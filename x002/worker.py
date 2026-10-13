"""一次性试运行的**工作进程**：冻结身份 → 首轮 → 判分 → A/B → 汇总。

工作进程只被 :mod:`x002.cli` 监督调用（也可由测试以同一入口拉起）；
它自己不做时间监督——总截止由父进程负责终止。

工作进程顺序（重要）
--------------------
1. 写运行头（冻结身份）到 ``records.jsonl`` 第一行；
2. 才开始加载模型与 tokenizer；
3. 逐题生成、判分、写记录；
4. 写 ``summary.json`` / ``summary.txt``。

因此"加载阻塞/加载异常"一定发生在**已完成身份落盘**之后，父进程超时也能拿到可汇总的日志。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys

from . import generator as generator_mod
from . import judge as judge_mod
from . import prompts
from . import pilot as pilot_mod
from .run import PROMPT_VERSION, JUDGE_VERSION, ARMS, Store, format_summary, make_key, sha16, summarize


# ---------------------------------------------------------------- 身份解析与冻结


OFFLINE_FIXTURE: dict | None = None


def load_offline_fixture(path: str | None) -> dict | None:
    """载入离线 fixture：**只用于离线检查**，提供身份与环境，避免访问 Hub 或导入重库。

    fixture 结构（全部可选）::

        {"model": {"path_or_id": ..., "revision": ..., "resolved_from": "fixture"},
         "dataset": {"dataset_id": ..., "split": ..., "revision": ..., "resolved_from": "fixture"},
         "env": {...},
         "pause_after_start_seconds": 3.0}
    """
    global OFFLINE_FIXTURE
    if not path:
        OFFLINE_FIXTURE = None
        return None
    with open(path, encoding="utf-8") as fh:
        OFFLINE_FIXTURE = json.load(fh)
    return OFFLINE_FIXTURE


def _fixture_pause() -> None:
    if OFFLINE_FIXTURE:
        pause = float(OFFLINE_FIXTURE.get("pause_after_start_seconds") or 0.0)
        if pause > 0:
            import time as _t

            _t.sleep(pause)


def _fixture_identity(kind: str) -> dict | None:
    if not OFFLINE_FIXTURE:
        return None
    ident = OFFLINE_FIXTURE.get(kind)
    if not ident:
        return None
    return {**ident, "resolved_from": ident.get("resolved_from", "fixture")}


def resolve_model_identity(model_cfg: dict) -> dict:
    """解析模型 commit；本地目录用文件指纹，HF 用 sha；解析不到且未显式给出则报错。

    离线 fixture 存在时**只读 fixture**，不向 Hub 查询。
    """
    fixed = _fixture_identity("model")
    if fixed is not None:
        return fixed
    ident = {"path_or_id": model_cfg["path_or_id"], "revision": model_cfg.get("revision")}
    p = model_cfg["path_or_id"]
    if os.path.isdir(p):
        ident["revision"] = ident.get("revision") or _dir_fingerprint(p)
        ident["resolved_from"] = "local_dir"
        return ident
    try:  # pragma: no cover - 需要网络
        from huggingface_hub import HfApi

        ident["revision"] = HfApi().model_info(p, revision=model_cfg.get("revision") or None).sha
        ident["resolved_from"] = "hf_api"
    except Exception as exc:  # pragma: no cover
        if not ident.get("revision"):
            raise RuntimeError(f"无法解析模型 revision（{p}）：{exc}") from exc
        ident["resolved_from"] = "config"
    return ident


def resolve_dataset_identity(data_cfg: dict) -> dict:
    fixed = _fixture_identity("dataset")
    if fixed is not None:
        return fixed
    ident = {"dataset_id": data_cfg["dataset_id"], "split": data_cfg.get("split", "test"),
             "revision": data_cfg.get("revision")}
    try:  # pragma: no cover - 需要网络
        from huggingface_hub import HfApi

        ident["revision"] = HfApi().dataset_info(
            data_cfg["dataset_id"], revision=data_cfg.get("revision") or None).sha
        ident["resolved_from"] = "hf_api"
    except Exception as exc:  # pragma: no cover
        if not ident.get("revision"):
            raise RuntimeError(f"无法解析数据集 revision（{data_cfg['dataset_id']}）：{exc}") from exc
        ident["resolved_from"] = "config"
    return ident


def _dir_fingerprint(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    for name in sorted(os.listdir(path)):
        fp = os.path.join(path, name)
        if os.path.isfile(fp):
            h.update(name.encode())
            h.update(str(os.path.getsize(fp)).encode())
    return "local:" + h.hexdigest()[:16]


_VERSION_PACKAGES = ("torch", "transformers", "datasets", "sympy", "huggingface_hub")

# **受控源码与配置清单**：只有这些路径参与"源码身份"摘要。
# 运行日志、离线检查产物、虚拟环境、模型/数据缓存、压缩件一律不参与（与 .gitignore 一致）。
# Git 模式与无 Git 归档模式使用**同一份清单**；两者都额外记录清单覆盖检查（清单中有多少文件实际存在）。
MANIFEST_PATHS = (
    ".gitignore",
    "config.json",
    "exposed_problems_r06.json",
    "requirements.txt",
    "run_offline_checks.sh",
    "run_worker.py",
    "README.md",
    "REVIEW_STATUS.md",
    "SERVER_SYNC.md",
    "VERSION_CONTROL.md",
    "x002/__init__.py",
    "x002/cli.py",
    "x002/generator.py",
    "x002/judge.py",
    "x002/pilot.py",
    "x002/prompts.py",
    "x002/run.py",
    "x002/worker.py",
    "tests/fake_backend.py",
    "tests/test_offline_checks.py",
)


def _git_lines(pkg_dir: str, *args: str) -> list[str] | None:
    """只读调用 git；失败（无 Git / 非仓库 / 超时）返回 None，不抛异常。"""
    import subprocess

    try:
        out = subprocess.run(["git", "-C", pkg_dir, *args],
                             capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    if "-z" in args:
        return [p for p in out.stdout.split("\0") if p]
    return [ln.strip() for ln in out.stdout.split("\n") if ln.strip()]


def canonical_manifest_paths() -> list[str]:
    """受控清单的**规范顺序**：按路径字节（UTF-8）升序。

    Git 模式与无 Git 归档模式都必须用这一个顺序计算摘要，否则同一份源码会因来源模式不同而得到不同
    ``files_sha16``。清单本身在模块里按可读性排列，这里统一规范化，不依赖书写顺序。
    """
    return sorted(MANIFEST_PATHS, key=lambda p: p.encode("utf-8"))


def _manifest_digest(pkg_dir: str, paths: list[str]) -> str:
    """按给定路径顺序计算源码摘要：路径字节 + NUL + 文件内容 SHA-256 十六进制。"""
    import hashlib

    digest = hashlib.sha256()
    for rel in paths:
        fp = os.path.join(pkg_dir, rel)
        if not os.path.isfile(fp):
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        with open(fp, "rb") as fh:
            digest.update(hashlib.sha256(fh.read()).hexdigest().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def capture_code_version(pkg_dir: str | None = None) -> dict:
    """记录**实际源码版本**：Git commit/工作树状态，以及**只按受控清单**计算的源码摘要。

    * 摘要只读受控源码与配置（``MANIFEST_PATHS`` 的规范顺序）：不递归扫描日志、``.venv``、
      模型或数据缓存，也不读 ``runs/``；新增运行日志或被忽略的环境文件**不会**改变源码摘要。
    * 已跟踪代码的改动必须能被发现：Git 模式下摘要覆盖 ``git ls-files``（含被改动的已跟踪文件），
      并且 ``dirty`` 如实记录 ``git status --porcelain``（含未跟踪的非忽略文件）。
    * 无 Git 的归档包：读清单里实际存在的文件，用**同一规范顺序**计算摘要，因此同一份完整源码在
      两种模式下得到**相同** ``files_sha16``；此时摘要包含已修改但未提交的内容，故同时给出
      ``source_mode`` 与 ``manifest_missing``，不把两种模式的差异掩盖掉。
    * ``pkg_dir`` 仅供离线检查在**临时 fixture** 上取证使用（默认取本包目录）；
      正常运行时不要传参，运行头记录的就是实际部署的代码目录。
    """
    import hashlib

    pkg_dir = os.path.abspath(pkg_dir) if pkg_dir else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    order = canonical_manifest_paths()
    info: dict = {
        "source_mode": None,        # "git" | "archive"
        "git": None,
        "dirty": None,
        "files_sha16": None,
        "manifest": order,
        "manifest_order": "sorted-by-path-bytes",
        "manifest_files": 0,
        "manifest_missing": [],
        "manifest_sha16": hashlib.sha256(
            "\n".join(order).encode("utf-8")).hexdigest()[:16],
        "coverage": {
            "manifest_only": True,
            "excludes": ["日志", "runs/ 运行输出", "虚拟环境", "模型与数据缓存", "压缩件"],
        },
        "note": "源码摘要只按受控源码/配置清单计算；运行日志、环境、缓存与权重不参与。",
    }
    tracked: list[str] | None = None
    if _git_lines(pkg_dir, "rev-parse", "--git-dir") is not None:
        commit = _git_lines(pkg_dir, "rev-parse", "HEAD")
        info["git"] = commit[0] if commit else None
        status = _git_lines(pkg_dir, "status", "--porcelain")
        info["dirty"] = bool(status) if status is not None else None
        tracked = _git_lines(pkg_dir, "ls-files", "-z")
        if tracked is not None:
            tracked = sorted(tracked)
    manifest = set(MANIFEST_PATHS)
    if tracked is not None:
        info["source_mode"] = "git"
        paths = [p for p in order if p in set(tracked)]        # 规范顺序 ∩ 已跟踪
        tracked_outside = sorted(p for p in tracked if p not in manifest)
        if tracked_outside:
            info["tracked_outside_manifest"] = tracked_outside
        info["manifest_files"] = len(paths)
        info["manifest_missing"] = sorted(manifest - set(tracked))
    else:
        info["source_mode"] = "archive"
        paths = [p for p in order if os.path.isfile(os.path.join(pkg_dir, p))]
        info["manifest_files"] = len(paths)
        info["manifest_missing"] = sorted(manifest - set(paths))

    info["files_sha16"] = _manifest_digest(pkg_dir, paths)
    return info


def capture_env() -> dict:
    """只读**包元数据**记录版本，不导入 torch/transformers/datasets/sympy。

    离线检查路径因此不会因为收集版本而加载重依赖（也不触发这些库的副作用）。
    """
    from importlib import metadata

    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in _VERSION_PACKAGES:
        try:
            env[name] = metadata.version(name)
        except Exception:
            env[name] = None
    return env


def _load_exposed_ids(cfg: dict) -> list[str]:
    """已暴露题号清单（R06 已跑的 12 题为默认来源）。"""
    p = cfg.get("pilot", {}).get("exposed_ids_path")
    if not p:
        return []
    path = p if os.path.isabs(p) else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", p)
    with open(os.path.abspath(path), encoding="utf-8") as fh:
        data = json.load(fh)
    ids = data.get("exposed_ids") if isinstance(data, dict) else data
    return [str(i) for i in (ids or [])]


def freeze_selection(reader, cfg: dict, dataset_revision: str | None = None) -> tuple[list[dict], dict]:
    """确定性选题。

    * 配置含 ``pilot`` 段时走**先导规则**（见 :mod:`x002.pilot`）：排除已暴露题、只取
      现行判分支持的标量金标准、按 (协议版本, 数据 commit, 种子, unique_id, 金标准) 的
      sha256 升序取前 N；逐层分母记录在 ``layers``。
    * 否则保持既有规则（``unique_id`` 升序取前 N），以便历史配置可复现。

    两条路径都与模型输出无关，都不按结果挑题。
    """
    d = cfg["data"]
    if cfg.get("pilot"):
        if not dataset_revision:
            raise RuntimeError("数据集 commit 未解析（pilot 选题需要用它排序）；拒绝继续，不用空串代替")
        return _freeze_selection_pilot(reader, cfg, d, dataset_revision)
    n_max = int(d.get("n_problems", 12))
    pool = sorted(reader.list_ids())
    if len(pool) < 2:
        raise RuntimeError(f"数据集可用题目过少（{len(pool)}），无法形成 A/B 比较")
    chosen = pool[: min(n_max, len(pool))]
    problems, missing = [], []
    for pid in chosen:
        row = reader.get(pid)
        if row is None:
            missing.append(pid)
            continue
        problems.append({"problem_id": pid, "problem": row["problem"], "gold": row["gold"]})
    if missing:
        raise RuntimeError(f"题目读取失败/缺失，拒绝继续：{missing}")
    if len(problems) < 2:
        raise RuntimeError("可用题目不足 2 道，拒绝继续")
    selected_ids = sorted(p["problem_id"] for p in problems)
    return problems, {"rule": "sort_unique_id_ascending_take_first_n",
                      "n_problems": len(problems), "pool_size": len(pool),
                      "selected_ids_sha16": hashlib.sha256(
                          ",".join(selected_ids).encode("utf-8")).hexdigest()[:16],
                      "dataset_revision_used_for_sorting": str(dataset_revision or "not_applicable")}


def _freeze_selection_pilot(reader, cfg: dict, d: dict, dataset_revision: str) -> tuple[list[dict], dict]:
    pilot_cfg = dict(cfg["pilot"])
    exposed = _load_exposed_ids(cfg)
    rows = reader.list_rows()
    result = pilot_mod.select_pilot_problems(
        rows,
        exposed_ids=exposed,
        dataset_revision=str(dataset_revision),
        seed=int(pilot_cfg.get("seed", 20261013)),
        max_problems=int(pilot_cfg.get("max_problems", 48)),
        protocol_version=str(pilot_cfg.get("protocol_version", pilot_mod.PROTOCOL_VERSION)),
    )
    selected = result["selected"]
    problems, missing = [], []
    for r in selected:
        row = reader.get(r["unique_id"])
        if row is None:
            missing.append(r["unique_id"])
            continue
        problems.append({
            "problem_id": r["unique_id"], "problem": row["problem"], "gold": row["gold"],
            "subject": r.get("subject"), "level": r.get("level"),
        })
    if missing:
        raise RuntimeError(f"选中的题目在数据集中读取失败，拒绝继续：{missing}")
    if len(problems) < 2:
        raise RuntimeError("按先导规则选出的题目不足 2 道，拒绝继续（不放宽筛选条件）")
    selection = {
        "rule": result["selection"]["rule"],
        "dataset_revision_used_for_sorting": str(dataset_revision),
        "protocol_version": result["selection"]["protocol_version"],
        "n_problems": len(problems),
        "pool_size": result["layers"]["pool_total"],
        "target": result["layers"]["target"],
        "shortfall": result["layers"]["shortfall"],
        "selected_ids_sha16": result["selection"]["selected_ids_sha16"],
        "composition": result["selection"]["composition"],
        "layers": result["layers"],
        "exposed_ids_count": len(result["selection"]["exposed_ids"]),
    }
    return problems, selection


def build_run_header(cfg, model_ident, dataset_ident, selection, problems, budget_seconds, out_dir) -> dict:
    return {
        "kind": "run",
        "schema": 3,
        "started_at": None,  # 由调用方填充（父进程或 worker 启动时）
        "out_dir": os.path.abspath(out_dir),
        "model": {**cfg["model"], **model_ident},
        "dataset": {**cfg["data"], **dataset_ident},
        "selection": selection,
        "problems": [{"problem_id": p["problem_id"],
                      "problem_sha16": sha16(p["problem"]),
                      "gold_sha16": sha16(p["gold"])} for p in problems],
        "feedback": cfg["feedback"]["text"],
        "feedback_sha16": sha16(cfg["feedback"]["text"]),
        "generation": dict(cfg["generation"]),
        "judge": dict(cfg["judge"]),
        "budget_seconds": budget_seconds,
        "run_protocol_version": (cfg.get("run_protocol_version")
                                 or cfg.get("pilot", {}).get("protocol_version")
                                 or pilot_mod.PROTOCOL_VERSION),
        "prompt_version": PROMPT_VERSION,
        "judge_version": JUDGE_VERSION,
        "env": (OFFLINE_FIXTURE or {}).get("env") or capture_env(),
        "code_version": capture_code_version(),
        "note": "一次性运行：本目录开始后拒绝再次生成，只允许汇总。",
    }


# ---------------------------------------------------------------- 业务循环


def run_attempt(
    cfg: dict,
    *,
    out_dir: str,
    budget_seconds: float,
    reader=None,
    backend_factory=None,
    clock=None,
    header_extra: dict | None = None,
) -> dict:
    """一次性执行：首轮 → 判分 → A/B → 汇总。返回 summary。"""
    import time

    clock = clock or time.monotonic
    os.makedirs(out_dir, exist_ok=True)
    store = Store(os.path.join(out_dir, cfg["paths"]["log_name"]))

    _fixture_pause()          # 离线检查：模拟"启动阶段耗时"（仅 fixture 提供时生效）
    model_ident = resolve_model_identity(cfg["model"])
    dataset_ident = resolve_dataset_identity(cfg["data"])
    reader = reader or DatasetReader(cfg, dataset_ident)
    # 同一 commit 同时支配：读数据（DatasetReader）、选题排序、运行头
    problems, selection = freeze_selection(reader, cfg, dataset_revision=dataset_ident.get("revision"))
    header = build_run_header(cfg, model_ident, dataset_ident, selection, problems, budget_seconds, out_dir)
    header["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
    if header_extra:
        header.update(header_extra)
    store.append(header)          # ← 加载之前先落盘身份

    factory = backend_factory or generator_mod.load_backend
    backend = factory({**cfg["model"], **model_ident}, cfg["generation"])

    # ---------- 首轮 ----------
    first_records: dict[str, dict] = {}
    for p in problems:
        messages = prompts.build_first_round(p["problem"])
        out = generator_mod.generate_once(
            backend, messages, seed=cfg["generation"]["seed"],
            max_new_tokens=cfg["generation"]["max_new_tokens"], expect_prior=False,
        )
        rec = _record(cfg, p, "first", "none", messages, out)
        store.append(rec)
        first_records[p["problem_id"]] = rec
        print(f"[first] {p['problem_id']} -> {rec['run_status']}/{rec['judge']['status']}", flush=True)

    # ---------- 错题分流 + A/B ----------
    eligible = sorted(pid for pid, r in first_records.items()
                      if r["run_status"] == "completed" and r["judge"]["status"] == "incorrect")
    if not eligible:
        print("[note] 无合格错题：第二轮未覆盖（不伪造错答）", flush=True)
    by_id = {p["problem_id"]: p for p in problems}
    for pid in eligible:
        p = by_id[pid]
        prior = first_records[pid]["response"]
        for arm in ARMS:
            messages = prompts.build_second_round(
                p["problem"], cfg["feedback"]["text"], prior if arm == ARMS[0] else None, arm)
            out = generator_mod.generate_once(
                backend, messages, seed=cfg["generation"]["seed"],
                max_new_tokens=cfg["generation"]["max_new_tokens"],
                expect_prior=(arm == ARMS[0]),
            )
            rec = _record(cfg, p, "second", arm, messages, out)
            store.append(rec)
            print(f"[second] {pid} {arm} -> {rec['run_status']}/{rec['judge']['status']}", flush=True)

    return write_summary(cfg, store)


def _record(cfg, problem: dict, stage: str, arm: str, messages: list[dict], out: dict) -> dict:
    judged = judge_mod.judge(out.get("text", ""), problem["gold"],
                             rtol=cfg["judge"]["numeric_rtol"],
                             symbolic=bool(cfg["judge"].get("symbolic", False)))
    return {
        "kind": "record",
        "key": make_key(problem["problem_id"], stage, arm),
        "problem_id": problem["problem_id"],
        "split": cfg["data"].get("split", "test"),
        "stage": stage,
        "arm": arm,
        "seed": cfg["generation"]["seed"],
        "messages": messages,
        "serialized_text": out.get("serialized_text"),
        "input_sha16": out.get("input_sha16"),
        "text_sha16": out.get("text_sha16"),
        "n_input_tokens": out.get("n_input_tokens"),
        "input_check": out.get("input_check"),
        "response": out.get("text", ""),
        "usage": out.get("usage", {}),
        "seconds": out.get("seconds"),
        "finish_reason": out.get("finish_reason"),
        "run_status": _run_status(out),
        "error": out.get("error"),
        "judge": judged,
        "feedback": cfg["feedback"]["text"] if stage == "second" else None,
        "gold_sha16": sha16(problem["gold"]),
        "gold_note": "标准答案只入日志用于追溯，不进入任何消息",
    }


def _run_status(out: dict) -> str:
    if out.get("finish_reason") == "error" or out.get("error"):
        return "error"
    if out.get("finish_reason") == "length":
        return "truncated"
    return "completed"


def write_summary(cfg, store: Store) -> dict:
    summary = summarize(store.records())
    out_dir = os.path.dirname(os.path.abspath(store.path))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    text = format_summary(summary)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text, flush=True)
    return summary


def row_problem_id(row, data_cfg: dict) -> str:
    """把数据集原始行映射为**稳定题号**：``id_field``（默认 ``unique_id``）+ ``id_prefix``（默认空）。

    * AIME 2024 用原始 ``id``（整数，如 60）→ ``aime24:60``；旧 MATH 配置不用改（默认字段/空前缀）。
    * 拒绝缺失或空身份；**整数 0 是合法身份**，不能用真值判断当缺失。
    * 不在这里去重：重复身份由调用方（:func:`DatasetReader` / :func:`freeze_selection`）显式报错，
      避免字典静默覆盖。
    """
    field = data_cfg.get("id_field") or "unique_id"
    if field not in row:
        raise RuntimeError(f"数据集缺少身份字段 {field!r}（行字段：{sorted(row)[:8]}）")
    raw = row[field]
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise RuntimeError(f"数据集身份字段 {field!r} 为空：{raw!r}")
    prefix = data_cfg.get("id_prefix") or ""
    return f"{prefix}{raw}"


def row_gold(row, data_cfg: dict):
    """按 ``answer_field`` 取金标准；``gold_unbox`` 打开时脱去外层 ``\\boxed{}``（见 judge.gold_value）。"""
    return judge_mod.gold_value(row[data_cfg["answer_field"]], unbox=bool(data_cfg.get("gold_unbox")))


def _index_rows(rows, data_cfg: dict) -> dict:
    """按稳定题号建索引；**重复身份显式报错**，不静默覆盖。"""
    index: dict = {}
    dupes: list[str] = []
    for row in rows:
        pid = row_problem_id(row, data_cfg)
        if pid in index:
            dupes.append(pid)
        index[pid] = row
    if dupes:
        raise RuntimeError(f"数据集存在重复题号，拒绝继续：{sorted(set(dupes))[:5]}")
    return index


class DatasetReader:
    """真实读题器（延迟导入 datasets）；只暴露 list_ids / list_rows / get。"""

    def __init__(self, cfg: dict, dataset_ident: dict):
        from datasets import load_dataset

        d = cfg["data"]
        ds = load_dataset(d["dataset_id"], split=d.get("split", "test"),
                          revision=dataset_ident.get("revision"))
        self._pf, self._af = d["problem_field"], d["answer_field"]
        self._data_cfg = d
        self._index = _index_rows(list(ds), d)

    def list_ids(self) -> list[str]:
        return list(self._index.keys())

    def list_rows(self) -> list[dict]:
        """返回全部题的 (unique_id, gold, subject, level)，供先导筛选；题面按需再取。"""
        rows = []
        for pid, row in self._index.items():
            rows.append({"unique_id": pid, "gold": row_gold(row, self._data_cfg),
                         "subject": row.get("subject"), "level": row.get("level")})
        return rows

    def get(self, pid: str):
        row = self._index.get(pid)
        if row is None:
            return None
        return {"problem": row[self._pf], "gold": row_gold(row, self._data_cfg)}


# ---------------------------------------------------------------- worker 入口


def _import_factory(spec: str):
    mod_name, _, attr = spec.partition(":")
    if not mod_name or not attr:
        raise ValueError(f"--backend-factory 需要 'module:callable' 形式，收到 {spec!r}")
    mod = __import__(mod_name, fromlist=[attr])
    return getattr(mod, attr)


def worker_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="X-002 工作进程（由 cli 监督调用）")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budget-seconds", type=float, required=True)
    ap.add_argument("--backend-factory", default=None,
                    help="测试接缝：'module:callable'，默认使用真实 transformers 后端")
    ap.add_argument("--problems-json", default=None, help="测试接缝：预置题目 JSON（[{problem_id,problem,gold}]）")
    ap.add_argument("--offline-fixture", default=None,
                    help="测试接缝：离线 fixture（身份/环境/启动延迟）；正式运行不提供该参数")
    args = ap.parse_args(argv)

    load_offline_fixture(args.offline_fixture)

    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    reader = None
    if args.problems_json:
        with open(args.problems_json, encoding="utf-8") as fh:
            rows = json.load(fh)
        reader = StaticReader(rows)
    factory = _import_factory(args.backend_factory) if args.backend_factory else None
    summary = run_attempt(cfg, out_dir=args.out_dir, budget_seconds=args.budget_seconds,
                          reader=reader, backend_factory=factory)
    return 0 if summary["first_round"]["n_records"] > 0 else 1


class StaticReader:
    """测试接缝：从内存列表读题（不联网、不导入 datasets）。"""

    def __init__(self, rows: list[dict]):
        self.rows = {r["problem_id"]: r for r in rows}

    def list_ids(self):
        return list(self.rows.keys())

    def list_rows(self):
        return [{"unique_id": r["problem_id"], "gold": r["gold"],
                 "subject": r.get("subject"), "level": r.get("level")} for r in self.rows.values()]

    def get(self, pid):
        r = self.rows.get(pid)
        return None if r is None else {"problem": r["problem"], "gold": r["gold"]}


if __name__ == "__main__":
    raise SystemExit(worker_main())
