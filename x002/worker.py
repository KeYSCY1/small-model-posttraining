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


def freeze_selection(reader, cfg: dict) -> tuple[list[dict], dict]:
    """确定性选题。

    * 配置含 ``pilot`` 段时走**先导规则**（见 :mod:`x002.pilot`）：排除已暴露题、只取
      现行判分支持的标量金标准、按 (协议版本, 数据 commit, 种子, unique_id, 金标准) 的
      sha256 升序取前 N；逐层分母记录在 ``layers``。
    * 否则保持既有规则（``unique_id`` 升序取前 N），以便历史配置可复现。

    两条路径都与模型输出无关，都不按结果挑题。
    """
    d = cfg["data"]
    if cfg.get("pilot"):
        return _freeze_selection_pilot(reader, cfg, d)
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
    return problems, {"rule": "sort_unique_id_ascending_take_first_n",
                      "n_problems": len(problems), "pool_size": len(pool)}


def _freeze_selection_pilot(reader, cfg: dict, d: dict) -> tuple[list[dict], dict]:
    pilot_cfg = dict(cfg["pilot"])
    exposed = _load_exposed_ids(cfg)
    rows = reader.list_rows()
    result = pilot_mod.select_pilot_problems(
        rows,
        exposed_ids=exposed,
        dataset_revision=str(cfg["data"].get("revision") or ""),
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
        "prompt_version": PROMPT_VERSION,
        "judge_version": JUDGE_VERSION,
        "env": (OFFLINE_FIXTURE or {}).get("env") or capture_env(),
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
    problems, selection = freeze_selection(reader, cfg)
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


class DatasetReader:
    """真实读题器（延迟导入 datasets）；只暴露 list_ids / get。"""

    def __init__(self, cfg: dict, dataset_ident: dict):
        from datasets import load_dataset

        d = cfg["data"]
        ds = load_dataset(d["dataset_id"], split=d.get("split", "test"),
                          revision=dataset_ident.get("revision"))
        self._pf, self._af = d["problem_field"], d["answer_field"]
        self._index = {row["unique_id"]: row for row in ds}

    def list_ids(self) -> list[str]:
        return list(self._index.keys())

    def list_rows(self) -> list[dict]:
        """返回全部题的 (unique_id, gold, subject, level)，供先导筛选；题面按需再取。"""
        rows = []
        for pid, row in self._index.items():
            rows.append({"unique_id": pid, "gold": row[self._af],
                         "subject": row.get("subject"), "level": row.get("level")})
        return rows

    def get(self, pid: str):
        row = self._index.get(pid)
        if row is None:
            return None
        return {"problem": row[self._pf], "gold": row[self._af]}


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
