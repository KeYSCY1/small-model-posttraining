"""记录、汇总与分母（一次性运行版本：不含续跑/恢复/预算账本）。

日志格式：``records.jsonl``，每行一条 JSON 记录，键为 ``(problem_id, stage, arm)``。
第一行为 ``{"kind": "run", ...}`` 的运行头（冻结身份、命令、时间），其后为各条生成记录。
汇总完全由 ``records.jsonl`` 派生，原始记录不可被汇总替代。

汇总口径（与已审一致）
----------------------
* 首轮分流：``correct`` / ``incorrect`` / ``undetermined`` / ``truncated`` / ``error``；
  只有**首轮正常完成且判错**的题进入第二轮。
* 第二轮同样区分 ``completed`` / ``truncated`` / ``error`` / ``unknown``。
* 默认比较只用 **determinate_pairs**（同题两臂都正常完成且判分可确定）；
  未纳入者逐条记入 ``excluded_from_pairs``。
* 另报 ``verified_success_rate_all_attempts``：分母为该臂**全部尝试**，名称与分母单列。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Iterable

__all__ = [
    "PROMPT_VERSION",
    "JUDGE_VERSION",
    "ARMS",
    "make_key",
    "sha16",
    "norm_text",
    "Store",
    "summarize",
    "format_summary",
    "bootstrap_ci",
    "first_bucket",
    "second_state",
]

PROMPT_VERSION = "x002-prompts-20261013.3"
JUDGE_VERSION = "x002-judge-20261013.3"
ARMS = ("A_prior_visible", "B_prior_hidden")
_WS_RE = re.compile(r"\s+")


def make_key(problem_id: str, stage: str, arm: str) -> str:
    return f"{problem_id}|{stage}|{arm}"


def sha16(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def norm_text(text: str) -> str:
    return _WS_RE.sub(" ", text or "").strip()


class Store:
    """只追加的 JSONL 记录库（一次性运行，无恢复语义）。"""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    def append(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def records(self) -> list[dict]:
        out: list[dict] = []
        if not os.path.exists(self.path):
            return out
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # 被终止时可能留下半行，跳过
        return out

    def header(self) -> dict | None:
        for r in self.records():
            if r.get("kind") == "run":
                return r
        return None


def first_bucket(record: dict) -> str:
    rs = record.get("run_status")
    if rs in ("truncated", "error"):
        return rs
    st = record.get("judge", {}).get("status")
    if st in ("correct", "incorrect"):
        return st
    if st == "undetermined":
        return "undetermined"
    return "error"


def second_state(record: dict) -> str:
    rs = record.get("run_status")
    if rs in ("truncated", "error"):
        return rs
    st = record.get("judge", {}).get("status")
    if st == "undetermined":
        return "unknown"
    if st in ("correct", "incorrect"):
        return "completed"
    return "error"


def bootstrap_ci(diffs: list[float], n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    import random

    if not diffs:
        return None, None, None
    mean = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return mean, means[int((alpha / 2) * n_boot)], means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]


def summarize(records: Iterable[dict]) -> dict:
    recs = [r for r in records if r.get("kind") != "run"]
    first = [r for r in recs if r.get("stage") == "first"]
    second = [r for r in recs if r.get("stage") == "second"]

    first_status: dict[str, int] = {}
    for r in first:
        b = first_bucket(r)
        first_status[b] = first_status.get(b, 0) + 1
    eligible_ids = sorted({r["problem_id"] for r in first if first_bucket(r) == "incorrect"})

    per_arm: dict[str, dict] = {}
    for arm in ARMS:
        rows = [r for r in second if r.get("arm") == arm]
        states: dict[str, int] = {}
        for r in rows:
            s = second_state(r)
            states[s] = states.get(s, 0) + 1
        per_arm[arm] = {
            "n_attempted": len(rows),
            "states": states,
            "n_determinate": states.get("completed", 0),
            # 分子要求"正常完成且判分正确"：截断/错误/UNKNOWN 不计成功，
            # 但仍留在"全部尝试"的分母里。
            "verified_success_rate_all_attempts": (
                sum(1 for r in rows
                    if r.get("run_status") == "completed" and r.get("judge", {}).get("status") == "correct")
                / len(rows) if rows else None
            ),
            "in_tokens": sum(r.get("usage", {}).get("input_tokens") or 0 for r in rows),
            "out_tokens": sum(r.get("usage", {}).get("output_tokens") or 0 for r in rows),
            "seconds": round(sum(r.get("seconds") or 0.0 for r in rows), 2),
        }

    by_key = {(r["problem_id"], r.get("arm")): r for r in second}
    pairs, diffs, excluded = [], [], []
    for pid in eligible_ids:
        ra, rb = by_key.get((pid, ARMS[0])), by_key.get((pid, ARMS[1]))
        if ra is None or rb is None:
            excluded.append({"problem_id": pid, "reason": "arm_record_missing"})
            continue
        sa, sb = second_state(ra), second_state(rb)
        if sa != "completed" or sb != "completed":
            excluded.append({"problem_id": pid, "reason": f"non_determinate:{sa}/{sb}"})
            continue
        ja, jb = ra["judge"]["status"], rb["judge"]["status"]
        if ja not in ("correct", "incorrect") or jb not in ("correct", "incorrect"):
            excluded.append({"problem_id": pid, "reason": "judge_undetermined"})
            continue
        va, vb = (1.0 if ja == "correct" else 0.0), (1.0 if jb == "correct" else 0.0)
        pairs.append({"problem_id": pid, "A_correct": va, "B_correct": vb})
        diffs.append(va - vb)

    mean, lo, hi = bootstrap_ci(diffs) if diffs else (None, None, None)
    return {
        "n_records": len(recs),
        "first_round": {"n_records": len(first), "status_counts": first_status,
                        "eligible_wrong_ids": eligible_ids, "n_eligible": len(eligible_ids)},
        "second_round": {"n_attempted_problems": len({r["problem_id"] for r in second}), "per_arm": per_arm},
        "determinate_pairs": {
            "name": "determinate_pairs（两臂均正常完成且判分可确定）",
            "n_pairs": len(pairs),
            "correction_rate_A": (sum(p["A_correct"] for p in pairs) / len(pairs)) if pairs else None,
            "correction_rate_B": (sum(p["B_correct"] for p in pairs) / len(pairs)) if pairs else None,
            "denominator": len(pairs),
            "observed_diff_A_minus_B": mean,
            # 先导/工程批次**不做区间推断**：不展示 bootstrap 区间，避免把退化输出当成 95%CI
            "interval_estimate": None,
            "interval_label": "未估计（仅描述）",
            "inference_note": INFERENCE_NOTE,
            "pairs": pairs,
            "note": "这是可判定配对子集，不代表全部入组题；未纳入者见 excluded_from_pairs。",
        },
        "excluded_from_pairs": excluded,
        "cost_note": "in/out tokens 与秒数为实际成本记录，不表示已固定总成本。",
        "scope_note": "极小工程样本不支持效应结论。",
    }


INFERENCE_NOTE = (
    "先导/工程批次不作区间推断：仅报告观察到的两臂修正率、配对数与成本；"
    "正式推断方法与样本量、重复单位确定后再定。"
)


def format_summary(summary: dict) -> str:
    f, s, d = summary["first_round"], summary["second_round"], summary["determinate_pairs"]
    lines = [
        "== X-002 汇总 ==",
        f"记录总数: {summary['n_records']}",
        f"首轮: {f['n_records']} 条 -> {f['status_counts']}",
        f"合格错题（首轮正常完成且判错）: {f['n_eligible']}",
        f"第二轮已尝试题数: {s['n_attempted_problems']}",
        "-- 第二轮各臂 --",
    ]
    for arm, st in s["per_arm"].items():
        rate = st["verified_success_rate_all_attempts"]
        lines.append(
            f"  {arm}: 尝试={st['n_attempted']} 状态={st['states']} 可判定={st['n_determinate']}"
            f" 已验证成功率(分母=全部尝试)={('%.3f' % rate) if rate is not None else 'NA'}"
            f" tok_in/out={st['in_tokens']}/{st['out_tokens']} 秒={st['seconds']}"
        )
    if d["n_pairs"]:
        lines.append(
            f"-- {d['name']}: n={d['n_pairs']} 修正率A={d['correction_rate_A']:.3f}"
            f" B={d['correction_rate_B']:.3f} 观察差(A-B)={d['observed_diff_A_minus_B']:.3f}"
            f" [{d['interval_label']}]"
        )
    else:
        lines.append(f"-- {d['name']}: 无有效配对（excluded={summary['excluded_from_pairs']}）--")
    lines.append(summary["determinate_pairs"]["inference_note"])
    lines.append(summary["cost_note"])
    lines.append(summary["scope_note"])
    return "\n".join(lines)
