"""先导批次选题：与模型输出无关的**预先确定**规则（X-002 专用）。

规则（写死，可复现）
--------------------
1. 只从**未暴露过**的题目里取：排除 ``exposed_ids``（R06 已跑的 12 个 ID 及其它已看过的 ID）。
2. 只保留**现行自动判分支持的标量参考答案**：由 :func:`x002.judge.gold_supported` 判定；
   区间、集合、坐标、复数、无有效答案一律排除（筛选只用于声明研究对象，**不把标准答案正文注入提示**）。
3. 排序键 = ``sha256(protocol_version | dataset_revision | seed | unique_id | 归一化参考答案)``，
   按该键升序取前 ``max_problems`` 道。排序与模型输出、与题面内容均无关，只依赖固定的
   (协议版本, 数据集 commit, 种子)。
4. 逐层记录分母：原始池 → 排除已暴露 → 排除非标量金标准 → 候选 → 最终入组；
   同时按学科与难度（若数据集提供）给出组成。

**诚实性**：本模块不做任何"补足数量"的操作。候选不足时返回实际可用的题目数，
并在 ``layers.shortfall`` 里写明与目标数的差额；调用方不得据此放宽筛选条件。
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping, Sequence

from .judge import gold_supported, normalize

__all__ = ["PROTOCOL_VERSION", "selection_key", "select_pilot_problems"]

PROTOCOL_VERSION = "x002-pilot-r07.1"


def selection_key(protocol_version: str, dataset_revision: str, seed: int, problem_id: str, gold: str) -> str:
    """稳定排序键：只依赖固定的协议/数据/种子与题目自身字段。"""
    payload = json.dumps(
        [protocol_version, dataset_revision, int(seed), problem_id, normalize(gold) or ""],
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _composition(rows: Sequence[Mapping]) -> dict:
    """按 subject / level 给出组成（缺失则记 unknown）。"""
    by_subject: dict[str, int] = {}
    by_level: dict[str, int] = {}
    for r in rows:
        s = str(r.get("subject") or "unknown")
        by_subject[s] = by_subject.get(s, 0) + 1
        lv = r.get("level")
        key = str(lv) if lv is not None else "unknown"
        by_level[key] = by_level.get(key, 0) + 1
    return {"by_subject": dict(sorted(by_subject.items())), "by_level": dict(sorted(by_level.items()))}


def select_pilot_problems(
    pool: Iterable[Mapping],
    *,
    exposed_ids: Iterable[str],
    dataset_revision: str,
    seed: int,
    max_problems: int,
    protocol_version: str = PROTOCOL_VERSION,
) -> dict:
    """按上述规则选题。

    ``pool`` 每项需含 ``unique_id``、``gold``；可选 ``problem``、``subject``、``level``。
    返回 ``{"selected": [...], "layers": {...}, "selection": {...}}``；
    ``selected`` 内**至少**含 unique_id 与 gold，供写入运行头与读题器使用。
    """
    rows = [dict(r) for r in pool]
    if not rows:
        raise ValueError("原始题池为空：无法选题（不得凭合成数据冒充真实题池）")
    missing = [i for i, r in enumerate(rows) if not r.get("unique_id") or "gold" not in r]
    if missing:
        raise ValueError(f"题池第 {missing[:5]} 项缺少 unique_id 或 gold 字段")

    exposed = set(exposed_ids)
    n_all = len(rows)

    kept_exposure = [r for r in rows if r["unique_id"] not in exposed]
    n_exposed_removed = n_all - len(kept_exposure)

    kept_scalar, removed_forms = [], []
    for r in kept_exposure:
        ok, reason = gold_supported(r["gold"])
        (kept_scalar if ok else removed_forms).append(r if ok else {**r, "excluded_reason": reason})

    ranked = sorted(
        kept_scalar,
        key=lambda r: selection_key(protocol_version, dataset_revision, seed, r["unique_id"], r["gold"]),
    )
    selected = ranked[: max(0, int(max_problems))]

    layers = {
        "pool_total": n_all,
        "exposed_ids_provided": len(exposed),
        "exposed_removed": n_exposed_removed,
        "after_exposure_filter": len(kept_exposure),
        "non_scalar_removed": len(removed_forms),
        "scalar_candidates": len(kept_scalar),
        "target": int(max_problems),
        "selected": len(selected),
        "shortfall": max(0, int(max_problems) - len(selected)),
        "removed_form_reasons": _reason_counts(removed_forms),
    }
    selection = {
        "protocol_version": protocol_version,
        "rule": "sha256(protocol|revision|seed|unique_id|normalized_gold) ascending, take first N",
        "dataset_revision": dataset_revision,
        "seed": int(seed),
        "max_problems": int(max_problems),
        "exposed_ids": sorted(exposed),
        "selected_ids": [r["unique_id"] for r in selected],
        "selected_ids_sha16": hashlib.sha256(
            ",".join(sorted(r["unique_id"] for r in selected)).encode("utf-8")
        ).hexdigest()[:16],
        "composition": _composition(selected),
        "composition_note": "按最终入组题目的 subject/level 统计；缺字段记 unknown。",
    }
    return {
        "selected": [
            {k: r.get(k) for k in ("unique_id", "gold", "problem", "subject", "level")}
            for r in selected
        ],
        "layers": layers,
        "selection": selection,
    }


def _reason_counts(rows: Sequence[Mapping]) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        key = str(r.get("excluded_reason") or "unknown")
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))
