"""X-002 消息构造、唯一编码路径与**基于来源与完整结构**的输入核对。

白名单（写死）
--------------
消息只由下列字段构成：``problem``（题面）、``feedback``（两臂共用的固定反馈）、
**仅 A 臂**的 ``prior_answer``（首轮旧答原文）、共用的 ``system``。

**外部答案/参考解字段（gold/solution）不参与消息构造**。题面或模型旧答里自然出现与标准
答案相同的字符串，**不算泄漏**——因此本模块不接收、也不使用 gold。

单一编码路径
------------
:func:`encode_messages` 是唯一入口：``apply_chat_template(..., tokenize=True,
add_generation_prompt=True)``。返回的 ``input_ids`` **直接交给推理并原样留证**，
不再从 messages 另行编码、也不再二次调用 tokenizer 编码（避免重复添加特殊 token）。
返回类型兼容普通 ``dict``、``UserDict``/``Mapping``（Hugging Face ``BatchEncoding`` 即继承
``UserDict``），不会把 Mapping 的键当 token IDs。

核对（失败即**阻止推理**）
--------------------------
:func:`verify_encoded` 按**来源与完整结构**检查：

1. 角色序列与期望一致；
2. 每条期望内容的**归一化完整正文**必须在实际解码文本里按顺序出现——
   因此"缺尾部"会失败，而不是靠"还有另一个特有片段"放行；
3. B 臂不得出现旧答正文（旧答根本没进入构造）。

共有字串、自然出现正确值、``\\boxed{}`` 包装都不会触发误阻止。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Sequence

__all__ = [
    "SYSTEM_PROMPT",
    "ARMS",
    "InputCheckError",
    "build_first_round",
    "build_second_round",
    "expected_contents",
    "encode_messages",
    "verify_encoded",
    "guard_encode",
]

SYSTEM_PROMPT = "你是一位严谨的数学解题助手。请给出清晰的解题过程和最终答案。"
ARMS = ("A_prior_visible", "B_prior_hidden")

_WS_RE = re.compile(r"\s+")
_SPECIAL_RE = re.compile(r"<\|[^|>]*\|>|<[^<>\s]{0,24}>")


class InputCheckError(RuntimeError):
    """输入核对或编码失败：调用方必须停止本次推理。"""


def build_first_round(problem: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]


def build_second_round(
    problem: str,
    feedback: str,
    prior_answer: str | None,
    arm: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
    """第二轮消息。**每次从零构造**，不携带任何先前会话对象。"""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm!r}")
    if not feedback or "上面的回答" in feedback:
        raise ValueError("feedback 必须两臂共用，且不得含只在 A 臂才有指代的表述")
    parts = [problem]
    if arm == "A_prior_visible":
        if not prior_answer:
            raise ValueError("A 臂必须显式给出非空 prior_answer")
        parts.append("此前的一次作答：\n" + prior_answer)
    parts.append(feedback)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def expected_contents(messages: Sequence[Mapping]) -> list[str]:
    return [str(m["content"]) for m in messages]


def _norm(text: str) -> str:
    """去掉模板特殊标记并折叠空白，用于完整正文比对。"""
    return _WS_RE.sub("", _SPECIAL_RE.sub("", text or ""))


def _extract_ids(encoded) -> list[int]:
    """兼容 dict / UserDict / BatchEncoding / 直接返回张量或列表。"""
    if encoded is None:
        raise InputCheckError("tokenizer 返回 None")
    ids = None
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise InputCheckError(f"编码结果缺少 input_ids（键：{sorted(map(str, encoded.keys()))}）")
        ids = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        ids = encoded.input_ids
    else:
        ids = encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    if not isinstance(ids, (list, tuple)):
        raise InputCheckError(f"无法从编码结果读取 token ids（类型 {type(ids).__name__}）")
    try:
        return [int(i) for i in ids]
    except (TypeError, ValueError) as exc:
        raise InputCheckError(f"token ids 不是整数序列：{exc}") from exc


def encode_messages(tokenizer, messages: Sequence[Mapping]) -> dict:
    """唯一编码路径。返回 ``{text, input_ids, sha16, n_tokens}``。"""
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        raise InputCheckError("tokenizer 没有 apply_chat_template，无法安全构造多轮输入")
    try:
        text = apply(list(messages), tokenize=False, add_generation_prompt=True)
        encoded = apply(list(messages), tokenize=True, add_generation_prompt=True)
    except Exception as exc:
        raise InputCheckError(f"apply_chat_template 失败：{exc}") from exc
    ids = _extract_ids(encoded)
    return {
        "text": text,
        "input_ids": ids,
        "n_tokens": len(ids),
        "sha16": hashlib.sha256(",".join(str(i) for i in ids).encode("utf-8")).hexdigest()[:16],
        "text_sha16": hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16],
        "add_generation_prompt": True,
    }


def _decode(tokenizer, ids: list[int]) -> str:
    try:
        return tokenizer.decode(ids, skip_special_tokens=True)
    except TypeError:  # pragma: no cover - 某些实现不接受关键字
        return tokenizer.decode(ids)
    except Exception as exc:  # pragma: no cover
        raise InputCheckError(f"decode 失败：{exc}") from exc


def verify_encoded(
    tokenizer,
    messages: Sequence[Mapping],
    encoded: dict,
    *,
    expect_prior: bool,
) -> dict:
    """按来源与完整结构核对；返回明细（``passed=False`` 时调用方必须中止）。"""
    contents = expected_contents(messages)
    roles = [str(m["role"]) for m in messages]
    decoded = _norm(_decode(tokenizer, encoded["input_ids"]))
    text_norm = _norm(encoded["text"])

    missing: list[dict] = []
    pos = 0
    for i, c in enumerate(contents):
        nc = _norm(c)
        found = decoded.find(nc, pos)
        if found < 0:
            found = decoded.find(nc)  # 允许模板重排顺序，但必须整段出现
        if found < 0:
            missing.append({"index": i, "role": roles[i], "reason": "content_missing_or_truncated",
                            "content_chars": len(nc)})
        else:
            pos = found + len(nc)

    result = {
        "passed": not missing,
        "expected_roles": roles,
        "n_contents": len(contents),
        "missing": missing,
        "expect_prior": expect_prior,
        "decoded_chars": len(decoded),
        "text_chars": len(text_norm),
        "input_sha16": encoded["sha16"],
        "text_sha16": encoded["text_sha16"],
        "n_tokens": encoded["n_tokens"],
        "add_generation_prompt": encoded.get("add_generation_prompt", True),
    }
    if not result["passed"]:
        result["reason"] = "content_missing_or_truncated"
    else:
        result["reason"] = "ok"
    return result


def check_encoded(tokenizer, messages: Sequence[Mapping], *, expect_prior: bool) -> dict:
    """编码 + 核对一步完成。核对失败抛 :class:`InputCheckError`（因此不会调用后端）。"""
    encoded = encode_messages(tokenizer, messages)
    result = verify_encoded(tokenizer, messages, encoded, expect_prior=expect_prior)
    if not result["passed"]:
        raise InputCheckError(f"输入核对失败：{result['reason']} {result['missing']}")
    return {"encoded": encoded, "check": result}


def guard_encode(tokenizer, messages: Sequence[Mapping], *, expect_prior: bool) -> dict:
    return check_encoded(tokenizer, messages, expect_prior=expect_prior)
