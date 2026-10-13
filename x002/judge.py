"""X-002 判分：答案抽取与数值等价判定（无模型依赖，可离线测试）。

候选抽取原则（本轮修正）
------------------------
只从**声明支持的包装**里抽候选，**不做数字前缀回退**：

* ``\\boxed{...}`` / ``\\fbox{...}``（含嵌套；空盒 → UNKNOWN）
* ``#### 42``
* 显式句子 ``final answer is X`` / ``answer is X`` / ``answer: X``，
  且 X 必须**整体**匹配一个受支持的标量 token（数值／分数／百分数／合法千分位／行内公式），
  **不是**只取前面的数字前缀。因此 ``The answer is 2.5 cm.`` 会抽不到受支持候选 → UNKNOWN，
  而不是抽成 ``2`` 判正确。

受支持的标量：整数、小数、简单分数 ``a/b``、百分数 ``50\\%``、合法千分位 ``1,000``、
``$...$`` / ``\\(...\\)`` 行内公式内的标量。

范围外一律 **UNKNOWN（不判错）**：带单位、区间、集合、元组/坐标、矩阵、分段函数、
非法逗号串（``1,2``）、多答案、空盒、抽不出答案、参考答案非标量。

符号（SymPy）判错路径默认关闭（``judge.symbolic=false``）；本轮环境未安装 SymPy，该分支未验证。
"""

from __future__ import annotations

import re
import unicodedata
from fractions import Fraction

__all__ = [
    "judge",
    "gold_supported",
    "extract_answer",
    "normalize",
    "numeric_value",
    "answers_equal",
    "is_valid_thousands",
    "classify_scalar",
]

_BOXED_RE = re.compile(r"\\(?:boxed|fbox)\s*\{")
_HASH_RE = re.compile(r"#+\s*([^\n#]+)")

# 受支持的标量 token（整体匹配才算候选）
# 受支持的标量 token：整数 / 小数 / 合法千分位 / 简单分数（不得含空格或单位）/ 百分数
_FRACTION = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)/[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)"
_PLAIN = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_PERCENT = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%"
# 末尾追加 (?!\.\d)：防止在 "2.5 cm." 里只匹配到整数前缀 "2"
_NUM_TOKEN = r"(?:" + _PERCENT + r"|" + _FRACTION + r"|" + _PLAIN + r")(?!\.\d)"
_MATH_TOKEN = r"\$[^$\n]{1,80}\$|\\\([^)\n]{1,80}\\\)"
# 候选之后只允许句末标点或行尾；后面还有别的正文（如单位 "cm"）就不算受支持候选，
# 从而避免"取数字前缀"式回退误判。
_SENT_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*"
    r"(?:(?P<num>" + _NUM_TOKEN + r")(?=\s*(?:[.!?。]|$|,))"
    r"|(?P<math>" + _MATH_TOKEN + r"))",
    re.IGNORECASE,
)


def _match_brace(text: str, open_idx: int) -> str | None:
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
    return None


def _strip_latex_wrappers(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = re.sub(r"^\\\(|\\\)$", "", s).strip()
    s = re.sub(r"\\(?:left|right|displaystyle|,|!|;|quad|qquad)\b", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", s)
    return s.strip()


def _clean_tail(s: str) -> str:
    """去掉句末标点但**保留小数**：``0.5.``→``0.5``；``12.``→``12``。"""
    out = s.strip()
    while out and out[-1] in ".。;；":
        if re.fullmatch(r"[+-]?\d+\.", out):
            out = out[:-1]
            break
        out = out[:-1]
    return out.strip()


_TRAILING_NUMBER_RE = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")


def _has_trailing_number(body: str, start: int) -> bool:
    """匹配位置之后、同一行内是否还存在别的数值 token。

    用于拒绝"数字前缀"式抽取：`1,2.`、`1, 2.`、`2.5, or 7.` 都会因此返回 UNKNOWN，
    而 `1,000.`、`1000.`、`7, so we are done` 不受影响（逗号后紧跟数字才被算作新数值）。
    """
    line_end = body.find("\n", start)
    tail = body[start : line_end if line_end != -1 else len(body)]
    return _TRAILING_NUMBER_RE.search(tail) is not None


def extract_answer(text: str) -> tuple[str | None, str]:
    """抽**唯一**受支持候选。返回 (candidate, reason)；抽不到时 candidate=None。"""
    if text is None:
        return None, "empty_response"
    body = text

    boxes: list[str] = []
    for m in _BOXED_RE.finditer(body):
        inner = _match_brace(body, m.end() - 1)
        if inner is not None:
            boxes.append(_clean_tail(_strip_latex_wrappers(inner)))
    if boxes:
        if any(b == "" for b in boxes):
            return None, "empty_box"
        cands = set(boxes)
        if len(cands) > 1:
            return None, f"multiple_boxed:{len(cands)}"
        return cands.pop(), "boxed"

    hashes = [_clean_tail(_strip_latex_wrappers(h)) for h in _HASH_RE.findall(body)]
    hashes = [h for h in hashes if h]
    if hashes:
        cands = set(hashes)
        if len(cands) > 1:
            return None, f"multiple_hash:{len(cands)}"
        return cands.pop(), "hash"

    sents: list[str] = []
    for m in _SENT_RE.finditer(body):
        raw = m.group("num") or m.group("math") or ""
        if m.group("num") and _has_trailing_number(body, m.end()):
            # 形如 "1,2." / "1, 2." / "2.5, or 7."：抽到的是数字前缀或存在多答案 → 不收候选
            continue
        cleaned = _clean_tail(_strip_latex_wrappers(raw))
        if cleaned:
            sents.append(cleaned)
    if sents:
        cands = set(sents)
        if len(cands) > 1:
            return None, f"multiple_sentence:{len(cands)}"
        return cands.pop(), "sentence"

    return None, "no_answer_found"


# ---------------------------------------------------------------- 归一化与比较

_WS_RE = re.compile(r"\s+")
_UNSUPPORTED_MARKERS = (
    "\\begin{cases}",
    "\\begin{pmatrix}",
    "\\begin{bmatrix}",
    "\\begin{array}",
    "\\in",
    "\\cup",
    "\\cap",
    "\\pm",
    "\\infty",
)
_THOUSANDS_RE = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_HAS_COMMA_RE = re.compile(r",")
_MATHY_RE = re.compile(r"^[0-9a-zA-Z\s\+\-\*/^_(){}\[\]\\.,!|<>=\\'\"$%]*$")

_SYMPY_CACHE: dict = {}


def _sympy_ready() -> bool:
    """SymPy 只在**真正启用符号判分**时才导入（默认关闭时模块导入不触碰 SymPy）。"""
    if "ok" not in _SYMPY_CACHE:
        try:
            from sympy import simplify  # type: ignore  # noqa: F401
            from sympy.parsing.sympy_parser import (  # type: ignore
                implicit_multiplication_application,
                parse_expr,
                standard_transformations,
            )

            _SYMPY_CACHE.update({"ok": True, "simplify": simplify, "parse_expr": parse_expr,
                                 "transforms": standard_transformations + (implicit_multiplication_application,)})
        except Exception:
            _SYMPY_CACHE["ok"] = False
    return bool(_SYMPY_CACHE["ok"])


def is_valid_thousands(s: str) -> bool:
    return bool(_THOUSANDS_RE.fullmatch(s.strip()))


def normalize(s: str | None) -> str | None:
    if s is None:
        return None
    out = unicodedata.normalize("NFKC", s).strip()
    out = _WS_RE.sub("", out)
    out = out.replace("−", "-").replace("×", "*").replace("÷", "/")
    out = out.replace("\\%", "%")
    out = out.replace("^\\circ", "").replace("\\degree", "")
    out = out.replace("\\left", "").replace("\\right", "")
    if _HAS_COMMA_RE.search(out) and is_valid_thousands(out):
        out = out.replace(",", "")
    return out


def _looks_unsupported(s: str | None) -> str | None:
    if s is None:
        return None
    low = s.lower()
    for marker in _UNSUPPORTED_MARKERS:
        if marker in low:
            return f"unsupported_form:{marker}"
    if re.search(r"[\[\(].*,.*[\]\)]", s):
        return "unsupported_form:interval_or_tuple"
    if _HAS_COMMA_RE.search(s):
        return "unsupported_form:malformed_comma_number"
    return None


_LATEX_EXPR_RE = re.compile(r"^\\?(?:sqrt|frac|pi|theta|alpha|beta|gamma|ln|log|sin|cos|tan)\b.*$")
_POLY_RE = re.compile(r"^[0-9a-zA-Z\^\+\-\*/()_.]+$")


def _is_supported_scalar(s: str | None) -> bool:
    """判定是否属于**声明支持**的标量：数值/分数/百分数/合法千分位/平方根等简单表达式。

    带单位的候选（如 ``2.5cm``）会在此被拒，从而返回 UNKNOWN 而不是判错。
    """
    if not s:
        return False
    t = s.replace("%", "")
    if _THOUSANDS_RE.fullmatch(t) or re.fullmatch(r"[+-]?\d+", t) or re.fullmatch(r"[+-]?\d+\.\d+", t):
        return True
    if re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)/[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)", s):
        return True
    if _LATEX_EXPR_RE.fullmatch(s):
        return True
    if re.search(r"[\^_/]", s) and _POLY_RE.fullmatch(s):
        return True
    return False


def _is_mathy(s: str | None) -> bool:
    return bool(s) and bool(_MATHY_RE.fullmatch(s))


def classify_scalar(s: str | None) -> str:
    if not s:
        return "empty"
    if s.endswith("%"):
        return "percent"
    if re.fullmatch(r"[+-]?\d+/\d+", s):
        return "fraction"
    if re.fullmatch(r"[+-]?(?:\d+\.\d+|\d+)", s):
        return "numeric"
    return "expression" if _is_mathy(s) else "text"


def numeric_value(s: str | None) -> tuple[float | None, str]:
    if s is None:
        return None, "none"
    t = s
    t = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", t)
    t = t.replace("\\", "").replace("%", "/100").strip().strip("$")
    if not t:
        return None, "empty"
    frac = re.fullmatch(r"\(?(-?\d+(?:\.\d+)?)\)?/\(?(-?\d+(?:\.\d+)?)\)?", t)
    if frac:
        try:
            return float(Fraction(frac.group(1)) / Fraction(frac.group(2))), "ok"
        except Exception:
            return None, "bad_fraction"
    try:
        return float(t), "ok"
    except Exception:
        return None, "not_numeric"


def answers_equal(
    candidate: str | None, gold: str | None, rtol: float = 1e-6, symbolic: bool = False
) -> tuple[bool, str, str]:
    cn, gn = candidate, gold
    if cn is None:
        return False, "none", "no_candidate"
    if gn is None:
        return False, "none", "no_gold"
    for bad in (_looks_unsupported(cn), _looks_unsupported(gn)):
        if bad:
            return False, "none", bad
    for value, which in ((cn, "candidate"), (gn, "gold")):
        if not _is_supported_scalar(value):
            return False, "none", f"unsupported_form:{which}_not_a_supported_scalar"
    if cn == gn:
        return True, "string", "exact_after_normalize"

    cv, creason = numeric_value(cn)
    gv, greason = numeric_value(gn)
    if cv is not None and gv is not None:
        if abs(cv - gv) <= rtol * max(1.0, abs(gv)):
            return True, "numeric", "numeric_tolerance"
        return False, "numeric", "numeric_mismatch"

    if symbolic and _is_mathy(cn) and _is_mathy(gn):
        eq = _sympy_equal(cn, gn)
        if eq is True:
            return True, "sympy", "symbolic_equal"
        if eq is False:
            return False, "sympy", "symbolic_unequal"
    return False, "none", f"undecidable(candidate={creason},gold={greason})"


def _sympy_equal(a: str, b: str) -> bool | None:
    if not _sympy_ready():
        return None
    try:
        parse_expr = _SYMPY_CACHE["parse_expr"]
        transforms = _SYMPY_CACHE["transforms"]
        ea = parse_expr(a.replace("\\", ""), transformations=transforms)
        eb = parse_expr(b.replace("\\", ""), transformations=transforms)
        return bool(_SYMPY_CACHE["simplify"](ea - eb) == 0)
    except Exception:
        return None


def gold_supported(gold: str | None) -> tuple[bool, str]:
    """参考答案是否属于**现行自动判分支持**的标量范围（供选题筛选使用）。

    只做"能否判分"的声明，不改判分行为；gold 不会因此进入任何模型提示。
    """
    if gold is None or not str(gold).strip():
        return False, "empty_gold"
    raw = _clean_tail(_strip_latex_wrappers(str(gold)))
    norm = normalize(raw)
    bad = _looks_unsupported(norm)
    if bad:
        return False, bad
    if not _is_supported_scalar(norm):
        return False, "unsupported_form:not_a_supported_scalar"
    return True, "supported"


def judge(response_text: str, gold: str, *, rtol: float = 1e-6, symbolic: bool = False) -> dict:
    """对外主入口。标准答案只在此处（判分）使用，不进入任何消息构造。"""
    cand, why = extract_answer(response_text)
    gold_raw = _strip_latex_wrappers(gold) if gold is not None else None
    gold_norm = normalize(_clean_tail(gold_raw)) if gold_raw is not None else None
    # symbolic_enabled 表示开关状态；不为记录可用性而导入 SymPy（真正用到时才探）
    base = {"symbolic_enabled": bool(symbolic)}
    if cand is None:
        return {**base, "status": "undetermined", "extracted": None, "gold_norm": gold_norm,
                "reason": why, "backend": "none"}
    equal, backend, reason = answers_equal(normalize(cand), gold_norm, rtol=rtol, symbolic=symbolic)
    status = "undetermined" if backend == "none" else ("correct" if equal else "incorrect")
    return {**base, "status": status, "extracted": cand, "gold_norm": gold_norm,
            "reason": reason, "backend": backend}
