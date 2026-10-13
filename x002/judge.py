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

多答案判定使用**完整的候选/答案区域**（不是只看候选紧后面那一行）：
候选之后到「段落边界（空行）」或「句末标点 + 下一句首字母大写」之间的整段文字，
只要再出现一个并列/多答案形态的数值（``,`` ``;`` ``、`` ``or`` ``and`` 之后接数值），
或**同一行内**还有另一个完整数值，就判 UNKNOWN；空行之后的另一段解释里的数字不算第二答案。
尚未覆盖：候选句本身不含 ``is``/``:`` 的并列数值（如“答案是 2，3”）仍按“抽不出答案”处理。

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

# 句子内的**并列连接词**：`, 123`、`; 3`、`、4`、`or 7`、`and -5`。
_LIST_CONNECTOR_RE = re.compile(r"[,;、]|\bor\b|\band\b", re.IGNORECASE)

# 句末标点 + 空白（含换行）+ 下一句首字母大写或数学式开头 → 句子边界
_SENTENCE_END_RE = re.compile(r"[.!?。！？]\s+(?=[A-Z$\\])")
# 段落边界：空行（含只有空白的行）
_PARAGRAPH_END_RE = re.compile(r"\n[ \t]*\n")
# **行首**并列连接词（`\nOr 3`）：是备选答案的延续，不构成句子边界
_LINE_CONTINUATION_RE = re.compile(r"(?m)^[ \t]*(?:or|and)\b", re.IGNORECASE)


def _answer_region(body: str, start: int) -> str:
    """候选位置之后的**答案区域**：取**最近**的边界，而不是只要后面有空行就一律放到空行。

    边界 = 空行，或「句末标点 + 空白 + 下一句首字母大写/数学式」——但**行首并列连接词**
    （``\\nOr 3``）不算边界，那是备选答案的延续，必须留在区域内被多答案规则看到。

    这样 ``The answer is 2. We used 3 terms.`` 与后面再加一段说明的文字给出**同一个**答案区域
    （区域止于 ``2.`` 之后的句末边界），而 ``The answer is 2.\\nOr 3.`` 会把 ``Or 3`` 留在区域内。
    """
    if start >= len(body):
        return ""
    span = body[start:]
    end = len(span)
    s_match = _SENTENCE_END_RE.search(span)
    while s_match is not None:
        after = span[s_match.end():]
        if _LINE_CONTINUATION_RE.match(after):
            # 只是行首并列连接词的延续：从它之后继续找真正的边界，同时不把它切掉
            s_match = _SENTENCE_END_RE.search(span, s_match.end())
            continue
        end = min(end, s_match.start())
        break
    p_match = _PARAGRAPH_END_RE.search(span)
    if p_match is not None:
        end = min(end, p_match.start())
    return span[:end]


def _starts_new_number(text: str) -> bool:
    """``text`` 是否以**新数值**开头（复用受支持的数值 token 语法，含正负号）。

    * 复用 ``_TRAILING_NUMBER_RE``（``[+-]?`` + 整数/合法千分位 + 可选小数 + 可选百分号），
      因此 ``,123``、``, -123``、``,-1,000``、``,+3``、``or -123`` 都按"另一个候选"处理；
    * ``1,000``：逗号前是数字、后面正好是千分位分组 → 不是新答案；
    * ``.`` / ``so we are done`` / 空 → 不是数值。
    """
    stripped = text.lstrip(" \t\n$\\")
    if _TRAILING_NUMBER_RE.match(stripped) and stripped != text:
        return True                                   # 空白/换行/公式符之后直接是数值（含符号）
    if not stripped:
        return False
    if re.match(r"(?:frac|dfrac|tfrac)\b", stripped):
        return True
    if not _TRAILING_NUMBER_RE.match(stripped):
        return False
    comma = re.match(r"[,;、][ \t]*", stripped)        # ``, 123`` / ``,123`` / ``, -123``
    if comma:
        after = re.sub(r"\s+", "", stripped[comma.end():])
        return not (stripped[:1] == "," and re.fullmatch(r"\d{3}", after))
    prev = text[len(text) - len(stripped) - 1] if len(stripped) < len(text) else ""
    if prev.isdigit() or prev in ",;、":
        return False                                  # ``,000``：紧跟在数字/逗号后的三位分组
    return True


def _separator_multi_answer(window: str) -> bool:
    """连接词之后是否紧跟**另一个数**（合法千分位分组除外）。

    * ``1,000`` → 分组恰好 3 位数字 → 合法千分位，不是多答案；
    * ``2,\\n123`` / ``2,\\nor 123`` / ``1, 2`` / ``2.5, or 7`` → 多答案；
    * ``7, so we are done`` → 连接词后不是数字 → 不是多答案。
    """
    for m in _LIST_CONNECTOR_RE.finditer(window):
        if _starts_new_number(window[m.end():]):
            return True
    return False


def _has_trailing_number(body: str, start: int) -> bool:
    """候选之后（**完整答案区域内**）是否还能读出另一个候选答案。

    数字、公式、``\\boxed{}``、``####`` 四类候选使用**同一条**区域规则：

    * 区域内出现并列形态（连接词之后紧跟数值，含跨换行）→ 多答案；
    * 或同一行内还存在别的完整数值 token。

    因此 ``2,\\nor 123`` / ``2,\\n123`` / ``\\boxed{2},\\nor 123`` / ``2.\\nOr 3.`` / ``1,2`` /
    ``$2.5$, or 7.`` 全部拒为 UNKNOWN；而 ``1,000.``、``\\boxed{2}`` 之后换行继续解释不受影响。
    """
    window = _answer_region(body, start)
    if _separator_multi_answer(window):
        return True
    line_end = window.find("\n")
    same_line = window if line_end == -1 else window[:line_end]
    return _TRAILING_NUMBER_RE.search(same_line) is not None


def extract_answer(text: str) -> tuple[str | None, str]:
    """抽**唯一**受支持候选。返回 (candidate, reason)；抽不到时 candidate=None。"""
    if text is None:
        return None, "empty_response"
    body = text

    boxes: list[dict] = []
    for m in _BOXED_RE.finditer(body):
        inner = _match_brace(body, m.end() - 1)
        if inner is not None:
            boxes.append({"inner": inner, "end": m.end() + len(inner) + 1, "empty": inner.strip() == ""})
    if boxes:
        if any(b["empty"] for b in boxes):
            return None, "empty_box"
        cands = {_clean_tail(_strip_latex_wrappers(b["inner"])) for b in boxes}
        if len(cands) > 1:
            return None, f"multiple_boxed:{len(cands)}"
        # 盒子里只有一个值，但盒后同一答案区域内还有别的候选（`\boxed{1}, or 2`）→ 多答案
        if any(_has_trailing_number(body, b["end"]) for b in boxes):
            return None, "multiple_boxed_after_box"
        return cands.pop(), "boxed"

    hashes: list[dict] = []
    for m in _HASH_RE.finditer(body):
        cleaned = _clean_tail(_strip_latex_wrappers(m.group(1)))
        if cleaned:
            hashes.append({"cleaned": cleaned, "end": m.end()})
    if hashes:
        cands = {h["cleaned"] for h in hashes}
        if len(cands) > 1:
            return None, f"multiple_hash:{len(cands)}"
        if any(_has_trailing_number(body, h["end"]) for h in hashes):
            return None, "multiple_hash_after_hash"
        return cands.pop(), "hash"

    sents: list[str] = []
    for m in _SENT_RE.finditer(body):
        raw = m.group("num") or m.group("math") or ""
        if _has_trailing_number(body, m.end()):
            # 数字候选与公式候选走**同一条**答案区域规则：
            # "1,2." / "1, 2." / "2.5, or 7." / "$2.5$, or 7." / "$1$, or $2$." / "1,\n2."
            # → 多答案/非法逗号串 → 不收候选（UNKNOWN）
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
