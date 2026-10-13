"""生成后端与单次生成封装。

* 真实后端只在这里 import torch / transformers，并**使用冻结后的 revision** 加载
  tokenizer 与模型（两个 ``from_pretrained`` 都收到同一个 commit）。
* 编码只做一次：由 :func:`generate` 调用 :mod:`x002.prompts` 的唯一路径，
  核对通过后把**同一份 input_ids** 直接交给后端；结果里保留序列化文本与摘要。
* 核对失败**不调用后端**，直接返回 ``input_check_failed`` 记录。
* 时间控制由外层进程监督负责（见 :mod:`x002.worker`）：本模块不建线程、不做软停止，
  以免留下继续计算的后台任务。
"""

from __future__ import annotations

from collections.abc import Mapping

from . import prompts

__all__ = ["TransformersBackend", "BackendError", "generate_once", "load_backend"]


class BackendError(RuntimeError):
    pass


class TransformersBackend:
    """单一 transformers 路径；加载参数由冻结身份决定。"""

    def __init__(self, model_cfg: dict, generation_cfg: dict):
        self.model_cfg = dict(model_cfg)
        self.generation = dict(generation_cfg)
        self._model = None
        self._tok = None
        self.load_seconds = None

    # --- 延迟加载 ---
    def load(self) -> None:
        if self._model is not None:
            return
        import time

        try:
            import torch  # noqa: F401   # 与 transformers 分开 import，便于区分失败来源
        except Exception as exc:  # pragma: no cover
            raise BackendError(
                "torch_import_failed：无法导入 torch，请按 README §2 安装与本机 CUDA 匹配的 torch；"
                f"原始错误：{type(exc).__name__}: {exc}"
            ) from exc
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise BackendError(
                "transformers_import_failed：无法从 transformers 导入 AutoModelForCausalLM/AutoTokenizer"
                "（常见原因：transformers 与 torch 版本不兼容、缺少可选依赖）；"
                f"原始错误：{type(exc).__name__}: {exc}"
            ) from exc
        m = self.model_cfg
        revision = m.get("revision") or None   # 冻结后的 commit
        t0 = time.monotonic()
        self._tok = AutoTokenizer.from_pretrained(
            m["path_or_id"],
            revision=revision,
            trust_remote_code=bool(m.get("trust_remote_code", False)),
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            m["path_or_id"],
            revision=revision,
            torch_dtype=m.get("dtype", "auto"),
            device_map="auto",
            trust_remote_code=bool(m.get("trust_remote_code", False)),
        )
        self._model.eval()
        self.load_seconds = time.monotonic() - t0

    @property
    def tokenizer(self):
        self.load()
        return self._tok

    def run(self, input_ids, *, max_new_tokens: int, seed: int) -> dict:
        import time

        import torch

        self.load()
        g = self.generation
        ids = torch.tensor([list(input_ids)], device=self._model.device)
        attn = torch.ones_like(ids)
        torch.manual_seed(int(seed))
        t0 = time.monotonic()
        try:
            with torch.no_grad():
                out = self._model.generate(
                    input_ids=ids,
                    attention_mask=attn,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=bool(g["do_sample"]),
                    temperature=float(g["temperature"]),
                    top_p=float(g["top_p"]),
                    pad_token_id=self._tok.eos_token_id,
                )
            gen = out[0][ids.shape[1] :]
            n_new = int(gen.shape[0])
            return {
                "text": self._tok.decode(gen, skip_special_tokens=True),
                "output_tokens": n_new,
                "input_tokens": int(ids.shape[1]),
                "finish_reason": "length" if n_new >= int(max_new_tokens) else "stop",
                "seconds": round(time.monotonic() - t0, 2),
                "error": None,
            }
        except Exception as exc:  # pragma: no cover - 运行期
            return {"text": "", "output_tokens": 0, "input_tokens": int(ids.shape[1]),
                    "finish_reason": "error", "seconds": round(time.monotonic() - t0, 2),
                    "error": repr(exc)}


def generate_once(
    backend,
    messages: list[dict],
    *,
    seed: int,
    max_new_tokens: int,
    expect_prior: bool,
) -> dict:
    """编码 → 核对 → 用**同一份** input_ids 生成。核对失败不调用后端。"""
    try:
        tokenizer = backend.tokenizer
    except Exception as exc:
        # 逐字保留底层类别与文本，便于区分 torch/transformers 导入失败与 tokenizer 初始化失败
        detail = f"{type(exc).__name__}: {exc}"
        if "torch_import_failed" in detail:
            return _error_result(f"torch_import_failed:{detail}", phase="import")
        if "transformers_import_failed" in detail:
            return _error_result(f"transformers_import_failed:{detail}", phase="import")
        if "缺少依赖" in detail:
            # 替身/后端在取 tokenizer 时才报缺依赖：与"导入期失败"区分，便于定位
            return _error_result(f"generate_import_failed:{detail}", phase="import")
        return _error_result(f"tokenizer_init_failed:{detail}", phase="load")
    try:
        bundle = prompts.check_encoded(tokenizer, messages, expect_prior=expect_prior)
    except prompts.InputCheckError as exc:
        return _error_result(f"input_check_failed:{exc}", phase="encode")
    except Exception as exc:
        return _error_result(f"encode_error:{exc}", phase="encode")

    encoded = bundle["encoded"]
    check = bundle["check"]
    try:
        out = backend.run(encoded["input_ids"], max_new_tokens=max_new_tokens, seed=seed)
    except Exception as exc:
        out = {"text": "", "output_tokens": 0, "input_tokens": len(encoded["input_ids"]),
               "finish_reason": "error", "seconds": None, "error": f"backend_error:{exc}"}
    return {
        "text": out.get("text", ""),
        "usage": {"input_tokens": out.get("input_tokens"), "output_tokens": out.get("output_tokens")},
        "seconds": out.get("seconds"),
        "finish_reason": out.get("finish_reason", "stop"),
        "error": out.get("error"),
        "serialized_text": encoded["text"],
        "input_sha16": encoded["sha16"],
        "text_sha16": encoded["text_sha16"],
        "n_input_tokens": encoded["n_tokens"],
        "input_check": check,
        "phase": "generate",
    }


def _error_result(reason: str, *, phase: str) -> dict:
    return {
        "text": "",
        "usage": {"input_tokens": None, "output_tokens": None},
        "seconds": None,
        "finish_reason": "error",
        "error": reason,
        "serialized_text": None,
        "input_sha16": None,
        "text_sha16": None,
        "n_input_tokens": None,
        "input_check": {"passed": False, "reason": reason.split(":")[0]},
        "phase": phase,
    }


def load_backend(model_cfg: dict, generation_cfg: dict) -> TransformersBackend:
    return TransformersBackend(model_cfg, generation_cfg)
