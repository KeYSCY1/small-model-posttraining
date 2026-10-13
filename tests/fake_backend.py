"""测试用替身后端（只替换外部模型，不复制业务循环）。

由真实入口以 ``--backend-factory tests.fake_backend:build`` 使用。
环境变量控制行为，便于在子进程里复现超时等场景：

* ``X002_FAKE_MODE``      : normal | blocking_load | blocking_run | missing_dep
* ``X002_FAKE_SLEEP``     : 阻塞秒数（默认 30）
* ``X002_FAKE_TOKENIZER`` : dict | userdict（编码返回类型）
* ``X002_FAKE_CAPTURE``   : 记录 from_pretrained 实际参数的 JSON 路径
* ``X002_FAKE_PLAN``      : 逐题计划 JSON（{problem_id: {problem, first, A, B}}）
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections import UserDict

_LAST_TEXT = {"v": ""}


def mark_ready(phase: str) -> None:
    """阶段就绪凭据：写入 ``$X002_READY_DIR/<phase>.ready``（含 pid 与时刻）。

    让测试能证明"确实到达了指定阻塞点"，而不是"还没进入就自称已测该阶段"。
    """
    d = os.environ.get("X002_READY_DIR")
    if not d:
        return
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{phase}.ready"), "w", encoding="utf-8") as fh:
        json.dump({"phase": phase, "pid": os.getpid(), "at": time.time()}, fh)


def ready_phases() -> set[str]:
    d = os.environ.get("X002_READY_DIR") or ""
    if not d or not os.path.isdir(d):
        return set()
    return {n[: -len(".ready")] for n in os.listdir(d) if n.endswith(".ready")}


def _install_sigterm_ignorer() -> None:
    """按 X002_IGNORE_SIGTERM 配置：忽略/挂起 SIGTERM，或派生一个同样忽略 SIGTERM 的子进程。

    用于复现"组长响应 SIGTERM 但同组子进程忽略 SIGTERM"与"忽略 SIGTERM 的阻塞调用"。
    """
    mode = os.environ.get("X002_IGNORE_SIGTERM", "")
    if not mode:
        return
    if mode == "self":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        mark_ready("sigterm_ignoring")          # 处理器装好之后才宣称就绪
    if mode == "child":
        env = dict(os.environ)
        env["X002_IGNORE_SIGTERM"] = "self"      # 子进程自己忽略 SIGTERM
        env["X002_FAKE_MODE"] = "blocking_load"
        env["X002_FAKE_SLEEP"] = os.environ.get("X002_FAKE_SLEEP", "30")
        subprocess.Popen([sys.executable, "-c", _CHILD_SNIPPET], env=env)   # 同组（不新开会话）
    if mode == "cooperative_leader_child":
        # 组长不忽略 SIGTERM（因此会正常退出），派生一个忽略 SIGTERM 的同组子进程
        env = dict(os.environ)
        env["X002_IGNORE_SIGTERM"] = "self"
        env["X002_FAKE_SLEEP"] = os.environ.get("X002_FAKE_SLEEP", "30")
        subprocess.Popen([sys.executable, "-c", _CHILD_SNIPPET], env=env)


_CHILD_SNIPPET = (
    "import os,signal,time,json\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "d=os.environ.get('X002_READY_DIR')\n"
    "os.makedirs(d, exist_ok=True) if d else None\n"
    "json.dump({'phase':'child_sigterm_ignoring','pid':os.getpid()},"
    " open(os.path.join(d,'child_sigterm_ignoring.ready'),'w')) if d else None\n"
    "time.sleep(float(os.environ.get('X002_FAKE_SLEEP','30')))\n"
)


class FakeTokenizer:
    """字符级假 tokenizer；编码返回类型可控。"""

    def __init__(self, ret_type: str = "dict"):
        self.ret_type = ret_type
        self.bos = 1
        self._vocab: dict[str, int] = {}
        self._rev: dict[int, str] = {}

    def _id(self, ch: str) -> int:
        if ch not in self._vocab:
            self._vocab[ch] = len(self._vocab) + 10
            self._rev[self._vocab[ch]] = ch
        return self._vocab[ch]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        _LAST_TEXT["v"] = "".join(str(m["content"]) for m in messages)
        parts = [f"<|{m['role']}|>{m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        text = "".join(parts)
        if not tokenize:
            return text
        ids = [self.bos] + [self._id(c) for c in text]
        if self.ret_type == "userdict":
            return UserDict({"input_ids": ids})
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self._rev.get(int(i), "") for i in ids)


class FakeBackend:
    def __init__(self, model_cfg: dict, generation_cfg: dict):
        _install_sigterm_ignorer()
        self.model_cfg = dict(model_cfg)
        self.generation = dict(generation_cfg)
        self.mode = os.environ.get("X002_FAKE_MODE", "normal")
        self.sleep = float(os.environ.get("X002_FAKE_SLEEP", "30"))
        self.ret_type = os.environ.get("X002_FAKE_TOKENIZER", "dict")
        self.capture = {"load_calls": [], "tokenizer_return": self.ret_type}
        self._cap_path = os.environ.get("X002_FAKE_CAPTURE")
        plan_path = os.environ.get("X002_FAKE_PLAN")
        self.plan = json.load(open(plan_path, encoding="utf-8")) if plan_path else {}
        self._tok = None

    # --- 加载 ---
    def _record(self, kind: str, revision):
        self.capture["load_calls"].append(
            {"kind": kind, "path_or_id": self.model_cfg["path_or_id"], "revision": revision})
        if self._cap_path:
            os.makedirs(os.path.dirname(os.path.abspath(self._cap_path)) or ".", exist_ok=True)
            with open(self._cap_path, "w", encoding="utf-8") as fh:
                json.dump(self.capture, fh, ensure_ascii=False, indent=2)

    def load(self):
        if self._tok is not None:
            return
        if self.mode == "missing_dep":
            raise RuntimeError("缺少依赖：模拟 torch 不可用")
        rev = self.model_cfg.get("revision")
        self._record("tokenizer", rev)
        self._record("model", rev)
        if self.mode == "blocking_load":
            mark_ready("blocking_load")          # 已进入加载阻塞点
            time.sleep(self.sleep)
        self._tok = FakeTokenizer(self.ret_type)

    @property
    def tokenizer(self):
        self.load()
        return self._tok

    # --- 生成 ---
    def run(self, input_ids, *, max_new_tokens: int, seed: int):
        if self.mode == "blocking_run":
            mark_ready("blocking_run")           # 已进入生成阻塞点
            time.sleep(self.sleep)
        blob = _LAST_TEXT["v"]
        pid = None
        for cand, entry in self.plan.items():
            if entry.get("problem", "") and entry["problem"] in blob:
                pid = cand
                break
        entry = self.plan.get(pid or "", {})
        if "此前的一次作答：" in blob:
            text = entry.get("A", "A 臂回答 \\boxed{1}")
        elif entry.get("feedback", "") in blob:
            text = entry.get("B", "B 臂回答 \\boxed{1}")
        else:
            text = entry.get("first", "首轮回答 \\boxed{0}")
        return {"text": text, "output_tokens": len(text), "input_tokens": len(input_ids),
                "finish_reason": "stop", "seconds": 0.01, "error": None}


def build(model_cfg: dict, generation_cfg: dict) -> FakeBackend:
    return FakeBackend(model_cfg, generation_cfg)
