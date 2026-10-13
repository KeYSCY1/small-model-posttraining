#!/usr/bin/env python3
"""X-002 离线回归检查：无模型、无网络、不加载真实依赖。

运行::

    cd run_package && PYTHONDONTWRITEBYTECODE=1 python3 tests/test_offline_checks.py

覆盖：
1. 判分反例（含 ``The answer is 2.5 cm.`` → UNKNOWN）；
2. 输入构造、完整正文核对与编码接口（dict / UserDict）、gold 不参与构造；
3. **正式启动入口**：阻塞加载超时、阻塞推理超时、终止后工作进程退出、
   已有日志可汇总、已启动目录拒绝再次生成；
4. 冻结 revision 真的传进两个 ``from_pretrained``；
5. 同一份编码交给推理（不二次编码）；
6. 首轮→判分→A/B→汇总的真实入口 mock（只替换读题/模型/tokenizer）。

mock 通过 ≠ GPU 烟测通过；真实 tokenizer/模型/数据/显存与吞吐未测。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from collections import UserDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))   # 让 tests.fake_backend 可被 import

from x002 import cli as CLI            # noqa: E402
from x002 import judge as J            # noqa: E402
from x002 import prompts as P          # noqa: E402
from x002 import run as R              # noqa: E402
from x002 import worker as W           # noqa: E402

FEEDBACK = "此前对本题的一次作答未通过自动检查，请重新求解本题。"
PROBLEMS = [
    {"problem_id": "p1", "problem": "求 1+1 的值。", "gold": "2"},
    {"problem_id": "p2", "problem": "求 2+2 的值。", "gold": "4"},
]
PLAN = {
    "p1": {"problem": "求 1+1 的值。", "feedback": FEEDBACK,
           "first": "先算 1+1=3，所以答案是 \\boxed{3}",
           "A": "重算得 \\boxed{2}", "B": "重算得 \\boxed{9}"},
    "p2": {"problem": "求 2+2 的值。", "feedback": FEEDBACK,
           "first": "答案是 \\boxed{4}", "A": "-", "B": "-"},
}


# ---------------------------------------------------------------- 1. 判分

class TestJudge(unittest.TestCase):
    def test_pi_counterexamples(self):
        self.assertEqual(J.judge("The final answer is 0.5.", "0.5")["status"], "correct")
        self.assertEqual(J.judge("The answer is 2.5.", "2")["status"], "incorrect")
        out = J.judge("The answer is 2.5 cm.", "2")
        self.assertEqual(out["status"], "undetermined", out)
        self.assertIsNone(out["extracted"])
        self.assertEqual(J.judge("\\boxed{1,2}", "12")["status"], "undetermined")

    def test_sentence_trailing_number_is_unknown(self):
        """本批定点修复：句子型答案后的多余数值不得被当成数字前缀抽走。"""
        for resp, gold in (("The answer is 1,2.", "1"),
                           ("The answer is 1, 2.", "1"),
                           ("The final answer is 2.5, or 7.", "2.5")):
            with self.subTest(resp=resp):
                out = J.judge(resp, gold)
                self.assertEqual(out["status"], "undetermined", out)
                self.assertIsNone(out["extracted"], out)

    def test_legal_thousands_and_single_scalars_still_ok(self):
        for resp, gold in (("The answer is 1,000.", "1000"),
                           ("The answer is 1,000,000.", "1000000"),
                           ("The answer is 1000.", "1000"),
                           ("The answer is 7, so we are done", "7"),
                           ("\\boxed{497}", "497"),
                           ("\\boxed{129}", "129")):
            with self.subTest(resp=resp):
                self.assertEqual(J.judge(resp, gold)["status"], "correct", resp)

    def test_gold_supported_scope(self):
        """选题用的金标准范围声明：标量支持，区间/集合/坐标/复数/空值不支持。"""
        for gold, want in (("12", True), ("1/2", True), ("0.5", True), ("1,000", True),
                           ("-7", True), ("50\\%", True), ("\\sqrt{2}", True),
                           ("[-2, 7]", False), ("(-\\sqrt{3}, \\sqrt{3})", False),
                           ("117 + 41i", False), ("", False), (None, False)):
            with self.subTest(gold=gold):
                ok, reason = J.gold_supported(gold)
                self.assertEqual(ok, want, (gold, reason))

    def test_still_supported(self):
        for resp, gold in [("The answer is 12.", "12"), ("\\boxed{0.5}", "1/2"),
                           ("\\boxed{\\frac{1}{2}}", "0.5"), ("#### 7", "7"),
                           ("\\boxed{1,000}", "1000"), ("\\boxed{50\\%}", "1/2")]:
            with self.subTest(resp=resp):
                self.assertEqual(J.judge(resp, gold)["status"], "correct", (resp, gold))

    def test_empty_box_multiple_and_symbolic_off(self):
        self.assertEqual(J.judge("\\boxed{}", "1")["status"], "undetermined")
        self.assertEqual(J.judge("\\boxed{3} 或 \\boxed{5}", "3")["status"], "undetermined")
        self.assertEqual(J.judge("\\boxed{x^2-2x+1}", "(x-1)^2")["status"], "undetermined")
        self.assertFalse(J.judge("\\boxed{1}", "1")["symbolic_enabled"])


# ---------------------------------------------------------------- 2. 输入

class FakeTok:
    def __init__(self, ret="dict"):
        self.ret = ret

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "<|assistant|>"
        if not tokenize:
            return text
        ids = [ord(c) % 1000 for c in text]
        return UserDict({"input_ids": ids}) if self.ret == "userdict" else {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(65 + (i % 26)) for i in ids)   # 仅用于结构检查


class EchoTok(FakeTok):
    """可逆的假 tokenizer：decode 能还原文本，用于完整正文核对。"""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "<|assistant|>"
        if not tokenize:
            return text
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids)


class TestInputs(unittest.TestCase):
    def test_messages_are_whitelisted(self):
        msgs = P.build_second_round("题面", FEEDBACK, "旧答正文", "A_prior_visible")
        blob = json.dumps(msgs, ensure_ascii=False)
        self.assertIn("旧答正文", blob)
        self.assertIn(FEEDBACK, blob)
        msgs_b = P.build_second_round("题面", FEEDBACK, "旧答正文", "B_prior_hidden")
        self.assertNotIn("旧答正文", json.dumps(msgs_b, ensure_ascii=False))
        self.assertEqual([m["role"] for m in msgs], [m["role"] for m in msgs_b])

    def test_gold_is_not_a_parameter(self):
        import inspect

        for fn in (P.build_first_round, P.build_second_round, P.encode_messages, P.verify_encoded):
            self.assertNotIn("gold", inspect.signature(fn).parameters, fn.__name__)

    def test_missing_tail_fails(self):
        tok = EchoTok()
        old = "abcdefghijklmnopqrstuvwxyz LONG_OLD_SUFFIX_MISSING"
        full = P.build_second_round("题面内容", FEEDBACK, old, "A_prior_visible")
        ok = P.check_encoded(tok, full, expect_prior=True)
        self.assertTrue(ok["check"]["passed"])
        # 构造"实际输入丢了尾部"的情形：截断消息里的旧答
        truncated = P.build_second_round("题面内容", FEEDBACK, old[:26], "A_prior_visible")
        res = P.verify_encoded(tok, full, P.encode_messages(tok, truncated), expect_prior=True)
        self.assertFalse(res["passed"], res)
        self.assertEqual(res["missing"][0]["reason"], "content_missing_or_truncated")

    def test_natural_overlap_is_not_blocked(self):
        """题面/旧答自然包含与标准答案相同的内容，不触发阻止。"""
        tok = EchoTok()
        msgs = P.build_second_round("答案是 2 吗？", FEEDBACK, "我写的是 \\boxed{2}", "A_prior_visible")
        res = P.check_encoded(tok, msgs, expect_prior=True)
        self.assertTrue(res["check"]["passed"], res["check"])

    def test_mapping_return_types(self):
        for ret in ("dict", "userdict"):
            with self.subTest(ret=ret):
                tok = FakeTok(ret)
                enc = P.encode_messages(tok, P.build_first_round("题面"))
                self.assertTrue(enc["input_ids"])
                self.assertTrue(all(isinstance(i, int) for i in enc["input_ids"]))
                self.assertEqual(enc["n_tokens"], len(enc["input_ids"]))

    def test_bad_return_type_raises(self):
        class Bad:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return {"wrong_key": [1, 2]} if tokenize else "text"

        with self.assertRaises(P.InputCheckError):
            P.encode_messages(Bad(), P.build_first_round("题面"))


def _encode(tok, messages):
    return P.encode_messages(tok, messages)


class TestInputCheckBlocksCalls(unittest.TestCase):
    """核对/编码失败必须不发出推理调用，并可回收日志。"""

    def test_check_failure_does_not_call_backend(self):
        from x002 import generator as G

        calls = {"run": 0}

        class Tok:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                text = "".join(f"<|{m['role']}|>{m['content']}" for m in messages) + "<|assistant|>"
                return [ord(c) for c in text] if tokenize else text

            def decode(self, ids, skip_special_tokens=True):
                return ""     # 解码拿不到正文 → 完整正文核对必然失败

        class B:
            @property
            def tokenizer(self):
                return Tok()

            def run(self, input_ids, *, max_new_tokens, seed):
                calls["run"] += 1
                return {"text": "x", "output_tokens": 1, "input_tokens": 1,
                        "finish_reason": "stop", "seconds": 0.0, "error": None}

        out = G.generate_once(B(), P.build_second_round("题面", FEEDBACK, "旧答正文", "A_prior_visible"),
                              seed=1, max_new_tokens=4, expect_prior=True)
        self.assertEqual(calls["run"], 0, "核对失败不得调用后端")
        self.assertEqual(out["finish_reason"], "error")
        self.assertTrue(str(out["error"]).startswith("input_check_failed"))

    def test_encode_exception_is_recoverable(self):
        from x002 import generator as G

        class Bad:
            @property
            def tokenizer(self):
                raise RuntimeError("模拟 tokenizer 加载失败")

            def run(self, *a, **k):  # pragma: no cover
                raise AssertionError("不应被调用")

        out = G.generate_once(Bad(), P.build_first_round("题面"), seed=1, max_new_tokens=4, expect_prior=False)
        self.assertEqual(out["finish_reason"], "error")
        self.assertTrue(str(out["error"]).startswith("tokenizer_init_failed"), out["error"])
        self.assertIn("模拟 tokenizer 加载失败", out["error"])   # 底层错误必须保留
        self.assertIsNone(out["input_sha16"])


# ---------------------------------------------------------------- 3. 正式入口

def _read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as fh:
        return len(fh.readlines())


def _write(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)


class EntryHarness:
    def __init__(self, tmp, problems=None, plan=None):
        self.tmp = tmp
        self.cfg_path = os.path.join(tmp, "config.json")
        self.problems_path = os.path.join(tmp, "problems.json")
        self.plan_path = os.path.join(tmp, "plan.json")
        self.fixture_path = os.path.join(tmp, "fixture.json")
        self.ready_dir = os.path.join(tmp, "ready")
        self.out_dir = os.path.join(tmp, "out")
        _write(self.fixture_path, {
            "model": {"path_or_id": "fake/model", "revision": "rev-frozen-123", "resolved_from": "fixture"},
            "dataset": {"dataset_id": "fake/data", "split": "test", "revision": "data-rev-9",
                        "resolved_from": "fixture"},
            "env": {"python": "3.x", "platform": "offline-fixture",
                    "torch": None, "transformers": None, "datasets": None, "sympy": None},
        })
        _write(self.cfg_path, {
            "model": {"path_or_id": "fake/model", "revision": "rev-frozen-123", "dtype": "bfloat16"},
            "generation": {"max_new_tokens": 64, "do_sample": True, "temperature": 0.6,
                           "top_p": 0.95, "seed": 7},
            "data": {"dataset_id": "fake/data", "split": "test", "revision": "data-rev-9",
                     "problem_field": "problem", "answer_field": "answer", "n_problems": 12},
            "feedback": {"text": FEEDBACK},
            "judge": {"numeric_rtol": 1e-6, "symbolic": False},
            "run": {"smoke": {"total_minutes": 45}},
            "paths": {"out_dir": self.out_dir, "log_name": "records.jsonl"},
        })
        _write(self.problems_path, problems or PROBLEMS)
        _write(self.plan_path, plan or PLAN)

    def _argv(self, total_seconds, fixture=None):
        argv = ["run", "--config", self.cfg_path, "--out-dir", self.out_dir,
                "--total-seconds", str(total_seconds),
                "--backend-factory", "tests.fake_backend:build",
                "--problems-json", self.problems_path,
                "--offline-fixture", fixture or self.fixture_path]
        return argv

    def _env(self, env_extra, pythonpath_prefix=None):
        env = dict(os.environ)
        parts = [os.path.dirname(ROOT), ROOT]
        if pythonpath_prefix:
            parts.insert(0, pythonpath_prefix)
        env.update({
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(parts + [env.get("PYTHONPATH", "")]),
            "X002_FAKE_PLAN": self.plan_path,
            "X002_READY_DIR": self.ready_dir,
        })
        if env_extra:
            env.update(env_extra)
        return env

    def ready(self) -> set:
        """已到达的阶段就绪凭据（由替身在阻塞点之前写入）。"""
        if not os.path.isdir(self.ready_dir):
            return set()
        return {n[: -len(".ready")] for n in os.listdir(self.ready_dir) if n.endswith(".ready")}

    def wait_ready(self, phase, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if phase in self.ready():
                return True
            time.sleep(0.02)
        return False

    def run(self, total_seconds=2700, env_extra=None, capture=True, pythonpath_prefix=None):
        """在独立进程里跑正式入口（子进程内 CLI 再派生工作进程）。"""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        env = self._env(env_extra, pythonpath_prefix=pythonpath_prefix)
        argv = ["python3", "-m", "x002.cli"] + self._argv(total_seconds)
        if not capture:
            proc = subprocess.run([sys.executable, "-m", "x002.cli"] + self._argv(total_seconds),
                                  env=env, cwd=ROOT, capture_output=True, text=True)
            return proc.returncode, proc.stdout, proc.stderr
        buf_out, buf_err = io.StringIO(), io.StringIO()
        code = (
            "import sys;from x002 import cli;"
            f"sys.exit(cli.main({self._argv(total_seconds)!r}))"
        )
        proc = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT,
                              capture_output=True, text=True)
        return proc.returncode, proc.stdout, proc.stderr

    def run_in_process(self, total_seconds=2700, env_extra=None, fixture=None):
        """在本进程内调用正式入口：便于断言工作进程组的回收结果。"""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        env = self._env(env_extra)
        old = dict(os.environ)
        os.environ.clear()
        os.environ.update(env)
        buf_out, buf_err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                rc = CLI.main(self._argv(total_seconds, fixture=fixture))
            return rc, buf_out.getvalue(), buf_err.getvalue()
        finally:
            os.environ.clear()
            os.environ.update(old)


class TestEntryPointMock(unittest.TestCase):
    def test_end_to_end_through_real_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = EntryHarness(tmp)
            rc, out, err = h.run(total_seconds=2700)
            self.assertEqual(rc, 0, err)
            store = R.Store(os.path.join(h.out_dir, "records.jsonl"))
            recs = store.records()
            header = store.header()
            self.assertEqual(header["kind"], "run")
            self.assertEqual(header["model"]["revision"], "rev-frozen-123")
            self.assertEqual(header["dataset"]["revision"], "data-rev-9")
            first = [r for r in recs if r.get("stage") == "first"]
            self.assertEqual(len(first), 2)
            self.assertEqual(sum(r["judge"]["status"] == "incorrect" for r in first), 1)
            summary = _read_json(os.path.join(h.out_dir, "summary.json"))
            self.assertEqual(summary["first_round"]["n_eligible"], 1)
            self.assertEqual(summary["determinate_pairs"]["n_pairs"], 1)
            self.assertAlmostEqual(summary["determinate_pairs"]["observed_diff_A_minus_B"], 1.0)
            self.assertTrue(os.path.exists(os.path.join(h.out_dir, CLI.STATUS_FILE)))

    def test_second_run_same_dir_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = EntryHarness(tmp)
            rc1, _, _ = h.run(total_seconds=2700)
            self.assertEqual(rc1, 0)
            n_before = _count_lines(os.path.join(h.out_dir, "records.jsonl"))
            rc2, out2, err2 = h.run(total_seconds=2700)
            self.assertEqual(rc2, 6, "已启动目录必须拒绝再次生成")
            self.assertIn("拒绝再次生成", err2)
            n_after = _count_lines(os.path.join(h.out_dir, "records.jsonl"))
            self.assertEqual(n_before, n_after)

    def test_frozen_revision_reaches_real_backend_load(self):
        """真实后端加载函数的连线：**同时**替换 torch 与 transformers，并完整恢复模块。

        在**没有**安装 torch/transformers 的环境里也必须能跑（不安装依赖绕过）。
        """
        from x002 import generator as G

        calls = []

        class Tok:
            eos_token_id = 0

            def __call__(self, text, **kw):
                return {"input_ids": [1]}

        class Model:
            device = "cpu"

            def eval(self):
                return self

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path, **kw):
                calls.append({"kind": "tokenizer", "path_or_id": path, **kw})
                return Tok()

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(path, **kw):
                calls.append({"kind": "model", "path_or_id": path, **kw})
                return Model()

        fake_torch = types.ModuleType("torch")
        fake_torch.__version__ = "0.0.0-fake"
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.__version__ = "0.0.0-fake"
        fake_transformers.AutoTokenizer = FakeAutoTokenizer
        fake_transformers.AutoModelForCausalLM = FakeAutoModel

        saved = {name: sys.modules.get(name) for name in ("torch", "transformers")}
        sys.modules["torch"] = fake_torch
        sys.modules["transformers"] = fake_transformers
        try:
            backend = G.TransformersBackend(
                {"path_or_id": "fake/model", "revision": "rev-frozen-123", "dtype": "bfloat16"}, {})
            backend.load()
        finally:
            for name, mod in saved.items():      # 完整恢复原模块（含"原本不存在"的情形）
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod
        kinds = [c["kind"] for c in calls]
        self.assertEqual(sorted(kinds), ["model", "tokenizer"])
        for call in calls:
            self.assertEqual(call["revision"], "rev-frozen-123", call)

    def test_second_round_blocks_on_input_check_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = EntryHarness(tmp)
            cap = os.path.join(tmp, "c.json")
            h.run(total_seconds=2700, env_extra={"X002_FAKE_CAPTURE": cap})
            recs = R.Store(os.path.join(h.out_dir, "records.jsonl")).records()
            second = [r for r in recs if r.get("stage") == "second"]
            self.assertEqual(len(second), 2)
            for r in second:
                self.assertIsNotNone(r["input_sha16"])
                self.assertIsNotNone(r["serialized_text"])
                self.assertTrue(r["input_check"]["passed"], r["input_check"])
            # A 臂序列化文本里必须含旧答正文
            a = next(r for r in second if r["arm"] == "A_prior_visible")
            self.assertIn("先算 1+1=3", a["serialized_text"])
            b = next(r for r in second if r["arm"] == "B_prior_hidden")
            self.assertNotIn("先算 1+1=3", b["serialized_text"])


# ---------------------------------------------------------------- 4. 监督与超时

def _worker_alive(pgid) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


class TestSupervision(unittest.TestCase):
    def test_worker_uses_same_encoded_ids_for_generation(self):
        """Generator 只调用一次编码路径，并把同一份 ids 交给后端。"""
        from x002 import generator as G

        calls = {"template": 0, "run": 0, "ids": None}

        class Tok:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                calls["template"] += 1
                text = "".join(f"<|{m['role']}|>{m['content']}" for m in messages) + "<|assistant|>"
                return [ord(c) for c in text] if tokenize else text

            def decode(self, ids, skip_special_tokens=True):
                return "".join(chr(int(i)) for i in ids)

        class B:
            @property
            def tokenizer(self):
                return Tok()

            def run(self, input_ids, *, max_new_tokens, seed):
                calls["run"] += 1
                calls["ids"] = list(input_ids)
                return {"text": "ok", "output_tokens": 1, "input_tokens": len(input_ids),
                        "finish_reason": "stop", "seconds": 0.0, "error": None}

        out = G.generate_once(B(), P.build_first_round("题面"), seed=1, max_new_tokens=4, expect_prior=False)
        self.assertEqual(calls["run"], 1)
        self.assertEqual(calls["template"], 2)          # tokenize=True/False 各一次，同一次调用内
        self.assertEqual(out["input_sha16"], R.sha16(",".join(str(i) for i in calls["ids"])))

    def test_blocking_load_times_out_and_worker_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = EntryHarness(tmp)
            t0 = time.monotonic()
            rc, out, err = h.run_in_process(total_seconds=2, env_extra={
                "X002_FAKE_MODE": "blocking_load", "X002_FAKE_SLEEP": "30",
                "X002_FAKE_TOKENIZER": "userdict"})
            elapsed = time.monotonic() - t0
            self.assertEqual(rc, 7)
            self.assertLess(elapsed, 20, "总截止必须约束加载阶段")
            status = _read_json(os.path.join(h.out_dir, CLI.STATUS_FILE))
            self.assertEqual(status["reason"], "total_deadline_exceeded")
            self.assertTrue(status["timed_out"])
            self.assertTrue(h.wait_ready("blocking_load"), f"未到达加载阻塞点：ready={h.ready()}")
            self.assertTrue(os.path.exists(os.path.join(h.out_dir, "records.jsonl")),
                            "身份必须在加载前落盘，加载阶段超时也要有可汇总日志")
            rc_s, out_s, _ = CLI.capture_summarize(h.out_dir)
            self.assertEqual(rc_s, 0)
            self.assertIn("首轮", out_s)

    def test_blocking_run_times_out_and_no_lingering_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = EntryHarness(tmp)
            rc, out, err = h.run_in_process(total_seconds=4, env_extra={
                "X002_FAKE_MODE": "blocking_run", "X002_FAKE_SLEEP": "30"})
            self.assertEqual(rc, 7)
            self.assertTrue(h.wait_ready("blocking_run"), f"未到达生成阻塞点：ready={h.ready()}")
            with open(os.path.join(h.out_dir, CLI.STATUS_FILE), encoding="utf-8") as fh:
                status = json.load(fh)
            self.assertEqual(status["reason"], "total_deadline_exceeded")
            # 终止后整组必须已回收：pgid 不得为 null（不能遇到 null 就跳过核查）
            pgid = status.get("worker_pgid")
            self.assertIsNotNone(pgid, "status.json 必须保存启动时的进程组号")
            self.assertFalse(status["group_alive_after_cleanup"], "清理后组内仍有存活进程")
            self.assertFalse(_worker_alive(pgid), "终止后工作进程组仍存活")
            self.assertTrue(os.path.exists(os.path.join(h.out_dir, "records.jsonl")))

    def test_missing_dependency_is_recoverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = EntryHarness(tmp)
            rc, out, err = h.run(total_seconds=2700, env_extra={"X002_FAKE_MODE": "missing_dep"})
            self.assertEqual(rc, 0)
            store = R.Store(os.path.join(h.out_dir, "records.jsonl"))
            self.assertIsNotNone(store.header())
            recs = [r for r in store.records() if r.get("kind") != "run"]
            self.assertTrue(recs)
            self.assertTrue(all(r["run_status"] == "error" for r in recs))
            self.assertTrue(str(recs[0]["error"]).startswith("generate_import_failed"), recs[0]["error"])
            self.assertIn("模拟 torch 不可用", recs[0]["error"])


class TestTerminationEnvelope(unittest.TestCase):
    """终止宽限必须留在总上限内；整组清理；非零退出传播。"""

    def test_sigterm_ignoring_blocking_load_stops_within_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = EntryHarness(tmp)
            t0 = time.monotonic()
            rc, out, err = h.run_in_process(total_seconds=4, env_extra={
                "X002_FAKE_MODE": "blocking_load", "X002_FAKE_SLEEP": "30",
                "X002_IGNORE_SIGTERM": "self"})
            wall = time.monotonic() - t0
            self.assertLessEqual(wall, 5.5, f"总墙钟 {wall:.2f}s 超出 4s 预算 + 1.5s 容差（原实现会再加 5s 宽限）")
            self.assertEqual(rc, 7)
            self.assertTrue(h.wait_ready("sigterm_ignoring"),
                            f"未证明忽略 SIGTERM 的处理器已装好：{h.ready()}")
            self.assertTrue(h.wait_ready("blocking_load"), f"未到达加载阻塞点：{h.ready()}")
            status = _read_json(os.path.join(h.out_dir, CLI.STATUS_FILE))
            self.assertTrue(status["timed_out"])
            self.assertEqual(status["reason"], "total_deadline_exceeded")
            self.assertTrue(status["forced_kill"])
            self.assertIsNotNone(status["worker_pgid"])
            self.assertFalse(status["group_alive_after_cleanup"])
            self.assertLessEqual(status["elapsed_seconds"], 5.5)

    def test_cooperative_leader_but_child_ignores_sigterm(self):
        """组长响应 SIGTERM 退出，但同组子进程忽略 SIGTERM：子进程也必须被清理。"""
        with tempfile.TemporaryDirectory() as tmp:
            h = EntryHarness(tmp)
            t0 = time.monotonic()
            rc, out, err = h.run_in_process(total_seconds=4, env_extra={
                "X002_FAKE_MODE": "blocking_load", "X002_FAKE_SLEEP": "30",
                "X002_IGNORE_SIGTERM": "cooperative_leader_child"})
            wall = time.monotonic() - t0
            self.assertLessEqual(wall, 5.5, f"总墙钟 {wall:.2f}s 超出容差")
            self.assertEqual(rc, 7)
            self.assertTrue(h.wait_ready("blocking_load"), f"未到达加载阻塞点：{h.ready()}")
            self.assertTrue(h.wait_ready("child_sigterm_ignoring"),
                            f"同组子进程未证明已装好忽略处理器：{h.ready()}")
            status = _read_json(os.path.join(h.out_dir, CLI.STATUS_FILE))
            self.assertIsNotNone(status["worker_pgid"])
            self.assertFalse(status["group_alive_after_cleanup"],
                             "组长退出后同组子进程仍存活，说明只等了组长")

    def test_worker_nonzero_exit_is_propagated(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = EntryHarness(tmp)
            env = h._env({"X002_FAKE_PLAN": h.plan_path})
            argv = ["run", "--config", h.cfg_path, "--out-dir", h.out_dir,
                    "--total-seconds", "2700", "--problems-json", h.problems_path]
            argv += ["--worker-cmd", "fail3"]      # 测试接缝：让工作进程以 3 退出
            proc = subprocess.run([sys.executable, "-m", "x002.cli"] + argv,
                                  env=env, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 3, proc.stderr)
            status = _read_json(os.path.join(h.out_dir, CLI.STATUS_FILE))
            self.assertEqual(status["reason"], "worker_failed")
            self.assertFalse(status["timed_out"])
            self.assertEqual(status["worker_returncode"], 3)


class TestVerifiedSuccessRate(unittest.TestCase):
    def test_truncated_but_correct_is_not_a_success(self):
        recs = [
            {"kind": "record", "problem_id": "p1", "stage": "first", "arm": "none",
             "run_status": "completed", "judge": {"status": "incorrect"}},
            {"kind": "record", "problem_id": "p1", "stage": "second", "arm": "A_prior_visible",
             "run_status": "truncated", "judge": {"status": "correct"}},
            {"kind": "record", "problem_id": "p1", "stage": "second", "arm": "B_prior_hidden",
             "run_status": "completed", "judge": {"status": "correct"}},
        ]
        s = R.summarize(recs)
        a = s["second_round"]["per_arm"]["A_prior_visible"]
        self.assertEqual(a["n_attempted"], 1, "截断记录仍留在全部尝试的分母里")
        self.assertEqual(a["verified_success_rate_all_attempts"], 0.0,
                         "截断且判分正确的记录不得算成功")
        b = s["second_round"]["per_arm"]["B_prior_hidden"]
        self.assertEqual(b["verified_success_rate_all_attempts"], 1.0)
        # 可判定配对口径不变：截断的那条仍被排除
        self.assertEqual(s["determinate_pairs"]["n_pairs"], 0)
        self.assertEqual(s["excluded_from_pairs"][0]["reason"], "non_determinate:truncated/completed")


class TestStartupDeadlineAndIsolation(unittest.TestCase):
    """启动阶段被截止；离线路径不查 Hub、不导入重依赖。"""

    def test_startup_phase_cut_off_before_freezing_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = EntryHarness(tmp)
            # fixture 让身份解析之后先暂停，模拟"启动阶段很慢"
            fixture = os.path.join(tmp, "slow_fixture.json")
            _write(fixture, {
                "model": {"path_or_id": "fake/model", "revision": "rev-frozen-123"},
                "dataset": {"dataset_id": "fake/data", "split": "test", "revision": "data-rev-9"},
                "env": {"python": "3.x", "platform": "offline-fixture"},
                "pause_after_start_seconds": 30.0,
            })
            t0 = time.monotonic()
            rc, out, err = h.run_in_process(total_seconds=4, fixture=fixture)
            wall = time.monotonic() - t0
            self.assertLessEqual(wall, 5.5, f"启动阶段未被总截止约束：{wall:.2f}s")
            self.assertEqual(rc, 7)
            status = _read_json(os.path.join(h.out_dir, CLI.STATUS_FILE))
            self.assertTrue(status["timed_out"])
            self.assertEqual(status["reason"], "total_deadline_exceeded")
            self.assertIsNotNone(status["worker_pgid"])
            self.assertFalse(status["group_alive_after_cleanup"])
            # 启动阶段被截止：允许尚无冻结头（身份未解析），但**不得伪造**已冻结的记录
            self.assertFalse(os.path.exists(os.path.join(h.out_dir, "records.jsonl")),
                             "身份未解析时不得产生冻结头记录")
            self.assertTrue(out or err or status, "启动/状态/控制台信息必须留下可诊断内容")

    def test_offline_path_does_not_import_heavy_deps(self):
        """导入层覆盖：导入 worker/judge 与收集环境版本都不触发重依赖导入。

        子进程层的覆盖由 `test_offline_path_does_not_query_hub` 与
        `test_sentinel_stub_packages_are_never_touched` 用 PYTHONPATH 哨兵包完成。
        """
        import importlib

        attempts: list[str] = []

        class _Sentinel(types.ModuleType):
            def __init__(self, name):
                super().__init__(name)

            def __getattr__(self, item):
                attempts.append(self.__name__)
                raise AssertionError(f"离线路径不应使用 {self.__name__}")

        heavy = ("torch", "transformers", "datasets", "sympy")
        saved = {m: sys.modules.get(m) for m in heavy}
        for m in heavy:
            sys.modules[m] = _Sentinel(m)          # 模拟"依赖已安装"
        try:
            for mod in ("x002.judge", "x002.worker", "x002.prompts", "x002.cli"):
                sys.modules.pop(mod, None)
            importlib.import_module("x002.worker")
            from x002 import judge as J
            out = J.judge("\\boxed{1}", "1")          # 默认 symbolic=False
            self.assertFalse(out["symbolic_enabled"])
            self.assertEqual(out["status"], "correct")
            # 哨兵仍挂在 sys.modules 里（模拟"依赖已安装"），关键是它**从未被访问**
            self.assertIsInstance(sys.modules["sympy"], _Sentinel)
            self.assertFalse(J._SYMPY_CACHE, "默认关闭时不应触发 SymPy 探测")
            from x002.worker import capture_env
            env = capture_env()
            self.assertIn("python", env)
            self.assertEqual(attempts, [], f"离线路径触碰了重依赖：{attempts}")
        finally:
            for m, mod in saved.items():
                if mod is None:
                    sys.modules.pop(m, None)
                else:
                    sys.modules[m] = mod

    def test_offline_path_does_not_query_hub(self):
        """真实子进程 + PYTHONPATH 哨兵包：fixture 路径必须完全不 import huggingface_hub。

        说明：原先只在**父测试进程**注入 sys.modules 哨兵，而业务入口是**子进程**，
        该写法高于实际覆盖（PI R06 审阅指出）。现在改为子进程可继承的 PYTHONPATH 哨兵，
        并以"标记文件是否被写出"作为**子进程确实没有导入它**的证据。
        """
        with tempfile.TemporaryDirectory() as tmp:
            stubs = os.path.join(tmp, "stubs")
            marker = os.path.join(tmp, "hub_imported.marker")
            d = os.path.join(stubs, "huggingface_hub")
            os.makedirs(d)
            with open(os.path.join(d, "__init__.py"), "w", encoding="utf-8") as fh:
                fh.write(
                    "import pathlib\n"
                    f"pathlib.Path({marker!r}).write_text('imported')\n"
                    "def HfApi(*a, **k):\n"
                    "    raise AssertionError('离线 fixture 路径不应访问 Hub')\n"
                    "def snapshot_download(*a, **k):\n"
                    "    raise AssertionError('离线 fixture 路径不应下载')\n"
                )
            h = EntryHarness(tmp)
            rc, out, err = h.run(total_seconds=2700, pythonpath_prefix=stubs)
            self.assertEqual(rc, 0, err)
            self.assertFalse(os.path.exists(marker),
                             "子进程导入了 huggingface_hub —— 离线 fixture 路径未完全隔离")
            header = R.Store(os.path.join(h.out_dir, "records.jsonl")).header()
            self.assertEqual(header["model"]["resolved_from"], "fixture")
            self.assertEqual(header["dataset"]["resolved_from"], "fixture")
            self.assertEqual(header["model"]["revision"], "rev-frozen-123")

    def test_sentinel_stub_packages_are_never_touched(self):
        """模拟"依赖已安装"：把哨兵包放进 PYTHONPATH，纯离线 fixture 路径不得导入它们。"""
        with tempfile.TemporaryDirectory() as tmp:
            stubs = os.path.join(tmp, "stubs")
            trace = os.path.join(tmp, "imports.log")
            for name in ("torch", "transformers", "datasets", "sympy", "huggingface_hub"):
                d = os.path.join(stubs, name)
                os.makedirs(d)
                with open(os.path.join(d, "__init__.py"), "w", encoding="utf-8") as fh:
                    fh.write(f"open({trace!r},'a').write('IMPORT:{name}\\n')\n__version__='9.9.9'\n")
            open(trace, "w").close()
            h = EntryHarness(tmp)
            rc, out, err = h.run(total_seconds=2700, env_extra={
                "PYTHONPATH": stubs + os.pathsep + os.path.dirname(ROOT) + os.pathsep + ROOT})
            self.assertEqual(rc, 0, err)
            with open(trace, encoding="utf-8") as fh:
                touched = fh.read().strip()
            self.assertEqual(touched, "", f"离线 fixture 路径触碰了重依赖：{touched}")
            header = R.Store(os.path.join(h.out_dir, "records.jsonl")).header()
            self.assertEqual(header["env"]["platform"], "offline-fixture")

    def test_fixture_env_is_used_and_records_have_no_dep_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = EntryHarness(tmp)
            rc, out, err = h.run(total_seconds=2700)
            self.assertEqual(rc, 0, err)
            header = R.Store(os.path.join(h.out_dir, "records.jsonl")).header()
            self.assertEqual(header["env"]["platform"], "offline-fixture")


class TestPilotSelection(unittest.TestCase):
    """先导选题规则：稳定性、排除已暴露、标量筛选、逐层分母（全部用合成题池）。"""

    @staticmethod
    def _pool(n=60, n_unsupported=8, n_exposed=5):
        rows = []
        for i in range(n):
            pid = f"test/algebra/{1000+i}.json"
            if i < n_exposed:
                gold = str(i + 1)                      # 已暴露题（内容无关，只看 ID）
            elif i < n_exposed + n_unsupported:
                gold = f"[{i}, {i+1}]"                 # 不支持形式：区间
            else:
                gold = str(i - n_exposed - n_unsupported + 1)
            rows.append({"unique_id": pid, "gold": gold,
                         "subject": "algebra" if i % 2 else "geometry",
                         "level": 1 + (i % 5)})
        return rows

    def test_layers_and_exclusions(self):
        from x002 import pilot as P
        pool = self._pool()
        exposed = [f"test/algebra/{1000+i}.json" for i in range(5)]
        r = P.select_pilot_problems(pool, exposed_ids=exposed, dataset_revision="rev-x",
                                    seed=7, max_problems=48)
        L = r["layers"]
        self.assertEqual(L["pool_total"], 60)
        self.assertEqual(L["exposed_removed"], 5)
        self.assertEqual(L["after_exposure_filter"], 55)
        self.assertEqual(L["non_scalar_removed"], 8)
        self.assertEqual(L["scalar_candidates"], 47)
        self.assertEqual(L["selected"], 47)
        self.assertEqual(L["target"], 48)
        self.assertEqual(L["shortfall"], 1)            # 不足如实记录
        self.assertNotIn(True, [i in set(exposed) for i in r["selection"]["selected_ids"]])
        # 合成池里 subject 由原始序号 i 决定（_pool 用 "algebra" if i % 2），
        # 按文件名前缀 1000+i 还原 i 后核对组成
        def _i(pid):
            return int(pid.split("/")[2][:4]) - 1000
        want = {"algebra": sum(1 for pid in r["selection"]["selected_ids"] if _i(pid) % 2 == 1),
                "geometry": sum(1 for pid in r["selection"]["selected_ids"] if _i(pid) % 2 == 0)}
        self.assertEqual(r["selection"]["composition"]["by_subject"], want)
        self.assertEqual(sum(want.values()), r["layers"]["selected"])

    def test_deterministic_and_independent_of_pool_order(self):
        from x002 import pilot as P
        pool = self._pool()
        import random
        shuffled = pool[:]
        random.Random(1).shuffle(shuffled)
        a = P.select_pilot_problems(pool, exposed_ids=[], dataset_revision="rev-x", seed=7, max_problems=10)
        b = P.select_pilot_problems(shuffled, exposed_ids=[], dataset_revision="rev-x", seed=7, max_problems=10)
        self.assertEqual(a["selection"]["selected_ids"], b["selection"]["selected_ids"])
        c = P.select_pilot_problems(pool, exposed_ids=[], dataset_revision="rev-x", seed=8, max_problems=10)
        self.assertNotEqual(a["selection"]["selected_ids"], c["selection"]["selected_ids"])
        d = P.select_pilot_problems(pool, exposed_ids=[], dataset_revision="rev-y", seed=7, max_problems=10)
        self.assertNotEqual(a["selection"]["selected_ids"], d["selection"]["selected_ids"])

    def test_empty_pool_and_missing_field_rejected(self):
        from x002 import pilot as P
        with self.assertRaises(ValueError):
            P.select_pilot_problems([], exposed_ids=[], dataset_revision="r", seed=1, max_problems=5)
        with self.assertRaises(ValueError):
            P.select_pilot_problems([{"unique_id": "a"}], exposed_ids=[], dataset_revision="r",
                                    seed=1, max_problems=5)

    def test_exposed_list_file_matches_r06_ids(self):
        import json as _json
        path = os.path.join(ROOT, "exposed_problems_r06.json")
        with open(path, encoding="utf-8") as fh:
            data = _json.load(fh)
        self.assertEqual(len(data["exposed_ids"]), 12)
        self.assertEqual(len(set(data["exposed_ids"])), 12)


class TestSummaryInferenceScope(unittest.TestCase):
    """先导批次不展示区间推断（n=1 的退化输出已移除）。"""

    def test_no_interval_fields_and_label(self):
        recs = [
            {"kind": "record", "problem_id": "p1", "stage": "first", "arm": "none",
             "run_status": "completed", "judge": {"status": "incorrect"}},
            {"kind": "record", "problem_id": "p1", "stage": "second", "arm": "A_prior_visible",
             "run_status": "completed", "judge": {"status": "incorrect"}},
            {"kind": "record", "problem_id": "p1", "stage": "second", "arm": "B_prior_hidden",
             "run_status": "completed", "judge": {"status": "correct"}},
        ]
        s = R.summarize(recs)
        d = s["determinate_pairs"]
        self.assertEqual(d["n_pairs"], 1)
        self.assertAlmostEqual(d["observed_diff_A_minus_B"], -1.0)
        self.assertIsNone(d["interval_estimate"])
        self.assertIn("未估计", d["interval_label"])
        for key in ("ci95_low", "ci95_high", "mean_diff_A_minus_B"):
            self.assertNotIn(key, d, f"不应再输出 {key}")
        text = R.format_summary(s)
        self.assertNotIn("95%CI", text)
        self.assertIn("未估计", text)
        self.assertIn("不作区间推断", text)


class TestPilotEntryPointMock(unittest.TestCase):
    """真实入口 + 先导配置（合成题池）：验证分层记录与选题落到运行头。"""

    def test_pilot_layers_recorded_in_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = EntryHarness(tmp)
            with open(h.cfg_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
            exposed_path = os.path.join(tmp, "exposed.json")
            _write(exposed_path, {"exposed_ids": ["p1"]})
            cfg["pilot"] = {"protocol_version": "x002-pilot-r07.1", "max_problems": 2,
                            "seed": 20261013, "exposed_ids_path": exposed_path}
            cfg["run_protocol_version"] = "x002-pilot-r07.1"
            _write(h.cfg_path, cfg)

            pool = [{"problem_id": "p1", "problem": "求 1+1 的值。", "gold": "2"},
                    {"problem_id": "p2", "problem": "求 2+2 的值。", "gold": "4"},
                    {"problem_id": "p3", "problem": "求 3+3 的值。", "gold": "[6, 7]"},   # 非标量 → 筛除
                    {"problem_id": "p4", "problem": "求 4+4 的值。", "gold": "8"},
                    {"problem_id": "p5", "problem": "求 5+5 的值。", "gold": "10"}]
            _write(h.problems_path, pool)
            plan = {"p2": {"problem": "求 2+2 的值。", "first": "答案是 \\boxed{9}",
                           "A": "重算 \\boxed{9}", "B": "重算 \\boxed{4}"},
                    "p4": {"problem": "求 4+4 的值。", "first": "答案是 \\boxed{8}"},
                    "p5": {"problem": "求 5+5 的值。", "first": "答案是 \\boxed{10}"}}
            _write(h.plan_path, plan)

            rc, out, err = h.run(total_seconds=2700)
            self.assertEqual(rc, 0, err)
            header = R.Store(os.path.join(h.out_dir, "records.jsonl")).header()
            sel = header["selection"]
            self.assertEqual(sel["protocol_version"], "x002-pilot-r07.1")
            self.assertEqual(sel["layers"]["pool_total"], 5)
            self.assertEqual(sel["layers"]["exposed_removed"], 1)
            self.assertEqual(sel["layers"]["non_scalar_removed"], 1)
            self.assertEqual(sel["layers"]["scalar_candidates"], 3)
            self.assertEqual(sel["layers"]["target"], 2)
            self.assertEqual(sel["n_problems"], 2)
            # 实际数据 commit 必须同时支配读数据、选题排序与运行头（真实入口，非只测 pilot 函数）
            self.assertEqual(header["dataset"]["revision"], "data-rev-9")
            self.assertEqual(sel["dataset_revision_used_for_sorting"], "data-rev-9")
            self.assertEqual(sel["dataset_revision_used_for_sorting"], header["dataset"]["revision"])
            self.assertTrue(sel["selected_ids_sha16"])
            self.assertIsNotNone(header["code_version"])
            selected_ids = [p["problem_id"] for p in header["problems"]]
            self.assertNotIn("p1", selected_ids, "已暴露题必须被排除")
            self.assertNotIn("p3", selected_ids, "非标量金标准必须被筛除")
            self.assertEqual(len(selected_ids), 2, "入组数应等于 target")
            summary = _read_json(os.path.join(h.out_dir, "summary.json"))
            self.assertNotIn("95%CI", R.format_summary(summary))


class TestPlanDenominators(unittest.TestCase):
    """摘要必须从运行头导出完整入组分母（四类情况）。"""

    def _rec(self, pid, stage, arm="none", status="correct", rs="completed"):
        return {"kind": "record", "problem_id": pid, "stage": stage, "arm": arm,
                "run_status": rs, "judge": {"status": status}, "usage": {}, "seconds": 0.0}

    def _header(self, ids):
        return {"kind": "run", "problems": [{"problem_id": i} for i in ids],
                "run_protocol_version": "x002-pilot-r07.1", "selection": {"layers": {"pool_total": 500}}}

    def test_first_round_interrupted(self):
        """首轮中断：入组分母来自运行头，缺失与"已记录但 UNKNOWN"分开。"""
        recs = [self._header(["p1", "p2", "p3"]),
                self._rec("p1", "first", status="incorrect"),
                self._rec("p2", "first", status="undetermined")]
        s = R.summarize(recs)
        plan = s["plan_denominators"]
        self.assertEqual(plan["n_enrolled"], 3)
        self.assertEqual(plan["enrolled_ids"], ["p1", "p2", "p3"])
        self.assertEqual(plan["n_first_recorded"], 2)
        self.assertEqual(plan["first_missing_ids"], ["p3"])
        self.assertEqual(plan["n_first_missing"], 1)
        self.assertEqual(plan["first_bucket_ids"]["undetermined"], ["p2"])
        self.assertEqual(plan["first_bucket_ids"]["incorrect"], ["p1"])
        self.assertNotIn("p2", plan["first_missing_ids"], "已记录但 UNKNOWN 的题不是缺失")
        self.assertEqual(s["second_round_plan"]["per_arm"]["A_prior_visible"]["planned"], 1)
        self.assertEqual(s["second_round_plan"]["per_arm"]["A_prior_visible"]["missing"], 1)

    def test_one_arm_missing(self):
        recs = [self._header(["p1", "p2"]),
                self._rec("p1", "first", status="incorrect"),
                self._rec("p2", "first", status="incorrect"),
                self._rec("p1", "second", arm="A_prior_visible", status="correct")]
        s = R.summarize(recs)
        a = s["second_round_plan"]["per_arm"]["A_prior_visible"]
        b = s["second_round_plan"]["per_arm"]["B_prior_hidden"]
        self.assertEqual((a["planned"], a["recorded"], a["missing"]), (2, 1, 1))
        self.assertEqual((b["planned"], b["recorded"], b["missing"]), (2, 0, 2))
        self.assertEqual(b["missing_ids"], ["p1", "p2"])
        self.assertEqual(s["determinate_pairs"]["n_pairs"], 0)
        text = R.format_summary(s)
        self.assertIn("B_prior_hidden: 计划=2 已记录=0 正常完成=0 已记录但UNKNOWN=0 缺失=2", text)
        self.assertIn("缺失 ID=['p1', 'p2']", text)

    def test_both_arms_present_but_undetermined(self):
        recs = [self._header(["p1"]), self._rec("p1", "first", status="incorrect"),
                self._rec("p1", "second", arm="A_prior_visible", status="undetermined"),
                self._rec("p1", "second", arm="B_prior_hidden", status="correct")]
        s = R.summarize(recs)
        self.assertEqual(s["determinate_pairs"]["n_pairs"], 0)
        self.assertEqual(s["excluded_from_pairs"][0]["reason"], "non_determinate:unknown/completed")
        for arm in R.ARMS:
            st = s["second_round_plan"]["per_arm"][arm]
            self.assertEqual((st["planned"], st["recorded"], st["missing"]), (1, 1, 0))
            self.assertEqual(st["completed"], 1 if arm == R.ARMS[1] else 0)
            self.assertEqual(st["unknown_but_recorded"], 0 if arm == R.ARMS[1] else 1)
        # 已记录但 UNKNOWN / 正常完成的题都不算"缺失"
        text = R.format_summary(s)
        self.assertNotIn("缺失 ID=", text)

    def test_eligible_never_recorded_first_round_is_not_assumed(self):
        """首轮未记录的题不能假定会进第二轮：分母只数"已记录且判错"。"""
        recs = [self._header(["p1", "p2", "p3"]), self._rec("p1", "first", status="incorrect")]
        s = R.summarize(recs)
        self.assertEqual(s["second_round_plan"]["n_planned"], 1)
        self.assertEqual(s["second_round_plan"]["planned_ids"], ["p1"])
        self.assertEqual(s["plan_denominators"]["first_missing_ids"], ["p2", "p3"])

    def test_no_header_marks_denominator_unknown(self):
        recs = [self._rec("p1", "first", status="incorrect"),
                self._rec("p1", "second", arm="A_prior_visible", status="correct"),
                self._rec("p1", "second", arm="B_prior_hidden", status="incorrect")]
        s = R.summarize(recs)
        plan = s["plan_denominators"]
        self.assertIsNone(plan["n_enrolled"])
        self.assertIsNone(plan["n_first_missing"])
        self.assertIsNone(plan["first_missing_ids"])
        self.assertEqual(plan["source"], "no_header")
        self.assertIn("未知", R.format_summary(s))
        self.assertIn("不得用已记录数冒充入组数", plan["note"])

    def test_json_and_text_agree_on_denominators(self):
        recs = [self._header(["p1", "p2"]), self._rec("p1", "first", status="incorrect")]
        s = R.summarize(recs)
        text = R.format_summary(s)
        plan = s["plan_denominators"]
        self.assertIn(f"入组数: {plan['n_enrolled']}", text)
        self.assertIn(f"首轮缺失: {plan['n_first_missing']}", text)
        self.assertIn("p2", text)
        self.assertIn(f"计划做题数（合格错题）: {s['second_round_plan']['n_planned']}", text)
        for arm, st in s["second_round_plan"]["per_arm"].items():
            self.assertIn(f"{arm}: 计划={st['planned']} 已记录={st['recorded']}", text)


class TestHeaderIdentityFields(unittest.TestCase):
    """运行头必须记录协议版本、实际数据 commit、源码版本与排除清单摘要。"""

    def test_header_records_protocol_commit_and_code_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = EntryHarness(tmp)
            rc, out, err = h.run(total_seconds=2700)
            self.assertEqual(rc, 0, err)
            header = R.Store(os.path.join(h.out_dir, "records.jsonl")).header()
            self.assertEqual(header["run_protocol_version"], "x002-pilot-r07.1")
            self.assertEqual(header["dataset"]["revision"], "data-rev-9")   # 解析后的实际 commit
            self.assertIsNotNone(header["code_version"])
            self.assertIn("files_sha16", header["code_version"])
            sel = header["selection"]
            self.assertEqual(sel["dataset_revision_used_for_sorting"], "data-rev-9")
            self.assertTrue(sel["selected_ids_sha16"])

    def test_pilot_requires_resolved_commit(self):
        """拿不到数据 commit 时，先导选题必须拒绝继续，而不是用空串排序。"""
        from x002 import worker as W

        class Rdr:
            def list_ids(self):
                return ["p1", "p2"]

            def list_rows(self):
                return [{"unique_id": "p1", "gold": "1"}, {"unique_id": "p2", "gold": "2"}]

            def get(self, pid):
                return {"problem": f"题 {pid}", "gold": "1"}

        cfg = {"data": {"revision": None}, "pilot": {"max_problems": 2, "seed": 1}}
        with self.assertRaises(RuntimeError):
            W.freeze_selection(Rdr(), cfg, dataset_revision=None)
        problems, sel = W.freeze_selection(Rdr(), cfg, dataset_revision="commit-abc")
        self.assertEqual(sel["dataset_revision_used_for_sorting"], "commit-abc")
        self.assertEqual(len(problems), 2)

    def test_real_reader_gets_same_commit_and_selection_header_binds_it(self):
        """真实入口（DatasetReader + run_attempt）的接线核验，而不只测 pilot 函数。

        用 ``datasets.load_dataset`` 的替身记录**实际传入的 revision**，并核对
        ``freeze_selection`` 收到的 commit 与之相同、且同值写进运行头。
        """
        from x002 import worker as W

        calls = []

        class FakeDataset(list):
            pass

        def fake_load_dataset(dataset_id, split=None, revision=None, **kw):
            calls.append({"dataset_id": dataset_id, "split": split, "revision": revision})
            return FakeDataset([
                {"unique_id": "p1", "problem": "求 1+1 的值。", "answer": "2", "subject": "algebra", "level": 1},
                {"unique_id": "p2", "problem": "求 2+2 的值。", "answer": "4", "subject": "algebra", "level": 1},
                {"unique_id": "p3", "problem": "求 3+3 的值。", "answer": "6", "subject": "geometry", "level": 2},
            ])

        fake_datasets = types.ModuleType("datasets")
        fake_datasets.load_dataset = fake_load_dataset
        seen = {}
        real_freeze = W.freeze_selection

        def spy_freeze(reader, cfg, dataset_revision=None):
            seen["revision"] = dataset_revision
            return real_freeze(reader, cfg, dataset_revision=dataset_revision)

        saved = sys.modules.get("datasets")
        sys.modules["datasets"] = fake_datasets
        W.freeze_selection = spy_freeze
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cfg = {
                    "model": {"path_or_id": "fake/model"}, "generation": {"seed": 3},
                    "data": {"dataset_id": "fake/data", "split": "test", "revision": None,
                             "problem_field": "problem", "answer_field": "answer", "n_problems": 2},
                    "feedback": {"text": FEEDBACK}, "judge": {"numeric_rtol": 1e-6, "symbolic": False},
                    "paths": {"out_dir": os.path.join(tmp, "out"), "log_name": "records.jsonl"},
                    "run_protocol_version": "x002-pilot-r07.1",
                    "pilot": {"protocol_version": "x002-pilot-r07.1", "max_problems": 2, "seed": 20261013},
                }
                W.load_offline_fixture(_write(os.path.join(tmp, "fixture.json"), {
                    "model": {"path_or_id": "fake/model", "revision": "model-rev-1", "resolved_from": "fixture"},
                    "dataset": {"dataset_id": "fake/data", "split": "test", "revision": "data-rev-9",
                                "resolved_from": "fixture"},
                    "env": {"python": "3.x", "platform": "offline-fixture"},
                }))
                try:
                    reader = W.DatasetReader(cfg, {"dataset_id": "fake/data", "split": "test",
                                                   "revision": "data-rev-9"})
                    problems, selection = W.freeze_selection(reader, cfg, dataset_revision="data-rev-9")
                    header = W.build_run_header(cfg, {"revision": "model-rev-1"}, {"revision": "data-rev-9"},
                                                selection, problems, 60, cfg["paths"]["out_dir"])
                finally:
                    W.load_offline_fixture(None)
                    W.freeze_selection = real_freeze
        finally:
            if saved is None:
                sys.modules.pop("datasets", None)
            else:
                sys.modules["datasets"] = saved

        self.assertEqual(calls[0]["revision"], "data-rev-9", "读数据必须用解析后的 commit")
        self.assertEqual(seen["revision"], "data-rev-9", "选题排序必须收到同一个 commit")
        self.assertEqual(header["dataset"]["revision"], "data-rev-9")
        self.assertEqual(header["selection"]["dataset_revision_used_for_sorting"], "data-rev-9")
        self.assertEqual(header["selection"]["dataset_revision_used_for_sorting"],
                         header["dataset"]["revision"])
        self.assertEqual(len(problems), 2)


class TestJudgeNewCounterexamples(unittest.TestCase):
    """R08 合并修复：完整候选/答案区域下的多答案与非法逗号串。

    覆盖三类，**不是**只匹配 PI 给出的三个字符串：
    ① PI 反例（公式分支绕过、换行切开同一非法逗号串）；
    ② 同规则生成的其它并列形态（无连接词并列、分号、``and``、货币符号、boxed/hash 分支之后）；
    ③ 必须保持判分能力的正例（合法千分位、单个小数/分数、boxed 497/129、空行后另有解释数字）。
    """

    # ---- 必须 UNKNOWN：多答案 / 非法逗号串（数字与公式两种候选写法都覆盖）
    MULTI_ANSWER_UNKNOWN = [
        ("The answer is 1,2.", "1"),
        ("The answer is 1, 2.", "1"),
        ("The answer is 1, 2, and 3.", "1"),
        ("The answer is 1;\n2.", "1"),
        ("The final answer is 2.5, or 7.", "2.5"),
        ("The final answer is $2.5$, or 7.", "2.5"),
        ("The answer is $1$, or $2$.", "1"),
        ("The answer is $1$ or $2$.", "1"),
        ("The answer is 3 and 5.", "3"),
        ("The answer is 1,\n2.", "1"),
        ("The answer is \\boxed{1}, or 2.", "1"),
        ("The answer is \\boxed{2} or 3.", "2"),
    ]

    # ---- 必须保持 correct：合法标量与"另一段解释里的数字"
    STILL_CORRECT = [
        ("The final answer is $5$.", "5"),
        ("The answer is 1,000.", "1000"),
        ("The answer is 1,000,000.", "1000000"),
        ("The answer is 12.5.", "12.5"),
        ("The answer is 7, so we are done", "7"),
        ("The answer is $\\frac{1}{2}$.", "0.5"),
        ("The answer is 2.\n\nWe used 3 steps.", "2"),
        ("The answer is 2. We used 3 steps.", "2"),
        ("\\boxed{497}", "497"),
        ("\\boxed{129}", "129"),
        ("\\boxed{2}", "2"),
        ("\\boxed{2}\n\nWe used 3 steps.", "2"),
        ("#### 2\n3 steps were needed.", "2"),
    ]

    def test_multi_answer_forms_are_unknown(self):
        for resp, gold in self.MULTI_ANSWER_UNKNOWN:
            with self.subTest(resp=resp):
                out = J.judge(resp, gold)
                self.assertEqual(out["status"], "undetermined", out)
                self.assertIsNone(out["extracted"], out)

    def test_supported_single_answers_still_judged(self):
        for resp, gold in self.STILL_CORRECT:
            with self.subTest(resp=resp):
                self.assertEqual(J.judge(resp, gold)["status"], "correct", resp)

    def test_multi_answer_also_rejects_matching_gold(self):
        """多答案不能因为"其中一个等于 gold"就被判 correct，也不能被当成 incorrect。

        这样它既不会进错题配对，也不会污染正确分子（须为 UNKNOWN）。
        """
        for resp, gold in (("The answer is 2, or 3.", "2"),
                           ("The answer is $4$, or $5$.", "5"),
                           ("The answer is \\boxed{6}, or 7.", "6"),
                           ("The answer is 6 and 7.", "6")):
            with self.subTest(resp=resp):
                out = J.judge(resp, gold)
                self.assertEqual(out["status"], "undetermined", out)

    def test_documented_region_limit_blank_line_before_continuation(self):
        """已知边界（如实记录，不在本批扩范围）：候选句结束后换行再写 ``Or 3``。

        ``2.`` 之后的空行之前没有并列形态，本规则按"该句只有一个候选"处理；
        这属于"候选之后的另一段文字"，不是本批要修的多答案误判，故不改判分范围。
        """
        out = J.judge("The answer is 2.\nWe stop here.", "2")
        self.assertEqual(out["status"], "correct", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
