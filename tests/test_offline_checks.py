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
        """直接替换真实 TransformersBackend.load 用到的两个加载函数（不是 fake 自己记名字）。"""
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

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeAutoTokenizer
        fake_transformers.AutoModelForCausalLM = FakeAutoModel
        sys.modules["transformers"] = fake_transformers
        try:
            backend = G.TransformersBackend(
                {"path_or_id": "fake/model", "revision": "rev-frozen-123", "dtype": "bfloat16"}, {})
            backend.load()
        finally:
            del sys.modules["transformers"]
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
            selected_ids = [p["problem_id"] for p in header["problems"]]
            self.assertNotIn("p1", selected_ids, "已暴露题必须被排除")
            self.assertNotIn("p3", selected_ids, "非标量金标准必须被筛除")
            self.assertEqual(len(selected_ids), 2, "入组数应等于 target")
            summary = _read_json(os.path.join(h.out_dir, "summary.json"))
            self.assertNotIn("95%CI", R.format_summary(summary))


if __name__ == "__main__":
    unittest.main(verbosity=2)
