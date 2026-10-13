#!/usr/bin/env python3
"""X-002 一次性工程试运行入口（父进程 = 监督者）。

推荐命令（唯一）::

    python3 -m x002.cli run --config config.json
    python3 -m x002.cli summarize --out-dir runs/x002_smoke

一次性语义
----------
* 输出目录在**加载/生成之前**写入启动标记 ``started.json``；
  已存在标记的目录**拒绝再次生成**（返回码 6），只允许 ``summarize``。
  上次失败或超时也不重跑，不自动新建目录绕过预算——原始日志留给 PI 判断。
* **从开始运行到模型加载、编码与生成**都受同一个外层截止监督：
  工作进程在独立进程组里运行，到点先 ``SIGTERM``、再升级 ``SIGKILL`` 并回收，
  不留下继续计算的后台线程/进程。
* 超时/退出原因由父进程写进 ``status.json``；已有 ``records.jsonl`` 仍可汇总。

没有续跑、没有覆盖、没有预算账本——这些功能已按 PI 裁决删除。
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

from . import run as run_mod
from .run import Store, format_summary, summarize

START_MARKER = "started.json"
STATUS_FILE = "status.json"
GRACE_SECONDS = 1.0
SIGKILL_GRACE_SECONDS = 1.0


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def cmd_run(args) -> int:
    out_dir = os.path.abspath(args.out_dir)
    marker = os.path.join(out_dir, START_MARKER)
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(marker):
        with open(marker, encoding="utf-8") as fh:
            started = json.load(fh)
        print(
            f"[reject] 该输出目录已于 {started.get('started_at')} 启动过一次性试运行，拒绝再次生成。\n"
            f"         原始日志仍在：{out_dir}\n"
            f"         只允许汇总：python3 -m x002.cli summarize --out-dir {out_dir}",
            file=sys.stderr,
        )
        return 6

    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)

    total_minutes = float(cfg["run"]["smoke"]["total_minutes"])
    if args.total_seconds is not None:
        budget_seconds = float(args.total_seconds)
    elif args.total_minutes is not None:
        budget_seconds = float(args.total_minutes) * 60
    else:
        budget_seconds = total_minutes * 60

    # 启动标记在加载/生成之前落盘
    with open(marker, "w", encoding="utf-8") as fh:
        json.dump({"started_at": _now(), "config": os.path.abspath(args.config),
                   "budget_seconds": budget_seconds, "pid": os.getpid(), "argv": sys.argv[1:]},
                  fh, ensure_ascii=False, indent=2)

    worker_args = [
        "--config", os.path.abspath(args.config),
        "--out-dir", out_dir,
        "--budget-seconds", str(budget_seconds),
    ]
    if args.backend_factory:
        worker_args += ["--backend-factory", args.backend_factory]
    if args.problems_json:
        worker_args += ["--problems-json", os.path.abspath(args.problems_json)]
    if args.offline_fixture:      # 仅离线检查路径提供；正式运行不传
        worker_args += ["--offline-fixture", os.path.abspath(args.offline_fixture)]

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.pathsep + env.get("PYTHONPATH", "")

    # 总上限是硬边界：终止宽限预留在上限之内，不再"等满预算再加宽限"
    t0 = time.monotonic()
    deadline = t0 + budget_seconds
    graceful_deadline = deadline - GRACE_SECONDS          # SIGTERM 的最晚发出时刻
    hard_deadline = graceful_deadline + GRACE_SECONDS     # 与 deadline 一致
    hard_cap = deadline + SIGKILL_GRACE_SECONDS           # 强杀后回收的最后期限

    # 直接运行工作进程入口脚本：本模块被当库调用时（离线检查）也用同一入口，
    # 且工作进程里 import 测试替身时能找到仓库根目录。
    if args.worker_cmd == "fail3":     # 测试接缝：模拟工作进程非零退出
        launch = [sys.executable, "-c", "import sys; sys.exit(3)"]
    elif args.worker_cmd == "fail0":
        launch = [sys.executable, "-c", "import sys; sys.exit(0)"]
    else:
        runner = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_worker.py")
        launch = [sys.executable, runner] + worker_args
    proc = subprocess.Popen(launch,
                            env=env, start_new_session=True,
                            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        worker_pgid = os.getpgid(proc.pid)   # 启动时立即保存进程组号
    except ProcessLookupError:  # pragma: no cover - 极早退出
        worker_pgid = proc.pid

    reason = "worker_exited"
    timed_out = False
    rc = None
    sigterm_at = None
    sigkill_at = None
    started_cleanup = False
    group_alive = _group_alive(worker_pgid)
    while True:
        now = time.monotonic()
        rc = proc.poll()
        if not group_alive:
            break
        if not timed_out and now >= graceful_deadline:
            timed_out = True
            reason = "total_deadline_exceeded"
            _killpg(worker_pgid, signal.SIGTERM)      # 先温和终止整组
            sigterm_at = now
            started_cleanup = True
        if started_cleanup and now >= hard_deadline:
            if sigkill_at is None:
                _killpg(worker_pgid, signal.SIGKILL)  # 宽限结束仍存活 → 强杀
                sigkill_at = now
            if now >= hard_cap:
                break
        group_alive = _group_alive(worker_pgid)
        if not group_alive:
            break
        time.sleep(0.05)

    killed_members = sorted(set(_group_members(worker_pgid))) if sigkill_at is not None else []
    group_alive_after = _group_alive(worker_pgid)
    if group_alive_after:                             # 最后一道确认
        _killpg(worker_pgid, signal.SIGKILL)
        sigkill_at = sigkill_at or time.monotonic()
        _wait_group_exit(worker_pgid, time.monotonic() + SIGKILL_GRACE_SECONDS)
        group_alive_after = _group_alive(worker_pgid)
    try:
        proc.wait(timeout=0.5)
    except subprocess.TimeoutExpired:                 # pragma: no cover
        pass
    cleanup = {"group_alive_before": group_alive, "killed_members": killed_members,
               "group_alive_after": group_alive_after}
    if cleanup["group_alive_after"]:   # 最后一道：只针对仍存活的成员强杀并确认
        _killpg(worker_pgid, signal.SIGKILL)
        _wait_group_exit(worker_pgid, time.monotonic() + SIGKILL_GRACE_SECONDS)
        cleanup["group_alive_after"] = _group_alive(worker_pgid)
    # 非零退出也要如实上报（宽限内主动退出仍区分 worker_exited）
    if not timed_out and rc not in (0, None):
        reason = "worker_failed"
    elapsed = time.monotonic() - t0

    status = {
        "kind": "status",
        "worker_pid": proc.pid,
        "worker_pgid": worker_pgid,              # 启动时保存的原始 pgid（不因组长退出而丢失）
        "finished_at": _now(),
        "reason": reason,
        "timed_out": timed_out,
        "worker_returncode": rc,
        "budget_seconds": budget_seconds,
        "elapsed_seconds": round(elapsed, 2),
        "grace_seconds": GRACE_SECONDS,
        "sigkill_grace_seconds": SIGKILL_GRACE_SECONDS,
        "term_sent_at_seconds": round(sigterm_at - t0, 2) if sigterm_at else None,
        "sigkill_sent_at_seconds": round(sigkill_at - t0, 2) if sigkill_at else None,
        "forced_kill": sigkill_at is not None,
        "group_members_alive_at_cleanup": cleanup["group_alive_before"],
        "group_members_killed": cleanup["killed_members"],
        "group_alive_after_cleanup": cleanup["group_alive_after"],
        "cmd": launch,
    }
    with open(os.path.join(out_dir, STATUS_FILE), "w", encoding="utf-8") as fh:
        json.dump(status, fh, ensure_ascii=False, indent=2)

    _summarize_dir(out_dir, cfg)
    print(f"[{reason}] 用时 {elapsed:.1f}s（上限 {budget_seconds:.0f}s，宽限 {GRACE_SECONDS:.0f}s 含在上限内）")
    if timed_out:
        return 7
    if rc not in (0, None):
        return rc if isinstance(rc, int) and 0 < rc < 126 else 1
    return 0


def _wait_group_exit(pgid, until: float) -> bool:
    """等到 until 为止；返回组是否已全部退出。"""
    while time.monotonic() < until:
        if not _group_alive(pgid):
            return True
        time.sleep(0.05)
    return not _group_alive(pgid)


def _group_alive(pgid: int | None) -> bool:
    """整组存活检查：用保存的 pgid 发信号 0；不因组长退出就认为全组结束。"""
    if not pgid:
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - 理论上不会发生在自己的子进程组
        return True


def _group_members(pgid: int | None) -> list[int]:
    """尽可能列出组内成员 pid（空列表表示无法确认或已无成员）。"""
    if not pgid:
        return []
    try:
        out = subprocess.run(["ps", "-o", "pid=", "-g", str(pgid)],
                             capture_output=True, text=True, timeout=2.0)
    except Exception:  # pragma: no cover
        return []
    pids = []
    for line in out.stdout.split("\n"):
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def _force_kill(proc, pgid, term_sent_at, sigkill_sent_at):
    """向整组发 SIGTERM；若已发过 SIGTERM 仍存活则升级 SIGKILL。"""
    now = time.monotonic()
    if term_sent_at is None:
        _killpg(pgid, signal.SIGTERM)
        term_sent_at = now
    else:
        _killpg(pgid, signal.SIGKILL)
        sigkill_sent_at = now
    try:
        proc.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        pass
    return term_sent_at, sigkill_sent_at


def _killpg(pgid, sig) -> None:
    if not pgid:
        return
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def cmd_summarize(args) -> int:
    out_dir = os.path.abspath(args.out_dir)
    store = Store(os.path.join(out_dir, "records.jsonl"))
    summary = summarize(store.records())
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    text = format_summary(summary)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    status_path = os.path.join(out_dir, STATUS_FILE)
    if os.path.exists(status_path):
        with open(status_path, encoding="utf-8") as fh:
            status = json.load(fh)
        print(f"[status] {status.get('reason')} 超时={status.get('timed_out')} "
              f"用时={status.get('elapsed_seconds')}s 上限={status.get('budget_seconds')}s")
    print(text)
    return 0


class _NS:
    """极简命名空间，供脚本化调用 summarize。"""

    def __init__(self, out_dir):
        self.out_dir = out_dir


def capture_summarize(out_dir: str) -> tuple[int, str, str]:
    """脚本化调用只读汇总，返回 (rc, stdout, stderr)。"""
    import io
    from contextlib import redirect_stderr, redirect_stdout

    buf_out, buf_err = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        rc = cmd_summarize(_NS(out_dir))
    return rc, buf_out.getvalue(), buf_err.getvalue()


def _summarize_dir(out_dir: str, cfg: dict | None) -> None:
    store = Store(os.path.join(out_dir, "records.jsonl"))
    summary = summarize(store.records())
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(format_summary(summary) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="X-002 一次性工程试运行")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="一次性运行（已启动过的目录会被拒绝）")
    r.add_argument("--config", default="config.json")
    r.add_argument("--out-dir", default=None, help="默认取配置里的 paths.out_dir")
    r.add_argument("--total-minutes", type=float, default=None, help="覆盖总截止（分钟）")
    r.add_argument("--total-seconds", type=float, default=None, help="覆盖总截止（秒；用于短预算检查）")
    r.add_argument("--backend-factory", default=None, help="测试接缝：'module:callable'")
    r.add_argument("--problems-json", default=None, help="测试接缝：预置题目 JSON")
    r.add_argument("--worker-cmd", default=None, choices=["fail3", "fail0"],
                   help="测试接缝：用固定退出码替代真实工作进程")
    r.add_argument("--offline-fixture", default=None,
                   help="测试接缝：离线 fixture（身份/环境/启动延迟）；正式运行不提供")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("summarize", help="只读汇总（可对已有日志运行）")
    s.add_argument("--out-dir", required=True)
    s.set_defaults(func=cmd_summarize)

    args = ap.parse_args(argv)
    if args.cmd == "run" and args.out_dir is None:
        with open(args.config, encoding="utf-8") as fh:
            args.out_dir = json.load(fh)["paths"]["out_dir"]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
