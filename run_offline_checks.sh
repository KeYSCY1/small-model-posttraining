#!/usr/bin/env bash
# X-002 离线检查包装：显式传播非零退出，避免管道/后续命令掩盖失败
# 用法：bash run_offline_checks.sh [输出日志路径，默认 offline_checks.log]
# 产物：$LOG（完整输出；同时回显到调用方终端）
set -uo pipefail

LOG="${1:-offline_checks.log}"

# 唯一一次实际检查：先落盘日志，再由调用方 tee/重定向
PYTHONDONTWRITEBYTECODE=1 python3 tests/test_offline_checks.py 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

if [ "$rc" -ne 0 ]; then
  echo "[FAIL] 离线检查未通过（退出码 ${rc}）；日志 ${LOG}" >&2
  echo "       停止后续步骤，回传 ${LOG} 与 git rev-parse HEAD。" >&2
  exit "$rc"
fi

# 结尾必须是 OK：非零退出或结尾异常都视为未通过
if ! tail -n 5 "$LOG" | grep -qx "OK"; then
  echo "[FAIL] 日志未以 OK 结尾，视为未通过；日志 ${LOG}" >&2
  exit 1
fi

echo "[OK] 离线检查通过；日志 ${LOG}"
