#!/usr/bin/env bash
# X-002 离线检查包装：显式传播非零退出，避免管道掩盖失败
# 用法：bash run_offline_checks.sh [输出日志路径]
set -Eeuo pipefail

LOG="${1:-offline_checks.log}"
set +e
PYTHONDONTWRITEBYTECODE=1 python3 tests/test_offline_checks.py > "${LOG}.raw" 2>&1
rc=$?
set -e
cat "${LOG}.raw" | tee "$LOG" >/dev/null
cat "$LOG" | tail -5

if [ "$rc" -ne 0 ]; then
  echo "[FAIL] 离线检查未通过（退出码 ${rc}）；日志 ${LOG}" >&2
  echo "       停止后续步骤，回传 ${LOG} 与 git rev-parse HEAD。" >&2
  exit "$rc"
fi
if ! grep -q "^OK$" "$LOG"; then
  echo "[FAIL] 日志未以 OK 结尾，视为未通过；日志 ${LOG}" >&2
  exit 1
fi
echo "[OK] 离线检查通过；日志 ${LOG}"
