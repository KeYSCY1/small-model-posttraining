#!/usr/bin/env bash
# X-002 离线检查包装：显式传播非零退出，避免管道/后续命令掩盖失败
# 用法：bash run_offline_checks.sh [输出日志路径，默认 offline_checks.log]
# 产物：$LOG（完整输出；同时回显到调用方终端）
#
# 注意：离线检查自身会在包目录内做"忽略规则"用例（创建并清理 offline_checks.log* 等同名文件），
# 因此日志先写到包外的临时文件，检查结束后再复制到 $LOG，避免被用例的清理步骤删掉。
set -uo pipefail

LOG="${1:-offline_checks.log}"
case "$LOG" in
  /*) ABS_LOG="$LOG" ;;
  *)  ABS_LOG="$(pwd)/$LOG" ;;
esac
TMP_LOG="$(mktemp -t x002_offline_checks.XXXXXX)" || exit 1
trap 'rm -f "$TMP_LOG"' EXIT

PYTHONDONTWRITEBYTECODE=1 python3 tests/test_offline_checks.py 2>&1 | tee "$TMP_LOG"
rc=${PIPESTATUS[0]}

mkdir -p "$(dirname "$ABS_LOG")" 2>/dev/null || true
cp -f "$TMP_LOG" "$ABS_LOG" 2>/dev/null || true

if [ "$rc" -ne 0 ]; then
  echo "[FAIL] 离线检查未通过（退出码 ${rc}）；日志 ${LOG}" >&2
  echo "       停止后续步骤，回传 ${LOG} 与 git rev-parse HEAD。" >&2
  exit "$rc"
fi

# 结尾必须是 OK：非零退出或结尾异常都视为未通过
if ! tail -n 5 "$TMP_LOG" | grep -qE '^OK( \([^)]*\))?$'; then
  echo "[FAIL] 日志未以 OK 结尾，视为未通过；日志 ${LOG}" >&2
  exit 1
fi

echo "[OK] 离线检查通过；日志 ${LOG}"
