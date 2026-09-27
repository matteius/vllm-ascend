#!/usr/bin/env bash
set -eo pipefail
evidence=/srv/ai/src/qwen38-w4-hardware-20260927
python_bin=/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python
test ! -e "$evidence/server-capacity-r6.log"
bash "$evidence/qwen38-w4-capacity-launch-r6.sh" > "$evidence/server-capacity-r6.log" 2>&1 &
server_pid=$!
echo "W4_SERVER_PID=$server_pid"
"$python_bin" "$evidence/qwen38-w4-watchdog.py" "$server_pid" > "$evidence/thermal-capacity-r6.log" 2>&1 &
monitor_pid=$!
trap 'kill -TERM "$monitor_pid" 2>/dev/null || true' EXIT
"$python_bin" "$evidence/qwen38-w4-http-smoke.py" "$server_pid" > "$evidence/http-smoke-capacity-r6.jsonl" 2>&1
echo QWEN_W4_CAPACITY_R6_SMOKE_COMPLETE
wait "$server_pid"
