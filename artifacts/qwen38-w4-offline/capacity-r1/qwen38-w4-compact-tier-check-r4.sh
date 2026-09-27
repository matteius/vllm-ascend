#!/usr/bin/env bash
set -eo pipefail
evidence=/srv/ai/src/qwen38-w4-hardware-20260927
source "$evidence/qwen38-w4-hardware-env.sh"
ready=0
for ((attempt=0; attempt<600; attempt++)); do
  if grep -q QWEN_W4_CAPACITY_R4_SEQUENCE_COMPLETE "$evidence/capacity-sequence-r4.log"; then
    ready=1
    break
  fi
  if grep -qE 'Traceback|AssertionError|RuntimeError' "$evidence/capacity-sequence-r4.log"; then
    exit 1
  fi
  sleep 2
done
test "$ready" = 1
metrics=$(curl -fsS --max-time 10 http://127.0.0.1:8002/metrics)
grep -qE '^vllm:num_requests_running\{.*\} 0(\.0)?$' <<< "$metrics"
grep -qE '^vllm:num_requests_waiting\{.*\} 0(\.0)?$' <<< "$metrics"
exec /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python -m pytest -q --noconftest \
  "$evidence/test_prefix_mamba_state_310.py"
