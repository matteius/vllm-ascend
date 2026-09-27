#!/usr/bin/env bash
# Full-size validation; leaves the experimental server running afterward.
set -euo pipefail
evidence=/srv/ai/src/qwen38-w4-hardware-20260927
python_bin=/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python
grep -q QWEN_W4_CAPACITY_R6_SEQUENCE_COMPLETE "$evidence/capacity-sequence-r6.log"
capacity=$(sed -nE 's/.*KV cache size: ([0-9,]+) tokens.*/\1/p' "$evidence/server-capacity-r6.log")
capacity=${capacity//,/}
[[ "$capacity" =~ ^[0-9]+$ ]]
test "$capacity" -ge 524288
metrics=$(curl -fsS --max-time 10 http://127.0.0.1:8002/metrics)
grep -qE '^vllm:num_requests_running\{.*\} 0(\.0)?$' <<< "$metrics"
grep -qE '^vllm:num_requests_waiting\{.*\} 0(\.0)?$' <<< "$metrics"
echo "DUAL262_START_UTC=$(date -u +%FT%TZ)"
npu-smi info
minimum_available_kib=$((12 * 1024 * 1024))
maximum_swap_growth_kib=$((2 * 1024 * 1024))
initial_swap_free_kib=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
"$python_bin" "$evidence/benchmark_capacity.py" \
  --model-dir /srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --prompt-tokens 261632 --output-tokens 512 --concurrency 2 --warm-prefixes \
  --output "$evidence/capacity-r6-dual262.jsonl" &
probe_pid=$!
# Long prefixes also retain host-side Mamba checkpoints. Abort only this
# client if host pressure grows; disconnect cancels its HTTP requests and
# leaves the server available for inspection. Do not risk a host OOM.
trap 'kill -TERM "$probe_pid" 2>/dev/null || true' EXIT
while kill -0 "$probe_pid" 2>/dev/null; do
  available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  swap_free_kib=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
  swap_growth_kib=$((initial_swap_free_kib - swap_free_kib))
  echo "HOST_MEMORY_UTC=$(date -u +%FT%TZ) available_kib=$available_kib swap_growth_kib=$swap_growth_kib"
  if ((available_kib < minimum_available_kib || swap_growth_kib > maximum_swap_growth_kib)); then
    echo DUAL262_HOST_MEMORY_GUARD
    kill -TERM "$probe_pid"
    wait "$probe_pid" || true
    trap - EXIT
    exit 1
  fi
  sleep 30
done
wait "$probe_pid"
trap - EXIT
jq -e -s 'map(select(.event=="summary")) | length==1 and all(.[]; .passed and .decode_overlap_s>0 and .observed_running_peak==2 and .metric_deltas.num_preemptions_total==0)' "$evidence/capacity-r6-dual262.jsonl"
jq -e -s 'map(select(.event=="result")) | length==2 and all(.[]; .done and .finish_reason=="length" and .usage.prompt_tokens==261632 and .usage.completion_tokens==512)' "$evidence/capacity-r6-dual262.jsonl"
curl -fsS --max-time 10 http://127.0.0.1:8002/health
npu-smi info
echo "DUAL262_COMPLETE_UTC=$(date -u +%FT%TZ)"
echo QWEN_W4_DUAL262_R6_PASS
