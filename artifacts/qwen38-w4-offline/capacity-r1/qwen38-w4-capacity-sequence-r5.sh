#!/usr/bin/env bash
set -euo pipefail
evidence=/srv/ai/src/qwen38-w4-hardware-20260927
python_bin=/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python
model=/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i
api_pid=$(sed -n 's/^W4_SERVER_PID=//p' "$evidence/server-capacity-r5-wrapper.log")
[[ "$api_pid" =~ ^[0-9]+$ ]]
ready=0
for ((attempt=0; attempt<600; attempt++)); do
  if grep -q QWEN_W4_CAPACITY_R5_SMOKE_COMPLETE "$evidence/server-capacity-r5-wrapper.log"; then
    ready=1
    break
  fi
  kill -0 "$api_pid"
  if grep -qE 'Traceback|HTTP_SMOKE_FAIL' "$evidence/http-smoke-capacity-r5.jsonl" 2>/dev/null; then
    exit 1
  fi
  sleep 2
done
test "$ready" = 1
mapfile -t engines < <(ps --ppid "$api_pid" -o pid=,args= | awk '$2=="VLLM::EngineCore" {print $1}')
test "${#engines[@]}" = 1
engine_pid=${engines[0]}
workers=()
for rank in 0 1 2 3; do
  mapfile -t found < <(ps --ppid "$engine_pid" -o pid=,args= | awk -v name="VLLM::Worker_TP${rank}_EP${rank}" '$2==name {print $1}')
  test "${#found[@]}" = 1
  workers+=("${found[0]}")
done
metrics=$(curl -fsS --max-time 10 http://127.0.0.1:8002/metrics)
grep -qE '^vllm:num_requests_running\{.*\} 0(\.0)?$' <<< "$metrics"
grep -qE '^vllm:num_requests_waiting\{.*\} 0(\.0)?$' <<< "$metrics"
"$python_bin" "$evidence/qwen38-w4-live-affinity-r2.py" \
  --api-pid "$api_pid" --engine-pid "$engine_pid" --workers "${workers[@]}" > "$evidence/capacity-r5-affinity.json"
echo CAPACITY_AFFINITY_APPLIED
"$python_bin" "$evidence/benchmark_capacity.py" --model-dir "$model" \
  --prompt-tokens 256 --output-tokens 512 --concurrency 1 \
  --output "$evidence/capacity-r5-short-single.jsonl"
echo CAPACITY_SINGLE_MTP_PASS
"$python_bin" "$evidence/benchmark_capacity.py" --model-dir "$model" \
  --prompt-tokens 256 --output-tokens 256 --concurrency 2 --warm-prefixes \
  --output "$evidence/capacity-r5-short-dual.jsonl"
jq -e -s 'map(select(.event=="summary")) | length==1 and all(.[]; .passed and .decode_overlap_s>0 and .observed_running_peak==2 and .metric_deltas.num_preemptions_total==0)' "$evidence/capacity-r5-short-dual.jsonl"
echo CAPACITY_SHORT_CONCURRENCY_PASS
# Revisit two independent cached prefixes. Request IDs/rows are recycled
# between probes; the separate NPU state-tier test forces resident eviction.
"$python_bin" "$evidence/benchmark_capacity.py" --model-dir "$model" \
  --prompt-tokens 8192 --output-tokens 48 --concurrency 2 --warm-prefixes --mode recall \
  --output "$evidence/capacity-r5-8k-dual-recall.jsonl"
echo CAPACITY_8K_RECALL_PASS
echo QWEN_W4_CAPACITY_R5_SEQUENCE_COMPLETE
