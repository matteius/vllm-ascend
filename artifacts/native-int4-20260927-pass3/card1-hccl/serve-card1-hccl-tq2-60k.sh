#!/usr/bin/env bash
set -eo pipefail
label=${1:?label}
task_queue=${2:?task queue mode}
aiv=${3:?0 or 1}
result_root=/srv/ai/src/native-int4-w4a8.KiuhBN/pass3/card1-hccl
source /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh
set -u
candidate_root=/srv/ai/src/native-int4-w4a8.KiuhBN
runtime_root=$candidate_root/runtime
export ASCEND_RT_VISIBLE_DEVICES=2,3
export TASK_QUEUE_ENABLE=$task_queue
if [[ "$aiv" == 1 ]]; then
  export HCCL_OP_EXPANSION_MODE=AIV
else
  unset HCCL_OP_EXPANSION_MODE || true
fi
export PYTHONPATH=$runtime_root:/srv/ai/src/vllm-opensensor:${PYTHONPATH:-}
export ASCEND_CUSTOM_OPP_PATH=$candidate_root/ops-pass3-route-cache/vendors/native_int4_w4a8_transformer:/srv/ai/src/qwen38-w4-native-build.zHZtEd/ops-routeplan-r4/vendors/qwen_w4_native_transformer:${ASCEND_CUSTOM_OPP_PATH:-}
export VLLM_ASCEND_KV_CACHE_FRACTION=0.80
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_ASCEND_LOG_REQUEST_TIMINGS=1
mkdir -p "$result_root"
env | grep -E '^(ASCEND_RT_VISIBLE_DEVICES|TASK_QUEUE_ENABLE|HCCL_OP_EXPANSION_MODE|VLLM_ASCEND_KV_CACHE_FRACTION)=' | sort > "$result_root/$label.env"
cd "$runtime_root"
overrides=$(cat "$candidate_root/native-overrides.json")
exec /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python -m vllm.entrypoints.cli.main serve \
  /srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --served-model-name "qwen38-w4-$label" --host 127.0.0.1 --port 8003 \
  --dtype float16 --tensor-parallel-size 2 \
  --no-async-scheduling --disable-custom-all-reduce \
  --max-model-len 60000 --max-num-batched-tokens 512 --max-num-seqs 4 \
  --gpu-memory-utilization 0.965 --language-model-only \
  --enable-expert-parallel --enable-ep-weight-filter \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 --mamba-cache-mode align --enable-prefix-caching \
  --enable-chunked-prefill --enable-prompt-tokens-details \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --cudagraph-metrics --enable-logging-iteration-details \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,6]}' \
  --hf-overrides "$overrides" --limit-mm-per-prompt '{"image":0,"video":0}'
