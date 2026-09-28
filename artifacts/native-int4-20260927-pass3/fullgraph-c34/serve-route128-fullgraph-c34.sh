#!/usr/bin/env bash
set -eo pipefail

runtime_root=${1:-/srv/ai/src/native-int4-w4a8.KiuhBN/runtime-route128}
visible_devices=${2:-0,1,2,3}
port=${3:-8004}
max_model_len=${4:-262144}
candidate_root=/srv/ai/src/native-int4-w4a8.KiuhBN
model_root=/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i

[[ "$visible_devices" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || {
  echo "visible_devices must name four NPUs" >&2
  exit 2
}
[[ "$port" =~ ^[0-9]+$ ]] || exit 2
[[ "$max_model_len" =~ ^[0-9]+$ ]] || exit 2
[[ -d "$runtime_root/vllm_ascend" ]] || {
  echo "missing runtime: $runtime_root" >&2
  exit 2
}
grep -Eq '^[[:space:]]*MAX_CUBE_ROUTES[[:space:]]*=[[:space:]]*128[[:space:]]*$' \
  "$runtime_root/vllm_ascend/models/qwen4_exp/w4_moe.py" || {
  echo "runtime is not the route-128 candidate: $runtime_root" >&2
  exit 2
}

# shellcheck source=/dev/null
source /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES=$visible_devices
export TASK_QUEUE_ENABLE=1
unset HCCL_OP_EXPANSION_MODE || true
export PYTHONPATH=$runtime_root:/srv/ai/src/vllm-opensensor:${PYTHONPATH:-}
export ASCEND_CUSTOM_OPP_PATH=$candidate_root/ops-pass3-route128/vendors/native_int4_w4a8_transformer:/srv/ai/src/qwen38-w4-native-build.zHZtEd/ops-routeplan-r4/vendors/qwen_w4_native_transformer:${ASCEND_CUSTOM_OPP_PATH:-}
export VLLM_ASCEND_KV_CACHE_FRACTION=0.80
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_ASCEND_LOG_REQUEST_TIMINGS=1

cd "$runtime_root"
overrides=$(cat "$candidate_root/native-overrides.json")
exec /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python -m vllm.entrypoints.cli.main serve \
  "$model_root" \
  --served-model-name qwen38-w4-route128-fullgraph-c34 \
  --host 127.0.0.1 --port "$port" \
  --dtype float16 --tensor-parallel-size 4 \
  --no-async-scheduling --disable-custom-all-reduce \
  --max-model-len "$max_model_len" --max-num-batched-tokens 512 --max-num-seqs 4 \
  --gpu-memory-utilization 0.94 --language-model-only \
  --enable-expert-parallel --enable-ep-weight-filter \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 --mamba-cache-mode align --enable-prefix-caching \
  --enable-chunked-prefill --enable-prompt-tokens-details \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --cudagraph-metrics --enable-logging-iteration-details \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[9,12]}' \
  --hf-overrides "$overrides" --limit-mm-per-prompt '{"image":0,"video":0}'
