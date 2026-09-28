#!/usr/bin/env bash
set -eo pipefail
max_len=${1:?maximum context length required}
mode=${2:?mtp1, mtp2, mtp3, plain, mtp2-noprefix, or plain-noprefix required}
utilization=${3:-0.94}
kv_fraction=${4:-0.80}
visible_devices=${5:-0,1}
port=${6:-8002}
label=${7:-single-card}
runtime_root=${8:-/srv/ai/src/native-int4-w4a8.KiuhBN/runtime}
[[ "$max_len" =~ ^[0-9]+$ ]] || exit 2
[[ "$visible_devices" =~ ^[0-9]+,[0-9]+$ ]] || exit 2
[[ "$port" =~ ^[0-9]+$ ]] || exit 2
[[ "$label" =~ ^[a-z0-9-]+$ ]] || exit 2
[[ -d "$runtime_root/vllm_ascend" ]] || exit 2
case "$mode" in
  mtp1)
    speculative_args=(--speculative-config '{"method":"mtp","num_speculative_tokens":1}')
    graph_sizes='[1,2]'
    cache_args=(--mamba-cache-mode align --enable-prefix-caching)
    ;;
  mtp2)
    speculative_args=(--speculative-config '{"method":"mtp","num_speculative_tokens":2}')
    graph_sizes='[1,2,3]'
    cache_args=(--mamba-cache-mode align --enable-prefix-caching)
    ;;
  mtp3)
    speculative_args=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
    graph_sizes='[1,2,3,4]'
    cache_args=(--mamba-cache-mode align --enable-prefix-caching)
    ;;
  plain)
    speculative_args=()
    graph_sizes='[1]'
    cache_args=(--mamba-cache-mode align --enable-prefix-caching)
    ;;
  mtp2-noprefix)
    speculative_args=(--speculative-config '{"method":"mtp","num_speculative_tokens":2}')
    graph_sizes='[1,2,3]'
    cache_args=(--mamba-cache-mode none --no-enable-prefix-caching)
    ;;
  plain-noprefix)
    speculative_args=()
    graph_sizes='[1]'
    cache_args=(--mamba-cache-mode none --no-enable-prefix-caching)
    ;;
  *) exit 2 ;;
esac
# shellcheck source=/dev/null
source /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh
set -u
candidate_root=/srv/ai/src/native-int4-w4a8.KiuhBN
export ASCEND_RT_VISIBLE_DEVICES=$visible_devices
export PYTHONPATH=$runtime_root:/srv/ai/src/vllm-opensensor:${PYTHONPATH:-}
export ASCEND_CUSTOM_OPP_PATH=$candidate_root/ops-pass3-route-cache/vendors/native_int4_w4a8_transformer:/srv/ai/src/qwen38-w4-native-build.zHZtEd/ops-routeplan-r4/vendors/qwen_w4_native_transformer:${ASCEND_CUSTOM_OPP_PATH:-}
export VLLM_ASCEND_KV_CACHE_FRACTION=$kv_fraction
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_ASCEND_LOG_REQUEST_TIMINGS=1
cd "$runtime_root"
overrides=$(cat "$candidate_root/native-overrides.json")
compilation_config=$(printf '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":%s}' "$graph_sizes")
exec /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python -m vllm.entrypoints.cli.main serve \
  /srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --served-model-name "qwen38-w4-$label" --host 127.0.0.1 --port "$port" \
  --dtype float16 --tensor-parallel-size 2 \
  --no-async-scheduling --disable-custom-all-reduce \
  --max-model-len "$max_len" --max-num-batched-tokens 512 --max-num-seqs 1 \
  --gpu-memory-utilization "$utilization" --language-model-only \
  --enable-expert-parallel --enable-ep-weight-filter \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 "${cache_args[@]}" --enable-chunked-prefill \
  --enable-prompt-tokens-details \
  "${speculative_args[@]}" \
  --cudagraph-metrics --enable-logging-iteration-details \
  --compilation-config "$compilation_config" \
  --hf-overrides "$overrides" --limit-mm-per-prompt '{"image":0,"video":0}'
