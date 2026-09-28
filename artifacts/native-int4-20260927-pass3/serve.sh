#!/usr/bin/env bash
set -eo pipefail
variant=${1:?validated pass3 variant required}
runtime_root=${2:-/srv/ai/src/native-int4-w4a8.KiuhBN/runtime}
max_num_seqs=${3:-2}
[[ "$variant" =~ ^[a-z0-9-]+$ ]] || exit 2
[[ -d "$runtime_root/vllm_ascend" ]] || exit 2
[[ "$max_num_seqs" =~ ^[1-4]$ ]] || exit 2
if [[ "$variant" == "paired-mtp3" ]]; then
  speculative_tokens=3
  graph_sizes='[1,2,3,4,8]'
else
  speculative_tokens=2
  graph_sizes='[1,2,3,6]'
fi
speculative_config=$(printf '{"method":"mtp","num_speculative_tokens":%d}' "$speculative_tokens")
compilation_config=$(printf '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":%s}' "$graph_sizes")
source /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh
candidate_root=/srv/ai/src/native-int4-w4a8.KiuhBN
export PYTHONPATH=$runtime_root:/srv/ai/src/vllm-opensensor:$PYTHONPATH
export ASCEND_CUSTOM_OPP_PATH=$candidate_root/ops-pass3-$variant/vendors/native_int4_w4a8_transformer:/srv/ai/src/qwen38-w4-native-build.zHZtEd/ops-routeplan-r4/vendors/qwen_w4_native_transformer:${ASCEND_CUSTOM_OPP_PATH:-}
export VLLM_ASCEND_KV_CACHE_FRACTION=0.80
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_ASCEND_LOG_REQUEST_TIMINGS=1
cd "$runtime_root"
overrides=$(cat "$candidate_root/native-overrides.json")
exec /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python -m vllm.entrypoints.cli.main serve \
  /srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --served-model-name qwen38-w4-experimental --host 0.0.0.0 --port 8001 \
  --dtype float16 --tensor-parallel-size 4 \
  --no-async-scheduling --disable-custom-all-reduce \
  --max-model-len 262144 --max-num-batched-tokens 512 --max-num-seqs "$max_num_seqs" \
  --gpu-memory-utilization 0.94 --language-model-only \
  --enable-expert-parallel --enable-ep-weight-filter \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 --mamba-cache-mode align --enable-chunked-prefill \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --speculative-config "$speculative_config" \
  --cudagraph-metrics --enable-logging-iteration-details \
  --compilation-config "$compilation_config" \
  --hf-overrides "$overrides" --limit-mm-per-prompt '{"image":0,"video":0}'
