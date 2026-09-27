#!/usr/bin/env bash
set -eo pipefail
variant=${1:?baseline or native required}
case "$variant" in baseline|native) ;; *) exit 2 ;; esac
source /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh
candidate_root=/srv/ai/src/native-int4-w4a8.KiuhBN
export PYTHONPATH=$candidate_root/runtime:/srv/ai/src/vllm-opensensor:$PYTHONPATH
export ASCEND_CUSTOM_OPP_PATH=$candidate_root/ops-delivery/vendors/native_int4_w4a8_transformer:/srv/ai/src/qwen38-w4-native-build.zHZtEd/ops-routeplan-r4/vendors/qwen_w4_native_transformer:${ASCEND_CUSTOM_OPP_PATH:-}
export VLLM_ASCEND_KV_CACHE_FRACTION=0.80
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_ASCEND_LOG_REQUEST_TIMINGS=1
cd "$candidate_root/runtime"
overrides=$(cat "$candidate_root/$variant-overrides.json")
exec /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python -m vllm.entrypoints.cli.main serve \
  /srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --served-model-name qwen38-w4-experimental --host 0.0.0.0 --port 8001 \
  --dtype float16 --tensor-parallel-size 4 \
  --no-async-scheduling --disable-custom-all-reduce \
  --max-model-len 262144 --max-num-batched-tokens 512 --max-num-seqs 2 \
  --gpu-memory-utilization 0.94 --language-model-only \
  --enable-expert-parallel --enable-ep-weight-filter \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 --mamba-cache-mode align --enable-chunked-prefill \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --cudagraph-metrics --enable-logging-iteration-details \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,6]}' \
  --hf-overrides "$overrides" --limit-mm-per-prompt '{"image":0,"video":0}'
