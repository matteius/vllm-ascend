#!/usr/bin/env bash
# Real-weight W4 service on the established Kilo port.
set -e
source /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh
vendor=/srv/ai/src/qwen38-w4-operator-build-r1/ops-byte-mask-r1/vendors/qwen_w4_probe_transformer
test -d "$vendor"
export ASCEND_CUSTOM_OPP_PATH=$vendor:${ASCEND_CUSTOM_OPP_PATH:-}
export VLLM_ASCEND_KV_CACHE_FRACTION=0.80
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_ASCEND_LOG_REQUEST_TIMINGS=1
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
  --hf-overrides '{"text_config":{"ascend_expert_quantization":{"backend":"cube_310_routed","bits":4,"format":"qwen4exp_w4a16_group_v1","group_size":128,"offset_dtype":"int8","packing":"signed_int4_low_nibble_first_in_axis","scale_dtype":"float16","symmetric":false}}}' \
  --limit-mm-per-prompt '{"image":0,"video":0}'
