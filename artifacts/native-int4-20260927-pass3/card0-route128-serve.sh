#!/usr/bin/env bash
set -eo pipefail
label=${1:-card0-route128}
shared_expert_execution=${2:-tp_sharded}
case "$shared_expert_execution" in
  tp_sharded)
    overrides_file=/srv/ai/src/native-int4-w4a8.KiuhBN/native-overrides.json
    ;;
  replicated|replicated_overlap)
    overrides_file=/srv/ai/src/native-int4-w4a8.KiuhBN/pass3/native-overrides-$shared_expert_execution.json
    ;;
  *)
    echo "unsupported shared expert execution: $shared_expert_execution" >&2
    exit 2
    ;;
esac
source /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh
set -u
candidate_root=/srv/ai/src/native-int4-w4a8.KiuhBN
runtime_root=$candidate_root/runtime-route128
export ASCEND_RT_VISIBLE_DEVICES=0,1
export TASK_QUEUE_ENABLE=1
unset HCCL_OP_EXPANSION_MODE || true
export PYTHONPATH=$runtime_root:/srv/ai/src/vllm-opensensor:${PYTHONPATH:-}
export ASCEND_CUSTOM_OPP_PATH=$candidate_root/ops-pass3-route128/vendors/native_int4_w4a8_transformer:/srv/ai/src/qwen38-w4-native-build.zHZtEd/ops-routeplan-r4/vendors/qwen_w4_native_transformer:${ASCEND_CUSTOM_OPP_PATH:-}
export VLLM_ASCEND_KV_CACHE_FRACTION=0.80
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_ASCEND_LOG_REQUEST_TIMINGS=1
cd "$runtime_root"
overrides=$(cat "$overrides_file")
exec /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python -m vllm.entrypoints.cli.main serve \
  /srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --served-model-name "qwen38-w4-$label" --host 127.0.0.1 --port 8002 \
  --dtype float16 --tensor-parallel-size 2 \
  --no-async-scheduling --disable-custom-all-reduce \
  --max-model-len 65536 --max-num-batched-tokens 512 --max-num-seqs 4 \
  --gpu-memory-utilization 0.965 --language-model-only \
  --enable-expert-parallel --enable-ep-weight-filter \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 --mamba-cache-mode align --enable-prefix-caching \
  --enable-chunked-prefill --enable-prompt-tokens-details \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --cudagraph-metrics --enable-logging-iteration-details \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,6]}' \
  --hf-overrides "$overrides" --limit-mm-per-prompt '{"image":0,"video":0}'
