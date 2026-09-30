#!/usr/bin/env bash
set -eo pipefail

source /srv/ai/bin/ascend-env.sh
source /srv/ai/venvs/qwen38-w4-test-ce1862/bin/activate
set -u

root=/srv/ai/src/glm-grouped-batchwrite-20260930
split_vendor=/srv/ai/src/kda-persistent-scores-opp/vendors/custom_transformer
grouped_vendor=/srv/ai/src/glm-grouped-batchwrite-20260930/opp-grouped-batchwrite/vendors/custom_transformer
mla_vendor=/srv/ai/src/glm-mla-opp/vendors/custom_transformer
base_vendor=/srv/ai/src/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export PYTHONPATH="$root:${PYTHONPATH:-}"
export ASCEND_CUSTOM_OPP_PATH="$split_vendor:$grouped_vendor:$mla_vendor:$base_vendor"
export LD_LIBRARY_PATH="$split_vendor/op_api/lib:$grouped_vendor/op_api/lib:$mla_vendor/op_api/lib:$base_vendor/op_api/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export SOC_VERSION=ascend310p1
export TASK_QUEUE_ENABLE=1
export OMP_NUM_THREADS=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export VLLM_ASCEND_310P_ENABLE_MLA=1
export VLLM_ASCEND_LOG_REQUEST_TIMINGS=1

cd "$root"
exec python -m vllm.entrypoints.cli.main serve \
  /srv/ai/models/GLM-5.3-Flash-W4through32-noclip-310p \
  --served-model-name glm53-flash-ascend-profile \
  --host 127.0.0.1 --port 8003 \
  --dtype float16 --tensor-parallel-size 4 \
  --max-model-len 16384 --max-num-seqs 4 --max-num-batched-tokens 512 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --gpu-memory-utilization 0.90 --num-gpu-blocks-override 128 \
  --enforce-eager --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser poolside_v1 \
  --hf-overrides '{"architectures":["Glm5NextW2ForCausalLM"]}' \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enable-logging-iteration-details
