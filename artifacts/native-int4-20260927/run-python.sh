#!/usr/bin/env bash
set -eo pipefail
source /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh
candidate_root=/srv/ai/src/native-int4-w4a8.KiuhBN
export PYTHONPATH=$candidate_root/runtime:/srv/ai/src/vllm-opensensor:$PYTHONPATH
export ASCEND_CUSTOM_OPP_PATH=$candidate_root/ops-delivery/vendors/native_int4_w4a8_transformer:/srv/ai/src/qwen38-w4-native-build.zHZtEd/ops-routeplan-r4/vendors/qwen_w4_native_transformer:${ASCEND_CUSTOM_OPP_PATH:-}
cd "$candidate_root/runtime"
exec taskset -c 0-5 /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python "$@"
