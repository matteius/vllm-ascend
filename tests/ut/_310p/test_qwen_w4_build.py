# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build integration regressions for the separate Qwen group-W4 operator."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_qwen_w4_is_in_310p_manifest_and_shared_helper_invalidation():
    script = (REPO_ROOT / "csrc/build_aclnn.sh").read_text()
    branch = script.split('if [[ "$SOC_VERSION" =~ ^ascend310 ]]', 1)[1].split('elif [[ "$SOC_VERSION"', 1)[0]
    assert '"qwen_w4_group_matmul_v310"' in branch
    assert '"qwen_w4_routed_matmul_v310"' in branch
    invalidation = script.split("invalidate_stale_kernel_cache()", 1)[1].split("invalidate_stale_host_cache()", 1)[0]
    assert '"${op_name}" == "qwen_w4_group_matmul_v310"' in invalidation
    assert 'find "${ROOT_DIR}/csrc/moe/common/kernel_utils"' in invalidation
    assert '"${op_name}" == "qwen_w4_routed_matmul_v310"' in invalidation
    assert 'find "${ROOT_DIR}/csrc/gmm/qwen_w4_group_matmul_v310/op_kernel"' in invalidation


def test_older_readonly_vendor_scripts_are_repaired_before_replacement():
    script = (REPO_ROOT / "csrc/build_aclnn.sh").read_text()
    install = script.split('custom_ops_install_dir="${ROOT_DIR}/vllm_ascend/_cann_ops_custom"', 1)[1]
    repair = 'chmod u+w "${custom_ops_install_dir}/vendors/custom_transformer/scripts"'
    cleanup = 'find "$custom_ops_install_dir" -mindepth 1 -maxdepth 1'
    assert install.index(repair) < install.index(cleanup)
