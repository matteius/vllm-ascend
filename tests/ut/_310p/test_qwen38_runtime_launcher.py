# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "examples" / "start_qwen38_flash_next_w4_310p.sh"


def test_custom_opp_order_survives_plugin_bootstrap():
    launcher = LAUNCHER.read_text()

    assert "PACKAGED_OPP=${RUNTIME_ROOT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer" in launcher
    assert (
        'ASCEND_CUSTOM_OPP_PATH="${COHERENT_OPP}:${PACKAGED_OPP}:${RETAINED_OPP}'
        '${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"'
    ) in launcher
    assert (
        'LD_LIBRARY_PATH="${COHERENT_OPP}/op_api/lib:${PACKAGED_OPP}/op_api/lib:'
        '${RETAINED_OPP}/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"'
    ) in launcher

    coherent_index = launcher.index("${COHERENT_OPP}:${PACKAGED_OPP}:${RETAINED_OPP}")
    bootstrap_comment_index = launcher.index("Plugin bootstrap prepends PACKAGED_OPP")
    assert bootstrap_comment_index < coherent_index


def test_matching_qsa_abi_precedes_retained_fallback():
    launcher = LAUNCHER.read_text()

    packaged_index = launcher.index("${PACKAGED_OPP}:${RETAINED_OPP}")
    qsa_abi_comment_index = launcher.index("QSA host API")
    assert qsa_abi_comment_index < packaged_index
