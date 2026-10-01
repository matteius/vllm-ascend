# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-contract tests for the 310P recurrent GDN state dtype."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OP_ROOT = REPO_ROOT / "csrc" / "attention" / "recurrent_gated_delta_rule_v310"


def _source(relative_path: str) -> str:
    return (OP_ROOT / relative_path).read_text()


def test_recurrent_gdn_advertises_fp16_and_fp32_state():
    op_def = _source("op_host/recurrent_gated_delta_rule_v310_def.cpp")
    aclnn = _source("op_host/op_api/aclnn_recurrent_gated_delta_rule_v310.cpp")

    assert op_def.count(".DataType({ge::DT_FLOAT16, ge::DT_FLOAT})") == 2
    assert "STATE_TYPE_SUPPORT_LIST = {DataType::DT_FLOAT16, DataType::DT_FLOAT}" in aclnn


def test_recurrent_gdn_registers_complete_dtype_signatures():
    op_def = _source("op_host/recurrent_gated_delta_rule_v310_def.cpp")
    dtype_lists = re.findall(r"\.DataType\(\{([^}]*)\}\)", op_def)
    format_lists = re.findall(r"\.FormatList\(\{([^}]*)\}\)", op_def)

    assert dtype_lists
    assert {len(dtype_list.split(",")) for dtype_list in dtype_lists} == {2}
    assert len(format_lists) == len(dtype_lists)
    assert {len(format_list.split(",")) for format_list in format_lists} == {2}
    assert ".DynamicFormatFlag(false)" in op_def


def test_recurrent_gdn_preserves_state_dtype_through_dispatch():
    infer = _source("op_host/recurrent_gated_delta_rule_v310_infershape.cpp")
    tiling = _source("op_host/recurrent_gated_delta_rule_v310_tiling.cpp")
    kernel = _source("op_kernel/recurrent_gated_delta_rule_v310.cpp")

    assert "SetOutputDataType(1, context->GetInputDataType(STATE_INDEX))" in infer
    assert "tilingKey_ = stateDtype_ == ge::DT_FLOAT ? 1 : 0" in tiling
    assert "RGDR<half, half, half>" in kernel
    assert "RGDR<half, float, half>" in kernel


def test_recurrent_gdn_ub_budget_uses_state_element_width():
    tiling = _source("op_host/recurrent_gated_delta_rule_v310_tiling.cpp")

    assert tiling.count("stateElementBytes_ * (1 + static_cast<int64_t>(") == 2
    assert "(2 + static_cast<int64_t>(2 * stateOutBufferNum))" not in tiling
    assert "(2 + static_cast<int64_t>(2 * selected.stateOutBufferNum))" not in tiling
