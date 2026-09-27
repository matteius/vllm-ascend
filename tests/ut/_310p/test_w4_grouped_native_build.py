# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registration and native-integer implementation guardrails."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_device_grouped_and_native_int4_are_registered_for_310p():
    build_script = (REPO_ROOT / "csrc/build_aclnn.sh").read_text()
    binding = (REPO_ROOT / "csrc/torch_binding.cpp").read_text()
    for op in ("qwen_w4_grouped_matmul_v310", "qwen_w4_a8_int4_matmul_v310", "qwen_w4_a8_pack_v310"):
        assert f'"{op}"' in build_script
        definition = (REPO_ROOT / "csrc/gmm" / op / "op_host" / f"{op}_def.cpp").read_text()
        assert 'AddConfig("ascend310p"' in definition
        assert op in binding
    assert '"${op_name}" == "qwen_w4_grouped_matmul_v310"' in build_script


def test_native_int4_does_not_dequantize_weight_bank_to_fp16():
    kernel = (REPO_ROOT / "csrc/gmm/qwen_w4_a8_int4_matmul_v310/op_kernel/qwen_w4_a8_int4_matmul_v310.cpp").read_text()
    assert "ReinterpretCast<int4b_t>()" in kernel
    assert "Mmad(c_.Get<int32_t>()" in kernel
    assert "Product(low_" in kernel and "Product(high_" in kernel
    assert "QwenW4GroupMatmulV310Cube" not in kernel
