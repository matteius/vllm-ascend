# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""310P correctness gates for the packed W4 reference backend (not its speed)."""

import pytest
import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.models.qwen4_exp.w4_moe import PackedExpertBank, dequantize, unpack_signed_int4


@pytest.fixture(autouse=True)
def require_310p():
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")


def test_all_packed_byte_values_unpack_exactly():
    packed = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8).reshape(16, 16)
    # Independent unsigned-byte reference, including both signed nibble halves.
    unsigned = packed.to(torch.int16) % 256
    nibbles = torch.stack((unsigned % 16, unsigned // 16), dim=-1).flatten(-2)
    expected = ((nibbles + 8) % 16 - 8).to(torch.int8)
    actual = unpack_signed_int4(packed.to("npu:0"))
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("outputs,inputs", [(640, 2560), (2560, 640)])
@pytest.mark.parametrize("tokens", [1, 7, 64])
def test_real_geometry_projection_matches_cpu_and_stays_packed(outputs, inputs, tokens):
    group_size = 128
    generator = torch.Generator().manual_seed(3104 + tokens)
    bank = PackedExpertBank(1, outputs, inputs, group_size)
    with torch.no_grad():
        bank.weight.copy_(torch.randint(-128, 128, bank.weight.shape, dtype=torch.int8, generator=generator))
        bank.weight_scale.copy_(torch.rand(bank.weight_scale.shape, generator=generator) * 0.01 + 0.001)
        bank.weight_offset.copy_(torch.randint(-8, 8, bank.weight_offset.shape, dtype=torch.int8, generator=generator))
    cpu_weight = dequantize(bank.weight[0], bank.weight_scale[0], bank.weight_offset[0], group_size)
    x = (torch.randn(tokens, inputs, generator=generator) * 0.1).half()
    expected = F.linear(x, cpu_weight.half())
    bank = bank.to("npu:0")
    npu_weight = dequantize(bank.weight[0], bank.weight_scale[0], bank.weight_offset[0], group_size)
    torch.testing.assert_close(npu_weight.cpu(), cpu_weight, rtol=0, atol=0)
    actual = bank.linear(x.to("npu:0"), 0)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.005, atol=0.01)
    assert torch.isfinite(actual).all()
    assert set(dict(bank.named_parameters())) == {"weight", "weight_scale", "weight_offset"}
    assert bank.weight.dtype == torch.int8
    assert bank.weight.shape == (1, outputs, inputs // 2)
    assert sum(value.numel() * value.element_size() for value in bank.parameters()) == (
        outputs * inputs // 2 + 3 * outputs * inputs // group_size
    )
