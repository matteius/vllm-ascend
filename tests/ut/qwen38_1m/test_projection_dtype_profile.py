# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tools.qwen4exp.profile_projection_dtypes_310 import ACTIVATION_SCALE, WEIGHT_SCALE, matched_inputs
from vllm_ascend.models.qwen4_exp.w4_moe import dequantize
from vllm_ascend.models.qwen4_exp.w4a8_int4 import quantize_activation_limbs


def unpack(packed):
    low, high = packed & 15, (packed >> 4) & 15
    pairs = torch.stack([low, high], -1).flatten(-2)
    return torch.where(pairs >= 8, pairs - 16, pairs).float()


@pytest.mark.parametrize("rows,width,outputs", [(1, 256, 128), (3, 2560, 1280), (16, 640, 2560)])
def test_dtype_fixture_represents_identical_matrix_arithmetic(rows, width, outputs):
    x, weight, qa, qw, packed, scales, offsets = matched_inputs(rows, width, outputs)
    assert torch.equal(x, qa.half() * ACTIVATION_SCALE)
    assert torch.equal(weight, qw.half() * WEIGHT_SCALE)
    assert torch.equal(dequantize(packed, scales, offsets, 128).half(), weight)
    low, high, activation_scale, sums = quantize_activation_limbs(x)
    assert torch.equal(unpack(low) + 16 * unpack(high) + 8, qa.float())
    assert torch.all(activation_scale == ACTIVATION_SCALE)
    assert torch.equal(sums, qa.float().reshape(rows, -1, 128).sum(-1))


def test_dtype_fixture_rejects_unsupported_shapes():
    with pytest.raises(ValueError):
        matched_inputs(0, 256, 128)
    with pytest.raises(ValueError):
        matched_inputs(1, 129, 128)
