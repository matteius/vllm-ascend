# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.ops.ple import ple_short_conv


def _require_310p() -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")


def _decode_reference(
    conv_input: torch.Tensor,
    gated: torch.Tensor,
    outer: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    conv = conv_input.float() * weight[:, -1].float()
    conv = conv * torch.sigmoid(conv)
    return outer.float() + gated.float() + conv


def test_ple_decode_short_conv_matches_current_tap_for_changing_inputs() -> None:
    _require_310p()
    generator = torch.Generator().manual_seed(310)
    shape = (3, 10240)
    # Keep operands in the accumulation dtype to isolate the reduced math.
    conv_input = torch.randn(shape, dtype=torch.float32, generator=generator).to("npu")
    gated = torch.randn(shape, dtype=torch.float32, generator=generator).to("npu")
    outer = torch.randn(shape, dtype=torch.float32, generator=generator).to("npu")
    weight = torch.randn((shape[1], 4), dtype=torch.float32, generator=generator).to("npu")
    current_weight = weight[:, -1].contiguous()

    def invoke() -> torch.Tensor:
        return ple_short_conv(
            conv_input,
            gated,
            outer,
            weight,
            dilation=3,
            activation="silu",
            accum_dtype=torch.float32,
            current_weight=current_weight,
        )

    torch.testing.assert_close(invoke().cpu(), _decode_reference(conv_input, gated, outer, weight).cpu())

    for phase in range(3):
        conv_input.copy_(torch.randn(shape, dtype=torch.float32, generator=generator).to("npu") * (phase + 1))
        gated.add_(0.0625)
        outer.sub_(0.03125)
        actual = invoke().cpu()
        expected = _decode_reference(conv_input, gated, outer, weight).cpu()
        torch.testing.assert_close(actual, expected)
