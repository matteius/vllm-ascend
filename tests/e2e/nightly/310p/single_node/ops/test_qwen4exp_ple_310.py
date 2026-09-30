# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.ops.ple import (
    ple_decode_310,
    ple_gate,
    ple_short_conv,
)
from vllm_ascend.utils import enable_custom_op


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


@pytest.mark.parametrize("num_tokens", [1, 3])
def test_fused_ple_decode_matches_fp32_reference_and_graph_replay(num_tokens: int) -> None:
    _require_310p()
    enable_custom_op()
    generator = torch.Generator().manual_seed(318 + num_tokens)
    group_size = 2560
    hc_groups = 4
    hidden_size = group_size * hc_groups
    projected = (
        torch.randn((num_tokens, hidden_size + group_size), dtype=torch.float16, generator=generator) * 0.1
    ).npu()
    hidden = (torch.randn((num_tokens, hidden_size), dtype=torch.float16, generator=generator) * 0.1).npu()
    norm_key = (torch.randn(hidden_size, dtype=torch.float16, generator=generator) * 0.05).npu()
    norm_query = (torch.randn(hidden_size, dtype=torch.float16, generator=generator) * 0.05).npu()
    norm_conv = (torch.randn(hidden_size, dtype=torch.float16, generator=generator) * 0.05).npu()
    current_weight = (torch.randn(hidden_size, dtype=torch.float32, generator=generator) * 0.05).npu()
    output = torch.empty_like(hidden)
    eps = 1e-6

    def reference() -> torch.Tensor:
        key, value = projected.split((hidden_size, group_size), dim=-1)
        gated, conv_input = ple_gate(
            key,
            value,
            hidden,
            norm_key,
            norm_query,
            norm_conv,
            eps,
            accum_dtype=torch.float32,
        )
        conv = conv_input * current_weight
        return (hidden.float() + gated + conv * torch.sigmoid(conv)).half()

    def invoke() -> torch.Tensor:
        return ple_decode_310(
            projected,
            hidden,
            norm_key,
            norm_query,
            norm_conv,
            current_weight,
            eps,
            output=output,
        )

    expected = reference()
    torch.testing.assert_close(invoke().cpu(), expected.cpu(), rtol=7e-3, atol=4e-3)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            invoke()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = invoke()

    previous = None
    for phase in range(3):
        projected.add_(0.015625 * (phase + 1))
        hidden.add_(0.00390625)
        expected = reference().cpu()
        captured.fill_(float("nan"))
        graph.replay()
        actual = captured.cpu()
        torch.testing.assert_close(actual, expected, rtol=7e-3, atol=4e-3)
        assert torch.isfinite(actual).all()
        if previous is not None:
            assert not torch.equal(actual, previous)
        previous = actual
