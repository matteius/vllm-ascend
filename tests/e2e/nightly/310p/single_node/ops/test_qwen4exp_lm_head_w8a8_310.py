# SPDX-License-Identifier: Apache-2.0
"""Real-geometry dynamic-W8A8 LM-head gates for Ascend 310P."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.lm_head_w8a8 import Qwen4ExpDynamicW8A8LMHeadMethod

HIDDEN_SIZE = 2560
LOCAL_VOCAB_SIZE = 62080
REFERENCE_VOCAB_SIZE = 64


def _require_310p() -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")


@pytest.fixture(scope="module")
def prepared_head():
    _require_310p()
    torch_npu.npu.set_device(0)
    torch.manual_seed(310)
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.empty(LOCAL_VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.float16, device="npu:0").uniform_(-0.02, 0.02),
            requires_grad=False,
        ),
    )
    method = Qwen4ExpDynamicW8A8LMHeadMethod()
    method.process_weights_after_loading(layer)
    yield SimpleNamespace(layer=layer, method=method)
    del layer
    torch_npu.npu.empty_cache()


@pytest.mark.parametrize("tokens", [1, 3])
def test_real_shard_matches_integer_reference(prepared_head, tokens: int) -> None:
    layer, method = prepared_head.layer, prepared_head.method
    x = torch.randn(tokens, HIDDEN_SIZE, dtype=torch.float16, device="npu:0")
    actual = method.apply(layer, x)
    assert actual.shape == (tokens, LOCAL_VOCAB_SIZE)

    x_cpu = x.cpu().float()
    input_scale = x_cpu.abs().amax(dim=-1, keepdim=True) / 127
    quantized_x = torch.round(x_cpu / input_scale).clamp(-127, 127)
    weight = layer.weight[:, :REFERENCE_VOCAB_SIZE].cpu().float()
    weight_scale = layer.weight_scale[:REFERENCE_VOCAB_SIZE].cpu().float()
    expected = (quantized_x @ weight) * input_scale * weight_scale
    torch.testing.assert_close(actual[:, :REFERENCE_VOCAB_SIZE].cpu(), expected.half(), rtol=0.002, atol=0.02)


def test_changing_input_graph_replay(prepared_head) -> None:
    layer, method = prepared_head.layer, prepared_head.method
    static_x = torch.randn(3, HIDDEN_SIZE, dtype=torch.float16, device="npu:0")

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            method.apply(layer, static_x)
    torch.npu.current_stream().wait_stream(stream)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = method.apply(layer, static_x)

    previous = None
    for multiplier in (0.5, -1.25):
        changed = torch.randn_like(static_x) * multiplier
        static_x.copy_(changed)
        graph.replay()
        expected = method.apply(layer, changed)
        actual = captured.cpu()
        torch.testing.assert_close(actual, expected.cpu(), rtol=0, atol=0)
        if previous is not None:
            assert not torch.equal(actual, previous)
        previous = actual.clone()
