# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact W4 down projection and route reduction on Ascend 310P."""

import pytest
import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.w4a8_int4 import (
    pack_activation_device,
    pack_native_metadata,
    pack_native_weight,
)
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(autouse=True, scope="module")
def require_kernel():
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()


def payload(routes: int):
    generator = torch.Generator().manual_seed(311)
    inputs = (torch.randn(routes, 640, generator=generator) * 0.1).half()
    codes = torch.randint(-128, 128, (3, 2560, 320), dtype=torch.int8, generator=generator)
    scales = (torch.rand(3, 2560, 5, generator=generator) * 0.02 + 0.001).half()
    offsets = torch.randint(-8, 8, scales.shape, dtype=torch.int8, generator=generator)
    packed_weights, weight_sums = zip(*(pack_native_weight(weight) for weight in codes))
    banks = (
        torch.stack(packed_weights).npu(),
        torch.stack([pack_native_metadata(scale) for scale in scales]).npu(),
        torch.stack([pack_native_metadata(offset).half() for offset in offsets]).npu(),
        torch.stack(weight_sums).npu(),
    )
    return inputs, banks


def ordered_reference(prepared, banks, route_ids, route_weights):
    route_rows = (
        torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310(*prepared, *banks, route_ids)
        .float()
        .reshape(route_weights.shape[0], route_weights.shape[1], 2560)
    )
    result = torch.zeros(route_weights.shape[0], 2560, dtype=torch.float32, device="npu")
    for route in range(route_weights.shape[1]):
        result = result + route_rows[:, route] * route_weights[:, route, None]
    return result


def test_down_reduce_meta_shape_and_dtype():
    output = torch.ops._C_ascend.npu_qwen_w4_a8_int4_down_reduce_310(
        torch.empty(3, 320, dtype=torch.int8, device="meta"),
        torch.empty(3, 320, dtype=torch.int8, device="meta"),
        torch.empty(3, 5, 8, dtype=torch.float32, device="meta"),
        torch.empty(3, 5, 8, dtype=torch.float32, device="meta"),
        torch.empty(3, 2560, 320, dtype=torch.int8, device="meta"),
        torch.empty(3, 2560, 5, dtype=torch.float16, device="meta"),
        torch.empty(3, 2560, 5, dtype=torch.float16, device="meta"),
        torch.empty(3, 2560, 5, dtype=torch.float16, device="meta"),
        torch.empty(30, dtype=torch.int32, device="meta"),
        torch.empty(3, 10, dtype=torch.float32, device="meta"),
    )
    assert output.shape == (3, 2560)
    assert output.dtype == torch.float32


def test_down_reduce_changing_inputs_routes_and_weights_graph():
    tokens, top_k = 3, 10
    inputs, banks = payload(tokens * top_k)
    prepared = list(pack_activation_device(inputs.npu()))
    route_ids = torch.zeros(tokens * top_k, dtype=torch.int32, device="npu")
    route_weights = torch.empty(tokens, top_k, dtype=torch.float32, device="npu")
    fused = torch.ops._C_ascend.npu_qwen_w4_a8_int4_down_reduce_310

    def invoke():
        return fused(*prepared, *banks, route_ids, route_weights)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            invoke()
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = invoke()

    for phase in range(5):
        changed = inputs * (phase - 1) + 0.0125 * phase
        changed_prepared = pack_activation_device(changed.npu())
        for target, value in zip(prepared, changed_prepared):
            target.copy_(value)
        ids = (torch.arange(tokens * top_k, dtype=torch.int32) * 7 + phase) % 6 - 1
        if phase == 2:
            ids.fill_(-1)
        elif phase == 4:
            ids.copy_(torch.tensor(([2, 0, 2, 1, 3, -1, 0, 2, 1, 4] * tokens), dtype=torch.int32))
        weights = torch.rand(tokens, top_k, generator=torch.Generator().manual_seed(500 + phase), dtype=torch.float32)
        weights[:, 0] += 0.000031 * (phase + 1)
        route_ids.copy_(ids)
        route_weights.copy_(weights)
        graph.replay()
        expected = ordered_reference(prepared, banks, route_ids, route_weights)
        torch.testing.assert_close(captured.cpu(), expected.cpu(), rtol=0, atol=0)
