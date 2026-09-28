"""Compare grouped and extended-routed W4 MoE graph replay at 90/120 routes."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import torch
import torch_npu

from vllm_ascend.models.qwen4_exp import w4_moe
from vllm_ascend.models.qwen4_exp.dtype_policy import Qwen4ExpDtypePolicy
from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE
from vllm_ascend.utils import enable_custom_op


def config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=2560,
        moe_intermediate_size=640,
        shared_expert_intermediate_size=0,
        num_experts=16,
        num_experts_per_tok=10,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        ascend_expert_quantization={
            "backend": "cube_310_int4_a8",
            "activation_quantization": "int8_per_group",
            "bits": 4,
            "format": "qwen4exp_w4a16_group_v1",
            "group_size": 128,
            "offset_dtype": "int8",
            "packing": "signed_int4_low_nibble_first_in_axis",
            "scale_dtype": "float16",
            "symmetric": False,
        },
    )


def capture(layer: W4SparseMoE, inputs: torch.Tensor, route_limit: int):
    w4_moe.MAX_CUBE_ROUTES = route_limit
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            layer(inputs)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = layer(inputs)
    return graph, output


def replay(graph: torch.npu.NPUGraph, iterations: int) -> float:
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        graph.replay()
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1000 / iterations


def main() -> None:
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    torch.manual_seed(20260928)
    layer = W4SparseMoE(config=config(), dtype_policy=Qwen4ExpDtypePolicy()).npu().eval()
    with torch.no_grad():
        layer.gate.normal_(std=0.01)
        for bank in layer.projections.values():
            bank.weight_scale.fill_(0.01)
    results = []
    for tokens in (9, 12):
        inputs = torch.randn(tokens, 2560, dtype=torch.float16, device="npu") * 0.05
        grouped_graph, grouped_output = capture(layer, inputs, 80)
        routed_graph, routed_output = capture(layer, inputs, 128)
        torch.testing.assert_close(routed_output.cpu(), grouped_output.cpu(), rtol=0, atol=0)
        grouped_ms = replay(grouped_graph, 200)
        routed_ms = replay(routed_graph, 200)
        results.append(
            {
                "tokens": tokens,
                "routes": tokens * 10,
                "grouped_ms": grouped_ms,
                "routed_ms": routed_ms,
                "speedup": grouped_ms / routed_ms,
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
