# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare separate/combined W4 gate-up on a real-weight TP partial.

This diagnostic holds both layouts for comparison; a production integration
must replace, not duplicate, the original packed banks. No runtime module is
patched globally and no collective or whole-model speed is measured here.
"""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu

from tools.qwen4exp.benchmark_w4_projection_310 import timed_ms
from tools.qwen4exp.profile_w4_layer_310 import load_layer
from vllm_ascend.models.qwen4_exp.model import _format_eager_linear_weights_npu
from vllm_ascend.models.qwen4_exp.moe import route_topk
from vllm_ascend.models.qwen4_exp.w4_moe import KINDS
from vllm_ascend.utils import enable_custom_op


def _combined_routed(layer, banks, block_input, weights, ids):
    tokens, hidden = block_input.shape
    inputs = block_input[:, None, :].expand(-1, layer.top_k, -1).reshape(-1, hidden).contiguous()
    local_ids = (ids - layer.expert_offset).to(torch.int32).flatten().contiguous()
    gate_up = torch.ops._C_ascend.npu_qwen_w4_routed_matmul_310(inputs, *banks, local_ids)
    gate, up = gate_up.to(layer.compute_dtype).chunk(2, dim=-1)
    activation = (F.silu(gate) * up).to(layer.params_dtype)
    output = layer.projections["down_proj"].routed_linear(activation, local_ids).to(layer.compute_dtype)
    return (output.reshape(tokens, layer.top_k, hidden) * weights.unsqueeze(-1)).sum(dim=1)


def _separate_routed(layer, banks, block_input, weights, ids):
    tokens, hidden = block_input.shape
    inputs = block_input[:, None, :].expand(-1, layer.top_k, -1).reshape(-1, hidden).contiguous()
    local_ids = (ids - layer.expert_offset).to(torch.int32).flatten().contiguous()
    gate, up = (
        torch.ops._C_ascend.npu_qwen_w4_routed_matmul_310(inputs, *bank, local_ids).to(layer.compute_dtype)
        for bank in banks
    )
    activation = (F.silu(gate) * up).to(layer.params_dtype)
    output = layer.projections["down_proj"].routed_linear(activation, local_ids).to(layer.compute_dtype)
    return (output.reshape(tokens, layer.top_k, hidden) * weights.unsqueeze(-1)).sum(dim=1)


def _comparison_paths(layer):
    if "gate_up_proj" in layer.projections:
        # Only this diagnostic creates copies to measure the former launches.
        bank = layer.projections["gate_up_proj"]
        width = layer.intermediate_size
        split_banks = tuple(
            tuple(getattr(bank, kind)[:, start : start + width].contiguous() for kind in KINDS) for start in (0, width)
        )
        return (
            partial(_separate_routed, layer, split_banks),
            layer._forward_routed,
            sum(tensor.nbytes for tensors in split_banks for tensor in tensors),
            sum(getattr(bank, kind).nbytes for kind in KINDS),
        )
    banks = tuple(
        torch.cat([getattr(layer.projections[projection], kind) for projection in ("gate_proj", "up_proj")], dim=1)
        for kind in KINDS
    )
    original_bytes = sum(
        getattr(layer.projections[projection], kind).nbytes for projection in ("gate_proj", "up_proj") for kind in KINDS
    )
    return (
        layer._forward_routed,
        partial(_combined_routed, layer, banks),
        original_bytes,
        sum(bank.nbytes for bank in banks),
    )


def _capture(function):
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            function()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = function()
    graph.replay()
    torch.npu.synchronize()
    return graph, output


@torch.inference_mode()
def main():
    # Layer setup, paired replay validation and timings are deliberately in
    # one process so the binary, weights and device state are identical.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-binary", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or min(args.tokens + [args.iterations, args.repeats]) <= 0 or max(args.tokens) > 8:
        parser.error("require new output, positive iterations/repeats, and 1–8 tokens")
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    layer = load_layer(args.model, 0, 0, 4, "cube_310_routed")
    _format_eager_linear_weights_npu(layer)
    original_forward, combined_forward, original_bytes, combined_bytes = _comparison_paths(layer)
    assert original_bytes == combined_bytes
    generator = torch.Generator().manual_seed(1024)
    kernel_sha = hashlib.sha256(args.kernel_binary.read_bytes()).hexdigest()
    with args.output.open("x") as output_file:
        for tokens in args.tokens:
            cpu_inputs = (torch.randn(tokens, layer.gate.shape[1], generator=generator) * 0.1).half()
            inputs = cpu_inputs.npu()
            layer._forward_routed = original_forward
            reference = layer(inputs).cpu()
            separate_eager = timed_ms(partial(layer, inputs), args.iterations, args.repeats)
            separate_graph, separate_output = _capture(partial(layer, inputs))
            layer._forward_routed = combined_forward
            torch.testing.assert_close(layer(inputs).cpu(), reference, rtol=0, atol=0)
            combined_eager = timed_ms(partial(layer, inputs), args.iterations, args.repeats)
            combined_graph, combined_output = _capture(partial(layer, inputs))
            torch.testing.assert_close(combined_output.cpu(), reference, rtol=0, atol=0)
            paired = {"separate": [], "combined": []}
            for repeat in range(args.repeats):
                # Alternate measurement order to limit order/temperature bias.
                order = ("separate", "combined") if repeat % 2 == 0 else ("combined", "separate")
                for name in order:
                    graph = separate_graph if name == "separate" else combined_graph
                    paired[name].append(timed_ms(graph.replay, args.iterations, 1)["median_ms"])
            for phase in range(4):
                changed = torch.zeros_like(cpu_inputs) if phase == 3 else torch.roll(cpu_inputs, phase + 1, dims=-1)
                inputs.copy_(changed)
                separate_graph.replay()
                combined_graph.replay()
                torch.testing.assert_close(combined_output.cpu(), separate_output.cpu(), rtol=0, atol=0)
            inputs.copy_(cpu_inputs)
            _, ids = route_topk(
                F.linear(inputs, layer.gate),
                layer.top_k,
                renormalize=layer.renormalize,
                routed_scaling_factor=layer.routed_scaling_factor,
            )
            selected = ids.cpu() - layer.expert_offset
            selected = selected[(selected >= 0) & (selected < layer.num_local_experts)]
            record = {
                "tokens": tokens,
                "layer": 0,
                "rank": 0,
                "tp_size": 4,
                "collective": False,
                "real_weights": True,
                "not_model_throughput": True,
                "all_outputs_exact": True,
                "changing_input_replays": 4,
                "local_routes": selected.numel(),
                "distinct_local_experts": selected.unique().numel(),
                "original_gate_up_bytes": original_bytes,
                "combined_gate_up_bytes": combined_bytes,
                "both_layouts_resident_for_diagnostic_only": True,
                "separate_eager": separate_eager,
                "combined_eager": combined_eager,
                "paired_graph_trials_ms": paired,
                "kernel_sha256": kernel_sha,
                "input_sha256": hashlib.sha256(cpu_inputs.numpy().tobytes()).hexdigest(),
            }
            line = json.dumps(record)
            print(line, flush=True)
            output_file.write(line + "\n")
            output_file.flush()
            del separate_graph, combined_graph, separate_output, combined_output


if __name__ == "__main__":
    main()
