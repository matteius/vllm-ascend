# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile one real-weight W4 MoE TP partial with synthetic activations.

This is an isolated layer diagnostic, NOT whole-model speed or quality evidence.
There is no collective: expert ownership and shared-expert slicing match one TP
rank, but its partial output is returned directly. Stop other NPU workloads and
use an external temperature watchdog when running this diagnostic.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torch_npu
from safetensors import safe_open

from tools.qwen4exp.benchmark_w4_projection_310 import timed_ms
from vllm_ascend.models.qwen4_exp.dtype_policy import Qwen4ExpDtypePolicy
from vllm_ascend.models.qwen4_exp.model import _format_eager_linear_weights_npu
from vllm_ascend.models.qwen4_exp.moe import route_topk
from vllm_ascend.models.qwen4_exp.w4_moe import EXPERT_NAME, W4SparseMoE
from vllm_ascend.utils import enable_custom_op


def load_layer(model, layer_number, rank, tp_size, backend):
    config = json.loads((model / "config.json").read_text())["text_config"]
    config["ascend_expert_quantization"]["backend"] = backend
    layer = W4SparseMoE(
        config=SimpleNamespace(**config), dtype_policy=Qwen4ExpDtypePolicy(), expert_sharding=(rank, tp_size)
    )
    # Deliberate diagnostic-only bypass. Never patch the running model's reducer.
    layer._tp_reduce = lambda partial: partial
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer_number}.mlp."
    shards = defaultdict(list)
    for name, shard in index.items():
        if not name.startswith(prefix):
            continue
        match = EXPERT_NAME.fullmatch(name)
        if match and not layer.expert_offset <= int(match[2]) < layer.expert_offset + layer.num_local_experts:
            continue
        shards[shard].append(name)
    shared = {}
    for filename, names in shards.items():
        with safe_open(model / filename, framework="pt", device="cpu") as shard:
            for name in names:
                tensor = shard.get_tensor(name)
                match = EXPERT_NAME.fullmatch(name)
                if match:
                    layer.load_projection(int(match[2]), match[3], match[4], tensor)
                elif name == prefix + "gate.weight":
                    layer.gate.copy_(tensor)
                else:
                    shared[name[len(prefix) :]] = tensor
    if layer.has_shared_expert:
        width = layer.local_shared_inter
        start, stop = rank * width, (rank + 1) * width
        layer.shared_gate_up.copy_(
            torch.cat(
                [
                    shared["shared_expert.gate_proj.weight"][start:stop],
                    shared["shared_expert.up_proj.weight"][start:stop],
                ]
            )
        )
        layer.shared_down.copy_(shared["shared_expert.down_proj.weight"][:, start:stop])
        layer.shared_expert_gate.copy_(shared["shared_expert_gate.weight"])
    return layer.npu().eval()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--backend", choices=["eager_dequant", "cube_310", "cube_310_tiled", "cube_310_routed"], default="cube_310"
    )
    parser.add_argument("--graph", action="store_true", help="validate and time device-routed layer replay")
    args = parser.parse_args()
    if args.output.exists() or (args.trace_dir and args.trace_dir.exists()):
        parser.error("choose new output/trace paths; preserve earlier evidence")
    if args.graph and args.backend != "cube_310_routed":
        parser.error("only cube_310_routed permits layer graph capture")
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    layer = load_layer(args.model, args.layer, args.rank, args.tp_size, args.backend)
    if args.graph:
        _format_eager_linear_weights_npu(layer)
    generator = torch.Generator().manual_seed(1024)
    with args.output.open("x") as output:
        for tokens in args.tokens:
            inputs = (torch.randn(tokens, layer.gate.shape[1], generator=generator) * 0.1).half().npu()
            _, ids = route_topk(
                F.linear(inputs, layer.gate),
                layer.top_k,
                renormalize=layer.renormalize,
                routed_scaling_factor=layer.routed_scaling_factor,
            )
            local = ids.cpu() - layer.expert_offset
            selected = local[(local >= 0) & (local < layer.num_local_experts)]
            result = layer(inputs).cpu()
            if not torch.isfinite(result).all():
                raise RuntimeError("non-finite partial MoE output")
            comparison = None
            if layer.device_routing:
                layer.device_routing = False
                reference = layer(inputs).cpu()
                torch.testing.assert_close(result, reference, rtol=0.01, atol=0.003)
                comparison = timed_ms(lambda inputs=inputs: layer(inputs), args.iterations, args.repeats)
                layer.device_routing = True
            replay = None
            graph_output_sha256 = None
            if args.graph:
                stream = torch.npu.Stream()
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    for _ in range(3):
                        layer(inputs)
                torch.npu.current_stream().wait_stream(stream)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    captured = layer(inputs)
                graph.replay()
                graph_result = captured.cpu()
                torch.testing.assert_close(graph_result, result, rtol=0.01, atol=0.003)
                graph_output_sha256 = hashlib.sha256(graph_result.numpy().tobytes()).hexdigest()
                replay = timed_ms(graph.replay, args.iterations, args.repeats)
                del graph, captured
            record = {
                "backend": args.backend,
                "layer": args.layer,
                "rank": args.rank,
                "tp_size": args.tp_size,
                "collective": False,
                "weights": "real_checkpoint",
                "activations": "synthetic_seed_1024_std_0.1",
                "input_sha256": hashlib.sha256(inputs.cpu().numpy().tobytes()).hexdigest(),
                "output_sha256": hashlib.sha256(result.numpy().tobytes()).hexdigest(),
                "graph_output_sha256": graph_output_sha256,
                "production_nz_projections": args.graph,
                "tokens": tokens,
                "local_routes": selected.numel(),
                "distinct_local_experts": selected.unique().numel(),
                "latency": timed_ms(lambda inputs=inputs: layer(inputs), args.iterations, args.repeats),
                "host_route_comparison": comparison,
                "graph_replay": replay,
                "torch_allocated_bytes": torch.npu.memory_allocated(),
                "torch_reserved_bytes": torch.npu.memory_reserved(),
            }
            line = json.dumps(record)
            print(line, flush=True)
            output.write(line + "\n")
            output.flush()
            if args.trace_dir:
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    record_shapes=True,
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        str(args.trace_dir / f"tokens-{tokens}")
                    ),
                ) as profiler:
                    for _ in range(3):
                        layer(inputs)
                        profiler.step()
                    torch.npu.synchronize()


if __name__ == "__main__":
    main()
