# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare real packed-W4 projections with eager dequantization on 310P.

Operator timings are not whole-model throughput. Run with the inference server
stopped, the same pinned runtime, and external temperature supervision.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu
from safetensors import safe_open

from vllm_ascend.models.qwen4_exp.w4_moe import dequantize, pack_cube_tiles
from vllm_ascend.utils import enable_custom_op


def timed_ms(function, iterations, repeats):
    for _ in range(3):
        function()
    torch.npu.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            function()
        torch.npu.synchronize()
        times.append((time.perf_counter() - start) * 1000 / iterations)
    return {"median_ms": statistics.median(times), "trials_ms": times}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 5, 64])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--label", default="unlabeled")
    parser.add_argument("--kernel-binary", type=Path)
    parser.add_argument(
        "--tiled", action="store_true", help="use lossless load-time NZ nibble encoding, without changing quantization"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; keep earlier measurements")
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires Ascend 310P")
    torch.npu.set_device(0)
    enable_custom_op()
    op = torch.ops._C_ascend.npu_qwen_w4_group_matmul_310
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    generator = torch.Generator().manual_seed(1024)
    binary_hash = hashlib.sha256(args.kernel_binary.read_bytes()).hexdigest() if args.kernel_binary else None
    with args.output.open("x") as output:
        for projection in ["gate_proj", "up_proj", "down_proj"]:
            prefix = f"model.language_model.layers.{args.layer}.mlp.experts.{args.expert}.{projection}"
            cpu_values = []
            for kind in ["weight", "weight_scale", "weight_offset"]:
                name = f"{prefix}.{kind}"
                with safe_open(args.model / index[name], framework="pt", device="cpu") as shard:
                    cpu_values.append(shard.get_tensor(name))
            cpu_weight = dequantize(*cpu_values, 128).half()
            packed_values = (
                [
                    pack_cube_tiles(value, kind)
                    for value, kind in zip(cpu_values, ["weight", "weight_scale", "weight_offset"])
                ]
                if args.tiled
                else cpu_values
            )
            device_values = [value.npu() for value in packed_values]
            canonical_device_values = [value.npu() for value in cpu_values] if args.tiled else device_values
            for tokens in args.tokens:
                x_cpu = (torch.randn(tokens, cpu_weight.shape[1], generator=generator) * 0.1).half()
                x = x_cpu.npu()
                expected = F.linear(x_cpu, cpu_weight)
                actual = op(x, *device_values, args.tiled).cpu()
                torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.003)
                eager = timed_ms(
                    lambda x=x, values=canonical_device_values: F.linear(x, dequantize(*values, 128).half()),
                    args.iterations,
                    args.repeats,
                )
                torch.npu.reset_peak_memory_stats()
                initial_bytes = torch.npu.memory_allocated()
                native = timed_ms(
                    lambda x=x, values=device_values: op(x, *values, args.tiled), args.iterations, args.repeats
                )
                scratch_peak = torch.npu.max_memory_allocated() - initial_bytes
                stream = torch.npu.Stream()
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    for _ in range(3):
                        op(x, *device_values, args.tiled)
                torch.npu.current_stream().wait_stream(stream)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    graph_output = op(x, *device_values, args.tiled)
                graph.replay()
                torch.testing.assert_close(graph_output.cpu(), expected, rtol=0.005, atol=0.003)
                replay = timed_ms(graph.replay, args.iterations, args.repeats)
                record = {
                    "label": args.label,
                    "kernel_binary": str(args.kernel_binary) if args.kernel_binary else None,
                    "kernel_sha256": binary_hash,
                    "projection": projection,
                    "layer": args.layer,
                    "expert": args.expert,
                    "tokens": tokens,
                    "tiled": args.tiled,
                    "weight_shape": list(cpu_weight.shape),
                    "max_abs_error": (actual - expected).abs().max().item(),
                    "eager_dequant": eager,
                    "native": native,
                    "graph_replay": replay,
                    "native_peak_delta_bytes": scratch_peak,
                    "iterations": args.iterations,
                    "torch": torch.__version__,
                    "torch_npu": torch_npu.__version__,
                }
                line = json.dumps(record)
                print(line, flush=True)
                output.write(line + "\n")
                output.flush()
                del graph, graph_output


if __name__ == "__main__":
    main()
