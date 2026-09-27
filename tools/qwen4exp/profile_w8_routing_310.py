# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B one real W8 expert bank; synthetic activations, no TP collective.

This measures routed MoE only, not the shared expert or full-model throughput.
The baseline is an explicitly supplied saved moe.py, never the live service.
Run on idle NPUs with external temperature supervision.
"""

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu
from safetensors import safe_open

from tools.qwen4exp.benchmark_w4_projection_310 import timed_ms
from vllm_ascend.models.qwen4_exp.moe import _w8a8_packed_grouped_experts_npu, route_topk
from vllm_ascend.utils import maybe_trans_nz


def load_bank(model, layer, rank, tp_size):
    config = json.loads((model / "config.json").read_text())["text_config"]
    experts = config["num_experts"]
    if experts % tp_size:
        raise ValueError("this diagnostic requires evenly divided expert shards")
    start, stop = rank * experts // tp_size, (rank + 1) * experts // tp_size
    index = json.loads((model / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer}.mlp."
    names = [prefix + "gate.weight"]
    for expert in range(start, stop):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for field in ("weight", "weight_scale", "weight_offset"):
                names.append(f"{prefix}experts.{expert}.{projection}.{field}")
    shards = defaultdict(list)
    for name in names:
        shards[index[name]].append(name)
    values = {}
    for filename, names in shards.items():
        with safe_open(model / filename, framework="pt", device="cpu") as shard:
            for name in names:
                value = shard.get_tensor(name)
                if name.endswith("weight_offset") and torch.count_nonzero(value):
                    raise ValueError("packed W8 kernel requires symmetric expert weights")
                values[name.removeprefix(prefix)] = value
    banks = []
    for projections in (("gate_proj", "up_proj"), ("down_proj",)):
        for field in ("weight", "weight_scale"):
            bank = torch.stack(
                [
                    torch.cat([values[f"experts.{expert}.{proj}.{field}"] for proj in projections], dim=0)
                    for expert in range(start, stop)
                ]
            )
            if field == "weight":
                if bank.dtype != torch.int8:
                    raise ValueError("expected INT8 checkpoint expert bank")
                bank = maybe_trans_nz(bank.npu())
            else:
                bank = bank.reshape(stop - start, -1).float().npu()
            banks.append(bank)
    return config, values["gate.weight"].half().npu(), start, banks


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 3, 6, 8])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or (args.trace_dir and args.trace_dir.exists()):
        parser.error("choose fresh output/trace paths")
    if min(args.iterations, args.repeats) < 1 or args.layer < 0:
        parser.error("iterations/repeats must be positive and layer nonnegative")
    if args.tp_size < 1 or not 0 <= args.rank < args.tp_size or any(t < 1 or t > 8 for t in args.tokens):
        parser.error("invalid shard or decode token count (1..8)")
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires Ascend 310P")
    spec = importlib.util.spec_from_file_location("vllm_ascend.models.qwen4_exp._saved_moe", args.baseline_source)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    config, gate, expert_offset, banks = load_bank(args.model, args.layer, args.rank, args.tp_size)
    generator = torch.Generator().manual_seed(1024)
    with args.output.open("x") as output:
        for tokens in args.tokens:
            x = (torch.randn(tokens, config["hidden_size"], generator=generator) * 0.1).half().npu()
            weights, ids = route_topk(F.linear(x, gate), config["num_experts_per_tok"])
            values = (x, weights, ids, *banks, expert_offset)
            expected = baseline._w8a8_packed_grouped_experts_npu(*values).cpu()
            actual = _w8a8_packed_grouped_experts_npu(*values).cpu()
            # Reordering movement/clearing must not change real-layer arithmetic.
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            for label, function in (
                ("baseline", baseline._w8a8_packed_grouped_experts_npu),
                ("candidate", _w8a8_packed_grouped_experts_npu),
            ):
                stream = torch.npu.Stream()
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    for _ in range(3):
                        function(*values)
                torch.npu.current_stream().wait_stream(stream)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    captured = function(*values)
                graph.replay()
                replay_result = captured.cpu()
                torch.testing.assert_close(replay_result, expected, atol=0, rtol=0)
                record = {
                    "label": label,
                    "tokens": tokens,
                    "weights": "real_checkpoint",
                    "scope": "routed_expert_bank_only_no_collective",
                    "layer": args.layer,
                    "rank": args.rank,
                    "tp_size": args.tp_size,
                    "input_sha256": hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
                    "output_sha256": hashlib.sha256(replay_result.numpy().tobytes()).hexdigest(),
                    "graph": timed_ms(graph.replay, args.iterations, args.repeats),
                }
                if args.trace_dir:
                    with torch_npu.profiler.profile(
                        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                        record_shapes=True,
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                            str(args.trace_dir / f"t{tokens}-{label}")
                        ),
                    ):
                        graph.replay()
                        torch.npu.synchronize()
                output.write(json.dumps(record) + "\n")
                output.flush()
                print(json.dumps(record), flush=True)
                del graph, captured


if __name__ == "__main__":
    main()
