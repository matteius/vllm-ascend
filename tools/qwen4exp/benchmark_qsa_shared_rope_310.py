# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate Q/K/index-query RoPE sharing under 310P graph replay.

This synthetic diagnostic excludes projections, normalization and attention;
it does not establish whole-model throughput. Run with other NPU jobs stopped.
"""

import argparse
import json
from pathlib import Path

import torch
import torch_npu

from tools.qwen4exp.benchmark_w4_projection_310 import timed_ms
from vllm_ascend.models.qwen4_exp.qsa import apply_partial_rope, partial_rope_cos_sin


def capture_rotations(values, positions, shared):
    def rotate():
        tables = partial_rope_cos_sin(positions, rotary_dim=64, base=10000.0, dtype=torch.float32) if shared else None
        return tuple(
            apply_partial_rope(value, positions, 64, 10000.0, torch.float32, cos_sin=tables) for value in values
        )

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            rotate()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        outputs = rotate()
    return graph, outputs


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or min(args.iterations, args.repeats) < 1:
        parser.error("use a new output path and positive iteration/repeat counts")
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires Ascend 310P")
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    generator = torch.Generator().manual_seed(1024)
    with args.output.open("x") as output:
        for tokens in (1, 2, 3, 5, 8):
            values = [
                torch.randn(tokens, heads, width, generator=generator).half().npu()
                for heads, width in ((6, 256), (1, 256), (4, 128))
            ]
            positions = torch.arange(23400, 23400 + tokens, dtype=torch.int64).npu()
            separate_graph, separate_outputs = capture_rotations(values, positions, False)
            shared_graph, shared_outputs = capture_rotations(values, positions, True)
            separate_graph.replay()
            shared_graph.replay()
            for separate, shared in zip(separate_outputs, shared_outputs):
                torch.testing.assert_close(shared.cpu(), separate.cpu(), rtol=0, atol=0)
            separate_ms = timed_ms(separate_graph.replay, args.iterations, args.repeats)
            shared_ms = timed_ms(shared_graph.replay, args.iterations, args.repeats)
            record = {
                "scope": "synthetic_query_rope_only_not_whole_model",
                "tokens": tokens,
                "query_shapes": [list(value.shape) for value in values],
                "position_start": 23400,
                "bitwise_equal": True,
                "separate_graph": separate_ms,
                "shared_graph": shared_ms,
            }
            line = json.dumps(record)
            print(line, flush=True)
            output.write(line + "\n")
            output.flush()


if __name__ == "__main__":
    main()
