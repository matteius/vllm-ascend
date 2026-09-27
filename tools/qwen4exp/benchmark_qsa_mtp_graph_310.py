# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare sparse and grouped QSA under graph replay, not whole-model speed.

Use the isolated 310P runtime with serving stopped and temperature supervision.
Both paths consume identical synthetic inputs and the existing paged cache.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch_npu

from tools.qwen4exp.benchmark_w4_projection_310 import timed_ms
from vllm_ascend.models.qwen4_exp.ops.qsa_batched_attention_310 import qsa_batched_prefill_310
from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import QSAGroupSelection
from vllm_ascend.models.qwen4_exp.ops.qsa_sparse_attention_310 import qsa_sparse_attention_310
from vllm_ascend.utils import enable_custom_op


def capture(function):
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
    return graph, output


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or min(args.iterations, args.repeats) < 1:
        parser.error("use a new output path and positive iteration/repeat counts")
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires Ascend 310P")
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    generator = torch.Generator().manual_seed(1024)
    block_size, head_dim, cache_blocks, num_groups = 128, 256, 192, 512
    with args.output.open("x") as output:
        for query_heads, kv_heads in ((6, 1), (12, 1), (24, 2)):
            shape = (cache_blocks, kv_heads * head_dim // 16, block_size, 16)
            caches = [
                torch_npu.npu_format_cast((torch.randn(shape, generator=generator) * 0.1).half().npu(), 29)
                for _ in range(2)
            ]
            block_table = torch.randperm(cache_blocks, generator=generator).to(torch.int32).unsqueeze(0).npu()
            for tokens in (1, 2, 3, 5, 8):
                query = (torch.randn(tokens, query_heads, head_dim, generator=generator) * 0.5).half().npu()
                selection = QSAGroupSelection(
                    group_indices=torch.stack(
                        [torch.randperm(5760, generator=generator)[:num_groups] for _ in range(tokens)]
                    ).to(dtype=torch.int32, device="npu:0"),
                    group_counts=torch.full((tokens,), num_groups, dtype=torch.int32, device="npu:0"),
                    tail_starts=torch.full((tokens,), 23040, dtype=torch.int32, device="npu:0"),
                    tail_counts=torch.arange(tokens, dtype=torch.int32, device="npu:0") % 4,
                )
                query_start_loc = torch.tensor([0, tokens], dtype=torch.int32, device="npu:0")
                group_list = torch.arange(1, 8 * kv_heads + 1, dtype=torch.int64, device="npu:0") * (
                    query_heads // kv_heads
                )
                inputs = (query, *caches, selection, block_table, query_start_loc)
                native_graph, native_output = capture(partial(qsa_sparse_attention_310, *inputs))
                grouped_graph, grouped_output = capture(
                    partial(qsa_batched_prefill_310, *inputs, scale=head_dim**-0.5, decode_group_list=group_list)
                )
                native_graph.replay()
                grouped_graph.replay()
                native_cpu, grouped_cpu = native_output.cpu(), grouped_output.cpu()
                torch.testing.assert_close(grouped_cpu, native_cpu, rtol=1e-2, atol=2e-3)
                record = {
                    "scope": "synthetic_qsa_only_not_whole_model",
                    "tokens": tokens,
                    "query_heads": query_heads,
                    "kv_heads": kv_heads,
                    "groups": num_groups,
                    "cache_tokens": cache_blocks * block_size,
                    "max_abs_difference": (grouped_cpu.float() - native_cpu.float()).abs().max().item(),
                    "sparse_graph": timed_ms(native_graph.replay, args.iterations, args.repeats),
                    "grouped_graph": timed_ms(grouped_graph.replay, args.iterations, args.repeats),
                }
                line = json.dumps(record)
                print(line, flush=True)
                output.write(line + "\n")
                output.flush()


if __name__ == "__main__":
    main()
