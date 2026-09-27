# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time QSA selection between breakable MTP graph segments, not model tok/s.

Use an idle Ascend 310P with external temperature supervision. The native
score reference checks group sets; nearly tied FP32 reductions can reorder
members within a set. This tool does not change the W8/default dispatch.
"""

import argparse
import json
from pathlib import Path

import torch
import torch_npu

from tools.qwen4exp.benchmark_w4_projection_310 import timed_ms
from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import _stable_topk_indices, qsa_indexer_select_groups_310
from vllm_ascend.utils import enable_custom_op


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output path to preserve earlier results")
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    generator = torch.Generator().manual_seed(1024)
    with args.output.open("x") as output:
        for capacity in (2048, 5856, 8192):
            pages = capacity // 16
            for tokens in (1, 2, 3, 5, 8):
                query = torch.randn((tokens, 4, 128), generator=generator).half().npu()
                cache = torch.randn((pages + 7, 19, 128), generator=generator).half().npu()
                table = torch.randperm(pages + 7, generator=generator)[:pages].int().unsqueeze(0).npu()
                boundaries = torch.tensor([0, tokens], dtype=torch.int32, device="npu:0")
                positions = torch.full((tokens,), capacity * 4 - 1, dtype=torch.int32, device="npu:0")

                def select(limit, query=query, cache=cache, table=table, boundaries=boundaries, positions=positions):
                    return qsa_indexer_select_groups_310(
                        query,
                        cache,
                        table,
                        boundaries,
                        positions,
                        compress_ratio=4,
                        token_topk=2048,
                        max_matmul_decode_tokens=limit,
                    )

                native_scores = torch.ops._C_ascend.npu_qsa_indexer_score_310(
                    query, cache, table, boundaries, positions, 4
                )
                native_set = _stable_topk_indices(native_scores, 512).sort(dim=1).values.cpu()
                for limit in (2, 8):
                    torch.testing.assert_close(
                        select(limit).group_indices.sort(dim=1).values.cpu(), native_set, rtol=0, atol=0
                    )
                baseline = timed_ms(lambda: select(2), args.iterations, args.repeats)
                candidate = timed_ms(lambda: select(8), args.iterations, args.repeats)
                record = {
                    "query_tokens": tokens,
                    "index_heads": 4,
                    "capacity_groups": capacity,
                    "default_selection": baseline,
                    "w4_mtp_selection": candidate,
                    "same_selected_group_set": True,
                    "execution": "eager_between_breakable_graph_segments",
                    "iterations": args.iterations,
                    "repeats": args.repeats,
                    "not_model_throughput": True,
                }
                line = json.dumps(record)
                print(line, flush=True)
                output.write(line + "\n")
                output.flush()


if __name__ == "__main__":
    main()
