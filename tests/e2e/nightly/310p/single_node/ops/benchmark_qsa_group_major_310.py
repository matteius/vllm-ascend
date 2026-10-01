# SPDX-License-Identifier: Apache-2.0
"""Compare per-query and group-major QSA prefill schedules on Ascend 310P.

This benchmark sweeps controlled overlap between adjacent query selections.
It reports the actual union-load reduction separately from device time so the
real-model overlap threshold for enabling the experimental backend can be
chosen from evidence rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.ops.qsa_batched_attention_310 import qsa_batched_prefill_310
from vllm_ascend.models.qwen4_exp.ops.qsa_group_major_attention_310 import (
    build_qsa_group_major_plan,
    qsa_group_major_prefill_310,
)
from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import QSAGroupSelection
from vllm_ascend.utils import enable_custom_op

_BLOCK_SIZE = 64
_COMPRESS_RATIO = 4
_HEAD_DIM = 256
_NUM_QUERY_HEADS = 24
_NUM_KV_HEADS = 2


def _selection_with_overlap(
    num_queries: int,
    selected_groups: int,
    visible_groups: int,
    overlap_fraction: float,
) -> QSAGroupSelection:
    shared_count = round(selected_groups * overlap_fraction)
    generator = torch.Generator().manual_seed(967 + shared_count)
    shuffled = torch.randperm(visible_groups, generator=generator, dtype=torch.int32)
    shared = shuffled[:shared_count]
    private_pool = shuffled[shared_count:]
    rows = []
    for _ in range(num_queries):
        private_order = torch.randperm(private_pool.numel(), generator=generator)
        private = private_pool[private_order[: selected_groups - shared_count]]
        rows.append(torch.cat((shared, private)))
    groups = torch.stack(rows)
    return QSAGroupSelection(
        group_indices=groups,
        group_counts=torch.full((num_queries,), selected_groups, dtype=torch.int32),
        tail_starts=torch.zeros(num_queries, dtype=torch.int32),
        tail_counts=torch.zeros(num_queries, dtype=torch.int32),
    )


def _to_device(selection: QSAGroupSelection, device: str) -> QSAGroupSelection:
    return QSAGroupSelection(
        selection.group_indices.to(device),
        selection.group_counts.to(device),
        selection.tail_starts.to(device),
        selection.tail_counts.to(device),
    )


def _measure(function, repeats: int) -> tuple[torch.Tensor, list[float]]:
    function()
    torch_npu.npu.synchronize()
    times = []
    output = None
    for _ in range(repeats):
        started = time.perf_counter()
        output = function()
        torch_npu.npu.synchronize()
        times.append((time.perf_counter() - started) * 1000)
    assert output is not None
    return output, times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--selected-groups", type=int, default=512)
    parser.add_argument("--visible-groups", type=int, default=10_000)
    parser.add_argument("--query-tiles", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--overlap", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires an Ascend 310P NPU")
    if min(args.queries, args.selected_groups, args.visible_groups, args.repeats, *args.query_tiles) <= 0:
        parser.error("query, group, repeat, and tile counts must be positive")
    if args.selected_groups > args.visible_groups:
        parser.error("selected groups cannot exceed visible groups")
    if any(fraction < 0.0 or fraction > 1.0 for fraction in args.overlap):
        parser.error("overlap fractions must be in [0, 1]")

    enable_custom_op()
    device = "npu:0"
    torch.manual_seed(971)
    visible_tokens = args.visible_groups * _COMPRESS_RATIO
    num_blocks = (visible_tokens + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    cache_shape = (num_blocks, _NUM_KV_HEADS * _HEAD_DIM // 16, _BLOCK_SIZE, 16)
    query = torch.randn((args.queries, _NUM_QUERY_HEADS, _HEAD_DIM), dtype=torch.float16, device=device) * 0.1
    key_cache = torch_npu.npu_format_cast(torch.randn(cache_shape, dtype=torch.float16, device=device) * 0.1, 29)
    value_cache = torch_npu.npu_format_cast(torch.randn(cache_shape, dtype=torch.float16, device=device) * 0.1, 29)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    query_start_loc = torch.tensor([0, args.queries], dtype=torch.int32, device=device)
    scale = _HEAD_DIM**-0.5

    results = []
    for overlap_fraction in args.overlap:
        selection_cpu = _selection_with_overlap(
            args.queries,
            args.selected_groups,
            args.visible_groups,
            overlap_fraction,
        )
        selection = _to_device(selection_cpu, device)

        def retained(selection: QSAGroupSelection = selection) -> torch.Tensor:
            return qsa_batched_prefill_310(
                query,
                key_cache,
                value_cache,
                selection,
                block_table,
                query_start_loc,
                scale=scale,
            )

        retained_output, retained_times = _measure(retained, args.repeats)
        for query_tile in args.query_tiles:
            source_reads = 0
            union_reads = 0
            for start in range(0, args.queries, query_tile):
                stop = min(start + query_tile, args.queries)
                tile_selection = QSAGroupSelection(
                    selection_cpu.group_indices[start:stop],
                    selection_cpu.group_counts[start:stop],
                    selection_cpu.tail_starts[start:stop],
                    selection_cpu.tail_counts[start:stop],
                )
                plan = build_qsa_group_major_plan(tile_selection)
                source_reads += int(tile_selection.group_counts.sum())
                source_reads += int((tile_selection.tail_counts > 0).sum())
                union_reads += plan.unique_group_reads

            def group_major(
                selection: QSAGroupSelection = selection,
                query_tile: int = query_tile,
            ) -> torch.Tensor:
                return qsa_group_major_prefill_310(
                    query,
                    key_cache,
                    value_cache,
                    selection,
                    block_table,
                    query_start_loc,
                    scale=scale,
                    query_tile=query_tile,
                )

            group_output, group_times = _measure(group_major, args.repeats)
            max_abs_error = float((retained_output.float() - group_output.float()).abs().max().cpu())
            retained_median = statistics.median(retained_times)
            group_median = statistics.median(group_times)
            results.append(
                {
                    "overlap_fraction": overlap_fraction,
                    "query_tile": query_tile,
                    "source_group_reads": source_reads,
                    "union_group_reads": union_reads,
                    "load_reduction": source_reads / union_reads,
                    "retained_median_ms": retained_median,
                    "group_major_median_ms": group_median,
                    "speedup": retained_median / group_median,
                    "max_abs_error": max_abs_error,
                }
            )
    print(json.dumps({"configuration": vars(args), "results": results}, indent=2))


if __name__ == "__main__":
    main()
