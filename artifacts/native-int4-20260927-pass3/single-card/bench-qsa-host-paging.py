"""Measure one QSA layer's device selection -> host KV -> device scratch path.

This is a transfer feasibility probe, not a working paged attention backend.
It preserves FP16 K/V and models 512 selected four-token groups out of 262K.
Run on an otherwise idle 310P3 NPU with ASCEND_RT_VISIBLE_DEVICES set.
"""

import argparse
import json
import statistics
import time

import torch
import torch_npu  # noqa: F401  # registers the NPU device

CONTEXT_TOKENS = 262_144
GROUP_WIDTH = 4
SELECTED_GROUPS = 512
KV_HEAD_DIM = 256
QSA_LAYERS = 12
WARMUP = 5


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")

    torch.set_num_threads(1)
    torch.npu.set_device(0)
    torch.manual_seed(20260928)
    rows_per_layer = SELECTED_GROUPS * GROUP_WIDTH
    group_count = CONTEXT_TOKENS // GROUP_WIDTH
    device_groups = torch.randperm(group_count, dtype=torch.int32, device="cpu")[:SELECTED_GROUPS].to("npu")
    host_groups = torch.empty(SELECTED_GROUPS, dtype=torch.int32, pin_memory=True)
    group_offsets = torch.arange(GROUP_WIDTH, dtype=torch.int64)

    # The full history may live in ordinary DRAM. Only the compact transfer
    # staging buffers must be pinned for asynchronous H2D copies.
    host_keys = torch.empty((CONTEXT_TOKENS, KV_HEAD_DIM), dtype=torch.float16).zero_()
    host_values = torch.empty_like(host_keys).zero_()
    staging_keys = torch.empty((rows_per_layer, KV_HEAD_DIM), dtype=torch.float16, pin_memory=True)
    staging_values = torch.empty_like(staging_keys, pin_memory=True)
    resident_keys = torch.empty_like(staging_keys, device="npu")
    resident_values = torch.empty_like(staging_values, device="npu")

    timings: dict[str, list[float]] = {"selection_ms": [], "gather_ms": [], "h2d_ms": [], "total_ms": []}
    for step in range(WARMUP + args.repeats):
        started = time.perf_counter()
        host_groups.copy_(device_groups, non_blocking=True)
        torch.npu.synchronize()
        after_selection = time.perf_counter()

        selected_rows = (host_groups.to(torch.int64)[:, None] * GROUP_WIDTH + group_offsets).reshape(-1)
        torch.index_select(host_keys, 0, selected_rows, out=staging_keys)
        torch.index_select(host_values, 0, selected_rows, out=staging_values)
        after_gather = time.perf_counter()

        resident_keys.copy_(staging_keys, non_blocking=True)
        resident_values.copy_(staging_values, non_blocking=True)
        torch.npu.synchronize()
        finished = time.perf_counter()
        if step >= WARMUP:
            timings["selection_ms"].append((after_selection - started) * 1000)
            timings["gather_ms"].append((after_gather - after_selection) * 1000)
            timings["h2d_ms"].append((finished - after_gather) * 1000)
            timings["total_ms"].append((finished - started) * 1000)

    print(
        json.dumps(
            {
                "context_tokens": CONTEXT_TOKENS,
                "selected_tokens": rows_per_layer,
                "qsa_layers": QSA_LAYERS,
                "bytes_h2d_per_layer": 2 * staging_keys.numel() * staging_keys.element_size(),
                "repeats": args.repeats,
                "median_ms": {name: statistics.median(values) for name, values in timings.items()},
                "p90_ms": {name: percentile(values, 0.90) for name, values in timings.items()},
                "estimated_12_layer_serial_ms": QSA_LAYERS * statistics.median(timings["total_ms"]),
            }
        )
    )


if __name__ == "__main__":
    main()
