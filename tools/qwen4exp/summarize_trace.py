# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize exported Ascend kernel CSVs without equating summed time to latency.

Each worker trace is reported separately. Task durations can overlap across
streams, so their sum and per-operation percentages are work-attribution
diagnostics, not critical-path latency or an end-to-end speed prediction.
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

PIPELINE_TIME_COLUMNS = (
    "aicore_time(us)",
    "vec_time(us)",
    "mac_time(us)",
    "scalar_time(us)",
    "mte1_time(us)",
    "mte2_time(us)",
    "mte3_time(us)",
)


def optional_metric(value: str | None) -> float | None:
    """Missing profiler counters are not measured zeroes."""
    try:
        parsed = Decimal(value.strip()) if value is not None else Decimal("NaN")
    except InvalidOperation:
        return None
    return float(parsed) if parsed.is_finite() and parsed >= 0 else None


def projection_shape_summary(rows: list[dict]) -> dict:
    result = {
        "name": rows[0]["Name"],
        "input_shapes": rows[0].get("Input Shapes", ""),
        "count": len(rows),
        "median_task_us": statistics.median(to_ns(row["Duration(us)"]) / 1000 for row in rows),
        "block_counts": sorted({int(row["Block Num"]) for row in rows if row.get("Block Num", "").isdigit()}),
        "pipeline_counters": {},
    }
    for column in PIPELINE_TIME_COLUMNS:
        values = [value for row in rows if (value := optional_metric(row.get(column))) is not None]
        if values:
            result["pipeline_counters"][column] = {"samples": len(values), "median_us": statistics.median(values)}
    return result


def to_ns(microseconds: str) -> int:
    """Avoid loss of sub-microsecond precision in epoch-sized timestamps."""
    return int(Decimal(microseconds.strip()) * 1000)


def union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    start, stop = ordered[0]
    total = 0
    for left, right in ordered[1:]:
        if left > stop:
            total += stop - start
            start, stop = left, right
        else:
            stop = max(stop, right)
    return total + stop - start


def category(name: str) -> str:
    compact = name.lower().replace("_", "")
    if "qwenw4routedmatmul" in compact:
        return "w4_routed_projection"
    if "qwenw4groupmatmul" in compact:
        return "w4_group_projection"
    if any(part in compact for part in ("hccl", "allreduce", "allgather", "reducescatter")):
        return "collective"
    if any(part in compact for part in ("qsa", "gdn", "kda", "gateddelta", "causalconv")):
        return "named_attention_or_recurrence"
    if any(part in compact for part in ("cast", "transdata", "transpose", "memcpy")):
        return "cast_layout_or_copy"
    if "matmul" in compact or "gemm" in compact:
        return "other_matrix_multiply"
    return "other"


def summarize_file(path: Path, top: int = 30) -> dict:
    intervals = []
    names = defaultdict(list)
    buckets = defaultdict(list)
    projection_shapes = defaultdict(list)
    devices = set()
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            start, duration = to_ns(row["Start Time(us)"]), to_ns(row["Duration(us)"])
            if duration < 0:
                raise ValueError(f"negative task duration in {path}")
            name = row["Name"]
            intervals.append((start, start + duration))
            names[name].append(duration)
            buckets[category(name)].append(duration)
            if category(name) in {"w4_routed_projection", "w4_group_projection"}:
                projection_shapes[(name, row.get("Input Shapes", ""))].append(row)
            devices.add(row["Device_id"])
    if len(devices) > 1:
        raise ValueError(f"multiple devices in {path}; cannot union unrelated device timelines")
    total = sum(sum(values) for values in names.values())
    span = max(right for _, right in intervals) - min(left for left, _ in intervals) if intervals else 0

    def entry(name, values):
        subtotal = sum(values)
        return {
            "name": name,
            "count": len(values),
            "summed_task_ms": subtotal / 1e6,
            "mean_task_us": statistics.mean(values) / 1000,
            "median_task_us": statistics.median(values) / 1000,
            "max_task_us": max(values) / 1000,
            "percent_summed_task_time": 100 * subtotal / total if total else 0,
        }

    return {
        "file": str(path),
        "device_ids": sorted(devices),
        "task_count": len(intervals),
        "task_span_ms": span / 1e6,
        "task_union_ms": union_ns(intervals) / 1e6,
        "summed_task_ms": total / 1e6,
        "projection_shapes": [projection_shape_summary(rows) for _, rows in sorted(projection_shapes.items())],
        "categories": [entry(name, values) for name, values in sorted(buckets.items())],
        "top_operations": [
            entry(name, values)
            for name, values in sorted(names.items(), key=lambda item: sum(item[1]), reverse=True)[:top]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    paths = sorted(args.trace_root.rglob("kernel_details.csv"))
    if not paths:
        parser.error("no exported kernel_details.csv found")
    report = {
        "metric_warning": (
            "Summed task times and reported hardware pipeline counters may overlap; "
            "neither is an additive critical-path latency or unprofiled throughput measurement."
        ),
        "traces": [summarize_file(path, args.top) for path in paths],
    }
    result = json.dumps(report, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as output:
            output.write(result)
    print(result, end="")


if __name__ == "__main__":
    main()
