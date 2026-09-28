"""Summarize the matched three/four-stream Ascend profiler captures."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from tools.qwen4exp.summarize_trace import category, summarize_file

RUNS = {
    "20260928034312907": "parallel4",
    "20260928034332238": "parallel3",
}


def capture_identity(path: Path) -> tuple[str, int]:
    name = path.parents[1].name
    rank = int(re.search(r"rank(\d+)", name).group(1))
    timestamp = re.search(r"_(2026\d+)_ascend_pt$", name).group(1)
    return RUNS[timestamp], rank


def kernel_summary(path: Path) -> dict:
    run, rank = capture_identity(path)
    durations: dict[str, list[float]] = defaultdict(list)
    starts = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            name = row["Name"]
            duration = float(row["Duration(us)"])
            durations[category(name)].append(duration)
            if name.startswith(("hcom_allReduce", "hcom_allGather")):
                starts[name] = float(row["Start Time(us)"])
    timeline = summarize_file(path, top=20)
    return {
        "run": run,
        "rank": rank,
        "task_span_ms": timeline["task_span_ms"],
        "task_union_ms": timeline["task_union_ms"],
        "categories": {
            name: {
                "count": len(values),
                "summed_ms": sum(values) / 1000,
                "median_us": statistics.median(values),
            }
            for name, values in durations.items()
        },
        "collective_starts_us": starts,
        "top_operations": timeline["top_operations"],
        "projection_shapes": timeline["projection_shapes"],
    }


def communication_summary(path: Path) -> dict:
    operations = json.loads(path.read_text())["step"]["collective"]
    # CANN appends one aggregate record whose values duplicate all operations.
    operations = {name: value for name, value in operations.items() if not name.startswith("Total")}
    result = {"count": len(operations)}
    for source, target in (
        ("Elapse Time(ms)", "elapsed_ms"),
        ("Transit Time(ms)", "transit_ms"),
        ("Wait Time(ms)", "wait_ms"),
        ("Synchronization Time(ms)", "synchronization_ms"),
    ):
        result[target] = sum(float(value["Communication Time Info"].get(source, 0)) for value in operations.values())
    result["transit_percent"] = 100 * result["transit_ms"] / result["elapsed_ms"]
    result["wait_percent"] = 100 * result["wait_ms"] / result["elapsed_ms"]
    return result


def arrival_summary(traces: list[dict], run: str) -> dict:
    selected = {trace["rank"]: trace["collective_starts_us"] for trace in traces if trace["run"] == run}
    common = set.intersection(*(set(starts) for starts in selected.values()))
    late = Counter()
    skews = []
    for name in common:
        starts = {rank: values[name] for rank, values in selected.items()}
        late[max(starts, key=starts.get)] += 1
        skews.append(max(starts.values()) - min(starts.values()))
    ordered = sorted(skews)
    return {
        "count": len(skews),
        "latest_rank_counts": dict(sorted(late.items())),
        "summed_skew_ms": sum(skews) / 1000,
        "median_skew_us": statistics.median(skews),
        "p90_skew_us": ordered[int(0.9 * (len(ordered) - 1))],
        "p99_skew_us": ordered[int(0.99 * (len(ordered) - 1))],
        "max_skew_us": max(skews),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.trace_root.glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
    traces = [kernel_summary(path) for path in paths]
    for trace, path in zip(traces, paths, strict=True):
        trace["communication"] = communication_summary(path.with_name("communication.json"))
    report = {
        "warning": "Profiled timings include profiler overhead; summed task work is not end-to-end latency.",
        "runs": RUNS,
        "arrival_skew": {run: arrival_summary(traces, run) for run in RUNS.values()},
        "traces": traces,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
