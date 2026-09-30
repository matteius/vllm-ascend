"""Compare GLM single- and four-stream NPU rank traces."""

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from summarize_trace import summarize_file

RUNS = {
    "20260930002643": "single",
    "20260930002821": "parallel4",
}


def identity(path):
    capture = path.parents[1].name
    rank = int(re.search(r"rank(\d+)", capture).group(1))
    stamp = re.search(r"_(2026\d+)_ascend_pt$", capture).group(1)
    return next(name for prefix, name in RUNS.items() if stamp.startswith(prefix)), rank


def communication(path):
    operations = json.loads(path.read_text())["step"]["collective"]
    operations = {name: value for name, value in operations.items() if not name.startswith("Total")}
    result = {"count": len(operations)}
    for field, key in (
        ("Elapse Time(ms)", "elapsed_ms"),
        ("Transit Time(ms)", "transit_ms"),
        ("Wait Time(ms)", "wait_ms"),
        ("Synchronization Time(ms)", "synchronization_ms"),
    ):
        result[key] = sum(float(item["Communication Time Info"].get(field, 0)) for item in operations.values())
    return result


def trace(path):
    run, rank = identity(path)
    grouped = defaultdict(list)
    collective_starts = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            name = row["Name"]
            if name == "W2GroupedBlockedDequantMatmulV310":
                shape = row["Input Shapes"].strip('"').split(";")[0]
                grouped[shape].append(float(row["Duration(us)"]))
            elif name.startswith(("hcom_allReduce", "hcom_allGather", "hcom_reduceScatter")):
                collective_starts[name] = float(row["Start Time(us)"])
    timeline = summarize_file(path, top=12)
    return {
        "run": run,
        "rank": rank,
        "task_span_ms": timeline["task_span_ms"],
        "task_union_ms": timeline["task_union_ms"],
        "categories": {
            entry["name"]: {"count": entry["count"], "summed_task_ms": entry["summed_task_ms"]}
            for entry in timeline["categories"]
        },
        "grouped_by_activation_shape": {
            shape: {
                "count": len(values),
                "summed_task_ms": sum(values) / 1000,
                "median_task_us": statistics.median(values),
            }
            for shape, values in sorted(grouped.items())
        },
        "communication": communication(path.with_name("communication.json")),
        "collective_starts": collective_starts,
    }


def arrival(traces, run):
    selected = {item["rank"]: item["collective_starts"] for item in traces if item["run"] == run}
    if len(selected) < 2:
        return None
    common = set.intersection(*(set(starts) for starts in selected.values()))
    latest = Counter()
    skews = []
    for name in common:
        starts = {rank: values[name] for rank, values in selected.items()}
        latest[max(starts, key=starts.get)] += 1
        skews.append(max(starts.values()) - min(starts.values()))
    if not skews:
        return {"common_collectives": 0}
    skews.sort()
    return {
        "ranks": sorted(selected),
        "common_collectives": len(skews),
        "latest_rank_counts": dict(sorted(latest.items())),
        "summed_skew_ms": sum(skews) / 1000,
        "median_skew_us": statistics.median(skews),
        "p90_skew_us": skews[int(0.9 * (len(skews) - 1))],
        "max_skew_us": skews[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.trace_root.glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
    traces = [trace(path) for path in paths]
    report = {
        "warning": "Profiled task sums are work attribution, not end to end latency.",
        "arrival_skew": {run: arrival(traces, run) for run in RUNS.values()},
        "traces": traces,
    }
    for item in traces:
        del item["collective_starts"]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
