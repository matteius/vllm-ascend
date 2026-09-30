"""Summarize grouped W2/W4 task shapes and hardware counters in one rank."""

import argparse
import csv
import statistics
from collections import defaultdict


def numeric(row, field):
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    grouped = defaultdict(list)
    with open(args.csv_path, newline="") as source:
        for row in csv.DictReader(source):
            if "W2GroupedBlockedDequantMatmulV310" in row["Name"]:
                grouped[(row["Input Shapes"], row["Block Num"])].append(row)
    print("grouped_task_count", sum(map(len, grouped.values())))
    for (shape, blocks), rows in sorted(grouped.items(), key=lambda item: -len(item[1])):
        print("shape", shape[:180], "blocks", blocks, "count", len(rows))
        for field in (
            "Duration(us)",
            "aicore_time(us)",
            "vec_time(us)",
            "mac_time(us)",
            "mte1_time(us)",
            "mte2_time(us)",
            "mte3_time(us)",
            "memory_bound",
            "cube_utilization(%)",
        ):
            values = [value for row in rows if (value := numeric(row, field)) is not None]
            if values:
                print(
                    field,
                    "median",
                    round(statistics.median(values), 3),
                    "mean",
                    round(statistics.mean(values), 3),
                    "min",
                    round(min(values), 3),
                    "max",
                    round(max(values), 3),
                )


if __name__ == "__main__":
    main()
