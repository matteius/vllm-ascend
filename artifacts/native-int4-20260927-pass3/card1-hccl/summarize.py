"""Summarize the isolated physical-card-1 HCCL/runtime sweep evidence."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent


def group(pattern: str) -> dict:
    files = sorted(ROOT.glob(pattern))
    rows = [json.loads(p.read_text()) for p in files]
    aggregate = [r["aggregate_decode_tok_s"] for r in rows]
    stream = [r["median_stream_decode_tok_s"] for r in rows]
    ttft = [s["ttft_s"] for r in rows for s in r["streams"]]
    decode = [s["decode_s"] for r in rows for s in r["streams"]]
    return {
        "files": [p.name for p in files],
        "runs": len(rows),
        "aggregate_decode_tok_s": aggregate,
        "median_aggregate_decode_tok_s": statistics.median(aggregate),
        "median_stream_decode_tok_s": statistics.median(stream),
        "max_ttft_s": max(ttft),
        "max_stream_decode_s": max(decode),
    }


summary = {
    "hardware": {"physical_card": "16409", "visible_devices": "2,3", "tensor_parallel_size": 2},
    "common": {
        "native_backend": "cube_310_int4_a8",
        "max_num_seqs": 4,
        "mtp_tokens": 2,
        "gpu_memory_utilization": 0.965,
        "kv_fraction": 0.80,
        "max_num_batched_tokens": 512,
    },
    "baseline": {
        "max_model_len": 65536,
        "task_queue_enable": 1,
        "hccl_op_expansion_mode": None,
        "c1": group("card1-hccl-baseline65k-tq1-c1-r*.json"),
        "c3": group("card1-hccl-baseline65k-tq1-c3-r*.json"),
        "c4": group("card1-hccl-baseline65k-tq1-c4-r*.json"),
    },
    "strict_host_isolation": {
        "affinity": {
            "api": [22, 23],
            "engine": [30, 31],
            "worker0": list(range(16, 22)),
            "worker1": list(range(24, 30)),
        },
        "c3": group("card1-hccl-baseline65k-tq1-isolated-c3-r*.json"),
        "c4": group("card1-hccl-baseline65k-tq1-isolated-c4-r*.json"),
    },
    "capacity": {
        "task_queue_1_80k": {
            "available_kv_gib": 5.73,
            "required_kv_gib": 6.46,
            "estimated_max_model_len": 70656,
            "result": "startup capacity failure",
        },
        "task_queue_2_65k": {
            "available_kv_gib": 5.03,
            "required_kv_gib": 5.31,
            "estimated_max_model_len": 61824,
            "result": "startup capacity failure",
        },
        "task_queue_2_60k_post_reboot": {
            "limiting_available_kv_gib": 4.67,
            "required_kv_gib": 4.88,
            "estimated_max_model_len": 57344,
            "result": "startup capacity failure after full load",
        },
        "task_queue_2_56k": {"result": "canceled by user at 3/1610 shards before throughput validation"},
        "task_queue_2_incremental_memory_gib_per_npu_pre_reboot_approx": 0.70,
    },
    "normalization": {
        "card1_baseline_c1_decode_tok_s": 19.447343556447983,
        "card0_route128_c1_decode_tok_s_reported_by_parallel_experiment": 20.142,
    },
    "status": "TQ2 throughput deferred for a later two-card study. No TQ2 throughput result was collected.",
}
for c in ("c3", "c4"):
    b = summary["baseline"][c]["median_aggregate_decode_tok_s"]
    i = summary["strict_host_isolation"][c]["median_aggregate_decode_tok_s"]
    summary["strict_host_isolation"][c]["delta_vs_baseline_percent"] = (i / b - 1) * 100
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
