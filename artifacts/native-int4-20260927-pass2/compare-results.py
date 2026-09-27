"""Summarize matched model runs; do not infer promotion from a sampled score."""

import argparse
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent


def rows(filename):
    return [json.loads(line) for line in (ROOT / filename).read_text().splitlines()]


def performance(label, context):
    measured = [row for row in rows(f"{label}-{context}.jsonl") if not row["warmup"]]
    assert len(measured) == 3 and all(row["usage"]["completion_tokens"] == 512 for row in measured)
    drafted = sum(row["drafted_delta"] for row in measured)
    accepted = sum(row["accepted_delta"] for row in measured)
    return {
        "requests": len(measured),
        "completion_tokens": sum(row["usage"]["completion_tokens"] for row in measured),
        "median_decode_tok_s": median(row["client_decode_tok_s"] for row in measured),
        "median_e2e_seconds": median(row["e2e_s"] for row in measured),
        "aggregate_e2e_tok_s": sum(row["usage"]["completion_tokens"] for row in measured)
        / sum(row["e2e_s"] for row in measured),
        "median_ttft_seconds": median(row["ttft_s"] for row in measured),
        "drafted_tokens": drafted,
        "accepted_tokens": accepted,
        "acceptance": accepted / drafted if drafted else None,
        "scope": "serial batch-one; long-prefix cache warmed by excluded 32-token request",
    }


def accuracy(baseline_file, candidate_file):
    baseline, native = rows(baseline_file), rows(candidate_file)
    assert len(baseline) == len(native) == 228
    assert all(
        (b["id"], b["prompt_sha256"], b["answer"]) == (n["id"], n["prompt_sha256"], n["answer"])
        for b, n in zip(baseline, native)
    )
    return {
        "scope": "fixed stratified zero-shot MMLU sample, not the official full-dataset score",
        "samples": len(baseline),
        "baseline_invalid": sum(row["prediction"] is None or row["finish_reason"] != "stop" for row in baseline),
        "native_invalid": sum(row["prediction"] is None or row["finish_reason"] != "stop" for row in native),
        "baseline_correct": sum(row["correct"] for row in baseline),
        "native_correct": sum(row["correct"] for row in native),
        "regressions": [b["id"] for b, n in zip(baseline, native) if b["correct"] and not n["correct"]],
        "improvements": [b["id"] for b, n in zip(baseline, native) if n["correct"] and not b["correct"]],
        "changed_predictions": sum(b["prediction"] != n["prediction"] for b, n in zip(baseline, native)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("balanced", "strided", "vector"), default="vector")
    args = parser.parse_args()
    candidate = args.candidate
    previous = "../native-int4-20260927"
    result = {
        "accuracy_vs_w4a16": accuracy(f"{previous}/baseline-mmlu.jsonl", f"{candidate}-mmlu.jsonl"),
        "accuracy_vs_previous_native": accuracy(f"{previous}/native-mmlu.jsonl", f"{candidate}-mmlu.jsonl"),
        "performance": {
            context: {
                "w4a16": performance(f"{previous}/baseline", context),
                "previous_native": performance(f"{previous}/native", context),
                "candidate": performance(candidate, context),
            }
            for context in ("short", "long")
        },
    }
    for values in result["performance"].values():
        values["decode_speedup_vs_previous_native"] = (
            values["candidate"]["median_decode_tok_s"] / values["previous_native"]["median_decode_tok_s"]
        )
        values["decode_speedup_vs_w4a16"] = (
            values["candidate"]["median_decode_tok_s"] / values["w4a16"]["median_decode_tok_s"]
        )
    (ROOT / "paired-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
