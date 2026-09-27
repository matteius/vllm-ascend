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
        "baseline_correct": sum(row["correct"] for row in baseline),
        "native_correct": sum(row["correct"] for row in native),
        "regressions": [b["id"] for b, n in zip(baseline, native) if b["correct"] and not n["correct"]],
        "improvements": [b["id"] for b, n in zip(baseline, native) if n["correct"] and not b["correct"]],
        "changed_predictions": sum(b["prediction"] != n["prediction"] for b, n in zip(baseline, native)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("native", "affine"), default="native")
    args = parser.parse_args()
    candidate = args.candidate
    result = {
        "accuracy": accuracy("baseline-mmlu.jsonl", f"{candidate}-mmlu.jsonl"),
        "performance": {
            context: {label: performance(label, context) for label in ("baseline", candidate)}
            for context in ("short", "long")
        },
    }
    if candidate == "affine" and (ROOT / "baseline-heldout.jsonl").exists():
        result["heldout_accuracy"] = accuracy("baseline-heldout.jsonl", "affine-heldout.jsonl")
    for values in result["performance"].values():
        values["decode_speedup"] = values[candidate]["median_decode_tok_s"] / values["baseline"]["median_decode_tok_s"]
        values["e2e_speedup"] = values[candidate]["aggregate_e2e_tok_s"] / values["baseline"]["aggregate_e2e_tok_s"]
    filename = "paired-summary.json" if candidate == "native" else "affine-paired-summary.json"
    (ROOT / filename).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
