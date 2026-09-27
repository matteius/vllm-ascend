# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired real-model multiple-choice gate via an already running local server.

Input JSONL: id, subject, prompt, answer. Output excludes dataset question text
and records its hash. This is a fixed sampled accuracy diagnostic, not a claim
to the official full-dataset score. The tool never changes server configuration.
"""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import regex as re


def score_answer(content: str | None, answer: str) -> tuple[str | None, bool]:
    match = re.fullmatch(r"\s*([ABCD])[.)]?\s*", content or "")
    predicted = match.group(1) if match else None
    return predicted, predicted == answer


def request_json(base_url: str, path: str, body=None):
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new evidence path")
    model = request_json(args.base_url, "/v1/models")["data"][0]["id"]
    samples = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    if not samples or any(s["answer"] not in "ABCD" or len(s["answer"]) != 1 for s in samples):
        parser.error("nonempty dataset with single-letter answers required")
    correct = invalid = 0
    started = time.monotonic()
    with args.output.open("x") as output:
        for completed, sample in enumerate(samples, start=1):
            begin = time.monotonic()
            result = request_json(
                args.base_url,
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Answer with exactly one letter: A, B, C, or D. No explanation."},
                        {"role": "user", "content": sample["prompt"]},
                    ],
                    "temperature": 0,
                    "seed": 1024,
                    "max_tokens": 16,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            choice = result["choices"][0]
            content = choice["message"].get("content")
            prediction, passed = score_answer(content, sample["answer"])
            # A truncated answer cannot count as a successful accuracy sample.
            passed = passed and choice["finish_reason"] == "stop"
            correct += passed
            invalid += prediction is None or choice["finish_reason"] != "stop"
            record = {
                "id": sample["id"],
                "subject": sample["subject"],
                "prompt_sha256": hashlib.sha256(sample["prompt"].encode()).hexdigest(),
                "label": args.label,
                "answer": sample["answer"],
                "prediction": prediction,
                "correct": passed,
                "content": content,
                "finish_reason": choice["finish_reason"],
                "usage": result["usage"],
                "elapsed_seconds": time.monotonic() - begin,
            }
            output.write(json.dumps(record) + "\n")
            output.flush()
            print(json.dumps({"id": sample["id"], "correct": passed, "completed": completed}), flush=True)
    summary = {
        "label": args.label,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "samples": len(samples),
        "correct": correct,
        "invalid": invalid,
        "accuracy": correct / len(samples),
        "elapsed_seconds": time.monotonic() - started,
        "scope": "fixed sampled zero-shot MMLU; not the official full-dataset score",
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
