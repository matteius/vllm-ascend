#!/usr/bin/env python3
"""Validate c3/c4 changing-input replay and optional eager-reference parity."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import threading
import time
import urllib.request
from pathlib import Path

METRICS = (
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)
PROMPT_STEMS = (
    "List twelve positive multiples of {value}, comma-separated, then stop.",
    "List twelve positive triangular numbers starting after {value}, comma-separated, then stop.",
)


def get_json(base_url: str, path: str) -> dict:
    with urllib.request.urlopen(base_url + path, timeout=10) as response:
        return json.load(response)


def get_metrics(base_url: str) -> dict[str, float]:
    with urllib.request.urlopen(base_url + "/metrics", timeout=10) as response:
        lines = response.read().decode().splitlines()
    values = {name: 0.0 for name in METRICS}
    for line in lines:
        for name in METRICS:
            if line.startswith(name + "{") or line.startswith(name + " "):
                values[name] += float(line.rsplit(" ", 1)[1])
    return values


def request_one(
    base_url: str,
    model: str,
    concurrency: int,
    prompt_variant: int,
    slot: int,
    barrier: threading.Barrier,
) -> dict:
    value = 19 + concurrency * 10 + slot
    prompt = PROMPT_STEMS[prompt_variant].format(value=value)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0,
        "seed": 20260928 + prompt_variant * 100 + slot,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    barrier.wait()
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    content = result["choices"][0]["message"]["content"]
    if not content:
        raise RuntimeError(f"c{concurrency} variant {prompt_variant} slot {slot} returned empty output")
    if result["usage"]["completion_tokens"] != body["max_tokens"]:
        raise RuntimeError(
            f"c{concurrency} variant {prompt_variant} slot {slot} returned "
            f"{result['usage']['completion_tokens']} completion tokens"
        )
    return {
        "concurrency": concurrency,
        "prompt_variant": prompt_variant,
        "slot": slot,
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "content": content,
        "finish_reason": result["choices"][0]["finish_reason"],
        "usage": result["usage"],
    }


def run_round(base_url: str, model: str, concurrency: int, prompt_variant: int) -> list[dict]:
    barrier = threading.Barrier(concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(request_one, base_url, model, concurrency, prompt_variant, slot, barrier)
            for slot in range(concurrency)
        ]
        return [future.result() for future in futures]


def validate_replay(results: list[dict]) -> dict[str, str]:
    hashes: dict[str, set[str]] = {}
    for result in results:
        key = f"c{result['concurrency']}-v{result['prompt_variant']}-s{result['slot']}"
        hashes.setdefault(key, set()).add(result["sha256"])
    unstable = {key: sorted(values) for key, values in hashes.items() if len(values) != 1}
    if unstable:
        raise RuntimeError(f"changing-input graph replay was unstable: {unstable}")
    stable = {key: next(iter(values)) for key, values in hashes.items()}
    for concurrency in (3, 4):
        for slot in range(concurrency):
            if stable[f"c{concurrency}-v0-s{slot}"] == stable[f"c{concurrency}-v1-s{slot}"]:
                raise RuntimeError(f"c{concurrency} slot {slot} ignored changed input")
    return stable


def validate_full_graph_log(path: Path) -> dict[str, int]:
    log = path.read_text(errors="replace")
    failures = (
        "507903",
        "event resources are insufficient",
        "stream is not registered with any allocator",
        "Engine core initialization failed",
    )
    present = [signature for signature in failures if signature in log]
    if present:
        raise RuntimeError(f"capture log contains failure signatures: {present}")
    if "Graph capturing finished" not in log or not re.search(r"2/2.*\[", log):
        raise RuntimeError("log does not prove that both retained graphs captured")
    counts: dict[str, int] = {"9": 0, "12": 0}
    for size in counts:
        full_rows = re.findall(
            rf"\|\s*{size}\s*\|\s*{size}\s*\|\s*0\s*\|\s*FULL\s*\|\s*(\d+)\s*\|",
            log,
        )
        none_rows = re.findall(
            rf"\|\s*{size}\s*\|\s*{size}\s*\|\s*0\s*\|\s*NONE\s*\|\s*(\d+)\s*\|",
            log,
        )
        counts[size] = sum(map(int, full_rows))
        if counts[size] == 0 or sum(map(int, none_rows)):
            raise RuntimeError(
                f"shape {size} did not dispatch exclusively to FULL graphs: "
                f"full={counts[size]}, none={sum(map(int, none_rows))}"
            )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8004")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--expect-full-graphs", action="store_true")
    parser.add_argument("--max-acceptance-drop", type=float, default=0.03)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite validation evidence")
    if args.expect_full_graphs and args.server_log is None:
        raise ValueError("--server-log is required with --expect-full-graphs")

    model = get_json(args.base_url, "/v1/models")["data"][0]["id"]
    before = get_metrics(args.base_url)
    results: list[dict] = []
    # A, B, A, B makes every graph replay consume different input buffers and
    # also provides a deterministic repeat for each input on both shapes.
    for concurrency in (3, 4):
        for prompt_variant in (0, 1, 0, 1):
            results.extend(run_round(args.base_url, model, concurrency, prompt_variant))
    after = get_metrics(args.base_url)
    drafted = after[METRICS[0]] - before[METRICS[0]]
    accepted = after[METRICS[1]] - before[METRICS[1]]
    if drafted <= 0 or accepted < 0 or accepted > drafted:
        raise RuntimeError(f"invalid MTP counters: accepted={accepted}, drafted={drafted}")
    acceptance = accepted / drafted
    hashes = validate_replay(results)

    graph_counts = None
    if args.expect_full_graphs:
        # CUDAGraph stats are emitted periodically by the API process.
        time.sleep(12)
        assert args.server_log is not None
        graph_counts = validate_full_graph_log(args.server_log)

    evidence = {
        "model": model,
        "hashes": hashes,
        "mtp": {"accepted": accepted, "drafted": drafted, "acceptance": acceptance},
        "full_graph_counts": graph_counts,
        "results": results,
    }
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")

    if args.reference is not None:
        reference = json.loads(args.reference.read_text())
        if hashes != reference["hashes"]:
            changed = sorted(set(hashes) | set(reference["hashes"]))
            changed = [key for key in changed if hashes.get(key) != reference["hashes"].get(key)]
            raise RuntimeError(f"candidate differs from eager/reference outputs: {changed}")
        reference_acceptance = float(reference["mtp"]["acceptance"])
        if acceptance < reference_acceptance - args.max_acceptance_drop:
            raise RuntimeError(
                f"MTP acceptance regressed: reference={reference_acceptance:.4f}, "
                f"candidate={acceptance:.4f}"
            )

    print(json.dumps({"status": "PASS", **evidence["mtp"], "full_graph_counts": graph_counts}))


if __name__ == "__main__":
    main()
