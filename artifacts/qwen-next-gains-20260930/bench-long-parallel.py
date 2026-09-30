"""Measure warm-prefix long-context decode at fixed concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path


def request_json(base_url: str, path: str, body: dict | None = None, timeout: int = 60) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def stream_one(
    base_url: str,
    model: str,
    prefix: str,
    index: int,
    max_tokens: int,
    barrier: threading.Barrier | None,
) -> dict:
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    prefix + f"\n\nIndependent stream {index}: explain stable merge sort and its complexity in detail."
                ),
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 20260930 + index,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    if barrier is not None:
        barrier.wait()
    started = time.perf_counter()
    first = last = None
    usage = None
    with urllib.request.urlopen(request, timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
                if text:
                    last = time.perf_counter()
                    first = last if first is None else first
    finished = time.perf_counter()
    if usage is None or first is None or last is None:
        raise RuntimeError(f"stream {index} returned no usage or text")
    completion_tokens = usage["completion_tokens"]
    return {
        "stream": index,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": completion_tokens,
        "ttft_s": first - started,
        "decode_s": last - first,
        "decode_tok_s": (completion_tokens - 1) / (last - first),
        "started": started,
        "first": first,
        "last": last,
        "finished": finished,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--prefix-file", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-prefix", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite benchmark evidence")
    model = request_json(args.base_url, "/v1/models")["data"][0]["id"]
    prefix = args.prefix_file.read_text()
    warm = None
    if args.warm_prefix:
        warm = stream_one(args.base_url, model, prefix, 10_000, 8, None)

    barrier = threading.Barrier(args.concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        streams = [
            future.result()
            for future in [
                pool.submit(stream_one, args.base_url, model, prefix, index, args.max_tokens, barrier)
                for index in range(args.concurrency)
            ]
        ]
    total_tokens = sum(stream["completion_tokens"] for stream in streams)
    first = min(stream["first"] for stream in streams)
    last = max(stream["last"] for stream in streams)
    result = {
        "concurrency": args.concurrency,
        "max_tokens_per_stream": args.max_tokens,
        "aggregate_decode_tok_s": (total_tokens - args.concurrency) / (last - first),
        "median_stream_decode_tok_s": statistics.median(stream["decode_tok_s"] for stream in streams),
        "warmup": warm,
        "streams": streams,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
