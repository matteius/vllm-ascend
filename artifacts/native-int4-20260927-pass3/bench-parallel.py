"""Measure aggregate streaming decode throughput at fixed concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path


def request_json(base_url: str, path: str) -> dict:
    with urllib.request.urlopen(base_url + path, timeout=10) as response:
        return json.load(response)


def stream_one(base_url: str, model: str, index: int, max_tokens: int, barrier: threading.Barrier) -> dict:
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"Independent throughput stream {index}. Explain integer sorting algorithms in detail.",
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 20260928 + index,
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
    barrier.wait()
    started = time.perf_counter()
    first = last = None
    usage = None
    with urllib.request.urlopen(request, timeout=600) as response:
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
        "started": started,
        "first": first,
        "last": last,
        "finished": finished,
        "ttft_s": first - started,
        "decode_s": last - first,
        "client_decode_tok_s": (completion_tokens - 1) / (last - first),
        "e2e_s": finished - started,
    }


def monitor(base_url: str, stop: threading.Event, samples: list[dict]) -> None:
    while not stop.is_set():
        try:
            with urllib.request.urlopen(base_url + "/metrics", timeout=2) as response:
                lines = response.read().decode().splitlines()
            sample = {"time": time.perf_counter(), "running": 0.0, "waiting": 0.0}
            for line in lines:
                for metric, field in (
                    ("vllm:num_requests_running", "running"),
                    ("vllm:num_requests_waiting", "waiting"),
                ):
                    if line.startswith(metric + "{"):
                        sample[field] += float(line.rsplit(" ", 1)[1])
            samples.append(sample)
        except OSError:
            pass
        stop.wait(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite benchmark evidence")
    if args.concurrency < 1 or args.max_tokens < 2:
        raise ValueError("concurrency and max-tokens must be positive")

    model = request_json(args.base_url, "/v1/models")["data"][0]["id"]
    barrier = threading.Barrier(args.concurrency)
    stop = threading.Event()
    samples: list[dict] = []
    monitor_thread = threading.Thread(target=monitor, args=(args.base_url, stop, samples), daemon=True)
    monitor_thread.start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(stream_one, args.base_url, model, index, args.max_tokens, barrier)
            for index in range(args.concurrency)
        ]
        streams = [future.result() for future in futures]
    stop.set()
    monitor_thread.join(timeout=2)

    total_tokens = sum(item["completion_tokens"] for item in streams)
    wall_started = min(item["started"] for item in streams)
    wall_first = min(item["first"] for item in streams)
    wall_last = max(item["last"] for item in streams)
    wall_finished = max(item["finished"] for item in streams)
    result = {
        "concurrency": args.concurrency,
        "max_tokens_per_stream": args.max_tokens,
        "total_completion_tokens": total_tokens,
        "aggregate_e2e_tok_s": total_tokens / (wall_finished - wall_started),
        "aggregate_decode_tok_s": (total_tokens - args.concurrency) / (wall_last - wall_first),
        "median_stream_decode_tok_s": statistics.median(item["client_decode_tok_s"] for item in streams),
        "max_running": max((sample["running"] for sample in samples), default=None),
        "max_waiting": max((sample["waiting"] for sample in samples), default=None),
        "streams": streams,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
