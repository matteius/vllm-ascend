"""Run one long-context streaming decode without a duplicate cold-prefill warmup."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

PROMPT = "Write a detailed Python implementation of an LRU cache with unit tests. Explain thread safety and complexity."


def request_json(base_url: str, path: str, body: dict | None = None, *, timeout: int = 30) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def metrics(base_url: str) -> dict[str, float]:
    with urllib.request.urlopen(base_url + "/metrics", timeout=10) as response:
        lines = response.read().decode().splitlines()
    names = (
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
    )
    values: dict[str, float] = {}
    for line in lines:
        for name in names:
            if line.startswith(name + "{"):
                values[name] = values.get(name, 0) + float(line.rsplit(" ", 1)[1])
    return values


def stream_completion(base_url: str, body: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = last = None
    chunks: list[str] = []
    usage = None
    finish_reason = None
    request_id = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            request_id = event.get("id", request_id)
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
                if text:
                    last = time.perf_counter()
                    first = last if first is None else first
                    chunks.append(text)
                finish_reason = choice.get("finish_reason") or finish_reason
    if usage is None or first is None or last is None:
        raise RuntimeError("Streaming completion had no text or usage")
    completion_tokens = usage["completion_tokens"]
    text = "".join(chunks)
    return {
        "request_id": request_id,
        "usage": usage,
        "finish_reason": finish_reason,
        "ttft_s": first - started,
        "decode_s": last - first,
        "client_decode_tok_s": (completion_tokens - 1) / (last - first),
        "e2e_s": time.perf_counter() - started,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prefix-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite benchmark evidence")
    before = metrics(args.base_url)
    if any(before.get(name, 0) for name in ("vllm:num_requests_running", "vllm:num_requests_waiting")):
        raise SystemExit("Benchmark server is busy")
    model = request_json(args.base_url, "/v1/models")["data"][0]["id"]
    prefix = args.prefix_file.read_text()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prefix + "\n\nTask:\n" + PROMPT}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "seed": 42,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    result = stream_completion(args.base_url, body, args.timeout)
    time.sleep(1)
    after = metrics(args.base_url)
    drafted = after.get("vllm:spec_decode_num_draft_tokens_total", 0) - before.get(
        "vllm:spec_decode_num_draft_tokens_total", 0
    )
    accepted = after.get("vllm:spec_decode_num_accepted_tokens_total", 0) - before.get(
        "vllm:spec_decode_num_accepted_tokens_total", 0
    )
    result.update(
        {
            "prefix_path": str(args.prefix_file),
            "prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
            "max_tokens": args.max_tokens,
            "drafted_delta": drafted,
            "accepted_delta": accepted,
            "acceptance": accepted / drafted if drafted else None,
        }
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
