"""Repeat deterministic four-request batches and compare exact HTTP outputs."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import sys
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8002"
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
PROMPTS = (
    "List the first 40 positive multiples of 7, comma-separated. No other text.",
    "List the first 40 positive multiples of 11, comma-separated. No other text.",
    "List the first 40 positive multiples of 13, comma-separated. No other text.",
    "List the first 40 positive multiples of 17, comma-separated. No other text.",
)


def request(model: str, index: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPTS[index]}],
        "max_tokens": 256,
        "temperature": 0,
        "seed": 20260928 + index,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE_URL + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    content = result["choices"][0]["message"]["content"]
    return {
        "index": index,
        "content": content,
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "finish_reason": result["choices"][0]["finish_reason"],
        "usage": result["usage"],
    }


def main() -> None:
    with urllib.request.urlopen(BASE_URL + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]
    batches = []
    for repeat in range(REPEATS):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            batch = list(pool.map(lambda index: request(model, index), range(4)))
        batches.append(batch)
        print(json.dumps({"repeat": repeat, "results": batch}), flush=True)
    for index in range(4):
        hashes = {batch[index]["sha256"] for batch in batches}
        if len(hashes) != 1:
            raise RuntimeError(f"request {index} changed across graph replays: {sorted(hashes)}")
    print("OVERLAP_REPLAY_STABLE", flush=True)


if __name__ == "__main__":
    main()
