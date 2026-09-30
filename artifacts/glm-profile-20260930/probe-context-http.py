#!/usr/bin/env python3
"""Exercise continued prefill and report the server's token count and latency."""

import argparse
import json
import time
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://192.168.53.187:8001/v1")
    parser.add_argument("--model", default="glm53-flash-ascend-profile")
    parser.add_argument("--repetitions", type=int, default=1500)
    parser.add_argument("--max-tokens", type=int, default=4)
    args = parser.parse_args()

    prompt = "Reply with OK after reading this: " + "blue green " * args.repetitions
    request = urllib.request.Request(
        args.url.rstrip("/") + "/chat/completions",
        data=json.dumps(
            {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": 0,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        print(f"HTTP {error.code} after {time.monotonic() - started:.2f}s")
        print(error.read().decode()[:4000])
        raise SystemExit(1) from error
    print(
        json.dumps(
            {
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "usage": result.get("usage"),
                "finish_reason": result["choices"][0]["finish_reason"],
                "content": result["choices"][0]["message"].get("content"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
