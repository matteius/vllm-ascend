"""Check GLM answer content separately from its required thinking prefix.

The checkpoint's chat template starts every assistant turn with ``<think>``.
Run against a server launched with ``--reasoning-parser glm47`` so the final
answer and reasoning are exposed in separate response fields.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

SECRET_CODE = "BLUE-ORCHID-7319"
FILLER = "The quick brown fox jumps over the lazy dog. Keep this context in mind.\n"


def build_prompt(repeats: int) -> str:
    return f"The secret code is {SECRET_CODE}.\n" + FILLER * repeats + "What is the secret code? Answer only the code."


def probe(base_url: str, model: str, repeats: int, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(repeats)}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.load(response)
    choice = result["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    return {
        "repeats": repeats,
        "elapsed_s": time.perf_counter() - start,
        "usage": result.get("usage"),
        "finish_reason": choice.get("finish_reason"),
        "content": content,
        "reasoning": reasoning,
        "content_contains_code": SECRET_CODE in content,
        "answer_is_exact": choice.get("finish_reason") == "stop" and content.strip() == SECRET_CODE,
        "content_has_thinking_marker": "<think>" in content or "</think>" in content,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://192.168.53.187:8001")
    parser.add_argument("--model", default="glm53-flash-ascend-profile")
    parser.add_argument("--repeats", type=int, nargs="+", default=[0])
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(repeats < 0 for repeats in args.repeats) or args.max_tokens <= 0:
        parser.error("repeats must be nonnegative and max-tokens must be positive")
    if args.output.exists():
        parser.error("output file already exists")

    records = [probe(args.base_url, args.model, repeats, args.max_tokens) for repeats in args.repeats]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    for record in records:
        usage = record["usage"] or {}
        print(
            f"repeats={record['repeats']} prompt_tokens={usage.get('prompt_tokens')} "
            f"finish={record['finish_reason']} exact_answer={record['answer_is_exact']} "
            f"raw_thinking={record['content_has_thinking_marker']}"
        )


if __name__ == "__main__":
    main()
