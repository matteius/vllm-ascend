# SPDX-License-Identifier: Apache-2.0
"""Small deterministic real-weight text/tool gate, separate from throughput."""

from __future__ import annotations

import argparse
import json
from contextlib import suppress
from pathlib import Path

from tools.qwen38_decode_study.benchmark import request_json

CASES = (
    ("Reply with only the integer: 17 * 23", "391"),
    ("Reply with only the integer: 144 / 12 + 7", "19"),
    ("Reverse ASCEND. Reply with only the reversed uppercase word.", "DNECSA"),
    ("What is the capital of Germany? Reply with only the city name.", "Berlin"),
    ("Reply with only the integer: how many distinct values are in [3, 1, 3, 2, 1]?", "3"),
    ("A list has 8 elements, indexed from zero. Reply with only the final valid index.", "7"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve evidence before requests; retain completed cases if a later one stalls.
    with args.output.open("x") as output:
        output.write("[]\n")
    model = request_json(args.base_url, "/v1/models")["data"][0]["id"]
    settings = {
        "model": model,
        "temperature": 0,
        "seed": 42,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    results = []
    for prompt, expected in CASES:
        response = request_json(
            args.base_url,
            "/v1/chat/completions",
            {
                **settings,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        choice = response["choices"][0]
        actual = (choice["message"].get("content") or "").strip()
        results.append({"prompt": prompt, "expected": expected, "pass": actual == expected, "response": response})
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    tool = {
        "type": "function",
        "function": {
            "name": "record_result",
            "description": "Record a task result.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "string"}, "value": {"type": "integer"}},
                "required": ["task_id", "value"],
                "additionalProperties": False,
            },
        },
    }
    messages = [{"role": "user", "content": "Call record_result with task_id T-001 and value 391. Do not add prose."}]
    response = request_json(
        args.base_url,
        "/v1/chat/completions",
        {
            **settings,
            "messages": messages,
            "tools": [tool],
            "tool_choice": "auto",
        },
    )
    choice = response["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    passed = False
    if len(calls) == 1:
        with suppress(KeyError, ValueError):
            passed = (
                calls[0]["function"]["name"] == "record_result"
                and json.loads(calls[0]["function"]["arguments"]) == {"task_id": "T-001", "value": 391}
                and choice["finish_reason"] == "tool_calls"
            )
    results.append({"prompt": messages[0]["content"], "pass": passed, "response": response})
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"passed": sum(case["pass"] for case in results), "total": len(results)}), flush=True)
    if not all(case["pass"] for case in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
