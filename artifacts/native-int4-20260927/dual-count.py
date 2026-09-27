"""Two long-enough concurrent decodes to exercise six-token MTP replay."""

import concurrent.futures
import json
import sys
import urllib.request
from pathlib import Path

import regex as re


def call(window):
    body = {
        "model": "qwen38-w4-experimental",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Independent window {window}: output integers 1 to 50 in ascending order, "
                    "separated by single spaces. No other text."
                ),
            }
        ],
        "temperature": 0,
        "seed": 1024,
        "max_tokens": 192,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        "http://127.0.0.1:8001/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    text = result["choices"][0]["message"]["content"]
    correct = [int(x) for x in re.findall(r"\d+", text)] == list(range(1, 51))
    return {"window": window, "correct": correct, "response": result}


def main():
    with Path(sys.argv[1]).open("x") as output, concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(call, window) for window in ("A", "B")]
        for future in futures:
            result = future.result()
            output.write(json.dumps(result) + "\n")
            output.flush()
            assert result["correct"] and result["response"]["choices"][0]["finish_reason"] == "stop", result
    print("DUAL_COUNT_PASS")


if __name__ == "__main__":
    main()
