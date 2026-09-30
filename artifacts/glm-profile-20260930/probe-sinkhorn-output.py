"""Capture deterministic GLM responses across mHC implementations."""

import json
import sys
import urllib.request

PROMPTS = (
    "What is 17 multiplied by 23? Give the result and one short calculation.",
    "Write a Python function that returns the sum of the first n integers.",
)
ENDPOINT = "http://192.168.53.187:8001/v1/chat/completions"


def main(output_path: str) -> None:
    records = []
    for prompt in PROMPTS:
        request_body = {
            "model": "glm53-flash-ascend-profile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 16,
            "logprobs": True,
            "top_logprobs": 3,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            ENDPOINT,
            json.dumps(request_body).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            records.append({"prompt": prompt, "response": json.load(response)})
    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(records, output, indent=2)


if __name__ == "__main__":
    main(sys.argv[1])
