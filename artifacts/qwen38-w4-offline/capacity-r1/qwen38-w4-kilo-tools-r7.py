"""Real streaming auto-tool calls and tool-result round trips; no tool execution."""

import argparse
import concurrent.futures
import json
import time
import urllib.request


def stream(base, messages, thinking):
    payload = {
        "model": "qwen38-w4-experimental",
        "messages": messages,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_access_code",
                    "description": "Look up the current access code for the named test session.",
                    "parameters": {
                        "type": "object",
                        "properties": {"session": {"type": "string"}},
                        "required": ["session"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "temperature": 0,
        "seed": 1024,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": thinking},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    content = reasoning = ""
    calls = {}
    finish = usage = None
    done = False
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                done = True
                break
            event = json.loads(raw)
            if "error" in event:
                raise RuntimeError(event)
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                content += delta.get("content") or ""
                reasoning += delta.get("reasoning") or delta.get("reasoning_content") or ""
                finish = choice.get("finish_reason") or finish
                for chunk in delta.get("tool_calls", []):
                    call = calls.setdefault(
                        chunk["index"],
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if chunk.get("id"):
                        call["id"] = chunk["id"]
                    for key in ("name", "arguments"):
                        call["function"][key] += chunk.get("function", {}).get(key) or ""
    if not done or not finish or not usage:
        raise AssertionError("Incomplete stream or missing usage")
    return {
        "content": content,
        "reasoning": reasoning,
        "tool_calls": [calls[index] for index in sorted(calls)],
        "finish_reason": finish,
        "usage": usage,
        "elapsed_s": time.monotonic() - start,
    }


def round_trip(base, label, thinking):
    messages = [
        {
            "role": "user",
            "content": (
                f"Use lookup_access_code to fetch the access code for session {label}. "
                "Do not guess it. After the tool returns, reply with only the code."
            ),
        }
    ]
    first = stream(base, messages, thinking)
    print(json.dumps({"event": "tool_call", "session": label, "thinking": thinking, **first}), flush=True)
    assert first["finish_reason"] == "tool_calls", first
    assert len(first["tool_calls"]) == 1, first
    call = first["tool_calls"][0]
    assert call["id"] and call["function"]["name"] == "lookup_access_code", call
    assert json.loads(call["function"]["arguments"]) == {"session": label}, call
    code = f"CODE-{label}-73491"
    messages.extend(
        [
            {"role": "assistant", "content": first["content"] or None, "tool_calls": first["tool_calls"]},
            {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"access_code": code})},
        ]
    )
    second = stream(base, messages, thinking)
    passed = second["finish_reason"] == "stop" and not second["tool_calls"] and second["content"].strip() == code
    print(json.dumps({"event": "tool_result", "session": label, "passed": passed, **second}), flush=True)
    assert passed, second


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://192.168.53.187:8001")
    args = parser.parse_args()
    print(json.dumps({"event": "start", "base_url": args.base_url, "model": "qwen38-w4-experimental"}), flush=True)
    round_trip(args.base_url, "single", False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(round_trip, args.base_url, label, True) for label in ("alpha", "beta")]
        for future in futures:
            future.result()
    print(json.dumps({"event": "summary", "passed": True, "round_trips": 3}), flush=True)


if __name__ == "__main__":
    main()
