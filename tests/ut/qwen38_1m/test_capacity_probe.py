# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import io
import json
import threading

import pytest

from tools.qwen4exp.benchmark_capacity import fit_prompt, parse_metrics, stream_request


def test_exact_prompt_budget_keeps_both_ends():
    assert fit_prompt([1, 2], [3, 4], [5, 6], 9) == [1, 2, 3, 4, 3, 4, 3, 5, 6]
    assert fit_prompt([1], [3], [2], 2) == [1, 2]


@pytest.mark.parametrize("filler,budget", [([], 4), ([3], 1)])
def test_invalid_budget(filler, budget):
    with pytest.raises(ValueError):
        fit_prompt([1], filler, [2], budget)


def test_metrics_do_not_confuse_waiting_with_waiting_by_reason():
    result = parse_metrics(
        'vllm:num_requests_running{engine="0"} 2.0\n'
        'vllm:num_requests_waiting{engine="0"} 1.0\n'
        'vllm:num_requests_waiting_by_reason{reason="capacity"} 1.0\n'
        'vllm:kv_cache_usage_perc{engine="0"} 8.5e-1\n'
        'vllm:num_preemptions_total{engine="0"} 3.0\n'
    )
    assert result["num_requests_running"] == 2
    assert result["num_requests_waiting"] == 1
    assert result["kv_cache_usage_perc"] == 0.85
    assert result["num_preemptions_total"] == 3


def test_completed_stream_requires_exact_prompt_usage(monkeypatch):
    events = [
        {"choices": [{"text": "test output", "finish_reason": None}]},
        {"choices": [{"text": "", "finish_reason": "length"}], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
    ]
    data = "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n"
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(data.encode()))
    payload = {"prompt": [1, 2], "max_tokens": 3}
    result = stream_request("http://localhost", payload, threading.Barrier(1), None)
    assert result["correct"] and result["done"]
    assert result["decode_tok_s"] > 0
    with pytest.raises(AssertionError, match="prompt length"):
        stream_request("http://localhost", {**payload, "prompt": [1]}, threading.Barrier(1), None)


def test_stream_error_is_not_a_pass(monkeypatch):
    data = b'data: {"error": {"message": "engine died"}}\n\n'
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(data))
    with pytest.raises(RuntimeError, match="engine died"):
        stream_request("http://localhost", {"prompt": [1], "max_tokens": 3}, threading.Barrier(1), None)
