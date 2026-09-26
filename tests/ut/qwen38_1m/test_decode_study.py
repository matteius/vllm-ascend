# SPDX-License-Identifier: Apache-2.0
"""Host gates for the isolated live-runtime experiment; no vLLM/NPU imports."""

import ast
import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tools.qwen38_decode_study import affinity, benchmark, prepare, summarize
from tools.qwen38_decode_study.grouped_draft import grouped_routed_experts, pack_draft_experts
from tools.qwen38_decode_study.guard import active_servers


def test_physical_core_pools_reserve_room_and_do_not_split_smt():
    topology = {cpu: (0, cpu % 32, (cpu % 32) // 4) for cpu in range(64)}
    groups = affinity.plan_groups(topology, set(range(64)), 4, 6)
    assert groups == [list(range(start, start + 6)) for start in (0, 8, 16, 24)]
    assert len(set(range(32)) - {cpu for group in groups for cpu in group}) == 8
    with pytest.raises(ValueError, match="shared"):
        affinity.validate_groups([[0], [32]], set(range(64)), topology)
    with pytest.raises(ValueError, match="unavailable"):
        affinity.validate_groups([[0], [8]], {0}, topology)
    with pytest.raises(ValueError, match="more than once"):
        affinity.validate_groups([[0, 0]], {0}, topology)


def test_restricted_cpuset_uses_available_smt_representatives():
    topology = {cpu: (0, cpu % 8, (cpu % 8) // 2) for cpu in range(16)}
    groups = affinity.plan_groups(topology, set(range(8, 16)), 2, 3)
    assert groups == [[8, 9, 10], [12, 13, 14]]
    with pytest.raises(ValueError, match="Not enough"):
        affinity.plan_groups(topology, {0, 1}, 4, 1)
    with pytest.raises(ValueError, match="one socket"):
        affinity.plan_groups({0: (0, 0, 0), 1: (1, 0, 0)}, {0, 1}, 2, 1)


def test_affinity_failure_restores_threads(monkeypatch):
    state = {0: {0, 1, 2, 3}, 10: {0, 1, 2, 3}, 11: {0, 1, 2, 3}}
    calls = []

    def set_affinity(tid, cpus):
        calls.append((tid, cpus))
        if tid == 11 and cpus == {0, 1}:
            raise PermissionError("test denial")
        state[tid] = cpus

    monkeypatch.setattr(affinity, "cpu_topology", lambda: {cpu: (0, cpu, 0) for cpu in range(4)})
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda tid: state[tid].copy())
    monkeypatch.setattr(affinity.os, "sched_setaffinity", set_affinity)
    monkeypatch.setattr(affinity.Path, "iterdir", lambda _: [SimpleNamespace(name="10"), SimpleNamespace(name="11")])
    with pytest.raises(PermissionError):
        affinity.bind_worker(0, ((0, 1), (2, 3)))
    assert state[10] == state[11] == {0, 1, 2, 3}
    assert calls[0] == (10, {0, 1})


def make_bank(offset=0):
    generator = torch.Generator().manual_seed(7)
    bank = nn.Module()
    bank.quantized_experts = True
    bank.num_local_experts = 2
    bank.expert_offset = offset
    for name, shape in (("gate_up_proj", (4, 6)), ("down_proj", (3, 4))):
        setattr(
            bank,
            name,
            nn.ParameterList(
                [
                    nn.Parameter(
                        torch.randint(-8, 9, shape, dtype=torch.int8, generator=generator), requires_grad=False
                    )
                    for _ in range(2)
                ]
            ),
        )
        setattr(
            bank,
            name + "_scale",
            nn.ParameterList(
                [
                    nn.Parameter(torch.full((shape[1],), 0.03125, dtype=torch.float16), requires_grad=False)
                    for _ in range(2)
                ]
            ),
        )
    return bank


def quantized_linear(x, weight, scale):
    activation_scale = (x.float().abs().amax(-1, keepdim=True) / 127).clamp_min(1e-8)
    activation = (x.float() / activation_scale).round().clamp(-127, 127) * activation_scale
    return (activation @ (weight.float() * scale[:, None]).t()).half()


class ReferenceOps:
    """CPU operator model deliberately leaves nonlocal output rows as NaN."""

    @staticmethod
    def npu_moe_init_routing_v2(x, ids, **kwargs):
        order = ids.flatten().argsort(stable=True)
        inverse = order.argsort().to(torch.int32)
        groups = torch.stack([(ids < i + 1).sum() for i in range(kwargs["expert_num"])])
        return x[order // ids.shape[1]], inverse, groups, None

    @staticmethod
    def npu_quant_grouped_matmul_dequant(x, weight, scale, groups):
        assert scale.dtype == torch.float32
        assert groups.dtype == torch.int64
        output = torch.full((len(x), weight.shape[1]), float("nan"), dtype=x.dtype)
        start = 0
        for expert in range(len(groups)):
            stop = int(groups[expert])
            if stop > start:
                output[start:stop] = quantized_linear(x[start:stop], weight[expert], scale[expert])
            start = stop
        return output

    @staticmethod
    def npu_swiglu(x):
        gate, up = x.chunk(2, -1)
        return torch.nn.functional.silu(gate) * up


@pytest.mark.parametrize("offset", [0, 2, 6])
@pytest.mark.parametrize("tokens", [1, 3])
def test_grouped_routes_match_per_route_reference_with_peer_nan_padding(offset, tokens):
    bank = make_bank(offset)
    old_weights = [[p.clone() for p in getattr(bank, name)] for name in ("gate_up_proj", "down_proj")]
    pack_draft_experts(bank, lambda value: value)
    x = torch.tensor([[0.3, -0.2, 0.9, -0.7]], dtype=torch.float16).repeat(tokens, 1)
    ids = torch.tensor([[3, 0, 2]]).repeat(tokens, 1)
    weights = torch.tensor([[0.312345, 0.487654, 0.2]]).repeat(tokens, 1)
    actual = grouped_routed_experts(bank, x, weights, ids, ReferenceOps)
    expected = torch.zeros(tokens, 4)
    for token in range(tokens):
        for slot in range(3):
            expert = int(ids[token, slot]) - offset
            if not 0 <= expert < 2:
                continue
            gate_up = quantized_linear(
                x[token : token + 1], old_weights[0][expert].t(), bank.gate_up_proj_scale[expert]
            )
            act = ReferenceOps.npu_swiglu(gate_up)
            output = quantized_linear(act, old_weights[1][expert].t(), bank.down_proj_scale[expert])
            expected[token] += output[0].float() * weights[token, slot]
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)


def test_packing_reuses_storage_preserves_names_and_rejects_double_pack():
    bank = make_bank()
    names = set(dict(bank.named_parameters()))
    expected = bank.gate_up_proj[1].clone()
    pack_draft_experts(bank, lambda value: value)
    assert set(dict(bank.named_parameters())) == names
    assert bank.gate_up_proj[1].untyped_storage().data_ptr() == bank.draft_gate_up_packed.untyped_storage().data_ptr()
    torch.testing.assert_close(bank.gate_up_proj[1], expected)
    assert not any("packed" in key for key in bank.state_dict())
    with pytest.raises(RuntimeError, match="already"):
        pack_draft_experts(bank, lambda value: value)
    bank = make_bank()
    bank.quantized_experts = False
    with pytest.raises(ValueError, match="quantized"):
        pack_draft_experts(bank, lambda value: value)


def test_launch_guard_finds_cli_and_workers_without_matching_its_own_script(tmp_path):
    for pid, name, args in [
        (10, "vllm", b"python\0/bin/vllm\0serve\0model\0"),
        (11, "VLLM::Worker_TP", b"worker\0"),
        (12, "python3", b"python3\0candidate_guard.py\0"),
    ]:
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "comm").write_text(name)
        (directory / "cmdline").write_bytes(args)
    assert active_servers(tmp_path) == [10, 11]


def test_patch_anchors_fail_closed_and_destinations_cannot_alias_baseline(tmp_path):
    with pytest.raises(ValueError, match="patch anchor"):
        prepare.replace_once("anchor anchor", "anchor", "replacement")
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="outside"):
        prepare.prepare(source, source / "candidate", tmp_path / "launcher", "control", None)
    with pytest.raises(FileNotFoundError):
        prepare.prepare(source, tmp_path / "candidate", tmp_path / "launcher", "control", None)
    assert not (tmp_path / "candidate").exists()


def test_affinity_patch_is_opt_in_and_keeps_arm_path():
    source = "from vllm.logger import logger\ndef bind_cpus(rank_id, npu_id):\n    if not is_arm_cpu():\n"
    source += (
        '        logger.info("CPU binding skipped: non-ARM CPU detected.")\n        return\n    CpuAlloc().run_all()\n'
    )
    result = prepare.patch_affinity(source, [[0, 1], [2, 3]])
    ast.parse(result)
    assert "bind_worker(rank_id, ((0, 1), (2, 3)))" in result
    assert "CpuAlloc().run_all()" in result


@pytest.mark.parametrize("draft_eager", [False, True])
def test_mtp_patch_selects_capture_boundary_and_packs_after_loading(draft_eager):
    source = """from .dtype_policy import Qwen4ExpDtypePolicy
class _MTPFP16MoE:
    def _forward_eager(self, x):
        logits = self.router(x)
        weights, ids = self.topk(logits)
        flat_ids = ids.flatten()
        output = self.experts(x)
        if self.local_shared_intermediate:
            output += self.shared(x)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        capture = Capture.current()
        if capture is None or not capture._capturing:
            return self._forward_eager(x)
        weak_x = x
        weak_output = x
        def run_experts_eager():
            weak_output.copy_(self._forward_eager(weak_x))
        capture.add_eager(run_experts_eager)
        return weak_output

class Model:
    def load_weights(self):
        loaded = set()
        return loaded
"""
    result = prepare.patch_mtp(source, draft_eager)
    tree = ast.parse(result)
    bank = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_MTPFP16MoE")
    forward = next(node for node in bank.body if node.name == "forward")
    if draft_eager:
        assert isinstance(forward.body[0], ast.Assign)
        assert "weak_output.copy_(self._forward_grouped(weak_x))" in result
        assert "self._forward_eager(" not in result
    else:
        assert isinstance(forward.body[0], ast.If)
    assert "grouped_routed_experts(self, x, weights, ids)" in result
    assert "pack_draft_experts(layer.mlp)" in result
    assert result.count("output += self.shared(x)") == 2
    with pytest.raises(ValueError, match="patch anchor"):
        prepare.patch_mtp(result, draft_eager)


def test_grouped_forward_does_not_read_device_tensors_on_host():
    bank = make_bank()
    pack_draft_experts(bank, lambda value: value)
    with patch.object(torch.Tensor, "tolist", side_effect=AssertionError("Unexpected host list")):
        result = grouped_routed_experts(
            bank, torch.ones(1, 4).half(), torch.tensor([[0.5, 0.5]]), torch.tensor([[0, 1]]), ReferenceOps
        )
    assert torch.isfinite(result).all()


def test_stream_timing_ignores_role_only_chunks_and_uses_usage(monkeypatch):
    events = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "hello "}}]},
        {"choices": [{"delta": {"content": "world"}, "finish_reason": "length"}]},
        {"choices": [], "usage": {"completion_tokens": 4, "prompt_tokens": 10}},
    ]
    stream = "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(stream.encode()))
    times = iter([0, 2, 4, 5])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    result = benchmark.stream_completion("http://localhost:8002", {})
    assert result["ttft_s"] == 2
    assert result["client_decode_tok_s"] == 1.5
    assert result["text"] == "hello world"
    assert result["finish_reason"] == "length"


def test_stream_failure_is_not_scored(monkeypatch):
    stream = b'data: {"error": "engine failed"}\n\n'
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(stream))
    with pytest.raises(RuntimeError, match="engine failed"):
        benchmark.stream_completion("http://localhost:8002", {})


def test_metrics_selects_exact_counter_names(monkeypatch):
    stream = b"""# HELP vllm:num_requests_running Running count
vllm:num_requests_running{engine="0"} 0
vllm:num_requests_waiting{engine="0"} 1
vllm:spec_decode_num_draft_tokens_total{engine="0"} 5
vllm:spec_decode_num_draft_tokens_total{engine="1"} 7
vllm:spec_decode_num_draft_tokens_created{engine="0"} 1790000000
"""
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(stream))
    assert benchmark.metrics("http://localhost:8002") == {
        "vllm:num_requests_running": 0,
        "vllm:num_requests_waiting": 1,
        "vllm:spec_decode_num_draft_tokens_total": 12,
    }


def test_summary_uses_server_gaps_and_rejects_unmatched_workloads():
    timings = summarize.server_timings(
        "request chatcmpl-test (length): prompt 32 computed + 0 cached in 500.0 ms; "
        "decode 512 tokens, 511 gaps in 25550.0 ms (50.0 ms/tok, 20.0 tok/s); queue 0.0 ms"
    )
    assert timings["chatcmpl-test"] == {"tokens": 512, "decode_tok_s": 20.0}
    row = {
        "request": {"messages": ["same prompt"]},
        "server_decode_tok_s": 20,
        "text_sha256": "same",
        "acceptance": 0.8,
        "usage": {"prompt_tokens": 32},
    }
    baseline = {("short", 512, 0): row}
    candidate = {("short", 512, 0): {**row, "server_decode_tok_s": 22}}
    result = summarize.compare(baseline, candidate)
    assert result["summary"][0]["median_paired_speedup"] == 1.1
    assert result["summary"][0]["identical_outputs"] == 1
    candidate["short", 512, 0]["request"] = {"messages": ["different prompt"]}
    with pytest.raises(ValueError, match="configuration differs"):
        summarize.compare(baseline, candidate)
