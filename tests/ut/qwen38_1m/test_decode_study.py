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
from tools.qwen38_decode_study.grouped_draft import grouped_routed_experts, pack_draft_experts, route_local_experts
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


def make_bank(offset=0, num_experts=2):
    generator = torch.Generator().manual_seed(7)
    bank = nn.Module()
    bank.quantized_experts = True
    bank.num_local_experts = num_experts
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
                    for _ in range(num_experts)
                ]
            ),
        )
        setattr(
            bank,
            name + "_scale",
            nn.ParameterList(
                [
                    nn.Parameter(torch.full((shape[1],), 0.03125, dtype=torch.float16), requires_grad=False)
                    for _ in range(num_experts)
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
        expert_count = kwargs["expert_num"]
        assert ids.dtype == torch.int32
        assert ((ids >= 0) & (ids < expert_count)).all(), "Routing expert IDs exceed the V2 count buffer"
        assert kwargs["active_expert_range"] == [0, expert_count]
        assert kwargs["active_num"] == ids.numel()
        order = ids.flatten().argsort(stable=True)
        inverse = order.argsort().to(torch.int32)
        groups = torch.bincount(ids.flatten().long(), minlength=expert_count).cumsum(0).int()
        return x[order // ids.shape[1]], inverse, groups, None

    @staticmethod
    def npu_quant_grouped_matmul_dequant(x, weight, scale, groups):
        assert scale.dtype == torch.float32
        assert groups.dtype == torch.int64
        assert len(groups) == len(weight) == len(scale), "Virtual expert must not reach matmul"
        assert ((groups >= 0) & (groups <= len(x))).all()
        assert (groups[1:] >= groups[:-1]).all()
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


@pytest.mark.parametrize("invalid_id", [-1, 128])
def test_routing_reference_rejects_ids_outside_declared_experts(invalid_id):
    with pytest.raises(AssertionError, match="expert IDs"):
        ReferenceOps.npu_moe_init_routing_v2(
            torch.ones(1, 4).half(),
            torch.tensor([[invalid_id]], dtype=torch.int32),
            expert_num=128,
            active_expert_range=[0, 128],
            active_num=1,
        )


@pytest.mark.parametrize("offset", [0, 128, 256, 384])
@pytest.mark.parametrize("tokens", [1, 2])
def test_routing_128_experts_preserves_inverse_and_local_counts_across_route_changes(offset, tokens):
    local_ids = torch.tensor([0, 1, 5, 10, 32, 64, 96, 120, 126, 127]) + offset
    peer_ids = (local_ids + 128) % 512
    mixed_ids = torch.where(torch.arange(10) % 2 == 0, local_ids, peer_ids)
    x = torch.arange(tokens * 2560).reshape(tokens, 2560).half()
    # Reuse inputs across populated -> empty -> mixed -> populated routing.
    # This is a CPU contract gate, not a substitute for NPU graph replay.
    for route_ids in (local_ids, peer_ids, mixed_ids, local_ids):
        ids = torch.stack([route_ids.roll(token) for token in range(tokens)])
        sorted_x, inverse, groups = route_local_experts(x, ids, 128, offset, ReferenceOps)
        assert groups.shape == (128,)
        assert groups.dtype == torch.int64
        expected_counts = torch.stack(
            [((ids >= offset) & (ids <= offset + expert)).sum() for expert in range(128)]
        )
        torch.testing.assert_close(groups, expected_counts)
        restored = sorted_x.index_select(0, inverse).view(tokens, 10, 2560)
        torch.testing.assert_close(restored, x[:, None, :].expand(-1, 10, -1))


def test_routing_requires_a_local_expert_bank():
    with pytest.raises(ValueError, match="positive"):
        route_local_experts(torch.ones(1, 4), torch.tensor([[0]]), 0, 0, ReferenceOps)


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


@pytest.mark.parametrize("offset", [0, 128, 256, 384])
def test_grouped_128_experts_keeps_virtual_expert_out_of_matmuls(offset):
    bank = make_bank(offset, num_experts=128)
    pack_draft_experts(bank, lambda value: value)
    x = torch.tensor([[0.3, -0.2, 0.9, -0.7]], dtype=torch.float16)
    weights = torch.linspace(0.05, 0.15, 10).reshape(1, 10)
    mixed_ids = torch.tensor([[127, 128, 255, 256, 383, 384, 511, 0, 10, 42]])
    peer_ids = (torch.arange(10).reshape(1, 10) + offset + 128) % 512
    for ids in (mixed_ids, peer_ids, mixed_ids):
        actual = grouped_routed_experts(bank, x, weights, ids, ReferenceOps)
        expected = torch.zeros_like(actual)
        for slot, global_id in enumerate(ids[0].tolist()):
            expert = global_id - offset
            if not 0 <= expert < bank.num_local_experts:
                continue
            gate_up = quantized_linear(x, bank.gate_up_proj[expert].t(), bank.gate_up_proj_scale[expert])
            output = quantized_linear(
                ReferenceOps.npu_swiglu(gate_up), bank.down_proj[expert].t(), bank.down_proj_scale[expert]
            )
            expected[0] += output[0].float() * weights[0, slot]
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


@pytest.mark.parametrize("draft_eager", [None, False, True])
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
        output = self._tp_reduce(output)
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
    result = prepare.patch_mtp(source) if draft_eager is None else prepare.patch_mtp(source, draft_eager)
    tree = ast.parse(result)
    bank = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_MTPFP16MoE")
    forward = next(node for node in bank.body if node.name == "forward")
    if draft_eager is not False:
        assert isinstance(forward.body[0], ast.Assign)
        assert "weak_output.copy_(self._forward_grouped(weak_x))" in result
        assert "self._forward_eager(" not in result
        assert "capture.add_eager(run_experts_eager)" in result
    else:
        assert isinstance(forward.body[0], ast.If)
    assert "grouped_routed_experts(self, x, weights, ids)" in result
    assert "pack_draft_experts(layer.mlp)" in result
    assert result.count("output += self.shared(x)") == 2
    grouped = next(node for node in bank.body if node.name == "_forward_grouped")
    assert "self._tp_reduce(output)" in ast.unparse(grouped)
    with pytest.raises(ValueError, match="patch anchor"):
        prepare.patch_mtp(result, draft_eager)


def test_target_routing_patch_only_replaces_small_batch_dispatch():
    source = """from .grouped_expert_dispatch import build_grouped_expert_dispatch
def _w8a8_packed_grouped_experts_npu(x, topk_ids, w13_weight, expert_offset):
    if x.device.type == "npu":
        # The 310P routing kernel computes the expert permutation, inverse
        num_local_experts = w13_weight.shape[0]
        local_ids = topk_ids.to(torch.int32)
        sorted_x, inverse_order, group_list, _ = torch_npu.npu_moe_init_routing_v2(x, local_ids)
        group_list = group_list.to(torch.int64)
        local_rows = torch.arange(num_tokens * top_k, device=x.device) < group_list[-1]
        return sorted_x, inverse_order, group_list, local_rows
    return build_grouped_expert_dispatch(topk_ids)
"""
    result = prepare.patch_target_routing(source)
    ast.parse(result)
    assert "npu_moe_init_routing_v2" not in result
    assert "x, topk_ids, w13_weight.shape[0], expert_offset, torch_npu" in result
    assert "from vllm_ascend._310p.qwen38_grouped_candidate import route_local_experts" in result
    assert "return build_grouped_expert_dispatch(topk_ids)" in result
    with pytest.raises(ValueError, match="routing patch anchor"):
        prepare.patch_target_routing(result)
    with pytest.raises(ValueError, match="routing patch anchor"):
        prepare.patch_target_routing(source + source)


@pytest.mark.parametrize("option,expected", [(None, None), ("--draft-eager", True), ("--draft-capture", False)])
def test_prepare_cli_leaves_draft_capture_opt_in(monkeypatch, capsys, option, expected):
    calls = []
    args = ["prepare", "--source", "base", "--destination", "new", "--launcher", "launch", "--arm", "grouped"]
    if option is not None:
        args.append(option)
    monkeypatch.setattr("sys.argv", args)
    monkeypatch.setattr(prepare, "prepare", lambda *values: calls.append(values) or {})
    prepare.main()
    assert calls[0][-1] is expected
    assert json.loads(capsys.readouterr().out) == {}


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
