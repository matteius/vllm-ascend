# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed W4, real loader, TP partial sums, and W8 isolation (CPU only)."""

import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

from tests.ut.qwen38_1m.test_model_load_and_moe import _single_rank_tp, _tiny_moe_config, _vllm_config
from vllm_ascend.models.qwen4_exp.dtype_policy import Qwen4ExpDtypePolicy
from vllm_ascend.models.qwen4_exp.model import AscendQwen4ExpForCausalLM, _EagerSparseMoE
from vllm_ascend.models.qwen4_exp.moe import route_topk
from vllm_ascend.models.qwen4_exp.w4_moe import (
    FORMAT,
    KINDS,
    MAX_CUBE_ROUTES,
    MAX_SHARED_EXPERT_OVERLAP_TOKENS,
    PackedExpertBank,
    W4SparseMoE,
    dequantize,
    pack_cube_tiles,
    require_eager_w4,
    unpack_signed_int4,
    w4_config,
)


def config(**kwargs):
    cfg = _tiny_moe_config(**kwargs)
    cfg.ascend_expert_quantization = {
        "format": FORMAT,
        "bits": 4,
        "symmetric": False,
        "group_size": 8,
        "packing": "signed_int4_low_nibble_first_in_axis",
        "backend": "eager_dequant",
        "scale_dtype": "float16",
        "offset_dtype": "int8",
    }
    return cfg


def build(cfg):
    from vllm.config import set_current_vllm_config

    vcfg = _vllm_config(cfg)
    vcfg.model_config.enforce_eager = True
    with _single_rank_tp(), set_current_vllm_config(vcfg):
        return AscendQwen4ExpForCausalLM(vllm_config=vcfg)


def pack(q):
    return ((q[..., ::2] & 15) | (q[..., 1::2] << 4)).to(torch.int8)


def payloads(cfg):
    generator = torch.Generator().manual_seed(104)
    group_size = cfg.ascend_expert_quantization["group_size"]
    for layer in range(cfg.num_hidden_layers):
        for expert in range(cfg.num_experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                output, inputs = (
                    (cfg.hidden_size, cfg.moe_intermediate_size)
                    if projection == "down_proj"
                    else (cfg.moe_intermediate_size, cfg.hidden_size)
                )
                values = {
                    "weight": pack(torch.randint(-8, 8, (output, inputs), dtype=torch.int8, generator=generator)),
                    "weight_scale": (
                        torch.rand(output, inputs // group_size, generator=generator) * 0.03 + 0.001
                    ).half(),
                    "weight_offset": torch.randint(-3, 4, (output, inputs // group_size), generator=generator).to(
                        torch.int8
                    ),
                }
                for kind, tensor in values.items():
                    yield f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.{kind}", tensor


def test_nibbles_all_values_and_group_zero_points():
    q = torch.arange(-8, 8, dtype=torch.int8).reshape(2, 8)
    torch.testing.assert_close(unpack_signed_int4(pack(q)), q)
    scale = torch.tensor([[0.1, 0.3], [0.2, 0.4]])
    offset = torch.tensor([[-3.0, 2.0], [1.0, -2.0]])
    expected = (q.reshape(2, 2, 4).float() - offset[..., None]) * scale[..., None]
    torch.testing.assert_close(dequantize(pack(q), scale, offset, 4), expected.flatten(-2))


def test_metadata_and_eager_gate_fail_closed():
    cfg = config()
    assert w4_config(SimpleNamespace()) is None
    require_eager_w4(SimpleNamespace(enforce_eager=False), SimpleNamespace())
    with pytest.raises(ValueError, match="enforce-eager"):
        require_eager_w4(SimpleNamespace(enforce_eager=False), cfg)
    require_eager_w4(SimpleNamespace(enforce_eager=True), cfg)
    for key, value in (
        ("bits", 8),
        ("packing", "different"),
        ("backend", "auto"),
        ("group_size", 3),
        ("group_size", 128),
    ):
        bad = config()
        bad.ascend_expert_quantization[key] = value
        with pytest.raises(ValueError):
            w4_config(bad)

    bad = config()
    bad.ascend_expert_quantization["shared_expert_execution"] = "automatic"
    with pytest.raises(ValueError, match="shared_expert_execution"):
        w4_config(bad)

    bad = config()
    bad.ascend_expert_quantization["lm_head_execution"] = "automatic"
    with pytest.raises(ValueError, match="lm_head_execution"):
        w4_config(bad)

    bad = config()
    bad.ascend_expert_quantization["lm_head_execution"] = "w8a8_dynamic"
    with pytest.raises(ValueError, match="native INT4 backend"):
        w4_config(bad)

    bad = config()
    bad.ascend_expert_quantization["ple_projection_execution"] = "automatic"
    with pytest.raises(ValueError, match="ple_projection_execution"):
        w4_config(bad)

    bad = config()
    bad.ascend_expert_quantization["mtp_expert_execution"] = "automatic"
    with pytest.raises(ValueError, match="mtp_expert_execution"):
        w4_config(bad)


@pytest.mark.parametrize(
    "mode,expected_width",
    [
        ("tp_sharded", 6),
        ("tp_sharded_overlap", 6),
        ("replicated", 24),
        ("replicated_overlap", 24),
        ("replicated_deferred", 24),
    ],
)
def test_shared_expert_execution_controls_resident_width(mode, expected_width):
    cfg = config(num_layers=1, num_experts=8, top_k=2, shared_inter=24)
    cfg.ascend_expert_quantization["shared_expert_execution"] = mode
    layer = W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy(), expert_sharding=(1, 4))
    assert layer.shared_expert_execution == mode
    assert layer.local_shared_inter == expected_width
    assert layer.shared_gate_up.shape == (2 * expected_width, cfg.hidden_size)
    assert layer.shared_down.shape == (cfg.hidden_size, expected_width)


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("tp_sharded", 10.0),
        ("tp_sharded_overlap", 10.0),
        ("replicated", 7.0),
        ("replicated_overlap", 7.0),
        ("replicated_deferred", 7.0),
    ],
)
def test_shared_expert_execution_preserves_reduction_semantics(mode, expected):
    cfg = config(num_layers=1, num_experts=4, top_k=1, shared_inter=8)
    cfg.ascend_expert_quantization["shared_expert_execution"] = mode
    layer = W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy(), expert_sharding=(0, 2))
    inputs = torch.zeros(2, cfg.hidden_size, dtype=torch.float16)
    routed = torch.full_like(inputs, 2.0, dtype=torch.float32)
    shared = torch.full_like(inputs, 3.0, dtype=torch.float32)
    with (
        patch("vllm_ascend.models.qwen4_exp.w4_moe.route_topk", return_value=(torch.ones(2, 1), torch.zeros(2, 1))),
        patch.object(layer, "_forward_host_routed", return_value=routed),
        patch.object(layer, "_forward_shared", return_value=shared) as shared_forward,
        patch.object(layer, "_tp_reduce", side_effect=lambda value: value * 2) as reduce,
    ):
        torch.testing.assert_close(layer(inputs), torch.full_like(inputs, expected))
    shared_forward.assert_called_once_with(inputs)
    reduce.assert_called_once()


def test_shared_expert_overlap_orders_auxiliary_stream_after_input():
    cfg = config(num_layers=1, num_experts=4, top_k=1, shared_inter=8)
    cfg.ascend_expert_quantization["shared_expert_execution"] = "replicated_overlap"
    layer = W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy(), expert_sharding=(0, 2))
    inputs, expected = MagicMock(), MagicMock()
    input_ready, shared_done = object(), object()
    main_stream, shared_stream = MagicMock(), MagicMock()
    main_stream.record_event.return_value = input_ready
    shared_stream.record_event.return_value = shared_done
    fake_utils = SimpleNamespace(
        current_stream=lambda: main_stream,
        npu_stream_switch=lambda stream: nullcontext(),
        shared_experts_calculation_stream=lambda: shared_stream,
    )
    fake_npu = SimpleNamespace(Stream=object)
    with (
        patch.dict(sys.modules, {"vllm_ascend.utils": fake_utils}),
        patch.object(torch, "npu", fake_npu, create=True),
        patch.object(layer, "_forward_shared", return_value=expected) as shared_forward,
    ):
        actual, returned_main, returned_done = layer._start_shared_overlap(inputs)
    assert actual is expected
    assert returned_main is main_stream
    assert returned_done is shared_done
    shared_stream.wait_event.assert_called_once_with(input_ready)
    inputs.record_stream.assert_called_once_with(shared_stream)
    expected.record_stream.assert_called_once_with(main_stream)
    shared_forward.assert_called_once_with(inputs)


@pytest.mark.parametrize(
    "tokens,expected",
    [
        (MAX_SHARED_EXPERT_OVERLAP_TOKENS, True),
        (MAX_SHARED_EXPERT_OVERLAP_TOKENS + 1, False),
        (4, False),
    ],
)
def test_tp_sharded_overlap_is_limited_to_decode_shapes_with_cube_headroom(tokens, expected):
    cfg = config(num_layers=1, num_experts=4, top_k=1, shared_inter=8)
    cfg.ascend_expert_quantization["shared_expert_execution"] = "tp_sharded_overlap"
    layer = W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy(), expert_sharding=(0, 2))
    inputs = SimpleNamespace(shape=(tokens, cfg.hidden_size), device=SimpleNamespace(type="npu"))
    assert layer._should_overlap_shared_expert(inputs) is expected


def test_tp_sharded_overlap_joins_shared_before_combined_reduce():
    cfg = config(num_layers=1, num_experts=4, top_k=1, shared_inter=8)
    cfg.hidden_size = cfg.moe_intermediate_size = 256
    cfg.ascend_expert_quantization.update(
        backend="cube_310_routed",
        group_size=128,
        shared_expert_execution="tp_sharded_overlap",
    )
    layer = W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy(), expert_sharding=(0, 2))
    inputs = MagicMock()
    inputs.shape = (1, cfg.hidden_size)
    inputs.device = SimpleNamespace(type="npu")
    routed = torch.full((1, cfg.hidden_size), 2.0, dtype=torch.float32)
    shared = torch.full_like(routed, 3.0)
    shared_done = object()
    call_order = []
    main_stream = MagicMock()
    main_stream.wait_event.side_effect = lambda event: call_order.append("wait")

    def start_shared(value):
        assert value is inputs
        call_order.append("shared")
        return shared, main_stream, shared_done

    def forward_routed(*args):
        call_order.append("routed")
        return routed

    def reduce(value):
        call_order.append("reduce")
        return value * 2

    with (
        patch("vllm_ascend.models.qwen4_exp.w4_moe.F.linear", return_value=torch.zeros(1, cfg.num_experts)),
        patch(
            "vllm_ascend.models.qwen4_exp.w4_moe.route_topk",
            return_value=(torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.long)),
        ),
        patch.object(layer, "_start_shared_overlap", side_effect=start_shared),
        patch.object(layer, "_forward_routed", side_effect=forward_routed),
        patch.object(layer, "_tp_reduce", side_effect=reduce),
    ):
        actual = layer(inputs)
    torch.testing.assert_close(actual, torch.full_like(actual, 10.0))
    assert call_order == ["shared", "routed", "wait", "reduce"]
    main_stream.wait_event.assert_called_once_with(shared_done)


def test_deferred_reduce_uses_events_and_records_cross_stream_tensors():
    cfg = config(num_layers=1, num_experts=4, top_k=1, shared_inter=8)
    cfg.ascend_expert_quantization["shared_expert_execution"] = "replicated_deferred"
    layer = W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy(), expert_sharding=(0, 2))
    routed, reduced = MagicMock(), MagicMock()
    routed_ready, reduce_done = object(), object()
    main_stream, reduce_stream = MagicMock(), MagicMock()
    main_stream.record_event.return_value = routed_ready
    reduce_stream.record_event.return_value = reduce_done
    fake_utils = SimpleNamespace(current_stream=lambda: main_stream, npu_stream_switch=lambda stream: nullcontext())
    stream_factory = MagicMock(return_value=reduce_stream)
    fake_npu = SimpleNamespace(Stream=stream_factory)
    with (
        patch.dict(sys.modules, {"vllm_ascend.utils": fake_utils}),
        patch.object(torch, "npu", fake_npu, create=True),
        patch.object(layer, "_tp_reduce", return_value=reduced) as reduce,
    ):
        actual, returned_main, returned_done = layer._start_deferred_reduce(routed)
    assert actual is reduced
    assert returned_main is main_stream
    assert returned_done is reduce_done
    stream_factory.assert_called_once_with()
    routed.record_stream.assert_called_once_with(reduce_stream)
    reduce_stream.wait_event.assert_called_once_with(routed_ready)
    reduce.assert_called_once_with(routed)
    reduced.record_stream.assert_called_once_with(main_stream)


def test_deferred_reduce_waits_after_shared_compute():
    cfg = config(num_layers=1, num_experts=4, top_k=1, shared_inter=8)
    cfg.hidden_size = cfg.moe_intermediate_size = 256
    cfg.ascend_expert_quantization.update(
        backend="cube_310_routed",
        group_size=128,
        shared_expert_execution="replicated_deferred",
    )
    layer = W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy(), expert_sharding=(0, 2))
    inputs = MagicMock()
    inputs.shape = (2, cfg.hidden_size)
    inputs.device = SimpleNamespace(type="npu")
    routed = torch.full((2, cfg.hidden_size), 2.0, dtype=torch.float32)
    shared = torch.full_like(routed, 3.0)
    reduced = routed * 2
    reduce_done = object()
    call_order = []
    main_stream = MagicMock()
    main_stream.wait_event.side_effect = lambda event: call_order.append("wait")

    def start_reduce(value):
        assert value is routed
        call_order.append("reduce")
        return reduced, main_stream, reduce_done

    def forward_shared(value):
        assert value is inputs
        call_order.append("shared")
        return shared

    with (
        patch("vllm_ascend.models.qwen4_exp.w4_moe.F.linear", return_value=torch.zeros(2, cfg.num_experts)),
        patch(
            "vllm_ascend.models.qwen4_exp.w4_moe.route_topk",
            return_value=(torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.long)),
        ),
        patch.object(layer, "_forward_routed", return_value=routed),
        patch.object(layer, "_start_deferred_reduce", side_effect=start_reduce),
        patch.object(layer, "_forward_shared", side_effect=forward_shared),
        patch.object(layer, "_tp_reduce", MagicMock()),
    ):
        actual = layer(inputs)
    torch.testing.assert_close(actual, torch.full_like(actual, 7.0))
    assert call_order == ["reduce", "shared", "wait"]
    main_stream.wait_event.assert_called_once_with(reduce_done)


@pytest.mark.parametrize(
    "mode,replicated",
    [("tp_sharded", False), ("tp_sharded_overlap", False), ("replicated", True)],
)
def test_shared_expert_loader_places_shard_or_full_replica(mode, replicated):
    prefix = "model.layers.0.mlp"
    hidden, shared, rank, tp_size = 8, 12, 2, 4
    local = shared if replicated else shared // tp_size
    params = {
        f"{prefix}.shared_gate_up": torch.zeros(2 * local, hidden, dtype=torch.float16),
        f"{prefix}.shared_down": torch.zeros(hidden, local, dtype=torch.float16),
    }
    gate = torch.arange(shared * hidden, dtype=torch.float16).reshape(shared, hidden)
    up = gate + 1000
    down = torch.arange(hidden * shared, dtype=torch.float16).reshape(hidden, shared) + 2000
    owner = SimpleNamespace(config=SimpleNamespace(ascend_expert_quantization={"shared_expert_execution": mode}))
    method = AscendQwen4ExpForCausalLM._place_shared_expert_tensor
    for projection, tensor in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
        loaded = method(
            owner,
            params,
            f"{prefix}.shared_expert.{projection}.weight",
            tensor,
            rank,
            tp_size,
        )
        assert loaded == f"{prefix}.{'shared_down' if projection == 'down_proj' else 'shared_gate_up'}"
    start = 0 if replicated else rank * local
    torch.testing.assert_close(params[f"{prefix}.shared_gate_up"][:local], gate[start : start + local])
    torch.testing.assert_close(params[f"{prefix}.shared_gate_up"][local:], up[start : start + local])
    torch.testing.assert_close(params[f"{prefix}.shared_down"], down[:, start : start + local])


def test_real_loader_chooses_w4_and_preserves_w8_default():
    cfg = config(num_layers=1)
    model = build(cfg)
    loaded = model.load_weights(payloads(cfg))
    layer = model.model.layers[0].mlp
    assert isinstance(layer, W4SparseMoE)
    assert not hasattr(layer, "w13_weight")
    assert len(loaded) == 9  # Three packed banks, scales and offsets, not FP16 shadows.
    for name, tensor in payloads(cfg):
        expert, projection, kind = name.split(".")[-3:]
        torch.testing.assert_close(getattr(layer.projections[projection], kind)[int(expert)], tensor)
    assert isinstance(build(_tiny_moe_config(num_layers=1)).model.layers[0].mlp, _EagerSparseMoE)


@pytest.mark.parametrize("backend", ["cube_310", "cube_310_tiled"])
def test_cube_backend_is_explicit_and_keeps_full_model_eager_gate(backend):
    cfg = config()
    cfg.hidden_size = 2560
    cfg.moe_intermediate_size = 640
    cfg.ascend_expert_quantization.update(group_size=128, backend=backend)
    assert w4_config(cfg)["backend"] == backend
    with pytest.raises(ValueError, match="enforce-eager"):
        require_eager_w4(SimpleNamespace(enforce_eager=False), cfg)
    cfg.ascend_expert_quantization["group_size"] = 64
    with pytest.raises(ValueError, match="group_size=128"):
        w4_config(cfg)
    cfg.ascend_expert_quantization["group_size"] = 128
    cfg.hidden_size = 4096
    with pytest.raises(ValueError, match="hidden_size"):
        w4_config(cfg)


def test_cube_backend_cannot_silently_use_cpu_reference():
    bank = PackedExpertBank(1, 256, 256, 128, backend="cube_310")
    with pytest.raises(ValueError, match="requires NPU"):
        bank.linear(torch.zeros(1, 256, dtype=torch.float16), 0)
    with pytest.raises(ValueError, match="unsupported"):
        PackedExpertBank(1, 256, 256, 128, backend="auto")


def test_only_device_routed_backend_allows_decode_graphs():
    cfg = config()
    cfg.hidden_size, cfg.moe_intermediate_size = 2560, 640
    cfg.ascend_expert_quantization.update(group_size=128, backend="cube_310_routed")
    require_eager_w4(SimpleNamespace(enforce_eager=False), cfg)
    bank = PackedExpertBank(1, 256, 256, 128, backend="cube_310_routed")
    with pytest.raises(ValueError, match="NPU"):
        bank.routed_linear(torch.zeros(1, 256).half(), torch.zeros(1, dtype=torch.int32))


def test_routed_slots_keep_token_order_and_nonzero_tp_expert_offset():
    cfg = config(num_layers=1, num_experts=7, top_k=3, shared_inter=0)
    cfg.hidden_size = cfg.moe_intermediate_size = 256
    cfg.ascend_expert_quantization.update(group_size=128, backend="cube_310_routed")
    layer = W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy(), expert_sharding=(1, 2))
    x = (torch.arange(512).reshape(2, 256) / 512).half()
    ids = torch.tensor([[0, 3, 6], [1, 4, 5]])
    weights = torch.tensor([[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]])
    local_ids = (ids - layer.expert_offset).flatten().to(torch.int32)

    def projection(inputs, selected):
        torch.testing.assert_close(selected, local_ids)
        owned = (selected >= 0) & (selected < layer.num_local_experts)
        factor = torch.where(owned, selected + 1, 0).to(inputs.dtype)
        return inputs * factor[:, None]

    expected = torch.zeros_like(x).float()
    for token in range(2):
        for slot in range(3):
            expert = int(ids[token, slot]) - layer.expert_offset
            if 0 <= expert < layer.num_local_experts:
                projected = (x[token] * (expert + 1)).float()
                activated = (F.silu(projected) * projected).half()
                expected[token] += (activated * (expert + 1)).float() * weights[token, slot]
    with (
        patch.object(
            layer.projections["gate_up_proj"],
            "routed_linear",
            side_effect=lambda inputs, selected: torch.cat([projection(inputs, selected)] * 2, dim=-1),
        ),
        patch.object(layer.projections["down_proj"], "routed_linear", side_effect=projection),
    ):
        torch.testing.assert_close(layer._forward_routed(x, weights, ids), expected)


def test_oversized_graph_fails_before_host_route_readback():
    cfg = config(num_layers=1, num_experts=7, top_k=3, shared_inter=0)
    cfg.hidden_size = cfg.moe_intermediate_size = 256
    cfg.ascend_expert_quantization.update(group_size=128, backend="cube_310_routed")
    layer = W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy())
    oversized_tokens = MAX_CUBE_ROUTES // layer.top_k + 1
    with (
        patch.object(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: True), create=True),
        patch.object(layer, "_forward_host_routed") as host,
    ):
        with pytest.raises(RuntimeError, match=f"exceeds {MAX_CUBE_ROUTES} routes"):
            layer(torch.zeros(oversized_tokens, 256).half())
        host.assert_not_called()


@pytest.mark.parametrize("outputs,inputs", [(640, 2560), (2560, 640)])
def test_tile_permutation_is_lossless_and_keeps_bytes(outputs, inputs):
    generator = torch.Generator().manual_seed(3132)
    weight = torch.randint(-128, 128, (outputs, inputs // 2), dtype=torch.int8, generator=generator)
    scale = torch.rand(outputs, inputs // 128, generator=generator).half()
    offset = torch.randint(-8, 8, scale.shape, dtype=torch.int8, generator=generator)
    for kind, tensor in [("weight", weight), ("weight_scale", scale), ("weight_offset", offset)]:
        packed = pack_cube_tiles(tensor, kind)
        assert packed.shape == tensor.shape and packed.dtype == tensor.dtype
        assert packed.numel() * packed.element_size() == tensor.numel() * tensor.element_size()
        assert packed.is_contiguous()
        if kind == "weight":
            encoded = packed.reshape(outputs // 32, inputs // 16, 16, 16)
            planes = torch.stack((encoded & 15, (encoded >> 4) & 15), -2)
            codes = planes.permute(0, 3, 4, 1, 2).reshape(outputs, inputs) - 8
            restored = (codes[:, 0::2] & 15) | ((codes[:, 1::2] & 15) << 4)
        else:
            restored = packed.reshape(outputs // 32, inputs // 128, 32).transpose(1, 2).reshape_as(tensor)
            if kind == "weight_offset":
                restored = restored - 8
        torch.testing.assert_close(restored, tensor, rtol=0, atol=0)


@pytest.mark.parametrize("outputs,inputs", [(32, 256), (640, 2560), (2560, 640)])
def test_concatenating_tiled_gate_up_equals_packing_concatenated_weights(outputs, inputs):
    generator = torch.Generator().manual_seed(1024)
    values = {
        "weight": torch.randint(-128, 128, (2, outputs, inputs // 2), dtype=torch.int8, generator=generator),
        "weight_scale": torch.rand(2, outputs, inputs // 128, generator=generator).half(),
        "weight_offset": torch.randint(-8, 8, (2, outputs, inputs // 128), dtype=torch.int8, generator=generator),
    }
    for kind, tensors in values.items():
        separate = torch.cat([pack_cube_tiles(tensor, kind) for tensor in tensors])
        combined = pack_cube_tiles(torch.cat(tuple(tensors)), kind)
        torch.testing.assert_close(separate, combined, rtol=0, atol=0)
        assert combined.nbytes == tensors.nbytes


@pytest.mark.parametrize("backend", ["cube_310_tiled", "cube_310_routed"])
def test_tiled_loader_permutes_each_expert_without_fp16_shadow(backend):
    cfg = config(num_layers=1, num_experts=3, top_k=1, shared_inter=0)
    cfg.hidden_size = cfg.moe_intermediate_size = 256
    cfg.ascend_expert_quantization.update(group_size=128, backend=backend)
    layer = W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy())
    values = {
        "weight": torch.arange(256 * 128).reshape(256, 128).to(torch.int8),
        "weight_scale": torch.full((256, 2), 0.25, dtype=torch.float16),
        "weight_offset": (torch.arange(512) % 16 - 8).reshape(256, 2).to(torch.int8),
    }
    for kind, tensor in values.items():
        layer.load_projection(2, "gate_proj", kind, tensor)
        bank = layer.projections["gate_up_proj" if backend == "cube_310_routed" else "gate_proj"]
        torch.testing.assert_close(getattr(bank, kind)[2, :256], pack_cube_tiles(tensor, kind), rtol=0, atol=0)
        assert not getattr(bank, kind)[0].any()
    assert len(list(bank.parameters())) == 3


def test_fused_gate_up_loader_preserves_layout_and_resident_bytes():
    cfg = config(num_layers=1, num_experts=7, top_k=3, shared_inter=0)
    cfg.hidden_size, cfg.moe_intermediate_size = 256, 384
    cfg.ascend_expert_quantization.update(group_size=128, backend="cube_310_routed")
    layer = W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy(), expert_sharding=(1, 2))
    generator = torch.Generator().manual_seed(13)
    for projection in ("up_proj", "gate_proj", "down_proj"):
        outputs, inputs = (256, 384) if projection == "down_proj" else (384, 256)
        tensors = (
            torch.randint(-128, 128, (outputs, inputs // 2), dtype=torch.int8, generator=generator),
            (torch.rand(outputs, inputs // 128, generator=generator) + 0.01).half(),
            torch.randint(-8, 8, (outputs, inputs // 128), dtype=torch.int8, generator=generator),
        )
        bank_name = "down_proj" if projection == "down_proj" else "gate_up_proj"
        start = 384 if projection == "up_proj" else 0
        for kind, tensor in zip(KINDS, tensors):
            assert layer.load_projection(0, projection, kind, tensor) is None  # peer
            assert layer.load_projection(4, projection, kind, tensor) == f"projections.{bank_name}.{kind}"
            stored = getattr(layer.projections[bank_name], kind)[4 - layer.expert_offset, start : start + outputs]
            torch.testing.assert_close(stored, pack_cube_tiles(tensor, kind), rtol=0, atol=0)
    assert set(layer.projections) == {"gate_up_proj", "down_proj"}
    elements = layer.num_local_experts * 3 * 256 * 384
    assert sum(parameter.nbytes for parameter in layer.projections.parameters()) == elements // 2 + elements // 128 * 3


def test_fused_gate_up_host_fallback_preserves_routes_and_rounding():
    cfg = config(num_layers=1, num_experts=3, top_k=2, shared_inter=0)
    cfg.hidden_size = cfg.moe_intermediate_size = 256
    cfg.ascend_expert_quantization.update(group_size=128, backend="cube_310_routed")
    layer = W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy())
    layer.device_routing = False  # diagnostic host routing must retain the fused layout
    x = (torch.arange(512).reshape(2, 256) / 512).half()
    ids, weights = torch.tensor([[0, 1], [1, 2]]), torch.tensor([[0.2, 0.8], [0.4, 0.6]])
    expected = torch.zeros_like(x).float()
    for token in range(2):
        for slot in range(2):
            factor = int(ids[token, slot]) + 1
            gate, up = (x[token] * factor).float(), (x[token] * (factor + 1)).float()
            expected[token] += ((F.silu(gate) * up).half() * factor).float() * weights[token, slot]
    with (
        patch.object(
            layer.projections["gate_up_proj"],
            "linear",
            side_effect=lambda inputs, expert: torch.cat((inputs * (expert + 1), inputs * (expert + 2)), -1),
        ),
        patch.object(
            layer.projections["down_proj"], "linear", side_effect=lambda inputs, expert: inputs * (expert + 1)
        ),
    ):
        torch.testing.assert_close(layer._forward_host_routed(x, weights, ids), expected, rtol=0, atol=0)


@pytest.mark.parametrize("missing_up", [False, True])
def test_real_loader_tracks_fused_parameters_but_requires_both_checkpoint_halves(missing_up):
    cfg = config(num_layers=1, num_experts=2, top_k=1, shared_inter=0)
    cfg.hidden_size = cfg.moe_intermediate_size = 256
    cfg.ascend_expert_quantization.update(group_size=128, backend="cube_310_routed")
    model = build(cfg)
    tensors = list(payloads(cfg))
    if missing_up:
        tensors = [(name, value) for name, value in tensors if ".up_proj." not in name]
        with pytest.raises(ValueError, match="incomplete W4 expert checkpoint"):
            model.load_weights(tensors)
    else:
        loaded = model.load_weights(tensors)
        assert loaded == {
            f"model.layers.0.mlp.projections.{projection}.{kind}"
            for projection in ("gate_up_proj", "down_proj")
            for kind in KINDS
        }
        assert loaded <= dict(model.named_parameters()).keys()


@pytest.mark.parametrize(
    "fault", ["missing", "empty", "duplicate", "dtype", "shape", "nan", "negative_scale", "offset", "unknown"]
)
def test_loader_rejects_malformed_checkpoint(fault):
    cfg = config(num_layers=1, num_experts=2, top_k=1)
    values = list(payloads(cfg))
    if fault == "missing":
        values.pop()
    elif fault == "empty":
        values = []
    elif fault == "duplicate":
        values.append(values[0])
    elif fault == "dtype":
        values[0] = values[0][0], values[0][1].float()
    elif fault == "shape":
        values[0] = values[0][0], values[0][1][:-1]
    elif fault == "nan":
        values[1][1].fill_(float("nan"))
    elif fault == "negative_scale":
        values[1][1].fill_(-1)
    elif fault == "offset":
        values[2][1].fill_(8)
    elif fault == "unknown":
        values[0] = values[0][0].replace(".weight", ".weight_packed"), values[0][1]
    with pytest.raises(ValueError):
        build(cfg).load_weights(values)


def test_forward_matches_independent_eager_reference_and_uneven_tp():
    torch.manual_seed(77)
    cfg = config(num_layers=1, num_experts=7, top_k=3, shared_inter=24)
    full = build(cfg).model.layers[0].mlp
    for name, tensor in payloads(cfg):
        expert, projection, kind = name.split(".")[-3:]
        full.load_projection(int(expert), projection, kind, tensor)
    for name in ("gate", "shared_gate_up", "shared_down", "shared_expert_gate"):
        getattr(full, name).data.normal_(std=0.1)
    inputs = torch.randn(5, cfg.hidden_size).half()
    weights, ids = route_topk(F.linear(inputs, full.gate), 3, renormalize=True)
    expected = torch.zeros_like(inputs).float()
    for token in range(len(inputs)):
        for slot, expert in enumerate(ids[token].tolist()):
            matrices = {}
            for name, bank in full.projections.items():
                packed = bank.weight[expert].long()
                lo, hi = packed & 15, (packed >> 4) & 15
                integers = torch.stack([lo, hi], -1).reshape(packed.shape[0], -1)
                integers = ((integers + 8) % 16 - 8).float()
                matrices[name] = (
                    (
                        (integers.reshape(*bank.weight_scale[expert].shape, 8) - bank.weight_offset[expert][..., None])
                        * bank.weight_scale[expert][..., None]
                    )
                    .flatten(-2)
                    .half()
                )
            gate = F.linear(inputs[token], matrices["gate_proj"]).float()
            up = F.linear(inputs[token], matrices["up_proj"]).float()
            expected[token] += (
                F.linear((F.silu(gate) * up).half(), matrices["down_proj"]).float() * weights[token, slot]
            )
    gate, up = F.linear(inputs.float(), full.shared_gate_up.float()).chunk(2, -1)
    expected += F.linear(F.silu(gate) * up, full.shared_down.float()) * torch.sigmoid(
        F.linear(inputs.float(), full.shared_expert_gate.float())
    )
    torch.testing.assert_close(full(inputs), expected.half(), atol=0.001, rtol=0.002)
    partials = []
    for rank in range(3):
        part = W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy(), expert_sharding=(rank, 3))
        for name, tensor in payloads(cfg):
            expert, projection, kind = name.split(".")[-3:]
            part.load_projection(int(expert), projection, kind, tensor)
        width, start = part.local_shared_inter, rank * part.local_shared_inter
        with torch.no_grad():
            part.gate.copy_(full.gate)
            part.shared_expert_gate.copy_(full.shared_expert_gate)
            gate, up = full.shared_gate_up.chunk(2, 0)
            part.shared_gate_up.copy_(torch.cat((gate[start : start + width], up[start : start + width])))
            part.shared_down.copy_(full.shared_down[:, start : start + width])
        with patch.object(part, "_tp_reduce", side_effect=lambda x: x) as reduce:
            partials.append(part(inputs).float())
            reduce.assert_called_once()
    torch.testing.assert_close(sum(partials).half(), full(inputs), atol=0.001, rtol=0.003)


def test_packed_resident_bytes():
    cfg = config(num_layers=1)
    layer = W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy())
    total = sum(param.numel() * param.element_size() for param in layer.projections.parameters())
    elements = 3 * cfg.num_experts * cfg.hidden_size * cfg.moe_intermediate_size
    assert total == elements // 2 + elements // 8 * 3  # packed bytes + FP16 scale and INT8 zero point


def test_modelslim_export_loads_into_real_model(tmp_path):
    pytest.importorskip("msmodelslim")
    from safetensors import safe_open
    from safetensors.torch import save_file

    from tests.ut.qwen38_1m.test_w4_export import export

    cfg = _tiny_moe_config(num_layers=1, num_experts=2, top_k=1)
    source = tmp_path / "original"
    source.mkdir()
    export.write_json(source / "config.json", {"model_type": "qwen4_exp", "text_config": vars(cfg)})
    tensors = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 32, 32).bfloat16(),
        "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(2, 32, 16).bfloat16(),
    }
    save_file(tensors, str(source / "model.safetensors"))
    export.write_json(
        source / "model.safetensors.index.json", {"weight_map": dict.fromkeys(tensors, "model.safetensors")}
    )
    dest = tmp_path / "quantized"
    export.convert(source, dest, group_size=8)
    quant_config = SimpleNamespace(**json.loads((dest / "config.json").read_text())["text_config"])
    model = build(quant_config)

    def weights():
        for shard in dest.glob("*.safetensors"):
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for name in list(handle.keys()):
                    yield name, handle.get_tensor(name)

    assert len(model.load_weights(weights())) == 9
    with torch.inference_mode():
        result = model.model.layers[0].mlp(torch.randn(3, 32).half())
    assert result.shape == (3, 32)
    assert torch.isfinite(result).all()
