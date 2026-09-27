# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed W4, real loader, TP partial sums, and W8 isolation (CPU only)."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from tests.ut.qwen38_1m.test_model_load_and_moe import _single_rank_tp, _tiny_moe_config, _vllm_config
from vllm_ascend.models.qwen4_exp.dtype_policy import Qwen4ExpDtypePolicy
from vllm_ascend.models.qwen4_exp.model import AscendQwen4ExpForCausalLM, _EagerSparseMoE
from vllm_ascend.models.qwen4_exp.moe import route_topk
from vllm_ascend.models.qwen4_exp.w4_moe import (
    FORMAT,
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
                    "weight_scale": (torch.rand(output, inputs // 8, generator=generator) * 0.03 + 0.001).half(),
                    "weight_offset": torch.randint(-3, 4, (output, inputs // 8), generator=generator).to(torch.int8),
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
        patch.object(layer.projections["gate_proj"], "routed_linear", side_effect=projection),
        patch.object(layer.projections["up_proj"], "routed_linear", side_effect=projection),
        patch.object(layer.projections["down_proj"], "routed_linear", side_effect=projection),
    ):
        torch.testing.assert_close(layer._forward_routed(x, weights, ids), expected)


def test_oversized_graph_fails_before_host_route_readback():
    cfg = config(num_layers=1, num_experts=7, top_k=3, shared_inter=0)
    cfg.hidden_size = cfg.moe_intermediate_size = 256
    cfg.ascend_expert_quantization.update(group_size=128, backend="cube_310_routed")
    layer = W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy())
    with (
        patch.object(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: True), create=True),
        patch.object(layer, "_forward_host_routed") as host,
    ):
        with pytest.raises(RuntimeError, match="exceeds 80 routes"):
            layer(torch.zeros(27, 256).half())
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
        bank = layer.projections["gate_proj"]
        torch.testing.assert_close(getattr(bank, kind)[2], pack_cube_tiles(tensor, kind), rtol=0, atol=0)
        assert not getattr(bank, kind)[0].any()
    assert len(list(layer.projections["gate_proj"].parameters())) == 3


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
