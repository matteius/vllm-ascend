# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU routing/decomposition tests; these do NOT validate device instructions."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from tests.ut.qwen38_1m.test_w4_moe import config
from vllm_ascend.models.qwen4_exp.dtype_policy import Qwen4ExpDtypePolicy
from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE, require_eager_w4, unpack_signed_int4
from vllm_ascend.models.qwen4_exp.w4a8_int4 import (
    NATIVE_INT4_BACKEND,
    pack_float_nibbles,
    pack_native_metadata,
    pack_native_weight,
    pack_nibbles,
    quantize_activation_limbs,
)


def make_layer(backend="cube_310_grouped"):
    cfg = config(num_layers=1, num_experts=7, top_k=3, shared_inter=0)
    cfg.hidden_size = cfg.moe_intermediate_size = 256
    cfg.ascend_expert_quantization.update(group_size=128, backend=backend)
    require_eager_w4(SimpleNamespace(enforce_eager=False), cfg)
    return W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy(), expert_sharding=(1, 2))


@pytest.mark.parametrize("tokens", [1, 8, 27, 128, 513])
@pytest.mark.parametrize("peers_only", [False, True])
def test_grouped_routes_match_slot_reference_without_host_readback(tokens, peers_only):
    layer = make_layer()
    x = torch.rand(tokens, 256).half() * 0.1
    ids = torch.arange(tokens * 3).reshape(tokens, 3) % (3 if peers_only else 7)
    weights = torch.tensor([0.125, 0.375, 0.5]).expand(tokens, -1)
    local = ids - layer.expert_offset
    factors = torch.where((local >= 0) & (local < layer.num_local_experts), local + 1, 0).half()
    projected = (x[:, None, :] * factors[..., None]).float()
    activation = (F.silu(projected) * projected).half()
    expected = ((activation * factors[..., None]).float() * weights[..., None]).sum(1)

    def projection(inputs, ends, fused=False):
        expert = (torch.arange(inputs.shape[0])[:, None] >= ends[None, :]).sum(1)
        factor = torch.where(expert < layer.num_local_experts, expert + 1, 0).half()
        out = inputs * factor[:, None]
        return torch.cat([out, out], -1) if fused else out

    with (
        patch.object(
            layer.projections["gate_up_proj"], "grouped_linear", side_effect=lambda x, e: projection(x, e, True)
        ),
        patch.object(layer.projections["down_proj"], "grouped_linear", side_effect=projection),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("routing CPU copy")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("routing Python list")),
        patch.object(torch.Tensor, "item", side_effect=AssertionError("routing scalar sync")),
    ):
        actual = layer._forward_grouped(x, weights, ids)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("backend", ["cube_310_grouped", NATIVE_INT4_BACKEND])
@pytest.mark.parametrize("tokens", [1, 8, 9, 128])
def test_backend_dispatch_never_falls_back_to_python(backend, tokens):
    layer = make_layer(backend)
    expected = torch.zeros(tokens, 256).float()
    with (
        patch.object(layer, "_tp_reduce", side_effect=lambda x: x),
        patch.object(layer, "_forward_host_routed", side_effect=AssertionError("host fallback")),
        patch.object(layer, "_forward_grouped", return_value=expected) as grouped,
        patch.object(layer, "_forward_routed", return_value=expected) as routed,
    ):
        layer(torch.zeros(tokens, 256).half())
    uses_native = backend == NATIVE_INT4_BACKEND
    assert grouped.call_count == int(uses_native or tokens * 3 > 80)
    assert routed.call_count == int(not uses_native and tokens * 3 <= 80)


def test_two_int4_limb_identity_covers_every_int8():
    q = torch.arange(-128, 128, dtype=torch.int16)
    low = unpack_signed_int4(pack_nibbles((q & 15) - 8)).int()
    high = unpack_signed_int4(pack_nibbles(q >> 4)).int()
    torch.testing.assert_close(low + 16 * high + 8, q.int(), rtol=0, atol=0)
    # All signed weight codes and all asymmetric zero points.
    for zero in range(-8, 8):
        weight = (torch.arange(256) % 16 - 8).int()
        exact = (q.int() * (weight - zero)).sum()
        decomposed = (low * weight).sum() + 16 * (high * weight).sum() + 8 * weight.sum() - zero * q.int().sum()
        assert decomposed == exact


@pytest.mark.parametrize("kind", ["zeros", "random", "outlier", "subnormal"])
def test_activation_quantization_contract(kind):
    x = torch.randn(7, 256).half()
    if kind == "zeros":
        x.zero_()
    elif kind == "outlier":
        x[:, 0] = 60000
    elif kind == "subnormal":
        x.mul_(0.000001)
    low, high, scale, total = quantize_activation_limbs(x)
    q = (unpack_signed_int4(low).int() + 16 * unpack_signed_int4(high).int() + 8).reshape(7, 2, 128)
    expected = (x.float().reshape(7, 2, 128) / scale[..., None]).round().clamp(-127, 127).int()
    torch.testing.assert_close(q, expected, rtol=0, atol=0)
    torch.testing.assert_close(total, q.sum(-1).float(), rtol=0, atol=0)
    assert torch.isfinite(scale).all() and (scale > 0).all()
    error = (x.float().reshape_as(q) - q * scale[..., None]).abs()
    assert (error <= scale[..., None] * 0.501).all()


def test_native_layout_roundtrip_and_weight_sum():
    packed = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8).repeat(128, 1)
    native, sums = pack_native_weight(packed)
    unpacked = native.reshape(8, 4, 2, 16, 32).permute(0, 3, 1, 2, 4).contiguous().view_as(packed)
    torch.testing.assert_close(unpacked, packed, rtol=0, atol=0)
    expected = unpack_signed_int4(packed).reshape(128, 4, 128).sum(-1).half()
    torch.testing.assert_close(sums, pack_native_metadata(expected), rtol=0, atol=0)


def test_float_packing_exact_for_every_nibble_pair():
    low, high = torch.meshgrid(torch.arange(-8, 8), torch.arange(-8, 8), indexing="ij")
    values = torch.stack((low.flatten(), high.flatten()), -1)
    torch.testing.assert_close(pack_float_nibbles(values.float()), pack_nibbles(values), rtol=0, atol=0)


def test_native_loader_builds_sums_and_accepts_original_offset_dtype():
    layer = make_layer(NATIVE_INT4_BACKEND)
    packed = torch.randint(-128, 128, (256, 128), dtype=torch.int8)
    scale = torch.rand(256, 2).half() + 0.01
    offset = torch.randint(-8, 8, (256, 2), dtype=torch.int8)
    for name, value in [("weight", packed), ("weight_scale", scale), ("weight_offset", offset)]:
        layer.load_projection(layer.expert_offset, "up_proj", name, value)
    bank = layer.projections["gate_up_proj"]
    native, sums = pack_native_weight(packed)
    torch.testing.assert_close(bank.weight[0, 256:], native)
    torch.testing.assert_close(bank.weight_sum[0, 256:], sums)
    torch.testing.assert_close(bank.weight_offset[0, 256:], pack_native_metadata(offset).half())
    torch.testing.assert_close(bank.weight_scale[0, 256:], pack_native_metadata(scale))
    with pytest.raises(RuntimeError, match="no Python expert fallback"):
        bank.linear(torch.zeros(1, 256).half(), 0)
