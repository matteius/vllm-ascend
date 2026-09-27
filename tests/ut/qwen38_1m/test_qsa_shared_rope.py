# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision and ownership gates for per-forward shared QSA query tables."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.qwen4_exp.qsa import (
    AscendQwen4ExpQSAAttention,
    _mrope_interleaved_dims,
    apply_partial_rope,
    partial_rope_cos_sin,
)


@pytest.mark.parametrize("accum_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("position_kind", ["text", "mrope", "mrope_first_axis"])
@pytest.mark.parametrize("start", [0, 23400, 163840, 1048576, 16777217])
def test_shared_tables_match_independent_query_rotations(accum_dtype, position_kind, start):
    positions = torch.arange(start, start + 5, dtype=torch.int64)
    kwargs = {}
    if position_kind != "text":
        positions = torch.stack((positions, positions.flip(0) * 2, positions % 3))
        if position_kind == "mrope":
            kwargs = {"mrope_section": [2, 1, 1], "mrope_interleaved": True}
    generator = torch.Generator().manual_seed(1024)
    tables = partial_rope_cos_sin(
        positions, rotary_dim=8, base=10000.0, dtype=accum_dtype, compute_dtype=accum_dtype, **kwargs
    )
    axes = torch.tensor(_mrope_interleaved_dims([2, 1, 1]), dtype=torch.int64)
    buffered_tables = partial_rope_cos_sin(
        positions,
        rotary_dim=8,
        base=10000.0,
        dtype=accum_dtype,
        compute_dtype=accum_dtype,
        frequency_axes=axes,
        **kwargs,
    )
    for reference, buffered in zip(tables, buffered_tables):
        torch.testing.assert_close(buffered, reference, rtol=0, atol=0)
    for heads, width in ((6, 32), (1, 32), (4, 16)):
        value = torch.randn(5, heads, width, generator=generator).to(accum_dtype)
        expected = apply_partial_rope(value, positions, 8, 10000.0, accum_dtype, **kwargs)
        actual = apply_partial_rope(value, positions, 8, 10000.0, accum_dtype, cos_sin=tables, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual[..., 8:], value[..., 8:], rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["half_precision", "wrong_tokens", "wrong_rotary_dim", "wrong_device"])
def test_shared_tables_reject_incompatible_storage(bad):
    positions = torch.arange(3)
    tables = partial_rope_cos_sin(positions, rotary_dim=8, base=10000.0, dtype=torch.float32)
    if bad == "half_precision":
        tables = tuple(table.half() for table in tables)
    elif bad == "wrong_tokens":
        tables = tuple(table[:1] for table in tables)
    elif bad == "wrong_rotary_dim":
        tables = tuple(table[:, :4] for table in tables)
    elif bad == "wrong_device":
        tables = tuple(table.to("meta") for table in tables)
    with pytest.raises(ValueError, match="shared RoPE tables must match"):
        apply_partial_rope(torch.ones(3, 2, 16), positions, 8, 10000.0, torch.float32, cos_sin=tables)


@pytest.mark.parametrize("bad", ["wrong_size", "wrong_dtype", "wrong_device"])
def test_mrope_tables_reject_incompatible_axis_map(bad):
    axes = torch.tensor(_mrope_interleaved_dims([2, 1, 1]), dtype=torch.int64)
    if bad == "wrong_size":
        axes = axes[:1]
    elif bad == "wrong_dtype":
        axes = axes.float()
    else:
        axes = axes.to("meta")
    with pytest.raises(ValueError, match="MRoPE frequency axes must"):
        partial_rope_cos_sin(
            torch.arange(3).expand(3, -1),
            rotary_dim=8,
            base=10000.0,
            dtype=torch.float32,
            mrope_section=[2, 1, 1],
            mrope_interleaved=True,
            frequency_axes=axes,
        )


def test_project_qk_shared_tables_preserve_norm_order_and_reject_distinct_positions():
    config = SimpleNamespace(
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=0.5,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    )
    module = AscendQwen4ExpQSAAttention(config=config, layer_idx=0)
    generator = torch.Generator().manual_seed(1024)
    query = torch.randn(3, 4, 16, generator=generator).half()
    key = torch.randn(3, 2, 16, generator=generator).half()
    positions = torch.tensor([128, 163841, 1048575])
    tables = partial_rope_cos_sin(positions, rotary_dim=8, base=10000.0, dtype=torch.float32)
    with torch.no_grad():
        module.q_norm_weight.copy_(torch.randn(16, generator=generator) * 0.1)
        module.k_norm_weight.copy_(torch.randn(16, generator=generator) * 0.1)
    expected = module.project_qk(query, key, positions, positions, accum_dtype=torch.float32)
    actual = module.project_qk(query, key, positions, positions, accum_dtype=torch.float32, cos_sin=tables)
    for before, after in zip(expected, actual):
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    with pytest.raises(ValueError, match="same positions tensor"):
        module.project_qk(query, key, positions, positions + 1, accum_dtype=torch.float32, cos_sin=tables)
