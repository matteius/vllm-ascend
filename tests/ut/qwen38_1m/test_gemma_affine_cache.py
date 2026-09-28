# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Gemma affine reuse keeps the original normalization math."""

import torch

from vllm_ascend.models.qwen4_exp.model import _cached_gemma_affine, _GatedResidual, _grouped_rms_norm


def test_cached_affine_matches_uncached_grouped_norm_bitwise():
    generator = torch.Generator().manual_seed(1024)
    values = torch.randn(3, 32, generator=generator).half()
    weight = torch.randn(32, generator=generator).half() * 0.1
    cache = _cached_gemma_affine(weight, torch.float32, None)
    assert _cached_gemma_affine(weight, torch.float32, cache) is cache
    expected = _grouped_rms_norm(values, weight, 1e-6, 8, torch.float32)
    actual = _grouped_rms_norm(values, weight, 1e-6, 8, torch.float32, affine=cache[1])
    assert torch.equal(actual, expected)

    with torch.no_grad():
        weight.add_(0.25)
    updated = _cached_gemma_affine(weight, torch.float32, cache)
    assert updated is not cache
    assert torch.equal(updated[1], 1.0 + weight.float())
    assert torch.equal(
        _grouped_rms_norm(values, weight, 1e-6, 8, torch.float32, affine=updated[1]),
        _grouped_rms_norm(values, weight, 1e-6, 8, torch.float32),
    )


def test_gated_residual_refreshes_affine_and_preserves_autograd():
    module = _GatedResidual(
        hc_count=2,
        hidden_size=8,
        lowrank=4,
        eps=1e-6,
        params_dtype=torch.float16,
        compute_dtype=torch.float32,
    )
    values = torch.randn(2, 16, generator=torch.Generator().manual_seed(17)).half()
    with torch.no_grad():
        module.hc_norm_weight.copy_(torch.linspace(-0.25, 0.25, 16).half())
        module.prepare_norm_affine()
        first_cache = module._hc_norm_affine_cache
        assert torch.equal(
            module._normalize(values), _grouped_rms_norm(values, module.hc_norm_weight, 1e-6, 8, torch.float32)
        )
        module.hc_norm_weight.add_(0.125)
        assert torch.equal(
            module._normalize(values), _grouped_rms_norm(values, module.hc_norm_weight, 1e-6, 8, torch.float32)
        )
        assert module._hc_norm_affine_cache is not first_cache

    module.hc_norm_weight.grad = None
    module._normalize(values).sum().backward()
    assert module.hc_norm_weight.grad is not None
    assert torch.isfinite(module.hc_norm_weight.grad).all()
