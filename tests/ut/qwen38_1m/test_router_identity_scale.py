# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact identity elimination; softmax, top-k ties and reduction order stay fixed."""

from unittest.mock import patch

import pytest
import torch

from tools.qwen4exp.profile_dispatch_reuse_310 import make_cases
from vllm_ascend.models.qwen4_exp.moe import route_topk


@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("scale", [0.0, 0.75, 1.0, 2.0])
@pytest.mark.parametrize("tied", [False, True])
def test_router_identity_preserves_existing_arithmetic(renormalize, scale, tied):
    logits = torch.randn(6, 256, generator=torch.Generator().manual_seed(1024)).half()
    if tied:
        logits.zero_()
    expected, expected_ids = logits.float().softmax(-1).topk(10, dim=-1)
    if renormalize:
        expected = expected / expected.sum(-1, keepdim=True)
    expected = expected * scale
    actual, ids = route_topk(logits, 10, renormalize=renormalize, routed_scaling_factor=scale)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)


def test_unit_scale_launches_no_multiplication():
    logits = torch.randn(3, 256)
    with patch.object(torch.Tensor, "__mul__", side_effect=AssertionError("redundant unit-scale multiply")):
        weights, ids = route_topk(logits, 10, routed_scaling_factor=1.0)
    assert weights.shape == ids.shape == (3, 10)


def test_dispatch_benchmark_compares_identical_operations():
    cases = make_cases(torch.tensor([3, 0, 2, 1]), torch.zeros(3, 512))
    for name, function in cases.items():
        # CPU index_copy requires INT64 indices; the all-INT32 alternative
        # is a 310P-only profiling case checked by the hardware diagnostic.
        if name.startswith("inverse_") and name != "inverse_index_copy_all_int32":
            torch.testing.assert_close(cases["inverse_sort"](), function(), rtol=0, atol=0)
    torch.testing.assert_close(cases["router_identity_multiply"](), cases["router_without_identity"](), rtol=0, atol=0)
