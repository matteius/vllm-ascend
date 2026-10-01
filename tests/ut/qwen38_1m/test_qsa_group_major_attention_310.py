# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.models.qwen4_exp.ops.qsa_group_major_attention_310 import (
    _group_major_attention,
    build_qsa_group_major_plan,
    qsa_group_major_prefill_310,
)
from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import QSAGroupSelection


def _selection(
    groups: list[list[int]],
    counts: list[int],
    tail_starts: list[int],
    tail_counts: list[int],
) -> QSAGroupSelection:
    return QSAGroupSelection(
        group_indices=torch.tensor(groups, dtype=torch.int32),
        group_counts=torch.tensor(counts, dtype=torch.int32),
        tail_starts=torch.tensor(tail_starts, dtype=torch.int32),
        tail_counts=torch.tensor(tail_counts, dtype=torch.int32),
    )


def test_group_major_plan_deduplicates_groups_and_preserves_query_masks() -> None:
    selection = _selection(
        groups=[[1, 3, 5], [3, 4, 5]],
        counts=[3, 3],
        tail_starts=[24, 28],
        tail_counts=[2, 1],
    )

    plan = build_qsa_group_major_plan(selection)

    torch.testing.assert_close(plan.group_indices, torch.tensor([1, 3, 4, 5, 6, 7], dtype=torch.int32))
    assert plan.unique_group_reads == 6
    mask = plan.token_mask.view(2, 6, 4)
    expected = torch.zeros_like(mask)
    expected[0, [0, 1, 3]] = True
    expected[0, 4, :2] = True
    expected[1, [1, 2, 3]] = True
    expected[1, 5, :1] = True
    assert torch.equal(mask, expected)


def test_group_major_plan_ignores_fixed_width_padding() -> None:
    selection = _selection(
        groups=[[2, 7, -1], [7, -1, -1]],
        counts=[2, 1],
        tail_starts=[32, 0],
        tail_counts=[3, 0],
    )

    plan = build_qsa_group_major_plan(selection)

    torch.testing.assert_close(plan.group_indices, torch.tensor([2, 7, 8], dtype=torch.int32))
    mask = plan.token_mask.view(2, 3, 4)
    assert mask[0, 0].all()
    assert mask[0, 1].all()
    assert torch.equal(mask[0, 2], torch.tensor([True, True, True, False]))
    assert not mask[1, 0].any()
    assert mask[1, 1].all()
    assert not mask[1, 2].any()


def test_group_major_attention_matches_independent_selected_attention() -> None:
    torch.manual_seed(951)
    selection = _selection(
        groups=[[0, 2, 4], [1, 2, 4]],
        counts=[3, 3],
        tail_starts=[20, 24],
        tail_counts=[2, 3],
    )
    plan = build_qsa_group_major_plan(selection)
    num_queries = 2
    num_query_heads = 4
    num_kv_heads = 2
    head_dim = 16
    dense_tokens = 28
    scale = head_dim**-0.5
    query = torch.randn(num_queries, num_query_heads, head_dim, dtype=torch.float16)
    dense_key = torch.randn(dense_tokens, num_kv_heads, head_dim, dtype=torch.float16)
    dense_value = torch.randn_like(dense_key)
    union_tokens = (plan.group_indices.to(torch.int64).unsqueeze(-1) * 4 + torch.arange(4, dtype=torch.int64)).reshape(
        -1
    )
    selected_keys = dense_key[union_tokens].permute(1, 2, 0).unsqueeze(0)
    selected_values = dense_value[union_tokens].permute(1, 0, 2).unsqueeze(0)

    actual = _group_major_attention(
        query,
        selected_keys,
        selected_values,
        plan.token_mask,
        scale=scale,
    )

    expected = torch.empty_like(actual)
    heads_per_kv_head = num_query_heads // num_kv_heads
    for row in range(num_queries):
        group_count = int(selection.group_counts[row])
        groups = selection.group_indices[row, :group_count].to(torch.int64)
        token_ids = (groups.unsqueeze(-1) * 4 + torch.arange(4)).reshape(-1)
        tail_count = int(selection.tail_counts[row])
        token_ids = torch.cat((token_ids, selection.tail_starts[row] + torch.arange(tail_count)))
        for query_head in range(num_query_heads):
            kv_head = query_head // heads_per_kv_head
            logits = dense_key[token_ids, kv_head].float() @ query[row, query_head].float() * scale
            expected[row, query_head] = (torch.softmax(logits, dim=0) @ dense_value[token_ids, kv_head].float()).half()

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_group_major_plan_rejects_empty_selection() -> None:
    selection = _selection(groups=[[-1]], counts=[0], tail_starts=[0], tail_counts=[0])
    with pytest.raises(ValueError, match="at least one"):
        build_qsa_group_major_plan(selection)


def test_group_major_native_entry_point_rejects_cpu() -> None:
    query = torch.zeros((1, 2, 16), dtype=torch.float16)
    cache = torch.zeros((1, 2, 4, 16), dtype=torch.float16)
    selection = _selection(groups=[[0]], counts=[1], tail_starts=[4], tail_counts=[0])
    with pytest.raises(RuntimeError, match="Ascend NPU"):
        qsa_group_major_prefill_310(
            query,
            cache,
            cache,
            selection,
            torch.zeros((1, 1), dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            scale=0.25,
        )
