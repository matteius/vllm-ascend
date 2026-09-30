# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for GLM host MLA page identity and sparse remapping."""

import numpy as np
import pytest
import torch

from vllm_ascend.models.glm5next.host_kv import (
    GlmHostKVLayer,
    remap_selected_pages,
    selected_page_columns,
)


def test_selected_pages_cover_dense_sparse_and_tail() -> None:
    groups = np.array([[0, 8, -1], [0, 24, -1]], dtype=np.int32)
    columns = selected_page_columns(
        groups,
        np.array([33, 2]),
        np.array([0, 96]),
        np.array([-1, 1]),
        pool_size=4,
    )
    assert [c.tolist() for c in columns] == [[0, 1], [0, 3]]


def test_remap_selected_pages_deduplicates_across_requests() -> None:
    table = np.array([[8, 3, 5, 7], [3, 9, 8, 10]], dtype=np.int32)
    pages, remapped = remap_selected_pages(
        table,
        [np.array([0, 1]), np.array([0, 2])],
        host_num_pages=11,
        hot_num_pages=3,
    )
    assert pages.tolist() == [3, 8]
    assert remapped[0, :2].tolist() == [1, 0]
    assert remapped[1, [0, 2]].tolist() == [0, 1]
    with pytest.raises(ValueError, match="only 1"):
        remap_selected_pages(table, [np.array([0, 1]), np.array([0, 2])], 11, 1)


def test_host_write_then_stage_exact_latent_rows() -> None:
    hot = torch.zeros((4, 32, 32, 16), dtype=torch.float16)
    layer = GlmHostKVLayer(logical_blocks=2, hot_cache=hot, block_size=64, pin_memory=False)
    slots = torch.tensor([0, 31, 32, 63, 64, 127], dtype=torch.int32)
    rows = torch.arange(len(slots), dtype=torch.float16)[:, None, None].expand(-1, 32, 16).clone()
    layer.write(rows, slots)
    table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    remapped = layer.stage(
        table,
        torch.tensor([[0, 8]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([64], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        pool_size=4,
    )
    assert remapped[0, [0, 1, 2]].tolist() == [0, 1, 2]
    assert hot[0, :, 0, :].eq(0).all()
    assert hot[0, :, 31, :].eq(1).all()
    assert hot[1, :, 0, :].eq(2).all()
    assert hot[2, :, 0, :].eq(4).all()


def test_reused_scheduler_page_overwrites_old_rows() -> None:
    hot = torch.empty((2, 32, 32, 16), dtype=torch.float16)
    layer = GlmHostKVLayer(logical_blocks=1, hot_cache=hot, block_size=64, pin_memory=False)
    slots = torch.arange(32, dtype=torch.int32)
    layer.write(torch.ones((32, 32, 16), dtype=torch.float16), slots)
    layer.write(torch.full((32, 32, 16), 7, dtype=torch.float16), slots)
    layer.stage(
        torch.tensor([[0, 1]], dtype=torch.int32),
        torch.zeros((1, 1), dtype=torch.int32),
        torch.tensor([32], dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.full((1,), -1, dtype=torch.int32),
        pool_size=4,
    )
    assert hot[0].eq(7).all()


def test_prefill_batches_queries_until_selected_pages_exceed_hot_capacity() -> None:
    hot = torch.empty((2, 32, 32, 16), dtype=torch.float16)
    layer = GlmHostKVLayer(logical_blocks=2, hot_cache=hot, block_size=64, pin_memory=False)
    table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    groups = torch.tensor([[0], [0], [16]], dtype=torch.int32)
    counts = torch.tensor([31, 35, 1], dtype=torch.int32)
    tails = torch.tensor([-1, -1, 0], dtype=torch.int32)
    starts = torch.zeros(3, dtype=torch.int32)
    segments = layer.prefill_segments(table, groups, counts, starts, tails, [3], pool_size=4)
    assert segments == [(0, 2, 0), (2, 3, 0)]
    remapped = layer.stage_prefill(table, groups[:2], counts[:2], starts[:2], tails[:2], pool_size=4)
    assert remapped.shape == (1, 4)
    assert remapped[0, :2].tolist() == [0, 1]
