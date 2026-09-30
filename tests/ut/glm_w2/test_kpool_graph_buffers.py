"""CPU checks for capture-stable GLM KPool metadata storage."""

import pytest
import torch

from vllm_ascend.attention.kpool_graph_buffers import KPoolGraphBufferSets


def test_draft_steps_keep_independent_stable_buffers() -> None:
    sets = KPoolGraphBufferSets(torch.device("cpu"), max_tokens=8, max_reqs=4, block_table_width=2)
    first_source = torch.zeros(8, dtype=torch.int32)
    second_source = torch.zeros(8, dtype=torch.int32)

    first = sets.get(first_source)
    first.positions[:2].copy_(torch.tensor([3, 4]))
    second = sets.get(second_source)

    assert sets.get(first_source) is first
    assert sets.get(second_source) is second
    assert first.slots.data_ptr() != second.slots.data_ptr()
    assert first.block_table.data_ptr() != second.block_table.data_ptr()
    assert first.positions[:2].tolist() == [3, 4]


def test_block_table_growth_is_scoped_and_frozen_after_capture() -> None:
    sets = KPoolGraphBufferSets(torch.device("cpu"), max_tokens=8, max_reqs=2, block_table_width=2)
    first_source = torch.zeros(8, dtype=torch.int32)
    second_source = torch.zeros(8, dtype=torch.int32)
    first = sets.get(first_source)
    second = sets.get(second_source)
    second_ptr = second.block_table.data_ptr()

    assert sets.ensure_table_capacity(first_source, num_reqs=3, width=5) is first
    assert first.block_table.shape == (3, 5)
    assert second.block_table.data_ptr() == second_ptr

    captured_ptr = first.block_table.data_ptr()
    sets.freeze(first_source)
    assert sets.ensure_table_capacity(first_source, num_reqs=3, width=5).block_table.data_ptr() == captured_ptr
    with pytest.raises(RuntimeError, match="captured block-table capacity"):
        sets.ensure_table_capacity(first_source, num_reqs=3, width=6)
