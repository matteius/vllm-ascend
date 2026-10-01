# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend._310p.prefix_mamba_state import (
    PrefixMambaStateTier,
    get_mamba_postprocess_block_ids,
    prefix_mamba_active_columns,
    prefix_mamba_resident_slot_count,
    prefix_mamba_slot_count,
)


def _tier(num_slots: int = 3) -> tuple[PrefixMambaStateTier, torch.Tensor]:
    states = torch.zeros((num_slots, 2), dtype=torch.float16)
    return PrefixMambaStateTier([(states,)], num_slots), states


def test_spill_and_restore_preserves_prefix_checkpoint() -> None:
    tier, states = _tier()
    first = tier.remap_table(np.array([[0, 101, 102]], dtype=np.int32), 3)
    assert first.tolist() == [[0, 1, 2]]
    states[1].fill_(7)
    states[2].fill_(8)

    second = tier.remap_table(np.array([[0, 103, 102]], dtype=np.int32), 3)
    assert second.tolist() == [[0, 1, 2]]
    torch.testing.assert_close(states[2], torch.full((2,), 8, dtype=torch.float16))
    states[1].fill_(9)

    restored = tier.remap_table(np.array([[0, 101, 103]], dtype=np.int32), 3)
    assert restored.tolist() == [[0, 2, 1]]
    torch.testing.assert_close(states[2], torch.full((2,), 7, dtype=torch.float16))
    torch.testing.assert_close(states[1], torch.full((2,), 9, dtype=torch.float16))


def test_reused_block_id_discards_old_checkpoint() -> None:
    tier, states = _tier()
    tier.remap_table(np.array([[0, 101, 102]], dtype=np.int32), 3)
    states[1].fill_(7)
    tier.remap_table(np.array([[0, 103, 102]], dtype=np.int32), 3)
    tier.invalidate([101])
    tier.remap_table(np.array([[0, 101, 103]], dtype=np.int32), 3)
    torch.testing.assert_close(states[2], torch.zeros(2, dtype=torch.float16))


def test_copy_on_write_duplicates_state() -> None:
    tier, states = _tier()
    tier.remap_table(np.array([[0, 101]], dtype=np.int32), 2)
    states[1].fill_(5)
    tier.invalidate([102])
    tier.copy(101, 102)
    tier.remap_table(np.array([[0, 102]], dtype=np.int32), 2)
    torch.testing.assert_close(states[2], torch.full((2,), 5, dtype=torch.float16))


def test_excess_live_states_fails_instead_of_aliasing() -> None:
    tier, _ = _tier()
    with pytest.raises(RuntimeError, match="references 3 states"):
        tier.remap_table(np.array([[101, 102, 103]], dtype=np.int32), 3)


def test_postprocess_uses_compact_slots_not_global_block_223() -> None:
    tier, states = _tier()
    raw = np.array([[0, 223, 224]], dtype=np.int32)
    mapped = tier.remap_table(raw, 3)
    batch = SimpleNamespace(
        block_table=[SimpleNamespace(get_numpy_array=lambda: raw)],
        _prefix_mamba_postprocess_tables={0: mapped},
    )
    block_ids = get_mamba_postprocess_block_ids(batch, 0, 0)
    assert block_ids.tolist() == [0, 1, 2]
    assert states[block_ids].shape == (3, 2)
    assert raw.tolist() == [[0, 223, 224]]
    del batch._prefix_mamba_postprocess_tables
    assert get_mamba_postprocess_block_ids(batch, 0, 0).tolist() == [0, 223, 224]


@pytest.mark.parametrize("requests,drafts,expected", [(1, 2, 64), (2, 2, 64), (16, 2, 97), (32, 4, 321)])
def test_shared_slot_count_covers_previous_and_current_candidate_windows(requests, drafts, expected):
    assert prefix_mamba_slot_count(requests, drafts) == expected


@pytest.mark.parametrize("requests,drafts", [(0, 1), (1, -1)])
def test_invalid_slot_count_fails(requests, drafts):
    with pytest.raises(ValueError):
        prefix_mamba_slot_count(requests, drafts)


def test_resident_slots_consume_device_headroom_after_attention() -> None:
    assert (
        prefix_mamba_resident_slot_count(
            minimum_slots=64,
            maximum_slots=1000,
            state_bytes_per_slot=100,
            physical_cache_bytes=30_000,
            attention_cache_bytes=5_000,
        )
        == 250
    )
    assert (
        prefix_mamba_resident_slot_count(
            minimum_slots=64,
            maximum_slots=100,
            state_bytes_per_slot=100,
            physical_cache_bytes=30_000,
            attention_cache_bytes=5_000,
        )
        == 100
    )


def test_resident_slots_preserve_minimum_when_headroom_is_small() -> None:
    assert (
        prefix_mamba_resident_slot_count(
            minimum_slots=64,
            maximum_slots=1000,
            state_bytes_per_slot=100,
            physical_cache_bytes=5_000,
            attention_cache_bytes=5_000,
        )
        == 64
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_slots": 1},
        {"maximum_slots": 63},
        {"state_bytes_per_slot": 0},
        {"physical_cache_bytes": -1},
        {"attention_cache_bytes": -1},
    ],
)
def test_resident_slot_budget_rejects_invalid_values(kwargs) -> None:
    values = {
        "minimum_slots": 64,
        "maximum_slots": 1000,
        "state_bytes_per_slot": 100,
        "physical_cache_bytes": 30_000,
        "attention_cache_bytes": 5_000,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match="residency budget"):
        prefix_mamba_resident_slot_count(**values)


def test_active_windows_follow_each_requests_progress_and_speculation():
    assert prefix_mamba_active_columns([1026, 5], [130944, 256], [128, 3], 128, 2) == (
        (1022, 1023, 1024, 1025),
        (1, 2, 3, 4),
    )
    # A cached checkpoint can be far from the final prefill chunk destination.
    assert prefix_mamba_active_columns([54, 3], [23424, 0], [3783, 512], 512, 0) == ((45, 53), (0,))
    # Running state is ahead of computed progress after rejected draft tokens.
    assert prefix_mamba_active_columns([6], [255], [3], 128, 2, [3]) == ((2, 3, 4, 5),)


@pytest.mark.parametrize(
    "used,computed,scheduled,block_size,drafts,previous",
    [
        ([2], [0, 1], [1], 128, 0, None),
        ([2], [0], [0], 128, 0, None),
        ([2], [-1], [1], 128, 0, None),
        ([2], [0], [1], 0, 0, None),
        ([2], [0], [1], 128, -1, None),
        ([2], [0], [1], 128, 0, []),
    ],
)
def test_active_windows_reject_invalid_metadata(used, computed, scheduled, block_size, drafts, previous):
    with pytest.raises(ValueError):
        prefix_mamba_active_columns(used, computed, scheduled, block_size, drafts, previous)


def test_missing_destination_or_previous_state_is_not_silently_aliased():
    with pytest.raises(RuntimeError, match="destination"):
        prefix_mamba_active_columns([1], [128], [1], 128, 0)
    with pytest.raises(RuntimeError, match="previous"):
        prefix_mamba_active_columns([2], [128], [1], 128, 0, [2])


def test_uneven_rows_ignore_stale_padding_and_inactive_matching_ids():
    tier, _ = _tier(5)
    raw = np.array([[101, 102, 103, 104], [201, 202, 103, 999]], dtype=np.int32)
    original = raw.copy()
    mapped = tier.remap_rows(raw, [4, 2], [(2, 3), (0, 1)])
    assert mapped[0, :2].tolist() == [0, 0]
    assert mapped[1, 2:].tolist() == [0, 0]
    assert len(set(mapped[mapped > 0].tolist())) == 4
    np.testing.assert_array_equal(raw, original)


def test_all_requests_live_ids_are_protected_before_any_eviction():
    tier, states = _tier()
    tier.remap_table(np.array([[10, 20]], dtype=np.int32), 2)
    states[tier.slot_for(10)].fill_(10)
    states[tier.slot_for(20)].fill_(20)
    # Staging row zero alone would evict the oldest ID 10, needed by row one.
    mapped = tier.remap_rows(np.array([[30], [10]], dtype=np.int32), [1, 1], [(0,), (0,)])
    assert mapped[0, 0] != mapped[1, 0]
    torch.testing.assert_close(states[mapped[1, 0]], torch.full((2,), 10, dtype=torch.float16))
    torch.testing.assert_close(states[mapped[0, 0]], torch.zeros(2, dtype=torch.float16))


def test_reordering_and_finished_request_prefix_reuse_preserve_all_layer_states():
    conv = torch.zeros((3, 4, 2), dtype=torch.float16)
    temporal = torch.zeros((3, 2, 2), dtype=torch.float32)
    tier = PrefixMambaStateTier([(conv, temporal)], 3)
    table = np.array([[101], [201]], dtype=np.int32)
    mapped = tier.remap_rows(table, [1, 1], [(0,), (0,)])
    conv[mapped[0, 0]].fill_(11)
    temporal[mapped[0, 0]].fill_(12)
    conv[mapped[1, 0]].fill_(21)
    temporal[mapped[1, 0]].fill_(22)
    reordered = tier.remap_rows(table[::-1], [1, 1], [(0,), (0,)])
    np.testing.assert_array_equal(reordered, mapped[::-1])
    # Request 101 finishes; 201 is condensed into row zero and another arrives.
    tier.remap_rows(np.array([[201], [301]], dtype=np.int32), [1, 1], [(0,), (0,)])
    restored = tier.remap_rows(table, [1, 1], [(0,), (0,)])
    torch.testing.assert_close(conv[restored[0, 0]], torch.full((4, 2), 11, dtype=torch.float16))
    torch.testing.assert_close(temporal[restored[0, 0]], torch.full((2, 2), 12, dtype=torch.float32))
    torch.testing.assert_close(conv[restored[1, 0]], torch.full((4, 2), 21, dtype=torch.float16))
    torch.testing.assert_close(temporal[restored[1, 0]], torch.full((2, 2), 22, dtype=torch.float32))


def test_shared_prefix_cow_and_recycled_ids_do_not_contaminate_other_window():
    tier, states = _tier()
    tier.remap_rows(np.array([[101], [101]], dtype=np.int32), [1, 1], [(0,), (0,)])
    states[tier.slot_for(101)].fill_(7)
    tier.copy(101, 201)
    mapped = tier.remap_rows(np.array([[101], [201]], dtype=np.int32), [1, 1], [(0,), (0,)])
    states[mapped[1, 0]].fill_(9)
    torch.testing.assert_close(states[mapped[0, 0]], torch.full((2,), 7, dtype=torch.float16))
    tier.invalidate([201])
    torch.testing.assert_close(states[mapped[1, 0]], torch.zeros(2, dtype=torch.float16))
    torch.testing.assert_close(states[mapped[0, 0]], torch.full((2,), 7, dtype=torch.float16))


def test_over_capacity_or_invalid_rows_fail_before_evicting_existing_states():
    tier, states = _tier()
    tier.remap_table(np.array([[101, 102]], dtype=np.int32), 2)
    states[1:].fill_(5)
    before = states.clone()
    resident = dict(tier._resident)
    with pytest.raises(RuntimeError, match="references 3 states"):
        tier.remap_rows(np.array([[201, 202], [301, 0]], dtype=np.int32), [2, 1], [(0, 1), (0,)])
    with pytest.raises(ValueError, match="window"):
        tier.remap_rows(np.array([[201, 202], [301, 0]], dtype=np.int32), [2, 1], [(0,), (1,)])
    assert dict(tier._resident) == resident
    torch.testing.assert_close(states, before)
