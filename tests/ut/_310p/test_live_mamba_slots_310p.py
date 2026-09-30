# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Four-request compact Mamba state isolation without prefix snapshots."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

from vllm_ascend._310p.prefix_mamba_state import LiveMambaRequestSlots, supports_compact_live_mamba_state


def test_glm_multi_request_uses_compact_live_mamba_state() -> None:
    assert supports_compact_live_mamba_state(4, "glm5_next_text")
    assert supports_compact_live_mamba_state(4, "qwen4_exp_text")
    assert supports_compact_live_mamba_state(1, None)
    assert not supports_compact_live_mamba_state(4, None)


def test_glm_live_state_rejects_runner_without_compact_remap() -> None:
    pytest.importorskip("torch_npu")
    from vllm_ascend._310p.model_runner_310p import NPUModelRunner310

    runner = object.__new__(NPUModelRunner310)
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None)
    runner.model_config = SimpleNamespace(use_mla=False, hf_text_config=SimpleNamespace(model_type="glm5_next_text"))
    runner.use_sparse = False
    runner.supports_compact_mamba_state = False
    runner._get_layer_kv_cache_specs = lambda _: {
        "model.layers.1.attn": MLAAttentionSpec(
            block_size=512,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.float16,
            model_version="glm5_next",
        ),
        "model.layers.0.linear_attn": MambaSpec(
            block_size=512,
            shapes=((3, 16), (1, 16, 16)),
            dtypes=(torch.float16, torch.float32),
            mamba_cache_mode="none",
        ),
    }

    with pytest.raises(ValueError, match="requires compact Mamba block-table remapping"):
        runner.initialize_kv_cache_tensors(SimpleNamespace())


def test_lanes_follow_request_identity_and_reuse_starts_zero() -> None:
    pytest.importorskip("torch_npu")
    from vllm_ascend._310p.model_runner_310p import NPUModelRunner310

    runner = object.__new__(NPUModelRunner310)
    runner.supports_compact_mamba_state = True
    runner.supports_prefix_mamba_state_tier = False
    runner.num_compact_mamba_blocks = 3
    runner._live_mamba_slots = LiveMambaRequestSlots(4, 3)
    states = torch.full((12, 2), 9.0)
    runner._compact_mamba_state_tensors = (states,)
    raw_mamba = np.array([[100 + row * 10 + column for column in range(5)] for row in range(4)], dtype=np.int32)
    device_mamba = torch.from_numpy(raw_mamba.copy())
    runner.input_batch = SimpleNamespace(
        req_ids=["a", "b", "c", "d"],
        block_table=SimpleNamespace(
            block_tables=[
                SimpleNamespace(is_mamba_group=False, block_table=SimpleNamespace(gpu=torch.zeros((4, 5)))),
                SimpleNamespace(is_mamba_group=True, block_table=SimpleNamespace(np=raw_mamba, gpu=device_mamba)),
            ]
        ),
    )
    runner.requests = {
        req_id: SimpleNamespace(block_ids=([0, 1, 2, 3, 4], raw_mamba[row].tolist()))
        for row, req_id in enumerate(runner.input_batch.req_ids)
    }

    runner._remap_compact_mamba_block_tables(4)
    expected = [[row * 3 + column % 3 for column in range(5)] for row in range(4)]
    np.testing.assert_array_equal(runner.input_batch._prefix_mamba_postprocess_tables[1], expected)
    torch.testing.assert_close(device_mamba, torch.tensor(expected, dtype=torch.int32))
    torch.testing.assert_close(states, torch.zeros_like(states))
    runner._stage_prefix_mamba_request_ids()
    for row, req_id in enumerate(runner.input_batch.req_ids):
        assert runner.requests[req_id].block_ids[1] == expected[row]
    runner._restore_prefix_mamba_request_ids()
    states[0:3].fill_(7)

    runner.input_batch.req_ids = ["d", "b", "c", "a"]
    runner._remap_compact_mamba_block_tables(4)
    np.testing.assert_array_equal(runner.input_batch._prefix_mamba_postprocess_tables[1][0], expected[3])
    np.testing.assert_array_equal(runner.input_batch._prefix_mamba_postprocess_tables[1][3], expected[0])
    torch.testing.assert_close(states[0:3], torch.full((3, 2), 7.0))

    del runner.requests["a"]
    runner.requests["e"] = SimpleNamespace(block_ids=([0, 1, 2, 3, 4], [500, 501, 502, 503, 504]))
    runner.input_batch.req_ids = ["d", "b", "c", "e"]
    runner._remap_compact_mamba_block_tables(4)
    np.testing.assert_array_equal(runner.input_batch._prefix_mamba_postprocess_tables[1][3], expected[0])
    torch.testing.assert_close(states[0:3], torch.zeros((3, 2)))


def test_no_lane_alias_when_a_request_is_temporarily_unscheduled() -> None:
    slots = LiveMambaRequestSlots(2, 3)
    assert slots.assign(["a", "b"], {"a", "b"}) == ((0, 1), (0, 1))
    assert slots.assign(["b"], {"a", "b"}) == ((1,), ())
    with pytest.raises(RuntimeError, match="No compact Mamba lane"):
        slots.assign(["b", "c"], {"a", "b", "c"})
    assert slots.assign(["b", "c"], {"b", "c"}) == ((1, 0), (0,))
