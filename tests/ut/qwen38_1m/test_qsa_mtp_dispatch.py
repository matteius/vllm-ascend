# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4 MTP dispatch and precreated graph metadata; W8 remains unchanged."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.qwen4_exp.dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY
from vllm_ascend.models.qwen4_exp.model import _QSAAttention
from vllm_ascend.models.qwen4_exp.w4_moe import FORMAT


@pytest.mark.parametrize("backend", [None, "eager_dequant", "cube_310", "cube_310_tiled", "cube_310_routed"])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_batched_qsa_limit_and_group_list_are_scoped_to_routed_w4(backend, tp_size):
    config = SimpleNamespace(
        hidden_size=256,
        moe_intermediate_size=256,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_head_dim=16,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )
    if backend is not None:
        config.ascend_expert_quantization = {
            "format": FORMAT,
            "bits": 4,
            "symmetric": False,
            "group_size": 128,
            "packing": "signed_int4_low_nibble_first_in_axis",
            "backend": backend,
            "scale_dtype": "float16",
            "offset_dtype": "int8",
        }
    module = _QSAAttention(
        config=config, layer_idx=0, dtype_policy=ASCEND_QWEN4EXP_DTYPE_POLICY, expert_sharding=(0, tp_size)
    )
    limit = 8 if backend == "cube_310_routed" else 2
    assert module._batched_qsa_max_decode_tokens == limit
    expected = torch.arange(1, limit * module.num_kv_heads + 1, dtype=torch.int64)
    expected *= module.num_heads // module.num_kv_heads
    torch.testing.assert_close(module._qsa_decode_group_list, expected)
    assert "_qsa_decode_group_list" not in module.state_dict()
    metadata = SimpleNamespace(num_decodes=1, num_prefills=0)
    for tokens in (1, 2, 3, 5, 8, 9):
        assert module._can_use_batched_qsa_decode(metadata, tokens, 512) == (tokens <= limit)
        assert not module._can_use_batched_qsa_decode(metadata, tokens, 255)
    assert module._can_use_batched_qsa_decode(metadata, limit, 256)
    for prefills, decodes in ((1, 0), (1, 1), (0, 0)):
        metadata.num_prefills, metadata.num_decodes = prefills, decodes
        assert not module._can_use_batched_qsa_decode(metadata, 2, 512)
