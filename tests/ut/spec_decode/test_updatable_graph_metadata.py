# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


def _metadata(query_lengths, kv_lengths, block_table):
    return SimpleNamespace(
        actual_seq_lengths_q=query_lengths,
        seq_lens_list=kv_lengths,
        block_tables=block_table,
    )


def test_updates_each_draft_step_and_attention_layer() -> None:
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer._runnable = MagicMock()
    backend = object()
    first = _metadata([1], [8], "first-blocks")
    second = _metadata([2], [9], "second-blocks")
    third = _metadata([3], [10], "third-blocks")

    with patch(
        "vllm_ascend.spec_decode.llm_base_proposer.use_updatable_graph",
        return_value=True,
    ):
        proposer._maybe_update_metadata(
            backend,
            [
                {"layers.0.attn": first, "layers.1.attn": second},
                {"layers.0.attn": third},
            ],
        )

    proposer._runnable.update_draft_model_metadata.assert_called_once_with(
        [
            {
                "layer_name": "layers.0.attn",
                "actual_seq_lengths": [1],
                "actual_seq_lengths_kv": [8],
                "block_table": "first-blocks",
            },
            {
                "layer_name": "layers.1.attn",
                "actual_seq_lengths": [2],
                "actual_seq_lengths_kv": [9],
                "block_table": "second-blocks",
            },
            {
                "layer_name": "layers.0.attn",
                "actual_seq_lengths": [3],
                "actual_seq_lengths_kv": [10],
                "block_table": "third-blocks",
            },
        ]
    )
    proposer._runnable.set_attn_backend.assert_called_once_with(backend)


def test_skips_metadata_update_for_non_updatable_graph() -> None:
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer._runnable = MagicMock()

    with patch(
        "vllm_ascend.spec_decode.llm_base_proposer.use_updatable_graph",
        return_value=False,
    ):
        proposer._maybe_update_metadata(object(), [])

    proposer._runnable.update_draft_model_metadata.assert_not_called()
    proposer._runnable.set_attn_backend.assert_not_called()


def test_skips_metadata_update_without_outer_graph_wrapper() -> None:
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer._runnable = lambda: None

    with patch(
        "vllm_ascend.spec_decode.llm_base_proposer.use_updatable_graph",
        return_value=True,
    ):
        proposer._maybe_update_metadata(object(), [])


def test_forwards_and_records_update_stream() -> None:
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer._runnable = MagicMock()
    update_stream = object()

    proposer.set_update_stream(update_stream)

    proposer._runnable.set_update_stream.assert_called_once_with(update_stream)
    assert proposer.update_stream is update_stream


def test_cache_only_group_resolves_backend_in_config_context() -> None:
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.vllm_config = object()
    group = SimpleNamespace(backend=MagicMock())
    group.backend.get_impl_cls.return_value = None

    with patch(
        "vllm_ascend.spec_decode.llm_base_proposer.set_current_vllm_config",
        return_value=nullcontext(),
    ) as config_context:
        assert proposer._is_cache_only_draft_attn_group(group)

    config_context.assert_called_once_with(proposer.vllm_config)
