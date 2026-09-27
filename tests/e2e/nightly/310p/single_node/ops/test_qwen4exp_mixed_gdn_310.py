# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed native GDN must equal independent prefill/spec state updates."""

from copy import copy
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("has_initial_state", [False, True])
@pytest.mark.parametrize("prefill_tokens", [17, 70])
@torch.inference_mode()
def test_native_mixed_matches_separate_and_preserves_other_slots(has_initial_state, prefill_tokens):
    pytest.importorskip("torch_npu")
    from vllm_ascend.models.qwen4_exp.model import _GDNAttention
    from vllm_ascend.utils import enable_custom_op, is_310p

    if not torch.npu.is_available() or not is_310p():
        pytest.skip("Ascend 310P required")
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    generator = torch.Generator().manual_seed(31)

    def random(shape):
        return (torch.randn(shape, generator=generator) * 0.1).half().npu()

    layer = _GDNAttention.__new__(_GDNAttention)
    torch.nn.Module.__init__(layer)
    layer.params = SimpleNamespace(conv_kernel_size=4, head_k_dim=128, head_v_dim=128)
    layer.num_k_heads, layer.num_v_heads = 4, 8
    layer.key_dim, layer.value_dim = 512, 1024
    layer.conv_dim = 2 * layer.key_dim + layer.value_dim
    layer.compute_dtype = torch.float32
    layer.conv_weight = torch.nn.Parameter(random((layer.conv_dim, 4)))
    layer.A_log = torch.nn.Parameter(random((layer.num_v_heads,)))
    layer.dt_bias = torch.nn.Parameter(random((layer.num_v_heads,)))
    initial_caches = (random((16, 5, layer.conv_dim)), random((16, 8, 128, 128)))
    layer.kv_cache = tuple(state.clone() for state in initial_caches)
    total_tokens = prefill_tokens + 3
    mixed = random((total_tokens, layer.conv_dim))
    a, b = random((total_tokens, 8)), random((total_tokens, 8))
    # Non-contiguous partition catches accidental concatenation of the outputs.
    spec_ids = [0, 2, total_tokens - 1]
    spec_index = torch.tensor(spec_ids, device="npu")
    non_index = torch.tensor([i for i in range(total_tokens) if i not in spec_ids], device="npu")
    spec_states = torch.tensor([[1, 2, 3]], dtype=torch.int32, device="npu")
    spec_query = torch.tensor([0, 3], dtype=torch.int32, device="npu")
    accepts = torch.tensor([2], dtype=torch.int32, device="npu")
    metadata = SimpleNamespace(
        num_prefills=1,
        num_prefill_tokens=prefill_tokens,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=1,
        num_spec_decode_tokens=3,
        num_actual_tokens=total_tokens,
        spec_sequence_masks=torch.tensor([True, False], device="npu"),
        spec_token_indx=spec_index,
        non_spec_token_indx=non_index,
        spec_state_indices_tensor=spec_states,
        non_spec_state_indices_tensor=torch.tensor([11], device="npu"),
        spec_query_start_loc=spec_query,
        non_spec_query_start_loc=torch.tensor([0, prefill_tokens], dtype=torch.int32, device="npu"),
        has_initial_state=torch.tensor([has_initial_state], device="npu"),
        spec_decode_metadata=SimpleNamespace(
            spec_causal_conv1d=SimpleNamespace(
                query_start_loc=spec_query, cache_indices=spec_states, num_accepted_tokens=accepts
            )
        ),
    )
    actual = layer._native_mixed_attention(mixed, a, b, metadata)
    actual_caches = tuple(state.clone() for state in layer.kv_cache)
    layer.kv_cache = tuple(state.clone() for state in initial_caches)
    expected = torch.empty_like(actual)
    for spec, indices in ((True, spec_index), (False, non_index)):
        view = copy(metadata)
        if spec:
            view.num_prefills = view.num_prefill_tokens = 0
            states, query, initial = spec_states, spec_query, None
        else:
            view.spec_sequence_masks = view.spec_decode_metadata = None
            view.num_spec_decodes = view.num_spec_decode_tokens = 0
            states = metadata.non_spec_state_indices_tensor
            query, initial = metadata.non_spec_query_start_loc, metadata.has_initial_state
        branch = layer._stateful_short_conv(mixed[indices], states, query, initial, metadata=view)
        q, k, v = torch.split(branch, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1)
        g, beta = layer._native_gating(a[indices], b[indices])
        expected[indices] = layer._native_delta_rule(
            q.reshape(-1, 4, 128),
            k.reshape(-1, 4, 128),
            v.reshape(-1, 8, 128),
            g,
            beta,
            view,
            states,
            query,
            initial,
        ).to(expected.dtype)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=1e-3, rtol=1e-3)
    untouched = torch.tensor([i for i in range(16) if i not in (1, 2, 3, 11)], device="npu")
    for actual_state, expected_state, original in zip(actual_caches, layer.kv_cache, initial_caches):
        torch.testing.assert_close(actual_state.cpu(), expected_state.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(actual_state[untouched].cpu(), original[untouched].cpu(), atol=0, rtol=0)
