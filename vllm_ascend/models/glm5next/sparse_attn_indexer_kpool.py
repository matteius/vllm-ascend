# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged GLM-5.3-Flash kpool sparse-attention indexer for Ascend."""

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp

from vllm_ascend.models.glm5next.kpool_ops import (
    compress_kpool,
    expand_kpool_groups,
    score_and_select_kpool_tokens,
    select_kpool_groups,
)


def _cache_tensor(cache_layer) -> torch.Tensor:
    cache = cache_layer.kv_cache
    if isinstance(cache, (tuple, list)):
        if len(cache) != 1:
            raise RuntimeError("GLM kpool cache must contain exactly one tensor")
        cache = cache[0]
    if not isinstance(cache, torch.Tensor) or cache.numel() == 0:
        raise RuntimeError("GLM kpool cache has not been bound to the model")
    return cache


@CustomOp.register("sparse_attn_indexer_kpool")
class SparseAttnIndexerKpool(CustomOp):
    """Compress complete pools, persist tails, and select causal pool IDs."""

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str | None,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        tail_cache=None,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.tail_cache = tail_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache

    def _write_pools(
        self,
        keys: torch.Tensor,
        gates: torch.Tensor,
        ape: torch.Tensor,
        positions: torch.Tensor,
        pool_size: int,
        indexer_metadata,
        state_metadata,
    ) -> None:
        """Write complete pools, including one continued from an earlier step."""
        num_tokens = keys.shape[0]
        device = keys.device
        state_cache = _cache_tensor(self.tail_cache)
        key_cache = _cache_tensor(self.k_cache)
        if state_cache.shape[1] != pool_size or state_cache.shape[2] != 2 * self.head_dim:
            raise RuntimeError("GLM kpool state cache has unexpected page geometry")
        if key_cache.shape[-2:] != (1, self.head_dim):
            raise RuntimeError("GLM kpool key cache has unexpected page geometry")

        state_slots = state_metadata.slot_mapping[:num_tokens].long()
        safe_state_slots = state_slots.clamp_min(0)
        state_blocks = torch.div(safe_state_slots, pool_size, rounding_mode="floor")
        # Gather before scattering this step: a previous step may have left
        # the first three members of a pool in the sliding state page.
        old_state = state_cache[state_blocks]
        offsets = torch.arange(pool_size - 1, -1, -1, device=device)
        local_indices = torch.arange(num_tokens, device=device)[:, None] - offsets[None, :]
        safe_local_indices = local_indices.clamp_min(0)
        boundaries = indexer_metadata.cum_query_lens
        if boundaries is None:
            raise RuntimeError("GLM kpool needs device query boundaries from the scheduler")
        request_ids = torch.searchsorted(
            boundaries,
            torch.arange(num_tokens, device=device, dtype=boundaries.dtype),
            right=True,
        )
        same_pool = (
            (local_indices >= 0)
            & (positions[safe_local_indices] == positions[:, None] - offsets[None, :])
            & (request_ids[safe_local_indices] == request_ids[:, None])
        )
        local_keys = keys[safe_local_indices].float()
        local_gates = gates[safe_local_indices].float()
        old_keys = old_state[:, :, : self.head_dim]
        old_gates = old_state[:, :, self.head_dim :]
        pool_keys = torch.where(same_pool[:, :, None], local_keys, old_keys)
        pool_gates = torch.where(same_pool[:, :, None], local_gates, old_gates)

        # Sliding state pages can recycle the same physical slots within a
        # large prefill. Only the request's final pool must survive this step.
        final_positions = indexer_metadata.raw_seq_lens[request_ids].long() - 1
        final_pool_starts = torch.div(final_positions, pool_size, rounding_mode="floor") * pool_size
        valid_state = (state_slots >= 0) & (positions >= final_pool_starts)
        state_cache[
            state_blocks[valid_state],
            safe_state_slots[valid_state] % pool_size,
        ] = torch.cat((keys[valid_state].float(), gates[valid_state].float()), dim=-1)

        completed = ((positions + 1) % pool_size == 0) & (indexer_metadata.slot_mapping[:num_tokens] >= 0)
        compressed = compress_kpool(pool_keys, pool_gates, ape)
        pool_slots = indexer_metadata.slot_mapping[:num_tokens][completed].long()
        block_size = key_cache.shape[1]
        # The shared GLM cache is a page-strided view of an int8 backing.
        # 310P has no IndexPutV2 binary for its BF16 advanced-indexed view;
        # write the same physical elements through a flat storage view.
        cache_elements = key_cache.untyped_storage().nbytes() // key_cache.element_size()
        flat_cache = key_cache.as_strided((cache_elements,), (1,), storage_offset=0)
        row_offsets = (
            key_cache.storage_offset()
            + torch.div(pool_slots, block_size, rounding_mode="floor") * key_cache.stride(0)
            + (pool_slots % block_size) * key_cache.stride(1)
        )
        element_offsets = (
            row_offsets[:, None] + (torch.arange(self.head_dim, device=device) * key_cache.stride(-1))[None, :]
        )
        flat_cache.index_copy_(
            0,
            element_offsets.reshape(-1),
            compressed[completed].to(key_cache.dtype).reshape(-1),
        )

    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if index_kpool <= 1 or gate_score is None or compress_ape is None or positions is None:
            raise ValueError("GLM kpool requires gate scores, APE, positions, and pool_size > 1")
        if self.skip_k_cache_insert or self.tail_cache is None:
            raise NotImplementedError("GLM kpool requires its paged key and compressor state caches")
        if isinstance(q_quant, tuple):
            raise ValueError("GLM kpool expects unquantized BF16 query vectors")
        forward_metadata = get_forward_context().attn_metadata
        if not isinstance(forward_metadata, dict):
            # The profiling run has no cache pages or valid request positions.
            return self.topk_indices_buffer
        indexer_metadata = forward_metadata[self.k_cache.prefix]
        state_metadata = forward_metadata[self.tail_cache.prefix]
        num_tokens = indexer_metadata.num_actual_tokens
        if num_tokens > min(k.shape[0], q_quant.shape[0], positions.shape[0]):
            raise RuntimeError("GLM kpool metadata exceeds the actual token tensors")
        positions = positions[:num_tokens]
        self._write_pools(
            k[:num_tokens],
            gate_score[:num_tokens],
            compress_ape,
            positions,
            index_kpool,
            indexer_metadata,
            state_metadata,
        )

        cache = _cache_tensor(self.k_cache)
        block_size = cache.shape[1]
        boundaries = indexer_metadata.cum_query_lens_cpu
        if boundaries is None:
            raise RuntimeError("GLM kpool needs host query boundaries from the scheduler")
        query_ends = boundaries.tolist()
        self.topk_indices_buffer[:num_tokens].fill_(-1)
        for request, (start, end) in enumerate(zip(query_ends[:-1], query_ends[1:])):
            if start == end:
                continue
            num_pools = int(indexer_metadata.seq_lens_cpu[request])
            if num_pools > indexer_metadata.block_table.shape[1] * block_size:
                raise RuntimeError("GLM kpool block table is shorter than the request's pooled keys")
            pool_ids = torch.arange(num_pools, device=cache.device, dtype=torch.long)
            page_ids = indexer_metadata.block_table[request, pool_ids // block_size].long()
            keys = cache[page_ids, pool_ids % block_size, 0]
            if num_pools <= self.topk_tokens // index_kpool:
                logits = torch.zeros(end - start, num_pools, device=cache.device)
                selected, _, tail_starts, tail_counts = select_kpool_groups(
                    logits, positions[start:end], self.topk_tokens, index_kpool
                )
                expanded = expand_kpool_groups(selected, tail_starts, tail_counts, index_kpool)
            else:
                expanded = score_and_select_kpool_tokens(
                    q_quant[start:end],
                    weights[start:end],
                    keys,
                    positions[start:end],
                    self.topk_tokens,
                    index_kpool,
                )
            self.topk_indices_buffer[start:end, : expanded.shape[1]] = expanded
        return self.topk_indices_buffer

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_oot(
            hidden_states,
            q_quant,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=compress_ape,
            index_kpool=index_kpool,
            positions=positions,
        )
