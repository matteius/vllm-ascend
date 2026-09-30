# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device tensor primitives for GLM's pooled sparse-attention indexer."""

import torch


def hadamard128(rows: torch.Tensor) -> torch.Tensor:
    """Apply the normalized 128-wide Walsh-Hadamard rotation in FP32."""
    if rows.shape[-1] != 128:
        raise ValueError(f"GLM kpool requires 128-wide index keys, got {rows.shape[-1]}")
    rotated = rows.float()
    for stride in (1, 2, 4, 8, 16, 32, 64):
        pairs = rotated.reshape(*rotated.shape[:-1], 128 // (2 * stride), 2, stride)
        rotated = torch.stack(
            (pairs[..., 0, :] + pairs[..., 1, :], pairs[..., 0, :] - pairs[..., 1, :]),
            dim=-2,
        ).reshape_as(rotated)
    return rotated * (128**-0.5)


def compress_kpool(
    keys: torch.Tensor,
    gate_scores: torch.Tensor,
    ape: torch.Tensor,
) -> torch.Tensor:
    """Compress full pools using GLM's per-dimension gated softmax."""
    if keys.shape != gate_scores.shape or keys.shape[-2:] != ape.shape:
        raise ValueError("K, gate, and APE shapes disagree for GLM kpool compression")
    probabilities = torch.softmax(gate_scores.float() + ape.float(), dim=-2)
    pooled = (keys.float() * probabilities).sum(dim=-2).to(torch.bfloat16)
    return hadamard128(pooled).to(torch.bfloat16)


def score_kpool(
    queries: torch.Tensor,
    weights: torch.Tensor,
    pooled_keys: torch.Tensor,
) -> torch.Tensor:
    """Head-weighted ReLU MQA logits over rotated, compressed keys."""
    if queries.ndim != 3 or queries.shape[-1] != 128:
        raise ValueError("GLM kpool queries must be [tokens, heads, 128]")
    if weights.shape != queries.shape[:2] or pooled_keys.shape[-1] != 128:
        raise ValueError("GLM kpool score inputs have inconsistent shapes")
    num_queries, num_heads, _ = queries.shape
    q_rot = hadamard128(queries).to(torch.bfloat16)
    logits = q_rot.reshape(-1, 128).float() @ pooled_keys.float().transpose(0, 1)
    logits = logits.reshape(num_queries, num_heads, -1).relu_()
    return (logits * weights.float().unsqueeze(-1)).sum(dim=1)


def select_kpool_groups(
    logits: torch.Tensor,
    positions: torch.Tensor,
    topk_tokens: int,
    pool_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select completed pools and retain the current incomplete pool as tail."""
    if logits.ndim != 2 or positions.ndim != 1 or logits.shape[0] != positions.shape[0]:
        raise ValueError("GLM kpool logits and positions must have matching rows")
    if pool_size <= 1 or topk_tokens <= 0 or topk_tokens % pool_size:
        raise ValueError("GLM kpool topk_tokens must be divisible by pool_size > 1")
    pool_budget = topk_tokens // pool_size
    sequence_lengths = positions.to(torch.int32) + 1
    completed = torch.div(sequence_lengths, pool_size, rounding_mode="floor")
    candidate_ids = torch.arange(logits.shape[1], device=logits.device, dtype=torch.int32)
    valid = candidate_ids[None, :] < completed[:, None]
    masked = logits.masked_fill(~valid, -torch.inf)
    selected = torch.full((logits.shape[0], pool_budget), -1, dtype=torch.int32, device=logits.device)
    count = min(pool_budget, logits.shape[1])
    if count:
        if count == logits.shape[1]:
            # Every candidate fits. Avoid a top-k and GatherElements on 310P.
            candidate_rows = candidate_ids[None, :].expand(logits.shape[0], -1)
            selected[:, :count] = torch.where(valid, candidate_rows, -1)
        else:
            topk = torch.topk(masked, count, dim=1).indices.to(torch.int32)
            selected[:, :count] = torch.where(topk < completed[:, None], topk, -1)
    group_counts = completed.clamp(max=pool_budget)
    tail_starts = completed * pool_size
    tail_counts = sequence_lengths - tail_starts
    return selected, group_counts, tail_starts, tail_counts


def expand_kpool_groups(
    selected: torch.Tensor,
    tail_starts: torch.Tensor,
    tail_counts: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Expand pool IDs to token IDs and append 0..pool_size-1 tail tokens."""
    offsets = torch.arange(pool_size, device=selected.device, dtype=torch.int32)
    expanded = selected[:, :, None] * pool_size + offsets[None, None, :]
    expanded = expanded.masked_fill(selected[:, :, None] < 0, -1).flatten(1)
    tail_offsets = offsets[:-1]
    tail = tail_starts[:, None] + tail_offsets[None, :]
    tail = tail.masked_fill(tail_offsets[None, :] >= tail_counts[:, None], -1)
    return torch.cat((expanded, tail), dim=1)
