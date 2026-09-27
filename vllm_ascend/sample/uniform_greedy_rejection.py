# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-shape greedy verification for short, uniformly sized draft batches."""

import torch

# Bound the unrolled device operations; ragged/wider batches keep the generic
# implementation, and one-token batches retain their existing special case.
MAX_UNIFORM_GREEDY_DRAFT_TOKENS = 8


def try_uniform_greedy_rejection(
    output_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_argmax: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    draft_tokens_per_req: list[int],
    max_spec_len: int,
    is_greedy: torch.Tensor | None = None,
    uniform_probs: torch.Tensor | None = None,
    synthetic_conditional_rates: torch.Tensor | None = None,
    synthetic_mode: bool = False,
) -> bool:
    """Update supported batches without data-dependent shapes or host reads.

    All requests must have exactly ``max_spec_len`` drafts. Scheduler-owned
    Python lengths establish the rectangular layout without reading the device
    cumulative-length tensor. Only acceptance stays dynamic, entirely on device.
    Rejected suffixes and non-greedy rows retain the caller's original contents.
    Returning False leaves output untouched for the generic fallback.
    """
    batch_size = output_token_ids.shape[0]
    if (
        not 1 < max_spec_len <= MAX_UNIFORM_GREEDY_DRAFT_TOKENS
        or batch_size == 0
        or len(draft_tokens_per_req) != batch_size
        or any(count != max_spec_len for count in draft_tokens_per_req)
        or draft_token_ids.numel() != batch_size * max_spec_len
        or target_argmax.numel() != batch_size * max_spec_len
        or bonus_token_ids.numel() != batch_size
        or output_token_ids.shape[1] < max_spec_len + 1
    ):
        return False
    if synthetic_mode:
        assert uniform_probs is not None and synthetic_conditional_rates is not None
        probabilities = uniform_probs.reshape(batch_size, max_spec_len)

    drafts = draft_token_ids.reshape(batch_size, max_spec_len)
    targets = target_argmax.reshape(batch_size, max_spec_len)
    active = (
        torch.ones(batch_size, dtype=torch.bool, device=output_token_ids.device) if is_greedy is None else is_greedy
    )
    # The bounded Python loop schedules fixed-shape device operations. It does
    # not inspect device values, construct a host draft-count tensor, use
    # nonzero/boolean indexing, or synchronize to find the accepted length.
    for position in range(max_spec_len):
        draft = drafts[:, position]
        target = targets[:, position]
        if synthetic_mode:
            accepted = (probabilities[:, position] < synthetic_conditional_rates[position]) & (draft >= 0)
            token = torch.where(accepted, draft, target)
        else:
            accepted = draft == target
            token = target
        column = output_token_ids[:, position]
        column.copy_(torch.where(active, token, column))
        active = active & accepted

    bonus_column = output_token_ids[:, max_spec_len]
    bonus_column.copy_(torch.where(active, bonus_token_ids.reshape(batch_size), bonus_column))
    return True
