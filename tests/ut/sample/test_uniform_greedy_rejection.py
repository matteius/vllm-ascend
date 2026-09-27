# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import pytest
import torch

from vllm_ascend.sample.uniform_greedy_rejection import try_uniform_greedy_rejection


@pytest.mark.parametrize("draft_count", [2, 3, 4, 8])
@pytest.mark.parametrize("output_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("greedy_mode", ["all", "mixed", "none"])
@pytest.mark.parametrize("synthetic", [False, True])
def test_uniform_rejection_every_acceptance_pattern(draft_count, output_dtype, greedy_mode, synthetic):
    patterns = list(itertools.product([False, True], repeat=draft_count))
    batch_size = len(patterns)
    accepts = torch.tensor(patterns)
    targets = torch.arange(batch_size * draft_count).reshape(batch_size, draft_count) + 100
    drafts = torch.where(accepts, targets, targets + 1000)
    probabilities = torch.where(accepts, 0.25, 0.5)
    rates = torch.full((draft_count,), 0.5)
    if synthetic:
        # Synthetic acceptance may emit a draft that differs from target;
        # negative IDs must reject even when the probability would accept.
        drafts = targets + 1000
        drafts[-1, -1] = -1
    bonuses = torch.arange(batch_size).reshape(-1, 1) + 10000
    greedy = None if greedy_mode == "all" else torch.arange(batch_size) % 3 != 0
    if greedy_mode == "none":
        greedy.zero_()
    # Extra output columns and non-greedy/rejected lanes must stay untouched.
    output = torch.arange(batch_size * (draft_count + 2), dtype=output_dtype).reshape(batch_size, -1)
    expected = output.clone()
    for row in range(batch_size):
        if greedy is not None and not greedy[row]:
            continue
        for position in range(draft_count):
            accepted = (
                probabilities[row, position] < rates[position] and drafts[row, position] >= 0
                if synthetic
                else drafts[row, position] == targets[row, position]
            )
            expected[row, position] = drafts[row, position] if synthetic and accepted else targets[row, position]
            if not accepted:
                break
        else:
            expected[row, draft_count] = bonuses[row, 0]
    # Exercise noncontiguous flattened input without changing request order.
    draft_storage = torch.stack((drafts.flatten(), torch.zeros_like(drafts.flatten())), dim=1)
    assert try_uniform_greedy_rejection(
        output,
        draft_storage[:, 0],
        targets.flatten(),
        bonuses,
        [draft_count] * batch_size,
        draft_count,
        greedy,
        probabilities.flatten() if synthetic else None,
        rates if synthetic else None,
        synthetic,
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("counts,width", [([0, 0], 0), ([1, 1], 1), ([2, 1], 2), ([2, 0], 2), ([9, 9], 9)])
def test_unsupported_batches_are_untouched(counts, width):
    output = torch.full((len(counts), width + 1), -7, dtype=torch.int32)
    original = output.clone()
    assert not try_uniform_greedy_rejection(
        output, torch.zeros(sum(counts)), torch.zeros(sum(counts)), torch.zeros(len(counts)), counts, width
    )
    torch.testing.assert_close(output, original)


def test_uniform_synthetic_probability_boundaries():
    output = torch.full((3, 3), -1, dtype=torch.int32)
    drafts = torch.tensor([4, 5, 6, 7, 8, 9])
    targets = torch.tensor([40, 50, 60, 70, 80, 90])
    assert try_uniform_greedy_rejection(
        output,
        drafts,
        targets,
        torch.tensor([100, 200, 300]),
        [2, 2, 2],
        2,
        uniform_probs=torch.tensor([0.0, 0.0, 0.0, 0.5, 0.0, 1.0]),
        synthetic_conditional_rates=torch.tensor([1.0, 0.5]),
        synthetic_mode=True,
    )
    torch.testing.assert_close(output, torch.tensor([[4, 5, 100], [6, 70, -1], [8, 90, -1]], dtype=torch.int32))


def test_uniform_rejection_does_not_read_device_scalars_or_build_host_tensors(monkeypatch):
    output = torch.full((1, 3), -1, dtype=torch.int32)
    drafts = torch.tensor([4, 5])
    targets = torch.tensor([4, 6])
    bonuses = torch.tensor([7])

    def forbidden(*args, **kwargs):
        pytest.fail("uniform rejection must not construct host tensors or inspect device values")

    with monkeypatch.context() as context:
        context.setattr(torch, "tensor", forbidden)
        context.setattr(torch, "nonzero", forbidden)
        context.setattr(torch.Tensor, "item", forbidden)
        context.setattr(torch.Tensor, "__bool__", forbidden)
        assert try_uniform_greedy_rejection(output, drafts, targets, bonuses, [2], 2)
    torch.testing.assert_close(output, torch.tensor([[4, 6, -1]], dtype=torch.int32))
