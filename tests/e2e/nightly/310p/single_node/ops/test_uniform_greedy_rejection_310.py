# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact device-side uniform rejection and changing-input graph replay."""

import pytest
import torch
import torch_npu

from vllm_ascend.sample.rejection_sampler import rejection_greedy_sample_pytorch


def _reference(drafts, targets, bonuses, greedy, probabilities, rates, synthetic, dtype):
    batch_size, width = drafts.shape
    output = torch.full((batch_size, width + 1), -17, dtype=dtype)
    for row in range(batch_size):
        if not greedy[row]:
            continue
        for position in range(width):
            accepted = (
                probabilities[row, position] < rates[position] and drafts[row, position] >= 0
                if synthetic
                else drafts[row, position] == targets[row, position]
            )
            output[row, position] = drafts[row, position] if synthetic and accepted else targets[row, position]
            if not accepted:
                break
        else:
            output[row, width] = bonuses[row]
    return output


@pytest.mark.parametrize("width", [2, 3, 4, 8])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("synthetic", [False, True])
@pytest.mark.parametrize("use_greedy_mask", [False, True])
def test_uniform_greedy_rejection_replay(width, batch_size, dtype, synthetic, use_greedy_mask):
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    drafts = torch.zeros(batch_size * width, dtype=torch.int64, device="npu")
    targets = torch.zeros_like(drafts)
    bonuses = torch.zeros(batch_size, dtype=torch.int64, device="npu")
    greedy = torch.ones(batch_size, dtype=torch.bool, device="npu")
    probabilities = torch.zeros(batch_size * width, device="npu")
    rates = torch.ones(width, device="npu")
    cumulative = torch.arange(1, batch_size + 1, dtype=torch.int32, device="npu") * width
    output = torch.empty((batch_size, width + 1), dtype=dtype, device="npu")

    def run():
        output.fill_(-17)
        rejection_greedy_sample_pytorch(
            output,
            cumulative,
            drafts,
            targets,
            bonuses,
            [width] * batch_size,
            width,
            greedy if use_greedy_mask else None,
            probabilities,
            rates,
            synthetic,
        )

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            run()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        run()
    for phase in range(width + 4):
        cpu_targets = torch.arange(batch_size * width).reshape(batch_size, width) + phase * 100 + 10
        cpu_drafts = cpu_targets.clone() + (1000 if synthetic else 0)
        cpu_probabilities = torch.zeros(batch_size, width)
        cpu_rates = torch.ones(width)
        cpu_bonuses = torch.arange(batch_size) + 10000 + phase
        cpu_greedy = torch.ones(batch_size, dtype=torch.bool)
        if phase < width:
            cpu_drafts[:, phase] += 1
            cpu_probabilities[:, phase] = 1.0  # Strict threshold equality rejects.
        elif phase == width + 1:
            cpu_drafts[:, 0] = -1
        elif phase == width + 2 and use_greedy_mask:
            cpu_greedy[::2] = False
        elif phase == width + 3 and use_greedy_mask:
            cpu_greedy.zero_()
        for device_value, cpu_value in (
            (drafts, cpu_drafts.flatten()),
            (targets, cpu_targets.flatten()),
            (bonuses, cpu_bonuses),
            (greedy, cpu_greedy),
            (probabilities, cpu_probabilities.flatten()),
            (rates, cpu_rates),
        ):
            device_value.copy_(cpu_value)
        expected = _reference(
            cpu_drafts, cpu_targets, cpu_bonuses, cpu_greedy, cpu_probabilities, cpu_rates, synthetic, dtype
        )
        run()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        output.fill_(777)  # Replay must clear stale accepted tokens from prior steps.
        graph.replay()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
