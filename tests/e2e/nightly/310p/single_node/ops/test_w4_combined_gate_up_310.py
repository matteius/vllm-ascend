# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate/up concatenation is lossless and replays changing route ownership."""

import pytest
import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.w4_moe import KINDS, pack_cube_tiles
from vllm_ascend.utils import enable_custom_op

INPUTS = 2560
OUTPUTS = 640
EXPERTS = 4
GROUP_SIZE = 128


def _make_banks():
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    generator = torch.Generator().manual_seed(1024)
    banks = []
    for _ in range(2):
        canonical = (
            torch.randint(-128, 128, (EXPERTS, OUTPUTS, INPUTS // 2), dtype=torch.int8, generator=generator),
            (torch.rand(EXPERTS, OUTPUTS, INPUTS // GROUP_SIZE, generator=generator) * 0.05 + 0.001).half(),
            torch.randint(-8, 8, (EXPERTS, OUTPUTS, INPUTS // GROUP_SIZE), dtype=torch.int8, generator=generator),
        )
        banks.append(
            tuple(
                torch.stack([pack_cube_tiles(value, kind) for value in values]).npu()
                for values, kind in zip(canonical, KINDS)
            )
        )
    combined = tuple(torch.cat((gate, up), dim=1) for gate, up in zip(*banks))
    assert sum(value.nbytes for value in combined) == sum(value.nbytes for bank in banks for value in bank)
    return generator, banks, combined


@pytest.mark.parametrize("rows", [1, 8, 128])
def test_combined_gate_up_group_prefill(rows):
    generator, banks, combined = _make_banks()
    inputs = (torch.randn(rows, INPUTS, generator=generator) * 0.1).half().npu()
    op = torch.ops._C_ascend.npu_qwen_w4_group_matmul_310
    for expert in range(EXPERTS):
        separate = torch.cat([op(inputs, *(value[expert] for value in bank), True) for bank in banks], -1)
        result = op(inputs, *(value[expert] for value in combined), True)
        torch.testing.assert_close(result.cpu(), separate.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("rows", [10, 30, 50, 80])
def test_combined_gate_up_changing_routes(rows):
    generator, banks, combined = _make_banks()
    inputs = (torch.randn(rows, INPUTS, generator=generator) * 0.1).half().npu()
    ids = (torch.arange(rows, dtype=torch.int32) % EXPERTS).npu()
    op = torch.ops._C_ascend.npu_qwen_w4_routed_matmul_310

    def run():
        return op(inputs, *combined, ids)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            run()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = run()
    patterns = (
        torch.arange(rows, dtype=torch.int32) % EXPERTS,
        torch.full((rows,), -1, dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        torch.arange(rows, dtype=torch.int32) % (EXPERTS + 2) - 1,
    )
    for phase, pattern in enumerate(patterns):
        ids.copy_(pattern)
        inputs.copy_((torch.randn(rows, INPUTS, generator=generator) * (phase + 1) * 0.1).half())
        reference = torch.cat([op(inputs, *bank, ids) for bank in banks], dim=-1).cpu()
        torch.testing.assert_close(run().cpu(), reference, rtol=0, atol=0)
        captured.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(captured.cpu(), reference, rtol=0, atol=0)
