# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device gates: exact grouped W4A16 and native two-INT4 W4A8 arithmetic."""

import pytest
import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.models.qwen4_exp.w4_moe import pack_cube_tiles
from vllm_ascend.models.qwen4_exp.w4a8_int4 import (
    pack_native_metadata,
    pack_native_weight,
    quantize_activation_limbs,
)
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(autouse=True, scope="module")
def require_kernel():
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    for name in ("npu_qwen_w4_grouped_matmul_310", "npu_qwen_w4_a8_int4_matmul_310"):
        assert hasattr(torch.ops._C_ascend, name), f"rebuild missing operator {name}"


def values(rows, inputs=256, outputs=128, seed=42):
    generator = torch.Generator().manual_seed(seed)
    x = (torch.randn(rows, inputs, generator=generator) * 0.1).half()
    codes = torch.randint(-128, 128, (3, outputs, inputs // 2), dtype=torch.int8, generator=generator)
    scales = (torch.rand(3, outputs, inputs // 128, generator=generator) * 0.02 + 0.001).half()
    offsets = torch.randint(-8, 8, scales.shape, dtype=torch.int8, generator=generator)
    return x, codes, scales, offsets


def unpack(packed):
    unsigned = packed.to(torch.int16) % 256
    nibble = torch.stack((unsigned % 16, unsigned // 16), -1).flatten(-2)
    return ((nibble + 8) % 16 - 8).float()


def reference(data, ends, native):
    x, codes, scales, offsets = data
    rows, inputs = x.shape
    result = torch.zeros(rows, codes.shape[1], dtype=torch.float16)
    start = 0
    for expert, end in enumerate(ends.tolist()):
        grouped = x[start:end].float().reshape(end - start, inputs // 128, 128)
        if native:
            scale = grouped.abs().amax(-1) / 127
            scale = torch.where(scale == 0, 1.0, scale)
            grouped = (grouped / scale[..., None]).round().clamp(-127, 127) * scale[..., None]
        weight = (
            (unpack(codes[expert]).reshape(*scales[expert].shape, 128) - offsets[expert].float()[..., None])
            * scales[expert].float()[..., None]
        ).flatten(-2)
        # W4A16 rounds each dequantized weight to FP16; W4A8 applies scales
        # after integer dot products. These are deliberately separate contracts.
        result[start:end] = F.linear(grouped.flatten(-2), weight if native else weight.half().float()).half()
        start = end
    return result


def device_arguments(data, ends, native):
    x, codes, scales, offsets = data
    if native:
        bank, sums = zip(*(pack_native_weight(expert) for expert in codes))
        return [
            *quantize_activation_limbs(x.npu()),
            torch.stack(bank).npu(),
            torch.stack([pack_native_metadata(s) for s in scales]).npu(),
            torch.stack([pack_native_metadata(z).half() for z in offsets]).npu(),
            torch.stack(sums).npu(),
            ends.npu(),
        ]
    return [
        x.npu(),
        *[
            torch.stack([pack_cube_tiles(e, kind) for e in tensor]).npu()
            for kind, tensor in zip(("weight", "weight_scale", "weight_offset"), (codes, scales, offsets))
        ],
        ends.npu(),
    ]


def invoke(arguments, native):
    if native:
        return torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310(*arguments)
    return torch.ops._C_ascend.npu_qwen_w4_grouped_matmul_310(*arguments)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("rows", [1, 15, 16, 17, 127, 128, 129, 513])
def test_empty_experts_m_tails_and_peer_zero(rows, native):
    data = values(rows)
    ends = torch.tensor([0, max(1, rows - 5), max(1, rows - 5)], dtype=torch.int64)
    actual = invoke(device_arguments(data, ends, native), native).cpu()
    torch.testing.assert_close(actual, reference(data, ends, native), rtol=0.005, atol=0.003)
    assert torch.count_nonzero(actual[ends[-1] :]) == 0
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("inputs,outputs", [(2560, 1280), (640, 2560)])
def test_real_projection_shapes(inputs, outputs, native):
    data = values(33, inputs=inputs, outputs=outputs)
    ends = torch.tensor([16, 17, 31], dtype=torch.int64)
    actual = invoke(device_arguments(data, ends, native), native).cpu()
    torch.testing.assert_close(actual, reference(data, ends, native), rtol=0.005, atol=0.003)


@pytest.mark.parametrize("native", [False, True])
def test_replay_changes_group_ends_weights_and_activations(native):
    data = values(129)
    ends = torch.tensor([16, 17, 128], dtype=torch.int64)
    arguments = device_arguments(data, ends, native)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            invoke(arguments, native)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = invoke(arguments, native)
    for phase, boundaries in enumerate(([0, 0, 0], [0, 129, 129], [1, 128, 129], [0, 0, 0], [129, 129, 129])):
        data = values(129, seed=100 + phase)
        ends = torch.tensor(boundaries, dtype=torch.int64)
        for target, source in zip(arguments, device_arguments(data, ends, native)):
            target.copy_(source)
        graph.replay()
        torch.testing.assert_close(output.cpu(), reference(data, ends, native), rtol=0.005, atol=0.003)


def test_native_integer_dot_exact_for_all_weight_bytes_and_zero_points():
    x = torch.zeros(16, 256).half()
    x[:, 0] = 127  # Unit activation scale in each group.
    x[:, 128] = -127
    for row in range(16):
        x[row, row + 1] = row - 8
        x[row, row + 129] = 119 + row % 9
    codes = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8).repeat(3, 64, 1).reshape(3, 128, 128)
    scales = torch.full((3, 128, 2), 0.125, dtype=torch.float16)
    offsets = (torch.arange(128) % 16 - 8).to(torch.int8)[None, :, None].expand(3, -1, 2).contiguous()
    data = x, codes, scales, offsets
    ends = torch.tensor([0, 16, 16], dtype=torch.int64)
    actual = invoke(device_arguments(data, ends, True), True).cpu()
    torch.testing.assert_close(actual, reference(data, ends, True), rtol=0, atol=0)


@pytest.mark.parametrize("native", [False, True])
def test_invalid_metadata_rejected_before_device_launch(native):
    arguments = device_arguments(values(17), torch.tensor([0, 16, 17]), native)
    wrong_dtype = list(arguments)
    wrong_dtype[-1] = wrong_dtype[-1].float()
    with pytest.raises(RuntimeError, match="dtype|expects"):
        invoke(wrong_dtype, native)
    wrong_count = list(arguments)
    wrong_count[-1] = wrong_count[-1][:-1].contiguous()
    with pytest.raises(RuntimeError, match="dimensions"):
        invoke(wrong_count, native)
    noncontiguous = list(arguments)
    noncontiguous[0] = noncontiguous[0][:, ::2]
    with pytest.raises(RuntimeError, match="contiguous"):
        invoke(noncontiguous, native)


@pytest.mark.parametrize("native", [False, True])
def test_meta_projection_shape_and_dtype(native):
    arguments = device_arguments(values(17), torch.tensor([0, 16, 17]), native)
    output = invoke([tensor.to("meta") for tensor in arguments], native)
    assert output.device.type == "meta"
    assert output.shape == (17, 128)
    assert output.dtype == torch.float16
