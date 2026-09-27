# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical and replay gates for Qwen's asymmetric packed-W4 Cube op."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.models.qwen4_exp.dtype_policy import Qwen4ExpDtypePolicy
from vllm_ascend.models.qwen4_exp.model import _format_eager_linear_weights_npu
from vllm_ascend.models.qwen4_exp.w4_moe import FORMAT, PackedExpertBank, W4SparseMoE, pack_cube_tiles
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(autouse=True, scope="module")
def require_kernel():
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    # Match NPUWorker310P.init_device: aclop JIT MatMul cannot be captured.
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    assert hasattr(torch.ops._C_ascend, "npu_qwen_w4_group_matmul_310"), "rebuild the Qwen W4 custom operator"


def make_inputs(tokens, outputs, inputs, seed=3104):
    generator = torch.Generator().manual_seed(seed)
    packed = torch.randint(-128, 128, (outputs, inputs // 2), dtype=torch.int8, generator=generator)
    scales = (torch.rand(outputs, inputs // 128, generator=generator) * 0.01 + 0.001).half()
    offsets = torch.randint(-8, 8, scales.shape, dtype=torch.int8, generator=generator)
    x = (torch.randn(tokens, inputs, generator=generator) * 0.1).half()
    return x, packed, scales, offsets


def reference(x, packed, scales, offsets):
    # Independent unsigned-byte reconstruction, not the runtime's dequantizer.
    unsigned = packed.to(torch.int16) % 256
    nibbles = torch.stack((unsigned % 16, unsigned // 16), dim=-1).flatten(-2)
    signed = (nibbles + 8) % 16 - 8
    groups = signed.float().reshape(*scales.shape, 128)
    weight = ((groups - offsets.float().unsqueeze(-1)) * scales.float().unsqueeze(-1)).flatten(-2).half()
    return F.linear(x, weight)


def kernel(x, packed, scales, offsets, tiled=False):
    return torch.ops._C_ascend.npu_qwen_w4_group_matmul_310(x, packed, scales, offsets, tiled)


def layout_values(values, tiled):
    if not tiled:
        return values
    return [values[0]] + [
        pack_cube_tiles(value, kind) for value, kind in zip(values[1:], ["weight", "weight_scale", "weight_offset"])
    ]


@pytest.mark.parametrize("outputs,inputs", [(640, 2560), (2560, 640)])
@pytest.mark.parametrize("tokens", [1, 2, 5, 7, 64, 128])
@pytest.mark.parametrize("tiled", [False, True])
def test_projection_matches_independent_cpu_reference(tokens, outputs, inputs, tiled):
    values = make_inputs(tokens, outputs, inputs)
    expected = reference(*values)
    actual = kernel(*(value.npu() for value in layout_values(values, tiled)), tiled=tiled).cpu()
    torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.003)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("inputs", [256, 384, 512, 640, 768, 896, 1280, 1536, 2304, 2560])
def test_tiled_k_batch_selection_matches_reference(inputs):
    # Exercise full-K, the wide 1280 batch, 512/256 batches, and the 128
    # fallback. 384 is not a power of two; 896 must not read a padded tail.
    values = make_inputs(2, 128, inputs)
    actual = kernel(*(value.npu() for value in layout_values(values, True)), tiled=True).cpu()
    torch.testing.assert_close(actual, reference(*values), rtol=0.005, atol=0.003)


@pytest.mark.parametrize("tokens", [1, 15, 16, 17, 63, 64, 65, 127, 128])
@pytest.mark.parametrize("inputs", [640, 2560])
def test_tiled_graph_replay_fractal_boundaries(tokens, inputs):
    # Exercise the resident-L1 B stride, both activation stages, and the
    # partial last M fractal. Change weights as well as A on every replay:
    # no previous tile/route/graph invocation may survive in local memory.
    values = make_inputs(tokens, 128, inputs)
    device_values = [value.npu() for value in layout_values(values, True)]
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            kernel(*device_values, tiled=True)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = kernel(*device_values, tiled=True)
    for phase in range(5):
        values = make_inputs(tokens, 128, inputs, seed=3110 + phase)
        for destination, source in zip(device_values, layout_values(values, True)):
            destination.copy_(source)
        graph.replay()
        torch.testing.assert_close(output.cpu(), reference(*values), rtol=0.005, atol=0.003)


@pytest.mark.parametrize("tiled", [False, True])
def test_every_byte_and_offset_with_basis_inputs(tiled):
    # Each output is one dequantized weight, removing matmul cancellation as
    # a way for incorrect nibble order or zero-point sign to escape detection.
    packed = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8).repeat(128, 1)
    scales = torch.full((128, 4), 0.125, dtype=torch.float16)
    offsets = (torch.arange(128, dtype=torch.int16) % 16 - 8).to(torch.int8).unsqueeze(1).expand(-1, 4).contiguous()
    for start in range(0, 512, 128):
        x = torch.eye(512, dtype=torch.float16)[start : start + 128]
        expected = reference(x, packed, scales, offsets)
        actual = kernel(
            *(value.npu() for value in layout_values([x, packed, scales, offsets], tiled)), tiled=tiled
        ).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 2, 5])
@pytest.mark.parametrize("tiled", [False, True])
def test_graph_replay_uses_changed_inputs_and_weights(tokens, tiled):
    values = make_inputs(tokens, 640, 2560)
    device_values = [value.npu() for value in layout_values(values, tiled)]
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            kernel(*device_values, tiled=tiled)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = kernel(*device_values, tiled=tiled)
    for seed in [21, 17, 99]:
        replacements = make_inputs(tokens, 640, 2560, seed)
        for target, source in zip(device_values, layout_values(replacements, tiled)):
            target.copy_(source)
        graph.replay()
        actual = captured.cpu()
        torch.testing.assert_close(actual, reference(*replacements), rtol=0.005, atol=0.003)


def test_wide_unpack_basis_at_batch_and_group_boundaries():
    inputs, outputs = 2560, 128
    columns = [0, 127, 128, 639, 640, 1279, 1280, 2559]
    x = torch.eye(inputs, dtype=torch.float16)[columns]
    packed = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8).repeat(outputs, inputs // 512)
    scales = torch.full((outputs, inputs // 128), 0.125, dtype=torch.float16)
    offsets = (torch.arange(outputs, dtype=torch.int16) % 16 - 8).to(torch.int8).unsqueeze(1)
    offsets = offsets.expand_as(scales).contiguous()
    actual = kernel(*(v.npu() for v in layout_values([x, packed, scales, offsets], True)), tiled=True).cpu()
    torch.testing.assert_close(actual, reference(x, packed, scales, offsets), rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["tokens", "scale_shape", "offset_dtype", "packed_shape", "noncontiguous"])
def test_invalid_inputs_fail_before_device_launch(bad):
    values = list(make_inputs(2, 640, 2560))
    if bad == "tokens":
        values[0] = torch.zeros(129, 2560, dtype=torch.float16)
    elif bad == "scale_shape":
        values[2] = values[2][:, :-1].contiguous()
    elif bad == "offset_dtype":
        values[3] = values[3].float()
    elif bad == "packed_shape":
        values[1] = values[1][:, :-1].contiguous()
    values = [value.npu() for value in values]
    if bad == "noncontiguous":
        values[0] = values[0].t().contiguous().t()
    with pytest.raises(RuntimeError, match="Qwen W4|unsupported"):
        kernel(*values)


@pytest.mark.parametrize("tiled", [False, True])
def test_meta_shape(tiled):
    values = make_inputs(5, 640, 2560)
    output = kernel(*(value.to("meta") for value in values), tiled=tiled)
    assert output.shape == (5, 640)
    assert output.dtype == torch.float16


@pytest.mark.parametrize("experts", [1, 3])
@pytest.mark.parametrize("tiled", [False, True])
def test_packed_bank_cube_dispatch_keeps_packed_parameters(experts, tiled):
    values = make_inputs(7, 640, 2560)
    bank = PackedExpertBank(experts, 640, 2560, 128, backend="cube_310_tiled" if tiled else "cube_310").npu()
    with torch.no_grad():
        for name, value in zip(["weight", "weight_scale", "weight_offset"], layout_values(values, tiled)[1:]):
            getattr(bank, name)[experts - 1].copy_(value)
    # Nonzero expert offsets must not pass the whole bank's storage shape or
    # its base address to CANN (a failure mode of older 310P banked operators).
    actual = bank.linear(values[0].npu(), experts - 1).cpu()
    torch.testing.assert_close(actual, reference(*values), rtol=0.005, atol=0.003)
    assert set(dict(bank.named_parameters())) == {"weight", "weight_scale", "weight_offset"}
    assert bank.weight.dtype == torch.int8
    assert sum(p.numel() * p.element_size() for p in bank.parameters()) == experts * (640 * 2560 // 2 + 3 * 640 * 20)


def test_input_device_guard_restores_current_device():
    if torch.npu.device_count() < 2:
        pytest.skip("requires two visible NPUs")
    values = make_inputs(2, 640, 2560)
    device_values = [value.to("npu:1") for value in values]
    torch.npu.set_device(0)
    actual = kernel(*device_values)
    assert actual.device.index == 1
    torch.testing.assert_close(actual.cpu(), reference(*values), rtol=0.005, atol=0.003)
    assert torch.npu.current_device() == 0


def routed_values(routes, outputs, inputs, experts=3, seed=616):
    values = [make_inputs(routes, outputs, inputs, seed + expert) for expert in range(experts)]
    x = values[0][0]
    canonical = [torch.stack([expert[field] for expert in values]) for field in range(1, 4)]
    encoded = [
        torch.stack([pack_cube_tiles(expert, kind) for expert in bank])
        for bank, kind in zip(canonical, ["weight", "weight_scale", "weight_offset"])
    ]
    ids = (torch.arange(routes, dtype=torch.int32) % (experts + 2)) - 1
    return [x, *encoded, ids], canonical


def routed_kernel(*values):
    return torch.ops._C_ascend.npu_qwen_w4_routed_matmul_310(*values)


def routed_reference(x, canonical, ids):
    result = torch.zeros(x.shape[0], canonical[0].shape[1], dtype=torch.float16)
    for row, expert in enumerate(ids.tolist()):
        if 0 <= expert < canonical[0].shape[0]:
            result[row] = reference(x[row : row + 1], *(bank[expert] for bank in canonical))[0]
    return result


@pytest.mark.parametrize("outputs,inputs", [(640, 2560), (2560, 640)])
@pytest.mark.parametrize("routes", [1, 10, 20, 50, 80])
def test_routed_projection_local_duplicate_and_peer_rows(routes, outputs, inputs):
    values, canonical = routed_values(routes, outputs, inputs)
    values[-1][0] = 2  # Ensure the single-route test uses a nonzero expert.
    actual = routed_kernel(*(value.npu() for value in values)).cpu()
    torch.testing.assert_close(actual, routed_reference(values[0], canonical, values[-1]), rtol=0.005, atol=0.003)
    assert (actual[(values[-1] < 0) | (values[-1] >= 3)] == 0).all()


@pytest.mark.parametrize("outputs,inputs", [(640, 2560), (2560, 640)])
@pytest.mark.parametrize("routes", [10, 20, 50])
def test_routed_graph_changes_experts_in_both_directions(routes, outputs, inputs):
    values, _ = routed_values(routes, outputs, inputs)
    device_values = [value.npu() for value in values]
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            routed_kernel(*device_values)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = routed_kernel(*device_values)
    for seed in [21, 17, 99, 18]:
        replacements, canonical = routed_values(routes, outputs, inputs, seed=seed)
        replacements[-1] = (replacements[-1] + seed) % 5 - 1
        if seed == 99:
            replacements[-1].fill_(-1)  # All-peer must erase prior local output.
        for target, source in zip(device_values, replacements):
            target.copy_(source)
        graph.replay()
        actual = captured.cpu()
        torch.testing.assert_close(
            actual, routed_reference(replacements[0], canonical, replacements[-1]), rtol=0.005, atol=0.003
        )


@pytest.mark.parametrize("routes", [2, 3, 19, 20, 21, 30, 50, 80])
@pytest.mark.parametrize("outputs,inputs", [(128, 256), (640, 2560), (2560, 640)])
def test_routed_reuse_owner_changes_on_graph_replay(routes, outputs, inputs):
    values, canonical = routed_values(routes, outputs, inputs)
    device_values = [value.npu() for value in values]
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            routed_kernel(*device_values)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = routed_kernel(*device_values)
    # All duplicates, a later first owner, no local rows, and mixed owners.
    # Row activations differ, so copying the owner's result to peers is wrong.
    for phase in range(5):
        ids = torch.full((routes,), 2, dtype=torch.int32)
        if phase == 1:
            ids[0] = -1
        elif phase == 2:
            ids.fill_(3)
        elif phase == 3:
            ids = torch.arange(routes, dtype=torch.int32) % 3
        elif phase == 4:
            ids[0], ids[-1] = 0, 1
        device_values[-1].copy_(ids)
        graph.replay()
        torch.testing.assert_close(captured.cpu(), routed_reference(values[0], canonical, ids), rtol=0.005, atol=0.003)


@pytest.mark.parametrize("bad", ["route_limit", "ids_dtype", "ids_shape", "bank_shape", "scale_shape", "noncontiguous"])
def test_routed_invalid_input_fails_before_launch(bad):
    values, _ = routed_values(10, 128, 256)
    if bad == "route_limit":
        values[0], values[-1] = torch.zeros(81, 256).half(), torch.zeros(81, dtype=torch.int32)
    elif bad == "ids_dtype":
        values[-1] = values[-1].long()
    elif bad == "ids_shape":
        values[-1] = values[-1][:-1]
    elif bad == "bank_shape":
        values[1] = values[1][:, :, :-1].contiguous()
    elif bad == "scale_shape":
        values[2] = values[2][:, :, :-1].contiguous()
    values = [value.npu() for value in values]
    if bad == "noncontiguous":
        values[0] = values[0].t().contiguous().t()
    with pytest.raises(RuntimeError, match="W4 routed"):
        routed_kernel(*values)


@pytest.mark.parametrize("outputs,inputs", [(640, 2560), (2560, 640)])
@pytest.mark.parametrize("routes", [10, 20, 21, 30, 50, 80])
def test_routed_many_unique_owners_and_peer_transitions(routes, outputs, inputs):
    # Persistent N-tile tasks must refresh every expert's metadata and drain
    # their UB/Cube stores before processing the next local or peer row.
    # Three-expert duplicate tests alone do not exercise eighty unique owners.
    values, canonical = routed_values(routes, outputs, inputs, experts=routes)
    values[-1] = torch.arange(routes, dtype=torch.int32)
    device_values = [value.npu() for value in values]
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            routed_kernel(*device_values)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = routed_kernel(*device_values)
    for phase in range(8):
        ids = (torch.arange(routes, dtype=torch.int32) + phase) % routes
        if phase % 4 == 1:
            ids[::2] = -1
        elif phase % 4 == 2:
            ids[1::2] = routes
        elif phase % 4 == 3:
            ids.fill_(-1)
        device_values[-1].copy_(ids)
        graph.replay()
        torch.testing.assert_close(captured.cpu(), routed_reference(values[0], canonical, ids), rtol=0.005, atol=0.003)


def test_routed_meta_shape():
    values, _ = routed_values(20, 640, 2560)
    output = routed_kernel(*(value.to("meta") for value in values))
    assert output.shape == (20, 640) and output.dtype == torch.float16


@pytest.mark.parametrize("tokens", [1, 2, 5, 8])
@pytest.mark.parametrize("nz", [False, True])
def test_routed_complete_moe_matches_host_route_and_replays(tokens, nz):
    metadata = dict(
        format=FORMAT,
        bits=4,
        symmetric=False,
        group_size=128,
        packing="signed_int4_low_nibble_first_in_axis",
        scale_dtype="float16",
        offset_dtype="int8",
    )
    cfg = SimpleNamespace(
        hidden_size=256,
        moe_intermediate_size=256,
        num_experts=7,
        num_experts_per_tok=3,
        shared_expert_intermediate_size=32,
        ascend_expert_quantization=dict(metadata, backend="cube_310_routed"),
    )
    layer = W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy()).npu()
    cfg.ascend_expert_quantization = dict(metadata, backend="cube_310_tiled")
    host = W4SparseMoE(config=cfg, dtype_policy=Qwen4ExpDtypePolicy()).npu()
    with torch.no_grad():
        for expert in range(7):
            for projection in layer.projections:
                values = make_inputs(1, 256, 256, seed=417 + expert)
                for kind, value in zip(["weight", "weight_scale", "weight_offset"], values[1:]):
                    layer.load_projection(expert, projection, kind, value)
        for name in ["gate", "shared_gate_up", "shared_down", "shared_expert_gate"]:
            torch.manual_seed(910)
            getattr(layer, name).copy_((torch.randn_like(getattr(layer, name).cpu()) * 0.1).half())
        host.load_state_dict(layer.state_dict())
        if nz:
            # Actual model post-load hook converts router/shared weights to
            # NZ. This exposed an uncapturable FP16->FP32 shared-weight Cast.
            _format_eager_linear_weights_npu(layer)
    x = (torch.randn(tokens, 256) * 0.1).half().npu()
    with torch.inference_mode():
        torch.testing.assert_close(layer(x).cpu(), host(x).cpu(), rtol=0.01, atol=0.003)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                layer(x)
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            captured = layer(x)
        for seed in [812, 314, 219]:
            torch.manual_seed(seed)
            x.copy_((torch.randn(tokens, 256) * 0.1).half())
            expected = host(x).cpu()
            graph.replay()
            torch.testing.assert_close(captured.cpu(), expected, rtol=0.01, atol=0.003)
