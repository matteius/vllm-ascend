# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""310P shared-query RoPE replay must read current positions and activations."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY
from vllm_ascend.models.qwen4_exp.model import _QSAAttention
from vllm_ascend.models.qwen4_exp.qsa import _mrope_interleaved_dims, apply_partial_rope, partial_rope_cos_sin
from vllm_ascend.models.qwen4_exp.w4_moe import FORMAT


@pytest.fixture(autouse=True)
def require_310p():
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)


@pytest.mark.parametrize("tokens", [1, 2, 3, 5, 8, 512])
@pytest.mark.parametrize("mrope", [False, True])
def test_shared_query_tables_replay_matches_separate_tables(tokens, mrope):
    generator = torch.Generator().manual_seed(1024)
    values = [
        torch.randn(tokens, heads, width, generator=generator).half().npu()
        for heads, width in ((6, 256), (1, 256), (4, 128))
    ]
    positions = torch.arange(tokens, dtype=torch.int64).npu()
    kwargs = {}
    frequency_axes = None
    if mrope:
        positions = torch.stack((positions, positions, positions))
        kwargs = {"mrope_section": [16, 8, 8], "mrope_interleaved": True}
        frequency_axes = torch.tensor(_mrope_interleaved_dims([16, 8, 8]), dtype=torch.int64, device="npu")

    def shared_rotations():
        tables = partial_rope_cos_sin(
            positions, rotary_dim=64, base=10000.0, dtype=torch.float32, frequency_axes=frequency_axes, **kwargs
        )
        return tuple(
            apply_partial_rope(value, positions, 64, 10000.0, torch.float32, cos_sin=tables, **kwargs)
            for value in values
        )

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            shared_rotations()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        outputs = shared_rotations()
    for start in (0, 23400, 163840, 1048576):
        cpu_positions = torch.arange(start, start + tokens, dtype=torch.int64)
        if mrope:
            cpu_positions = torch.stack((cpu_positions, cpu_positions.flip(0), cpu_positions % 17))
        positions.copy_(cpu_positions)
        for value in values:
            value.copy_(torch.randn(value.shape, generator=generator).half())
        expected = tuple(apply_partial_rope(value, positions, 64, 10000.0, torch.float32, **kwargs) for value in values)
        graph.replay()
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual.cpu(), reference.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("backend", [None, "eager_dequant", "cube_310", "cube_310_tiled", "cube_310_routed"])
def test_query_table_reuse_is_only_enabled_for_alternate_routed_w4(backend):
    config = SimpleNamespace(
        hidden_size=256,
        moe_intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=0.5,
        rope_parameters={"mrope_section": [2, 1, 1], "mrope_interleaved": True},
    )
    if backend is not None:
        config.ascend_expert_quantization = {
            "backend": backend,
            "format": FORMAT,
            "bits": 4,
            "group_size": 128,
            "packing": "signed_int4_low_nibble_first_in_axis",
            "symmetric": False,
            "scale_dtype": "float16",
            "offset_dtype": "int8",
        }
    layer = _QSAAttention(config=config, layer_idx=0, dtype_policy=ASCEND_QWEN4EXP_DTYPE_POLICY)
    assert layer.reuse_query_rope is (backend == "cube_310_routed")
    axes = layer._query_rope_frequency_axes
    assert "_query_rope_frequency_axes" not in layer.state_dict()
    if backend == "cube_310_routed":
        assert "_query_rope_frequency_axes" in dict(layer.named_buffers())
        torch.testing.assert_close(axes.cpu(), torch.tensor(_mrope_interleaved_dims([2, 1, 1])))
    else:
        assert axes is None
