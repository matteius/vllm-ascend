# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.ut.qwen38_1m.test_w4_moe import build, config
from vllm_ascend.models.qwen4_exp.lm_head_w8a8 import (
    Qwen4ExpDynamicW8A8LMHeadMethod,
    enable_dynamic_w8a8_lm_head,
    quantize_lm_head_weight,
)


def test_weight_quantization_is_per_output_channel_and_handles_zero_rows():
    weight = torch.tensor([[0.0, 0.0, 0.0], [-2.0, 0.25, 2.0]], dtype=torch.float16)
    quantized, scale = quantize_lm_head_weight(weight)
    assert quantized.dtype == torch.int8
    assert scale.dtype == torch.float32
    assert scale[0] == 1
    assert torch.equal(quantized[0], torch.zeros(3, dtype=torch.int8))
    torch.testing.assert_close(quantized[1], torch.tensor([-127, 16, 127], dtype=torch.int8))


def test_cpu_reference_uses_postload_quantized_weight():
    layer = torch.nn.Linear(4, 3, bias=False).half()
    source = torch.tensor(
        [[-1.0, -0.5, 0.5, 1.0], [0.0, 0.25, 0.5, 0.75], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float16,
    )
    layer.weight.data.copy_(source)
    method = Qwen4ExpDynamicW8A8LMHeadMethod()
    method.process_weights_after_loading(layer)
    x = torch.tensor([[1.0, -2.0, 0.5, 0.25]], dtype=torch.float16)
    x_scale = x.float().abs().amax(dim=-1, keepdim=True) / 127
    quantized_x = torch.round(x.float() / x_scale).clamp(-127, 127)
    expected = (quantized_x @ layer.weight.float()) * x_scale * layer.weight_scale.float()
    torch.testing.assert_close(method.apply(layer, x), expected.half())
    with pytest.raises(ValueError, match="does not support bias"):
        method.apply(layer, x, torch.zeros(3))


def test_model_selects_w8a8_head_only_from_explicit_metadata():
    fp16_cfg = config()
    fp16_model = build(fp16_cfg)
    assert not isinstance(fp16_model.lm_head.quant_method, Qwen4ExpDynamicW8A8LMHeadMethod)

    head = SimpleNamespace(
        embedding_dim=32,
        num_embeddings_per_partition=64,
        weight=torch.nn.Parameter(torch.zeros(64, 32, dtype=torch.float16), requires_grad=False),
        bias=None,
    )
    enable_dynamic_w8a8_lm_head(head, tied_embeddings=False)
    assert isinstance(head.quant_method, Qwen4ExpDynamicW8A8LMHeadMethod)


def test_w8a8_head_rejects_tied_embeddings():
    layer = SimpleNamespace()
    with pytest.raises(ValueError, match="tied word embeddings"):
        enable_dynamic_w8a8_lm_head(layer, tied_embeddings=True)

    layer = SimpleNamespace(
        embedding_dim=32,
        num_embeddings_per_partition=64,
        weight=torch.nn.Parameter(torch.zeros(64, 32, dtype=torch.float16), requires_grad=False),
        bias=None,
    )
    with pytest.raises(ValueError, match="LoRA"):
        enable_dynamic_w8a8_lm_head(layer, tied_embeddings=False, lora_enabled=True)

    layer.embedding_dim = 30
    with pytest.raises(ValueError, match="divisible by 16"):
        enable_dynamic_w8a8_lm_head(layer, tied_embeddings=False)


def test_postload_rejects_repeat_processing():
    layer = torch.nn.Linear(4, 3, bias=False).half()
    method = Qwen4ExpDynamicW8A8LMHeadMethod()
    method.process_weights_after_loading(layer)
    with pytest.raises(RuntimeError, match="already prepared"):
        method.process_weights_after_loading(layer)
