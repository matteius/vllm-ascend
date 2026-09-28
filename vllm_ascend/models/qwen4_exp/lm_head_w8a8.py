# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit dynamic-W8A8 vocabulary projection for Qwen4Exp on 310P."""

from __future__ import annotations

import torch
from torch import nn
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

INT8_SYMMETRIC_MAX = 127
QUANT_MATMUL_ALIGNMENT = 16


def quantize_lm_head_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an ``[vocab, hidden]`` weight per output channel."""
    weight_fp32 = weight.to(torch.float32)
    absmax = weight_fp32.abs().amax(dim=1)
    scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax / INT8_SYMMETRIC_MAX)
    quantized = torch.round(weight_fp32 / scale.unsqueeze(1)).clamp(
        -INT8_SYMMETRIC_MAX,
        INT8_SYMMETRIC_MAX,
    )
    return quantized.to(torch.int8), scale


class Qwen4ExpDynamicW8A8LMHeadMethod(QuantizeMethodBase):
    """Quantize the loaded head once, then dynamically quantize activations."""

    def create_weights(self, layer: nn.Module, *weight_args, **extra_weight_attrs) -> None:
        raise RuntimeError("Qwen4Exp creates and loads its LM head before W8A8 conversion")

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if hasattr(layer, "weight_scale"):
            raise RuntimeError("dynamic-W8A8 LM head was already prepared")
        if layer.weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("dynamic-W8A8 LM head requires a floating-point checkpoint weight")
        quantized_weight, weight_scale = quantize_lm_head_weight(layer.weight.data)
        if quantized_weight.device.type == "npu":
            from vllm_ascend.utils import maybe_trans_nz

            quantized_weight = maybe_trans_nz(quantized_weight)
            runtime_weight = quantized_weight.transpose(0, 1)
        else:
            runtime_weight = quantized_weight.transpose(0, 1).contiguous()
        # QuantMatmul consumes a logical [hidden, local_vocab] right operand.
        layer.weight.requires_grad_(False)
        layer.weight.data = runtime_weight
        layer.register_parameter(
            "weight_scale",
            nn.Parameter(weight_scale, requires_grad=False),
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise ValueError("dynamic-W8A8 LM head does not support bias")
        if x.device.type != "npu":
            x_fp32 = x.to(torch.float32)
            absmax = x_fp32.abs().amax(dim=-1, keepdim=True)
            input_scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax / INT8_SYMMETRIC_MAX)
            quantized_x = torch.round(x_fp32 / input_scale).clamp(
                -INT8_SYMMETRIC_MAX,
                INT8_SYMMETRIC_MAX,
            )
            output = quantized_x @ layer.weight.to(torch.float32)
            return (output * input_scale * layer.weight_scale.to(torch.float32)).to(x.dtype)

        import torch_npu

        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        needs_unsqueeze = quantized_x.dim() == 3 and quantized_x.shape[1] == 1
        if needs_unsqueeze:
            quantized_x = quantized_x.squeeze(dim=1)
            pertoken_scale = pertoken_scale.squeeze(dim=1)
        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight.data,
            layer.weight_scale,
            pertoken_scale=pertoken_scale,
            bias=None,
            output_dtype=x.dtype,
        )
        return output.unsqueeze(dim=1) if needs_unsqueeze else output


def enable_dynamic_w8a8_lm_head(
    lm_head: nn.Module,
    *,
    tied_embeddings: bool,
    lora_enabled: bool = False,
) -> None:
    """Install the opt-in method without changing the checkpoint loader."""
    if tied_embeddings:
        raise ValueError("lm_head_execution=w8a8_dynamic is incompatible with tied word embeddings")
    if lora_enabled:
        raise ValueError("lm_head_execution=w8a8_dynamic does not support LoRA or runtime weight reload")
    hidden_size = int(lm_head.embedding_dim)
    local_vocab_size = int(lm_head.num_embeddings_per_partition)
    if hidden_size % QUANT_MATMUL_ALIGNMENT or local_vocab_size % QUANT_MATMUL_ALIGNMENT:
        raise ValueError(
            "lm_head_execution=w8a8_dynamic requires hidden and local vocabulary dimensions divisible by 16"
        )
    if lm_head.weight.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("lm_head_execution=w8a8_dynamic requires an FP16 or BF16 checkpoint weight")
    if getattr(lm_head, "bias", None) is not None:
        raise ValueError("lm_head_execution=w8a8_dynamic does not support bias")
    lm_head.quant_method = Qwen4ExpDynamicW8A8LMHeadMethod()
