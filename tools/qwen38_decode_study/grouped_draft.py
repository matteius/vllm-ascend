# SPDX-License-Identifier: Apache-2.0
"""Experimental grouped W8A8 draft; activation quantization differs from W8A16.

Installed only into an isolated snapshot by prepare.py. The existing draft's
INT8 weights and rounded FP16 weight scales are retained. Real-checkpoint
acceptance, quality, empty-rank, and graph replay gates are mandatory.

The 2026-09-26 TP4 graph trial stalled with a device-memory fault. This
candidate is rejected for serving and retained only for failure reproduction.
"""

from __future__ import annotations

import torch
from torch import nn


def pack_draft_experts(bank, format_weight=None) -> None:
    """Pack once after loading, sharing storage with the original named weights."""
    if not bank.quantized_experts:
        raise ValueError("Grouped draft requires the live runtime's quantized MTP bank")
    if hasattr(bank, "draft_gate_up_packed"):
        raise RuntimeError("Draft weights have already been packed")
    if format_weight is None:
        # Worker-only dependency; CPU reference tests inject an identity format.
        from vllm_ascend.utils import maybe_trans_nz

        format_weight = maybe_trans_nz
    for projection, packed_name in (("gate_up_proj", "draft_gate_up"), ("down_proj", "draft_down")):
        weights = getattr(bank, projection)
        if len(weights) != bank.num_local_experts or any(weight.dtype != torch.int8 for weight in weights):
            raise ValueError("Expected one INT8 parameter per local draft expert")
        packed = format_weight(torch.stack([weight.t() for weight in weights]).contiguous())
        bank.register_buffer(packed_name + "_packed", packed, persistent=False)
        # Parameter names still satisfy vLLM's loaded-weight accounting, while
        # replacing the original storage avoids retaining two weight banks.
        setattr(
            bank, projection, nn.ParameterList([nn.Parameter(w.t(), requires_grad=False) for w in packed.unbind(0)])
        )
        scales = torch.stack(list(getattr(bank, projection + "_scale"))).float().contiguous()
        bank.register_buffer(packed_name + "_scale", scales, persistent=False)


def grouped_routed_experts(bank, x, weights, ids, ops=None):
    """Return FP32 routed outputs with device-only dispatch, including empty ranks.

    The operator dynamically quantizes x and the SwiGLU activation to INT8;
    this is deliberately a separate experimental arm, not exact W8A16 parity.
    """
    if ops is None:
        import torch_npu

        ops = torch_npu
    num_tokens, hidden = x.shape
    top_k = ids.shape[1]
    local_ids = ids.to(torch.int32) - bank.expert_offset
    local_ids = torch.where((local_ids >= 0) & (local_ids < bank.num_local_experts), local_ids, bank.num_local_experts)
    sorted_x, inverse_order, group_list, _ = ops.npu_moe_init_routing_v2(
        x,
        local_ids,
        active_num=num_tokens * top_k,
        expert_num=bank.num_local_experts,
        drop_pad_mode=0,
        active_expert_range=[0, bank.num_local_experts],
        quant_mode=-1,
        row_idx_type=0,
    )
    group_list = group_list.to(torch.int64)
    valid_rows = torch.arange(num_tokens * top_k, device=x.device) < group_list[-1]
    gate_up = ops.npu_quant_grouped_matmul_dequant(
        sorted_x, bank.draft_gate_up_packed, bank.draft_gate_up_scale, group_list
    )
    # CANN may leave peer-owned rows unwritten. Select zero rather than multiply
    # by zero so stale NaN/Inf values cannot contaminate the reduction.
    gate_up = torch.where(valid_rows[:, None], gate_up, 0)
    activated = ops.npu_swiglu(gate_up)
    routed = ops.npu_quant_grouped_matmul_dequant(activated, bank.draft_down_packed, bank.draft_down_scale, group_list)
    routed = torch.where(valid_rows[:, None], routed, 0)
    # Keep the existing draft's FP32 route-weight multiply and top-k slot order.
    original_order = routed.float().index_select(0, inverse_order)
    return (original_order * weights.float().reshape(-1, 1)).view(num_tokens, top_k, hidden).sum(dim=1)
