# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host contracts for the native W4 down-projection route reduction."""

from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEDULE = REPO_ROOT / "csrc/gmm/qwen_w4_a8_int4_matmul_v310/op_kernel/native_int4_schedule.h"
OP_ROOT = REPO_ROOT / "csrc/gmm/qwen_w4_a8_int4_down_reduce_v310"
MODEL = REPO_ROOT / "vllm_ascend/models/qwen4_exp/w4_moe.py"


def ordered_reference(
    route_outputs: torch.Tensor, route_ids: torch.Tensor, route_weights: torch.Tensor, experts: int
) -> torch.Tensor:
    """Mirror the fused epilogue, including its existing FP16 boundary."""
    tokens, top_k = route_weights.shape
    staged = route_outputs.to(torch.float16)
    result = torch.zeros(tokens, route_outputs.shape[1], dtype=torch.float32)
    for token in range(tokens):
        for route in range(top_k):
            slot = token * top_k + route
            if 0 <= route_ids[slot] < experts:
                result[token] += staged[slot].float() * route_weights[token, route]
    return result


def test_reference_preserves_fp16_projection_fp32_weight_and_slot_order():
    outputs = torch.tensor(
        [[1.0003, -0.7503], [1000.4, -1000.4], [0.3334, 0.6668], [float("nan"), float("nan")]],
        dtype=torch.float32,
    )
    ids = torch.tensor([0, 1, 0, -1], dtype=torch.int32)
    # These weights deliberately lose information if converted to FP16.
    weights = torch.tensor([[0.33331, 0.33337, 0.33343, 0.00019]], dtype=torch.float32)
    actual = ordered_reference(outputs, ids, weights, experts=2)
    staged = outputs.half().float()
    expected = torch.zeros(1, 2, dtype=torch.float32)
    for slot in range(3):
        expected[0] += staged[slot] * weights[0, slot]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    rounded_weight_result = torch.zeros_like(expected)
    for slot in range(3):
        rounded_weight_result[0] += staged[slot] * weights[0, slot].half().float()
    assert not torch.equal(actual, rounded_weight_result)
    assert torch.isfinite(actual).all(), "the nonlocal NaN row must be skipped before it is read"


def test_kernel_orders_ownership_rounding_weighting_and_reduction():
    source = SCHEDULE.read_text()
    combine = source.split("void CombineRoutes", 1)[1].split("void Store", 1)[0]
    ownership = "if (expert < 0 || expert >= experts_) continue;"
    promote = "Cast(values, staged[slot * N], RoundMode::CAST_NONE, N);"
    weight = "Muls(values, values, routeWeights_.GetValue(slot), N);"
    accumulate = "Add(accumulator, accumulator, values, N);"
    assert combine.index(ownership) < combine.index(promote) < combine.index(weight) < combine.index(accumulate)
    assert "for (uint32_t route = 0; route < topK_; ++route)" in combine
    assert "static_cast<half>(routeWeights" not in combine
    project = source.split("void Project", 1)[1]
    assert project.index("Cast(out, accumulator") < project.index("StageRoutes(out, row, count)")


def test_model_uses_one_fused_down_launch_for_c1_and_keeps_bounded_fallback():
    source = MODEL.read_text()
    routed = source.split("def _forward_routed", 1)[1].split("def _forward_host_routed", 1)[0]
    fused_guard = "if self.fused_native_down_reduce and tokens * self.top_k <= MAX_FUSED_DOWN_ROUTES:"
    fused_call = 'return self.projections["down_proj"].native_down_reduce(prepared_activation, local_ids, weights)'
    fallback_call = 'output = self.projections["down_proj"].native_linear(prepared_activation, local_ids)'
    tail = "output.reshape(tokens, self.top_k, hidden) * weights.unsqueeze(-1)"
    assert routed.index(fused_guard) < routed.index(fused_call) < routed.index(fallback_call) < routed.index(tail)
    assert "MAX_FUSED_DOWN_ROUTES = 30" in source


def test_operator_is_built_registered_and_returns_token_major_fp32():
    manifest = (REPO_ROOT / "csrc/build_aclnn.sh").read_text()
    binding = (REPO_ROOT / "csrc/torch_binding.cpp").read_text()
    meta = (REPO_ROOT / "csrc/torch_binding_meta.cpp").read_text()
    adapter = (OP_ROOT / "qwen_w4_a8_int4_down_reduce_310_torch_adpt.h").read_text()
    tiling = (OP_ROOT / "op_host/qwen_w4_a8_int4_down_reduce_v310_tiling.cpp").read_text()
    assert '"qwen_w4_a8_int4_down_reduce_v310"' in manifest
    assert '"qwen_w4_a8_swiglu_pack_v310"' in manifest
    assert 'ops.def("npu_qwen_w4_a8_int4_down_reduce_310' in binding
    assert 'ops.def("npu_qwen_w4_a8_swiglu_pack_310' in binding
    assert 'ops.impl("npu_qwen_w4_a8_int4_down_reduce_310"' in meta
    assert 'ops.impl("npu_qwen_w4_a8_swiglu_pack_310"' in meta
    assert "route_weights.scalar_type() == at::kFloat" in adapter
    assert "at::empty({tokens, OUTPUTS}, low.options().dtype(at::kFloat))" in adapter
    assert "routes != tokens * topK || activationRows != routes" in tiling
    assert "data.set_broadcastFactor(1)" in tiling
    assert "context->SetBlockDim(blocks)" in tiling


def test_gate_keeps_retained_projection_and_mtp_layouts():
    model = (REPO_ROOT / "vllm_ascend/models/qwen4_exp/model.py").read_text()
    mtp = (REPO_ROOT / "vllm_ascend/models/qwen4_exp/mtp.py").read_text()
    assert "enable_dynamic_w8a8_lm_head" in model
    assert "enable_dynamic_w8a8_lm_head" in mtp
    assert "in_proj_qkvzba" not in model
    assert "qkvgik_proj" not in model
    assert '"input_mix_weight_down_block_inject"' in model
    assert "mtp_grouped_w8a16" not in mtp
    assert "npu_qwen_mtp_grouped_w8a16_310" not in mtp
