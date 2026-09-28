// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef QWEN_W4A8_INT4_DOWN_REDUCE_TORCH_ADPT_H
#define QWEN_W4A8_INT4_DOWN_REDUCE_TORCH_ADPT_H
namespace vllm_ascend {
at::Tensor npu_qwen_w4_a8_int4_down_reduce_310(
    const at::Tensor& low, const at::Tensor& high, const at::Tensor& activation_scale,
    const at::Tensor& activation_sum, const at::Tensor& codes, const at::Tensor& scale, const at::Tensor& offset,
    const at::Tensor& weight_sum, const at::Tensor& route_ids, const at::Tensor& route_weights) {
  constexpr int64_t GROUP = 128, INPUTS = 640, OUTPUTS = 2560, METADATA_LANES = 8, MAX_ROUTES = 30;
  TORCH_CHECK(low.device().type() == c10::DeviceType::PrivateUse1, "fused native INT4 down-reduce requires NPU");
  for (const auto& tensor : {low, high, activation_scale, activation_sum, codes, scale, offset, weight_sum, route_ids,
                             route_weights}) {
    TORCH_CHECK(tensor.device() == low.device() && tensor.is_contiguous(),
                "fused native INT4 down-reduce inputs must be contiguous on one NPU");
  }
  TORCH_CHECK(low.scalar_type() == at::kChar && high.scalar_type() == at::kChar && codes.scalar_type() == at::kChar &&
                  activation_scale.scalar_type() == at::kFloat && activation_sum.scalar_type() == at::kFloat &&
                  scale.scalar_type() == at::kHalf && offset.scalar_type() == at::kHalf &&
                  weight_sum.scalar_type() == at::kHalf && route_ids.scalar_type() == at::kInt &&
                  route_weights.scalar_type() == at::kFloat,
              "invalid fused native INT4 down-reduce dtypes");
  TORCH_CHECK(low.dim() == 2 && high.dim() == 2 && activation_scale.dim() == 3 && activation_sum.dim() == 3 &&
                  codes.dim() == 3 && scale.dim() == 3 && offset.dim() == 3 && weight_sum.dim() == 3 &&
                  route_ids.dim() == 1 && route_weights.dim() == 2,
              "invalid fused native INT4 down-reduce ranks");
  const int64_t activation_rows = low.size(0), tokens = route_weights.size(0), experts = codes.size(0);
  const int64_t top_k = route_weights.size(1);
  const int64_t routes = route_ids.size(0);
  TORCH_CHECK(tokens > 0 && top_k > 0 && routes == tokens * top_k && activation_rows == routes &&
                  routes <= MAX_ROUTES && experts > 0 &&
                  low.size(1) * 2 == INPUTS && high.sizes() == low.sizes() && codes.size(1) == OUTPUTS &&
                  codes.size(2) * 2 == INPUTS && route_weights.size(0) == tokens,
              "unsupported fused native INT4 down-reduce dimensions");
  TORCH_CHECK(activation_scale.size(0) == activation_rows && activation_scale.size(1) == INPUTS / GROUP &&
                  activation_scale.size(2) == METADATA_LANES && activation_sum.sizes() == activation_scale.sizes() &&
                  scale.size(0) == experts && scale.size(1) == OUTPUTS && scale.size(2) == INPUTS / GROUP &&
                  offset.sizes() == scale.sizes() && weight_sum.sizes() == scale.sizes(),
              "fused native INT4 down-reduce metadata shape mismatch");
  const c10_npu::OptionalNPUGuard guard(low.device());
  auto out = at::empty({tokens, OUTPUTS}, low.options().dtype(at::kFloat));
  EXEC_NPU_CMD(aclnnQwenW4A8Int4DownReduceV310, low, high, activation_scale, activation_sum, codes, scale, offset,
               weight_sum, route_ids, route_weights, out);
  return out;
}
}  // namespace vllm_ascend
#endif
