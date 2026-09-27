// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef QWEN_W4_ROUTED_MATMUL_TORCH_ADPT_H
#define QWEN_W4_ROUTED_MATMUL_TORCH_ADPT_H
namespace vllm_ascend {
at::Tensor npu_qwen_w4_routed_matmul_310(const at::Tensor& x, const at::Tensor& codes, const at::Tensor& scale,
                                         const at::Tensor& offset, const at::Tensor& expert_ids) {
  constexpr int64_t MAX_ROUTES = 80, GROUP_SIZE = 128, MIN_K = 256, MAX_K = 2560;
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "W4 routed input must be on NPU");
  for (const auto& tensor : {codes, scale, offset, expert_ids}) {
    TORCH_CHECK(tensor.device() == x.device(), "all W4 routed inputs must be on the same NPU");
    TORCH_CHECK(tensor.is_contiguous(), "W4 routed banks/ids must be contiguous");
  }
  TORCH_CHECK(x.is_contiguous(), "W4 routed x must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kHalf && scale.scalar_type() == at::kHalf && codes.scalar_type() == at::kChar &&
                  offset.scalar_type() == at::kChar && expert_ids.scalar_type() == at::kInt,
              "W4 routed expects FP16 x/scale, INT8 codes/offset, INT32 expert_ids");
  TORCH_CHECK(x.dim() == 2 && codes.dim() == 3 && scale.dim() == 3 && offset.dim() == 3 && expert_ids.dim() == 1,
              "W4 routed expects x[R,K], banks[E,N,*], ids[R]");
  TORCH_CHECK(x.size(0) > 0 && x.size(0) <= MAX_ROUTES && expert_ids.size(0) == x.size(0) && x.size(1) >= MIN_K &&
                  x.size(1) <= MAX_K && x.size(1) % GROUP_SIZE == 0 && codes.size(0) > 0 && codes.size(1) > 0 &&
                  codes.size(1) % GROUP_SIZE == 0 && codes.size(2) * 2 == x.size(1),
              "unsupported W4 routed dimensions");
  TORCH_CHECK(scale.size(0) == codes.size(0) && scale.size(1) == codes.size(1) &&
                  scale.size(2) == x.size(1) / GROUP_SIZE && offset.sizes() == scale.sizes(),
              "W4 routed metadata must match [E,N,K/128]");
  const c10_npu::OptionalNPUGuard guard(x.device());
  auto out = at::empty({x.size(0), codes.size(1)}, x.options());
  EXEC_NPU_CMD(aclnnQwenW4RoutedMatmulV310, x, codes, scale, offset, expert_ids, out);
  return out;
}
}  // namespace vllm_ascend
#endif
