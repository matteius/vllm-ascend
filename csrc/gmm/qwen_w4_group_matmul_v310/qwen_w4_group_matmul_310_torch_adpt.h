/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef QWEN_W4_GROUP_MATMUL_V310_TORCH_ADPT_H
#define QWEN_W4_GROUP_MATMUL_V310_TORCH_ADPT_H
namespace vllm_ascend {

// out[T, N] (fp16) = x[T, K] (fp16) @ W^T,
// W[n,k] = (unpack_signed(codes)[n,k] - offset[n,k/128]) * scale[n,k/128].
// Canonical signed int8 codes [N,K/2], low nibble first; FP16 scale and int8
// offset. tiled=true instead interprets pack_cube_tiles' lossless NZ encoding
// (paired N channels, equally biased codes/offsets) with the same shapes.
at::Tensor npu_qwen_w4_group_matmul_310(const at::Tensor& x, const at::Tensor& codes, const at::Tensor& scale,
                                        const at::Tensor& offset, bool tiled) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
  TORCH_CHECK(codes.device() == x.device() && scale.device() == x.device() && offset.device() == x.device(),
              "all Qwen W4 inputs must be on the same NPU");
  TORCH_CHECK(x.scalar_type() == at::kHalf && scale.scalar_type() == at::kHalf && codes.scalar_type() == at::kChar &&
                  offset.scalar_type() == at::kChar,
              "Qwen W4 expects FP16 x/scale and INT8 codes/offset");
  TORCH_CHECK(x.dim() == 2 && codes.dim() == 2 && scale.dim() == 2 && offset.dim() == 2,
              "Qwen W4 inputs must be matrices");
  TORCH_CHECK(x.is_contiguous() && codes.is_contiguous() && scale.is_contiguous() && offset.is_contiguous(),
              "Qwen W4 inputs must be contiguous ND tensors");
  TORCH_CHECK(x.size(0) > 0 && x.size(0) <= 128 && x.size(1) >= 256 && x.size(1) <= 2560 && x.size(1) % 128 == 0 &&
                  codes.size(0) > 0 && codes.size(0) % 128 == 0 && codes.size(1) * 2 == x.size(1),
              "unsupported Qwen W4 matrix dimensions");
  TORCH_CHECK(scale.size(0) == codes.size(0) && scale.size(1) == x.size(1) / 128 && offset.sizes() == scale.sizes(),
              "Qwen W4 scale/offset must be [N,K/128]");
  const c10_npu::OptionalNPUGuard npu_guard(x.device());
  at::Tensor out = at::empty({x.size(0), codes.size(0)}, x.options());
  EXEC_NPU_CMD(aclnnQwenW4GroupMatmulV310, x, codes, scale, offset, tiled, out);
  return out;
}

}  // namespace vllm_ascend
#endif  // QWEN_W4_GROUP_MATMUL_V310_TORCH_ADPT_H
