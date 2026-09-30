// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef W2_GROUPED_BLOCKED_DEQUANT_MATMUL_V310_TILING_DATA_H
#define W2_GROUPED_BLOCKED_DEQUANT_MATMUL_V310_TILING_DATA_H
#include <cstdint>
struct W2GroupedBlockedDequantMatmulKernelTilingData {
  int64_t numRows;
  int64_t numExperts;
  int64_t nDim;
  int64_t kDim;
  int64_t codesPerByte;
  int64_t nzPacked;
};
#endif
