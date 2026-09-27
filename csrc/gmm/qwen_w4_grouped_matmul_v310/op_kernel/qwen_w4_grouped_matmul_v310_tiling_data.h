// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef QWEN_W4_GROUPED_MATMUL_TILING_DATA_H
#define QWEN_W4_GROUPED_MATMUL_TILING_DATA_H
#include <cstdint>
struct QwenW4GroupedKernelTilingData {
  int64_t numRows;
  int64_t numExperts;
  int64_t nDim;
  int64_t kDim;
};
#endif
