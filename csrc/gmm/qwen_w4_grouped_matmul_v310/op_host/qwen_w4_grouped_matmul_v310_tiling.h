// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef QWEN_W4_GROUPED_MATMUL_TILING_H
#define QWEN_W4_GROUPED_MATMUL_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(QwenW4GroupedMatmulTilingData)
TILING_DATA_FIELD_DEF(int64_t, numRows);
TILING_DATA_FIELD_DEF(int64_t, numExperts);
TILING_DATA_FIELD_DEF(int64_t, nDim);
TILING_DATA_FIELD_DEF(int64_t, kDim);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(QwenW4GroupedMatmulV310, QwenW4GroupedMatmulTilingData)
}  // namespace optiling
#endif
