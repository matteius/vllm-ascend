// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef W2_GROUPED_BLOCKED_DEQUANT_MATMUL_V310_TILING_H
#define W2_GROUPED_BLOCKED_DEQUANT_MATMUL_V310_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(W2GroupedBlockedDequantMatmulTilingData)
TILING_DATA_FIELD_DEF(int64_t, numRows);
TILING_DATA_FIELD_DEF(int64_t, numExperts);
TILING_DATA_FIELD_DEF(int64_t, nDim);
TILING_DATA_FIELD_DEF(int64_t, kDim);
TILING_DATA_FIELD_DEF(int64_t, codesPerByte);
TILING_DATA_FIELD_DEF(int64_t, nzPacked);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(W2GroupedBlockedDequantMatmulV310, W2GroupedBlockedDequantMatmulTilingData)
}  // namespace optiling
#endif
