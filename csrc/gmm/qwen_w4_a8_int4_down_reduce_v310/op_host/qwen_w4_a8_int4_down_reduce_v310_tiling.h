// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef QWEN_W4A8_INT4_DOWN_REDUCE_TILING_H
#define QWEN_W4A8_INT4_DOWN_REDUCE_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(QwenW4A8Int4DownReduceTilingData)
TILING_DATA_FIELD_DEF(int64_t, numRows);
TILING_DATA_FIELD_DEF(int64_t, numExperts);
TILING_DATA_FIELD_DEF(int64_t, nDim);
TILING_DATA_FIELD_DEF(int64_t, kDim);
TILING_DATA_FIELD_DEF(int64_t, metadataLanes);
TILING_DATA_FIELD_DEF(int64_t, routed);
TILING_DATA_FIELD_DEF(int64_t, broadcastFactor);
TILING_DATA_FIELD_DEF(int64_t, topK);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(QwenW4A8Int4DownReduceV310, QwenW4A8Int4DownReduceTilingData)
}  // namespace optiling
#endif
