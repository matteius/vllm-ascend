// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef QWEN_W4_A8_SWIGLU_PACK_V310_TILING_H
#define QWEN_W4_A8_SWIGLU_PACK_V310_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(QwenW4A8SwigluPackTilingData)
TILING_DATA_FIELD_DEF(int64_t, rows);
TILING_DATA_FIELD_DEF(int64_t, groups_per_row);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(QwenW4A8SwigluPackV310, QwenW4A8SwigluPackTilingData)
}  // namespace optiling
#endif
