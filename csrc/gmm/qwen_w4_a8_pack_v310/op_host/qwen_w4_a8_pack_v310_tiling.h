// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef QWEN_W4_A8_PACK_TILING_H
#define QWEN_W4_A8_PACK_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(QwenW4A8PackTilingData)
TILING_DATA_FIELD_DEF(int64_t, groups);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(QwenW4A8PackV310, QwenW4A8PackTilingData)
}  // namespace optiling
#endif
