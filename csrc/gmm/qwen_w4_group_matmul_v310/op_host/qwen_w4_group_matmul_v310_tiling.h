/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qwen_w4_group_matmul_v310_tiling.h
 * \brief
 */

#ifndef ASCEND_OPS_QWEN_W4_GROUP_MATMUL_V310_TILING_H
#define ASCEND_OPS_QWEN_W4_GROUP_MATMUL_V310_TILING_H

#include <cstdint>

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "tiling_base/error_log.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(QwenW4GroupMatmulTilingData)
TILING_DATA_FIELD_DEF(int64_t, numTokens);     // T
TILING_DATA_FIELD_DEF(int64_t, nDim);          // N
TILING_DATA_FIELD_DEF(int64_t, kDim);          // K
TILING_DATA_FIELD_DEF(int64_t, codesPerByte);  // 2 signed W4 values per byte
TILING_DATA_FIELD_DEF(int64_t, tiled);         // lossless load-time NZ nibble encoding
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QwenW4GroupMatmulV310, QwenW4GroupMatmulTilingData)

}  // namespace optiling

#endif  // ASCEND_OPS_QWEN_W4_GROUP_MATMUL_V310_TILING_H
