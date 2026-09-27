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
 * \file qwen_w4_group_matmul_v310_tiling.cpp
 * \brief
 */

#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_util.h"
#include "qwen_w4_group_matmul_v310_tiling.h"

namespace optiling {

constexpr uint32_t X_INDEX = 0;
constexpr uint32_t CODES_INDEX = 1;
constexpr uint32_t SCALE_INDEX = 2;
constexpr uint32_t OFFSET_INDEX = 3;
constexpr int64_t OUTPUT_TILE = 32;
constexpr int64_t OUTPUT_ALIGNMENT = 128;
constexpr int64_t INPUT_TILE = 128;
constexpr int64_t MIN_INPUT_DIM = 256;
constexpr int64_t MAX_INPUT_DIM = 2560;
constexpr int64_t MAX_TOKENS = 128;

static ge::graphStatus QwenW4GroupMatmulTilingFunc(gert::TilingContext* context) {
  auto platformInfoPtr = context->GetPlatformInfo();
  OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  uint32_t coreNum = ascendcPlatform.GetCoreNumAic();
  OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "aivCoreNum is 0"), return ge::GRAPH_FAILED);

  auto xShapePtr = context->GetInputShape(X_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
  auto codesShapePtr = context->GetInputShape(CODES_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context, codesShapePtr);

  auto xShape = xShapePtr->GetStorageShape();
  auto codesShape = codesShapePtr->GetStorageShape();
  OP_CHECK_IF(xShape.GetDimNum() != 2, OP_LOGE(context, "x must be 2D [T, K]"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(codesShape.GetDimNum() != 2, OP_LOGE(context, "codes must be 2D [N, K]"), return ge::GRAPH_FAILED);

  const int64_t T = xShape.GetDim(0);
  const int64_t K = xShape.GetDim(1);
  const int64_t N = codesShape.GetDim(0);
  const int64_t packedK = codesShape.GetDim(1);
  OP_CHECK_IF(packedK <= 0 || K % packedK != 0, OP_LOGE(context, "codes.shape[1] must divide x.shape[1] (K)"),
              return ge::GRAPH_FAILED);
  const int64_t codesPerByte = K / packedK;
  OP_CHECK_IF(codesPerByte != 2, OP_LOGE(context, "packed codes must contain 2 signed W4 values per byte"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(T <= 0 || N <= 0 || K <= 0, OP_LOGE(context, "T/N/K must be positive"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(N % OUTPUT_ALIGNMENT != 0 || K % INPUT_TILE != 0 || K < MIN_INPUT_DIM,
              OP_LOGE(context, "N and K must be multiples of 128 and K must be at least 256"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(T > MAX_TOKENS || K > MAX_INPUT_DIM, OP_LOGE(context, "Qwen W4 kernel supports T <= 128 and K <= 2560"),
              return ge::GRAPH_FAILED);
  for (uint32_t index : {SCALE_INDEX, OFFSET_INDEX}) {
    auto metadataPtr = context->GetInputShape(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, metadataPtr);
    auto metadata = metadataPtr->GetStorageShape();
    OP_CHECK_IF(metadata.GetDimNum() != 2 || metadata.GetDim(0) != N || metadata.GetDim(1) != K / INPUT_TILE,
                OP_LOGE(context, "scale and offset must be [N,K/128]"), return ge::GRAPH_FAILED);
  }

  // Canonical CANN tiling-data pattern (local optiling object + SaveToBuffer),
  // required once workspace > 0 so the RunForWorkspace probe path works.
  QwenW4GroupMatmulTilingData tilingData;
  tilingData.set_numTokens(T);
  tilingData.set_nDim(N);
  tilingData.set_kDim(K);
  tilingData.set_codesPerByte(codesPerByte);
  auto attrs = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  auto tiled = attrs->GetAttrPointer<bool>(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, tiled);
  tilingData.set_tiled(*tiled ? 1 : 0);

  const int64_t nBlocks = (N + OUTPUT_TILE - 1) / OUTPUT_TILE;
  uint32_t blockDim = static_cast<uint32_t>(nBlocks);

  // One logical block owns each output tile. The runtime schedules excess
  // blocks over the physical cores, while every block starts with clean
  // Cube pipeline state and a private [32,K] packed-NZ workspace. Splitting
  // the actual N=640 gate/up projection into 20 rather than five tiles
  // distributes unpacking work more evenly across physical AI cores.
  // Scratch totals N*K*2 bytes for this invocation only, not a persistent
  // FP16 expert bank. Producing NZ directly avoids an extra ND-to-NZ pass.
  // The synchronized Cube epilogue casts FP32 accumulation to FP16 output.
  const size_t wdqBytes =
      static_cast<size_t>(nBlocks) * static_cast<size_t>(OUTPUT_TILE) * static_cast<size_t>(K) * sizeof(uint16_t);
  // GetUserWorkspace(workspace) returns (workspace + GetLibApiWorkSpaceSize()),
  // so the reported size must include that system reserve or the kernel writes
  // run past the allocation (invalid GM address).
  const size_t sysRsv = ascendcPlatform.GetLibApiWorkSpaceSize();
  size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
  currentWorkspace[0] = sysRsv + wdqBytes;

  context->SetBlockDim(blockDim);
  context->SetTilingKey(0);

  tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForQwenW4GroupMatmul(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }

struct QwenW4GroupMatmulCompileInfo {
  uint64_t ubSize = 0;
  uint32_t coreNum = 0;
};

IMPL_OP_OPTILING(QwenW4GroupMatmulV310)
    .Tiling(QwenW4GroupMatmulTilingFunc)
    .TilingParse<QwenW4GroupMatmulCompileInfo>(TilingParseForQwenW4GroupMatmul);
}  // namespace optiling
