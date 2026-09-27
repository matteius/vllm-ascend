// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <algorithm>
#include "qwen_w4_grouped_matmul_v310_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {
constexpr int64_t MAX_ROUTES = 5120;
constexpr int64_t GROUP_SIZE = 128;
constexpr int64_t OUTPUT_TILE = 32;
constexpr int64_t MIN_K = 256;
constexpr int64_t MAX_K = 2560;

static ge::graphStatus TileQwenW4Grouped(gert::TilingContext* context) {
  auto platform = context->GetPlatformInfo();
  OP_CHECK_NULL_WITH_CONTEXT(context, platform);
  platform_ascendc::PlatformAscendC device(platform);
  for (uint32_t i = 0; i < 5; ++i) {
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(i));
  }
  const auto x = context->GetInputShape(0)->GetStorageShape();
  const auto codes = context->GetInputShape(1)->GetStorageShape();
  const auto scales = context->GetInputShape(2)->GetStorageShape();
  const auto offsets = context->GetInputShape(3)->GetStorageShape();
  const auto ends = context->GetInputShape(4)->GetStorageShape();
  OP_CHECK_IF(x.GetDimNum() != 2 || codes.GetDimNum() != 3 || scales.GetDimNum() != 3 || offsets.GetDimNum() != 3 ||
                  ends.GetDimNum() != 1,
              OP_LOGE(context, "expected x[R,K], banks[E,N,*], group_ends[E]"), return ge::GRAPH_FAILED);
  const int64_t rows = x.GetDim(0), k = x.GetDim(1), experts = codes.GetDim(0), n = codes.GetDim(1);
  OP_CHECK_IF(rows <= 0 || rows > MAX_ROUTES || experts <= 0 || n <= 0 || n % GROUP_SIZE != 0 || k < MIN_K ||
                  k > MAX_K || k % GROUP_SIZE != 0 || codes.GetDim(2) * 2 != k || ends.GetDim(0) != experts,
              OP_LOGE(context, "invalid Qwen W4 grouped dimensions"), return ge::GRAPH_FAILED);
  for (const auto& shape : {scales, offsets}) {
    OP_CHECK_IF(shape.GetDim(0) != experts || shape.GetDim(1) != n || shape.GetDim(2) != k / GROUP_SIZE,
                OP_LOGE(context, "W4 metadata must be [E,N,K/128]"), return ge::GRAPH_FAILED);
  }
  const uint32_t cores = device.GetCoreNumAic();
  OP_CHECK_IF(cores == 0, OP_LOGE(context, "no AI cores"), return ge::GRAPH_FAILED);
  const uint32_t blocks = std::min<int64_t>(cores, experts * (n / OUTPUT_TILE));
  QwenW4GroupedMatmulTilingData data;
  data.set_numRows(rows);
  data.set_numExperts(experts);
  data.set_nDim(n);
  data.set_kDim(k);
  auto workspace = context->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
  // Per physical block, never per route. Tiled dequantization stays in L1.
  workspace[0] = device.GetLibApiWorkSpaceSize() + blocks * OUTPUT_TILE * k * sizeof(uint16_t);
  context->SetBlockDim(blocks);
  context->SetTilingKey(0);
  data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(data.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus ParseQwenW4Grouped(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
struct QwenW4GroupedCompileInfo {};
IMPL_OP_OPTILING(QwenW4GroupedMatmulV310)
    .Tiling(TileQwenW4Grouped)
    .TilingParse<QwenW4GroupedCompileInfo>(ParseQwenW4Grouped);
}  // namespace optiling
