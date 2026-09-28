// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "qwen_w4_a8_int4_down_reduce_v310_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {
constexpr int64_t GROUP_SIZE = 128;
constexpr int64_t MODEL_INPUTS = 640;
constexpr int64_t MODEL_OUTPUTS = 2560;
constexpr int64_t METADATA_LANES = 8;
constexpr int64_t OUTPUT_TILE = 320;
constexpr int64_t MAX_ROUTES = 30;

static ge::graphStatus TileQwenW4A8Int4DownReduce(gert::TilingContext* context) {
  auto platform = context->GetPlatformInfo();
  OP_CHECK_NULL_WITH_CONTEXT(context, platform);
  platform_ascendc::PlatformAscendC device(platform);
  for (uint32_t i = 0; i < 10; ++i) OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(i));
  const auto low = context->GetInputShape(0)->GetStorageShape();
  const auto high = context->GetInputShape(1)->GetStorageShape();
  const auto xs = context->GetInputShape(2)->GetStorageShape();
  const auto sums = context->GetInputShape(3)->GetStorageShape();
  const auto codes = context->GetInputShape(4)->GetStorageShape();
  const auto scales = context->GetInputShape(5)->GetStorageShape();
  const auto offsets = context->GetInputShape(6)->GetStorageShape();
  const auto weightSums = context->GetInputShape(7)->GetStorageShape();
  const auto routeIds = context->GetInputShape(8)->GetStorageShape();
  const auto routeWeights = context->GetInputShape(9)->GetStorageShape();
  OP_CHECK_IF(low.GetDimNum() != 2 || high.GetDimNum() != 2 || xs.GetDimNum() != 3 || sums.GetDimNum() != 3 ||
                  codes.GetDimNum() != 3 || scales.GetDimNum() != 3 || offsets.GetDimNum() != 3 ||
                  weightSums.GetDimNum() != 3 || routeIds.GetDimNum() != 1 || routeWeights.GetDimNum() != 2,
              OP_LOGE(context, "invalid fused W4 down-reduce ranks"), return ge::GRAPH_FAILED);
  const int64_t activationRows = low.GetDim(0);
  const int64_t tokens = routeWeights.GetDim(0);
  const int64_t experts = codes.GetDim(0);
  const int64_t routes = routeIds.GetDim(0);
  const int64_t topK = routeWeights.GetDim(1);
  const int64_t groups = MODEL_INPUTS / GROUP_SIZE;
  OP_CHECK_IF(tokens <= 0 || topK <= 0 || routes != tokens * topK || activationRows != routes ||
                  routes > MAX_ROUTES || experts <= 0 || low.GetDim(1) * 2 != MODEL_INPUTS ||
                  high.GetDim(1) * 2 != MODEL_INPUTS || codes.GetDim(1) != MODEL_OUTPUTS ||
                  codes.GetDim(2) * 2 != MODEL_INPUTS,
              OP_LOGE(context, "invalid fused W4 down-reduce dimensions"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(high.GetDim(0) != activationRows || xs.GetDim(0) != activationRows || xs.GetDim(1) != groups ||
                  xs.GetDim(2) != METADATA_LANES || sums.GetDim(0) != activationRows || sums.GetDim(1) != groups ||
                  sums.GetDim(2) != METADATA_LANES,
              OP_LOGE(context, "invalid fused W4 activation metadata"), return ge::GRAPH_FAILED);
  for (const auto& shape : {scales, offsets, weightSums}) {
    OP_CHECK_IF(shape.GetDim(0) != experts || shape.GetDim(1) != MODEL_OUTPUTS || shape.GetDim(2) != groups,
                OP_LOGE(context, "invalid fused W4 weight metadata"), return ge::GRAPH_FAILED);
  }
  const uint32_t blocks = MODEL_OUTPUTS / OUTPUT_TILE;
  OP_CHECK_IF(device.GetCoreNumAic() < blocks, OP_LOGE(context, "fused W4 down-reduce requires eight AI cores"),
              return ge::GRAPH_FAILED);
  QwenW4A8Int4DownReduceTilingData data;
  data.set_numRows(routes);
  data.set_numExperts(experts);
  data.set_nDim(MODEL_OUTPUTS);
  data.set_kDim(MODEL_INPUTS);
  data.set_metadataLanes(METADATA_LANES);
  data.set_routed(1);
  data.set_broadcastFactor(1);
  data.set_topK(topK);
  auto workspace = context->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
  workspace[0] = device.GetLibApiWorkSpaceSize();
  context->SetBlockDim(blocks);
  context->SetTilingKey(0);
  data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(data.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus ParseQwenW4A8Int4DownReduce(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
struct QwenW4A8Int4DownReduceCompileInfo {};
IMPL_OP_OPTILING(QwenW4A8Int4DownReduceV310)
    .Tiling(TileQwenW4A8Int4DownReduce)
    .TilingParse<QwenW4A8Int4DownReduceCompileInfo>(ParseQwenW4A8Int4DownReduce);
}  // namespace optiling
