#include "mhc_sinkhorn_v310_tiling.h"

#include <algorithm>
#include <cmath>

#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
namespace {

constexpr int64_t MHC_STREAMS = 4;
constexpr int64_t MAX_ITERATIONS = 64;

ge::graphStatus Tiling(gert::TilingContext *context)
{
    const auto logits = context->GetInputShape(0)->GetStorageShape();
    OP_CHECK_IF(logits.GetDimNum() != 3 ||
                    logits.GetDim(1) != MHC_STREAMS ||
                    logits.GetDim(2) != MHC_STREAMS ||
                    logits.GetDim(0) <= 0,
                OP_LOGE(context, "logits must be [tokens,4,4]"),
                return ge::GRAPH_FAILED);

    const int64_t *iterations = context->GetAttrs()->GetInt(0);
    const float *epsilon = context->GetAttrs()->GetFloat(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, iterations);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsilon);
    OP_CHECK_IF(*iterations < 1 || *iterations > MAX_ITERATIONS ||
                    !std::isfinite(*epsilon) || *epsilon < 0.0f,
                OP_LOGE(context, "invalid Sinkhorn iterations or epsilon"),
                return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t coreCount = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreCount == 0,
                OP_LOGE(context, "AIV core count is zero"),
                return ge::GRAPH_FAILED);
    const int64_t numRows = logits.GetDim(0);
    const uint32_t blockDim = static_cast<uint32_t>(
        std::min<int64_t>(numRows, coreCount));

    MhcSinkhornV310TilingData data;
    data.set_numRows(numRows);
    data.set_rowsPerCore((numRows + blockDim - 1) / blockDim);
    data.set_iterations(*iterations);
    data.set_epsilon(*epsilon);

    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);
    data.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Parse(gert::TilingParseContext *) { return ge::GRAPH_SUCCESS; }
struct CompileInfo {};

}  // namespace

IMPL_OP_OPTILING(MhcSinkhornV310).Tiling(Tiling).TilingParse<CompileInfo>(Parse);

}  // namespace optiling
