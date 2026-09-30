#include "qwen4_exp_ple_decode_v310_tiling.h"

#include <algorithm>
#include <cmath>

#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
namespace {

constexpr int64_t VECTOR_ALIGNMENT = 16;
constexpr int64_t MAX_DECODE_TOKENS = 8;
constexpr int64_t MAX_GROUP_SIZE = 4096;

ge::graphStatus Tiling(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t coreCount = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreCount == 0, OP_LOGE(context, "AIV core count is zero"), return ge::GRAPH_FAILED);

    const auto projected = context->GetInputShape(0)->GetStorageShape();
    const auto hidden = context->GetInputShape(1)->GetStorageShape();
    OP_CHECK_IF(projected.GetDimNum() != 2 || hidden.GetDimNum() != 2,
                OP_LOGE(context, "projected and hidden must be rank-2"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(projected.GetDim(0) != hidden.GetDim(0) || projected.GetDim(1) <= hidden.GetDim(1),
                OP_LOGE(context, "PLE projected and hidden dimensions do not match"),
                return ge::GRAPH_FAILED);
    const int64_t numTokens = hidden.GetDim(0);
    const int64_t groupSize = projected.GetDim(1) - hidden.GetDim(1);
    OP_CHECK_IF(numTokens <= 0 || numTokens > MAX_DECODE_TOKENS,
                OP_LOGE(context, "PLE decode token count must be in [1,8]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupSize <= 0 || groupSize > MAX_GROUP_SIZE || groupSize % VECTOR_ALIGNMENT != 0,
                OP_LOGE(context, "PLE group size must be a multiple of 16 up to 4096"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(hidden.GetDim(1) % groupSize != 0,
                OP_LOGE(context, "PLE hidden size must be divisible by value width"), return ge::GRAPH_FAILED);
    const int64_t hcGroups = hidden.GetDim(1) / groupSize;

    for (int32_t input = 2; input <= 4; ++input) {
        const auto weight = context->GetInputShape(input)->GetStorageShape();
        OP_CHECK_IF(weight.GetDimNum() != 1 || weight.GetDim(0) != hidden.GetDim(1),
                    OP_LOGE(context, "PLE norm weights must match hidden width"), return ge::GRAPH_FAILED);
    }
    const auto convWeight = context->GetInputShape(5)->GetStorageShape();
    OP_CHECK_IF(convWeight.GetDimNum() != 1 || convWeight.GetDim(0) != hidden.GetDim(1),
                OP_LOGE(context, "PLE current convolution weight must match hidden width"),
                return ge::GRAPH_FAILED);

    const float *normEps = context->GetAttrs()->GetFloat(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, normEps);
    OP_CHECK_IF(*normEps <= 0.0f, OP_LOGE(context, "norm epsilon must be positive"), return ge::GRAPH_FAILED);

    const int64_t tasks = numTokens * hcGroups;
    int64_t reduceWidth = 1;
    while (reduceWidth * 2 <= groupSize) {
        reduceWidth *= 2;
    }
    const uint32_t blockDim = static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(tasks, coreCount)));
    Qwen4ExpPleDecodeV310TilingData data;
    data.set_numTokens(numTokens);
    data.set_hcGroups(hcGroups);
    data.set_groupSize(groupSize);
    data.set_tasksPerCore((tasks + blockDim - 1) / blockDim);
    data.set_reduceWidth(reduceWidth);
    data.set_normEps(*normEps);
    data.set_invGroupSize(1.0f / static_cast<float>(groupSize));
    data.set_invSqrtGroupSize(1.0f / std::sqrt(static_cast<float>(groupSize)));

    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Parse(gert::TilingParseContext *) { return ge::GRAPH_SUCCESS; }
struct CompileInfo {};

}  // namespace

IMPL_OP_OPTILING(Qwen4ExpPleDecodeV310).Tiling(Tiling).TilingParse<CompileInfo>(Parse);

}  // namespace optiling
