#include "mla_cache_write_v310_tiling.h"

#include <algorithm>

#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
namespace {

constexpr int64_t NZ_INNER = 16;
constexpr int64_t MAX_LATENT_HEAD_DIM = 4096;

ge::graphStatus Tiling(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t coreCount = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreCount == 0, OP_LOGE(context, "AIV core count is zero"), return ge::GRAPH_FAILED);

    const auto cache = context->GetInputShape(0)->GetStorageShape();
    const auto rows = context->GetInputShape(1)->GetStorageShape();
    const auto slots = context->GetInputShape(2)->GetStorageShape();
    OP_CHECK_IF(cache.GetDimNum() != 4 || cache.GetDim(3) != NZ_INNER,
                OP_LOGE(context, "cache must be [blocks,head_dim/16,block_size,16]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(rows.GetDimNum() != 3 || rows.GetDim(2) != NZ_INNER,
                OP_LOGE(context, "rows must be [tokens,head_dim/16,16]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(slots.GetDimNum() != 1 || slots.GetDim(0) != rows.GetDim(0),
                OP_LOGE(context, "slot mapping must be [tokens]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(rows.GetDim(1) != cache.GetDim(1),
                OP_LOGE(context, "row and cache latent widths differ"), return ge::GRAPH_FAILED);
    const int64_t headDim = rows.GetDim(1) * NZ_INNER;
    OP_CHECK_IF(headDim <= 0 || headDim > MAX_LATENT_HEAD_DIM,
                OP_LOGE(context, "latent head dimension must be in [16,4096]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cache.GetDim(2) <= 0 || cache.GetDim(2) > 65535,
                OP_LOGE(context, "cache block size is outside the DMA stride range"),
                return ge::GRAPH_FAILED);

    const int64_t *pageStride = context->GetAttrs()->GetInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, pageStride);
    const int64_t logicalPageStride = cache.GetDim(1) * cache.GetDim(2) * NZ_INNER;
    OP_CHECK_IF(*pageStride < logicalPageStride,
                OP_LOGE(context, "page stride is smaller than the logical cache page"),
                return ge::GRAPH_FAILED);

    const int64_t numRows = rows.GetDim(0);
    const uint32_t blockDim = static_cast<uint32_t>(std::max<int64_t>(
        1, std::min<int64_t>(numRows, coreCount)));
    MlaCacheWriteV310TilingData data;
    data.set_numRows(numRows);
    data.set_channelBlocks(rows.GetDim(1));
    data.set_blockSize(cache.GetDim(2));
    data.set_pageStride(*pageStride);
    data.set_rowsPerCore((numRows + blockDim - 1) / blockDim);

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

IMPL_OP_OPTILING(MlaCacheWriteV310).Tiling(Tiling).TilingParse<CompileInfo>(Parse);

}  // namespace optiling
