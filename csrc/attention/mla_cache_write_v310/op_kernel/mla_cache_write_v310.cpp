#include "mla_cache_write_v310.h"

extern "C" __global__ __aicore__ void mla_cache_write_v310(
    GM_ADDR cache, GM_ADDR rows, GM_ADDR slotMapping, GM_ADDR cacheOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MlaCacheWriteV310TilingData);
    GET_TILING_DATA_WITH_STRUCT(MlaCacheWriteV310TilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    NsMlaCacheWrite::MlaCacheWriteV310 op;
    op.Init(cache, rows, slotMapping, cacheOut, &tilingData, &pipe);
    op.Process();
}
