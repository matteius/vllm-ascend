#include "mhc_sinkhorn_v310.h"

extern "C" __global__ __aicore__ void mhc_sinkhorn_v310(
    GM_ADDR logits, GM_ADDR mix, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MhcSinkhornV310TilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcSinkhornV310TilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    NsMhcSinkhorn::MhcSinkhornV310 op;
    op.Init(logits, mix, &tilingData, &pipe);
    op.Process();
}
