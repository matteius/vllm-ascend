#include "qwen4_exp_ple_decode_v310.h"

extern "C" __global__ __aicore__ void qwen4_exp_ple_decode_v310(
    GM_ADDR projected, GM_ADDR hidden,
    GM_ADDR normKeyWeight, GM_ADDR normQueryWeight,
    GM_ADDR normConvWeight, GM_ADDR currentConvWeight,
    GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(Qwen4ExpPleDecodeV310TilingData);
    GET_TILING_DATA_WITH_STRUCT(Qwen4ExpPleDecodeV310TilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    NsQwen4ExpPleDecode::Qwen4ExpPleDecodeV310 op;
    op.Init(projected, hidden, normKeyWeight, normQueryWeight,
            normConvWeight, currentConvWeight, output, &tilingData, &pipe);
    op.Process();
}
