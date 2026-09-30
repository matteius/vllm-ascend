#ifndef QWEN4EXP_PLE_DECODE_V310_TILING_DATA_H
#define QWEN4EXP_PLE_DECODE_V310_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

struct Qwen4ExpPleDecodeV310TilingData {
    int64_t numTokens;
    int64_t hcGroups;
    int64_t groupSize;
    int64_t tasksPerCore;
    int64_t reduceWidth;
    float normEps;
    float invGroupSize;
    float invSqrtGroupSize;
};

#endif
