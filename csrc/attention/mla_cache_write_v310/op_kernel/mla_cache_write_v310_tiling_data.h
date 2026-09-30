#ifndef MLA_CACHE_WRITE_V310_TILING_DATA_H
#define MLA_CACHE_WRITE_V310_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

struct MlaCacheWriteV310TilingData {
    int64_t numRows;
    int64_t channelBlocks;
    int64_t blockSize;
    int64_t pageStride;
    int64_t rowsPerCore;
};

#endif
