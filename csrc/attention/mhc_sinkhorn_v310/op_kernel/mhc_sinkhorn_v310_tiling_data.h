#ifndef MHC_SINKHORN_V310_TILING_DATA_H
#define MHC_SINKHORN_V310_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

struct MhcSinkhornV310TilingData {
    int64_t numRows;
    int64_t rowsPerCore;
    int64_t iterations;
    float epsilon;
};

#endif
