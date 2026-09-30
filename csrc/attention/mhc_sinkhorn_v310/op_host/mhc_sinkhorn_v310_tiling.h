#ifndef MHC_SINKHORN_V310_TILING_H
#define MHC_SINKHORN_V310_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MhcSinkhornV310TilingData)
    TILING_DATA_FIELD_DEF(int64_t, numRows);
    TILING_DATA_FIELD_DEF(int64_t, rowsPerCore);
    TILING_DATA_FIELD_DEF(int64_t, iterations);
    TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MhcSinkhornV310, MhcSinkhornV310TilingData)
}  // namespace optiling

#endif
