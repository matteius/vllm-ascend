#ifndef ASCEND_OPS_MLA_CACHE_WRITE_V310_TILING_H
#define ASCEND_OPS_MLA_CACHE_WRITE_V310_TILING_H

#include <cstdint>

#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MlaCacheWriteV310TilingData)
    TILING_DATA_FIELD_DEF(int64_t, numRows);
    TILING_DATA_FIELD_DEF(int64_t, channelBlocks);
    TILING_DATA_FIELD_DEF(int64_t, blockSize);
    TILING_DATA_FIELD_DEF(int64_t, pageStride);
    TILING_DATA_FIELD_DEF(int64_t, rowsPerCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MlaCacheWriteV310, MlaCacheWriteV310TilingData)

}  // namespace optiling

#endif
