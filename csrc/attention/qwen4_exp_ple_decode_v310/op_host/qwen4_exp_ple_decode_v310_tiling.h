#ifndef ASCEND_OPS_QWEN4EXP_PLE_DECODE_V310_TILING_H
#define ASCEND_OPS_QWEN4EXP_PLE_DECODE_V310_TILING_H

#include <cstdint>

#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(Qwen4ExpPleDecodeV310TilingData)
    TILING_DATA_FIELD_DEF(int64_t, numTokens);
    TILING_DATA_FIELD_DEF(int64_t, hcGroups);
    TILING_DATA_FIELD_DEF(int64_t, groupSize);
    TILING_DATA_FIELD_DEF(int64_t, tasksPerCore);
    TILING_DATA_FIELD_DEF(int64_t, reduceWidth);
    TILING_DATA_FIELD_DEF(float, normEps);
    TILING_DATA_FIELD_DEF(float, invGroupSize);
    TILING_DATA_FIELD_DEF(float, invSqrtGroupSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Qwen4ExpPleDecodeV310, Qwen4ExpPleDecodeV310TilingData)

}  // namespace optiling

#endif
