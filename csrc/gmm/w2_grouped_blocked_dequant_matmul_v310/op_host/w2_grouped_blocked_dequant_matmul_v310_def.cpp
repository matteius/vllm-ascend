// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "register/op_def_registry.h"

namespace ops {
class W2GroupedBlockedDequantMatmulV310 : public OpDef {
 public:
  explicit W2GroupedBlockedDequantMatmulV310(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
        .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("codes")
        .ParamType(REQUIRED)
        .DataType({ge::DT_UINT8, ge::DT_INT8})
        .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("blockScale")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
        .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("groupEnds")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64, ge::DT_INT64})
        .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
        .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("coreType.value", "AiCore");
    this->AICore().AddConfig("ascend310p", config);
  }
};
OP_ADD(W2GroupedBlockedDequantMatmulV310);
}  // namespace ops
