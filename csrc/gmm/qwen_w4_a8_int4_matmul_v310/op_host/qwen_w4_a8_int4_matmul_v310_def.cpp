// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "register/op_def_registry.h"

namespace ops {
class QwenW4A8Int4MatmulV310 : public OpDef {
 public:
  explicit QwenW4A8Int4MatmulV310(const char* name) : OpDef(name) {
    this->Input("low")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_INT8})
        .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("high")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_INT8})
        .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("activation_scale")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
        .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("activation_sum")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
        .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("codes")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_INT8})
        .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("scale")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
        .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("offset")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
        .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("weight_sum")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
        .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("group_ends")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64, ge::DT_INT32})
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
OP_ADD(QwenW4A8Int4MatmulV310);
}  // namespace ops
