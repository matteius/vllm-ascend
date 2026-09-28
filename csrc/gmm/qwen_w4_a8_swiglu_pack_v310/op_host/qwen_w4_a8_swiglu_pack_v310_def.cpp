// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "register/op_def_registry.h"

namespace ops {
class QwenW4A8SwigluPackV310 : public OpDef {
 public:
  explicit QwenW4A8SwigluPackV310(const char* name) : OpDef(name) {
    this->Input("gate_up")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    for (const char* output : {"low", "high"}) {
      this->Output(output).ParamType(REQUIRED).DataType({ge::DT_INT8}).FormatList({ge::FORMAT_ND}).AutoContiguous();
    }
    for (const char* output : {"scale", "sum"}) {
      this->Output(output).ParamType(REQUIRED).DataType({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND}).AutoContiguous();
    }
    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(false)
        .ExtendCfgInfo("coreType.value", "AiCore");
    this->AICore().AddConfig("ascend310p", config);
  }
};
OP_ADD(QwenW4A8SwigluPackV310);
}  // namespace ops
