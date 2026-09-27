// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "register/op_def_registry.h"

namespace ops {
class QwenW4RoutedMatmulV310 : public OpDef {
 public:
  explicit QwenW4RoutedMatmulV310(const char* name) : OpDef(name) {
    this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
    this->Input("codes").ParamType(REQUIRED).DataType({ge::DT_INT8}).FormatList({ge::FORMAT_ND}).AutoContiguous();
    this->Input("scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
    this->Input("offset").ParamType(REQUIRED).DataType({ge::DT_INT8}).FormatList({ge::FORMAT_ND}).AutoContiguous();
    this->Input("expert_ids").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
    this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
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
OP_ADD(QwenW4RoutedMatmulV310);
}  // namespace ops
