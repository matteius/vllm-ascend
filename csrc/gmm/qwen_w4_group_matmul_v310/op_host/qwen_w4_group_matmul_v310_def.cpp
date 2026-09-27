/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qwen_w4_group_matmul_v310_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {

class QwenW4GroupMatmulV310 : public OpDef {
 public:
  explicit QwenW4GroupMatmulV310(const char* name) : OpDef(name) {
    this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
    this->Input("codes").ParamType(REQUIRED).DataType({ge::DT_INT8}).FormatList({ge::FORMAT_ND}).AutoContiguous();
    this->Input("scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
    this->Input("offset").ParamType(REQUIRED).DataType({ge::DT_INT8}).FormatList({ge::FORMAT_ND}).AutoContiguous();

    this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();

    // Same byte count and quantization, optionally re-encoded at load time
    // to [N/32,K/128,8,16,16] NZ nibble pairs (n, n+16) and
    // [N/32,K/128,32] metadata. Both codes and offsets are biased by 8.
    this->Attr("tiled").AttrType(OPTIONAL).Bool(false);

    OpAICoreConfig aicoreConfig;
    aicoreConfig.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("coreType.value", "AiCore");
    this->AICore().AddConfig("ascend310p", aicoreConfig);
  }
};
OP_ADD(QwenW4GroupMatmulV310);

}  // namespace ops
