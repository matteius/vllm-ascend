/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "register/op_def_registry.h"

namespace ops {
class ChunkKdaFwd : public OpDef {
public:
    explicit ChunkKdaFwd(const char *name) : OpDef(name)
    {
        // dav-m200 Cube MMAD has no BF16 input mode, and its vector pipeline
        // cannot cast BF16 gate or beta values to the FP32 accumulators used by
        // KDA. Register only the signature implemented by Ascend 310P.
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("g").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("beta").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("a_log").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("dt_bias").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("initial_state").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("chunk_indices").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        // Carries only an executor dependency between the physical 310P
        // stages. The kernel does not inspect its value.
        this->Input("stage_dependency")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // Standalone executor kernels materialize these casts. The dav-m200
        // unified image cannot reliably lower FP32-to-FP16 vector Cast.
        this->Input("gk_fp16")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("beta_fp16")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("attn_out").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("final_state").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("gk").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("Aqk").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("Akk").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("w").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("u").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("qg").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("kg").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("v_new").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("h").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("qg_scaled").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("u_seed").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("score_scratch").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("score_matrices").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("stage_token")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("layout").AttrType(OPTIONAL).String("BSND");
        this->Attr("scale").AttrType(REQUIRED).Float(1.0);
        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);
        this->Attr("safe_gate").AttrType(REQUIRED).Bool(false);
        this->Attr("lower_bound").AttrType(OPTIONAL).Float(-5.0);
        this->Attr("use_gate_in_kernel").AttrType(REQUIRED).Bool(false);
        this->Attr("state_v_first").AttrType(OPTIONAL).Bool(false);
        this->Attr("stage").AttrType(OPTIONAL).Int(-1);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend310p", config);
    }
};

OP_ADD(ChunkKdaFwd);
} // namespace ops
