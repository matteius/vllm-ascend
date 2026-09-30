#include "register/op_def_registry.h"

namespace ops {

class Qwen4ExpPleDecodeV310 : public OpDef {
public:
    explicit Qwen4ExpPleDecodeV310(const char *name) : OpDef(name)
    {
        this->Input("projected")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("hidden")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("normKeyWeight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("normQueryWeight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("normConvWeight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("currentConvWeight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("output").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND});
        this->Attr("normEps").AttrType(OPTIONAL).Float(1e-6f);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("coreType.value", "AiCore");
        this->AICore().AddConfig("ascend310p", config);
    }
};

OP_ADD(Qwen4ExpPleDecodeV310);

}  // namespace ops
