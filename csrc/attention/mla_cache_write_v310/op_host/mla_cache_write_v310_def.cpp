#include "register/op_def_registry.h"

namespace ops {

class MlaCacheWriteV310 : public OpDef {
public:
    explicit MlaCacheWriteV310(const char *name) : OpDef(name)
    {
        this->Input("cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("rows")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("slotMapping")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Attr("pageStride").AttrType(OPTIONAL).Int(0);

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

OP_ADD(MlaCacheWriteV310);

}  // namespace ops
