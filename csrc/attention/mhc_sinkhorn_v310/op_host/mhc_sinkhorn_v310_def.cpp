#include "register/op_def_registry.h"

namespace ops {

class MhcSinkhornV310 : public OpDef {
public:
    explicit MhcSinkhornV310(const char *name) : OpDef(name)
    {
        this->Input("logits")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("mix")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Attr("iterations").AttrType(OPTIONAL).Int(20);
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6f);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("coreType.value", "AiCore");
        this->AICore().AddConfig("ascend310p", config);
    }
};

OP_ADD(MhcSinkhornV310);

}  // namespace ops
