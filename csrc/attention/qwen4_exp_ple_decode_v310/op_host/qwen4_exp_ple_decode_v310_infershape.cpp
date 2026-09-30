#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

namespace ops {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *hidden = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, hidden);
    gert::Shape *output = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);
    *output = *hidden;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(2));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Qwen4ExpPleDecodeV310).InferShape(InferShape).InferDataType(InferDataType);

}  // namespace ops
