// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"
namespace ops {
static ge::graphStatus InferPack(gert::InferShapeContext* context) {
  auto x = context->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, x);
  OP_CHECK_IF(x->GetDimNum() != 2, OP_LOGE(context, "expected matrix"), return ge::GRAPH_FAILED);
  for (uint32_t i = 0; i < 4; ++i) {
    auto out = context->GetOutputShape(i);
    OP_CHECK_NULL_WITH_CONTEXT(context, out);
    out->SetDimNum(i < 2 ? 2 : 3);
    out->SetDim(0, x->GetDim(0));
    out->SetDim(1, x->GetDim(1) < 0 ? -1 : x->GetDim(1) / (i < 2 ? 2 : 128));
    if (i >= 2) out->SetDim(2, 8);
  }
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus DtypePack(gert::InferDataTypeContext* context) {
  for (uint32_t i = 0; i < 4; ++i) context->SetOutputDataType(i, i < 2 ? ge::DT_INT8 : ge::DT_FLOAT);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(QwenW4A8PackV310).InferShape(InferPack).InferDataType(DtypePack);
}  // namespace ops
