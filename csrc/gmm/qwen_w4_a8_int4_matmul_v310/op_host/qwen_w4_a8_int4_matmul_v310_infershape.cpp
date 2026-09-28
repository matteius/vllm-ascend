// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"
namespace ops {
static ge::graphStatus InferQwenW4A8Int4(gert::InferShapeContext* context) {
  auto x = context->GetInputShape(0);
  auto codes = context->GetInputShape(4);
  auto ends = context->GetInputShape(8);
  auto y = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, x);
  OP_CHECK_NULL_WITH_CONTEXT(context, codes);
  OP_CHECK_NULL_WITH_CONTEXT(context, ends);
  OP_CHECK_NULL_WITH_CONTEXT(context, y);
  y->SetDimNum(2);
  y->SetDim(0, context->GetInputDesc(8)->GetDataType() == ge::DT_INT32 ? ends->GetDim(0) : x->GetDim(0));
  y->SetDim(1, codes->GetDim(1));
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus DtypeQwenW4A8Int4(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, ge::DT_FLOAT16);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(QwenW4A8Int4MatmulV310).InferShape(InferQwenW4A8Int4).InferDataType(DtypeQwenW4A8Int4);
}  // namespace ops
