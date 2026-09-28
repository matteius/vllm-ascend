// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"
namespace ops {
static ge::graphStatus InferQwenW4A8Int4DownReduce(gert::InferShapeContext* context) {
  auto weights = context->GetInputShape(9);
  auto codes = context->GetInputShape(4);
  auto y = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, weights);
  OP_CHECK_NULL_WITH_CONTEXT(context, codes);
  OP_CHECK_NULL_WITH_CONTEXT(context, y);
  y->SetDimNum(2);
  y->SetDim(0, weights->GetDim(0));
  y->SetDim(1, codes->GetDim(1));
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus DtypeQwenW4A8Int4DownReduce(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, ge::DT_FLOAT);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(QwenW4A8Int4DownReduceV310)
    .InferShape(InferQwenW4A8Int4DownReduce)
    .InferDataType(DtypeQwenW4A8Int4DownReduce);
}  // namespace ops
