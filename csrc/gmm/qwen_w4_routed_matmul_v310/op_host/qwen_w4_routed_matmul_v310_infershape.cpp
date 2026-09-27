// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"
namespace ops {
static ge::graphStatus InferQwenW4Routed(gert::InferShapeContext* context) {
  auto x = context->GetInputShape(0);
  auto codes = context->GetInputShape(1);
  auto y = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, x);
  OP_CHECK_NULL_WITH_CONTEXT(context, codes);
  OP_CHECK_NULL_WITH_CONTEXT(context, y);
  y->SetDimNum(2);
  y->SetDim(0, x->GetDim(0));
  y->SetDim(1, codes->GetDim(1));
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus DtypeQwenW4Routed(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, context->GetInputDataType(0));
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(QwenW4RoutedMatmulV310).InferShape(InferQwenW4Routed).InferDataType(DtypeQwenW4Routed);
}  // namespace ops
