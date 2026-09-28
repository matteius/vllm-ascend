// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

namespace ops {
static ge::graphStatus InferSwigluPack(gert::InferShapeContext* context) {
  auto input = context->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, input);
  OP_CHECK_IF(input->GetDimNum() != 2, OP_LOGE(context, "expected matrix"), return ge::GRAPH_FAILED);
  const int64_t gate_up_width = input->GetDim(1);
  const int64_t activation_width = gate_up_width < 0 ? -1 : gate_up_width / 2;
  for (uint32_t i = 0; i < 4; ++i) {
    auto output = context->GetOutputShape(i);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);
    output->SetDimNum(i < 2 ? 2 : 3);
    output->SetDim(0, input->GetDim(0));
    output->SetDim(1, activation_width < 0 ? -1 : activation_width / (i < 2 ? 2 : 128));
    if (i >= 2) output->SetDim(2, 8);
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DtypeSwigluPack(gert::InferDataTypeContext* context) {
  for (uint32_t i = 0; i < 4; ++i) context->SetOutputDataType(i, i < 2 ? ge::DT_INT8 : ge::DT_FLOAT);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QwenW4A8SwigluPackV310)
    .InferShape(InferSwigluPack)
    .InferDataType(DtypeSwigluPack);
}  // namespace ops
