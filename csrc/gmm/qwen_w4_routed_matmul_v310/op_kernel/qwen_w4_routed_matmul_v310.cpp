// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "compat_310p.h"
#include "qwen_w4_group_matmul_v310_tiling_data.h"
#include "qwen_w4_group_matmul_v310.h"
#include "qwen_w4_routed_matmul_v310_tiling_data.h"

namespace {
__aicore__ inline void ZeroPeerRoute(GM_ADDR y, int64_t outputOffset) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
  pipe.InitBuffer(buffer, NsQwenW4::QW4_TILE_N * sizeof(half));
  auto zero = buffer.Get<half>();
  AscendC::Duplicate(zero, static_cast<half>(0), NsQwenW4::QW4_TILE_N);
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
  AscendC::GlobalTensor<half> output;
  output.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));
  AscendC::DataCopy(output[outputOffset], zero, NsQwenW4::QW4_TILE_N);
  AscendC::PipeBarrier<PIPE_ALL>();
}
}  // namespace

extern "C" __global__ __aicore__ void qwen_w4_routed_matmul_v310(GM_ADDR x, GM_ADDR codes, GM_ADDR scale,
                                                                 GM_ADDR offset, GM_ADDR expert_ids, GM_ADDR y,
                                                                 GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
  auto td = reinterpret_cast<__gm__ QwenW4RoutedKernelTilingData*>(tiling);
  const int64_t n = td->nDim, k = td->kDim;
  const uint32_t nTiles = n / NsQwenW4::QW4_TILE_N;
  const uint32_t route = AscendC::GetBlockIdx() / nTiles;
  const uint32_t tile = AscendC::GetBlockIdx() % nTiles;
  AscendC::GlobalTensor<int32_t> ids;
  ids.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expert_ids));
  const int32_t expert = ids.GetValue(route);
  // Negative and >=E IDs are peer-owned routes after subtracting the local
  // expert offset. Always overwrite output, including local->peer replays.
  if (expert < 0 || expert >= td->numExperts) {
    ZeroPeerRoute(y, route * n + tile * NsQwenW4::QW4_TILE_N);
    return;
  }
  NsQwenW4::QwenW4GroupMatmulV310Cube op;
  const int64_t metadataOffset = static_cast<int64_t>(expert) * n * (k / NsQwenW4::QW4_GROUP_SIZE);
  op.InitGeometry(x + route * k * sizeof(half), codes + static_cast<int64_t>(expert) * n * k / 2,
                  scale + metadataOffset * sizeof(half), offset + metadataOffset, y + route * n * sizeof(half),
                  AscendC::GetUserWorkspace(workspace), 1, n, k, true);
  op.SetTileRange(tile, nTiles);
  op.Process();
}
