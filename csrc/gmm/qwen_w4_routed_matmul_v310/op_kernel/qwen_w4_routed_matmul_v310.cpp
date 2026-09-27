// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "compat_310p.h"
#include "qwen_w4_group_matmul_v310_tiling_data.h"
#include "qwen_w4_group_matmul_v310.h"
#include "qwen_w4_routed_matmul_v310_tiling_data.h"

namespace {
// Bounded two-token verification batch (Qwen top-k=10). Larger batches keep
// the existing one-row path instead of allocating quadratic output scratch.
constexpr uint32_t MAX_REUSED_ROUTES = 20;

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
  bool reuse = false;
  if (td->numRows <= MAX_REUSED_ROUTES) {
    for (uint32_t previous = 0; previous < route; ++previous) {
      if (ids.GetValue(previous) == expert) {
        // The first route writes this row, including on changing-ID replay.
        return;
      }
    }
    for (uint32_t next = route + 1; next < td->numRows; ++next) {
      if (ids.GetValue(next) == expert) {
        reuse = true;
        break;
      }
    }
  }
  NsQwenW4::QwenW4GroupMatmulV310Cube op;
  GM_ADDR user = AscendC::GetUserWorkspace(workspace);
  GM_ADDR projectionOutput = y + route * n * sizeof(half);
  if (reuse) {
    // The first region remains the original per-route unpack workspace.
    // Each owner gets an independent [R,N] projection result after it.
    projectionOutput = user + (td->numRows * n * k + route * td->numRows * n) * sizeof(half);
  }
  const int64_t metadataOffset = static_cast<int64_t>(expert) * n * (k / NsQwenW4::QW4_GROUP_SIZE);
  op.InitGeometry(reuse ? x : x + route * k * sizeof(half), codes + static_cast<int64_t>(expert) * n * k / 2,
                  scale + metadataOffset * sizeof(half), offset + metadataOffset, projectionOutput, user,
                  reuse ? td->numRows : 1, n, k, true);
  op.SetTileRange(tile, nTiles);
  op.Process();
  if (reuse) {
    op.CopyMatchingRows(y, expert_ids, expert, tile);
  }
}
