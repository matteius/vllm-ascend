// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "compat_310p.h"
#include "qwen_w4_group_matmul_v310_tiling_data.h"
#include "qwen_w4_group_matmul_v310.h"
#include "qwen_w4_routed_matmul_v310_tiling_data.h"

extern "C" __global__ __aicore__ void qwen_w4_routed_matmul_v310(GM_ADDR x, GM_ADDR codes, GM_ADDR scale,
                                                                 GM_ADDR offset, GM_ADDR expert_ids, GM_ADDR y,
                                                                 GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
  auto td = reinterpret_cast<__gm__ QwenW4RoutedKernelTilingData*>(tiling);
  const int64_t n = td->nDim, k = td->kDim;
  const uint32_t nTiles = n / NsQwenW4::QW4_TILE_N;
  // A verification batch has many peer-owned rows but very few local
  // experts. One persistent owner per N tile avoids scheduling an AI-core
  // task and constructing CATLASS resources for every peer/duplicate row.
  // Host validation bounds this to at most 80 routes (eight top-k=10 rows),
  // allowing MTP verification with more than two tokens to reuse experts too.
  const uint32_t tile = AscendC::GetBlockIdx();
  AscendC::GlobalTensor<int32_t> ids;
  ids.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expert_ids));
  NsQwenW4::QwenW4GroupMatmulV310Cube op;
  GM_ADDR user = AscendC::GetUserWorkspace(workspace);
  for (uint32_t route = 0; route < td->numRows; ++route) {
    const int32_t expert = ids.GetValue(route);
    // Always overwrite peer-owned rows, including local->peer replays.
    if (expert < 0 || expert >= td->numExperts) {
      op.ZeroOutputTile(y, route * n + tile * NsQwenW4::QW4_TILE_N);
      continue;
    }
    bool reuse = false;
    bool ownedByEarlierRoute = false;
    for (uint32_t previous = 0; previous < route; ++previous) {
      if (ids.GetValue(previous) == expert) {
        ownedByEarlierRoute = true;
        break;
      }
    }
    if (ownedByEarlierRoute) {
      continue;
    }
    for (uint32_t next = route + 1; next < td->numRows; ++next) {
      if (ids.GetValue(next) == expert) {
        reuse = true;
        break;
      }
    }
    GM_ADDR projectionOutput = y + route * n * sizeof(half);
    if (reuse) {
      projectionOutput = user + (td->numRows * n * k + route * td->numRows * n) * sizeof(half);
    }
    const int64_t metadataOffset = static_cast<int64_t>(expert) * n * (k / NsQwenW4::QW4_GROUP_SIZE);
    op.InitGeometry(reuse ? x : x + route * k * sizeof(half), codes + static_cast<int64_t>(expert) * n * k / 2,
                    scale + metadataOffset * sizeof(half), offset + metadataOffset, projectionOutput, user,
                    reuse ? td->numRows : 1, n, k, true);
    op.SetTileRange(tile, nTiles);
    // Keep the original unpack-workspace mapping: one region per logical
    // route/tile, independent of the added batched projection-output scratch.
    op.SetWorkspaceTile(route * nTiles + tile);
    op.Process();
    if (reuse) {
      op.CopyMatchingRows(y, expert_ids, expert, tile);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
  }
}
