// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "compat_310p.h"
#include "qwen_w4_group_matmul_v310_tiling_data.h"
#include "qwen_w4_group_matmul_v310.h"
#include "qwen_w4_grouped_matmul_v310_tiling_data.h"

// Device-side group ends describe fixed-size sorted routes; no R*N*K scratch.
extern "C" __global__ __aicore__ void qwen_w4_grouped_matmul_v310(GM_ADDR x, GM_ADDR codes, GM_ADDR scale,
                                                                  GM_ADDR offset, GM_ADDR group_ends, GM_ADDR y,
                                                                  GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
  auto td = reinterpret_cast<__gm__ QwenW4GroupedKernelTilingData*>(tiling);
  const int64_t n = td->nDim, k = td->kDim;
  const uint32_t nTiles = n / NsQwenW4::QW4_TILE_N;
  AscendC::GlobalTensor<int64_t> ends;
  ends.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(group_ends));
  NsQwenW4::QwenW4GroupMatmulV310Cube op;
  GM_ADDR user = AscendC::GetUserWorkspace(workspace);
  // One owner per expert/N tile. Skip empty groups entirely on device.
  for (int64_t expert = 0; expert < td->numExperts; ++expert) {
    const int64_t start = expert == 0 ? 0 : ends.GetValue(expert - 1);
    const int64_t end = ends.GetValue(expert);
    if (start < 0 || end < start || end > td->numRows) {
      ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "invalid W4 group boundaries"); });
      return;
    }
    if (start == end && expert + 1 != td->numExperts) continue;
    for (uint32_t tile = AscendC::GetBlockIdx(); tile < nTiles; tile += AscendC::GetBlockNum()) {
      if (expert + 1 == td->numExperts) {
        // Overwrite peer rows on every replay. Bounded zero tiles fit helper UB.
        constexpr int64_t ZERO_ROWS = 64;
        for (int64_t row = end; row < td->numRows; row += ZERO_ROWS) {
          const uint32_t count = td->numRows - row < ZERO_ROWS ? td->numRows - row : ZERO_ROWS;
          op.ZeroOutputRows(y + row * n * sizeof(half), count, n, tile);
        }
      }
      const int64_t metadata = expert * n * (k / NsQwenW4::QW4_GROUP_SIZE);
      if (end > start) {
        // The helper streams bounded M tiles through one resident L1 weight
        // tile, so large groups no longer repeat weight unpacking per 128 rows.
        op.InitGeometry(x + start * k * sizeof(half), codes + expert * n * k / 2, scale + metadata * sizeof(half),
                        offset + metadata, y + start * n * sizeof(half), user, end - start, n, k, true);
        op.SetTileRange(tile, nTiles);
        op.SetWorkspaceTile(AscendC::GetBlockIdx());
        op.Process();
        AscendC::PipeBarrier<PIPE_ALL>();
      }
    }
  }
}
