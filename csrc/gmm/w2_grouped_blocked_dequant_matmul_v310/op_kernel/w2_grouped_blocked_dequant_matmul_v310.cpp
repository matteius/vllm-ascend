// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "compat_310p.h"
#include "w2_blocked_dequant_matmul_v310.h"
#include "w2_grouped_blocked_dequant_matmul_v310_tiling_data.h"

namespace {
constexpr int64_t SPARSE_DECODE_MAX_ROWS = 32;
constexpr int64_t SPARSE_DECODE_MIN_EXPERTS = 64;
}

extern "C" __global__ __aicore__ void w2_grouped_blocked_dequant_matmul_v310(
    GM_ADDR x, GM_ADDR codes, GM_ADDR blockScale, GM_ADDR groupEnds, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
  auto td = reinterpret_cast<__gm__ W2GroupedBlockedDequantMatmulKernelTilingData*>(tiling);
  AscendC::GlobalTensor<int64_t> ends;
  ends.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(groupEnds));
  const int64_t n = td->nDim;
  const int64_t k = td->kDim;
  const int64_t packedK = k / td->codesPerByte;
  const int64_t scaleStride = (n / NsW2::W2_BLOCK_SIZE) * (k / NsW2::W2_BLOCK_SIZE);
  GM_ADDR user = AscendC::GetUserWorkspace(workspace);
  NsW2::W2BlockedDequantMatmulV310Cube op;
  int64_t start = 0;
  // Decode routes touch only a few of the 72 local experts. The cumulative
  // boundaries are monotone by construction, so upper-bound search skips
  // empty experts without fetching every boundary on every Cube core.
  if (td->nzPacked != 0 && td->numRows <= SPARSE_DECODE_MAX_ROWS &&
      td->numExperts >= SPARSE_DECODE_MIN_EXPERTS) {
    int64_t nextExpert = 0;
    while (nextExpert < td->numExperts) {
      int64_t low = nextExpert;
      int64_t high = td->numExperts;
      while (low < high) {
        const int64_t mid = low + (high - low) / 2;
        if (ends.GetValue(mid) <= start) {
          low = mid + 1;
        } else {
          high = mid;
        }
      }
      if (low == td->numExperts) {
        break;
      }
      const int64_t end = ends.GetValue(low);
      if (end <= start || end > td->numRows) {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "invalid grouped W2/W4 boundaries"); });
        return;
      }
      op.InitGeometry(x + start * k * sizeof(half),
                      codes + low * n * packedK,
                      blockScale + low * scaleStride * sizeof(float),
                      y + start * n * sizeof(half), user, end - start, n, k,
                      td->codesPerByte, true);
      op.Process();
      AscendC::PipeBarrier<PIPE_ALL>();
      start = end;
      nextExpert = low + 1;
    }
    return;
  }
  for (int64_t expert = 0; expert < td->numExperts; ++expert) {
    const int64_t end = ends.GetValue(expert);
    if (end < start || end > td->numRows) {
      ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "invalid grouped W2/W4 boundaries"); });
      return;
    }
    if (end > start) {
      op.InitGeometry(x + start * k * sizeof(half),
                      codes + expert * n * packedK,
                      blockScale + expert * scaleStride * sizeof(float),
                      y + start * n * sizeof(half), user, end - start, n, k,
                      td->codesPerByte, td->nzPacked != 0);
      op.Process();
      AscendC::PipeBarrier<PIPE_ALL>();
    }
    start = end;
  }
}
