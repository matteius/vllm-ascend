// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "compat_310p.h"
#include "qwen_w4_group_matmul_v310_tiling_data.h"
#include "qwen_w4_group_matmul_v310.h"
#include "qwen_w4_routed_matmul_v310_tiling_data.h"

namespace {
// Matches the host's route limit. This plan is private to a persistent
// AI-core block, rebuilt on every launch/replay, and never stored on the host.
constexpr uint32_t MAX_ROUTE_ROWS = 80;
struct RoutePlan {
  uint32_t groups;
  int32_t experts[MAX_ROUTE_ROWS];
  uint32_t starts[MAX_ROUTE_ROWS + 1];
  uint32_t rows[MAX_ROUTE_ROWS];

  __aicore__ inline void Build(GM_ADDR expertIds, uint32_t routeCount, int64_t expertCount) {
    AscendC::GlobalTensor<int32_t> ids;
    ids.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expertIds));
    uint32_t routeGroups[MAX_ROUTE_ROWS];
    uint32_t counts[MAX_ROUTE_ROWS];
    groups = 0;
    for (uint32_t row = 0; row < routeCount; ++row) {
      const int32_t expert = ids.GetValue(row);
      routeGroups[row] = MAX_ROUTE_ROWS;
      if (expert < 0 || expert >= expertCount) {
        continue;
      }
      uint32_t group = 0;
      while (group < groups && experts[group] != expert) {
        ++group;
      }
      if (group == groups) {
        experts[group] = expert;
        counts[group] = 0;
        ++groups;
      }
      routeGroups[row] = group;
      ++counts[group];
    }
    starts[0] = 0;
    for (uint32_t group = 0; group < groups; ++group) {
      starts[group + 1] = starts[group] + counts[group];
      counts[group] = starts[group];
    }
    // Preserve both first-owner order and ascending row order within each
    // expert. No arithmetic/reduction order changes and no duplicate writes.
    for (uint32_t row = 0; row < routeCount; ++row) {
      const uint32_t group = routeGroups[row];
      if (group != MAX_ROUTE_ROWS) {
        rows[counts[group]++] = row;
      }
    }
  }
};
}  // namespace

extern "C" __global__ __aicore__ void qwen_w4_routed_matmul_v310(GM_ADDR x, GM_ADDR codes, GM_ADDR scale,
                                                                 GM_ADDR offset, GM_ADDR expert_ids, GM_ADDR y,
                                                                 GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
  auto td = reinterpret_cast<__gm__ QwenW4RoutedKernelTilingData*>(tiling);
  const int64_t n = td->nDim, k = td->kDim;
  const uint32_t nTiles = n / NsQwenW4::QW4_TILE_N;
  // A verification batch has many peer-owned rows but very few local
  // experts. One persistent owner per physical core reuses its route plan
  // and compact activations across all of that core's N tiles.
  // Host validation bounds this to at most 80 routes (eight top-k=10 rows),
  // allowing MTP verification with more than two tokens to reuse experts too.
  const uint32_t core = AscendC::GetBlockIdx();
  const uint32_t cores = AscendC::GetBlockNum();
  RoutePlan plan;
  plan.Build(expert_ids, td->numRows, td->numExperts);
  NsQwenW4::QwenW4GroupMatmulV310Cube op;
  GM_ADDR user = AscendC::GetUserWorkspace(workspace);
  for (uint32_t tile = core; tile < nTiles; tile += cores) {
    op.ZeroOutputRows(y, td->numRows, n, tile);
  }
  for (uint32_t group = 0; group < plan.groups; ++group) {
    const int32_t expert = plan.experts[group];
    const uint32_t* matchedRows = plan.rows + plan.starts[group];
    const uint32_t route = matchedRows[0];
    const uint32_t projectionRows = plan.starts[group + 1] - plan.starts[group];
    const bool reuse = projectionRows > 1;
    GM_ADDR projectionInput = x + route * k * sizeof(half);
    GM_ADDR projectionOutput = y + route * n * sizeof(half);
    if (reuse) {
      // Resident-L1 weights leave the original R*N*K workspace unused.
      // Reserve R*32*K elements per core; compact input needs only R*K.
      // This is disjoint across cores and reused across their N tiles.
      projectionInput = user + static_cast<int64_t>(core) * td->numRows * NsQwenW4::QW4_TILE_N * k * sizeof(half);
      op.GatherPlannedRows(x, projectionInput, matchedRows, projectionRows, k);
      // Store the existing FP16 accumulator conversion directly to its
      // final route rows, eliminating the scratch write/read/scatter pass.
      projectionOutput = y;
    }
    const int64_t metadataOffset = static_cast<int64_t>(expert) * n * (k / NsQwenW4::QW4_GROUP_SIZE);
    for (uint32_t tile = core; tile < nTiles; tile += cores) {
      // Reinitialize/drain the Cube pipeline for every N tile. The route
      // plan and gathered GM activations remain valid across these calls.
      op.InitGeometry(projectionInput, codes + static_cast<int64_t>(expert) * n * k / 2,
                      scale + metadataOffset * sizeof(half), offset + metadataOffset, projectionOutput, user,
                      projectionRows, n, k, true);
      if (reuse) {
        op.SetOutputRows(matchedRows);
      }
      op.SetTileRange(tile, nTiles);
      // The resident-L1 path does not use this legacy unpack region.
      op.SetWorkspaceTile(route * nTiles + tile);
      op.Process();
      AscendC::PipeBarrier<PIPE_ALL>();
    }
  }
}
