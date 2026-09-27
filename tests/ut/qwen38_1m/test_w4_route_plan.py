# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the kernel's scalar route planner with a host GM-read stub."""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_compiled_route_plan_is_stable_bounded_and_reads_each_id_once(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    source = (REPO_ROOT / "csrc/gmm/qwen_w4_routed_matmul_v310/op_kernel/qwen_w4_routed_matmul_v310.cpp").read_text()
    planner = source[source.index("namespace {") : source.index('extern "C"')]
    test = (
        r"""
#include <cassert>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>
#define __aicore__
#define __gm__
using GM_ADDR = uint8_t*;
namespace AscendC {
template <typename T> struct GlobalTensor {
  T* data;
  static inline uint32_t reads = 0;
  void SetGlobalBuffer(T* pointer) { data = pointer; }
  T GetValue(uint32_t row) { ++reads; return data[row]; }
};
}
"""
        + planner
        + r"""
int main() {
  std::mt19937 generator(1024);
  RoutePlan plan;
  for (uint32_t count = 1; count <= MAX_ROUTE_ROWS; ++count) {
    for (uint32_t phase = 0; phase < 40; ++phase) {
      const int64_t experts = phase % 2 == 0 ? 128 : 3;
      std::vector<int32_t> ids(count);
      std::vector<int32_t> owners;
      std::vector<std::vector<uint32_t>> rows;
      for (uint32_t row = 0; row < count; ++row) {
        int32_t id = static_cast<int32_t>(generator() % 150) - 5;
        if (phase == 0) id = 0;
        if (phase == 1) id = -1;
        if (phase == 2) id = row;
        if (phase == 3) id = row % 3;
        if (phase == 4) id = row % 2 == 0 ? std::numeric_limits<int32_t>::min()
                                        : std::numeric_limits<int32_t>::max();
        ids[row] = id;
        if (id < 0 || id >= experts) continue;
        uint32_t group = 0;
        while (group < owners.size() && owners[group] != id) ++group;
        if (group == owners.size()) { owners.push_back(id); rows.emplace_back(); }
        rows[group].push_back(row);
      }
      AscendC::GlobalTensor<int32_t>::reads = 0;
      plan.Build(reinterpret_cast<GM_ADDR>(ids.data()), count, experts);
      assert(AscendC::GlobalTensor<int32_t>::reads == count);
      assert(plan.groups == owners.size());
      assert(plan.starts[0] == 0);
      for (uint32_t group = 0; group < plan.groups; ++group) {
        assert(plan.experts[group] == owners[group]);
        assert(plan.starts[group + 1] - plan.starts[group] == rows[group].size());
        for (uint32_t i = 0; i < rows[group].size(); ++i)
          assert(plan.rows[plan.starts[group] + i] == rows[group][i]);
      }
      assert(plan.starts[plan.groups] <= count);
    }
  }
}
"""
    )
    path = tmp_path / "route_plan.cpp"
    path.write_text(test)
    executable = tmp_path / "route_plan"
    subprocess.run([compiler, "-std=c++17", "-O2", str(path), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True, timeout=30)


def test_route_plan_limit_matches_host_guard():
    root = REPO_ROOT / "csrc/gmm/qwen_w4_routed_matmul_v310"
    kernel = (root / "op_kernel/qwen_w4_routed_matmul_v310.cpp").read_text()
    host = (root / "op_host/qwen_w4_routed_matmul_v310_tiling.cpp").read_text()
    assert "constexpr uint32_t MAX_ROUTE_ROWS = 80;" in kernel
    assert "constexpr int64_t MAX_ROUTES = 80;" in host
