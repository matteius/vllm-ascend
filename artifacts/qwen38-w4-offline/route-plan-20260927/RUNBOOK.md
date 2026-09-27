# 远端隔离运行与复测

本轮不覆盖 home launcher、checkpoint 或旧 vendor；结束后停止 benchmark 服务。
下列命令仅针对 `matteius@192.168.53.187` 的隔离目录，端口固定 8001。
先确认端口/NPU 空闲，不与其他服务同时启动。

```bash
ssh matteius@192.168.53.187
cd /srv/ai/src/qwen38-w4-native-build.zHZtEd
```

## 服务候选

```bash
# W4：MTP2，graph [1,2,3,6]，262144/2，memory 0.94。
bash qwen38-w4-routeplan-serve.sh

# W8：MTP1，graph [1,2]，160000/1，memory 0.965。
bash qwen38-w8-routeplan-serve.sh candidate

# W8：本轮较快的已验证配置，MTP2，graph [1,2,3]。
bash qwen38-w8-routeplan-serve.sh candidate-k2
```

三个命令是互斥选项，不要依次同时后台启动。`baseline` 使用私有生产 runtime 副本。
W4 使用 `runtime` + `ops-routeplan-r4`；W8 candidate 使用 `w8-routeplan-candidate`。
本轮显式 PYTHONPATH 仅用于隔离 A/B，没有改全局安装。

benchmark 在模型 ready 后将 TP0–3 **所有已有线程**分别绑到
`0–5 / 8–13 / 16–21 / 24–29`，并记录完整 affinity JSON。
仅运行上面的 serve 命令不包含该后处理；不能据此保证重现表中的吞吐。
完整协议在远端 `qwen38-w8-routeplan-check.py` 与
`qwen38-w4-tile-server-check.py`：验证目标 API/worker 血缘、绑定、真实输出 smoke、
三次 512-token short 与三次 ~23.4K-prefix long 请求。
这些 checker 会发送 benchmark 请求；使用时提供新的 evidence label，避免覆盖旧结果。

## 算子与图测试

```bash
bash qwen38-w4-reuse-check.sh routeplan-r4 fresh-w4-tests \
  -m pytest --confcutdir=tests/e2e/nightly/310p/single_node/ops -q \
  tests/e2e/nightly/310p/single_node/ops/test_qwen_w4_group_matmul_310.py

bash qwen38-w8-reuse-check.sh routeplan-r4 fresh-w8-rope-tests \
  -m pytest --confcutdir=tests/e2e/nightly/310p/single_node/ops -q \
  tests/e2e/nightly/310p/single_node/ops/test_qwen4exp_shared_rope_310.py
```

wrapper 使用真实 NPU，包含 900 秒超时与温度监控。
真实 checkpoint 的单层 W8 A/B 工具为 `tools/qwen4exp/profile_w8_routing_310.py`，
必须显式提供保存的 baseline `moe.py`，不用运行中的服务作临时基准。

## 证据与限制

- W4：`routeplan-http-candidate-*.json*`、`routeplan-dual-count.jsonl`、
  `routeplan-server-graph.log`、`routeplan-trace-summary.json`。
- W8：远端 `w8-routeplan-{baseline,candidate,candidate-k2}-*`。
- 原始 profiler 数据保留在各 `*-traces` 目录；`profile_runtime.py analyse`
  离线导出后，可用 `summarize_trace.py` 汇总。不要覆盖已导出的目录。
- 48 层真实权重；没有 dummy-only 验收。本轮只验证 text、MTP、graph、
  expert 分片和 W4 小请求并发；不增加量化质量/满窗口容量承诺。
- 保留旧预留 workspace；缩减该预算需另做 graph 内存与长窗口验证。
- 不启用 flashcomm1/多模态；不重跑 `128K × 16`，超出当前实验并发配置。
- 保留 `.965` 硬上限，不通过增加内存比例让实验勉强启动。
