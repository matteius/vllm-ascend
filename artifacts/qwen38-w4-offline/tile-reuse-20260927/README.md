# Qwen W4 tile 复用与 W4/W8 路由优化（2026-09-27）

## 范围与结论

本轮保留 Flash-Next 的 48 层、TP4/EP、W4A16 G128、MTP2、decode ACLGraph，
不修改模型文件、KV 分配或量化精度。修改基于 `ad3be2926`，测试工作树同时包含
用户此前已暂存的 byte-mask unpack 补丁；该补丁不属于本轮提交。

- W4：同一 expert/N tile 解包后保留在 L1，跨 M128 分块复用。
- W4：下一 K chunk 解包与当前 Cube 计算重叠；小 M 使用容量约束内的较大 K tile。
- W4/W8 共用：省去 router 的恒等乘法 `* 1.0`，保持 softmax/top-k/归一化顺序。
- W4/W8 共用：有界 INT64 索引经 INT32 转 FP32，避免直接转换的 AI-CPU 路径。
  expert 排序和逆排列排序均应用此优化；整数范围外保留原来的整数路径。
- 不采用本轮实验的通用 scatter 替换：改进转换之后，AI-Core 排序仍更快。

这里的“共用”指 `route_topk` 和 `build_grouped_expert_dispatch` 的 W4/W8 调用方，
不是把 W4 解包内核套到 W8。未改动旧 W8 启动脚本，也未测量本轮 W8 整机 tok/s。

## 微基准（不能换算成整机 tok/s）

310P、NPU0、CPU6–7、空闲服务；每图展开 10 次，30 次 replay，5 组。
projection 使用相同的合成权重、数据和运算；没有路由或 collective。

| 单 expert projection | 原版 µs | 最终候选 µs | 耗时变化 |
| --- | ---: | ---: | ---: |
| gate/up，M512 | 1884.62 | 1782.48 | -5.4% |
| down，M512 | 998.52 | 931.16 | -6.7% |
| routed gate/up，M3 | 121.40 | 121.50 | 基本不变 |
| routed down，M3 | 92.19 | 92.04 | 基本不变 |

对应 `reuse-baseline-a.jsonl`、`wide-candidate-a.jsonl`。
`reuse-candidate-a.jsonl` 和 `pipeline-candidate-a.jsonl` 分离了仅复用与流水化两个阶段：
可测收益主要来自多 M tile 的解包复用，流水化和更大 K tile 尚无额外显著收益。
实际 MoE 每 expert 行数可能小于 128，因此不能把 M512 的收益套用到整个 prefill。

逆排列的单独比较见 `dispatch-math-r2.jsonl` 至 `dispatch-math-r7.jsonl`：

- 原先 INT64→FP32 后排序，约 129–169 µs；
- 通用 INT64 scatter，约 118–191 µs，大 batch 比旧排序还慢；
- INT64 `index_copy_`，约 40–87 µs；
- 经 INT32 转换后排序，约 13–42 µs；这是最终选择。
- 其他 INT32/FP32 scatter/index-copy 组合并非稳定更快，保留测量结果而不启用。
- router 恒等乘法删除，单 router 约节省 0.9 µs；没有改变归一化算法。

`dispatch-r4-t3-kernels.csv` 与 `dispatch-r6-t3-kernels.csv` 显示：
直接转换使用 `CastAiCpu`，经 INT32 的两个转换使用 `CastAiCore`。
该 trace 是 eager 诊断，不能用其中带 profiler 的单次耗时替代 replay 基准。

有界 expert id 已在转窄前把 peer route 替换成 sentinel；不会把大的全局 id 截断成 local id。
所有索引保持精确；没有 NPU `.item()`、主机计数读取、新环境变量或常驻 FP16 expert bank。

## 真实模型与 layer A/B

基线与候选均为相同 checkpoint、MTP2、图配置、CPU affinity 和请求集合，
每种上下文有三个不同 coding prompt，各生成 512 tokens，不把 warmup 算入中位数。

| decode tok/s 中位数 | 基线 | 候选 |
| --- | ---: | ---: |
| 短上下文 | 14.62 | 14.71 |
| 约 23.4K 上下文 | 14.83 | 14.60 |

结论是整机 decode 基本不变，不能宣称大幅提速。见 `tile-http-*-short.jsonl` 与
`tile-http-*-long.jsonl`；记录包含 acceptance、TTFT 和输出 hash。
固定采样参数没有使两次运行的全文 hash 一致，不能宣称整模型 bitwise deterministic。
本轮候选长上下文 warmup 是真正 cold prefill：23407 tokens，TTFT 100.08 秒；
基线 warmup 已有缓存命中，故这两条 warmup 不能用于声称 cold prefill 改善或退化。

额外隔离了真实 layer-0 权重、相同合成输入和相同优化后的 Python 路由，
仅切换旧/新 W4 vendor，未执行 collective。`tile-layer-{baseline,candidate}.jsonl`
覆盖 T=1/3/8/32/512：五组 eager 与 graph 输出 hash 均与基线完全一致。
graph 耗时变化约 -0.6% 至 +1.2%，没有证明普通真实路由分布下的 layer 加速；
这也解释了为何单 expert M512 的复用收益没有直接成为模型 decode 收益。

HTTP smoke 检查算术、1–50、Python 输出；`tile-profile-and-dual.jsonl`
另外记录 profile 请求和两个同时发送的独立算术请求，结果均正确。
`tile-dual-count.jsonl` 再同时运行两次完整的 1–50 生成，两个答案均正确；
`server-startup-and-graphs.log` 确认双请求的 6-token 批次使用 `FULL` 图 replay。
`graph-replay-evidence.json` 记录每个 rank 的 110 次 `aclmdlRIExecuteAsync` 调用，
这是 breakable 图的执行调用数，不是 token 数或 decode step 数。
原始四 rank trace 保留在远端 `tile-server-traces/`，摘要为 `tile-trace-summary.json`。
rank0 的 30-route gate/up 投影中，Cube MAC counter 中位数约 9.9 µs，
整个 task 约 560 µs；说明只削减 Cube 算术不足以解决其余开销。
各硬件流水计数会重叠，不能相加或视为串行临界路径时间。

验收时服务 API PID 为 1014763，四个 worker 各占约 39260 MB process memory（npu-smi 标记），
设备总占用约 40.2–41.2 GB（npu-smi 的 MB 数值）；完成检查后保留服务，不停止。

## 验证证据

| 层次 | 结果与覆盖 |
| --- | --- |
| CPU | `cpu-tests.log`：101 passed，1 skipped；路由、整数边界、恒等运算、W4 contract/build guards |
| NPU 核心 | `wide-tests-r1.log`：186 passed；投影、路由、grouped、组合内核 |
| NPU 最终路由 | `final-tests-r4.log`：39 passed；跨 M128 尾块、peer 清零、输入变化的图 replay |
| 数学精度 | router 与排列要求完全相等；W4 projection 使用既有误差门限，不声称所有模型输出逐位一致 |
| 真实模型 | 由 `tile-http-candidate-smoke.jsonl` 和 short/long HTTP 基准给出最终结果 |

硬件最终 r2 内核相对 r1 只增加容量静态断言和 K 搜索下界保护，合法尺寸的计算相同。
初次未隔离上级 conftest 的测试收集缺少 `modelscope`；使用 ops 目录的 `--confcutdir` 后通过，
未为此升级或安装运行时依赖。

Ruff 检查通过。仓库 scoped pre-commit 的 symbolic-meta 检查仍报告既有
`csrc/torch_binding_meta.cpp:655` 的 `empty_symint` 写法；本轮没有改动该文件。
不能把这称为全仓测试/检查通过。

## 运行约束与功能状态

| 功能 | 本轮范围 |
| --- | --- |
| MTP2 / ACLGraph / EP | 保持开启；真实服务 smoke 和图指标确认 |
| 最大窗口 / 并发 | 保持 262144 / 2；本轮未重跑双满长容量测试 |
| 内存 | `gpu-memory-utilization=0.94`、KV fraction 0.80 不变；不增加常驻解包权重 |
| flashcomm1 | 未启用；保持现有通信配置，不纳入本轮变量 |
| 多模态 | text-only；image/video 额度为 0，未验证多模态 |
| 128K × 16 | 不在本轮容量目标；用户指定两窗口内存预算，未尝试 |
| W8 整机性能 | 未测；只验证共用路由逻辑，不报告推算出的 W8 tok/s |

## 简明运行手册

目标机器 `matteius@192.168.53.187`。用户指定本机工作树与端口 8001，
因此不使用技能默认 Docker 路径和 8000 端口。试验隔离 runtime 继续使用已有 PYTHONPATH。
不要在仍有服务占用 NPU/8001 时启动第二个实例，也不要覆盖已有证据日志。

远端候选根目录：`/srv/ai/src/qwen38-w4-native-build.zHZtEd`。
启动命令（只在原服务已正常退出后执行）：

```bash
bash /srv/ai/src/qwen38-w4-native-build.zHZtEd/qwen38-w4-tile-serve.sh
```

该脚本选择 `ops-pipeline-wide-r2`，保持原 checkpoint、CPU affinity 布局、
MTP2、图尺寸 `[1,2,3,6]`、chunk 512、max-seqs 2、端口 8001。
CPU affinity 由验收脚本在 worker 就绪后设置，不是 serve 脚本本身设置。
实际运行由 tmux `qwen-w4-tile-reuse-r1` 的 supervisor 和 96°C watchdog 管理。

```bash
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8001/v1/models
curl -fsS http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-w4-experimental","messages":[{"role":"user","content":"What is 17 * 19? Reply only with the integer."}],"temperature":0,"seed":1024,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

预期 `323`，不能只看启动完成或 HTTP 200。完整验收还检查 1–50、Python 输出与两个并发算术请求。
服务日志 `tile-server.log`，温度日志 `tile-server-thermal.log`，HTTP 验收日志 `tile-http-candidate.log`。
冷/热 prefill、decode、MTP acceptance 分开记录；不把首 token 前的时间混入 decode tok/s。

本轮不启用 softmax 分母消去、低精度 activation 替换或归约重排：
实数恒等变换不保证 FP32 舍入/top-k tie 不变。后续自定义 permutation kernel
可以利用“每个目的位置恰好一个 writer”，但需超过已优化排序再替换它。
