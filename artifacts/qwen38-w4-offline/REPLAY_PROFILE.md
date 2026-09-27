# W4 MTP graph replay：整模型 profile 与专家复用

## 基线与方法

本轮不修改生产 W8。仍使用隔离 W4 checkpoint / venv / plugin、TP4/EP、
MTP k=1、FULL_DECODE_ONLY `[1,2]`、32k context、单请求、512-token
prefill chunk、memory utilization 0.90。上一轮三个 512-token 请求的
短/长热上下文中位数分别为 **10.722 / 10.660 tok/s**；同机 W8 为
**19.073 / 18.091 tok/s**。目标尚未达成。

profile 前三个真实权重 smoke 均正确完成。通过 `/start_profile` 收集
延迟四步后的八个 worker steps，128-token 请求完整结束，再 `/stop_profile`。
带 profiler 的请求不能作为速度基准。worker 为 daemon，在线解析拒绝执行；
原始 trace 已成功写入，随后在独立进程调用
`torch_npu.profiler.profiler.analyse(..., max_process_number=4)` 完成离线导出。
停止时的 RECORD schedule warning 保留，不能把整个请求当成完整 trace。
八步中每 rank 恰有 1152 次 W4 projection（8 × 48 × 3）。

原始 host 目录：
`/srv/ai/src/qwen38-w4-hardware-20260927/replay-profile-r3`。
本地 CSV 副本：`/tmp/qwen38-w4-full-replay-r3`。
可审计摘要和请求证据位于 [`replay-r3/`](replay-r3/)。

```bash
python -m tools.qwen4exp.summarize_trace /tmp/qwen38-w4-full-replay-r3 \
  --output /tmp/qwen38-w4-replay-new-summary.json
```

工具逐 rank 汇总，分别报告任务时间总和、时间区间并集、首尾跨度；
不能把并行 rank 的时间相加后当作延迟。时间戳以 Decimal 转整数纳秒，
避免 epoch 微秒浮点数丢失精度。四个 CPU tests 验证精度、重叠和设备隔离。

| rank | 首尾跨度 ms | 任务并集 ms | W4 projection 总和 ms | 每步 W4 总和 ms |
| --- | --- | --- | --- | --- |
| 0 | 1453.250 | 994.724 | 464.113 | 58.014 |
| 1 | 1453.214 | 978.843 | 457.452 | 57.182 |
| 2 | 1453.014 | 1042.110 | 520.699 | 65.087 |
| 3 | 1453.198 | 1007.051 | 483.388 | 60.424 |

W4 projection 占每 rank 累计任务时间的 46.5–49.8%，是最大具名 kernel
成本。其它 NZ Matmul、layout/Cast、host/stream 等待仍不可忽略。
这些归因数值不是独立删去算子后的加速预测；最终必须看无 profiler 的
客户端 `(completion_tokens-1)/(elapsed-TTFT)`。

## 候选：同一 verification batch 内复用专家解包

原 routed kernel 对每个 route 单独解包，即便两个 verification tokens
选择同一专家也会重复恢复全部权重。新候选仅在 routes ≤ 20 时：

1. 每次 replay 在设备读取 IDs，首个同专家 route 成为 owner。
2. 唯一专家仍走原单行 projection；重复专家一次解包后做 batched Cube。
3. owner 的临时 `[R,N]` 输出只复制匹配 ID 的行到最终输出；其它专家
   不写这些行，peer route 仍写精确零。
4. routes > 20 保持原路径，避免放大二次输出 scratch。

checkpoint、常驻 packed bank 和数学量化公式均不变。不建立全专家
FP16/W8 shadow。临时 workspace 在原 `R*N*K*2` 之外，对 R≤20 增加
`R*R*N*2` bytes：R=20/N=2560 时最多 2,048,000 bytes，生命周期仅为算子。

97 个 NPU tests 全部通过（40.99 s），含新增 15 项重复 owner/全 peer/
owner 位置变化的 graph replay，覆盖 R=2/3/19/20/21 及三个投影尺寸。
37 个 CPU/build/trace tests 通过（4.84 s）。真实 layer-0 partial 的
synthetic-input replay 也正确：M=1/2/5 为 0.644/0.586/3.581 ms；
该样例没有重复本地专家，因此不代表候选的加速收益。

构建证据：host `build-reuse-r1.log`。group kernel 不变，routed kernel
SHA256 为 `a260a4d4c0e29a598d11d64e543f1a7c0f7d98fe386dbe963eaedbb51a21a09a`。
binding ABI 不变，无需重建 binding。整模型三个真实权重 smoke 均正确结束：
乘法 323、1–50 计数和 Python `[0,4,16]`。计数为 12.076 tok/s；
三个短 coding 512-token 请求为 11.763 / 11.066 / 11.203，中位数
**11.203 tok/s（+4.49%）**。23.4k 热前缀为 11.575 / 11.050 / 10.474，
中位数 **11.050 tok/s（+3.66%）**。所有请求都完整生成 512 tokens；
MTP 接受率会随输出变化，不能把整个提升都归为纯 kernel 加速。
短 LRU 的 drafted/accepted 与原基线完全相同（265/246），其速度由
11.210 提高到 11.763。长上下文冷 warmup 的 TTFT 仍为 367.412 s，
明显慢于 W8，不作为 decode 提速掩盖。测量完成后才停止该实验 server。

## 下一候选：更宽的 tiled unpack

仅对 K=2560 将 K batch 从 512 扩至 1280；其它形状保留原 divisor 选择。
删除 tiled 路径不使用的 expanded scale/offset buffers，将低 nibble 输出
作为中间 FP16 scratch，峰值 188416 bytes，小于实际分配的 192 KiB UB。
不同 group 的 Sub 和 Mul 分别批量提交，各阶段之间同步，保持每元素
计算顺序不变。重复 route 的早退发生在 CATLASS resource 初始化之前。
静态断言约束 scratch 上界，新增测试验证 wide batch/group 边界。

构建独立安装到 `ops-wide-r1`，没有覆盖上一候选运行时的 vendor。
101 项 NPU numerical / replay tests 已通过（41.53 s），真实 layer-0
TP partial 也通过：M=1/2/5 replay 为 0.638/0.564/3.484 ms。
37 项 CPU/build/trace tests 重跑通过（5.08 s）。整模型三个真实 smoke
已正确完成，计数为 **12.266 tok/s**。两-token verification 的 runtime
表明确显示 FULL，MTP accepted 计数递增。model-load 仍为 19.3821 GiB/rank。
构建记录为 `build-wide-r2.log`，服务日志 `server-routed-mtp-wide-r1.log`。

三条短 coding 请求为 **12.072 / 11.003 / 11.310 tok/s**，中位数
**11.310 tok/s**：比专家复用版本高 0.95%，比原 routed 基线高 5.48%。
样本少且 acceptance 有波动，不能将这点中位数差异过度推广。
第一题仍是 drafted/accepted=265/246，decode 从 43.442 降到 42.330 s。
约 23.4k 的 warm-prefix 三题为 **11.589 / 11.239 / 10.787 tok/s**，
中位数 **11.239 tok/s**，比专家复用版本高 1.71%，比原 routed 基线高
5.43%。全部完成 512 tokens，cached_tokens=23168；acceptance 分别为
91.76% / 86.18% / 79.65%。冷 warmup TTFT=364.452 s，仍明显慢于 W8。
所有 speed claim 都仍低于生产 W8。原始记录为 `w4-wide-mtp-long-r1.jsonl`。

- group SHA256：`5e8da07bbb170fffb51ff6080cce77a44e6a0a6027811ff30ea0c436b46e0151`
- routed SHA256：`42d7464b7fe0a4aa0d6b4952455b6cdac99d1c16b2ddb9b50ff84dec0d69f38d`

## 第三个候选：persistent N-tile route 调度

R≤20 时仅启动 N/32 个 logical blocks，每个 block 顺序处理该 N tile 的
local/peer routes，复用同一个 CATLASS resource。重复专家仍由首个 owner
批量计算；peer rows 仍在每次 replay 清零。为隔离变量，解包 workspace
仍按 `route*(N/32)+tile` 独立分配，不复用 physical-core workspace，
也不保留 FP16 expert bank。R>20 仍使用逐 route/tile 调度与小型 zero-only
peer task。没有新增环境变量，W8 路径不变。

首版 `persistent-r1` 通过 107 NPU tests，但创建完整 resource 的 peer 路径
使五-token layer replay 增至 3.755 ms，因此第二版恢复大 batch 的轻量
peer task。`persistent-r2` 再次 **107 passed / 50.45 s**，37 CPU tests
也通过（4.80 s）。新增六项测试覆盖 10/20/21 routes、多达 21 个不同
expert，以及八轮 local↔peer 动态 graph replay。

真实 layer0 权重、synthetic activations、无 collective 的 graph replay：

| tokens | wide (ms) | persistent-r2 (ms) |
| --- | ---: | ---: |
| 1 | 0.637756 | 0.592170 |
| 2 | 0.564023 | 0.423398 |
| 5 | 3.484067 | 3.569738 |

两-token diagnostic 减少 24.93%，但五-token 仍慢 2.46%；不声称所有
batch 都提速，也不把单层结果当整模型 tok/s。
构建日志 `build-persistent-r2.log`，独立 vendor `ops-persistent-r2`；
旧 vendor 均保留。group binary SHA 与 wide 相同，routed SHA256：
`a64d35f3f393287254a24b91df1ccf041b6dc12756eae43430c028a66abcbb19`。

完整模型 `server-routed-mtp-persistent-r1.log` 已通过三个真实 smoke：
323、1–50 和 `[0,4,16]` 都正确终止；计数 decode **12.608 tok/s**。
两-token verification runtime 表再次确认 FULL；MTP drafted/accepted
计数递增。model-load 仍为 **19.3821 GiB/rank**，没有增加常驻权重内存。
三题短 coding 为 **12.287 / 11.265 / 11.830 tok/s**，中位数
**11.830 tok/s**，比 wide 高 4.60%，比原 routed 基线高 10.33%。
全部完成 512 tokens；acceptance=92.11% / 77.43% / 86.50%。第三题的
acceptance 也上升，不能把全部中位数差异归因于 kernel。第一题输出 SHA
与 wide 相同，accepted 少一个，decode 从 42.330 降到 41.589 s；第二题
drafted/accepted 同为 288/223，decode 从 46.442 降到 45.361 s。
约 23.4k 热前缀三题为 **12.150 / 11.052 / 10.743 tok/s**，中位数
**11.052 tok/s**，低于 wide 的 11.239；acceptance 分别为
96.17% / 79.65% / 75.34%。三题均正常完成 512 tokens，cached_tokens=23168。
冷 warmup TTFT=364.375 s；该段 prefill 与下一候选的 CPU 编译有短暂重叠，
不能作为严格隔离的冷 prefill A/B。编译在 decode 前完成，NPU tests 在
所有吞吐请求结束后才运行。短/长结果均仍明显低于 W8，不声称普遍提速。

## 第四个候选：80-route persistent 与 MTP draft-count sweep

将相同 owner/peer/重复 expert 调度扩展至现有上限 80 routes（八个
top-k=10 tokens）。保留每个 logical route/tile 的独立 unpack workspace，
额外 `[R,R,N]` projection-output scratch 在 R=80/N=2560 为
32,768,000 bytes（31.25 MiB）。不新增常驻 expanded expert bank，不改变 W8。
新增 30/50/80 routes 的 owner、unique experts 和动态 replay 回归。

**122 NPU tests passed / 107.28 s**；37 CPU/build tests passed / 4.79 s。
真实 layer0、synthetic activations、TP4 rank0、无 collective 的 replay
在 tokens=1/2/3/5/8 为 **0.590 / 0.421 / 2.681 / 2.136 / 6.465 ms**。
新增三-token case 会推进 synthetic-input RNG，因此五-token 的 local
experts 从旧样例的 14 变成 8；2.136 与旧 3.570 ms **不是 matched A/B**，
不能据此声称提速。随后在服务空闲并完全停止后以相同 `[1,2,5]` 列表
重跑：0.593 / 0.423 / **3.524 ms**；五-token 同为 14 个 local experts，
比旧 3.570 ms 仅降低 1.29%。allocator reserved 由旧 683,671,552 增至
1,048,576,000 bytes，这是单层诊断的 allocator 高水位，不是新增常驻权重。
不能按单层结果估计整模型 MTP k=2/k=4 的收益。

构建记录 `build-persistent-r3.log`，独立 vendor `ops-persistent-r3`，
group SHA 与上一版相同；routed SHA256：
`e667ab7e366d4bfba0291364d7a097a64bafc6d7548a9fe9bba163aebb01ad37`。
测量顺序为先结束上一服务的全部吞吐请求，确认空闲并停掉准确 PID，
再运行 NPU tests/layer checks，最后启动 MTP k=2、FULL_DECODE_ONLY `[1,3]`。
后续 k=4 必须单独通过真实 smoke 与五-token FULL replay；启动成功不算通过。

k=2 已通过全部三个真实 smoke，1–50 为 **14.600 tok/s**。
runtime 表明确显示三-token batch `FULL`；model-load 仍为
19.3821 GiB/rank，capture 报告 0.38 GiB。三题 coding 为
**13.482 / 11.259 / 11.972 tok/s**，中位数 **11.972 tok/s**，
仅比 k=1 的 11.830 高 1.20%。drafted/accepted 分别为 378/322、448/289、
424/299，即 acceptance=85.19% / 64.51% / 70.52%。三题都完成 512 tokens。
不能把更高的 counting 速度当成 coding 的典型速度，也不声称达到 W8。
证据为 `server-routed-mtp-mtp2-r1.log`、`w4-mtp2-r1-short.jsonl`。

k=4 也通过三个真实 smoke：323、1–50、`[0,4,16]` 正确终止，
counting 为 **16.179 tok/s**。runtime 表确认五-token `FULL`；capture
报告 0.50 GiB，常驻 model-load 仍为 19.3821 GiB/rank。首题 512-token
coding 为 12.377 tok/s，低于 k=2 同题的 13.482；不能用 counting 替代
实际 coding 结果。三题最终为 **12.377 / 9.877 / 10.155 tok/s**，
中位数 **10.155 tok/s**，比 k=2 慢 15.18%；全部完成 512 tokens。
drafted/accepted=548/375、680/344、660/346，acceptance 分别为
68.43% / 50.59% / 52.42%。更长 verification 加上更低接受率抵消了
每步生成更多 tokens 的收益；不推荐把 k=4 当成 coding 提速配置。
约 23.4k 的 k=4 长 benchmark 已开始，独立记录稳定性与长请求速度，
不改写这个短请求回归结论。

| 已完成短 coding sweep | 中位 tok/s | FULL verification tokens | capture 报告 GiB |
| --- | ---: | ---: | ---: |
| W4 k=1 / 20-route persistent | 11.830 | 2 | 0.30 |
| W4 k=2 / 80-route persistent | 11.972 | 3 | 0.38 |
| W4 k=4 / 80-route persistent | 10.155 | 5 | 0.50 |
| 生产 W8 k=1 基线 | 19.073 | 2 | 不在本轮重测 |

W4 k=1 与更大 k 同时改变了大 batch kernel 分支；不是单独改变 k 的
纯 causal A/B。k=2/k=4 使用同一 binary，prompt/sampling/长度协议相同。
小样本不能说明 k=2 的 1.20% 中位数优势显著。当前目标仍未达成。

## 复现与交接

简短 runbook（仅已授权的隔离 host 环境，不替换 W8）：

```bash
cd /srv/ai/src/qwen38-w4-hardware-20260927
bash qwen38-w4-server-routed-check-r1.sh wide-r1 \
  /srv/ai/src/qwen38-w4-operator-build-r1/ops-wide-r1/vendors/qwen_w4_probe_transformer
# 首次运行后用新 label 保留旧日志；脚本先执行三个 real-weight smoke。
# 确认 HTTP_SMOKE_PASS 后，在另一终端运行：
bash qwen38-w4-wide-benchmark-r1.sh

# persistent 候选使用独立 vendor；同样需要新的 label/output 路径重跑：
bash qwen38-w4-server-routed-check-r1.sh persistent-r1 \
  /srv/ai/src/qwen38-w4-operator-build-r1/ops-persistent-r2/vendors/qwen_w4_probe_transformer
bash qwen38-w4-persistent-benchmark-r1.sh

# 80-route kernel 与更大 MTP verification（重跑必须使用新的 label）：
bash qwen38-w4-server-routed-check-r1.sh mtp2-r1 \
  /srv/ai/src/qwen38-w4-operator-build-r1/ops-persistent-r3/vendors/qwen_w4_probe_transformer 2
bash qwen38-w4-mtp-short-benchmark.sh mtp2-r1
# k=4 对应最后一个参数 4，capture sizes 自动配为 [1,5]。
```

服务配置见本页开头和 tutorial，仍为 :8002 / TP4 / EP /
FULL_DECODE_ONLY；本轮增加 k=2/k=4（分别 `[1,3]` / `[1,5]`），
不将启动成功当作验证通过。所有候选均保留 packed W4 bank。

本轮 scoped manual pre-commit 的 Ruff、codespell、typos、markdownlint、
secret scan 等均通过；全局 `check-symbolic-meta` 仍因未改动的
`csrc/torch_binding_meta.cpp:655`（已有 W2 meta 的 `empty_symint({`）失败，
已与 HEAD 对照，不把这个结果写成全库 lint 全绿。三个 C++ 文件另行
手动 clang-format（仓库 hook 本身排除 csrc）。未跑全库多型号硬件测试。

另外在临时快照 `/tmp/qwen-w4-replay-format.lZNQ6h` 执行了完整
`bash format.sh ci`，日志 `/tmp/qwen-w4-replay-format-ci.log`；失败项还包括
已有 prefill 工具的 Ruff 问题、其它模型/文档的格式及拼写、禁用 `re`
imports 等。自动格式修改只留在临时快照，没有污染任务 worktree 或生产文件。

## 尚未修复的 profile 线索

- rank0 有 384 个 AI_CPU `Cast`，八步共 60.698 ms（约 7.587 ms/step）；
  结合相邻 RoPE operations 和源码，优先检查整数 positions 的重复转换及
  Q/K/indexer 的频率与 sin/cos 重算。当前是归因线索，不是已验证加速。
- `trans_TransData_10` 恰每步一次、约 2.311 ms；输入为 FP16
  `FRACTAL_Z [4,640,16,16]`，输出为 `[2560,1,16,16]`，前后是
  `[10240,1,1,4]` depthwise filters 和 `Conv2D3`，不是大 lm_head 权重搬运。
  `[1,10240,1,11] → [1,10240,1,2]` 也指向短 dilation convolution。
- 小型 INT64 FloorDiv AI_CPU 八步约 14.254 ms，不能将不同维度的
  cache/position 算术未经范围证明就改为 INT32。

这些时间可能重叠，也受 profiler 扰动；不把它们直接相加为可获得的 tok/s。
投影仍是首要瓶颈。persistent 候选隔离检查 route/peer logical-block
调度开销，未来复用 workspace 必须重新验证 Cube pipeline 生命周期；不能重复
旧 wide + physical-workspace 实验曾出现的数值错误。

## 边界

已验证的基线及专家复用版本为真实权重、MTP、decode ACLGraph、EP；
没有用 dummy 代替真实权重。更宽 unpack 也已通过真实 smoke 与短/长请求。
persistent 20-route 候选已完成算子、真实权重单层和整模型 smoke，
短/长 coding benchmark 均完成；80-route 候选也已完成 MTP k=2 的真实
smoke、三-token FULL replay 和短 coding；k=4 的真实 smoke 与五-token
FULL replay、短 coding 也完成，长 coding 仍在执行。
本轮只隔离 batch-one decode，不声称 W4 已验证 128k×16、双 160k 或
完整任务质量。flashcomm1 和多模态未在本轮验证；容量和量化质量仍需后续
独立评测，不能伪造 accuracy YAML 分数。
