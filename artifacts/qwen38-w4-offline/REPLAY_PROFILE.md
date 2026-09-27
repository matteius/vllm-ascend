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
约 23.4k 的 k=4 长 benchmark 也完成：**9.838 / 7.914 / 6.653 tok/s**，
中位数 **7.914 tok/s**，低于 k=1 的 11.052。三题均完成 512 tokens，
cached_tokens=23168，drafted/accepted=492/389、608/359、720/335。
acceptance=79.07% / 59.05% / 46.53%。冷 warmup TTFT=371.591 s。
本轮 CPU 编译和后续 NPU tests 都在最后一个请求完成后才启动。
长请求也不支持将 k=4 作为 coding 提速配置。

| 已完成短 coding sweep | 中位 tok/s | FULL verification tokens | capture 报告 GiB |
| --- | ---: | ---: | ---: |
| W4 k=1 / 20-route persistent | 11.830 | 2 | 0.30 |
| W4 k=2 / 80-route persistent | 11.972 | 3 | 0.38 |
| W4 k=4 / 80-route persistent | 10.155 | 5 | 0.50 |
| 生产 W8 k=1 基线 | 19.073 | 2 | 不在本轮重测 |

W4 k=1 与更大 k 同时改变了大 batch kernel 分支；不是单独改变 k 的
纯 causal A/B。k=2/k=4 使用同一 binary，prompt/sampling/长度协议相同。
小样本不能说明 k=2 的 1.20% 中位数优势显著。当前目标仍未达成。

## 第五个候选：Qwen-only resident-L1 decoded tile

独立 vendor `ops-l1-r1` 把 tiled packed-W4 的 FP16 tile 通过 MTE3 从 UB
直接搬到 L1，移除该 tile 的 GM 写入及随后的 GM→L1 读取。每个 AI core
保留最大 `[32,2560]` FP16 B tile（163,840 bytes），旁边为两个
`[128,128]` FP16 activation stages（共 65,536 bytes）。总 L1 上限
229,376 bytes；L0 double buffering、升序 K=128 的 FP32 accumulation 和
FP16 output 保持不变。跨 MTE3/MTE1/M/V 的事件显式同步，所有本地
tile 在每次调用/route/replay 都重新生成，绝不缓存跨请求的 decoded experts。

只改 Qwen kernel 内部的 tiled 分支，canonical reference 和共享 W8/GLM
CATLASS helper 不变。此轮先保留旧 tiling 的 GM workspace 大小，以隔离
执行路径；尚未将其减少。因此不把移除 memory traffic 写成 allocator
或长窗口容量提升。没有新增常驻 expanded expert bank。

新增 M=1/15/16/17/63/64/65/127/128、K=640/2560 的图重放，五轮都更换
输入和权重，覆盖 L1 B stride、双 activation stages、M fractal 尾部。
**140 NPU tests passed / 108.69 s**；37 CPU/build tests passed / 4.78 s。
构建 `build-l1-r1.log` 的 group/routed binary SHA256 分别为：

```text
303e5bb7283bfd74daffb3297f412f5d71ace8b04f7be7b6bf088a38a9bc82b8
df4872e5ace484548c6b94dd88bc8df2f6d39d480ad6fa23e6a481680536b0e0
```

真实 layer0、相同 `[1,2,3,5,8]` synthetic input 顺序、相同 local routes
和 experts 的 replay 对比如下（无 TP collective，不是整模型速度）：

| tokens | persistent-r3 ms | L1-r1 ms | latency 降低 |
| --- | ---: | ---: | ---: |
| 1 | 0.590 | 0.524 | 11.06% |
| 2 | 0.421 | 0.390 | 7.29% |
| 3 | 2.681 | 2.434 | 9.22% |
| 5 | 2.136 | 1.956 | 8.44% |
| 8 | 6.465 | 6.133 | 5.14% |

allocator allocated/reserved 与相同形状的 persistent-r3 一致。
证据 `layer-l1-r1.jsonl`。真实 k=2 + FULL `[1,3]`（日志 label `mtp2-r2`）
通过三个正确性 smoke；counting 为 15.373 tok/s，Python 和 323 正确
终止。runtime 表确认三-token `FULL`，MTP drafted/accepted 增长，
model-load 仍为 19.3821 GiB/rank，capture 0.38 GiB。

三题短 coding 为 **14.207 / 11.895 / 12.470 tok/s**，中位数
**12.470 tok/s**，相对相同 k=2 的 persistent-r3（11.972）提高 **4.16%**。
全部完成 512 tokens。drafted/accepted 为 376/323、446/289、426/299；
接受率 85.90% / 64.80% / 70.19%。首题输出 SHA 与旧 k=2 相同，后两题
不同；接受数量略有变化，不能把所有增益都解释为 kernel 的纯因果作用。
第一题 decode 从 37.902 降到 35.969 s，第二题从 45.388 降到 42.958 s，
第三题从 42.682 降到 40.978 s。三题均改善，但小样本不是统计显著性证明。

证据为 `http-routed-mtp-smoke-mtp2-r2.jsonl`、`w4-mtp2-r2-short.jsonl`。
约 23.4k 的长 benchmark 已全部完成：**10.483 / 9.283 / 9.147 tok/s**，
中位数 **9.283 tok/s**，三题都完成 512 tokens、命中 23,168 prefix tokens。
drafted/accepted 为 380/321、426/298、432/297；接受率
84.47% / 69.95% / 68.75%。冷 TTFT 为 359.882 s；热 TTFT 为
4.622 / 4.653 / 5.184 s。原始证据 `w4-mtp2-r2-long.jsonl`。
该组合低于旧 k=1 的 11.052 tok/s，不能将短 prompt 的 k=2 优势推广到
长窗口。这里同时改变了 kernel 与 k，不将差异单独归因于 L1。
不以单层或 counting 数据宣称达到生产 W8 的 19.073/18.091 tok/s。

## 第六个候选：批量清零 peer 输出

`ops-zero-r1` 保留 L1 数学路径，只改 routed 输出初始化：每个 persistent
N tile 的 block 先在 UB 准备 `[R,32]` 的零，再通过一次 strided UB→GM
copy 清零该 tile 的所有 route 行，最后由本地专家覆盖所属行。host 限制
R≤80，临时零 buffer 最大 5 KiB，与原 CATLASS UB 复用，不新增分配。
MTE3 完成后才复用 UB / 写入 local outputs；不同 blocks 的 N tile 不重叠。
peer 分支不再每行执行 Duplicate、store 和 pipeline drain。

回归新增 R=1 的三种尺寸，并在每次 graph replay 前用 NaN 污染输出，
要求本地输出数值正确、peer 输出逐元素精确为零。覆盖 all-local、all-peer、
mixed、owner 改变和 R=80 上限。**143 NPU tests passed / 108.73 s**；
**37 CPU/build tests passed / 4.82 s**。group binary 与 L1 完全相同；
routed binary SHA256 为：

```text
47f845c1129a24814b02cc489e72e9b72feda56c94e9779ed6975938ae63bc48
```

真实 layer0、同一 `[1,2,3,5,8]` input 顺序的 replay 数值检查通过。
allocator allocated/reserved 与 L1 各对应形状完全相同。
原始证据 `layer-zero-r1.jsonl`，下面不是整模型速度：

| tokens | L1-r1 ms | zero-r1 ms | latency 降低 |
| --- | ---: | ---: | ---: |
| 1 | 0.524 | 0.528 | -0.75% |
| 2 | 0.390 | 0.334 | 14.35% |
| 3 | 2.434 | 2.390 | 1.80% |
| 5 | 1.956 | 1.865 | 4.65% |
| 8 | 6.133 | 5.942 | 3.11% |

整模型 label `mtp2-r3` 使用相同 MTP k=2 / FULL `[1,3]` 配置，
三个完整 smoke 已正确结束，counting 为 15.687 tok/s，不作 coding
速度替代。runtime 表显示三-token `FULL` 重放，drafted/accepted 计数
递增，model-load 仍为 19.3821 GiB/rank、capture 0.38 GiB。
原始证据 `http-routed-mtp-smoke-mtp2-r3.jsonl`。
同协议短 coding 完成：**14.288 / 11.902 / 12.552 tok/s**，中位数
**12.552**，比 L1 的 12.470 高 **0.66%**；全部正常完成 512 tokens。
drafted/accepted 为 382/321、454/284、430/296，接受率为
84.03% / 62.56% / 68.84%。三题输出 SHA 均与旧样本不同，不能把很小的
中位数差异当成统计显著的整模型提速，也不能据此宣称 15 tok/s coding。
短证据 `w4-mtp2-r3-short.jsonl`。约 23.4k 长测量完成：
**11.023 / 9.709 / 8.866 tok/s**，中位数 **9.709**，比 L1 的
9.283 高 4.59%，但第三题反而下降，且三题输出都变化。全部完整生成
512 tokens，cached_tokens=23168；drafted/accepted 为
364/331、412/306、450/286。冷 warmup TTFT=358.539 s，三个热请求
TTFT=4.614/4.689/5.066 s。不能推广为所有长请求的收益，也仍低于
旧 k=1 的 11.052 和 W8 的 18.091。原始证据 `w4-mtp2-r3-long.jsonl`。
构建 `build-zero-r1.log`；回归 `kernel-zero-r1-tests.log`；W8 环境未替换。

## 第七个候选：共享 Q/K/index-query RoPE 表

只在显式 `cube_310_routed` W4 backend 开启：同一次 forward 的 Q、K、
index-query 共享由当前 positions 计算的表，保留 FP32/FP64 accumulation
与原先 norm→rotation 顺序。index-key 仍使用自身的 group-start positions
和原精度；不把整型 positions 缩窄，不跨请求缓存动态表。
W8、eager-W4、其它 Cube backend 默认保持旧计算路径。

首次 NPU gate 的六项 text replay 通过，但 MRoPE capture 失败：
`_rope_frequency_positions` 从 Python list 创建 axis tensor，触发 capture
中禁止的同步 H2D copy（107030）。这不是通过；日志保留为
`kernel-rope-r1-tests.log`。修正为 W4 module 的非持久 constant-axis buffer，
init 时创建，query 与 index-key 共用；动态坐标仍每次 replay 读取。
未修改 checkpoint 格式、operator binary、专家 bank 或生产 W8 安装。

CPU 回归 **113 passed / 5.09 s**，覆盖 FP32/FP64、三轴 MRoPE、
0/23400/163840/1048576/16777217 positions、Passthrough dims 与错误表/axis
metadata。两项未改动的旧 full-model mock test 因本机 vLLM 已删除
`logits_processor.get_tensor_model_parallel_world_size` hook，在模型构造前
失败；本次选择性 gate 排除这两项，不声称整个 CPU suite 全绿。
修正版 **160 NPU tests passed / 106.74 s**：143 项已有 W4 kernel
回归与 17 项新增 RoPE/backend gate；12 项 replay 覆盖 1/2/3/5/8/512
tokens、text/MRoPE、四轮动态位置/activation，并要求 bitwise parity。
这不等于完整多模态模型验证。NPU 日志 `kernel-rope-r2-tests.log`。

独立 graph microbenchmark（50 次/组，5 组）输出 bitwise equal；
只计三路 query RoPE，不含 norm、projection、attention 或 collective：

| tokens | 三份独立表 ms | 共享一份表 ms |
| --- | --- | --- |
| 1 | 0.453359 | 0.186866 |
| 2 | 0.474302 | 0.198493 |
| 3 | 0.477039 | 0.194466 |
| 5 | 0.499273 | 0.204444 |
| 8 | 0.481998 | 0.201357 |

原始记录 `rope-r2-microbench.jsonl`；三-token 局部延迟降低 59.24%，
不能直接折算整模型提速。真实 MTP k=2 / FULL `[1,3]` 的新 run 为
`mtp2-r4`，沿用 `ops-zero-r1` binary、采样/上下文/内存配置。
三个 smoke 已正确结束，counting 为 15.871 tok/s（不替代 coding
benchmark）。runtime stats 显示三-token FULL replay；MTP accepted /
drafted 计数增长，capture 仍为 0.38 GiB、KV cache 12.91 GiB。
证据 `http-routed-mtp-smoke-mtp2-r4.jsonl` 与 host
`server-routed-mtp-mtp2-r4.log`。短 coding 三题完成 512 tokens：
**14.412 / 12.091 / 13.034 tok/s**，中位数 **13.034**，比 zero-r1
12.552 高 **3.84%**，仍低于同机 W8 的 19.073。第一题输出 SHA 与
旧版本相同，速度提高 0.87%；其它两题输出变化，drafted/accepted 为
384/320、456/283、422/301。不能把整段中位数提升完全归因于共享表。
model-load 仍为 19.3821 GiB/rank；kernel、checkpoint 和原 W8 launcher
均未改动。原始结果 `w4-mtp2-r4-short.jsonl`。

约 23.4k 长测试也完成：**10.852 / 10.120 / 9.004 tok/s**，中位数
**10.120**，相对 zero-r1 的 9.709 高 4.23%。三题均生成 512 tokens，
复用 23,168 prefix tokens；drafted/accepted 为 378/323、402/311、
452/286。冷 TTFT 358.790 秒，热 TTFT 4.518/4.690/5.079 秒。
三个输出 SHA 均改变，不能将这次中位数差异完全归因于 kernel 延迟；
第一题低于 zero-r1 的 11.023。原始结果 `w4-mtp2-r4-long.jsonl`。
下一步使用同一 kernel + RoPE 比较 k=1，避免混淆 kernel 与 k。

同一 `ops-zero-r1` 与 RoPE 的 k=1 新 run `mtp1-r1` 已通过三个真实
smoke；两-token verification 的 runtime stats 为 FULL，accepted 计数增长。
短 coding 三题为 **13.417 / 12.392 / 12.787 tok/s**，中位数 **12.787**。
k=2 的 13.034 仅高 1.93%；第三题输出 SHA 完全一致，其余两题不同。
k=1 drafted/accepted 为 266/246、287/225、278/234；不要把更高
acceptance rate 本身等同于更高 tok/s。

k=1 的约 23.4k 三题完成：**13.213 / 12.754 / 12.269 tok/s**，中位数
**12.754**，比相同代码 k=2 的 10.120 高 **26.03%**。全部生成 512 tokens、
复用 23,168 prefix tokens；drafted/accepted 为 261/251、271/241、279/232。
冷 TTFT 356.128 秒，热 TTFT 4.560/4.562/5.094 秒。三个输出都与 k=2
不同，不是 bitwise-identical workload；不能把差异完全视为每步加速。
原始证据 `w4-mtp1-r1-short.jsonl`、`w4-mtp1-r1-long.jsonl`，
host `server-routed-mtp-mtp1-r1.log` 显示两-token FULL replay。

源码定位到一个重要 dispatch 边界：`_BATCHED_QSA_MAX_DECODE_TOKENS=2`，
且 `_qsa_decode_group_list` 也只按两-token 创建。满足其它条件时 k=1
走 grouped batched QSA，k=2 的三-token（及 k=4 的五-token）则回退
`qsa_sparse_attention_310`。这可能解释部分长上下文差距；尚未修改或
以 matched profile 验证因果。下一独立候选应只对 routed-W4 扩展这个
阈值及预创建 group-list，增加 T=3/5/8、KV heads=1/2 的 dynamic replay
和真实长上下文 gates；不能全局改变 W8 默认或直接宣称可获得的 tok/s。

## 第八个候选：只计算同一专家的匹配行

之前 reuse 分支虽然只解包一次专家，但矩阵乘仍覆盖整个 R-route 输入，
最后才筛掉其它专家的结果。新候选在 NPU 上 gather 匹配 activation 行，
仅对 compact 行执行同样的 K=128 顺序 FP32 accumulation，再 scatter 回
原 route。每个 persistent N tile 的 gather scratch 独立，复用原 R*N*K
workspace；UB 每 16 行 flush，最大 80 KiB，workspace ABI/packed bank
大小均不变。singleton 路径不 gather，canonical W4/W8/GLM 未修改。

新 graph 回归固定 R=80，逐次改变匹配数为 0/1/2/3/8/15/16/17/
31/32/33/65/79/80、随机位置与 expert owner、activation；每次先用 NaN
污染输出，并与独立 CPU dequant reference 对比。另覆盖第二专家覆盖
scratch、peer 精确清零、K=1024 与两种真实投影形状。
本地 113 项 CPU/build tests 通过（仍排除两个已有 vLLM mock 不兼容项）；
独立 vendor `ops-compact-r1` 构建成功，**163 项 NPU tests 全通过，
138.18 秒**，包含新增 compact 边界与之前的 canonical/tiled、完整 MoE、
MRoPE replay gates。`kernel-compact-r1-tests.log` 保留 15 条已有 warnings，
不把 warnings 隐去。group kernel SHA 与 zero-r1 相同：
`303e5bb7283bfd74daffb3297f412f5d71ace8b04f7be7b6bf088a38a9bc82b8`；
routed kernel SHA：
`1ced910285a0714b09e167fc84dd8490c0c9ea1c68cb397d07847227961efece`。

同一真实权重 partial-layer、相同 seed/input 顺序的 graph 诊断如下；
没有 collective，不能替代整模型 throughput：

| tokens | local routes / distinct experts | zero-r1 ms | compact-r1 ms |
| --- | --- | --- | --- |
| 1 | 2 / 2 | 0.528335 | 0.505768 |
| 2 | 1 / 1 | 0.334313 | 0.339203 |
| 3 | 11 / 11 | 2.389827 | 2.379788 |
| 5 | 8 / 8 | 1.865002 | 1.861959 |
| 8 | 28 / 19 | 5.942223 | 4.513013 |

只有最后一组包含重复本地专家，延迟降低 **24.05%**；其余组不走 gather，
差异不能视为这次算法的收益。原始记录 `layer-compact-r1.jsonl`。
真实整模型 `mtp2-r5` 的三个 smoke 已正确结束：323、完整 1–50、
Python `[0,4,16]`。counting 为 16.287 tok/s，不代替 sustained coding。
保持 k=2 / FULL `[1,3]`，请求 runtime stats 确认三-token FULL replay；
capture 仍为 0.38 GiB，KV cache 12.91 GiB，没有同时更改 QSA dispatch。
证据 `http-routed-mtp-smoke-mtp2-r5.jsonl`，host
`server-routed-mtp-mtp2-r5.log`。

三题 short coding 均正常生成 512 tokens：**15.000 / 12.669 / 12.926
tok/s**，中位数 **12.926**，比之前 k=2 的 13.034 低 **0.82%**。
第一题达到 15 不等于整体达到目标；三个 output SHA 都改变，
drafted/accepted 为 380/321、448/289、438/293，第三题 acceptance 从
0.713 降到 0.669。只能确认局部 repeated-expert kernel 的收益，不能
宣称 whole-model median 提升。原始证据 `w4-mtp2-r5-short.jsonl`。
约 23.4k 长三题已完成：**10.946 / 9.777 / 9.251 tok/s**，中位数
**9.777**，比旧 10.120 低 **3.39%**。全部生成 512 tokens、复用
23,168 prefix tokens；drafted/accepted 为 380/323、422/302、444/290。
冷 TTFT 358.307 秒，热 TTFT 4.522/4.674/5.159 秒；三个输出 SHA 均变化。
因此局部 kernel 收益没有转化为本轮整模型中位数提升，仍未达到 W8。
原始记录 `w4-mtp2-r5-long.jsonl`，host sequence 已输出
`QWEN_W4_COMPACT_MTP2_SEQUENCE_COMPLETE`；服务仍留在 :8002，未自动
替换 W8 launcher。下一独立优先级为上面的 QSA decode dispatch。

## 第九个候选：W4 MTP 的 grouped QSA dispatch

仅对显式 `cube_310_routed`，将 batched decode 上限从 2 扩展到 8，
预创建的 int64 group-list 同步覆盖 `8 * local_kv_heads`。W8、eager-W4、
其它 Cube backend 仍为 2；prefill、混合批次、少于 256 个 selection groups
和多请求的原 dispatch 限制不变。没有新增环境变量、host routing、cache
复制或常驻 expert bank；使用现有 NZ gather + grouped matmul 实现。

本地 **128 个 CPU tests 通过**（5.04 秒，仍排除两个已有 vLLM mock
不兼容项），含 backend × TP1/2/4 的 group-list 所有权、非持久化、
decode/prefill/selection-width 边界检查。新增 30 个 NPU replay cases，
覆盖 T=1/2/3/5/8、Q/KV heads=6/1、12/1、24/2、256/512 groups；同一
graph 逐次改变 Q/K/V、随机物理页表、selected groups/counts 和 tail，
并先以 NaN 污染输出，与旧 native sparse attention 比较。

首次测试进程没有对齐 serving 的 `jit_compile=False`，在 capture `Mul`
时报 legacy aclop 不支持；保留 `qsa-mtp-r1-tests.log`。测试改为 serving
已有的 ACLNN 模式后，完整 QSA 文件 **72 passed，49.45 秒**，保留
2,207 条旧 CANN/Python deprecation warnings，不将其隐去。
成功证据 `qsa-mtp-r2-tests.log`，未改变 serving 的编译模式。

独立 graph benchmark 为 synthetic QSA-only、24,576-token cache、512
selection groups，30 replays × 5 trials；不是整模型 tok/s：

| query tokens | sparse ms（Q/KV=6/1） | grouped ms（Q/KV=6/1） |
| --- | --- | --- |
| 1 | 1.303978 | 0.193993 |
| 2 | 1.687083 | 0.249664 |
| 3 | 2.071416 | 0.296986 |
| 5 | 3.061050 | 0.375953 |
| 8 | 3.071496 | 0.411330 |

12/1、24/2 也完成，共 15 个形状；最大绝对输出差 7.63e-6。
原始记录 `qsa-mtp-graph-r1.jsonl`；工具
`tools/qwen4exp/benchmark_qsa_mtp_graph_310.py`。三-token 局部延迟下降
85.66%，不能将此百分比用于整模型。kernel 仍为 `ops-compact-r1`，
真实模型 `mtp2-r6` 保持 MTP k=2 + FULL `[1,3]`；三个真实 smoke
均正确完成，runtime 表确认三-token FULL replay。短 coding 为
**15.167 / 12.555 / 13.601 tok/s**，中位数 **13.601**，比 12.926
高 5.22%。drafted/accepted=378/322、456/283、422/302；输出 SHA
均变化，不能将全部差异归于 QSA。weights 19.38 GiB/rank、graph
0.38 GiB、KV 12.91 GiB，W8 launcher 未改。23.4k benchmark 已完成：
**12.856 / 10.933 / 10.919 tok/s**，中位数 **10.933**（比 9.777 高
11.82%）。三题均生成 512 tokens、复用 23,168 prefix tokens；
drafted/accepted=364/331、428/299、426/298，输出 SHA 均变化。
cold TTFT **357.285 秒**，三个热 TTFT 4.566/4.716/5.270 秒。
原始证据 `w4-mtp2-r6-{short,long}.jsonl`、
`http-routed-mtp-smoke-mtp2-r6.jsonl`；host server log 为
`server-routed-mtp-mtp2-r6.log`。目标仍未达成：W8 为 19.073/18.091，
此前 k=1 长上下文中位数 12.754 也仍高于本轮 k=2。

下一独立诊断是 `replay-profile-r4` 的真实 k=2 长上下文 trace，保留
当前 compact kernel、shared RoPE 和 grouped QSA。benchmark 全部
完成、server idle 且 PID 身份核验后才停止旧隔离服务；只重启 :8002。
先跑三个真实 smoke，再以相同 prefix warmup，随后延迟四步采八步，
128-token profiled 请求必须完整结束。profiled tok/s 不作为性能基准；
旧 r3 k=1 trace 的各类比例不能直接当成当前归因。

## Current k=2 trace: replay-profile-r4

The fresh long-context trace completed on all four ranks with the compact
projection kernel, shared RoPE, and W4 grouped-QSA dispatch. All three HTTP
smokes passed. The profiling request completed 128 tokens after a 32-token
warmup, with 23,168 cached tokens out of a 23,407-token prompt. Runtime graph
metrics report three-token `FULL` replay. The profiled request's 11.289 tok/s
is **not** an unprofiled performance result.

Each rank contains exactly 1,152 routed projections: eight captured iterations
times 48 main-model layers times three projections. The inputs contain 30
routes, consistent with MTP k=2. Offline export was run after collection;
`profile-analysis-r4.log` ends with `QWEN_W4_PROFILE_R4_ANALYSIS_COMPLETE`.

| Attribution across ranks | Summed task time / eight iterations | Share of summed task time |
| --- | --- | --- |
| W4 routed projections | 394.027–434.678 ms | 30.64–32.83% |
| QSA index scoring | 307.785–311.666 ms | 23.28–24.24% |
| NZ matmul (`MatMulV2`) | 189.119–191.182 ms | 14.34–14.87% |
| AI-CPU `Cast` | 27.337–30.884 ms | 2.06–2.36% |

This is work attribution, not an additive critical-path model. The device task
spans are 1.792–1.801 seconds; task unions are 1.285–1.319 seconds. Neither
uncovered time nor a pipeline counter is by itself proof of a host bottleneck.
The 192 AI-CPU casts are still present, but the old r3 cast count/time cannot
be reused for the changed runtime.

For gate/up, median task duration is 305.547–346.081 us, with 20 blocks.
Reported median scalar time is 126.422–144.559 us, vector 87.786–106.640 us,
and Cube 6.887–8.609 us. Down projection uses 80 blocks with task duration
391.224–413.854 us, scalar 186.877–197.938 us, vector 78.531–95.549 us, and
Cube 5.739–7.174 us. Hardware pipeline counters can overlap and are not
summed to predict acceleration. Missing counters are preserved as missing,
not converted to zero; the summary parser has 11 passing CPU tests.

Evidence: `replay-r4/kernel-summary.json`, `replay-r4/profile-request.jsonl`,
and `replay-r4/http-smoke.jsonl`. Raw trace files remain under the isolated
host's `replay-profile-r4/` directory. No W8 launcher or runtime was changed.

The next isolated candidate caches the at-most-80 route IDs in 512 bytes at
the tail of each persistent block's UB. Owner detection, matching-input
gather and output scatter reuse this list. IDs are loaded on every kernel
execution, with aligned DMA plus a bounded scalar tail, rather than rounded-up
reads beyond the input. The projection math and order, weights, scratch ABI,
MTP settings and QSA dispatch are unchanged. This is a testable scalar-load
hypothesis, not a claim that it removes all the reported scalar time.

An independent follow-up is the index scorer's repeated query loads/casts.
Rank 0 records 112 calls: 104 have `[3,4,128]` queries with one request and
eight have the same query shape with a three-request metadata layout. The
one-request calls have median duration about 2.72 ms, versus about 3.47 ms
for the latter; the shape alone does not establish the Python call site.
In `QsaIndexerScoreV310::Compute`, each visible compressed-key group reloads
and casts every head of the same query. Caching those query heads per token
inside a core is a plausible arithmetic-preserving candidate, but has not
been implemented, measured or applied to W8. Keep it separate from route-ID
cache A/B evidence.

### 被拒绝的 route-ID cache 候选

`ops-route-ids-r1` 编译成功，193 项 NPU tests 全通过（175.80 秒，15 条
已有 warnings），其中增加了 DMA 八-ID 边界及 scalar tail 的动态 replay
覆盖。group kernel SHA 没变；routed SHA 为
`1074c9e241ce4d86bbaf3e18a4700a08b0b704bab90cf95e506e084d0787849c`。
但相同真实 layer-0 权重、seed、input 顺序的 partial-layer graph A/B
没有实际收益：

| tokens | compact 基线 ms | route-ID UB ms | 延迟变化 |
| --- | --- | --- | --- |
| 1 | 0.506741 | 0.510981 | +0.84% |
| 2 | 0.338894 | 0.341191 | +0.68% |
| 3 | 2.387205 | 2.394501 | +0.31% |
| 5 | 1.873130 | 1.867247 | -0.31% |
| 8 | 4.508377 | 4.565189 | +1.26% |

因此不启动这个 kernel 的整模型 A/B、不合入 runtime 源码，也不宣称
tok/s 增益。候选补丁及新增测试保存在
`replay-r4/route-id-cache-rejected.patch`，原始记录为
`replay-r4/layer-route-ids-{before-r1,r1}.jsonl`。已验证的 compact kernel
仍被后续服务使用；不是恢复或改动 W8 production。

### 第十个候选：W4 三/五-token QSA score GEMM

进一步源码检查发现 index scoring 也有独立的两-token dispatch 上限：
`_QSA_MATMUL_DECODE_MAX_TOKENS=2`。k=2 的三-token target verification
因此走 native per-group score，而旧 k=1 可以走已实现的 FP32 GEMM。
这一发现优先于新写 native query-cache kernel。

为 selector 添加 keyword-only `max_matmul_decode_tokens`，默认仍为 2。
只从 routed-W4 attention 传入已有八-token bound；W8/其它 backend
仍传 2。eager 路径及 breakable-graph 的 `select_current_groups` 回调
都传入相同值。单 request、至少 2,048 groups、现有长 prefill dispatch
和中间 tensor 大小限制均保留；没有新增环境变量、NPU kernel、host
routing 或改变缓存格式。score/select 仍按原设计在 graph segments
之间执行，以处理不断增长的 visible table，不能把它描述成无 graph
的模型。query/weight projection 和 decode graph 继续保留。

63 项本地 CPU tests 通过（3.93 秒，16 条已有 warnings）。最初误用
仅 ModelSlim 的 CPU venv 缺少 vLLM/zmq；切换到已有 vLLM venv、pytest
临时 overlay 和本地 vLLM checkout 后通过，并未更改生产依赖。
NPU QSA 文件 **84 passed / 50.33 秒**（2,207 条已有 warnings），包括
12 个新 cases：T=3/5/8、index heads=4/16、2,048/5,856 groups；逐轮
改变 query、cache、物理页表及 visible count，覆盖 0/511/512/最大值、
tail 和全零 query 的精确 tie。与 native scorer 比较同一 group set，
允许集合内近似相等 score 的 FP32 reduction 排序差异，不宣称 bitwise
输出一致。原有 downstream QSA dynamic graph tests 也全部通过。

15 个 score+stable-selection 局部 A/B 均完成，30 iterations × 5 trials；
符合真实 eager-between-segments 调用方式，不是全模型 throughput：

| groups | query tokens | 默认 selection ms | W4 MTP selection ms |
| --- | --- | --- | --- |
| 2048 | 3 | 1.121252 | 0.605082 |
| 2048 | 5 | 1.740284 | 0.502085 |
| 2048 | 8 | 2.666677 | 0.553600 |
| 5856 | 3 | 2.921905 | 0.506435 |
| 5856 | 5 | 4.698810 | 0.504334 |
| 5856 | 8 | 7.335426 | 0.505778 |
| 8192 | 3 | 4.000451 | 0.510273 |
| 8192 | 5 | 6.408426 | 0.590703 |
| 8192 | 8 | 10.217293 | 0.586686 |

1/2-token 两列是同一个 dispatch，只反映测量波动，不能计作算法收益。
工具为 `tools/qwen4exp/benchmark_qsa_score_mtp_310.py`，原始记录
`replay-r4/qsa-score-mtp-r1.jsonl`。新的真实模型 run `mtp2-r7` 继续使用
compact kernel、MTP k=2、FULL `[1,3]`、TP4/EP、32,768 context、.90
memory fraction。三个真实 smoke 均正确终止：323、1–50、`[0,4,16]`，
计数 16.305 tok/s。runtime 表确认三-token `FULL`，MTP counters 递增。
短三题为 **14.907/12.977/13.363 tok/s**，中位数 **13.363**，比旧
13.601 低 1.74%；该修复不在短前缀的评分 dispatch 生效，不宣称短题提速。
长三题为 **14.440/13.683/12.241 tok/s**，中位数 **13.683**，比旧
10.933 高 **25.15%**。全部六题生成 512 tokens；长题均复用 23,168
prefix tokens，prompt tokens 为 23,407/23,418/23,424。长题
drafted/accepted=384/320、402/310、446/288，三个输出 SHA 都变化，
不能把整个中位数差异归因于单一算子。短题 drafted/accepted=
388/318、442/291、428/298。冷 warmup TTFT **356.843 秒**仍未改善；
长题热 TTFT=4.431/4.661/4.971 秒。

model-load 仍为 19.3821 GiB/rank，graph capture 0.38 GiB；没有新增
常驻 expanded expert bank。最高记录温度 75°C，W8 launcher SHA256
仍为 `ae81d75eec7516fc6f455a3a071e52410ba6c477c34e0114df647f4c125c77df`。
本地原始 smoke、短/长请求记录保存在 `replay-r4/`。六题和 final marker
完成、确认无运行/等待请求后，才向准确 API PID 3377501 发 TERM，
开始下一项隔离 kernel 实验；关闭期间的 EngineDeadError 保留，不把
主动关闭与 benchmark 运行故障混为一谈。本轮 CPU gate 重跑为
63 passed / 3.79 秒，另两个 build-manifest tests 通过。
W8 的 19.073/18.091 tok/s 仍未达到；32k 单会话结果不是容量或完整
coding accuracy 评测，不能据此给 W4 生产等价结论。

## 未保留：small-M Cube K=512 stages

为降低 projection 中 copy/event/scalar 指令数量，试验仅对 compact
M≤32 使用 512-wide Cube K stage，M>32 保持 128；K=640 的最后
128 个元素单独处理。L1 activation 总分配不增加，double-buffered
L0A/L0B 分别最多 64 KiB，编译时断言边界；packed bank、解包和共享
W8/GLM helper 不变。独立 vendor 为 `ops-cube-k-r1`。

第一次新增测试误用不受支持的 K=128，binding 正确拒绝，未发生 NPU
kernel fault。改用受支持的 K=256，仍通过 K=640 覆盖 128-wide tail；
最终 **195 NPU tests passed / 137.12 秒**，15 条已有 warnings。
覆盖 M=31/32/33 dispatch 边界、改变权重的 replay、compact owner
长度变化和 peer rows。构建 binary SHA256：

```text
group:  da3a88dee3753c7733dde5ccdfc16f2a8684b1c9faadbcecaa24c4005c0f2531
routed: d020321d7606711da74d239e74ab2a17a5c736a8acecbc61b60acd5d26a11b31
```

但相同真实 layer-0、synthetic input/route 序列、无 collective 的
graph replay 全部略慢，不能进入完整模型：

| tokens | compact 基线 ms | K=512 候选 ms | 变化 |
| --- | --- | --- | --- |
| 1 | 0.507837 | 0.521943 | +2.78% |
| 2 | 0.343229 | 0.345853 | +0.76% |
| 3 | 2.382508 | 2.437904 | +2.33% |
| 5 | 1.861499 | 1.895122 | +1.81% |
| 8 | 4.519393 | 4.600229 | +1.79% |

候选源码及新增测试仅归档在 `replay-r4/cube-k-rejected.patch`，未进入
运行源码；matched JSONL 也在该目录。没有用这个 binary 启动模型。
下一项是以已验证 compact kernel + 两个 QSA dispatch 修复，重新测
MTP k=4 的 `[1,5]` FULL replay；旧 k=4 慢速结果早于这些修复，
不能当成当前版本结果。新 run `mtp4-r2` 的结果见下一节。

## QSA 修复后的 k=4 实测

`mtp4-r2` 使用相同 compact binary、共享 RoPE 和两个 W4-only QSA
dispatch 修复，仅改 MTP k=4 / FULL `[1,5]`。三个真实 smoke 正确，
1–50 counting 为 20.198 tok/s；不能用这个简单题代替 coding benchmark。
运行时表反复确认五-token FULL replay，drafted/accepted counters 增长。
六个无 profiler coding 请求均完成 512 tokens，结果如下：

| context | 三题 decode tok/s | 中位数 | 相对同代码 k=2 |
| --- | --- | --- | --- |
| 短 | 15.783 / 11.199 / 12.189 | 12.189 | -8.79% |
| 约 23.4k | 15.843 / 12.025 / 11.392 | 12.025 | -12.12% |

短题 drafted/accepted 为 536/378、748/328、688/339；长题为
492/389、636/352、680/345。第一题改善，但后两题 acceptance 不足以
抵消更多 draft/verify 工作。全部输出 SHA 与 k=2 不同，不能声称是
bitwise 等价的 workload。长题均复用 23,168 tokens，hot TTFT 为
4.660/4.728/5.258 秒；cold warmup TTFT=361.316 秒。

权重仍约 19.38 GiB/rank，graph 从 k=2 的 0.38 增为 0.50 GiB；
watchdog 最高 75°C。没有启用 rejected Cube binary，也未改生产 W8；
home launcher SHA256 仍为
`ae81d75eec7516fc6f455a3a071e52410ba6c477c34e0114df647f4c125c77df`。
原始请求文件为 `replay-r4/w4-mtp4-r2-{short,long}.jsonl` 与
`http-routed-mtp-smoke-mtp4-r2.jsonl`。服务/温度日志在 host
`/srv/ai/src/qwen38-w4-hardware-20260927/` 下的
`server-routed-mtp-mtp4-r2.log` / `server-routed-thermal-mtp4-r2.log`。

因此保留 k=2 作为下一次 profile 配置，不将 k=4 推为更快默认。
完成全部请求并确认 idle 后，原 API PID 3479201 正常退出；新的
`replay-profile-r5` 已启动（API PID 3549264），采样仍为 delay=4、
max=8 steps。其 trace 尚未完成；不能把旧 profile 的瓶颈比例当成
QSA score 修复后的实测，也不能把带 profiler 的请求算作速度基准。

## QSA score 修复后的新 trace：replay-profile-r5

保留 compact vendor、MTP k=2、FULL `[1,3]` 和两个 W4-only QSA
dispatch 修复。三个真实 smoke 正确结束，冷前缀 warmup 完成后采集
delay=4 / max_iterations=8 的 worker trace；128-token profile 请求
完整结束，cached_tokens=23168。其 12.538 tok/s 含 profiler 扰动，
不能替代无 profiler 的三题中位数。冷 warmup TTFT=357.803 秒。
四个 rank 均导出成功，每 rank 1152 次 W4 projection，符合
8 × 48 × 3；host 原始目录为 `replay-profile-r5`，本地 CSV 副本
为 `/tmp/qwen38-w4-full-replay-r5`，可审计摘要见 [`replay-r5/`](replay-r5/)。

| rank | task span ms | task union ms | W4 projection 总和 ms | W4 占累计 task time |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1596.860 | 1048.758 | 432.687 | 41.18% |
| 1 | 1593.259 | 1072.832 | 451.840 | 42.00% |
| 2 | 1594.879 | 1092.102 | 473.386 | 43.03% |
| 3 | 1598.390 | 1060.511 | 442.861 | 41.50% |

原 native `QsaIndexerScoreV310` 从 r4 每 rank 112 calls 降至 8，
累计 22.504–24.431 ms，剩余 shape 为 `[3,4,128]` query 搭配
三请求页表；不能直接套用单请求 GEMM。NZ MatMul 累计约 189–190 ms。
INT64 FloorDiv 仍有 361–363 calls、17.554–23.059 ms，192 次 AiCPU
Cast 为 26.108–32.765 ms。上述时间可能重叠，不是可直接相加的提速。

rank0 的八个最长 `aten::copy_` 为 99.378–113.128 ms；各自内部
同时有 76.636–87.286 ms 的 device-task 区间并集。这些 host 等待
不等价于纯 memcpy 成本。其父算子是 `aten::to/_to_copy`，紧邻
`argmax` 后的 host-to-device copy。源码中的通用 greedy rejection
path 有未 pin 的 draft-count H2D、动态 boolean indexing 和
`if torch.any(...)`；k=1 的专用路径不经过该函数。它是下一步的
设备端固定形状 sampling 诊断线索，后续候选见下；未证明为全部等待的来源。
不能将 sampled host self time 与 device time 再次相加。

## 第十一个候选：复用 QSA causal position 商

selection 原 GEMM 分支重复计算三次 `(position + 1) // ratio`；
native 分支也算两次。新 helper 只做一次 INT64 floor division，
共同产生 visible groups、tail start 和 tail count。保留负 padding、
大整数、clamp 与稳定 ties 的语义，不改成 INT32，不新增同步或环境变量。

80 个 CPU tests 通过（3.80 秒），新增 17 项覆盖非连续输入、空 batch、
负位置、ratio=3/4 和超过 INT32 的位置。92 个 NPU QSA tests 通过
（31.74 秒），包括新增八项整数语义回归及原有动态 replay 检查。
新的 `benchmark_qsa_geometry_310` 比较两个源码 snapshot、记录 SHA256，
20 个 capacity/token 组合的四个 selection fields 全部逐元素精确一致。

| capacity groups | query tokens | 原 selection ms | 复用后 ms | 变化 |
| --- | ---: | ---: | ---: | ---: |
| 512 | 3 | 0.413127 | 0.366174 | -11.37% |
| 2048 | 3 | 0.513780 | 0.399147 | -22.31% |
| 5856 | 3 | 0.509781 | 0.396926 | -22.14% |
| 8192 | 3 | 0.517764 | 0.402167 | -22.33% |

每个 snapshot 为 30 iterations × 5 trials，reference 先于 candidate，
不是随机交错实验；全部 20 个局部中位数降低 6.14–26.77%，但这仅是
breakable graph 间的 eager callback，不是整模型加速比例。采集结束、
server 空闲且准确 PID tree 完全退出后才运行 NPU tests 和 timings。
温度监督保留，W8 launcher SHA256 仍为
`ae81d75eec7516fc6f455a3a071e52410ba6c477c34e0114df647f4c125c77df`。

完整模型 `mtp2-r8` 保留同一 compact binary、MTP k=2 和 FULL
`[1,3]`；三个真实 smoke 正确结束，runtime 三-token FULL 与 MTP
计数递增均有证据。短三题全部完成 512 tokens，为
14.919 / 13.183 / 13.213 tok/s，中位数 **13.213**，比 r7 的
13.363 低 1.12%。drafted/accepted=382/321、432/297、430/298；
只有第一题的输出 SHA 与 r7 相同，acceptance 仍变化；短题没有形成
提速证据。长三题已完成，14.639 / 12.990 / 12.119 tok/s，中位数
**12.990**，比 r7 的 13.683 低 5.07%。每题生成 512 tokens、复用
23,168 prefix tokens；drafted/accepted=378/322、422/301、450/288，
输出 SHA 与 acceptance 改变。冷 TTFT 358.321 秒，没有整模型提速
证据。尚未达到 W8 的 19.073 / 18.091，不把局部 selection 降时
当作已实现的 generation speed 提升。

## 第十二个候选：固定形状 multi-draft greedy rejection

新增 `sample/uniform_greedy_rejection.py`，仅在各请求 draft 长度一致且
为 2–8 时提前处理。scheduler 的 Python lengths 用于确认矩形布局，
接受/拒绝和 greedy mask 保留在设备上。静态有界循环使用 `where`
与逐列 copy，避免 pageable draft-count H2D、动态 boolean indexing、
`nonzero`、`if torch.any`。拒绝后的 suffix、非 greedy 行以及多余
output columns 保留调用方原值。k=1、ragged、空 draft 或超过八个
draft 仍走原路径；不改 random sampling 或概率。没有新增环境变量。

55 项 CPU tests 覆盖至八 drafts 的全部 acceptance patterns、mixed
greedy、int32/int64 输出、非连续输入、synthetic probability 边界、
负 draft ID 和 fallback 不改输出。64 项 310P tests 通过实际 dispatcher
验证 eager 与 graph 完全一致，capture 后改变 drafts、targets、bonus、
概率和 greedy mask，并用 777 污染 output 检查 replay 清理。合计
119 项通过（15.86 秒），既有 sampler UT 另有 21 项通过（11.09 秒）。

第一轮 microbenchmark 在 old reference 的全接受 case 因 bonus int64 /
output int32 不匹配报错；这是工具未遵守旧接口 dtype 的问题，不是新
算子故障。已把 benchmark bonus 改为与实际 sampler 一样的 int32，
保留失败日志 `uniform-rejection-r1-bench.log`，使用新输出 r2 重跑，
没有覆盖失败证据。源码和 runtime 候选未因此修改。

| drafts | batch | 全接受 | 原 eager ms | 新 eager ms |
| ---: | ---: | --- | ---: | ---: |
| 2 | 1 | 否 | 1.824482 | 0.206545 |
| 2 | 1 | 是 | 2.022500 | 0.181590 |
| 4 | 1 | 否 | 1.679657 | 0.304908 |
| 4 | 1 | 是 | 1.964977 | 0.342370 |
| 8 | 1 | 否 | 1.687348 | 0.613797 |
| 8 | 1 | 是 | 2.122682 | 0.569095 |

全部 18 个 cases（drafts 2/4/8 × batch 1/4/16 × accept/reject）
输出精确一致，50 iterations × 5 trials 的局部中位数降低 63.62–91.98%。
old reference 先运行、candidate 后运行，并非随机交错；这些是独立
sampler timings，不把约 100 ms 的 profile host copy wait 当成可消除
的 sampling 成本，也不外推等比例整模型 gain。原始数据及源码 SHA256
见 `replay-r5/uniform-rejection-r2.jsonl`。

`mtp2-r8` 六个请求和 smoke 完成、server 空闲且准确 PID tree 正常退出
后才替换隔离 W4 runtime 的 sampler 并运行上述检查。W8 环境和 launcher
未修改。新的 `mtp2-r9` 使用同一 compact kernel、TP4/EP、MTP k=2、
FULL `[1,3]` 和同一短/23.4k benchmark protocol。三个真实 smoke 已正确
结束（323、完整 1–50、`[0, 4, 16]`），runtime 三-token FULL replay
计数及两 draft positions 的 acceptance 均有证据。权重仍为
19.3821 GiB/rank，graph 仍为 0.38 GiB。

短三题各完成 512 tokens：**15.358 / 13.239 / 13.729 tok/s**，中位数
**13.729**，比 r8 的 13.213 高 3.90%。drafted/accepted 为
382/321、440/291、426/300；所有输出 SHA 改变，第一题虽然计数与
r8 相同但输出并非 bitwise 相同。不能把全部差异归因于 sampler，
也不能用 counting smoke 的 16.549 tok/s 代替 coding median。
`replay-r5/w4-mtp2-r9-short.jsonl` 与相邻 smoke JSONL 保留完整结果。
长三题也完成 512 tokens：**15.510 / 13.689 / 12.834 tok/s**，
中位数 **13.689**（比 r8 的 12.990 高 5.38%，与 r7 的 13.683
接近）。均复用 23,168 prefix tokens，cold warmup TTFT 357.555 秒；
drafted/accepted 为 364/329、408/308、436/293，不能用局部 sampler
收益直接解释全部差异。完整结果见 `replay-r5/w4-mtp2-r9-long.jsonl`。
W8 19.073/18.091 目标未达成。完成标记、空闲 metrics 及 PID tree
复核后正常停止隔离 :8002，以执行下一项 gate/up 合并诊断；未恢复
或修改 W8。历史 API PID 3684371、engine 3685195、workers
3685525–3685528；host 日志 `server-routed-mtp-mtp2-r9.log`。

本候选 scoped manual hooks 全部通过，只有同一未改动的
`check-symbolic-meta` 第 655 行问题需跳过；不是全库 lint 全绿。

## 第十三个候选：合并 packed gate/up projection

gate/up 使用相同激活与 expert IDs，沿 N 合并两个已编码 bank，
在现有 kernel 中将两次 N=640 调用变成一次 N=1280 调用。N tile
仍为 32、K reduction 仍为 128，未重编译或修改 kernel，也未增加
常驻 FP16/W8 shadow。生产分配直接创建 `gate_up_proj`，checkpoint
的两个原始投影逐 expert 写入对应半区；不存在常驻旧 bank 或完整
bank 的加载时拼接。只有诊断程序同时保留两种布局用于对照。
非 routed W4 后端和生产 W8 保持原路径。host-prefill fallback
也使用合并 bank，但保留现有 FP32 SiLU/乘法及 FP16 down 输入。

CPU 证明 pack(concat) 与 concat(pack) 逐位相同；增加非零 rank、
不均匀 ownership、up-first 加载、缺少 up checkpoint 必须拒绝、
真实 model loader 返回参数名及 packed 总字节数测试。focused CPU
suite **73 passed / 1 skipped**（缺少可选 msmodelslim）。隔离 host
**38 passed / 1 skipped**，包含 31 个 CPU UT 和 7 个 NPU cases：
prefill rows 1/8/128，routed rows 10/30/50/80；后者 replay 四种
不同 ownership，先以 NaN 污染输出，确认 local/peer 切换仍精确覆盖。
随后完整 `tests/ut/qwen38_1m` suite 为 **950 passed / 7 skipped**，
50.04 秒；不是全库或多型号硬件测试。

真实 layer-0 / TP-rank-0 partial、synthetic activations、无 collective
的图回放对照（30 iterations × 5 trials，交错测量顺序）：

| tokens | separate ms | combined ms | 减少 |
| --- | --- | --- | --- |
| 1 | 0.503440 | 0.479937 | 4.67% |
| 2 | 0.342808 | 0.319739 | 6.73% |
| 3 | 2.379038 | 2.159219 | 9.24% |
| 5 | 1.858981 | 1.697172 | 8.70% |
| 8 | 4.498129 | 4.088642 | 9.10% |

每个 shape 的 eager 和四次 changing-input replay 输出均精确一致。
gate/up 字节数两边均为 219,545,600；这不是整模型吞吐测试。
集成前 r1 也全部正确、减少 5.86–9.13%。证据为
`replay-r5/gate-up-r{1,2}.jsonl` 和相邻 test logs。原 routed kernel
SHA256 为 `1ced910285a0714b09e167fc84dd8490c0c9ea1c68cb397d07847227961efece`，
新 `w4_moe.py` SHA256 为
`49d2a6b639ae068d8568819a7bb83d90dabac8285ed3a55a3755d652871fadf1`。

仅在上述 gates 完成后启动 `mtp2-r10`，保持 :8002 / TP4+EP /
MTP k=2 / FULL `[1,3]` / .90 / 32k / seq1。三个真实 smoke 均正确：
323、完整 1–50、`[0, 4, 16]`。三-token FULL runtime replay 有计数。
权重仍为 19.38 GiB/rank；graph memory **0.53–0.54 GiB**，
相比 r9 的约 0.38 GiB 增加约 150 MiB，不能把 packed-bank 不变
表述为全部运行显存不变。

短三题各完成 512 tokens：**15.928 / 13.148 / 13.841 tok/s**，
中位数 **13.841**，比 r9 的 13.729 高 **0.82%**。第一题 draft
计数与 r9 一致（382/321），但输出 SHA 改变；第二/三题分别为
460/281、438/292，接受率下降。局部 layer 的 4.67–9.24% gain
未等比例转化为生成速度，不能宣称已达到 W8 或完整任务质量等价。
原始记录 `replay-r5/w4-mtp2-r10-short.jsonl` 和相邻 smoke JSONL。
23.4k 三题也完成 512 tokens：**15.599 / 13.903 / 12.821 tok/s**，
中位数 **13.903**，比 r9 的 13.689 高 **1.57%**。三题均复用
23,168 prefix tokens，drafted/accepted 为 374/324、416/304、450/287；
cold warmup TTFT 324.894 秒。原始记录为
`replay-r5/w4-mtp2-r10-long.jsonl`。所有六个 coding 请求均为
`finish_reason=length` 的固定 512-token 吞吐测试，不是完整 coding
任务正确率评测。完成标记和 running/waiting=0 已复核，无 runtime
ERROR/Traceback，server 保持运行；API 3782927、engine 3783652、
TP workers 3784185–3784188，host log
`server-routed-mtp-mtp2-r10.log`。生产 W8 launcher
SHA256 仍为 `ae81d75eec7516fc6f455a3a071e52410ba6c477c34e0114df647f4c125c77df`。

## k=1 重测和 CPU affinity 对照

gate/up 合并后再测 MTP k=1 / FULL `[1,2]`，其它设置不变。
短三题 14.265 / 12.984 / 13.496，23.4k 三题
14.063 / 13.283 / 13.035 tok/s，中位数 **13.496 / 13.283**。
三题均为 512 tokens / finish=length；长题复用 23,168 prefix tokens，
cold warmup TTFT 323.856 秒。三个真实 smoke 正确，two-token FULL
runtime 计数可见，不是 eager 或无 MTP 的结果。

W8 worker 启动日志及实际 affinity 为 `(0–5)/(8–13)/(16–21)/(24–29)`；
W4 日志为 `CPU binding skipped: non-ARM CPU detected`，四个
worker 原 mask 都是 `0–63`。在同一 W4 服务完成请求且空闲后，
核对 API/engine/worker 父子关系，仅绑定四个 worker 的所有线程，
rank0 共 87 线程、其它 ranks 各 86 线程；没有改 IRQ、W8 或 kernel。

再次通过三个 smoke 后，同服务、不重启的 affinity 对照为：

| case | unbound median tok/s | bound median tok/s | 变化 |
| --- | --- | --- | --- |
| short | 13.496 | 13.853 | +2.64% |
| 23.4k | 13.283 | 13.781 | +3.75% |

bound 短三题 14.675 / 13.572 / 13.853，长三题
14.261 / 13.781 / 13.329。输出 SHA 与 acceptance 也变化，不能将
整个差值都归因于 affinity；这是单次 A/B，并非统计显著性结论。
`decode_s / drafted_delta` 粗略 step 代理由短 134–135 / 长 138–139
降到 131–132 / 134–135 ms，不是 profiler 精确 step latency。
**仍未达到 W8 19.073 / 18.091，未将此 host 特定绑定写入通用默认值。**

证据在 `replay-r5/w4-mtp1-r2-{short,long}.jsonl` 和
`w4-mtp1-affinity-r1-{short,long}.jsonl`，相邻 smoke 文件保留答案。
host `mtp1-affinity-r1-applied.json` 保存旧/新 thread masks，
`affinity-sequence-r1.log` 有最终完成标记。API 3868105 的这组对照
已完整结束，确认 idle 和准确 PID tree 后正常停止，以做下一 kernel
的隔离硬件验证；不再是运行中的服务。W8 launcher SHA256 未变。

## 被拒绝的第十四个候选：整段 A 预加载到 L1

将能放入剩余 L1 的完整 activation 预加载一次，替代每个 K=128
tile 的 GM→L1 copy/event，保留原 reduction 次序、双 L0 和原
大 M fallback。K=2560 的 resident/fallback 边界为 M=64/65。
只修改 Qwen W4 的 tiled kernel；没有常驻 FP16/W8 shadow。

独立 `ops-preload-a-r1` 构建成功，compact 和 candidate 各通过
**170 个 NPU tests**（147.49 / 145.55 秒），覆盖 projection、所有
byte/offset、changing ownership、64→65→64、NaN scratch、完整
MoE 和共享 RoPE/合并 gate-up。旧 full-MoE 测试已修正为按 checkpoint
gate/up/down 分别加载到两种布局，不再复制不兼容的 state_dict。

真实 layer-0 / rank-0 partial、固定 seed 合成输入、无 collective，
tokens 1/2/3/5/8，40 iterations × 5 trials；进程均绑定 CPU 0–5，
按 baseline-A / candidate-A / candidate-B / baseline-B 测试。
每种 shape 的输入、eager 输出和 graph 输出在四轮均 SHA256 相同。
graph timing 合并每侧十次 trial 取中位数：

| tokens | compact ms | preload-A ms | 变慢 |
| --- | --- | --- | --- |
| 1 | 0.467028 | 0.493831 | 5.74% |
| 2 | 0.317500 | 0.329619 | 3.82% |
| 3 | 2.150634 | 2.279928 | 6.01% |
| 5 | 1.684382 | 1.781025 | 5.74% |
| 8 | 4.081794 | 4.312513 | 5.65% |

**正确但更慢，不进入服务默认路径。** 主 kernel source 保持已验证
compact 版本；仅保留增强后的测试和 profile 工具输出 hash。
候选 diff 在 `replay-r5/preload-a-rejected.patch`，完整 A-B-B-A 记录
为 `layer-preload-a-r1-*.jsonl`，两份 `kernel-preload-a-r1-*-tests.log`
保留测试证据。host 构建日志 `build-preload-a-r2.log`，group/routed
binary SHA256 分别为
`f0eb23d970156ae9a60c384cf268fc9a01c419b68898007767cd9fd979ac5790` /
`8b86e3f6106c3045623e2e16fb21e1318523a3354494a36c5f48a6c1529957b3`。

完成这些隔离测试后，启动 compact + MTP k=2 / FULL `[1,3]`
的 affinity 对照。此时尚未得到该新整模型 benchmark 的结果；
不可用上述 layer 测试代替整模型吞吐或质量评测。

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

# L1 候选，同样先通过真实 smoke，再测三条 512-token coding：
bash qwen38-w4-server-routed-check-r1.sh mtp2-r2 \
  /srv/ai/src/qwen38-w4-operator-build-r1/ops-l1-r1/vendors/qwen_w4_probe_transformer 2
bash qwen38-w4-mtp-short-benchmark.sh mtp2-r2

# 批量清零候选；143 项 NPU tests / layer parity 后才启动：
bash qwen38-w4-server-routed-check-r1.sh mtp2-r3 \
  /srv/ai/src/qwen38-w4-operator-build-r1/ops-zero-r1/vendors/qwen_w4_probe_transformer 2
bash qwen38-w4-mtp-short-benchmark.sh mtp2-r3

# 共享 RoPE 候选使用同一 vendor；先 stage 新 model.py/qsa.py 并通过
# 160 项 NPU gate（其中 17 项 RoPE），再用新的 run label 启动：
bash qwen38-w4-server-routed-check-r1.sh mtp2-r4 \
  /srv/ai/src/qwen38-w4-operator-build-r1/ops-zero-r1/vendors/qwen_w4_probe_transformer 2
bash qwen38-w4-mtp-short-benchmark.sh mtp2-r4

# compact matching activation rows, same MTP k=2 / FULL [1,3]
bash qwen38-w4-server-routed-check-r1.sh mtp2-r5 \
  /srv/ai/src/qwen38-w4-operator-build-r1/ops-compact-r1/vendors/qwen_w4_probe_transformer 2
bash qwen38-w4-mtp-short-benchmark.sh mtp2-r5

# same compact kernel, W4-only grouped QSA for MTP verification
bash qwen38-w4-server-routed-check-r1.sh mtp2-r6 \
  /srv/ai/src/qwen38-w4-operator-build-r1/ops-compact-r1/vendors/qwen_w4_probe_transformer 2
bash qwen38-w4-mtp-short-benchmark.sh mtp2-r6

# same kernel, W4-only QSA score GEMM for wider verification batches
bash qwen38-w4-server-routed-check-r1.sh mtp2-r7 \
  /srv/ai/src/qwen38-w4-operator-build-r1/ops-compact-r1/vendors/qwen_w4_probe_transformer 2
bash qwen38-w4-mtp-short-benchmark.sh mtp2-r7

# 当前 gate/up 合并候选：先 stage w4_moe.py，运行独立 NPU/real-layer gate；
# 此脚本验证结果后启动 MTP k=2、自动执行真实 smoke 和短/长 benchmark。
# 重跑需更改脚本中的 run label/output，不能覆盖旧证据。
bash qwen38-w4-gate-up-mtp2-r1.sh
```

服务配置见本页开头和 tutorial，仍为 :8002 / TP4 / EP /
FULL_DECODE_ONLY；本轮增加 k=2/k=4（分别 `[1,3]` / `[1,5]`），
不将启动成功当作验证通过。所有候选均保留 packed W4 bank。

本轮 scoped manual pre-commit 的 Ruff、codespell、typos、markdownlint、
secret scan 等均通过；全局 `check-symbolic-meta` 仍因未改动的
`csrc/torch_binding_meta.cpp:655`（已有 W2 meta 的 `empty_symint({`）失败，
已与 HEAD 对照，不把这个结果写成全库 lint 全绿。三个 C++ 文件另行
手动 clang-format（仓库 hook 本身排除 csrc）。未跑全库多型号硬件测试。
L1 候选的 raw JSONL request ID 曾被 typos 把随机 hex 子串当成拼写错误；
仅为 `"request_id": "chatcmpl-<hex>"` 增加精确 ignore regex，原始证据
保持逐字节不变，不豁免其它 benchmark 文本或源代码。

另外在临时快照 `/tmp/qwen-w4-replay-format.lZNQ6h` 执行了完整
`bash format.sh ci`，日志 `/tmp/qwen-w4-replay-format-ci.log`；失败项还包括
已有 prefill 工具的 Ruff 问题、其它模型/文档的格式及拼写、禁用 `re`
imports 等。自动格式修改只留在临时快照，没有污染任务 worktree 或生产文件。

## 尚未修复的 profile 线索

- rank0 有 384 个 AI_CPU `Cast`，八步共 60.698 ms（约 7.587 ms/step）；
  结合相邻 RoPE operations 和源码，优先检查整数 positions 的重复转换及
  Q/K/indexer 的频率与 sin/cos 重算。当前是归因线索，不是已验证加速。
  `project_qk` 的 Q/K 与 `model.py` 的 index-query 使用相同 rotary 参数
  和 positions，本轮已实现隔离 W4 候选，在一次 forward 内共享 compute-dtype 的表；index-key
  的压缩组位置不同，不能直接复用 query 表。保留 CPU FP64、NPU FP32、
  MRoPE 三轴和 graph replay 的动态 positions，不能缓存旧请求的表。
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
FULL replay、短/长 coding 也完成。
L1 候选通过 140 项 NPU tests、真实 layer 和三个整模型 smoke，
k=2 三-token FULL replay 与短/长 coding 均完成。批量清零版本也完成
三个真实 smoke、短/长 coding 和三-token FULL replay；RoPE 新候选也通过
三个真实 smoke 和三-token FULL replay，短/23.4k coding 中位数
**13.034 / 10.120 tok/s**。
匹配 k=1 已完成 **12.787 / 12.754 tok/s**；compact-row 版本通过
163 项 NPU tests、真实 partial-layer 和三个完整模型 smoke，短/23.4k coding
中位数 **12.926 / 9.777 tok/s**，均已完成，不宣称 W8 速度达标。
W4 grouped-QSA 候选通过 128 项 CPU、72 项 NPU tests 和三个真实
smoke，三-token FULL replay 与六个 512-token coding 请求全部完成；
短/长中位数 **13.601 / 10.933 tok/s**，仍未达到 W8。
进一步的 W4-only score-GEMM bound 通过 84 项 NPU QSA tests 和完整
smoke、MTP k=2、三-token FULL replay；短/长中位数为
**13.363 / 13.683 tok/s**，长题有所改善，但仍低于 W8。
本轮只隔离 batch-one decode，不声称 W4 已验证 128k×16、双 160k 或
完整任务质量。flashcomm1 和多模态未在本轮验证；容量和量化质量仍需后续
独立评测，不能伪造 accuracy YAML 分数。
