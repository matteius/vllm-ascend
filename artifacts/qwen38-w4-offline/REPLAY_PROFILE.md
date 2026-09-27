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
本轮只隔离 batch-one decode，不声称 W4 已验证 128k×16、双 160k 或
完整任务质量。flashcomm1 和多模态未在本轮验证；容量和量化质量仍需后续
独立评测，不能伪造 accuracy YAML 分数。
