# Qwen W4 Cube、设备 routing 与 MTP/graph 验证

## 结论与边界

最新 `cube_310_routed` 已通过真实整模型 MTP k=1 + decode graph 的三个
正确答案 smoke；1–50 计数为 11.592 tok/s，约为 tiled eager 的 2.71 倍。
这不是完整任务质量评测，也不是生产速度达标；长短上下文基准另行记录。

目标仍是超过同机 W8 + MTP + graph 的生产生成速度，而不是只超过
旧 W4 reference。已测 W8 短上下文中位数为 19.073 tok/s，约 23.4k
热上下文为 18.091 tok/s。本文的 projection 数据不能换算为整模型 tok/s。

模型为 48 层、512 routed experts、top-k=10、hidden=2560、expert
intermediate=640；TP4 每 rank 保留 128 个专家。量化 checkpoint 不变：
signed W4、低 nibble 在前、每行 group-128 FP16 scale 和 INT8 offset。
恢复权重为 `(q - offset) * scale`。没有建立全专家 FP16 或 W8 shadow。

## 实现

- 新增独立 `QwenW4GroupMatmulV310` CANN 算子及
  `torch.ops._C_ascend.npu_qwen_w4_group_matmul_310`；不改变旧 W2 算子。
- 每 logical tile 处理 32 个输出通道。向量解包直接 gather 到 Cube NZ
  布局，去掉每个 K tile 的八次 transpose、DMA 和对应同步。
- layout index 由 16 个 scalar seed 加 vector Adds 生成，不再逐元素构建。
- Cube 使用 FP32 accumulation、FP16 输出。adapter 检查 dtype、维度、
  contiguity、device，并用 device guard 保证非当前 NPU 正确执行。
- 新增显式 `cube_310` 后端；`eager_dequant` 仍是默认。metadata 不匹配、
  算子缺失或 CPU 输入直接报错，不静默回退。
- `cube_310`/`cube_310_tiled` 的 host routing 仍有 `.cpu().tolist()`，
  因此这两个后端**仍要求 `--enforce-eager`**。第三轮新增的
  `cube_310_routed` 才允许 bounded decode graph，不改变旧后端限制。
- 修复旧/复制的 vendor `scripts` 目录只读时，升级清理失败的问题。

## 实测 projection

使用真实 layer-0/expert-0 gate/up/down 权重，输入为 seed 1024 的随机
FP16 tensor。每个 shape 预热三次，30 次调用 × 三组计时，表中为中位数。
计时包含 host enqueue 与最终 synchronize；三个 projection 串行测量。

| M=2 | eager dequant + linear | Cube | 单算子 graph replay |
| --- | --- | --- | --- |
| gate | 4.020 ms | 0.320 ms | 0.330 ms |
| up | 4.003 ms | 0.319 ms | 0.329 ms |
| down | 4.007 ms | 0.291 ms | 0.300 ms |

原始数据：[`cube-r1/kernel-benchmark-r1.jsonl`](cube-r1/kernel-benchmark-r1.jsonl)。
M=1/2/5/64 均通过独立 CPU reference。单算子 replay 没有额外提速；不能
据此声称整模型 graph/MTP 已完成。每次调用临时 workspace 为 N×K×2
bytes 加 CANN reserve，M=2 的 peak allocation delta 约 5.13 MiB，
并非永久展开的专家 bank。

保留的迭代记录：

- `kernel-probe-benchmark-r1.jsonl`：首版 N=128 tile，约 0.67–0.80 ms。
- `kernel-probe-benchmark-vector-r3.jsonl`：向量构建 layout index。
- `kernel-probe-benchmark-n32-r1.jsonl`：N=32 tile。
- `kernel-probe-benchmark-nz-r1.jsonl`：直接 NZ gather。
- `kernel-benchmark-r1.jsonl`：正常 `_C_ascend` binding，无测试 alias。

这些在 `cube-r1/` 下。中间 `vector-r1/r2` 结果被排除：前者命中旧编译
cache，后者编译报错却被打包工具用旧 `.o` 伪装成成功。后续强制移走旧
W4 `.o`、检查错误日志并记录 binary SHA256，才接受结果。

## 第二轮：无 Gather 的 NZ packed layout

新增显式 `cube_310_tiled`，仅改变加载后的内存编码，不改 checkpoint：

- 将 signed nibble `q` 编码成 `q+8`，offset 同时加 8；差值严格不变。
- 每个 byte 的 low/high nibble 分别保存输出通道 `n` 和 `n+16`。
  物理 shape 为 `[N/32,K/128,8,16,16]`，metadata 为 `[N/32,K/128,32]`。
  对外参数 shape/dtype/bytes 不变，不保留 FP16/W8 全专家 shadow。
- 用精确 FP16 算术拆 byte：`high=RINT(byte/16-15/32)`、
  `low=byte-16*high`。任意 byte 都不会落在舍入 tie 上，310P 支持该 RINT。
- low/high plane 已是 Cube NZ；去掉运行时 weight Gather/transpose。
- 使用 zero block/repeat stride 的向量 Sub/Mul 广播 16-channel metadata，
  不生成大份 coefficient tensor，也不使用 metadata Gather。
- gate/up 每批解包 K=512；down 一批处理 K=640。其它已验证尺寸用完整 K
  或可整除的 512/256/128 batch，避免越界读取 K tail。

真实 layer-0/expert-0，M=2，仍是 30 次调用 × 三组：

| projection | 首轮 Cube | tiled arithmetic | NZ + vector broadcast |
| --- | --- | --- | --- |
| gate | 0.320 ms | 0.268 ms | 0.079 ms |
| up | 0.319 ms | 0.267 ms | 0.079 ms |
| down | 0.291 ms | 0.247 ms | 0.081 ms |

原始记录：`cube-r1/kernel-benchmark-tiled-broadcast-r1.jsonl`。
本轮 **51 个 NPU tests 全部通过**：两种 layout、每个 byte/offset、
改变输入/权重的 graph replay、非零 expert storage offset、非当前 device，
以及 K=256/384/512/768/896/1536/2304 的 batch 选择。
kernel SHA256：`56a6513bed73e51c56b629395fca54f096c32d97ae3a9651e2cf7c8938fe8c0a`。
新增 bool `tiled=False` 的正常 binding SHA256：
`2d153f7e9982c6912d692f9d5eb35df3ad6d728d2521b0694b5f3af72751e743`。

没有采用以下实验：byte LUT 正确但变慢；NZ + metadata Gather 正确但
变慢至约 0.62/0.62/0.53 ms；宽 K batch 与 physical-core workspace reuse
组合出现数值错误（`wide-r1`），未进入整模型测试。最终实现保留每个
logical N tile 的独立 workspace，使用无 Gather 的 broadcast 路径。
上述失败/回退不是生产 W8 回退，W8 runtime 始终未改。

第二轮整模型 TP4、真实权重、`cube_310_tiled`、eager、无 MTP 的三个
请求已完整正确结束（均有 stop、DONE 与独立答案检查）：

| 问题 | completion tokens | 总耗时 | TTFT | decode tok/s |
| --- | --- | --- | --- | --- |
| 17 × 19 → 323 | 4 | 2.604 s | 1.897 s | 4.247 |
| 1–50 完整计数 | 190 | 45.158 s | 0.999 s | 4.280 |
| Python 输出 → [0, 4, 16] | 11 | 3.540 s | 1.182 s | 4.241 |

计数请求从 3.013 提升到 4.280 tok/s（约 +42%），但仍未达到 W8。
三个短 smoke 不是完整代表性吞吐/量化质量评测，也未启用整模型 graphs/MTP。
原始证据：`cube-r1/http-tiled-smoke-r1.jsonl`，使用修正后的统一 decode 公式。
rank 权重仍为 18.6917 GiB；load-time encoding 使加载耗时增至约 233 秒，
代价发生在启动而非每次 decode。独立服务保留在 loopback `:8002`，
96°C watchdog 继续监督。生产 W8 `:8001` 在本维护窗口中仍停止，文件未改。

## 第三轮：设备端 routing 与 MTP/graph gate

新增独立 `QwenW4RoutedMatmulV310` 与显式 `cube_310_routed` 后端。
输入为静态 route slots、设备上的 INT32 local expert IDs，以及仍压缩的
`[E,N,K/2]` NZ-packed bank。每次 replay 读取当次 ID；peer-owned routes
写精确零值，不能保留上次 local expert 输出。gate/up/down 三次调用替代
Python 逐专家 dispatch，weighted reduction 留在设备上，TP all-reduce
与 shared expert 的语义不变。

初版限定最多 80 routes（本模型 8 tokens × top-k 10），仅供 bounded
decode。大 prefill 保留原 grouped host routing；超过限制的 graph capture
直接报错，不能静默捕获 `.cpu().tolist()`。临时 workspace 上界为
`routes*N*K*2 + CANN reserve`，不是常驻专家展开。

本轮 **78 个 NPU tests 通过**：原 51 项、routed projection、local/peer 双向
切换与全 peer 的 replay、错误输入、meta 和完整 MoE-layer replay。
第一次 layer capture 因测试未设置 worker 使用的 `jit_compile=False` 而
命中不可捕获的 aclop MatMul；对齐真实 worker 设置后全套通过，没有修改
生产环境设置。CPU suite 与 build 回归合计 **33 passed**。

第一次真实模型启动成功加载 target + MTP（19.3821 GiB/rank），但在
graph capture 阶段失败：生产 post-load hook 已把 shared-expert weights
转成 NZ，W4 原路径每次 `.to(float32)` 命中不支持捕获的 aclop Cast。
因此未达到 ready，也没有把该次当作推理成功。修正仅限 routed backend：
NPU shared projections 使用与 W8 一致的 resident FP16 operand policy；
CPU/reference 的 FP32 路径不变。增加实际 post-load NZ hook 的 regression，
扩展到 **82 个 NPU tests 全部通过**，包括完整 layer 的 changing-input replay。

真实 layer-0、TP4 rank-0 partial、随机激活（不含 collective）中，M=1
有两个本地专家：host route 1.258 ms，device route 0.768 ms，graph replay
0.747 ms。M=2 为 0.804/0.705/0.650 ms；M=5 为 6.556/3.739/3.713 ms。
结果已对照同一权重的 host-routed layer；原始数据
`cube-r1/layer-routed-r1.jsonl`。不能把这组 partial latency 换算为全模型 tok/s。

修正 shared projection 并应用真实 NZ post-load hook 后，第二次 real-layer
检查仍通过：M=1 的 host/device/replay 为 1.219/0.664/0.638 ms，M=2 为
0.664/0.599/0.573 ms。原始数据 `cube-r1/layer-routed-r2.jsonl`；该比较的
两条 route 路径使用同一 FP16 shared policy，不等同于旧 FP32 shared baseline。

本轮 native binding SHA256：
`1baed5288873315a337f25383833b7ea6c351ba8c6ca51e5c8d5b07210b85004`。
group kernel（共享 helper geometry 重构后）SHA256：
`f67b7437d7839cf5c1c6ec13b1ccb4ddb271bc7c73f1ce3a468b02c1c3496906`。
routed kernel SHA256：
`c6088d51622d30c213be2a53c4a05ec7110b7464432b0a02ba36cb80bb967923`。

第二次真实 W4 启动完成：TP4/EP、MTP k=1、FULL_DECODE_ONLY、capture
sizes `[1,2]`，32,768 context / 单请求 / 512-token prefill chunk。
权重为 19.3821 GiB/rank；graph capture 日志报告 0.30 GiB。
不使用 `--enforce-eager`，编译 mode=0 不等于关闭 ACL graph。

三个请求均完整正确且有 stop/DONE：

| 问题 | completion tokens | 总耗时 | TTFT | decode tok/s |
| --- | --- | --- | --- | --- |
| 17 × 19 → 323 | 4 | 1.425 s | 1.073 s | 8.536 |
| 1–50 完整计数 | 190 | 17.307 s | 1.003 s | 11.592 |
| Python 输出 → [0, 4, 16] | 11 | 2.050 s | 1.192 s | 11.653 |

原始记录 `cube-r1/http-routed-mtp-smoke-r2.jsonl`。服务日志的 graph
统计为两 token verification 的 `Runtime Mode = FULL`，不是只有配置或
capture 成功；MTP 的 accepted/drafted 计数均递增。更换 expert IDs、
local→peer route 清零以及 NZ shared weights 的 replay 数值回归也通过。
服务保留在 loopback `:8002`，W8 文件和 launcher 未改。

### 同协议短上下文 sustained 对照

同机、相同三条 coding/diagnostic prompts、temperature=0、seed=42、
thinking=false、ignore_eos=true，每题固定输出 512 tokens：

| prompt | W4 decode tok/s | W4 draft 接受率 | W8 decode tok/s | W8 draft 接受率 |
| --- | --- | --- | --- | --- |
| LRU cache + tests | 11.210 | 92.83% | 20.282 | 93.94% |
| producer/consumer queue | 10.367 | 78.40% | 18.515 | 78.40% |
| inference diagnosis | 10.722 | 83.81% | 19.073 | 83.15% |

W4 中位数 **10.722 tok/s**，低于 W8 **19.073**。W4 TTFT 分别为
0.981/1.224/1.273 秒。原始请求、返回文本、usage 与 metrics delta 在
`cube-r1/w4-routed-mtp-short-r2.jsonl`；汇总在
`cube-r1/routed-mtp-results-r2.json`。两种 quant 可能生成不同文本；这是
同 workload 的速度比较，不是相同 token 轨迹或质量等价证明。

W4 使用完整结束时间重算 client decode；W8 表格沿用保存的 server timing
（gaps/elapsed_ms），原始 client timing 也可复核。配置差异明确保留：
W4 是 32k / prefill512 / memory.90 / KV.65，W8 是 160k / prefill2048 /
memory.965 / KV.88，不能声称所有 runtime 参数完全相同。

尤其 queue 题两边都是 drafted=287、accepted=225，W4 decode 49.291 秒，
W8 27.599 秒。接受率并没有解释这项差距；每次 verification + draft 的
执行成本仍明显偏高。下一步应采集全模型 replay trace，分离 W4 projections、
attention/GDN、MTP 与 collective，再决定是否融合 gate/up 或复用相同 expert
的解包结果。以上是待验证优化方向，不是已经测得的具体瓶颈占比。

### 约 23.4k 上下文 sustained 对照

相同 long-prefix（SHA256 在 JSON 汇总中）、同三题 / 512 tokens：

| prompt | prompt / cached tokens | W4 decode tok/s | draft 接受率 | 热 TTFT |
| --- | --- | --- | --- | --- |
| LRU cache + tests | 23407 / 23168 | 10.923 | 92.48% | 4.565 s |
| producer/consumer queue | 23418 / 23168 | 10.660 | 88.24% | 4.657 s |
| inference diagnosis | 23424 / 23168 | 10.198 | 79.93% | 5.119 s |

W4 中位数 **10.660 tok/s**，W8 是 **18.091 tok/s**；短/长六个请求
均达到 512 tokens 正常 length 结束，服务无 ERROR/Traceback。
原始数据 `cube-r1/w4-routed-mtp-long-r2.jsonl`。先行 32-token warmup 的
冷 TTFT 为 **368.799 秒**，不能混入 decode 吞吐或隐去该 prefill 缺陷。
本次测试最高 80°C，96°C watchdog 未触发。W4 服务仍保留，速度目标未完成。

## 整模型第一轮 Cube 结果

TP4、真实 W4 checkpoint、`cube_310`、eager、无 MTP，三个 HTTP smoke 均
完整结束且答案正确。不是仅有 startup/HTTP 200：

| 问题 | completion tokens | 总耗时 | TTFT | decode tok/s |
| --- | --- | --- | --- | --- |
| 17 × 19 → 323 | 4 | 3.815 s | 2.831 s | 3.048 |
| 1–50 完整计数 | 190 | 64.609 s | 1.884 s | 3.013 |
| Python 输出 → [0, 4, 16] | 11 | 5.670 s | 2.311 s | 2.977 |

decode 统一用 `(completion_tokens - 1)/(总耗时 - TTFT)`。保存的
[`http-smoke-raw.jsonl`](cube-r1/http-smoke-raw.jsonl) 来自旧 smoke client，
其中原 `decode_tokens_per_second` 用最后可见 delta 时间而非完整结束时间，
偏乐观；此表从原始时间重算，不覆盖原始证据。

rank 权重占用仍为 **18.6917 GiB**；profile peak activation 0.39 GiB，
graph 0 GiB，warm torch allocated/reserved 为 23.08/23.14 GiB。
本轮 8k/单请求不是容量上限。完整服务日志保留在 host
`/srv/ai/src/qwen38-w4-hardware-20260927/server-cube-r1.log`。

后续 layer-0 真实权重、随机激活、TP4 rank-0 partial（没有 all-reduce）
profile 显示：M=1、两个本地专家，未 profile 的中位耗时 2.409 ms。
三次采样共 222 个 device tasks，W4 projection 的 18 次调用占
5524.895/6093.725 μs，即 90.67%。因此下一优化先解决解包/DMA，
再处理 host routing；不能把该 partial latency 外推成整模型吞吐。
原始 layer 计时为 [`layer-profile-r1.jsonl`](cube-r1/layer-profile-r1.jsonl)。

## 硬件与可复现性

四个逻辑 Ascend 310P3，Python 3.12.13，torch 2.13.0+cpu，
torch_npu 2.13.0.rc1，CANN 9.1.0；未升级系统依赖。

- checkpoint：`/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i`
- source：`/srv/ai/src/qwen38-w4-ce1862e52`
- 独立 Python：`/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python`
- evidence：`/srv/ai/src/qwen38-w4-hardware-20260927`
- 首轮 Cube binding SHA256：`54a80e20456ba3d86c2d92b46b9eb1b5ad29ff9e3b2241397371c2d44cf4a4c8`
- 首轮 Cube kernel SHA256：`8cef7bd46643226c71683afe83f86eed563ad36f8a19221ac1154fac55ee62c2`

本次使用正常新 Torch binding + 独立 W4 operator vendor package，保留
已验证的 attention/GDN package。完整 CANN 包首轮编译完成，但安装旧复制
目录时遇到只读权限；第二轮重建被主动终止，避免再次等待无关 KDA 编译。
没有修改生产 W8 model、venv、runtime 或 home launcher。

独立 W4 vendor：
`/srv/ai/src/qwen38-w4-operator-build-r1/ops/vendors/qwen_w4_probe_transformer`。
该名字表示开发期独立安装包；最终 native 测试不使用 Python alias。
标准源码 build manifest 也已包含新算子，干净部署可用完整构建流程。

## 验证状态

| 项目 | 状态 / 证据 |
| --- | --- |
| CPU packing/export/backend/build guards | 合计 33 passed（含 lossless tiled layout） |
| 首轮新算子 NPU correctness/replay | 正常 binding 25 passed（含非零 expert storage offset） |
| NZ tiled / canonical 扩展回归 | 正常 binding 51 passed；全部 byte、offset 与 K-batch 边界 |
| byte/zero-point exhaustive test | 256 个 byte、16 个 offset、basis inputs 精确一致 |
| 多设备与错误输入 | device guard、shape/dtype/contiguity 拒绝测试通过 |
| 真实权重 projection | M=1/2/5/64 对照 CPU 通过 |
| 真实整模型 TP4/EP | 三个请求完整正确，约 3 tok/s；尚未达到 W8 |
| NZ tiled 整模型 TP4/EP | 三个请求完整正确，4.24–4.28 tok/s；仍为 eager，无 MTP |
| dummy 整模型 | 未运行；不能替代真实 checkpoint 测试 |
| Routed/NZ/operator/layer NPU 回归 | 82 passed；包含动态 expert ID、真实 NZ post-load 与完整 MoE replay |
| 整模型 ACLGraph | routed 后端的 FULL_DECODE_ONLY 通过；verification batch=2 的运行统计为 FULL |
| MTP | k=1 已通过三个正确真实权重请求；不是只验证启动 |
| FlashComm1 | 未启用；保持原 TP/EP collective 路径 |
| Multimodal | 未测，本轮 language-model-only |
| 128k + batch 16 | 未测；最新 32k/单请求隔离速度验证不是容量上限 |

没有完整评测集分数，因此不编造 accuracy YAML 的指标值，
也不声称量化质量等价。无 GitHub issue 发帖或 push。

本轮 CPU/build 合计 33 tests 通过。已在独立 disposable worktree
执行 `bash format.sh ci`；全仓存在原有 Ruff/拼写/Markdown/forbidden-import
失败，未改动无关文件。变更文件的 scoped hooks 通过，唯一失败是全文件扫描
发现原有 `torch_binding_meta.cpp:655` 的 `empty_symint({`；新增 meta 使用
显式 `c10::SymDimVector`，NPU meta 测试通过。原始 lint 日志保留在
`/tmp/qwen-w4-format-ci-routed-r1.log` 和 `/tmp/qwen-w4-format-scoped-routed-r3.log`。
原始 benchmark 文本中的 HTTP 工具 `wrk` 加入精确拼写白名单；没有篡改输出。

## 紧凑运行手册

本任务使用真实 host 路径和 loopback `:8002`，不使用 skill 默认容器路径。
仅在服务器空闲的维护窗口运行，并保留 96°C watchdog。

```bash
source /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh
export ASCEND_CUSTOM_OPP_PATH=/srv/ai/src/qwen38-w4-operator-build-r1/ops/vendors/qwen_w4_probe_transformer:${ASCEND_CUSTOM_OPP_PATH:-}
export VLLM_ASCEND_KV_CACHE_FRACTION=0.65
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_ASCEND_LOG_REQUEST_TIMINGS=1
/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python -m vllm.entrypoints.cli.main serve \
  /srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --served-model-name qwen38-w4-experimental --host 127.0.0.1 --port 8002 \
  --dtype float16 --tensor-parallel-size 4 \
  --no-async-scheduling --disable-custom-all-reduce \
  --max-model-len 32768 --max-num-batched-tokens 512 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 --language-model-only \
  --enable-expert-parallel --enable-ep-weight-filter \
  --reasoning-parser qwen3 --mamba-cache-mode align --enable-chunked-prefill \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --cudagraph-metrics --enable-logging-iteration-details \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}' \
  --hf-overrides '{"text_config":{"ascend_expert_quantization":{"backend":"cube_310_routed","bits":4,"format":"qwen4exp_w4a16_group_v1","group_size":128,"offset_dtype":"int8","packing":"signed_int4_low_nibble_first_in_axis","scale_dtype":"float16","symmetric":false}}}' \
  --limit-mm-per-prompt '{"image":0,"video":0}'
```

这是最新通过的 MTP/graph 路径，需要两种 W4 operator 与匹配 binding。
checkpoint 默认仍为 reference，不会静默改变原 W8 启动方式。
如需 eager 隔离：切换为 `cube_310_tiled` 或去掉 `--hf-overrides`，
同时加 `--enforce-eager` 并移除 speculative/compilation-config 参数。
不能把 `TORCHDYNAMO_DISABLE=1` 当作 graph 正确性验证。

```bash
curl -sS http://127.0.0.1:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-w4-experimental","messages":[{"role":"user","content":"What is 17 * 19? Reply with just the integer."}],"temperature":0,"seed":1024,"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}}'
```

应完整返回 `323` 且 `finish_reason=stop`；完整 smoke 还要求 1–50、Python
输出题正确，不能只检查 HTTP 200。
