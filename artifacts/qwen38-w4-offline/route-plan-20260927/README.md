# W4/W8 decode 路由复用与直接输出（2026-09-27）

## 范围

针对实际小批量 W4 routed decode 内核，不是上一轮的 grouped 排序路径。
保留 48 层、W4A16 G128、TP4/EP、MTP2、FULL_DECODE_ONLY、端口 8001，
max-model-len 262144、max-num-seqs 2、内存比例 0.94、KV fraction 0.80。
不改 checkpoint、量化精度或已有 home 启动脚本。
用户随后授权同时推进 W8、独占 NPU，最终无需保留运行服务。
W8 原服务单独复制作 baseline；候选为最新隔离 runtime，MTP1/2 分别测量。
W8 保留 160000 context、max-num-seqs 1、prefill 2048、内存比例 0.965。

基于 `42c5e1f03`，硬件工作树仍包含用户此前暂存的 byte-mask unpack 补丁；
它不是本轮提交的一部分。已有暂存内容与日志必须保留。

## 修改与分阶段实验

- r1：每任务只读取每个 expert ID 一次，构造保持首 owner/行顺序的有界计划；
  复用计划进行 gather，批量暂存输出以减少逐行往返同步。
- r2：把最终 FP16 输出直接写入计划指定的行，删除临时输出的 GM 写入、重读、scatter。
- r3：尝试每个物理 AI core 一个持久任务，直接在一次 Process 内遍历多个 N tile。
  数值 gate 失败（63 failed / 135 passed），没有进入服务。
- r4：保留跨 N tile 的计划/输入 gather 复用，但每 tile 独立 InitGeometry/Process，
  显式排空流水线。198 个 NPU tests 全部通过；真实 layer 输出 hash 与 baseline 一致。
  不跨核共享可变状态，不增加全局同步或主机读回。
- W8：fused routing 从两 token 扩到八 token，覆盖三 token MTP verification；
  去掉两个 grouped GEMM 之间多余的 peer 清零；先 FP16 inverse-gather、再转 FP32。
  仍在最终 reduction 前清零 peer，保留 FP16 route-weight rounding、FP32 weighting/sum。
- W8：启用已用于 device-routed W4 的共享 Q/K/index-query RoPE tables 与八-token QSA dispatch。
  没有缓存跨 replay 的 position 值，也不复用 index-key 的不同位置表。
  同时带入此前共用 router 恒等乘法删除、INT32 中转索引转换；不把 W4 解包应用于 W8。

路由计划最多 80 行。peer ID（包括 INT32 两端）被排除；每条有效 route 恰好归属一个 group。
所有输出行先清零，以避免 graph replay 从 local 变为 peer 后留下旧结果。
直接输出保留原有 FP32 累加到 FP16 的转换，只改变 store 地址。
旧 workspace 预算暂时保留，本轮不宣称降低显存或增加容量。

## 验证方法

- CPU：用 GM-read stub 编译执行内核中的同一段 route planner，验证 1–80 行、
  3200 组输入、顺序、边界和每 ID 一次读取；另运行 W4 MoE/build 回归测试。
- NPU：独立 CPU 数学参考、单/多 expert、极端 peer ID、16 行 gather 边界、
  变化输入/权重/归属的图 replay、完整 MoE、非整除 N tile 分配。
- 真实权重 layer：layer 0、同一合成输入、TP rank 0 partial，无 collective；
  比较 T=1/3/6/8 的输出 hash 和 graph latency。
- 全模型：相同 512-token coding 请求，单独记录 TTFT、decode tok/s、MTP acceptance；
  算术、1–50、Python 输出和两并发请求作为 smoke/graph gate。

仅 layer hash 一致不能证明全模型逐位确定性。profiler 计数可重叠，不能相加成端到端耗时。
本轮不重跑双 262K 满窗口容量测试，也不启用 flashcomm1 或多模态。

## 实测结果

W4 真实 layer-0、TP rank0 partial（含该层 shared expert，无 collective），graph replay：

| Tokens | baseline ms | r4 ms | 耗时下降 |
| --- | ---: | ---: | ---: |
| 1 | 0.46535 | 0.43384 | 6.8% |
| 3 | 0.70625 | 0.65910 | 6.7% |
| 6 | 2.77031 | 2.56324 | 7.5% |
| 8 | 3.65308 | 3.11467 | 14.7% |

W8 真实 layer-0 routed expert bank（不含 shared expert/collective），graph replay：

| Tokens | baseline ms | candidate ms | 耗时下降 |
| --- | ---: | ---: | ---: |
| 1 | 0.16188 | 0.15245 | 5.8% |
| 2 | 0.16443 | 0.15801 | 3.9% |
| 3 | 0.71022 | 0.43535 | 38.7% |
| 6 | 0.75969 | 0.47477 | 37.5% |
| 8 | 1.04986 | 0.75242 | 28.3% |

上述各自 A/B 使用相同输入，所有 eager/graph 输出 hash 完全相同。
另测 layer 47、TP shard 3 的真实 W8 bank，五种 token 数全部逐位一致；
三-token graph 0.48414 → 0.24054ms，避免只凭 layer 0 判断。
W4/W8 两张表的组件范围和输入不同，不能直接据此计算量化间的速度比。
W8 layer A/B 已使用新版共用 dispatch helper，不包含更旧生产 helper 的直接 INT64→FP32 开销。

W4 三个短 prompt、每个 512 output tokens：median 14.268 → 15.274 tok/s。
acceptance / 输出 hash 有变化；不是等 acceptance 或全模型 bitwise-deterministic 比较。
W4 ~23.4K prompt：17.211 / 14.228 / 13.923 tok/s，median 14.228。
冷 TTFT 99.984s；缓存复用后 TTFT 1.607 / 1.705 / 2.370s。
本轮没有新跑长 prompt baseline，不宣称长上下文提速；acceptance 从 0.889 降至 0.641。
两个并发 1–50 请求全部正确，graph 日志记录六-token FULL replay；
另两个并发算术请求 713 / 1073 全部正确。不等同于两个满 262K 窗口压力测试。
W8 真实权重 MTP1 baseline / candidate 均完成 smoke、short/long 各三次和 profile。
short median 18.647 → 19.829 tok/s（+6.3%）；long 18.526 → 19.547（+5.5%）。
冷 TTFT 45.508 → 46.063s，未改善。相同一组请求仍可能有不同输出和 acceptance。
MTP2 candidate 完成同样真实服务 gate：short 24.441 / 20.820 / 21.140 tok/s，
median 21.140；long 23.587 / 21.744 / 20.869，median 21.744。
相对 baseline median，分别 +13.4% / +17.4%；相对优化 MTP1 则 +6.6% / +11.2%。
冷 long TTFT 46.422s，仍不是本轮改善项。推荐 **已测量范围内** 的 W8 MTP2 配置，
不宣称 Kilo 所有请求都达到峰值 24.44 tok/s，也不宣称 W4 已追平 W8。

| 配置 | short median tok/s | long median tok/s |
| --- | ---: | ---: |
| W4 MTP2 baseline | 14.268 | 本轮未测 |
| W4 MTP2 r4 | 15.274 | 14.228 |
| W8 MTP1 生产副本 | 18.647 | 18.526 |
| W8 MTP1 candidate | 19.829 | 19.547 |
| W8 MTP2 candidate | 21.140 | 21.744 |

W4 保留双窗口配置，W8 本轮保留单窗口配置；不是量化精度或容量的完全同条件比较。
各服务温度峰值：W4 72°C，W8 baseline 75°C、MTP1 77°C、MTP2 78°C。
没有人为提高时钟/内存上限，不把温度日志等同于固定频率实验。

W4 TP0 的三-token verification trace（每个投影 240 次）：
gate/up task median 560.352 → 530.286us；down 386.979 → 327.461us。
逻辑 block 数从 40/80 改为 8/8；只有 planner/gather/store 路径改变，
MAC counter 仍为 9.913 / 4.956us。标量/搬运/解包仍是优先调查方向，
不能把重叠硬件计数简单相加或当成纯算术耗时。

W8 TP0 MTP1 同协议六-iteration capture（含一次 prefill）：
task 数 67797 → 64870；`Cast` 240 → 120 次，合计 41.630 → 17.175ms；
`aclnnInplaceCopy_CastAiCpu_Cast` 150 → 30 次，合计 22.305 → 4.838ms。
576 次 quant grouped matmul 合计仍约 84.6ms，没有把算术精度下降当作优化。
这些是整段 capture 的计数，不是每 token 或每 decode step 的节省。
candidate 包含此前已完成的 INT32 中转转换等共同 runtime 改进，
不能把整个 bundle 的收益全归因于本轮的 fused-routing 一项。
`w8-replay-proof.json` 记录各四个 rank 的真实 `aclmdlRIExecuteAsync`：
baseline / MTP1 candidate 各 90 次，MTP2 candidate 各 110 次；
W4 四个 rank 各 110 次，另有双请求六-token FULL 日志。不是仅观察 capture/startup。

CPU focused suite：155 passed。W4 NPU：198 passed。W8 routing/NZ/QDQ/replay：32 passed。
共享 RoPE/作用域 NPU suite：17 passed，覆盖变化 position/activation 的逐位一致 replay。
最终扩大 CPU 回归（增加 W8 parity / TP sharding）：197 passed、1 skipped；
与此前 155 项重叠，不能把两个计数相加。
W8 首次 graph test 因测试未设置 `jit_compile=False` 在 aclop Cast capture 失败；
匹配实际服务的非 JIT 配置及 load-time NZ bank 后通过。
局部 pre-commit 除既有 `csrc/torch_binding_meta.cpp:655` symbolic-meta 问题外通过；
本轮不改该无关代码，不声称全库 lint 全绿。

## 运行位置

远端：`matteius@192.168.53.187`。
隔离构建目录：`/srv/ai/src/qwen38-w4-native-build.zHZtEd`。
保留每个 `ops-routeplan-r*` vendor，旧 `ops-pipeline-wide-r2` 未覆盖。
遵循用户指定的本机工作树和 8001，不使用技能默认 Docker 路径/8000。

## 结论与下一步

本轮通过有界路线复用、直接输出和 W8 三-token 快路径改善 decode，
没有修改 checkpoint、量化精度或长上下文数值策略。
W4 仍落后于 W8，优先继续降低解包/标量流水线控制成本；
更激进的跨 step 热 expert 缓存与 workspace 缩减要单独验证显存及图状态一致性。
参见同目录 `RUNBOOK.md`；吞吐原始数据、真实响应、输出 hash、温度和 profile 均保留。
最终所有 benchmark 服务停止，8001 无 listener，四个逻辑 NPU 无运行进程；
见 `npu-after.log`。home 生产启动脚本未改，没有推送到远端 Git 分支。
