# 单卡双 NPU 上下文与 QSA 主 KV 分层存储

基线测试卡为物理卡 8217 的逻辑 NPU 0、1；后续并行实验同时使用物理卡
16409 的逻辑 NPU 2、3。真实 W4 checkpoint、TP2+EP、原生 INT4 后端、
`gpu_memory_utilization=0.965`；每个 NPU 加载模型权重 38.5613 GB。
端口 8002、8003 仅作隔离测试，最终恢复 TP4 服务。

## 已测容量与吞吐

| `max_model_len` | 解码模式 | 设备 KV 池报告 | 三次 512-token 短请求中位 decode | 近上限请求中位 decode |
| ---: | --- | ---: | ---: | ---: |
| 24,576 | MTP2 | 78,787 token | 18.77 tok/s | 18.54 tok/s，prompt 23,936–23,953 token |
| 65,536 | MTP2 | 81,794 token | 未单独计时 | 17.03 tok/s，prompt 64,035–64,052 token |
| 80,000 | MTP2 | 82,260 token | 未单独计时 | 16.57 tok/s，prompt 79,236–79,253 token |
| 82,000 | MTP2 | 82,251 token | 18.66 tok/s | 18.07–18.62 tok/s，prompt 79,236 token |
| 100,000 | plain | 168,147 token | 未单独计时 | 9.67 tok/s，prompt 99,236–99,253 token |
| 160,000 | plain | 168,789 token | 未单独计时 | 8.33 tok/s，prompt 159,236–159,253 token |

24,576 轮的近上限冷 prefill 首字耗时约 101 秒，后续请求复用 prefix cache，
首字耗时约 2–3 秒。因此 decode tok/s 与冷 prefill 延迟分别报告。
短请求三个原始值为 20.71、18.52、18.77 tok/s；长请求为
19.77、18.54、17.61 tok/s。固定前缀由 `make-prefix.py` 根据 tokenizer 精确生成；
原始 HTTP 流式结果和 prompt token 计数在同目录 JSONL 中。
65,536 轮的三次近上限值为 18.98、17.03、16.57 tok/s；64,035-token 冷 prefill
首字耗时 286 秒，三次复用前缀的首字耗时 1.32–1.46 秒。
80,000 轮的三次近上限值为 18.55、16.57、16.21 tok/s；79,236-token 冷
prefill 首字耗时 359.71 秒，三次复用前缀的首字耗时 1.33、3.30、2.27 秒。
这轮 MTP2 接受率依次为 0.850、0.721、0.688。固定 228 题 zero-shot MMLU
抽样得到 204/228（89.47%），无无效答案；该数字不是完整官方 MMLU 分数。
原始记录为 `bench-tp2-80000-near-limit.jsonl`、
`tp2-card1-80000-mtp2-mmlu.jsonl` 及其 `summary.json`。

继续把 MTP2 声明上限推到 82,000 也成功：KV 池只有 82,251 token，最大并发
显示为 1.00×。79,236-token 冷 prefill 首字耗时 360.51 秒，随后 512-token
decode 为 18.07 tok/s，MTP 接受率 0.857；紧接的一次相同输出为 18.62 tok/s。
但第一条完整输出占用剩余空间后，下一条请求无法保留整段 79K prefix，重新进入
冷 prefill。故 82K 是已验证的容量边缘，不是适合频繁复用长前缀的稳健服务点；
80K 留有更实用的缓存余量。该 82K 服务的三次短 512-token 请求为 20.35、
17.50、18.66 tok/s，中位 18.66 tok/s；三次 1,024-token 请求为 20.81、
18.44、17.89 tok/s，中位 18.44 tok/s。固定 228 题为 205/228、零无效答案。

MTP3 在相同 80,000 配置下能完整加载 38.5613 GB/NPU 权重，但 KV 规划需要
6.57 GiB/NPU，实际只有 5.67 GiB/NPU，启动器给出的最大长度估计为 68,736。
因此后续 MTP3 对照改为 `max_model_len=68000`，不能把 MTP2 的 80K 容量直接
外推到 MTP3。

68K MTP3 的 67,236-token 冷 prefill 首字耗时 306.53 秒。三次 512-token
decode 为 18.04、15.16、11.10 tok/s，中位 15.16 tok/s；接受率分别为
0.772、0.610、0.358。第三次请求因容量已接近上限而无法复用固定前缀，重新执行
冷 prefill（首字 307.75 秒）。它比 65K MTP2 的 17.03 tok/s 中位更慢，且
缓存行为更脆弱，因此不应为单卡长上下文选择 MTP3。

减少到 MTP1 也没有增加容量：90K 启动只有 5.81 GiB/NPU 可用于 KV，而需求为
7.14 GiB/NPU，规划器估计最大长度 73,088，低于已验证的 80K MTP2。不同声明
长度会改变 profiling 临时内存，不能按 speculative token 数量线性推算 KV 空间。

关闭 speculative model 的 plain 模式成功启动 100K，报告 12.10 GiB/NPU 可用
KV、168,147-token KV 池和 1.68× 最大并发。99,236-token 冷 prefill 首字耗时
445.92 秒；三次缓存后 512-token decode 为 9.66、9.67、9.68 tok/s，中位
9.67 tok/s。它把已验证容量从 MTP2 的 80K 提到至少 100K，但 decode 只有
MTP2 80K 中位 16.57 tok/s 的约 58%。

相同 plain 模式进一步成功启动 `max_model_len=160000`，报告 12.10 GiB/NPU、
168,789-token KV 池和 1.05× 最大并发。真实请求为 159,236 prompt token 加
512 output token。三次完整 prefix-cache 命中后的 decode 为 8.328、8.328、
8.343 tok/s，中位 **8.328 tok/s**；首字耗时为 0.60、5.34、3.32 秒。
最初的通用客户端在长冷 prefill 期间达到 socket timeout 并断开，但服务保持健康，
已完成的 130,048 prefix token 被缓存；专用 1,800 秒客户端随后计算剩余 29,184
token，首字耗时 163.06 秒，并以 8.234 tok/s 完成 512-token decode。因此这轮
验证了完整 159.7K 输入输出容量与稳定 decode，没有记录一条连续客户端连接下的
完整冷 TTFT。原始记录为 `bench-tp2-card2-160000-plain-u965-one-shot.json`、
`bench-tp2-card2-160000-plain-u965-cached.jsonl` 和对应 server log。

另一个严格等价候选在上下文不超过 QSA 的 2,048-token 选择预算时跳过不会被
消费的 index-query 投影、RMSNorm 和 RoPE。24,576 配置的三次短请求为
20.69、17.88、18.97 tok/s，中位 18.97 tok/s；同配置基线中位 18.77 tok/s，
约提升 1.1%，在这组三次测量的波动范围内。第一组输出 SHA 与基线一致，后两组
因 MTP 接受路径变化而不同；不能把这个候选称为明显吞吐提升。

另一个严格等价候选在一次 backbone forward 内让 12 个 QSA 层共享 query 与
index-key RoPE 表，把每次目标 forward 的 24 次表计算降为 2 次。24,576 配置的
短请求为 21.01、18.70、19.27 tok/s，中位 19.27 tok/s，比 18.77 基线高 2.7%。
23.9K 请求为 20.08、18.16、17.68 tok/s，中位 18.16 tok/s，比 18.54 基线低
2.1%；这组三次的 MTP 接受率中位也从 0.753 降到 0.696。两种上下文合看属于小幅
波动而非稳定的大提升，但没有明显回退，故保留等价计算复用并继续走准确率门禁。
与既有 pass3 完全相同的数据集哈希 `de4f0a…` 上，RoPE 复用得到 205/228，
无无效答案；既有 pass3 各候选为 203–206/228，单卡 80K MTP2 为 204/228。
另一个误用 `dd128…` held-out 文件的额外结果为 206/228，单独保留并明确标名，
不参与上述 A/B。

两个等价候选合并后，短请求为 21.28、17.93、18.97 tok/s，中位 18.97 tok/s，
第二组三次为 21.07、18.20、16.83 tok/s；六次合并中位为 18.59 tok/s，
相对三次基线中位 18.77 tok/s 基本持平且略低。因此保留它们的依据是消除重复/
无用计算且没有明显回退，而不是把前三次波动称为新的吞吐档位。相同 `de4f0a…`
准确率门禁仍为 205/228、零无效答案，与单独 RoPE 复用完全相同。另一次 23.9K
上下文复测为 20.00、18.67、17.63 tok/s，中位 18.67 tok/s；MTP 接受率依次为
0.813、0.725、0.664。它与既有 23.9K 基线中位 18.54 tok/s 一致。

同一个已捕获图随后跨越 QSA 精确 dense/sparse 分支边界重放：实际 prompt 1,936
token（小于 2,048）时输出 128 token、20.70 tok/s、MTP 接受率 0.851；实际
prompt 2,085 token（大于 2,048）时输出 128 token、20.59 tok/s、接受率 0.823。
两次变化输入均成功，证明分支判断和共享 RoPE 表没有把首次捕获的位置或形状固化。
原始记录为 `bench-tp2-card1-qsa-combined-boundary-1900.jsonl`、
`bench-tp2-card1-qsa-combined-boundary-2050.jsonl` 和
`bench-tp2-card1-qsa-combined-long-repeat3.jsonl`。

数学复查还加入了静态 Gemma norm-affine 缓存，每个目标 forward 删除 97 对重复
FP32 cast/add，单层 MTP 的每个 draft step 再删除 5 对；额外内存约 3.96 MiB/NPU。
独立单卡六次短请求为 21.14、18.10、17.40、20.54、18.30、18.24 tok/s，
合并中位 18.27 tok/s，比同配置三次基线中位 18.77 低约 2.6%，但处在请求间的
MTP 接受率波动内。相同 `de4f0a…` 准确率门禁为 205/228、零无效答案。它严格
保持原始 `1 + FP16 weight -> FP32` 运算结果，且没有明显回退，故按保留小幅等价
简化的策略进入最终 TP4 候选。

`max_model_len=262144`、利用率 0.94 的启动曾失败：规划器要求每 NPU
20.89 GiB KV 池，但当时只剩 1.23 GiB。这个失败是**声明的最大请求**
和当前内存规划布局的预检查，不代表一般会话已经使用 262K token。
0.98 和 0.99 的启动在加载前因空闲内存低于利用率目标失败；后续使用用户确认的
0.965。不同 `max_model_len` 会影响启动时预留，故不能用 0.94 的 13,952-token
估计代替 0.965 的真实启动结果。

## 权重是否还能缩小

路由专家并非 FP32：每 NPU 的 256 个专家 × 48 层、gate/up/down 三个投影
使用每字节两个 INT4 code，合计约 28.125 GiB/NPU。原生算子的每组 scale、
offset 与预计算 correction sum 各约 0.879 GiB/NPU，均为 FP16。
Checkpoint 的 offset 为 INT8，但当前原生装载时转换为 FP16；改回 INT8
最多节省约 0.439 GiB/NPU，并需改算子。取消预计算 sum 最多再节省约
0.879 GiB/NPU，但会把计算放回每步解码热路径。这些局部节省不足以解释
当前 262K 规划器相对设备 KV 预算的十几 GiB 差额；优先处理 KV 布局。
其余注意力、GDN、共享专家等权重主要为 FP16，量化它们需要单独的精度
许可、核实现和真实模型验证。

## 长上下文可行路径

本模型 48 层中有 12 层 QSA。TP2 时每个 NPU 持有一个 256 维 KV head。
262,144 token 的主 QSA K/V 裸数据量为
`12 × 262144 × 1 × 256 × 2(K,V) × 2(FP16) = 3 GiB/NPU`。
压缩索引的完整历史约 192 MiB/NPU（12 层、4:1 压缩、128 维 FP16），
且索引器每步最多选 512 个四 token 组。当前约 20.89 GiB 的申请量远高于
主 QSA 裸数据。vLLM 把 12 层 QSA 放在一个组，把 36 层 GDN 按 12 层
分成三个组；四组共享一个块池，每个物理块按**最大的组**定宽。
QSA 的完整历史因此以 GDN 组的较大块步长计费，虽然 GDN `align` 模式
只需要少量活动状态。还需记录每组实际页大小和块数，才能把 20.89 GiB
逐项重现。

优先验证是否能把 QSA 长历史页与 GDN 的少量活动状态分开分配。
如果它们能在现有设备预算内独立容纳，262K 可能无需主 KV 的主机分页；
这需要真实权重启动和长 prompt 验证，不能仅凭裸字节数承诺。

对单个连续长会话还有一个不改精度的容量模式：关闭 prefix cache，并把 GDN
设置为 `mamba_cache_mode=none`。这样每个 GDN 组只保留当前递归状态，不再为
未来前缀复用保留对齐检查点；QSA 的完整 FP16 历史仍留在 NPU，MTP2 也保留。
代价是重复提交相同长前缀时必须重新 prefill。`serve-tp2.sh` 的
`mtp2-noprefix` 模式用于单独验证该配置，且长请求使用一次性基准，避免在没有
prefix cache 时重复执行数次 260K 冷 prefill。

真实 262,144 启动否定了这条捷径：关闭 prefix cache 后规划器仍要求
20.86 GiB/NPU，几乎等于 `align` 轮的 20.89 GiB。声明 262K 后的 profiling
只留下 2.92 GiB/NPU，启动器估计最大长度为 35,712。原因是共享块池仍按最宽
QSA 组定 stride，GDN 的逻辑块表即使只保留当前状态也没有变成独立的小物理池。
所以仅切换 Mamba cache mode 不足以获得 262K；需要真正拆分物理池或分页主 QSA K/V。

主机分页是另一条精度不变的路径：压缩索引与当前写入页留在 NPU，旧的
FP16 主 K/V 放在主机 DRAM。选择结果仍在 NPU 生成，再把被选中的四 token 组
搬入约 2 MiB/层的设备暂存区。12 层每输出 token 最多约 24 MiB/NPU 的 H2D
数据（不计额外索引/对齐流量）；20 tok/s 约为 480 MiB/s/NPU。
`bench-qsa-host-paging.py` 测量设备选择索引 D2H 同步、主机随机收集和选中
K/V H2D 的串行成本。它只是传输可行性探针，尚未实现分页注意力。
在空闲的第二张物理卡上，30 次测量的单层中位为选择结果 D2H 同步
0.075 ms、主机随机收集 0.101 ms、2 MiB H2D 0.723 ms，合计 0.902 ms；
12 层简单串行外推为每输出 token 10.82 ms。该探针与第一张卡的冷 prefill
并行，不能当作完整分页服务的 tok/s，也未覆盖 prefill 读回、图重放和
主机块管理。JSON 原始结果在 `bench-qsa-host-paging-card2.json`。

服务停止后又在同一张物理卡的两个逻辑 NPU 上各做 1,000 次复测。单层 total
中位为 0.849/0.874 ms，p90 为 0.957/0.976 ms；其中 2 MiB H2D 中位为
0.646/0.664 ms。12 层完全串行外推为 10.19/10.49 ms/token，与 30 次探针一致。
原始结果为 `bench-qsa-host-paging-card1-1000.json`。这仍只是传输探针；实际分页
后端能否把部分传输与注意力/路由重叠，将决定最终吞吐损失是否接近该串行上限。

当前 310P QSA 算子直接读取设备 `key_cache`/`value_cache` 和块表；
现有 `SimpleCPUOffloadConnector` 只缓存可复用前缀块，不能释放活动请求的旧页。
另一套 sparse-KV offload 属于不同 SFA 后端，并要求分离的 prefill/decode 配置，
不能直接用于此模型。真正的 QSA 分页实现还需管理主机块生命周期、保留设备
当前写入页、在选择后读取旧页、处理 chunked prefill 的大量选中组，并让
ACLGraph 在变化的索引/地址下安全重放。验证门槛是算子逐位对照、真实模型
长 prompt 准确率、MTP 接受率、图重放和端到端 tok/s。
