# 原生 INT4 第三轮：稀疏路由与激活复用

后续保留的 c1 权重流水线结果见
[`WEIGHT-PIPELINE-C1-RESULTS.md`](WEIGHT-PIPELINE-C1-RESULTS.md)：c1 中位
29.86 tok/s，c4 aggregate 中位 59.52 tok/s，固定质量门禁 206/228、零无效答案。

基于 `7a04af2e9`，分支 `perf/native-int4-pass3-20260927`。模型为 Qwen3.8 Flash Next 的
MoE W4 checkpoint；路由专家权重保持打包 INT4，激活按模型元数据许可进行 per-group INT8
量化，Cube 计算为 INT4×INT4→INT32，输出为 FP16。注意力、GDN 和共享专家仍为 FP16。
W8A8 路径、W4A16 路径、模型级后端选择策略及激活精度许可均未修改。

## 代码与验证

- `native_int4_schedule.h` 对稀疏专家只修正有效行，在 N 方向执行向量校正；小 M
  时把两份 INT4 激活拼成一次 Mmad，保留大 M 的双 Mmad 路径。独立的 decode/MTP
  与 prefill 调度及原有完整 K 权重重用保持不变。
- native decode 的 gate/up 输入按唯一 token 打包一次，再由设备算子按 top-k 固定倍数
  映射到输出路由。down 输入依专家而异，仍逐路由打包。host tiling、Torch shape/meta、
  kernel 地址计算均检查路由数整除输入行数，grouped prefill 不广播。
- 最后一组实验将最多 80 个路由 ID 的连续部分批量搬到片上；不足 32 字节的尾部用
  标量读取，避免越界。失效专家路由仍先清零，changing-input/IDs graph replay 仍覆盖。
- 有效行与配对候选分别通过 109/109 个 NPU 测试；激活复用与路由缓存候选分别通过
  117/117 个 NPU 测试，包括多形状精确逐位对照、动态输入和路由 ID 图重放。
  合并等价 QSA 与 norm-affine 简化后的 CPU 回归为 1101 passed、7 skipped。`provenance.json` 保存独立安装包
  的目标文件和源码哈希。没有新增环境变量，也没有改 model runner。

## 吞吐与质量

同一台 4×Ascend 310P3、同一真实 checkpoint、TP4+EP、MTP2（另列 MTP3）、
FULL_DECODE_ONLY graph、capture `[1,2,3,6]`、scheduler 上限 2（并发扫描时为 4）、512-token prefill chunk，
使用相同 CPU affinity。三次串行 512-token 请求的 decode **中位 tok/s**，warmup 不计。
长上下文计时请求复用 23168 个 prefix token；它们不是 23.4k 冷 prefill 的计时。

| 候选 | 短上下文 | ~23.4k 上下文 | 固定 228 题 MMLU 抽样 |
| --- | ---: | ---: | ---: |
| 第二轮 native | 19.28 | 18.34 | 204/228 |
| 有效行校正，MTP2 | 21.66 | 20.97 | 206/228 |
| 配对 Mmad，MTP3 | 22.08 | 20.91 | 206/228 |
| 配对 Mmad + 激活复用，MTP2 | 22.04 | 21.88 | 203/228 |
| 再加路由 ID 批量加载，MTP2 | 22.24 | 21.32 | 206/228 |
| 最终等价 QSA + norm-affine，MTP2 | **23.27** | 待完成 | TP2 门禁 205/228 |

最长额外做了三次 1024-token 短上下文请求，路由缓存候选为
25.19、23.19、21.67 tok/s，**中位 23.19 tok/s**。因此 25 tok/s 只在单次
高接受率请求出现，不能作为该配置的持续速度。第二轮至本轮路由缓存候选在
512-token 短上下文提升约 15.4%，长上下文提升约 16.2%，仍低于 25–30 目标。

这不是逐候选交错重跑：短请求接受率和输出不同，尤其首个计时请求较快；
因此不把单次峰值称为可持续吞吐。单层真实权重固定输入的图重放在
`layer-*.jsonl`，只说明算子延迟和输出 SHA 一致，**不是 tok/s**。
完整三轮 HTTP 结果在 `*-short.jsonl`、`*-long.jsonl`，逐题结果在 `*-mmlu.jsonl`。

最终 TP4 的三次 512-token 短请求为 25.88、21.78、23.27 tok/s，中位
**23.27 tok/s**；相对路由缓存候选的 22.24 提升 4.6%，相对第二轮 19.28
提升 20.7%。高接受率请求已两次超过 25 tok/s，但持续中位仍未达到 25。

### 并发吞吐

最终服务以 `--max-num-seqs 4` 重启，KV planner 分配 1,176,084 token，
因此对 262,144-token 请求报告 **4.49x** 理论容量。这表示 KV 块容量足以
容纳 4 个该上限的 sequence，不是 4 个 262k prompt 的实测速度。并发实测为
每路 25-token prompt + 1,024 个生成 token，四路同时起跑：

| 并发路数 | aggregate decode tok/s | 每路中位 tok/s | max running | max waiting |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 25.09 | 25.09 | 1 | 0 |
| 2 | 38.35 | 19.40 | 2 | 0 |
| 3（首次） | 25.60 | 8.58 | 3 | 0 |
| 3（profile 后重跑） | 39.07 | 13.41 | 3 | 0 |
| 4 | **50.07** | **12.64** | 4 | 0 |

四路的 end-to-end aggregate 为 49.83 tok/s，各路 decode 为 12.55–12.87 tok/s。
三路首次异常慢，但在完成 profiler 并以相同配置重启后，同样的
3×1024-token 请求达到 39.07 aggregate tok/s。因此不把 25.60 归因为固定
batch-shape 分支；两次数据都保留在 `route-cache-parallel3-1024*.json`。
尝试为四路扩展 graph capture shape 时，Ascend collective capture 报
`endCaptureErr 507903`；当前服务保留已验证的 `[1,2,3,6]`，四路所需的其他
shape 回退 eager，但 scheduler 实测可同时运行 4 路。

### Decode 热点实测

通过 vLLM `--profiler-config` 在同一 TP4 进程内分别采集 12 个四路和三路
decode iteration，使用 Ascend PipeUtilization 计数器。Profiler 本身会降速，以下只做
工作归因，不代替上面未 profile 的 tok/s。Rank 0 的汇总 task time 为：

| 类别 | 4 路 | 3 路 | 3 路占比 |
| --- | ---: | ---: | ---: |
| HCCL collective | 940.3 ms | **1217.0 ms** | **48.9%** |
| 其他 matmul | 334.8 ms | 324.5 ms | 13.0% |
| cast/layout/copy | 299.0 ms | 290.9 ms | 11.7% |
| native INT4 projection | 274.7 ms | 243.1 ms | 9.8% |
| attention/GDN | 86.1 ms | 75.9 ms | 3.1% |

三路与四路都使用 grouped native-INT4：模型 top-k=10，9/12 个 model row 分别展开为
90/120 个 route，均超过 80-route 分界。INT4 kernel 在 90 route 上的总时间反而更少，
因此不是三路 cliff 的主因。HCCL communication report 显示 rank 0 三路的
collective elapsed 中 95.7% 是 wait，只有 2.7% 是 transit；大多数消息仅为 92–184 KiB。
每个 iteration 约有 106 个 collective，使得小的 rank 到达差被重复放大。

三路中 rank 3 在 1272 个 collective 中有 846 次（66.5%）最后到达；到达差中位
934.9 us，p90 1.62 ms，最大 18.18 ms。四路的中位到达差接近（943 us），
但最后到达更均匀地分散在各 rank，最大值只有 5.34 ms。这证明主要热点是
all-reduce 前的 rank-arrival skew/同步，而不是实际的跨卡传输带宽。三路未 profile
重跑从 25.60 恢复到 39.07 tok/s，说明极端 cliff 是瞬时同步尾延迟，而不是固定的
90-route INT4 kernel 回归。具体上游 kernel/host launch 如何造成最慢 rank 仍需逐层打点。
`profile-hotspots.json` 保留 8 个 rank/run 的分类、projection shape 和到达差；
`analyze-hotspots.py` 可从远端保留的原始 trace 重生该报告。

MMLU 为固定零样本 228 题，不是官方全量准确率。所有已完成候选的 228 个回答
都以 `finish_reason=stop` 正常结束，最多生成 2 token，质量脚本允许 16 token，
所以三题变化不是输出上限截断。样本差异不能证明总体精度变化；选择候选时以
可重复的 tok/s、算子逐位对照和无 invalid 回答共同判断。

## 运行与范围

`build-variant.sh` 将算子编译进不可覆盖的 `ops-pass3-<label>` 包；`serve.sh`
显式使用 `cube_310_int4_a8` 与 `activation_quantization=int8_per_group`，不会静默
改变激活精度。远端固定根目录：`/srv/ai/src/native-int4-w4a8.KiuhBN`。例：

```bash
python /srv/ai/src/native-int4-w4a8.KiuhBN/pass3/launch-server.py route-cache
# 第三个参数将 scheduler 上限提高到 4；未提供时保持已测的 2。
python /srv/ai/src/native-int4-w4a8.KiuhBN/pass3/launch-server.py route-cache \
  /srv/ai/src/native-int4-w4a8.KiuhBN/runtime-qsa-combined 4
python /srv/ai/src/native-int4-w4a8.KiuhBN/pass3/validate-server.py route-cache
python /srv/ai/src/native-int4-w4a8.KiuhBN/pass3/verify-service.py route-cache
```

本轮验证了真实权重 HTTP smoke、四会话并发计数、TP4+EP、MTP 与 ACLGraph；VL 关闭，
flashcomm1、128k+bs16 和 4×262144 实际填满的长 prompt 未重新验证。
本轮无 dummy-only 推断；性能与质量数据均来自真实权重。

## 数学热路径复查

额外复查发现 MTP eager 专家路径仍在每个 draft pass 执行 `counts.tolist()`，随后
用 Python 循环逐专家做 W8A16 GEMM；这是同步和小矩阵开销的明确来源。但 310P
当前 grouped matmul 不支持所需的 INT8 权重 antiquant 组合，现有 grouped W8A16
文件也明确标为未来路径，不能安全地直接接到新硬件的 API。下一步需要一个 310P
专用 grouped W8A16 AI Core 算子，再做 MTP 接受率和真实 tok/s 验证；本轮没有加入
只会在运行时失败的占位分支。

已实现的等价简化是在一次 backbone forward 内让 12 个 QSA 层共享 query 与
index-key RoPE 表；每次目标 forward 的表计算从 24 次降到 2 次。缓存仅活在本次
forward，key 包含输入 positions 的存储、版本、形状、dtype、RoPE/MRoPE 几何和
压缩率，因此 changing-input graph replay 仍从当前 positions 重新计算。单卡真实
权重结果和准确率见 `single-card/README.md`。

另一项等价简化缓存静态 Gemma norm 的 FP32 `1 + weight` affine。目标 forward
由此删除 97 对重复 cast/add，单层 MTP 每个 draft step 再删除 5 对；每 NPU 增加
约 3.96 MiB。缓存不进入 state dict，并根据权重指针、版本、设备和 dtype 失效；
训练/启用 autograd 时仍走原始路径。单卡六次 512-token decode 合并中位为
18.27 tok/s，相对 18.77 的基线中位低约 2.6%，在 MTP 接受率和逐请求波动范围内；
固定 228 题为 205/228、零无效答案。因计算严格等价且没有明显回退，按本轮保留
小幅等价简化的策略合入；最终 TP4 数字另列于上表/后续记录。
