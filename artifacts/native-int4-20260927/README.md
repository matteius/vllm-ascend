# 原生 INT4 W4A8：310P 实验记录

本次只开发独立 `cube_310_int4_a8` 后端。W8A8 路径不变；W4A16 继续使用
原有后端。权重参与原生 INT4×INT4→INT32 乘法，激活动态量化为 INT8 后拆成
两段 INT4。没有新增全局 dtype 开关或环境变量。

## 实现

- 在模型构造、权重布局选择和图捕获前，按 W4 元数据、激活许可、设备和形状
  解析后端。显式原生 override 必须声明 `activation_quantization=int8_per_group`。
  缺失许可、算子不可用或形状不支持时立即报错。`auto` 的原生晋级仍需独立资格门槛。
- `qwen_w4_a8_pack_v310` 融合 FP16→INT8 动态量化、两段 INT4 打包、scale 与
  sum 计算。最终保留对称量化，每批处理 8 个 G128 组，metadata 以 8 个 FP32 lane 广播。
  `auto` 保持 W4A16；显式原生 override 可供实验使用，不代表精度晋级通过。
- 原生 matmul 使用 M16/N64 decode、M32/N64 稀疏 prefill、M128/N64 密集
  prefill。完整 K 的打包权重切片驻留 L1，L0B 双缓冲预取下一组。
- 小批次直接读取设备 INT32 专家 ID，在芯片上生成路由计划并按原路由位置写回。
  prefill 按设备 INT64 累积边界分组，在 top-k 展开前复用量化结果。
  peer 行每次都写零，避免图重放沿用旧值。
- scale、zero-point、activation sum 修正使用向量操作，消除旧内核的逐行 GM
  标量读取。旧 rank-2 metadata 调度保留为独立数值参考，不用于模型热路径。

## 环境及证据

独立分支 `feat/native-int4-w4a8-20260927`，基于 `4e997da32`。
未改动原工作区或另一个离线 W4 工作区的未提交内容。
远端独立目录 `/srv/ai/src/native-int4-w4a8.KiuhBN`；保留的算子安装到 `ops-delivery`。
整模型对称量化测速使用 `ops-final`；`ops-delivery` 仅增加 N≤5120 的 host/binding
形状检查，三个设备 `.o` 完全相同。最终服务脚本使用 `ops-delivery`。
实际包版本、上游 vLLM 提交及二进制哈希见 `runtime-provenance.json` 和 `retained-provenance.json`。

实际设备：4×Ascend 310P3，CANN 9.1.0，torch 2.13.0+cpu，
torch-npu 2.13.0rc1，transformers 5.17.0。没有升级环境。
W4 检查点路径为 `/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i`。

`serve-baseline-r8.sh` 记录实际 W4A16 启动；`serve.sh native` 使用最终包。
两组使用 TP4+EP、MTP2、FULL_DECODE_ONLY、capture sizes `[1,2,3,6]`、
max model length 262144、512-token prefill chunk、max sequences 2、相同 CPU affinity。
W4A16 使用原有 `ops-routeplan-r4`，native 使用独立新算子。
显式原生配置在启动前注册延迟算子，Meta 使用显式符号维度，没有修改 W4A16 计算。

## 算子和单层验证

- 最终保留版本：79 个 NPU 测试通过，见 `device-tests-retained.log.gz`；最终设备二进制和
  模型源码哈希与整模型测速版本一致。N≤5120 的 host/binding 检查也已验证。
  恢复最终源码后再次编译的三个设备二进制也逐字节相同，见 `retained-rebuild-verification.json`。
- 最终 Qwen 和构建清单 CPU 回归：1091 passed、7 skipped，见 `cpu-retained-suite.log.gz`。
- 构建清单与 HTTP 评分测试：10 passed，见 `cpu-build-and-http.log.gz`。
- 更改文件的检查通过，见 `format-retained.log.gz`；完整 `bash format.sh ci` 仍有既有无关 Ruff、拼写、Markdown、
  禁止 import 失败，见 `format-ci.log.gz`。自动格式化的无关更改已撤回。
  同文件旧 W2 Meta 的隐式 shape initializer 改为显式 `SymDimVector`，使 Meta 检查通过。

单层测量是真实 layer 0 / TP rank 0 权重、固定随机输入、无 collective 的诊断，
不是模型 tokens/s。以下两组在同一次脚本中使用相同输入：

| tokens | native eager ms | W4A16 eager ms | native graph ms |
| --- | ---: | ---: | ---: |
| 1 | 0.528 | 0.482 | 0.360 |
| 3 | 0.576 | 0.676 | 0.538 |
| 32 | 8.171 | 8.496 | 8.176 |
| 512 | 28.135 | 28.916 | 28.011 |

相对 W4A16 的单层 L2 误差为 0.0092–0.0123，cosine 最低 0.999924。
一 token eager 仍慢于 W4A16；不能由较大的单投影收益推断所有场景均加速。
原始结果见 `operator-evidence/layer-native-r8.jsonl`。

## 整模型验证

`validate-server.py` 依次检查真实权重 smoke、双会话计数、固定 MMLU 样本、
3 个短上下文与 3 个约 23.4k 上下文的 512-token 流式请求。
质量请求必须完整停止；吞吐请求故意固定输出长度，并保存输出、usage、MTP 计数。

MMLU 固定为 57 科目各 4 题，共 228 题；temperature=0、zero-shot、关闭 thinking。
这是整个模型的抽样准确率门槛，不是官方完整 MMLU 分数。
`mmlu-manifest.json` 保留题目 ID、prompt 哈希和作者数据包 SHA256；
`reproduce-mmlu.py` 可以精确重建，不在 Git 中复制题目文本。
首轮对称量化：W4A16 得分 206/228，native 得分 203/228，两者 invalid=0。
只改变了 3 个答案，均为退步，因此没有自动晋级。

| 首轮对称量化，MTP2 | W4A16 tok/s | native tok/s | 加速 |
| --- | ---: | ---: | ---: |
| 短上下文，decode median | 15.18 | 17.18 | 13.2% |
| 约 23.4k，decode median | 14.44 | 16.69 | 15.6% |
| 短上下文，aggregate end-to-end | 15.03 | 17.23 | 14.7% |
| 约 23.4k，aggregate end-to-end | 14.42 | 16.24 | 12.7% |

每个速度条目来自 3 个串行 512-token 请求，不是大批量并发吞吐。
MTP 接受率：短上下文 W4A16 71.79%、native 72.68%；长上下文 75.86%、75.20%。
这些速度属于对称量化版本，不能直接用于后续量化修订。

长上下文 warmup 包含一次完整冷 prefill；三个计时请求会复用 23168 个 prefix token。
不能把这些请求的 TTFT 当作冷 prefill 速度。基线冷 prefill TTFT 为 101.68 秒。
首轮配对准确率、逐题变化、接受率和吞吐见 `paired-summary.json`。

## 验证范围

| 功能 | 证据和限制 |
| --- | --- |
| 真实权重 | 全部模型实验使用同一个 W4 检查点；未使用 dummy 代替模型验收 |
| TP4 + EP | 四个 rank 加载各自专家并完成真实 HTTP 请求 |
| ACLGraph | FULL_DECODE_ONLY；设备测试改变激活、专家 ID 与分组边界后重放 |
| MTP2 | 实际 speculative 请求及接受计数，不只检查启动 |
| flashcomm1 | 未启用，保持原 W4A16 对照通信配置 |
| 多模态 | 未测，实验明确使用 language-model-only |
| 容量 | 配置上限 262144、bs2；实测最长约 23.4k，不声称完成 128k+bs16 容量验证 |

固定抽样 HTTP 评分与通用 lm_eval 任务口径不同，未将该分数填入官方完整 MMLU
模型 YAML，以免生成错误的回归门槛。题目 manifest、评分器和逐题证据独立提供。

## 未采用的 affine 量化修订

额外尝试包括零点的非对称 INT8 激活量化，保持两次原生 INT4 乘法，向量修正
激活零点。它通过 103 个 NPU 测试、1101 个 CPU 测试（7 skipped），日志为
`operator-evidence/device-tests-affine-expanded.log.gz` 和 `cpu-affine-suite.log.gz`，但不作为最终实现：

| 指标 | W4A16 | 保留的对称 native | affine 实验 |
| --- | ---: | ---: | ---: |
| 短上下文 decode median tok/s | 15.18 | 17.18 | 16.56 |
| 约 23.4k decode median tok/s | 14.44 | 16.69 | 16.03 |
| 首轮 228 题正确数 | 206 | 203 | 204 |

Affine 没有恢复原来的三个错误答案，仅在另一题得到改善，且两个吞吐场景均更慢。
因此保留速度更高的对称实现。原始逐题、吞吐和 MTP 结果见 `affine-*.jsonl`，
对照汇总见 `affine-paired-summary.json`。该版本首个短请求为 18.49 tok/s，
后续为 16.31、16.56，不能将首个请求当作三次测量的中位数。

预先固定的独立 228 题样本上，affine 得到 206/228；该版本被淘汰后未继续重启
W4A16 做此样本的配对对照，所以这个单侧结果不用于证明精度改善或晋级。
`mmlu-heldout-manifest.json` 记录不与首轮重叠的样本；没有用它继续调参。
`operator-evidence/affine-experiment.patch.gz` 可在最终源码上重建实验修订；需重新
编译 bindings 和 CANN 算子。`affine-provenance.json` 保存实际测试包的哈希。

## 已解决问题和未采用方案

- 扩大 N 后必须按 K fragment、再按 N fragment 排布 L0B。初版沿用小 tile
  顺序导致数值错误，修正后完整设备回归通过。
- dav_m200 的 WholeReduceMax 即使请求 ONLY_VALUE 仍输出 value/index 对。
  使用明确的 VALUE_INDEX 和 Gather 提取最大值。FP32 Compare/Select lane 表达也
  与预期不同；零组 scale 用有限 FP16 最小非零幅值构造精确指示器，避免标量读取。
- N128 调度和常驻/Gather metadata 版本数值正确，但真实 MoE 层较慢，未采用。
  r5/r6/r7 原始结果保留在 `operator-evidence/`。
- CANN 增量构建会保留旧 op info 和 `.done` 标记。dtype 扩展后重新生成 op info、
  编译参数和两个 dtype 变体，重新编译后再做设备回归。另一次实验发现编译阶段
  报错后仍能生成安装包并夹带旧 `.o`，因此同时检查日志错误、设备二进制哈希和数值结果。

## 复现

在本次主机上，先启动明确的实验服务，再运行对应验证，不能同时启动两个占用同一
NPU/端口的服务。脚本中的绝对路径需要按环境修改。

```bash
bash serve.sh baseline  # 或 native
python validate-server.py baseline  # 或 native；需要对应 server-*.pid
bash run-device-tests.sh
python reproduce-mmlu.py --archive /path/to/data.tar --output /tmp/mmlu-228.jsonl
python -m tools.qwen4exp.evaluate_w4a8_http \
  --base-url http://127.0.0.1:8001 --dataset /tmp/mmlu-228.jsonl \
  --output /tmp/native-results.jsonl --label native
python compare-results.py
```

最终结论：完成可显式选择的原生 INT4 后端和实际吞吐/精度验证，但**不晋级为自动默认**。
速度提升不能抵消或隐去 203/228 对 206/228 的精度差距；整数算子正确也不表示模型
精度更高。后续晋级需要更大规模的配对精度验证和明确的精度容忍标准。

模型启动成功不是验收通过；必须检查请求完成、输出正确、图重放以及完整模型指标。
配置中的 262144 上下文上限不等于本次已验证的实际长度。VL、flashcomm1、EPLB、
128k+bs16 容量和更大并发不在本次配对实验内；它们不能从文本 TP4+EP/MTP2 结果推断。

## 服务恢复

已结束本次 W4 实验服务，按私有快照恢复原 W8 服务。argv、cwd 和保存的完整环境
均匹配；API 启动完成，`17 * 19` 请求返回 `323`，finish_reason=stop。
公开确认文件为 `w8-restoration-verification.json`；启动快照和服务日志留在远端私有目录，
没有加入 Git。原工作区及离线 W4 工作区的未提交更改保持原状。
