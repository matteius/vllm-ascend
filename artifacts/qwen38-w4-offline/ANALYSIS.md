# Qwen W4 离线适配记录

## 范围与隔离

本次仅在独立 worktree 中开发，并在本机 CPU 上生成独立量化目录。
未修改运行中的 W8 模型、启动脚本、服务器、NPU 环境或原始 BF16 权重。
用户要求离线，因此不执行适配技能默认的启动、重启及真实 NPU 推理步骤。
此交付是离线实验候选，不是生产验收。

## 架构与原因

Qwen Flash Next 主干有 48 层、每层 512 个 routed experts，top-k 为 10；
hidden size 为 2560，expert intermediate size 为 640。当前 W8 loader
严格要求 INT8 权重和逐通道 FP32 参数，不能把压缩 INT4 张量冒充 INT8 加载。
通用 Ascend W4A16 fused-MoE 格式也不能直接套用到既有 310P 专用 Qwen 路径。

新增格式显式描述 packing、group size、scale/zero-point dtype 和 backend。
ModelSlim IR 执行非对称逐组 min/max RTN；采用官方 INT4 packing 函数。
group=128，scale=FP16，zero point=INT8，resident routed experts 理论约
58.8867 GiB。其余浮点权重、KV、SSM、graph/workspace 不包含在这个数字中。

首版 eager 后端按路由仅展开当前 expert，不保留 FP16/INT8 副本。
它需要 CPU 路由同步，不能作为图回放或 15 tok/s 性能证明；
显式要求 `--enforce-eager`，并拒绝通用 `--quantization` 覆盖。
W8 未带新 metadata 时，仍进入原 `_EagerSparseMoE` 和原 post-load 路径。
PLE 仅为新格式切换 safetensors index 文件名，继续懒加载 host table。

## 验证边界

| 项目 | 离线证据 | 仍缺少的真实证据 |
| --- | --- | --- |
| INT4 packing、zero point、scale | CPU 正负 nibble、严格 dtype/shape 测试 | 310P 算子兼容性 |
| loader | 合成完整 checkpoint、缺失/重复/非法参数拒绝 | 完整真实模型启动和请求 |
| TP expert/shared 求和 | 不均匀 expert 分片数值对照 | NPU 通信及端到端 TP |
| ModelSlim 产物 | 原始 BF16 → 独立 W4 文件、可恢复构建、hash | 模型困惑度和任务正确率 |
| PLE | 两种 index 路径、lazy lookup 测试 | W4 长上下文推理 |
| ACLGraph | 初始后端明确禁止 | packed kernel 和 replay 测试 |
| MTP | 保留浮点 checkpoint，不改原实现 | W4 target 的接受率和质量 |
| multimodal / flashcomm1 / EPLB | 未测试 | 后续独立验证 |
| 128k × 16 / 双 160k | 不执行：离线范围，无 NPU 容量验证 | 实测内存、正确率与并发吞吐 |

RTN 不等于 GPTQ/AWQ，局部重构误差不等于模型回答质量。不能用旧 W8
测试结果或其它平台四比特结果替代当前候选的真实验收。
未生成带虚构 accuracy 数值的自动 CI 配置，亦未发布 checkpoint 或推送分支。

## 本次构建与验证环境

- 独立 worktree：`/run/media/matteius/20TB-drive/qwen38-w4-offline-20260926`
- 分支：`feat/qwen38-w4-offline-20260926`
- 基线：`aa136a9d5bd22809b078307d15f2c0010dba9094`
- 原始模型：`Qwen/Qwen3.8-Flash-Next`，本地下载记录 revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- ModelSlim：`01d1bf92e088b70acc807c3b98391106d460cbfa`
- 转换使用独立 CPU 环境：PyTorch `2.11.0+cpu`，4 个计算线程。
- 输出：`/run/media/matteius/20TB-drive/models/Qwen3.8-Flash-Next-W4A16-G128-300i`

### 完整真实权重产物

构建已完成：48 层、73,728 个 expert projections，输出 **1,610 shards /
222,746 tensors / 181,637,185,528 bytes（169.1628 GiB tensor payload）**。
`build-journal.json` 为 `complete=true`，最终 index 已生成；每个输出 shard
有 SHA256 receipt。最后复核原始 index/config hash 与所有原始 shard 的
size/mtime，均与构建前一致；tokenizer、chat template、generation config
的复制文件 hash 与原件一致。

原 W8 routed experts（含逐通道参数）为 113.2031 GiB，新 W4 为 58.8867 GiB，
静态存储减少 **54.3164 GiB**，TP4 均匀分片约 **13.5791 GiB/rank**。
该差值不是实际可用 KV 增量，尚需真实 NPU workspace、缓存及质量验证。

完整 header/index 审计通过：routed experts 58.8867 GiB，PLE 95.3679 GiB，
其余张量 14.9082 GiB，合计 169.1628 GiB。抽样层为 0/12/24/36/47，
expert 为 0/255/511，每组检查 gate/up/down，共 45 个真实 projection。

| 抽样指标 | 最小值 | 平均值 | 最大值 |
| --- | --- | --- | --- |
| 相对权重 RMSE | 0.100542 | 0.103625 | 0.118554 |
| 权重 cosine | 0.993009 | 0.994629 | 0.994936 |
| Gaussian 输入 projection 相对 RMSE | 0.099507 | 0.103506 | 0.116563 |

重构和输出均为有限值。完整结果见 [OFFLINE_AUDIT.json](OFFLINE_AUDIT.json)，
模型目录同时保留 `offline-audit.json`，日志为 `/tmp/qwen38-w4-audit-20260926.log`。
这些数字仅验证局部重构，不是模型困惑度、编码质量或 NPU 吞吐结果。

### CPU 测试与基线对照

专项测试 **82 passed**；完整 Qwen 目录为 **833 passed / 19 failed / 4 errors**。
相同主机、相同 vLLM 源码和测试命令的未修改基线为
**811 passed / 19 failed / 4 errors**。失败和错误的测试标识集合完全一致，
新增 22 个通过项；不能把完整测试目录描述成全绿。
基线问题涉及既有测试 fixture、当前 vLLM 配置上下文与接口差异。

相关日志位于本机 `/tmp/`：

- `qwen38-w4-focused-tests-final-20260926.log`
- `qwen38-w4-host-tests-final-20260926.log`
- `qwen38-w4-baseline-tests-final-20260926.log`
- `qwen38-modelslim-w4-compact-build-20260926.log`
- `qwen38-w4-format-ci-20260926.log`
- `qwen38-w4-format-scoped-final-20260926.log`

已执行要求的 `bash format.sh ci`，仓库全量检查存在既有问题，未修改无关文件。
修改文件的适用 hooks 全部通过；仅跳过全局 `check-symbolic-meta`，其报错来自
基线已有的 `csrc/torch_binding_meta.cpp:655`，该 hook 不受文件列表限制。
未调用真实 NPU、HTTP 推理服务或 SSH；没有模型质量和 tok/s 验收结果。

## 紧凑运行说明

详见教程 `docs/source/tutorials/models/Qwen3.8-Flash-Next-W4-Offline.md`。
首先离线执行 converter，再执行 audit。后续获准使用 NPU 时，在独立环境、
独立端口以 TP4 + eager + 8k context 验证；先不启用 MTP/graph。
显存比例不超过既定 0.965 上限。必须完成实际回答，不能仅凭服务启动成功验收。

## 后续优先级

1. 使用原 held-out 编程任务和困惑度数据评估 RTN；若不达标再加入校准。
2. 为 310P 实现/验证真正直接消费 groupwise packed W4 的算子。
3. 路由留在设备端，验证 graph replay，随后独立验证 MTP。
4. 在实际显存账本上测量上下文和并发窗口，再考虑生产切换。
