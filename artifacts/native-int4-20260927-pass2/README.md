# 原生 INT4 第二轮性能优化

基于 `a5a65d677`，分支 `perf/native-int4-pass2-20260927`。
只修改原生 INT4 matmul 调度；激活量化、权重格式、模型后端选择策略和 W8 路径不变。
最终包为远端 `/srv/ai/src/native-int4-w4a8.KiuhBN/ops-pass2-vector`。

## 改动

- M16 decode/MTP 和 M32 稀疏 prefill 根据输出维度、实际 block 数选择 N160、N80
  或 N64，让模型的 N1280/2560/5120 投影均匀分配到八个计算核。
  M128 密集 prefill 保留 N64，避免超出 UB 容量。
- scale、zero-point、weight sum 各使用一次跨 strip 的 DMA；L1→L0A/L0B 用
  strided repeat 批量装载。保留完整 K 权重驻留和 L0B 双缓冲。
- 修正按运算阶段遍历全部独立输出 strip，阶段间同步，减少逐 strip barrier；
  FP32 运算顺序、两次 INT4 MMAD 和最终 FP16 输出精度不变。

## 单层诊断

真实 checkpoint layer 0 / TP rank 0，无 collective；输入固定。
每个形状五轮，每轮 30 次 graph replay。以下为中位延迟，单位 ms，**不是 tok/s**。

| tokens | 原生旧版 | metadata | balanced | strided | 最终 vector |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3563 | 0.3560 | 0.2799 | 0.2934 | 0.2661 |
| 3 | 0.5174 | 0.5179 | 0.4122 | 0.4175 | 0.3794 |
| 6 | 1.9303 | 1.9430 | 1.4702 | 1.4585 | 1.3347 |
| 32 | 7.9872 | 8.0030 | 6.2014 | 6.1130 | 5.7159 |
| 512 | 28.0121 | 28.0236 | 21.5941 | 21.5523 | 20.4356 |

metadata 单独无明显收益；balanced 加入 N160/N80；strided 再批量装载 L0；
vector 再减少向量同步。最终相对旧版降低延迟 25–31%。
所有变体、所有形状的 eager 和 graph 输出 SHA256 完全一致。
见 `layer-*.jsonl`、`layer-summary.json`。

## 验证与复现

- 原始 79 个 NPU 测试验证 metadata；扩展后的 109 个测试分别验证其余三个变体。
  增加 N80 分支、不同输出宽度的 changing-input / changing-route 图重放。
  N64 fallback、N160、密集 prefill、peer 清零和专家边界仍覆盖。
- CPU 回归 1091 passed、7 skipped。日志为 `cpu-tests.log.gz`。
- `provenance.json` 记录每个变体的设备源码和三个 `.o` 哈希；本地最终源码与
  实际编译源码一致。模型 Python 源码及激活 pack 二进制与上一轮完全相同。
- 环境沿用上一轮：4×Ascend 310P3、CANN 9.1.0、同一私有 venv、同一 W4 checkpoint。
  未升级依赖。完整配置见上一轮报告；启动脚本保持 TP4+EP、MTP2、FULL_DECODE_ONLY、
  capture `[1,2,3,6]`、512-token chunk、bs2、相同 CPU affinity。

本目录脚本使用实际主机绝对路径。`build-variant.sh` 为每次构建创建独立安装目录，
保存源码、失效旧 `.done`，并检查 CANN 内层编译错误；不得在已有标签上覆盖构建。
`serve.sh vector` 显式启用已经许可 INT8 激活的 native override。
`validate-server.py vector` 依次执行真实权重 smoke、双会话检查、三次短/长上下文测速，
再运行与上一轮完全相同的 228 题 MMLU 抽样。题目 manifest 和复现脚本见上一轮目录。

```bash
bash serve.sh vector
python validate-server.py vector  # 需要记录对应 server-vector.pid
python compare-results.py --candidate vector
```

## 整模型吞吐

每项为三个串行 512-token 请求的 decode 中位 tok/s；warmup 不计入。
对照为上一轮同一主机、同一 checkpoint 和服务配置的记录，不是本轮交错重跑。

| 上下文 | W4A16 对照 | 上一轮 native | 本轮 native | 相对上一轮 |
| --- | ---: | ---: | ---: | ---: |
| 短 | 15.18 | 17.18 | 19.28 | +12.2% |
| 约 23.4k | 14.44 | 16.69 | 18.34 | +9.9% |

本轮各次 decode tok/s：短 21.34、18.80、19.28；长 20.43、18.34、17.49。
聚合 end-to-end 吞吐为短 19.42、长 17.73 tok/s。MTP 接受率为 74.15%、73.55%；
上一轮 native 分别为 72.68%、75.20%。输入配对，但生成输出及接受率并非全部相同。
模型吞吐变化包含这些请求差异；单层相同输入、逐位一致的结果用于单独验证内核收益。

长上下文冷 prefill 的 TTFT 从上一轮 99.80 秒降为 85.11 秒。之后三个计时请求复用
23168 个 prefix token，不能把它们的 TTFT 当作冷 prefill。见 `vector-short.jsonl`、
`vector-long.jsonl`。仍只验证到约 23.4k 上下文，配置上限 262144 不代表容量验收。

## 整模型质量和保留状态

同一固定 MMLU 样本完成 228 题，204 正确、0 invalid；上一轮 native 为 203，
W4A16 为 206。本轮相对上一轮有四个答案变化：一题退步、两题改善、一题错项改变。
这不是全量 MMLU，也不能据一题的净增加声称普遍精度提升。单层逐位一致的结论不
扩展为整模型所有生成文本逐位一致。逐题差异和性能汇总见 `paired-summary.json`。

真实权重 HTTP smoke、双会话计数、TP4+EP、MTP2、ACLGraph 均通过。
VL、flashcomm1、128k+bs16 和更大并发沿用上一轮的未验证范围。
本轮更改文件的 manual pre-commit 检查及 C++ clang-format 检查通过；
完整仓库格式检查上一轮已有无关失败，见 `../native-int4-20260927/format-ci.log.gz`。

按用户明确要求，**保留优化后的 native W4 服务运行**，没有恢复 W8。
API 为 `http://192.168.53.187:8001`，model 为 `qwen38-w4-experimental`，
API PID 为 2160698。`service-verification.json` 确认健康检查、四个 worker、
CPU affinity、原生后端/激活许可，以及实际安装包哈希均匹配测试版本。
私有 W8 启动快照仍留在主机，没有复制到仓库。自动后端策略不变；当前服务通过
模型级 override 明确选择 native W4A8。
