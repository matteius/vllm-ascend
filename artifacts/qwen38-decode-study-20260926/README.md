# Qwen3.8 Flash Next 解码实验（2026-09-26）

## 范围与基线

本实验以 NPU 主机实际运行的 `qwen4exp-serving-20260923` 为基线，
不以落后于线上优化的本地 `main` 替换它。源码与已编译扩展的 SHA256
记录在 [baseline-hashes.json](baseline-hashes.json)。原启动脚本也单独备份。

- 主机：`matteius@192.168.53.187`，Threadripper 3970X，4 × Ascend 310P3。
- 模型：`/srv/ai/models/Qwen3.8-Flash-Next-W8A8-DYNAMIC-300i`。
- 运行版本：torch `2.13.0+cpu`、torch-npu `2.13.0rc1`、
  transformers `5.17.0`、vLLM `0.1.dev1+g3ab5dda29.empty`；未升级依赖。
- 全部分组保持 TP4、单活跃请求、160000 上下文、2048 prefill chunk、
  MTP K=1、`FULL_DECODE_ONLY`、prefix caching、内存上限 0.965。
- 用户先要求推迟重启，随后明确授权停止服务。实验使用本机端口 8002，
  原服务端口为 8001。NPU worker 停止发送 SIGTERM；操作者不发送 SIGKILL。

## 分组与代码

| 分组 | CPU 绑定 | MTP 专家路径 |
| --- | --- | --- |
| control | 原实现，x86 跳过 | 原 W8A16、CPU counts、逐专家执行 |
| affinity | 每 rank 六个物理核 | 同 control |
| grouped | 同 control | 设备分组路由和两次 W8A8 grouped matmul |
| combined | 同 affinity | 同 grouped |
| combined-eager | 同 affinity | grouped，但保留原 eager callback |

绑定池为 `0-5`、`8-13`、`16-21`、`24-29`；不把同一物理核的 SMT
兄弟分给不同 rank。还有八个物理核未纳入 worker 池；API/系统线程没有
额外绑核。完整拓扑见 [cpu-plan.json](cpu-plan.json)。

分组 draft 保留原 INT8 权重和 FP16 舍入后的权重尺度，打包为 NZ 格式，
原参数以 view 共享底层存储。**算子增加了动态 INT8 激活量化，故它不是
W8A16 的数值等价替换。**必须同时核对接受率、目标模型输出和性能。
非本 rank 路由以 sentinel 排在末尾；未写输出行用 `where` 清零，避免
NaN × 0 污染。该路径没有每步专家计数 D2H 或 eager callback。
`combined-eager` 是故障隔离用 fallback；它保留 callback，只替换内部专家计算。

实现位于 `tools/qwen38_decode_study/`，由 `prepare.py` 在独立副本中
应用最小补丁。没有新环境变量，也没有修改线上源码树。源文件哈希不符、
目标目录已存在或目标指向基线内部时拒绝准备。启动 guard 发现其他 vLLM
进程时只报错退出，不会代替用户停止进程。

## 测量协议

- 温度 0、seed 42、thinking 关闭、`ignore_eos=true`，逐个请求。
- 短提示：三种固定编程/分析提示，各生成 512 和 2048 token。
- 长提示：同一 180 模块合成代码前缀（约 23407 token），三种任务各
  生成 512 token；先单独进行 32-token warmup/cold prefill。
- JSONL 保留完整请求、输出、SHA256、usage、客户端时序和 MTP counter
  差值。`summarize.py` 按请求 ID 读取服务端 token-gap 时序，避免将
  prefill、聚合吞吐或 SSE chunk 数当成单请求解码速率。
- 按相同请求逐项配对；不同请求配置拒绝比较。报告输出完全一致的数量。
- 独立语义 smoke 含六个文本题和一次严格参数工具调用。control 为 6/7：
  `ASCEND` 逆序应为 `DNECSA`，基线却输出 `DNESCA`；工具调用通过。
  后续分组不能使基线已通过的题目退化。

这是小规模合成工作负载实验，不是完整 GPQA、160K 正确性或并发能力验证。
MTP/图模式使用真实 checkpoint；没有以 dummy 结果代替真实权重验证。
多模态、EP、flashcomm1、128K × 16 均不属于本次单请求优化范围。

## 实测结果与选择

完整请求指标、输出哈希、smoke 响应及远端原始文件校验值见
[results.json](results.json)。以下为三个不同提示的中位数，**不是三次同提示重复**。
增幅列是逐项配对倍率的中位数，因此不等于两列中位数之比。

| 工作负载 | control tok/s | affinity tok/s | 配对增幅 |
| --- | --- | --- | --- |
| 短提示，512 输出 | 17.90 | 19.01 | +4.60% |
| 短提示，2048 输出 | 18.40 | 19.52 | +5.49% |
| 约 23K 提示，512 输出 | 17.88 | 18.67 | +5.05% |

冷前缀 warmup 的 TTFT 为 45.65 → 45.35 秒；没有显示明显 prefill 改善。
后续长提示命中 23168 个前缀 token，但解码时仍使用完整约 23K 上下文。
按 `(1 + acceptance) / tok_s` 粗估每轮延迟，短提示约 101–102 → 97 ms，
长提示约 105.5 → 100.3 ms。这只是接受率归一化估计，不是 NPU profiler 测量。

affinity 的七项 smoke 与 control 一致（6/7，同一逆序题失败，工具调用通过）。
九项长输出哈希均不相同，不能宣称 bit-exact；control 自身在不同输出长度
设置下也出现措辞差异，尚未验证严格确定性或大规模质量等价。
此次受控 control 速率高于用户先前的 15–17 tok/s，不能把整个差额算作优化收益。

**选择 affinity；拒绝 grouped 上线。**grouped 启动和六次短文本 HTTP 请求成功，
但工具调用持续解码停滞，内核记录 `devmm` 非法设备内存访问；没有有效吞吐结果。
具体根因尚未隔离，不能把它归因于某个算子或量化误差。`combined` 未执行。
`combined-eager` 随后的启动被残留设备状态阻塞，在 ready 前终止；其数值和性能
都未完成验证。故障记录见 [failures.json](failures.json)。

grouped 停止发送 SIGTERM；日志显示 vLLM 的 abort-mode shutdown 内部以 0 秒超时
强制结束 EngineCore。后续三个 worker 卡在驱动，芯片复位因引用仍在而返回
`EBUSY`。worker 最终因设备超时退出，再次复位仍报 `Hotreset is executing`。
无 NPU worker 的启动中 API 后续另发 SIGINT。初次交付时尚未恢复服务。
该内存错误出现在停滞后的 shutdown/recovery 阶段；尚未证明它就是首次停滞的
原因，也未证明 grouped MTP 无法修复。

后续只读代码审计发现了 310P 路由的非法专家 ID 契约问题，并区分了确定缺陷
与尚未隔离的 graph/collective 风险，详见 [failure-audit.md](failure-audit.md)。

用户随后要求修复但暂不测试。新候选生成器已改用合法的虚拟 peer 路由槽，
同时修正 target/draft 的该调用路径，并默认保留 MTP eager callback。
这些改动尚未测试或部署；历史实验结果不能作为修复验证。
变更及后续验证说明见 [routing-fix.md](routing-fix.md)。

## 服务恢复（23:16 UTC）

用户随后要求启动约 19 tok/s 的 affinity 版本。设备仍无法完成简单计算，
执行主机重启后，四张 NPU 均通过真实张量计算检查，再启动正常端口服务。
[recovery.json](recovery.json) 保存恢复证据：

- 地址：`http://192.168.53.187:8001`，model ID `qwen38-flash-next-w8a8`。
- source root 为 `affinity`；四个 rank 的所有线程分别绑定到既定 CPU 池。
- 语义 smoke 为 6/7，与基线相同；严格参数工具调用通过。
- 512-token 持续生成的服务端解码速率 **19.22 tok/s**，MTP 接受率 89.26%。
- 持久 tmux 会话：`qwen38-affinity`；上下文上限仍为 160000。
- grouped draft 未启用。本次恢复没有重新运行完整 A/B 矩阵。

## 操作与证据

NPU 主机实验根目录：`/srv/ai/src/qwen38-decode-study-20260926`。
各分组保存 `serve.log`、`smoke.json`、`benchmark.jsonl`、
`long-benchmark.jsonl`、`candidate-manifest.json` 和温度/内存快照。
`harness.log` 保存执行顺序，`run-status.json` 保存完成与恢复状态。

主机上的准备命令示例（只生成副本，不启动）：

```bash
cd /srv/ai/src/qwen38-decode-study-20260926/study-tools
python3 -m tools.qwen38_decode_study.affinity > ../cpu-plan.json
python3 -m tools.qwen38_decode_study.prepare \
  --source ../baseline --destination ../new-affinity \
  --launcher ../original-launcher.sh --arm affinity --cpu-plan ../cpu-plan.json
bash ../new-affinity/launch-candidate.sh --show
```

服务已独占 NPU 后，使用持久 tmux 会话启动生成的脚本；不要与线上实例
同时启动。正常启动脚本仍为 `~/start_qwen38_flashnext_mtp_graph.sh`，默认 source root
已改为本实验的 `affinity` 副本；原始 source tree 完整保留。
恢复必须检查 `/v1/models` 和真实 completion，不能只看 startup 日志。

原始配置回滚（服务停止后执行）：

```bash
cp /srv/ai/src/qwen38-decode-study-20260926/original-launcher.sh \
  ~/start_qwen38_flashnext_mtp_graph.sh
```

合成长前缀可以逐字节重建：

```bash
python3 -m tools.qwen38_decode_study.long_prefix > long-prefix.txt
```

本地 host 测试：

```bash
python3 -m pytest --confcutdir=tests/ut/qwen38_1m -q \
  tests/ut/qwen38_1m/test_decode_study.py
```

本地 20 项 host 测试通过。变更文件的格式、拼写、secret scan 等检查通过。全仓 symbolic-meta hook
发现已有 `csrc/torch_binding_meta.cpp:655` 的 `empty_symint({...})` 问题；
`git show HEAD` 确认该问题在本次修改前已存在。本实验未改动该文件。
