# W4 设备分组、原生 INT4 与 profiling（2026-09-27）

## 范围与门禁

保持原 W8、W4 checkpoint、端口 8001、MTP k=2、decode graphs、TP4/EP、
CPU affinity 及双窗口配置。所有候选构建位于远端隔离目录：
`/srv/ai/src/qwen38-w4-native-build.zHZtEd`。
旧运行环境 `/srv/ai/src/qwen38-w4-ce1862e52` 未改写；恢复启动脚本为
隔离目录中的 `baseline-launch-preserved.sh`。

模型仍为 48 层混合 QSA/GDN MoE；本次只改变 W4 expert 投影与路由。
共享 expert、QSA、GDN、MTP 权重不重新量化。

## 根因与实现

- 旧 `cube_310_routed` 仅支持最多 80 条路由；大 batch 回退到
  `.cpu().tolist()` 和 Python expert 循环。
- 新 `cube_310_grouped` 保留小 decode kernel，大 batch 使用设备排序、
  累计 group ends 和设备分组 Cube kernel；空 expert 跳过，peer 行每次清零。
  不读取设备路由到 CPU，按静态 shape 分块（最多 512 tokens/5120 routes）。
- 新 `cube_310_int4_a8` **实验选项，不是性能合格的默认后端**。
  使用 CANN 9.1 dav_m200 `Mmad<int4b_t,int4b_t,int32_t>` 的 `mad_s4`，
  W4 权重直接装入 Cube，不解包成 FP16 矩阵。
  FP16 activation 每行、每 128 元素量化成 INT8，然后拆成两份 signed INT4：
  `a8 = lo + 16*hi + 8`。
  `dot(a8,q-z) = dot(lo,q)+16*dot(hi,q)+8*sum(q)-z*sum(a8)`。
  拆分与整数乘法精确；activation 量化是新增误差，不等价于 W4A16。
- 新 Python profiler 工具支持有限 NPU capture、可选 py-spy 多进程采样、
  cProfile `.pstats` → DOT 调用图。工具默认 dry-run，不自行重启服务或安装依赖。

## 已完成验证

- 最终 CPU focused tests：95 passed、7 skipped（W4、export、byte-mask、
  grouped/native、profiling、dtype fixture、QSA decode 调度与构建注册；跳过硬件项）。
- NPU `device-tests-r1.log` 至 `device-tests-r3.log`：各 23 passed；
  `device-tests-r4.log` 增加非法 metadata 拒绝与 Meta shape 检查后 27 passed。
  覆盖 1/15/16/17/127/128/129/513 行、空 expert、peer 清零、真实投影尺寸、
  修改输入/权重/group ends 后的 graph replay，以及所有权重字节/zero points。
- 真权重单层、合成 activation（seed 1024），rank0 TP partial，无 collective：

| tokens | Python 路由 W4A16 ms | 设备分组 W4A16 ms | 原生 W4A8 ms（首次） |
| ---: | ---: | ---: | ---: |
| 3 | 1.41 | 0.80 | 10.18 |
| 32 | 16.74 | 8.67 | 45.79 |
| 512 | 127.25 | 28.81 | 550.55 |

数据见 `layer-cube_310_grouped-r2.jsonl` 与 `layer-cube_310_int4_a8-r2.jsonl`。
重复 r3 设备分组 512 tokens 为 28.78 ms，对照 126.78 ms（4.4×）。
原生 INT4 消除重复空 expert 边界读取后仍不够快；不能凭 INT4 指令就推断加速。
W4A8 单层相对 L2 约 0.0092–0.0123、cosine ≥0.999924，低于预设误差门限
（L2≤0.05、cosine≥0.999），但**尚未证明全模型质量**。

native r2 trace 的单 token 测试中，cast/layout/copy 占设备 task 时间约 65.6%，
主要是 INT16 bitwise 操作触发 AiCPU Cast。已改为精确的小整数 FP32 打包运算，
避免 INT16 bitwise 路径；所有 nibble 对的 CPU 一致性测试与新 NPU gate 通过。
r4 单 token 的三次调用 cast/layout task 总计从 16.00 ms 降到 2.34 ms；
512 tokens 原生路径中位数从约 548 ms 降到 169 ms，但同轮 W4A16 对照为 35.67 ms。
r4 测试时完整服务仍驻留但 idle，host timing 波动大；不能将该变化直接推算为模型 tok/s。
原生路径仍有每组两次整数乘法、16×16 小 tile、逐行 scale 修正等开销。

## 独立 dtype profiling

`profile_projection_dtypes_310.py` 在 310P 上比较同一组矩阵数值：
gate/up 为 K=2560、N=1280，down 为 K=640、N=2560；
M=1/3/6/16/32/128/512，共 122 个计时配置、14 份 NPU trace。
其中 94 个完整投影配置分别校验 eager 与 graph 输出；其余 28 个单独测 activation 打包。
每项 5 轮、每轮 30 次，graph 内展开 10 次以摊薄 Python replay 开销。
服务驻留但无请求，benchmark 绑定 CPU 6–7，最高温度 58°C。

下表为 graph 的 NPU event 中位数，单位 µs：

| 投影 / M | FP16 已排布 | INT8 已排布 | INT8 含 activation 量化 | packed W4A16 grouped | native INT4 已打包 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate/up / 3 | 17.58 | 27.05 | 30.54 | 109.17 | 256.58 |
| down / 3 | 10.67 | 9.53 | 12.63 | 66.43 | 131.28 |
| gate/up / 512 | 96.39 | 56.89 | 91.21 | 1883.57 | 19278.60 |
| down / 512 | 58.98 | 33.88 | 57.17 | 998.85 | 9710.09 |

完整数据：`dtype-matrix-r1.jsonl`；设备 task 摘要：`dtype-trace-summary-r1.json`。
native INT4 activation 拆分单独耗时：gate/up M=3 为 63.34 µs、M=512 为 1262.03 µs；
相应 native 完整投影为 319.36 µs、20573.21 µs。trace 中 prepared native kernel 本身
也明显更慢，不能只归因于 Python dispatch 或 activation 打包。

**结论不是“INT4 硬件慢”或“INT8 总最快”。** Stock INT8 在 M=512 的 prepared
matmul 比 FP16 快约 1.7×，但计入 activation 量化后优势只约 3–5%；
小 M 的 gate/up 反而是 FP16 更快。现有两个 custom W4 kernel 的解包、
tile 利用率、每组 scale/zero-point 修正仍是主要优化对象。

这是单 expert、热权重、固定 scale、无 router/collective 的实现对比，**不是峰值 TOPS
测试，也不能直接换算全模型 tok/s**。FP16/INT8 已排布权重占用分别为 W4 codes 的
4×/2×；没有把完整 expert bank 改存 FP16/INT8。真实 checkpoint 每 128 元素的
scale 不同，无法简单将 nibble 扩展为 INT8 后调用一次 whole-K INT8 GEMM 来保持原计算。
fixture 统一 scale 只是为了让所有路径的矩阵数值完全相同。

下一步优先优化 packed W4A16 的 tile 解包与复用；另一个候选是保持 W4 存储、
按 tile 扩展 INT8 并正确处理每组 scale。后者仍需要 activation 精度门禁和完整
路由/通信/生成 benchmark；本次没有因此改变服务 dtype。

## 全模型结果

全 48 层真实权重，算术、1–50 计数、Python 输出均正确；并行两请求返回 713、1073。
MTP k=2，TP4/EP，图捕获完成（0.94 GiB），各 worker 的 CPU affinity 已核验。
未改 checkpoint。以下对照是此前相同 prompts/settings 的 byte-mask-r1 证据，
**不是本轮重新交替启动的 A/B**。MTP 接受率与生成文本不同，不宣称 decode 加速。

| 测量 | 之前 W4A16 | 设备分组 W4A16 |
| --- | ---: | ---: |
| 23,407 uncached tokens → first token | 305.10 s | 100.21 s |
| 相同长 prefix 再用 TTFT（3 请求） | 3.80–4.35 s | 1.62–2.30 s |
| 短 prompt，512 输出 tokens（3 请求） | 13.86–16.96 tok/s | 14.13–16.68 tok/s |
| 长 prompt，512 输出 tokens（3 请求） | 13.92–16.23 tok/s | 14.31–16.77 tok/s |

详见 `full-test-r1.log`、`http-smoke-r1.jsonl`、`validation-summary.json`。
服务仍留在端口 8001，**使用 exact W4A16 grouped，不使用实验 W4A8**。
容量配置仍为 `max_model_len=262144`、`max_num_seqs=2`，不代表本次重跑了双满窗口。
128K×16 超出本任务的现有双窗口内存预算，不作为本次 gate。

| 功能/验证阶段 | 本轮状态 |
| --- | --- |
| 真实权重 | 全模型 HTTP 和输出检查通过；不是仅 startup |
| dummy | 未使用；低层合成 tensor 只证明算子，不等价于模型质量 |
| MTP | k=2，接受率有非零增量，多 token decode 通过 |
| ACLGraph | 服务 trace 含 88 个 `aclmdlRIExecuteAsync` 调用；低层 replay 更换数据/路由测试通过 |
| EP | TP4/EP4 全模型与并发通过 |
| flashcomm1 / 多模态 | 沿用既存文本服务配置，本轮未改变或重新验收 |
| 双满 262K | 配置保留，本轮只验证并发短请求，未重新跑满 |
| W4A8 模型质量/速度 | 只有真实单层+合成 activation 精度 gate；速度未通过，不启用 |

服务离线 trace 已解析（`server-trace-summary-r1.json`）：rank0 cold capture 的
grouped W4 projection 占 summed task time 40.58%，collective 14.13%，
cast/layout/copy 10.44%。这些是工作量归因，不能相加成关键路径延迟。
warm capture 同时含 prefill 和 decode，不能标记为纯 decode profile。
pipeline 硬件计数器未收集，不用空值推断 Cube 利用率。

## 限制与失败记录

- W4A8 性能门禁失败，不切换默认服务；全模型 benchmark accuracy 未评估，
  不填造 accuracy YAML 或宣称可替代 W4A16。
- 本地全套 Qwen CPU 测试：998 passed、7 skipped、19 failed、4 errors；
  构造测试引用当前 vLLM checkout 已移除的 logits-processor TP helper 等接口。
  focused 新增测试通过；不能宣称全套通过。
- 全仓 `bash format.sh ci` 有既存 lint/import/meta 问题。其产生的无关格式修改已恢复。
- 最终 scoped lint 的已知阻塞是原有 `torch_binding_meta.cpp` 第 655 行的
  `empty_symint` 写法；新增 Meta kernel 使用显式 `SymDimVector`。
- 测试工作树包含此前已暂存的 byte-mask 优化；本次提交不吸收那些既有改动。
- 首次 layer harness 覆盖了 CANN 的 PYTHONPATH，导致 CANN 编译模块导入失败；
  r2 起保留 CANN 环境后通过。未改依赖版本。
- py-spy 未安装，未自动安装；已实际收集的是 layer cProfile 与 NPU traces。
- vLLM daemon worker 不能解析 trace；`profile_runtime analyse` 使用同一 CANN 环境
  在独立 host 进程离线解析。原始服务 trace 保留在远端 `server-traces-r1/`。
  首次缺少 CANN 动态库环境及 parser 导入路径问题已在工具验收中发现并修正。
- 不启动 CPU speculative 路径；不修改 flashcomm1 或多模态路径。

## 紧凑 runbook

设备分组选项：保持 checkpoint 原 metadata，仅将
`text_config.ascend_expert_quantization.backend` 改为 `cube_310_grouped`。
原生 INT4 仅用于实验，将该字段改为 `cube_310_int4_a8`，不得当作精度/速度验收。
两者都需要重新编译 Python extension 和对应 CANN custom operators。

本机已准备好的直接 serve 命令及守护脚本位于远端隔离目录：
`serve-grouped.sh`、`supervise-grouped.sh`。仍使用 8001，保留旧 launcher 以便恢复。

```bash
python -m tools.qwen4exp.profile_runtime config \
  --trace-dir /path/to/new-traces --phase cold-prefill --steps 6
# 将输出 JSON 作为下次启动的 --profiler-config；不要在线改配置。
python -m tools.qwen4exp.profile_runtime capture \
  --phase cold-prefill --steps 6 --trace-dir /path/to/new-traces \
  --output /path/to/new-capture --execute -- python benchmark_client.py
python -m tools.qwen4exp.profile_runtime cprofile-dot layer.pstats \
  --output layer.dot
# 先 source 当前安装的 CANN 环境，再离线解析 daemon worker 原始数据：
python -m tools.qwen4exp.profile_runtime analyse /path/to/new-traces --processes 1
python -m tools.qwen4exp.summarize_trace /path/to/new-traces --output summary.json
# 独立 dtype 对比：先停止并发请求及其他 profiler，保留当前服务配置。
python -m tools.qwen4exp.profile_projection_dtypes_310 \
  --output /path/to/new-dtype.jsonl --rows 1 3 6 16 32 128 512 \
  --iterations 30 --repeats 5 --graph-unroll 10 \
  --trace-dir /path/to/new-dtype-traces
```

decode capture 前先预热同一 prefix；delay 只是 iteration 数，不会自动辨认 prefill。
profiled 请求不能用于生产 tok/s 对比；CPU 累计时间包含等待 NPU，不能当成 kernel 时间。
只有可信本机 `.pstats` 可输入，因为 pstats 使用 pickle。
