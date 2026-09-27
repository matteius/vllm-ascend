# W4 容量与并发（2026-09-27）

用户将优先级改为容量/并发，接受约15 tok/s。保持生产W8不变；只使用
隔离的W4 runtime、8002、real weights、TP4+EP、FULL graphs；MTP配置
按轮次标明，关闭MTP的隔离测试不能当成15 tok/s部署通过。
所有配置均不超过用户的 `.965` memory-utilization 上限。

## 已测单请求基线

- byte-mask + affinity：短题median14.966，23.4k median15.097 tok/s。
- 每组3题、每题512输出token，finish=length；23.4k cold TTFT305.102秒。
- 权重19.382 GiB/rank；matched W8为33.040，节省13.658 GiB/rank。
- 旧启动参数32768/seq1/.90/KVfraction .65；planner报307627 tokens。
  这是单请求模式下的配置估计，不是307k上下文实测。

## 第一轮容量启动：失败，不能据此宣称262k或双160k

`capacity-r1/server-capacity-r1.log.gz` 无损保存完整启动失败（含原始CR、
上游日志拼写，不修饰诊断文本）：

- native max262144，seq2，.94，KVfraction .80，FULL `[1,2,3,6]`。
- 权重仍19.382 GiB/rank。
- 多组cache预算系数 **0.2653**：5444608 / 20521984 bytes/block。
- 调整后advertised KV仅 **4.60 GiB**。
- 单个262144请求需要 **10.45 GiB** advertised KV，启动被拒绝。
- planner给出estimated maximum114432 tokens，非实测最大值。
- 进程已自行退出；NPU无剩余任务，W8 launcher SHA未变。

根因：`NPUModelRunner310.__init__` 将prefix Mamba state tier限制在
`max_num_reqs == 1`；第二个request关闭compact state。310P也不能将
NZ attention和contiguous Mamba state叠放到upstream shared backing，
因此worker为所有per-layer buffers缩减planner预算。不能仅删除预算缩放，
否则实际分配可能超预算；也不能仅删除single-request guard：active-column
逻辑读取request0，必须先实现/测试各request的独立状态映射及生命周期。

这不是W4节省失效，而是当前runtime在单/多请求间切换了state布局。
下一项内存优化应针对multi-request compact Mamba tier，不是更低位权重。

## 第二轮：启动及单请求通过，双请求 MTP k2 失败

无runtime源代码改变，只把max-model-len降为98304，其余同第一轮。
`capacity-r1/qwen38-w4-capacity-*.sh` 保存启动与自动测试脚本。
顺序：三个real-weight smoke → CPU affinity → 单/双短prompt512输出
→ 双8k passcode recall → 双48k（48640输入+512输出）。任一gate失败停止
后续压测，服务健康则保留，不自动重启/停止。

- 启动通过，planner报113301 tokens；warmup后各rank allocated约33.58、
  reserved34.02 GiB，graph约0.94 GiB。这仍非实测长窗上限。
- 三个真实smoke通过；单请求256输入+512输出，finish=length，
  **15.936 tok/s**、TTFT2.885s、MTP draft/accept=390/318、无preemption。
  这是一个样本，不能替代上述三题median。
- 双请求各自32-token warmup通过；真正同时提交256输入+512输出时，
  第二次MTP draft发生 **2 vs256 rows** shape mismatch，所有rank退出。
  `qwen4_exp/model.py:549` residual combine只是报错点；调用栈来自
  `llm_base_proposer.py:1587` 的后续draft。该分支仍用prefill
  `num_input_tokens=256`读取hidden buffer，但attention metadata是2个请求。
- scheduler dump是两个请求各128个prefill tokens；尚未到双decode，
  不能宣称并发吞吐通过。双8k及双48k测试均未执行。
- `capacity-r1/server-capacity-r2.log.gz`、`capacity-sequence-r2.log.gz`
  和相邻JSONL保留原始失败；服务自行退出，未发生NPU OOM。

## 第三轮：MTP k1 混合 batch 失败

同98304/seq2/.94/fraction .80，仅改MTP k1及FULL `[1,2,4]`。
没有修改runtime源文件。三个smoke及单请求通过，256输入+512输出
**14.000 tok/s**，draft/accept=273/238。双请求warmup通过，真正并发
时GDN抛出 `NotImplementedError: Qwen4Exp GDN mixed speculative/non-speculative batches`。
双8k未执行；服务自行退出。本轮不是15 tok/s k2的等价配置。
原始日志及JSONL见 `capacity-r1/*r3*`。

## 第四轮：修复多请求 compact Mamba layout

实现不再以request0的进度映射所有行：

- 各request按独立computed/scheduled长度、上一running state列选择当前及
  上一候选窗口；所有请求live IDs的并集在LRU eviction前统一保护。
- slot按全局block ID而非batch行绑定；condense、换行、共享prefix不串状态。
- scheduler原始IDs不变；forward、pre-copy及postprocess使用同一compact映射。
- finished/preempted/resumed/new request清理旧running列；fresh/recycled ID
  失效后再处理CoW；忽略block-table行尾垃圾，避免误拷贝其他cache group。
- 64-slot共享pool不乘request数；若并发/speculation要求更大则按
  `max(64, 1 + max_num_reqs * 2 * (1 + num_spec_tokens))`扩展。
  worker先扣除固定Mamba pool，再计算attention KV预算，未绕过memory上限。
- 单请求非prefix路径保持原逻辑；没有新增环境变量或hot-path设备`.item()`。

125个定向CPU UT通过（另有2 subtests）；覆盖长/短窗口、候选回滚、
LRU、CoW、复用ID、row reorder、生命周期及分配/预算。
新增真实NPU state-tier回归：2/4窗口各96步、FP16 conv + FP32 temporal，
强制超出63 resident slots，校验spill/restore、CoW及recycle后的逐元素一致性。

硬件隔离配置：262144/seq2/.94/fraction .80，FULL `[1,2]`，**不启用MTP**。
真实HTTP顺序：三个smoke → affinity → 双256+256 → 双8k独立passcode。
启动及三个真实smoke通过；每rank固定Mamba pool **941359104 bytes**
（0.8767 GiB）。planner报 **472599 tokens / 262144单窗1.80x**；warmup
allocated25.41、reserved25.60 GiB，graph0.40 GiB，权重18.69 GiB。
此轮无MTP，不能把与r2的全部容量/权重差值归因于compact修改。

双256输入+256输出均finish=length，running峰值2，decode overlap30.071秒，
无preemption；每窗口约8.45 tok/s，含TTFT总吞吐14.323 tok/s。
日志记录两token **FULL** runtime执行（例如count84），不是仅捕图成功。
双8k recall也通过：各自独立passcode均正确，cached_tokens各7680，
finish=stop，running峰值2，decode overlap1.266秒，preemption=0。
NPU tier测试 **2 passed（20.91s）**：2/4窗口各96步，历史checkpoint
超出63个resident slots后恢复，FP16/FP32逐元素完全一致，CoW及recycle通过。
该合成state-tier测试不是4-session模型吞吐测试。

服务保留在8002（API152592，engine153295，workers153648–153651）；
W8 home launcher hash未变。`capacity-r1/*r4*`保存脚本、原始日志、
smoke/并发JSONL、affinity及源文件hash。262k是已启动的配置上限，
472599是planner共享容量估计；**尚未实测双160k或单262k**。
MTP k1/k2的独立阻塞尚未修复，不宣称多窗口15 tok/s已达成。

校验限制：当前本地vLLM main环境的Qwen全套为941 passed/19 failed/4 errors，
主要是旧CPU fixture仍patch已移除的logits-processor TP函数，以及缺少
current-vLLM-config context；在独立导出的修改前HEAD中也复现assembly与
registration失败。不以历史957 passed代替本轮结果。定向layout/runner/worker
125 UT及HTTP client 6 UT通过。scoped manual hooks除既有
`csrc/torch_binding_meta.cpp:655` symbolic-meta失败外通过；未修改该文件。

复现：向同一隔离runtime部署本commit的3个runtime源文件后，按顺序运行
`qwen38-w4-capacity-server-r4.sh`、`qwen38-w4-capacity-sequence-r4.sh`、
`qwen38-w4-compact-tier-check-r4.sh`。脚本使用host绝对路径；不改生产W8。
EP及FULL graph已验证；MTP多请求失败如上；flashcomm1关闭，文本模型
隔离测试不覆盖multimodal；没有dummy权重或编造accuracy-eval指标。
128k/bs16尚未测：当前W4路由算子上限8 decode tokens，且本轮优先双窗口。

`tools/qwen4exp/benchmark_capacity.py` 直接提交精确token IDs。
各session独立随机prefix；长测试先各自warm再并发，避免cold prefill排队
被混入稳态decode。记录usage、finish、decode overlap、running峰值、
KV使用峰值、preemption和MTP draft/accept。共享同session的warm prefix
是有意设计，不是跨session共享容量。合成recall不代表完整长文本accuracy。

当前MTP k2每请求3个verification tokens，双请求6 tokens在设备路由
上限8 tokens内。四请求需要12 tokens，当前W4最多80 routes
（10 experts/token），不能直接捕图；四请求k1是另一配置，尚未测。

## 为什么W4未快于W8

当前是W4A16，不是W4A8。专家权重的packed nibbles需要解包、按128组
scale/offset反量化，然后执行FP16 Cube matmul；生产W8A8使用native
INT8投影。节省存储带宽不等于消除解包、搬运和调度成本，MTP接受率
也会影响最终吞吐。这里是本runtime的实现结果，不是4-bit的一般上限。
