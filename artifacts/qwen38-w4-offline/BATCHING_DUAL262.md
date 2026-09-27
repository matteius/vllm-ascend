# W4 多窗口 MTP / 262k 容量修复（2026-09-27）

## 目标与隔离

在 compact Mamba 修复 `b7dafff1a` 上修复多窗口 MTP，并评估两个独立
262,144-token 窗口。仅改 private W4 runtime / loopback `8002`；W8 启动脚本、
runtime、权重不改。TP4 + EP，MTP k2，FULL_DECODE_ONLY [1,2,3,6]，
chunk512，max-seqs2，utilization .94（不超过 .965），KV fraction .80。
沿用原环境，不升级 transformers，不启用 CPU speculative fallback。

## 根因和修改

1. r2 的第二次 draft 仍使用首轮 prefill 的 256 行，但 metadata 只描述两个
   request。修正非 FULL 的 MTP 后续 draft 为每 request 一行；FULL 仍保留
   capture bucket。单 request 的广播也不能作为正确性依据。
2. r3 的 GDN 明确拒绝 mixed prefill/spec decode。按 spec/non-spec token
   indices 分开执行 native conv / recurrent 或 chunk delta，再按原 token
   顺序合并。metadata views 每 step 缓存、跨层共享；不改全局 forward metadata。
3. compact prefix state 是固定 resident pool，而 attention 的 planner page
   被补齐至 Mamba 大小。310P allocator 的 K/V 和 QSA index 是无此 padding 的
   独立 tensor。预算应扣除固定 pool，再按实际 attention page 字节换算为
   scheduler 的 virtual page 字节。包含 QSA index side cache；不增加 utilization。

r5 首次预算调整未消除 attention padding，仍只有 408,613 shared tokens；
不能据此宣称 dual262。r6 加入 real_page_size_bytes 修正。日志中的 planner
budget 不是 NPU 实占，须与 allocated/reserved/npu-smi 对照。

## 已验证证据（r5）

- 三个 real-weight HTTP smoke 正确（包含 1–50），不是 startup-only。
- single 256+512：16.018 tok/s；dual 256+256：10.349 / 9.838 tok/s，
  decode overlap 24.149s，draft424 / accepted299，零 preemption。
- dual8192 recall：两个独立 passcode 均正确，cached7680 各自。
- server log 有 mixed step（128 prefill + 3 decode tokens）及 FULL6 replay。
- NPU native mixed vs separate：4 passed；17/70 prefill tokens、初始状态有/无、
  speculative accepted=2；输出和全状态一致，未涉及 slots bit-exact 不变。
- 最终 focused CPU：156 passed + 2 subtests；新增 batching/client：36 passed。
- 扩展 GDN/MTP CPU：114 passed（不包含过期 fixture 文件）；另一次包含
  test_qwen4exp_mtp.py 的运行有7个既有 fixture 失败，均为 upstream 删除
  logits_processor.get_tensor_model_parallel_world_size；不宣称全库绿。

原始证据在 `capacity-r1/` 的 r5 JSONL / gzip logs；native 测试
`mixed-gdn-r5-tests.log.gz`。本轮没有完整任务准确率评测；synthetic passcode
不代表长上下文 coding accuracy，也不承诺 temperature=0 bitwise deterministic。

## r6（短窗口通过，完整长测已中止）

actual attention=1,830,400 bytes/block，planner=5,444,608 bytes/block。
限制 rank 规划9,552 blocks，shared容量1,215,533 tokens，262144理论并发4.64x。
实际仍只启用max-seqs2，不声称4-session支持。allocated36.60GiB、reserved37.07GiB；
npu-smi每rank约39,700–40,674MiB。planner显示约48–49GiB是virtual budget，
不是在46GiB NPU上超配；实际attention预算约16.28GiB，fixed state约0.899GiB。

重复验证已通过：三个HTTP smoke；single256+512为16.049 tok/s；dual256+256
为11.267/11.111 tok/s，overlap22.633s；256/8192非对称batch均完整生成256tokens；
dual8k两条独立passcode均正确、各自cached7680；所有probe零preemption。
FULL6 replay在运行时计数表中确认，不仅是capture配置。

尚未据此认定 dual262 成功。完整长测脚本`qwen38-w4-dual262-r6.sh`
会先独立warm两条261632-token前缀，再并发每条生成512tokens（含输出总262144）；
要求两条完整结束、decode overlap、无preemption。于13:12 UTC启动，
tmux `qwen-w4-dual262-r6`，日志`dual262-r6.log`和`capacity-r6-dual262.jsonl`。
长前缀还会增加host Mamba snapshots。脚本监控MemAvailable和swap增长：
低于12GiB可用或swap增长超过2GiB，仅取消测试client并断开请求，保留server。
NPU allocation充足不等于host容量、长上下文准确率和吞吐已经验证。

13:31:34 UTC host guard 在 swap 增长2,190,352KiB时取消测试client。
第一条cold prefill约完成113,664tokens，第二条尚未开始，无完整窗口结果；
当时MemAvailable=61,376,700KiB，未出现NPU OOM或preemption。
这是预设保守swap增长阈值触发，不是dual262硬件容量上限的证明。
server在请求断开后恢复idle；用户随后选择直接用Kilo测试，不重跑长测。
最终guard日志与未完成JSONL已保存至本目录的`capacity-r1/`。
`server-capacity-r6-gates.log.gz`是短窗口gates完成后的日志快照，不含长测最终结果。

## 版本与质量检查

- vLLM：`3ab5dda29acabea01f6a63d0806bdbbb4a27bde5`；CANN9.1.0，
  原W4 test venv和byte-mask-r1算子，未更换依赖。
- private runtime三源文件SHA256与本地一致：model.py `a2ef381d7b7c22cf`，
  llm_base_proposer.py `26d81a89831062f2`，worker.py `173a392d682b0d28`。
- W8 home launcher SHA256仍为
  `ae81d75eec7516fc6f455a3a071e52410ba6c477c34e0114df647f4c125c77df`。
- scoped manual pre-commit（含markdownlint/shellcheck）通过，仅跳过既有、
  未改动的check-symbolic-meta问题；`git diff --check`通过。
- 未运行完整nightly/full-repo suite；不把既有fixture失败隐去或称全库通过。

## 运行与验证

host `matteius@192.168.53.187`，taskdir
`/srv/ai/src/qwen38-w4-hardware-20260927`：

```bash
bash /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-capacity-server-r6.sh
# 另一个终端：等待 smoke 后应用经过 PID-tree 校验的 affinity 并运行 gates
bash /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-capacity-sequence-r6.sh
curl -fsS http://127.0.0.1:8002/health
```

启动参数见同目录保存的 launch-r6.sh。证据输出不可覆盖；再次启动应使用新的
run-id 路径。不使用 production home launcher。若 native/MTP gate 失败，保留
失败证据并修复，不把禁用 MTP 的服务器标为成功。

## 特性界限

MTP k2 / TP4+EP / FULL graph：r5和r6短窗口real-weight均已过。
flashcomm1 未启用，本轮不变；language-model-only，multimodal 未验证。
目标 max context=262144（含输出），max-seqs2；128k+bs16 不在本轮支持范围，
W4 routed kernel 小 batch 优化最多八个 token，不能据此放开更大验证 batch。
dual160 不是已确定的硬件极限；双262仍需要实际长窗口验证。
