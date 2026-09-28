# Grouped MTP 故障代码审计

## 结论与范围

**不能判定 grouped MTP 的故障不可避免。已发现可修复的路由契约错误，
但尚未证明它触发了这次停滞。** 图捕获边界也随专家计算一起改变，原实验
未隔离这两个变量。应先修正路由，再分别验证 eager 和 graph。

本次只读取故障副本、日志、已安装 CANN 源码和对应 torch-npu 源码，
运行本地 CPU 检查；没有修改或重启线上服务，没有执行 NPU 算子。
线上仍为已验证约 19.22 tok/s 的 affinity 版本；审计结束前 `/health` 返回成功。
这不是修复交付，也没有新增 grouped 性能或正确性结论。

审计对象是远端 `qwen38-decode-study-20260926/grouped`，不是本地较旧模型实现。
远端 grouped helper 与本地 helper 的函数 AST 相同，只有说明文字不同。
源码定位、版本和 SHA256 见 [failure-audit-sources.json](failure-audit-sources.json)。

## P1：非本 rank 专家被传成非法 expert ID

位置：[grouped_draft.py](../../tools/qwen38_decode_study/grouped_draft.py)，
`grouped_routed_experts` 的 `local_ids` 构造及 `npu_moe_init_routing_v2` 调用。

- 当前配置：512 个专家、TP4，每 rank 128 个本地专家，每 token 选择 10 个专家。
- 非本地专家统一编码为 `128`，同时给算子传 `expert_num=128`。
- CANN V2 在输出专家计数/累积计数时要求 ID 位于 `[0, expertNum-1]`；
  这里合法范围是 `0..127`，不是 `0..128`。
  依据：[CANN 9.1 V2 接口契约](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnMoeInitRoutingV2.md)。
- `active_expert_range=[0,128]` **不能在此路径过滤 sentinel**。
  torch-npu 报告的源码版本对应 op-plugin `a365032e8d96befd8e554afba3d60f342a4e6578`。
  该版本的 310P 分支直接调用 `aclnnMoeInitRoutingV2`，强制累积计数 flag 为 1，
  不向内核传递 active range 或 `row_idx_type`。
  依据：[固定版本 wrapper，第 268–310 行](https://github.com/Ascend/op-plugin/blob/a365032e8d96befd8e554afba3d60f342a4e6578/op_plugin/ops/opapi/MoeInitRoutingV2KernelNpuOpApi.cpp#L268)。

已安装源码中的具体危险路径：

```text
ops_transformer/ascendc/moe_init_routing_v2/
  moe_init_routing_v2.cpp:135        310P 分支，tiling key 20000
  moe_v2_init_routing_fullload.h:195  ComputeExpertTokenCountOrCumsum
  moe_v2_init_routing_fullload.h:217  SetValue(lastExpertId, ...)
  moe_v2_init_routing_fullload.h:224  最后一个 expert ID 直接作为写入索引
  moe_v2_init_routing_fullload.h:347  按 expertNum 分配计数 UB
```

128 个 INT32 的对齐缓冲区恰好为 512 字节，没有供索引 128 使用的空间。
在该 full-load 路径中，只要存在 sentinel，最终就会写入逻辑和物理分配之外。
事后对 matmul 输出做 `where(..., 0)` 无法防止此前的路由缓冲区越界。

CPU 拦截未修改的 helper，并按上述内核循环模拟写入索引，结果见
[routing-contract-audit.json](routing-contract-audit.json)：

| 输入情况 | 实际传入的 ID 契约 | full-load 越界写索引 |
| --- | --- | --- |
| rank 0 全本地 | 合法 | 无 |
| rank 0 混合本地/远端 | 非法 | 128 |
| rank 0 全远端 | 非法 | 128 |
| rank 1 全远端 | 非法 | 128 |

这是 **CPU 调用参数检查和内核控制流模型**，不是 NPU 故障复现。
故障时未采集实际 tiling key、算子轨迹和输入，不能宣称越界路径当时一定执行。
torch-npu 源码按安装版本的 git revision 定位；不能排除未报告的构建补丁。

重要反证：稳定 target 的远端 `moe.py:233–262` 已使用类似约定；rank 0 甚至
保留超出本地范围的全局 ID，其他 rank 使用 128。这是共享的潜在缺陷，
不是 grouped draft 独有的新错误。稳定运行并不能使非法输入变为合法，
但它降低了把此缺陷直接认定为此次停滞根因的把握。

修复方向：使用契约合法的设备路由。一个候选是分配第 129 个虚拟专家计数槽，
以 `expert_num=129`、完整范围 `[0,129]` 路由，随后只把前 128 个累计边界
交给真实权重的 grouped matmul，并保留末尾 peer 行清零。虚拟专家没有权重，
不能把第 129 个边界直接传给只有 128 个专家的 matmul。该方案仍需真实 310P
形状、全空 rank 和 graph replay 验证；也可采用已有设备排序/计数路径。

## P2：CPU mock 把非法输入定义成了成功行为

位置：[test_decode_study.py](../../tests/ut/qwen38_1m/test_decode_study.py)，
`ReferenceOps.npu_moe_init_routing_v2`。

mock 对全部 ID 排序，但用 `(ids < i + 1).sum()` 生成每个合法专家的累积计数，
因此它自动忽略 sentinel，绕过了真实接口的输入契约和 full-load 直接索引行为。
测试使用每 rank 两个专家，也没有覆盖真实的 128 专家对齐边界。
现有 20 项 CPU 测试全部通过，只能证明其参考模型下的数学行为。

验证命令：`python3 -m pytest --confcutdir=tests/ut/qwen38_1m -q
tests/ut/qwen38_1m/test_decode_study.py`。文档/证据文件的 manual pre-commit
检查通过，只有仓库既有的 `check-symbolic-meta` 在未改动的
`csrc/torch_binding_meta.cpp:655` 报错；没有修改该文件。

修复验证必须增加严格契约检查，以及真实 `E=128, K=10, hidden=2560` 的
混合路由、全本地、全远端、连续改变路由的测试。不能只重复现有 mock。
权重格式测试注入 identity formatter，不能证明 NZ view 的设备行为。

## 待隔离：同时移除了整个 draft block 的 eager 边界

位置：[prepare.py](../../tools/qwen38_decode_study/prepare.py)，`patch_mtp`。
生成后的远端 `mtp.py:187–225` 在 NPU 上直接返回 `_forward_grouped(x)`，
绕过原 `capture.add_eager(run_experts_eager)`，因此进入图的内容包括：

1. 路由、两个 grouped matmul、SwiGLU 和 gather。
2. shared expert 计算。
3. FP32 tensor-parallel all-reduce。

原 callback 在 capture 外执行，并把结果复制到供后续图段使用的输出缓冲区。
移除它改变了临时张量分配、算子 workspace、collective 和 replay 的执行环境。
它不是单纯消除 Python 专家循环，因此无法凭当前 A/B 结果区分这些原因。

这属于未验证的风险，**不是已经证明的 graph/HCCL bug**。检查到的 all-reduce
链路最终调用 `dist.all_reduce`；没有证据指向另一个 raw-pointer communicator。
日志中三个 rank 持续忙、一个 rank 空闲，与 collective 等待相容，但同样可能
由某个 rank 较早卡在计算算子造成，不能据此给 HCCL 定责。

`combined-eager` 曾准备为隔离实验，但在故障后的设备初始化阶段就被阻塞，
没有进入有效推理；这不能算作 eager grouped 也失败的证据。

## 已排查但没有证据支持的结论

- 不能依据 A2/A3 的 V3 routing 文档认定此处一定返回 `-1` gather 索引。
  310P 分支调用的是 V2，dropless 路径构造完整逆排列。
- 没有从已读 grouped kernel 循环中发现“全空 rank 必然死循环”；合法全零
  累积计数会跳过专家计算。仍需设备验证其完整执行和 replay 行为。
- `persistent=False` 只影响 state dict；packed 权重仍由注册 buffer 持有。
  不能据此宣称权重已被释放或有确定的 use-after-free。
- W8A16 → W8A8 动态激活量化确实改变数值与可能的接受率，但数值差异本身
  不是非法设备地址的解释，也未测得 grouped 的持续吞吐收益。

## 故障时间线与证明边界

`grouped/serve.log` 显示六个仅生成 2–4 token 的文本请求完成。
19:06:17 UTC 是最后一个非零吞吐统计窗口；19:06:27 已为零。
19:07:15 出现 shared-memory broadcast 等待超时，19:07:35 收到 SIGTERM。
随后 vLLM 的 `mode=abort timeout=0s` 强制终止 EngineCore。
保存的 dmesg 在 shutdown/recovery 附近记录 rank 0 的非法设备地址。

因此应区分“首次停滞”和“终止后的设备内存错误/恢复失败”。没有逐算子证据
将它们连成唯一因果链，也没有证据证明必须永久放弃 grouped MTP。
原始时间线与恢复信息见 [failures.json](failures.json) 和 [recovery.json](recovery.json)。

## 下一次隔离顺序（尚未执行）

1. 修正非法 ID，严格检查计数边界、逆排列、peer 行清零和输出有限性。
2. 在可用测试窗口先执行单 NPU 的真实尺寸 routing + 两次 matmul，
   覆盖全空和混合路由；先 eager，再持续改变 ID 的 graph replay。
3. TP4 保留原 eager callback，只替换专家计算，验证输出和接受率。
4. 分开测试专家计算捕获与 all-reduce 捕获，记录每个 rank 最后完成的阶段。
5. 最后运行长输出、工具调用、性能 A/B；原 19 tok/s 服务配置保留作恢复基线。

推荐继续调查；当前证据足以要求修正代码并重新隔离，不足以承诺性能增益或
宣称已修复原故障。
