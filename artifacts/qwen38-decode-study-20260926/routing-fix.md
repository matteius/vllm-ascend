# Grouped MTP 路由修复（尚未测试）

用户要求修复代码，但暂不测试。本次只修改本地候选生成器、helper、回归测试
和说明文档；未执行测试、lint、候选生成、NPU 调用或远端部署，未重启线上服务。
历史 [failure-audit.md](failure-audit.md) 和 JSON 证据对应修复前版本，保留原样。

## 代码变更

- `grouped_draft.py` 新增共享的 `route_local_experts`：每 rank 128 个真实
  专家加 1 个虚拟 peer 路由槽，传 `expert_num=129` 与完整范围 `[0,129]`。
  所有非本地 ID（包括 rank 0 的高全局 ID）统一映射到合法的 128。
- 路由返回后只保留前 128 个累计计数，并转换为 matmul 要求的 INT64。
  两次 matmul 不接收虚拟专家计数，也不添加虚拟权重；末尾 peer 输出仍用
  `where` 清零，避免未写行中的 NaN/Inf 污染结果。
- `prepare.py` 为新建 grouped/combined 副本同时修正 draft 和 target 的
  小 batch fused routing，使用同一 helper。target 的大 prefill 路径保持原实现。
  原实验副本和基线 SHA256 固定值保持不变。
- grouped draft 默认保留原 eager callback，包括 shared expert 和 TP all-reduce。
  `--draft-eager` 保留兼容；完整捕获需要显式 `--draft-capture`。
  manifest 记录 callback 选择和 target/draft 路由修复状态。

## 已编写、未执行的回归测试

`tests/ut/qwen38_1m/test_decode_study.py`：

- CPU routing mock 严格拒绝声明范围外的 ID，包括旧版 sentinel 128。
- matmul mock 校验累计计数长度与真实专家权重数量一致，防止虚拟槽泄漏。
- 128 专家、10 路由、2560 hidden、四个 TP offset、1/2 token 的路由契约，
  覆盖全本地、全远端、混合以及连续路由变化；这不等同于设备 graph replay。
- 128 专家的小矩阵数值参考、全空 rank、peer 行 NaN 清零。
- MTP 默认保留 callback 和 all-reduce、显式捕获选择、target patch 的范围及
  缺失/重复锚点拒绝行为、CLI 默认参数。

## 后续验证状态

全部为 **NOT RUN**，包括现有 CPU 测试和新增测试。上一轮审计的“20 项通过”
不能用于本次修复。未声称解决了原停滞、通过 NPU 验证或取得吞吐提升。
129 个 routing slots 对当前 CANN 的真实执行与图重放仍需验证。

待用户恢复测试后，先运行 CPU 回归，再验证单 NPU 路由和 matmul，随后
TP4 保留 eager callback；完整 draft capture 单独隔离。对应的候选准备命令
默认会保留 callback；本次没有运行该命令：

```bash
python3 -m tools.qwen38_decode_study.prepare \
  --source /srv/ai/src/qwen38-decode-study-20260926/baseline \
  --destination /srv/ai/src/qwen38-decode-study-20260926/grouped-routing-fixed \
  --launcher /srv/ai/src/qwen38-decode-study-20260926/original-launcher.sh \
  --arm grouped
```

仅在包含本次修复的工具目录执行上述准备命令；它只生成新副本，不启动服务。
