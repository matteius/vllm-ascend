# Qwen3.8：310P 原生 INT4 后端

## Introduction

`cube_310_int4_a8` 是独立的 W4A8 专家投影后端。权重始终以 INT4
参与 Cube 乘法，累加为 INT32；每个动态量化的 INT8 激活拆成两个 INT4
乘积。它不会在计算时把权重展开成 FP16。输出、路由器、共享专家、注意力及
原有 MTP 权重策略保持既有配置。

本实现复用 [离线 W4 检查点](Qwen3.8-Flash-Next-W4-Offline.md)；
它改变激活精度，必须显式获得模型量化策略许可。

## Supported Features

| 配置 | 选择结果 |
| --- | --- |
| W8A8，无 W4 元数据 | 既有 INT8 路径 |
| W4，`activation_quantization=float16` 或省略 | 既有 W4A16 路径 |
| W4，`int8_per_group` 且显式原生后端 | 支持设备/形状检查通过后使用原生 INT4 |
| `backend=auto` | 在模块创建、权重布局选择与图捕获前决定；未完成晋级门槛时保留 W4A16 |
| 显式原生后端不受支持 | 启动报错，不静默回退或改精度 |

支持 Ascend 310P、G128、K=256…2560、N=128…5120，且 K/N 为 128 的倍数。
激活准备要求连续、有限的 FP16 输入，最多 5120 行。decode/MTP 的直接路由接口最多
80 行，接受 INT32 专家 ID；负数及越界 ID 是 peer 路由，每次重放均写零。
分组接口接受 INT64 累积行边界。两者是同一算子的明确输入契约。

调度使用 M16/N64 decode、M32/N64 稀疏 prefill，以及 M128/N64 密集
prefill。量化与打包一次处理 8 个 G128 组；prefill 在 top-k 展开前复用激活准备。
L1 保留完整 K 的打包权重切片，L0B 双缓冲预取；scale/zero-point 修正使用
向量运算，不逐行读取 GM 标量。

## Environment Preparation

本次使用现有四张 310P3、固定 CANN/PyTorch 环境和独立源码副本。
没有验证 A2/A3 Docker 镜像，不应将该算子包安装到其他芯片环境。
同时编译 `qwen_w4_a8_pack_v310`、`qwen_w4_a8_int4_matmul_v310`
及 PyTorch bindings；保留原 W4A16 算子包供独立对照。

```bash
cd csrc
bash build.sh --pkg \
  --ops=qwen_w4_a8_pack_v310,qwen_w4_a8_int4_matmul_v310 \
  --soc=ascend310p --vendor_name=native_int4_w4a8
```

增量构建时，修改算子输入 dtype 需要重新生成 op info、编译参数与内核。
本次发现旧的生成文件会被复用；仅重新链接 host 库不足以更新设备二进制。

## Deployment

保留检查点其他量化字段，仅覆盖模型级别的两个字段：

```json
{
  "backend": "cube_310_int4_a8",
  "activation_quantization": "int8_per_group"
}
```

完整 HF override 及本次服务器命令保存在
[实验目录](../../../../artifacts/native-int4-20260927/)。
`serve.sh native` 与 `serve.sh baseline` 使用同一检查点、TP4+EP、MTP2、
FULL_DECODE_ONLY、capture sizes `[1,2,3,6]`、512-token prefill chunk 和两个会话。
脚本包含本次主机的绝对路径，仅用于复现实验，不是生产启动配置。
实验服务使用已释放的 8001 端口；W8 的启动配置独立保留。

## Functional Verification

```bash
curl http://127.0.0.1:8001/v1/models
curl http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-w4-experimental","messages":[{"role":"user","content":"17 * 19 = ? Reply with the number only."}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

设备回归包括 pack 的随机、零、次正规数、离群值与舍入边界，分组边界缓存尾部，
以及输入和专家 ID 同时改变的图重放。独立参考使用旧原生 INT4 调度及 CPU 量化。
这证明整数分解和设备调度一致，不等价于模型精度测试。

## Accuracy Evaluation

固定抽取 MMLU 57 个科目各 4 题，共 228 题，zero-shot、temperature=0、
关闭 thinking。仅完整结束的单个 A/B/C/D 答案计分。保留题目 ID、prompt 哈希、
逐题答案与结果，不将这个抽样结果写成官方完整 MMLU 分数。

使用作者数据包和 `reproduce-mmlu.py` 可按 manifest 精确重建样本。
`tools.qwen4exp.evaluate_w4a8_http` 对已运行的模型评估，不修改服务器配置。
同一 228 题样本：W4A16 为 206，最终原生实现为 203；因此原生只接受显式实验
配置，`auto` 不晋级。原生整数乘法通过数值测试，不意味着 INT8 激活量化比 FP16
激活更准确。另一个 affine 修订为 204，但速度更低，未采用。

## Performance

完整模型速度必须与 W4A16 在相同硬件、亲和性、MTP 和图配置下比较。
算子或单层速度不能代表 tokens/s。原生 INT4 实测结果、MTP 接受率、短/长上下文
吞吐及限制统一保存在实验报告和原始证据中。

| MTP2，单流 decode median | W4A16 | 原生 INT4 |
| --- | ---: | ---: |
| 短上下文 tok/s | 15.18 | 17.18 |
| 约 23.4k 上下文 tok/s | 14.44 | 16.69 |

每项为 3 次 512-token 请求；长上下文计时请求复用了预热前缀。
包含请求首 token 延迟的 aggregate end-to-end 分别为 15.03→17.23 和 14.42→16.24
tok/s。长上下文冷 prefill 单独记录，不混入上述 decode tok/s。
实际配对汇总见 [实验报告](../../../../artifacts/native-int4-20260927/README.md)。
