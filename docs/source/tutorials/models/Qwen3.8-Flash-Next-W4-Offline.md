# Qwen3.8 Flash Next: offline W4 candidate

## Introduction

This is an **experimental alternate checkpoint**, not a replacement for the
validated W8A8 + MTP + decode-graph deployment. It quantizes only the 48 target
layers' routed experts directly from the original BF16 checkpoint. The router,
shared experts, attention, vision, embeddings, PLE and MTP checkpoint tensors
remain floating-point (BF16 is exported as FP16 for 310P; FP32 stays FP32).

The converter uses **ModelSlim IR asymmetric per-group min/max RTN**, group size
128. It does not perform GPTQ, AWQ, activation calibration, or accuracy evaluation.
The custom checkpoint format is not interchangeable with stock AWQ/GPTQ or the
generic Ascend W4A16 fused-MoE format.

The full offline export completed on 2026-09-26: 1,610 shards, 222,746 tensors,
and 169.1628 GiB of tensor payload including the host PLE table and floating
MTP weights. A later authorized TP4/310P hardware smoke loaded the full checkpoint
and completed three correct arithmetic/counting/Python-output answers. This does not establish production
speed or broad model quality; see the hardware results below.

## Supported Features

| Feature | W4 candidate status |
| --- | --- |
| ModelSlim conversion and packed checkpoint | Offline path implemented |
| Strict loader, signed packing, group zero points | CPU tests and real-weight 310P smoke |
| Contiguous expert-TP ownership, shared-expert TP | CPU numerical tests, including uneven expert ownership |
| PLE disk-backed lazy lookup | Preserved; W4 explicitly selects the standard HF index |
| W8 runtime, MTP and graph defaults | Unchanged unless W4 checkpoint metadata is present |
| W4 NPU inference | Full checkpoint loaded on TP4/310P; three completed correct answers and seven operator regressions passed |
| Experimental W4 Cube projection | Separate group/routed 310P operators; 82 NPU regressions and real-weight TP4 smokes passed |
| W4 ACLGraph | FULL_DECODE_ONLY verified with cube_310_routed; older backends still require eager |
| W4 MTP | k=1 verified with real weights, three correct completed answers and increasing acceptance counters |
| W4 multimodal / flashcomm1 / EPLB | Not validated; language-model-only with existing collectives |
| Long context / concurrent sessions | Not validated for W4 |

Uneven expert ownership is not proof of whole-model TP3/TP6 support: attention,
shared-expert and MTP divisibility constraints still apply.

## Environment Preparation

Use the existing pinned `opensensor/vllm` fork and an isolated worktree containing
this plugin change. Do not upgrade the serving environment, install this worktree
over it, or change the W8 launcher to try this candidate.

Conversion needs Python, CPU PyTorch, safetensors, and the local ModelSlim checkout.
It never imports the model, initializes an accelerator, or contacts the server.
Allow approximately 170 GiB of destination disk space plus temporary working space.
The source must contain its original `config.json`, index, and all shards.
No A2/A3 Docker image is prescribed here: this path targets the existing 310P
environment, and no new container or NPU environment was validated offline.

```bash
MODELSLIM_PYTHON=/path/to/msmodelslim/.venv-cpu/bin/python
"$MODELSLIM_PYTHON" tools/quantization/qwen38_modelslim_w4.py \
  --source /path/to/original/Qwen3.8-Flash-Next \
  --output /path/to/separate/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --group-size 128 --threads 4
```

After an interrupted build, rerun the same command with `--resume`. Resume checks
source/config/settings/tool revision identity and hashes of completed output
shards. It rejects a nonempty unrelated destination and source/output overlap.
The final model index is published only after all tensors are exported. Preserve
`build-journal.json`, `build-receipts/`, and `quantization_provenance.json`.

### Storage contract

Selection is explicit in `config.json` → `text_config.ascend_expert_quantization`:

```json
{
  "format": "qwen4exp_w4a16_group_v1",
  "bits": 4,
  "group_size": 128,
  "symmetric": false,
  "packing": "signed_int4_low_nibble_first_in_axis",
  "backend": "eager_dequant",
  "scale_dtype": "float16",
  "offset_dtype": "int8"
}
```

Each `model.language_model.layers.L.mlp.experts.E.{gate,up,down}_proj`
contains `weight` (INT8 bytes, two signed INT4 values per byte, low nibble first),
`weight_scale` (FP16), and `weight_offset` (INT8 signed zero point).
Packing is along the input dimension. Dequantization is `(q - offset) * scale`,
with one scale/zero-point pair per output row and input group of 128.

The resident routed-expert bank is `0.5 + 3/128` bytes per weight: **58.8867 GiB**
for this model, excluding all other tensors, caches and runtime workspace.
That number is not a whole-model fit or context-capacity guarantee. PLE remains
on disk/host and is not counted as NPU expert memory.

## Deployment

**Do not use the W8 production launcher for this checkpoint.** Its generic
quantization flags are incompatible. The following is the experimental
hardware-smoke configuration, not a production serving profile. It does not
stop another process and uses port 8002 instead of the production port.

After separately activating the pinned Ascend environment and sourcing its CANN
setup, from the isolated plugin worktree (or its installed test environment):

```bash
export SOC_VERSION=ascend310p1
export VLLM_ASCEND_ENABLE_310P=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export TASK_QUEUE_ENABLE=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_ASCEND_KV_CACHE_FRACTION=0.65
python -m vllm.entrypoints.cli.main serve /path/to/separate/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --served-model-name qwen38-w4-experimental \
  --host 127.0.0.1 --port 8002 \
  --dtype float16 --tensor-parallel-size 4 \
  --no-async-scheduling --disable-custom-all-reduce \
  --max-model-len 32768 --max-num-batched-tokens 512 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 --language-model-only \
  --enable-expert-parallel --enable-ep-weight-filter \
  --reasoning-parser qwen3 --mamba-cache-mode align --enable-chunked-prefill \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --cudagraph-metrics --enable-logging-iteration-details \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}' \
  --hf-overrides '{"text_config":{"ascend_expert_quantization":{"backend":"cube_310_routed","bits":4,"format":"qwen4exp_w4a16_group_v1","group_size":128,"offset_dtype":"int8","packing":"signed_int4_low_nibble_first_in_axis","scale_dtype":"float16","symmetric":false}}}' \
  --limit-mm-per-prompt '{"image":0,"video":0}'
```

Omit `--quantization ascend`:
the model-specific metadata selects W4, and generic quantization is rejected.
Never raise memory utilization above the established 0.965 cap. The model config
advertises 262,144 positions; W4's usable maximum has **not** been measured.

The command requires an isolated installation rebuilt with both
`QwenW4GroupMatmulV310` and `QwenW4RoutedMatmulV310` plus matching Torch bindings.
It leaves the checkpoint's default and the W8 runtime unchanged. Detailed
commands and build provenance are in `artifacts/qwen38-w4-offline/CUBE_KERNEL.md`.
The `cube_310_tiled` variant losslessly re-encodes nibbles in Cube NZ order at
load time and biases codes/offsets equally. It preserves parameter shapes,
dtypes, byte counts, and the quantization formula; the checkpoint is unchanged.
The matching operator is mandatory because the in-memory byte layout differs.
Neither group-only variant makes full-model graph capture safe while routing
uses the CPU. The new `cube_310_routed` variant reads expert IDs on device and
supports bounded decode graphs up to 80 routes (8 tokens for this top-k=10 model).
Larger prefill uses the existing grouped host route; oversized capture fails
explicitly. Workspace is bounded by `routes*N*K*2 + CANN reserve`, not a
persistent expanded expert bank. For eager isolation, select `cube_310_tiled`
or remove the overrides, add `--enforce-eager`, and remove speculative and
compilation configuration. Do not use `TORCHDYNAMO_DISABLE=1` as graph evidence.

## Functional Verification

Offline artifact audit, after conversion completes:

```bash
"$MODELSLIM_PYTHON" tools/quantization/audit_qwen38_w4.py \
  /path/to/separate/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --report /tmp/qwen38-w4-audit.json
```

This checks all tensor names, shapes, dtypes, byte accounting, untouched-tensor
inventory, and sampled expert reconstruction. It is not full-model inference.

For an explicitly authorized NPU test:

```bash
curl --fail http://127.0.0.1:8002/v1/models
curl --fail http://127.0.0.1:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-w4-experimental","messages":[{"role":"user","content":"Return the integers 1 through 50 in order."}],"temperature":0,"seed":1024,"max_tokens":512,"chat_template_kwargs":{"enable_thinking":false}}'
```

Require a completed correct response and clean worker logs; startup alone is not
a pass. Next compare W8 and W4 on identical held-out coding prompts, prompts that
cross cache-block boundaries, and perplexity/answer accuracy. Longer context and
concurrency require separate gates. Check actual FULL runtime-mode statistics
for two-token verification batches and increasing drafted/accepted counters;
configuration or startup alone is not graph/MTP evidence. No accuracy-threshold CI YAML is
provided yet: there are no real W4 accuracy results from which to set thresholds.

## Accuracy Evaluation

Full-model evaluation is pending. The completed artifact passed a full
header/index audit. Across 45 sampled real expert projections, mean relative
weight RMSE was 0.103625 and mean weight cosine was 0.994629; all sampled
reconstructions were finite. Weight cosine and Gaussian-input errors are diagnostics, not
task-quality evidence. Earlier W8 or other-platform four-bit results cannot be
transferred to this RTN artifact.

## Performance

The original three TP4 hardware requests decoded at approximately **0.23 tok/s**, not
15 tok/s. The reference backend keeps only packed weights resident and
dequantizes selected experts one at a time, with host routing synchronization.
Each rank reported **18.6917 GiB** model-load memory with MTP and graphs disabled.
That is not a measured maximum context or concurrency capacity.

The first accelerated `cube_310` implementation completed the same three
correct smokes at approximately **3 tok/s**, with the same 18.6917 GiB/rank
model memory. This is not production parity: the refreshed W8 + MTP + graph
baseline is 19.073 tok/s at short context and 18.091 tok/s near 23.4k context.
Whole-model graphs/MTP were not enabled in that first run. See
`artifacts/qwen38-w4-offline/CUBE_KERNEL.md` for measurements
and the distinction between raw smoke timing and corrected decode throughput.

The subsequent `cube_310_tiled` backend passes the same three correct smokes
at **4.24–4.28 tok/s**, retaining 18.6917 GiB/rank. Its 51 operator tests pass,
including changed-weight replay and K-batch boundaries. Load-time re-encoding
adds startup work (about 233 seconds for model loading in this run). These
smokes remain eager/no-MTP and do not establish production parity.

Detailed provenance, timings and limitations are in
`artifacts/qwen38-w4-offline/HARDWARE_SMOKE.md`. Hardware regression coverage is
`tests/e2e/nightly/310p/single_node/ops/test_qwen4exp_w4_310.py`.

### 最新 MTP + graph 验证

`cube_310_routed` 已完成 TP4 真实整模型 MTP k=1 + FULL_DECODE_ONLY
验证。三个 smoke 均完整正确，1–50 为 **11.592 tok/s**，Python 输出题为
11.653 tok/s；这些短 smoke 不能代替 sustained coding throughput。
model-load 为 19.3821 GiB/rank，graph capture 报告 0.30 GiB。
82 项 NPU operator/layer 回归与 33 项 CPU/build tests 通过。

首次完整 graph 启动因 NZ shared weights 的 FP32 Cast 失败；修正仅限
routed NPU 后端，改为 resident FP16 operand policy，与 W8 一致。
增加真实 NZ post-load 和改变输入/route 的 replay 回归后，实际服务的
两-token verification batch 显示 `Runtime Mode = FULL`，MTP 接受计数
也递增；不是只验证 capture。最新速度证据与 runbook 在上述 CUBE_KERNEL。

同协议三条 512-token prompts 的 sustained 中位数为：短上下文 **10.722**、
约 23.4k 上下文 **10.660 tok/s**。六题都正常完成，最高温度 80°C。
长前缀冷 TTFT 为 368.799 秒，prefill 仍有明显性能缺陷。下一步是完整
replay 的瓶颈 profile；仍未达到 W8 的 19.073/18.091 tok/s。
不能靠常驻 INT8/FP16 全专家展开
制造 W4 提速；必须保留量化内存收益与动态 replay 正确性。
