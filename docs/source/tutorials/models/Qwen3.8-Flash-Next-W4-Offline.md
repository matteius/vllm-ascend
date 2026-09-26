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
MTP weights. This is an artifact-build result, not an NPU inference or quality pass.

## Supported Features

| Feature | W4 candidate status |
| --- | --- |
| ModelSlim conversion and packed checkpoint | Offline path implemented |
| Strict loader, signed packing, group zero points | CPU tests |
| Contiguous expert-TP ownership, shared-expert TP | CPU numerical tests, including uneven expert ownership |
| PLE disk-backed lazy lookup | Preserved; W4 explicitly selects the standard HF index |
| W8 runtime, MTP and graph defaults | Unchanged unless W4 checkpoint metadata is present |
| W4 NPU inference | Not validated; requires a separate maintenance-window test |
| W4 ACLGraph | Unsupported by the initial eager backend; initialization rejects graph mode |
| W4 MTP / multimodal / flashcomm1 / EPLB | Not validated; leave disabled for the first W4 smoke |
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

**Do not use the W8 production launcher for this checkpoint.** Its quantization
and graph flags are intentionally incompatible. The following is a future
maintenance-window smoke command, not a validated serving profile. It does not
stop another process and uses port 8002 instead of the production port.

After separately activating the pinned Ascend environment and sourcing its CANN
setup, from the isolated plugin worktree (or its installed test environment):

```bash
export SOC_VERSION=ascend310p1
export VLLM_ASCEND_ENABLE_310P=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
vllm serve /path/to/separate/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --served-model-name qwen38-w4-experimental \
  --host 127.0.0.1 --port 8002 \
  --dtype float16 --tensor-parallel-size 4 \
  --enforce-eager --no-async-scheduling --disable-custom-all-reduce \
  --max-model-len 8192 --max-num-batched-tokens 512 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 --language-model-only \
  --enable-expert-parallel --enable-ep-weight-filter \
  --reasoning-parser qwen3 --mamba-cache-mode align
```

Omit `--quantization ascend` and all speculative/graph flags in this first smoke:
the model-specific metadata selects W4, and generic quantization is rejected.
Never raise memory utilization above the established 0.965 cap. The model config
advertises 262,144 positions; W4's usable maximum has **not** been measured.

## Functional Verification

Offline artifact audit, after conversion completes:

```bash
"$MODELSLIM_PYTHON" tools/quantization/audit_qwen38_w4.py \
  /path/to/separate/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --report /tmp/qwen38-w4-audit.json
```

This checks all tensor names, shapes, dtypes, byte accounting, untouched-tensor
inventory, and sampled expert reconstruction. It is not full-model inference.

For the later, explicitly authorized NPU test:

```bash
curl --fail http://127.0.0.1:8002/v1/models
curl --fail http://127.0.0.1:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-w4-experimental","messages":[{"role":"user","content":"Return the integers 1 through 50 in order."}],"temperature":0,"seed":1024,"max_tokens":512}'
```

Require a completed correct response and clean worker logs; startup alone is not
a pass. Next compare W8 and W4 on identical held-out coding prompts, prompts that
cross cache-block boundaries, and perplexity/answer accuracy. Only then evaluate
MTP, longer context and concurrency separately. No accuracy-threshold CI YAML is
provided yet: there are no real W4 accuracy results from which to set thresholds.

## Accuracy Evaluation

Full-model evaluation is pending. The completed artifact passed a full
header/index audit. Across 45 sampled real expert projections, mean relative
weight RMSE was 0.103625 and mean weight cosine was 0.994629; all sampled
reconstructions were finite. Weight cosine and Gaussian-input errors are diagnostics, not
task-quality evidence. Earlier W8 or other-platform four-bit results cannot be
transferred to this RTN artifact.

## Performance

No W4 tok/s result is available. The initial reference backend keeps only packed
weights resident and dequantizes selected experts one at a time. It performs a
host routing synchronization and is expected to be slower than the W8 grouped
kernel. It must not be advertised as preserving 15 tok/s.

The next performance gate is a 310P-compatible packed groupwise W4 matmul with
device-side routing and bounded scratch space, followed by graph-replay and
MTP acceptance tests. Do not obtain apparent W4 speed by permanently expanding
all experts to INT8/FP16: that would erase the intended memory saving.
