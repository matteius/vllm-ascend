# 310P grouped W8A16 MTP expert dispatch

## Decision

Do not route the MTP expert bank through `torch_npu.npu_grouped_matmul` on
310P. The pinned 310P path supports the single-expert
`npu_weight_quant_batchmatmul`, but its grouped-matmul implementation does not
support the required FP16-activation, INT8-weight, per-output antiquant scale
combination. The public GroupedMatmul contract likewise requires antiquant
inputs to be null for Atlas inference products. Calling the generic API would
therefore create a branch that fails only on the target hardware.

The repository's custom `MoeGroupedMatmul` is not a substitute. Its operator
definition accepts matching FP16/BF16 input and weight types, and its kernel
instantiates both cube operands with the same type. It has no scale input or
INT8-to-FP16 tile conversion. Its kernel also takes the first tensor-list
weight pointer and computes bank offsets from that address, while the MTP
weights are independent `ParameterList` allocations.

The safe implementation requires a dedicated 310P grouped W8A16 kernel. The
one independent change ready now is one-time FRACTAL_NZ formatting of each
static MTP INT8 weight after loading. It uses the standard W8A16 linear
post-load policy and removes the per-call `TransposeAiCore` work without
changing activation precision, quantization, routing, or expert accumulation.

Public operator reference:

- <https://gitee.com/ascend/cann-ops-adv/blob/master/docs/GroupedMatmulV3.md>

## Measured opportunity

The four-rank profile contains 262 MTP `WeightQuantBatchMatmulV2` calls and 262
matching `WeightQuantBatchMatmulV2_TransposeAiCore_Transpose` calls in the
captured interval. The transpose calls consume 12.43 ms in that interval. The
per-iteration cast/layout audit attributes about 1.45 ms to this MTP path.

Grouping changes each MTP draft layer from two weight-only matmuls per active
local expert to two grouped launches total:

1. grouped gate/up projection;
2. SwiGLU;
3. grouped down projection.

It also removes `counts.tolist()`, the host synchronization, the Python expert
loop, and the current MTP graph break. The one-time NZ formatting removes the
transpose portion before the grouped kernel exists and remains useful to both
the existing fallback and the grouped kernel.

For the real TP4 geometry, each rank owns 128 experts:

| Projection | Per-expert logical weight | Per-expert scale |
|---|---:|---:|
| gate/up | INT8 `[2560, 1280]` | FP16 `[1280]` |
| down | INT8 `[640, 2560]` | FP16 `[2560]` |

Keeping the MTP bank INT8 uses 600 MiB/rank for routed weights. Expanding it to
FP16 solely to use the existing non-quantized grouped kernel would require an
additional 600 MiB/rank, so that is not an acceptable production workaround
for the long-context configuration.

## Operator contract

Add a new operator rather than widening `MoeGroupedMatmul`:

```text
qwen_mtp_w8a16_grouped_v310(
    Tensor x,
    Tensor[] weight,
    Tensor[] weight_scale,
    Tensor group_list,
    int group_list_type=0,
) -> Tensor
```

Contract:

- `x`: FP16 ND `[routes, K]` in stable expert-sorted order.
- `weight`: exactly `E_local` INT8 FRACTAL_NZ tensors, each logically `[K, N]`.
- `weight_scale`: exactly `E_local` FP16 ND vectors `[N]`.
- `group_list`: INT64 cumulative group ends `[E_local]`; empty experts repeat
  the previous end. `group_list[-1] <= routes` because peer-owned routes occupy
  a trailing sentinel range.
- output: FP16 ND `[routes, N]`. Rows at and after `group_list[-1]` must be
  explicitly zero, so graph replay cannot expose stale peer-route data.
- offsets: absent. The MTP quantizer is symmetric and stores no zero point.
- numerical contract: match `npu_weight_quant_batchmatmul` with FP16 input,
  INT8 weight, and FP16 per-output-channel scale. Do not quantize activations.

The tensor-list ABI avoids a second full weight bank. The kernel must obtain
each expert address through `ListTensorDesc::GetDataPtr(expert)`; it must not
assume independent parameters are adjacent.

## Kernel schedule

Create `csrc/gmm/qwen_mtp_w8a16_grouped_v310/` with separate schedules:

- decode/MTP: `routes <= 128`, static `K/N` specializations for
  `2560x1280` and `640x2560`, persistent group metadata, and expert/N-tile
  work stealing across the eight cube cores;
- prefill/fallback: tiled M with the same cumulative group descriptor and no
  host-visible active-expert count.

310P has no grouped pseudo-quant path for this datatype combination. The
kernel must either:

1. load an INT8 FRACTAL_NZ B tile, cast it to FP16 in UB, multiply by the N-axis
   scale vector, move the FP16 tile to L1, and feed the FP16 cube matmul; or
2. use a validated multi-stage decomposition equivalent to the pinned
   `WeightQuantBatchMatmulV2` implementation.

The first approach is easier to validate and preserves activation precision,
but needs explicit unified-core vector/cube event ordering because 310P has no
fixpipe. Reuse the 310P L0C-to-UB compatibility pattern in
`csrc/moe/common/kernel_utils/block/block_mmad_pingpong_tla_multi.hpp` rather
than importing an A2/A3 schedule.

## Exact integration sites

1. Add the operator directory and register it from the existing custom-op
   CMake hierarchy.
2. Add its generated aclnn header and wrapper beside `moe_grouped_matmul` in
   `csrc/torch_binding.cpp`; register the schema beside the existing grouped
   operators.
3. In `vllm_ascend/models/qwen4_exp/mtp.py`, import
   `build_grouped_expert_dispatch` and add
   `_forward_grouped_quantized_npu` to `_MTPFP16MoE`.
4. Build the descriptor with `weight_dtype=self.policy.accumulation_dtype`,
   gather `x` by `dispatch.token_indices[dispatch.order]`, and call the new op
   for gate/up and down.
5. Apply router weights in FP32, gather by `dispatch.inverse_order`, reshape to
   `[tokens, top_k, hidden]`, and sum in the same dimension/order as the current
   path.
6. Select the grouped path only for quantized NPU experts. Keep `_forward_eager`
   as the FP16/CPU and rollback path.
7. In `forward`, return the grouped path directly during graph capture. Retain
   `capture.add_eager` only for the host-dispatched fallback.

## Required tests before selection

Add `tests/e2e/nightly/310p/single_node/ops/test_qwen4exp_mtp_grouped_w8a16_310.py`
with these gates:

- both real projection geometries and a small diagnostic geometry;
- token counts 1, 3, 4, 8, and 12 (10 to 120 routes);
- nonzero TP expert offsets, empty local experts, all-peer routes, repeated
  expert ids, and exactly 128 local experts;
- direct parity against the current per-expert
  `npu_weight_quant_batchmatmul` reference;
- changing expert IDs and activations under ACL graph replay;
- no host synchronization (`counts.tolist`, `item`, or selected-ID copy) in the
  grouped path;
- profiler assertion that one gate/up and one down grouped kernel replace the
  per-expert W8A16/transpose launches.

Promotion additionally requires the fixed 228-question real-model gate, MTP
accepted/drafted counts by position, changing-input service replay, c1/c3/c4
throughput, and peak allocated/reserved memory. Measure the one-time NZ change
separately first, then the grouped kernel, so the transpose gain and dispatch
gain remain attributable.

## Current blocker

This workstation lacks the pinned torch-npu/CANN 310P compiler and runtime, and
the task explicitly excludes remote NPU use. The kernel cannot be compiled or
validated here. Implementing a Python call to generic grouped matmul would
contradict the target operator contract; implementing uncompiled AscendC source
would be an equally unsafe runtime branch. The operator design above is the
smallest complete next patch once a 310P build/test lane is available.
