# GLM 310P decode: offline optimization design (2026-09-30)

The NPUs are reserved. This is a source and prior-trace analysis, not a
post-merge latency measurement. The current `serve-fp16-mhc.sh` uses eager
execution, four-way expert parallelism, NZ-packed routed codes, fused
gate/up storage, fused mHC Sinkhorn, and FP16 mHC state. Its measured one-stream
decode was 2.043 tokens/s before the latest source merge. The 14–15 s grouped
projection attribution in `artifacts/glm-profile-20260930/README.md` predates
the NZ-packed and mHC changes. A new operator trace must reset the ranking.

## What the current expert kernel actually computes

`_apply_device_grouped` calls the 310P grouped operator once for fused gate/up
and once for down. The operator's grouped wrapper walks expert boundaries on
device. For each active expert, `DequantTileToNz` expands W2/W4 codes and
applies one FP16-rounded scale per 32×32 weight block, writes the full FP16
NZ weight tile to global-memory workspace, then CATLASS reads that workspace
back for FP16 Cube matmul. The routed activation is FP16 on this active path;
the W2A8 policy describes an older fallback, not the serving kernel. Both
the checkpoint's W4 layers 3–32 and W2 layers 33–44 use this FP16 Cube path.

Each expert has 25,165,824 weights: fused gate/up is 4096×4096 and down is
4096×2048. Its packed codes occupy 12 MiB at W4 or 6 MiB at W2, while its
materialized FP16 weights occupy 48 MiB. If eight routed experts per token are
evenly spread over four ranks, one rank processes two per layer on average.
Across 42 routed layers this implies about **0.844 GiB of packed-code reads**
and **3.938 GiB of transient FP16 workspace writes plus 3.938 GiB of reads**
per generated token per rank. Those are a traffic model, not measured HBM
bytes: cache hits, repeated experts, scheduling, and multiple tokens per
expert change physical traffic. They also do not imply 3.938 GiB of *resident*
extra HBM; the workspace is reused. The source nevertheless shows a large
write/read round trip that does no model math.

## Ranked candidates

1. **Feed dequantized tiles to Cube without full global-memory staging.**
   Preserve packed W2/W4 banks and the existing FP16 scale-rounding order.
   Dequantize a 128×128 NZ tile into local storage and have the matmul consume
   it before advancing. This targets both the vector-to-GM write and the
   GM-to-Cube read, plus their ordering barriers. It is the highest-upside
   kernel change but needs an explicit CANN 9.1/310P data-path probe: the
   [documented UB-to-L1 API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/DataCopy_UBToL1_highdim_split.md)
   may relay through GM on some configurations. A
   relay would reduce neither traffic nor latency. A small isolated 128×128
   decode-plus-MMAD prototype should establish whether a direct local path
   exists before changing the grouped operator. Keep the current NZ-packed
   operator as the parity baseline.
2. **Compact the grouped expert walk for decode.** The wrapper scans all 72
   local experts on every Cube block, although one token can route to at most
   eight and an even TP4 split has about two active locally. A fixed-width
   device descriptor of sorted `(expert_id, start, end)` groups could scan
   active routes only, without a host synchronization or dynamic Python shape.
   At one-stream decode, this changes the number of boundary checks from 72
   toward 0–8 per operator invocation; it does not reduce weight dequantization.
   A source-level implementation must handle peer-owned routes, repeated
   expert IDs, empty local groups, and the prefill case where many experts
   become active. Benchmark this separately before adopting it.
3. **Fuse the remaining MoE epilogue after down projection.** Current code
   scales each sorted route, reverses its permutation, reshapes to
   `[tokens, top_k, hidden]`, sums routes, then adds the shared expert. A
   device epilogue could accumulate per-token FP32 outputs and remove the
   inverse permutation and temporary route tensor. It needs deterministic
   accumulation order or an accepted numerical tolerance, plus a plan for
   peer-owned zero routes. The output tensor currently starts at zero because
   peers occupy rows after the last local group.
4. **Store scales already rounded to FP16.** The active kernel loads FP32
   scales and casts each scalar to FP16 before multiplying decoded codes.
   Pre-rounding at conversion time is algebraically the same input to the
   current FP16 multiply, provided NaN/Inf and signed-zero behavior are
   checked. The indexed scale payload is 1.107 GiB across the checkpoint, so
   this could save at most about 0.138 GiB per TP4 rank. It is a capacity
   experiment, not a likely explanation for decode latency. The operator ABI
   and loader would need coordinated changes.

Avoid a persistent full FP16 expert cache as a default. Just two fully
expanded experts per layer consume about 3.938 GiB **resident per rank**, which
directly competes with the context KV budget. A bounded cache would be
worthwhile only if real routing traces show enough cross-token expert reuse.

## KPool math guardrail

The GLM indexer scores each head against a pooled key, applies ReLU, then
weights and sums heads:

`score(pool) = sum_h weight[h] * relu(dot(query[h], key[pool]))`.

The [Transformers reference](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/modeling_glm5_next.py)
and our local `score_kpool` both use this order.
An upstream Triton path folds the weighted heads into one query before the
dot product. That is not equivalent: with heads selecting coordinates 0 and
1, key A `(1, -2)`, key B `(0.25, 0.25)`, and positive unit head weights,
the reference scores A=1 and B=0.5, while the folded query scores A=-1 and
B=0.5. The host regression test checks that the ordering reverses after the
Hadamard and BF16 rounding. A future fused 310P scorer must keep headwise
ReLU unless a model-quality evaluation explicitly accepts an approximation.

The KPool writer also uses host query-boundary lists and per-request loops;
stable metadata alone does not make GLM graph drafting safe. The current
indexer selects a 2048-token history plus causal tail. At 16,384 context,
there are 4096 pooled keys per sparse layer, so the score/top-k path needs a
device-side paged gather and fixed-shape selection for graph capture. A first
310P score kernel should tile pools, compute all 32 head dot products and
headwise ReLUs inside the tile, then write only one reduced FP32 score per
pool. The current PyTorch implementation materializes 32 scores per pool
before reducing them; at 16,384 context that temporary has 131,072 FP32
values per query versus 4096 reduced scores. Retain `torch.topk` initially,
then fuse selection only if a trace shows it matters. This is a stronger
long-context target than the completed-pool compression filter alone.

## First hardware return gate

Profile the running FP16-mHC/NZ build at one and four streams, separating
prefill from decode. Capture operator time for grouped gate/up and down,
KPool score/write, mHC, KDA, dispatch, and collectives. Compare physical
memory traffic and task span before selecting one of the kernel candidates.
For any new grouped kernel, require W2 and W4 FP16 output parity on single
and repeated experts, zero local routes, peer routes, and long prefill; then
run matched greedy outputs/logprobs and context-capacity checks. Do not infer
generation speed from the pre-NZ profile or an isolated operator benchmark.
