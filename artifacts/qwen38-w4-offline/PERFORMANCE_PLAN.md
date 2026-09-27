# W4 generation-speed target

User-directed next goal: match and then exceed the approximately 15 tok/s W8
production profile while retaining packed W4 memory savings and correct output.
This is an active target, not an achieved result.

The first gate is complete: three W4 real-weight smoke answers, seven 310P
regressions, and restoration of unchanged W8. Fresh W8 sustained medians are
19.073 tok/s at short context and 18.091 tok/s around 23.4k context (three
distinct 512-token prompts each). See [HARDWARE_SMOKE.md](HARDWARE_SMOKE.md)
and [HARDWARE_RESULTS.json](HARDWARE_RESULTS.json). W4 remains approximately
0.23 tok/s on its short smoke; the accelerated backend is not implemented yet.

## Acceptance criteria

- Compare against a freshly measured W8 baseline on the same four NPUs,
  prompts, sampling settings, output lengths, and comparable context lengths.
- Report TTFT, decode-only throughput, end-to-end throughput, and MTP acceptance
  separately. A faster step is not sufficient if fewer tokens are accepted.
- Require completed correct responses and numerical operator tests before
  interpreting throughput. Include short coding checks and realistic Kilo-style
  contexts; a counting prompt alone is not production equivalence.
- Retain packed expert banks. Scratch must be bounded by active tiles/routes,
  not a permanent FP16/INT8 expansion of all experts.
- Keep the W8 checkpoint, environment, home launcher and runtime unchanged.
  Maintain the 0.965 memory-utilization ceiling and thermal watchdog.

## Ordered work

1. Finish TP4 eager-W4 smoke and capture the restored W8 baseline. Preserve
   commands, imports, timings, worker logs, and memory accounting.
2. Profile one real W4 MoE layer before integration. In particular, its shared
   expert currently casts resident FP16 weights to FP32 on every call, unlike
   W8's zero-copy NPU `_linear` policy. Check whether that repeats the earlier
   Cast/layout bottleneck, and validate any dtype change against a numerical
   reference rather than assuming its whole-model contribution.
3. Implement a separate 310P packed-Qwen-W4 projection operator and benchmark
   actual checkpoint weights at `[N,K]=[640,2560]` and `[2560,640]`, initially
   `M=1,2,5`, then prefill shapes. Check exact signed nibble order, group-128
   zero points, FP16 scale handling, and accumulation/rounding against the
   existing reference. Gate integration on measured wins, not compilation.
4. Move top-k routing, local-expert masking, gather and weighted scatter onto
   device. Batch active routes and gate/up projections to avoid per-expert
   Python launches and `.cpu().tolist()` synchronization. Preserve TP ownership
   and shared-expert/all-reduce semantics.
5. Validate fixed-address scratch and graph replay with changing token IDs,
   routing, batch sizes and cache-block boundaries. Do not just remove the
   eager-only guard from a host-synchronizing implementation.
6. Enable W4 target + existing floating MTP only after loader/runner checks.
   Measure accepted tokens per step, correctness and actual generation speed
   against W8; tune speculation only from those measurements.
7. Recheck long-context memory and concurrent-session capacity after the fast
   kernel's real scratch/graph usage is known. Do not infer it from payload
   savings or startup cache estimates alone.

## Existing kernel candidate and compatibility boundary

`csrc/gmm/w2_blocked_dequant_matmul_v310/` contains an existing arch-20-compatible
packed W2/W4 dequant-to-NZ/Cube implementation. It is useful as a build/pipeline
reference, but **not numerically compatible as-is**:

| Contract | Existing blocked operator | Qwen W4 checkpoint |
| --- | --- | --- |
| Weight packing | Signed W2/W4, low field first | Signed W4, low nibble first |
| Scale grid | Shared across `[32,32]` blocks, FP32 | Per output row and 128 input columns, FP16 |
| Zero point | None | Signed INT8 per row/group; subtract before scaling |
| Input/output | FP16 projections | FP16 projections |
| Existing small-token bound | Wrapper limits `M <= 128` | Decode initially `M=1,2,5` |

Preserve the existing DeepSeek/GLM operator behavior. A new op or a strictly
separate dispatch contract must implement Qwen's scales and offsets. Its
workspace also needs measurement: the current tiling allocates one `[128,K]`
FP16 NZ area per logical output tile, so the total temporary can equal one
expanded projection even though the whole expert bank stays packed.

The current eager baseline is approximately 0.23 tok/s. The required speedup
is substantial; no claim is made yet that adapting one kernel alone reaches
or exceeds the production target.

The native `aclnnWeightQuantBatchMatmulV2` is not a drop-in packed-INT4 escape
route on this hardware. Its CANN 9.1 documentation distinguishes Atlas
inference products (INT8 weights) from the A2/A3 INT4 support. Keep platform
and format constraints separate; do not infer 310P INT4 support from the
generic dtype table. See the official
[CANN operator constraints](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnWeightQuantBatchMatmulV2.md).
