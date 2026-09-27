# W4 generation-speed target

User-directed next goal: match and then exceed the approximately 15 tok/s W8
production profile while retaining packed W4 memory savings and correct output.
This is an active target, not an achieved result.

The first gate is complete: three W4 real-weight smoke answers, seven 310P
regressions, and restoration of unchanged W8. Fresh W8 sustained medians are
19.073 tok/s at short context and 18.091 tok/s around 23.4k context (three
distinct 512-token prompts each). See [HARDWARE_SMOKE.md](HARDWARE_SMOKE.md)
and [HARDWARE_RESULTS.json](HARDWARE_RESULTS.json). The first accelerated Cube
backend now completes all three real-weight smokes at approximately 3 tok/s,
up from the 0.23 tok/s reference path. The second, load-time-NZ-encoded backend
passes the same three smokes at 4.24–4.28 tok/s (+42% on counting). It still
uses host routing and eager execution; it does not yet match W8.
See [CUBE_KERNEL.md](CUBE_KERNEL.md).

The first profile-guided priority was packed-weight unpacking. A real layer-0 TP
partial with synthetic one-token input spends 91% of its profiled device time
inside the six Cube projections for two local experts. Shared-expert casts
are not the dominant cost in that diagnostic. Lossless load-time NZ nibble
encoding, exact arithmetic unpacking and vector metadata broadcast now reduce
M=2 gate/up/down projections to approximately 0.079/0.079/0.081 ms, from
0.320/0.319/0.291 ms. All 51 operator tests pass, including changing-weight
graph replay and K-batch boundaries. This is not a whole-model throughput
claim. The explicit `cube_310_routed` backend now keeps bounded decode routing
on device. All 82 NPU projection/layer tests pass, including graph replays that
change expert IDs and clear formerly local rows, and the production NZ weight
post-load layout. A real layer-0 TP4 partial passes parity against host routing:
its latest M=1 sample drops from 1.219 ms to 0.638 ms under replay (no collective;
not whole-model throughput). Full-model W4 + MTP k=1 + decode graphs now passes
the three correct smokes. Counting reaches 11.592 tok/s, versus 4.280 eager.
Two-token verification batches execute in FULL graph mode, and draft acceptance
counters increase. The completed matched short-context 512-token benchmark is
11.210/10.367/10.722 tok/s (median 10.722), versus W8's median 19.073. Acceptance
is 92.83/78.40/83.81%, close to W8 on the same prompts. In particular both queue
runs draft 287 and accept 225 tokens, but W4 takes 49.291 seconds to decode versus
27.599 for W8: MTP acceptance alone is not the remaining bottleneck. The completed
23.4k benchmark is 10.923/10.660/10.198 tok/s (median 10.660 versus W8 18.091).
Its cold warmup TTFT is 368.799 seconds; prefill remains a serious limitation.
All six 512-token samples completed without worker errors; maximum observed
temperature was 80°C. Production W8 parity is not yet achieved. Next collect a
full-model replay profile before deciding between projection fusion, route/expert
reuse, attention/GDN or communication changes.

The eight-step full-model replay profile is now complete. Routed W4 projections
account for 46.5–49.8% of each rank's summed task time (not critical-path latency).
Within-batch expert-unpack reuse passes 97 NPU tests and full real-weight smokes;
matched short/23.4k medians improve to 11.203/11.050 tok/s. The wider-unpack
candidate passes 101 NPU tests, real-weight layer replay, and all three full-model
smokes. Its completed short/23.4k coding medians are 11.310/11.239 tok/s.
See [REPLAY_PROFILE.md](REPLAY_PROFILE.md). No production parity claim.

Persistent N-tile route scheduling now passes 107 NPU tests and real-weight
layer/full-model smokes. Its MTP k=1 + FULL graph short/long medians are
11.830/11.052 tok/s. The long median is below wide's 11.239, with different
acceptance; this is not a universal throughput win. There is no resident-weight
memory increase.
The five-token synthetic layer diagnostic has not improved (3.570 vs 3.484 ms),
so do not extrapolate the two-token diagnostic's 24.93% gain to all batch sizes.
The next candidate extends bounded expert reuse/scheduling to 80 routes. All
122 NPU tests pass. The corrected matched five-token partial-layer replay is
3.524 ms versus 3.570; an earlier 2.136 ms sample used different synthetic
inputs and is not a matched speedup measurement. MTP k=2 passes all three real
smokes and three-token FULL replay; its short coding median is 11.972 tok/s
(13.482/11.259/11.972), only 1.20% above k=1. Lower acceptance limits the gain.
MTP k=4 also passes all three real smokes and five-token FULL replay, but its
completed short median regresses to 10.155 tok/s (12.377/9.877/10.155), versus
the misleadingly faster 16.179 tok/s counting smoke. Its long-context run also
completed, at 9.838/7.914/6.653 tok/s (median 7.914), below k=1's 11.052.
All three completed 512 tokens, with 23,168 cached tokens; acceptance was
79.07%/59.05%/46.53%. Do not promote k=4 as a coding acceleration. Graph capture reports
0.38 GiB for k=2 and 0.50 GiB for k=4, with model-load still 19.3821 GiB/rank.
Additional projection-output scratch is bounded at 31.25 MiB
for 80 routes and N=2560, without a resident expanded expert bank.
Independent next trace lead: reuse Q/K/index-query RoPE tables, preserving
FP32/FP64 policy and MRoPE coordinates instead of narrowing position integers.

The next kernel candidate removes the dequantized tile's UB→GM→L1 round trip.
The previous `DequantTileToNz` writes a full FP16 N=32 tile to per-route global
scratch before CATLASS reloads it. The installed CANN 9.1
`dav_m200/kernel_operator_data_copy_impl.h` supplies `DataCopyUB2L1Impl` through
`copy_ubuf_to_cbuf` outside the vector-only build. This is an API lead, not
evidence that the proposed kernel works or is faster. A private Qwen-only
prototype must bound L1 alongside the activation tiles, preserve exact NZ
layout and MTE3/MTE1/M lifetimes, and pass the full numerical/replay suite
before a full-model MTP/graph benchmark. Do not modify the shared W8/GLM block
kernel speculatively or count theoretical traffic savings as a tok/s result.

The Qwen-only `ops-l1-r1` prototype now passes 140 numerical/replay NPU tests
(108.69 s), including 18 new M-fractal/activation-stage boundary cases, and
37 CPU/build tests (4.78 s). Matched real-layer graph replay at 1/2/3/5/8
tokens is 0.524/0.390/2.434/1.956/6.133 ms, 5.14–11.06% below persistent-r3.
Full-model k=2 + FULL `[1,3]` under label `mtp2-r2` passes all three real
smokes and shows three-token FULL runtime replay with rising MTP counters.
Short coding is 14.207/11.895/12.470 tok/s, median **12.470**, up **4.16%**
from the previous k=2 median of 11.972. All requests completed 512 tokens;
drafted/accepted=376/323,446/289,426/299. Only the first output hash matches
the older run, and acceptance also varies slightly, so this is not a pure
kernel causal estimate. The 23.4k-context run completed at
10.483/9.283/9.147 tok/s (median 9.283), with all three producing 512 tokens
and reusing 23,168 prefix tokens. Cold TTFT was 359.882 s. This is below
the earlier k=1 median of 11.052; do not extrapolate the short k=2 advantage
to longer contexts. Kernel and k both differ from that k=1 run. This still misses
the measured W8 baseline; counting's 15.373 tok/s is not coding throughput.
Canonical W4 and the shared W8/GLM helper remain unchanged. The existing GM
workspace allocation is deliberately retained for this first execution-path
A/B, so allocator memory/capacity savings are not yet claimed.

The batched peer-output initialization candidate is now implemented in the
isolated `ops-zero-r1` vendor. One strided zero-fill per persistent N tile
replaces per-peer vector fills/stores/drains; local experts overwrite their
rows after the initialization is drained. It retains the existing bounded
workspace and uses at most 5 KiB of the same UB, not a new allocation.
143 NPU tests pass (108.73 s), including NaN-poisoned replay outputs and
exact peer-zero checks; 37 CPU/build tests pass (4.82 s). Matched real-layer
1/2/3/5/8-token graph medians are 0.528/0.334/2.390/1.865/5.942 ms.
Two-token latency is 14.35% lower than L1; three-token is 1.80% lower;
single-token is 0.75% higher. Full-model `mtp2-r3` passes three complete
correct smokes with k=2 and verified three-token FULL replay. The three
512-token short coding requests completed at 14.288/11.902/12.552 tok/s,
median 12.552, only 0.66% above L1's 12.470. All output hashes and acceptance
counts differ; this does not establish a statistically significant or clean
causal throughput gain. The long-context run completed at 11.023/9.709/8.866
tok/s (median 9.709), with all three generating 512 tokens and reusing 23,168
prefix tokens. Cold TTFT was 358.539 s. The median is 4.59% above L1, but the
third prompt regresses and output/acceptance changes; this is not a universal
long-context win. W8 remains unchanged.
Before choosing a larger k for deployment, measure k=1 with the same latest
kernel at long context: the existing k=2/k=1 comparison also changes kernels.

The next isolated W4 candidate shares current-position RoPE tables across Q,
K and index-query, preserving normalization order and accumulation precision.
Index-key positions remain separate. Initial MRoPE graph capture exposed a
synchronous axis-tensor H2D copy; a nonpersistent module buffer now creates
that constant before capture. CPU gates pass 113 tests; two existing model
mock tests are excluded because their vLLM hook no longer exists locally.
All 160 NPU tests pass (106.74 s), including 17 new backend/replay tests.
The isolated three-token query-RoPE graph drops from 0.477 to 0.194 ms
with bitwise-equal outputs (50 iterations, five trials); this excludes the
rest of the model. Real-weight MTP k=2 + FULL gates under label `mtp2-r4`
pass all three correct smokes and verify three-token FULL runtime replay.
Three short 512-token coding requests complete at 14.412/12.091/13.034 tok/s,
median 13.034 (+3.84% versus 12.552). Only the first output hash matches the
previous run; its gain is 0.87%. Drafted/accepted=384/320,456/283,422/301.
Do not attribute the entire median difference to table sharing or claim W8
parity. The 23.4k run completed at 10.852/10.120/9.004 tok/s, median 10.120
(+4.23% versus zero-r1), but all outputs differ and the first prompt regresses.
All three generated 512 tokens with 23,168 prefix tokens reused; cold TTFT
358.790 s. Compare k=1 with this same kernel and RoPE path, not with a
different kernel's earlier result. This matched k=1 run is now complete:
short 13.417/12.392/12.787 (median 12.787), long 13.213/12.754/12.269
(median 12.754) tok/s. All six generated 512 tokens with MTP acceptance and
two-token FULL runtime replay. Long k=1 is 26.03% above k=2; output hashes
differ, so this is not bitwise-identical workload evidence.
Source inspection also found the faster grouped QSA decode dispatch is
capped at two tokens, with a two-token precreated group-list. Three/five-token
MTP verification falls back to the other sparse path. After the separately
queued compact-expert candidate, validate a W4-only expansion of this bound
with operator/replay gates and real long-context A/B; leave W8 defaults intact.

The compact-expert candidate now gathers only the matching activation rows
for a reused expert, into disjoint existing per-N-tile scratch, before the same
FP32 K=128-order Cube reduction. No CPU routing or expanded bank is added.
All 163 NPU tests pass (138.18 s), including changing compact lengths/owners
across graph replay. Matched real-weight partial-layer M=8 replay drops from
5.942 to 4.513 ms (-24.05%); M=1/2/3/5 samples contain no duplicate local
experts and stay broadly similar. Real model `mtp2-r5` passes all three smokes
and three-token FULL replay. Short coding 15.000/12.669/12.926 has median
12.926, 0.82% below 13.034; all hashes differ and acceptance also changes.
This is not a whole-model throughput win. The long benchmark also completed:
10.946/9.777/9.251, median 9.777 tok/s (-3.39% versus 10.120), with all
output hashes changed. All three generated 512 tokens and reused 23,168
prefix tokens; cold TTFT 358.307 s. The two-query-token QSA batched decode
cap is the next independent priority, with MTP and graphs still required.

The W4-only grouped QSA dispatch extension is implemented for up to eight
query tokens, including precreated group-list sizing. W8 and other W4
backends retain the two-token bound. 128 CPU tests and all 72 NPU QSA tests
pass, including 30 dynamic replay cases across 1/2/3/5/8 tokens and local
KV-head counts 1/2. The first NPU test attempt used legacy JIT mode and failed
capture; the rerun aligns the test with serving's existing ACLNN mode.
At the TP4 shape (6 query heads, 1 KV head), isolated three-token replay
falls from 2.071 to 0.297 ms. This does not establish whole-model speed.
Real-weight `mtp2-r6` uses the same compact kernel, MTP k=2 and FULL [1,3].
All three correct smokes pass, and runtime tables confirm three-token FULL
replay. Short coding rates are 15.167/12.555/13.601 tok/s, median 13.601;
all output hashes and acceptance differ from the prior run. The 23.4k
benchmark completes at 12.856/10.933/10.919 tok/s, median 10.933 (+11.82%
versus 9.777). All three generate 512 tokens, reuse 23,168 cached tokens,
and have changed output hashes; cold TTFT is 357.285 s. This is not W8
parity, and even the earlier k=1 long median (12.754) remains higher.
The next measurement is a fresh full-model k=2 long-context trace, keeping
the compact kernel, shared RoPE and expanded QSA dispatch together. The
older k=1 trace predates these optimizations; do not assume its percentages
still describe the current bottleneck. Profiled requests are not speed tests.

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
