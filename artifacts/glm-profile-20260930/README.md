# GLM 5.3 Flash 310P end to end profile

## Setup

- Host: Threadripper, four 310P3 chips on two cards.
- Model: `GLM-5.3-Flash-W4through32-noclip-310p`, TP4, 151.58 GiB checkpoint,
  35.6075 GB loaded weights per rank.
- Server: `serve-profile.sh` on `127.0.0.1:8003`, eager mode, 128 token maximum
  context, four concurrent sequences. The same model process served all
  measurements. The known good grouped package remained active throughout.
- Raw NPU traces and full CSV exports are on the host at
  `/srv/ai/src/glm-grouped-offline-20260929/glm-profile-20260930/npu`.
  They are too large to copy into this repository.

## Client results

The client uses streaming chat completions, measures first and last content
timestamps, and reports decode rate as `(completion_tokens - streams) /
(last_content - first_content)`. Each prompt was 25 tokens; each stream
generated 32 tokens with EOS ignored.

| Run | Streams | Output tokens | TTFT | Aggregate decode | Aggregate end to end |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1 | 32 | 8.62 s | 0.764 tok/s | 0.651 tok/s |
| Baseline | 4 | 128 | 8.56–20.8 s | 1.474 tok/s | 1.381 tok/s |
| NPU profiler active | 1 | 32 | 8.66 s | 0.726 tok/s | 0.623 tok/s |
| NPU profiler active | 4 | 128 | 8.57–20.94 s | 1.447 tok/s | 1.358 tok/s |

Exact timings are in `baseline-s1.json`, `baseline-s4.json`,
`profiled-s1.json`, and `profiled-s4.json`. Profiled rates include tracing
overhead and must not replace the baseline rates.

## NPU profile

The torch NPU profiler captured 16 engine iterations per workload, including
prefill and decode. `analyze-ranks.py` reads the exported per-rank
`kernel_details.csv` files and CANN communication summaries. The complete
four-rank result is in `rank-summary-final.json`.

| Capture | Task span per rank | Grouped projection tasks per rank | Collective tasks per rank |
| --- | ---: | ---: | ---: |
| One stream, 16 iterations | 29.8 s | 14.2–15.4 s | 8.3–9.6 s |
| Four streams, 16 iterations | 49.6 s | 28.4–29.9 s | 11.1–12.8 s |

Per-rank NPU task union covers about 92–96% of each captured interval.
These task sums are work attribution, not additive end to end latency.

The CANN communication summaries show only 18–51 ms of collective transit
over 1,472 collectives per rank, while 8.1–12.6 s is recorded as waiting.
All four ranks alternate as the last to arrive: the counts are
403/315/377/377 for one stream and 422/296/356/398 for four streams. Median
four-rank arrival spread is 3.3 ms and 3.0 ms respectively; the 90th
percentile grows from 26.3 ms to 38.8 ms at four streams. No single rank
consistently limits the run.

The grouped projection uses eight Cube blocks. For decode, its median task is
about 4.3 ms at eight routed rows and 8.6 ms at 32 routed rows. The kernel's
reported vector dequant time is far larger than its MAC time; the exact
hardware counters are saved in `grouped-hardware-rank1-s4.txt`. Prefill
activates more experts and costs about 49–60 ms per grouped projection in
this trace.

An isolated synthetic bank makes the expert-count effect explicit. With two
active local experts across eight routed rows, W4 gate/up and W2 down take
4.40 and 4.64 ms respectively. With eight active local experts across 32 rows,
they take 17.31 and 18.21 ms. This benchmark uses synthetic codes and fixed
routes; it is evidence about kernel scaling, not model output speed.

## Python sample and isolated checks

`worker0-python.speedscope.json` contains 1,332 py-spy samples taken during
a single-stream decode request. Most main-thread samples landed in the MoE
combine's NPU scalar creation, but the NPU timeline shows that work is queued
behind nearly continuous device activity. A small scalar `torch.add` check
reduced isolated combine latency from 0.098 ms to 0.062 ms with identical
FP32 output; that saving does not explain the end to end latency.

Two grouped-kernel ideas were tested in separate operator packages while the
server remained on the known good package:

1. A clean integer-shift unpack binary had a distinct SHA-256 hash but failed
   all five W2/W4 hardware parity cases. It was never used for inference.
2. A clean build that reused dequant lookup tables across experts passed all
   five hardware parity cases. Its measured projection improvement was under
   1% at the GLM decode shapes, so it was not promoted.

Earlier apparent parity and flat timing for those ideas came from the CMake
build directory reusing the old `.o` despite changed source. Both experiments
were rebuilt from fresh build directories and their binary hashes were
checked. The experimental source edits were then reverted; the live server
was never reloaded.

## Reproduction

- `serve-profile.sh` starts the server with a 16-iteration torch NPU profiler.
- The benchmark client is
  `../native-int4-20260927-pass3/bench-parallel.py`; the capture controller is
  `../../tools/qwen4exp/profile_runtime.py`. Copies of both were run from the
  host profile directory. For each concurrency,
  `capture --phase decode --steps 16 --delay 0 --execute` wrapped
  `bench-parallel.py --concurrency {1,4} --max-tokens 32`. The first capture
  step includes prefill.
- `bench-grouped-projection.py` compares W2/W4 grouped projection latency at
  eight and 32 routed rows without loading the model.
- `bench-moe-combine.py` compares the isolated MoE combine expressions.
- `analyze-grouped.py` summarizes grouped kernel shapes and hardware counters.
- `analyze-ranks.py` combines per-rank timelines and HCCL wait/arrival data.

The eight raw traces and their parsed CSVs remain on the host. The server
remains running for subsequent measurements.

## Batched NZ workspace write

The grouped W2/W4 kernel formerly wrote each decoded 16x16 NZ fractal to GM
and waited for MTE3 before reusing its UB buffer. It now stages all eight
fractals of a 16x128 tile in UB and performs one contiguous GM write. This
removes seven MTE3 transfers and seven pairs of V/MTE3 synchronization events
per tile. The dequantization math and NZ layout are unchanged.

A distinct candidate kernel object (SHA-256 prefix `237bbdb13d7b`) passed all
five hardware parity cases in `test_grouped_candidate.py` against the FP16
reference and the known-good single-expert operator. The live baseline object
has SHA-256 prefix `0f631e374438`. Four isolated projection shapes were run
in baseline/candidate/baseline/candidate order while the model remained loaded.
The table averages each pair of 30-run medians.

| Projection | Routed rows | Baseline | Batched write | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| W4 gate/up, N=2048 K=4096 | 8 | 4.424 ms | 3.436 ms | 22.3% |
| W2 down, N=4096 K=2048 | 8 | 4.661 ms | 3.669 ms | 21.3% |
| W4 gate/up, N=2048 K=4096 | 32 | 17.405 ms | 13.524 ms | 22.3% |
| W2 down, N=4096 K=2048 | 32 | 18.232 ms | 14.384 ms | 21.1% |

The candidate package is isolated at
`/srv/ai/src/glm-grouped-batchwrite-20260930/opp-grouped-batchwrite`; it is a
copy of the known-good package with the freshly compiled grouped object and
metadata substituted. `serve-optimized.sh` selects that package, enables the
`poolside_v1` parser for the checkpoint's XML tool-call format, and expands the
context for Kilo. The remote TCP forward in `forward-8001.py` maps the
Threadripper LAN endpoint on port 8001 to the vLLM loopback port 8003.

The optimized server's unprofiled streaming check used the same 25-token
prompts and 32-token completions as the baseline:

| Streams | Baseline decode | Optimized decode | Improvement |
| --- | ---: | ---: | ---: |
| 1 | 0.764 tok/s | 0.912 tok/s | 19.3% |
| 4 | 1.474 tok/s | 1.822 tok/s | 23.6% |

Exact timings are in `optimized-s1.json` and `optimized-s4.json`. These runs
changed the context and tool configuration as well as the grouped kernel, so
the end-to-end figures are directional evidence. The isolated package A/B above
measures the kernel change under fixed settings.

The first optimized launch attempted 320 forced KV blocks and a 4096-token
context. It ran out of NPU memory allocating KV tensors. The server was then
started with 128 forced blocks and a 2048-token context; startup and inference
passed. The tool parser accepts GLM XML calls in a direct parser check and an
OpenAI request with `tool_choice=auto` returns HTTP 200. That 64-token model
sample did not emit a valid tool call, so end-to-end tool behavior is not yet
validated.

A Kilo prompt then exceeded the 512-token prefill batch. The second prefill
chunk entered the 310P MLA context-merge path, which raises
`NotImplementedError` and stops the engine. The launch now sets the batch limit
equal to the 2048-token model limit and explicitly disables chunked prefill and
prefix caching. This is a serving correctness fix; its memory and Kilo behavior
must be checked after the restart.

## NVIDIA NVFP4 source screening

The local `GLM-5.3-Flash-NVFP4` directory is the NVIDIA ModelOpt checkpoint.
It supplied the earlier GPU golden output. The running W4-through-32/W2-late
Ascend checkpoint instead derives from the ZAI FP8 checkpoint, as recorded in
its conversion manifest. The NVIDIA release is a post-training quantization,
and its card reports benchmark results close to the BF16 source. These are
vendor results, not measurements of the Ascend conversion.

To screen a requant, we decoded two NVIDIA routed-expert matrices from their
E2M1 codes and per-16 scales, applied our signed-int W4/W2 no-clip 32x32
quantizer, and compared both the proposed conversion and the existing Ascend
matrix against the ZAI FP8 dequantized matrix. Relative error is the L2 norm
of the difference divided by the FP8 matrix L2 norm; lower is better.

| Expert matrix | Existing FP8 → Ascend error | NVIDIA NVFP4 → Ascend error |
| --- | ---: | ---: |
| Layer 3, expert 0, gate projection, W4 | 0.131704 | 0.162255 |
| Layer 33, expert 0, gate projection, W2 | 0.634864 | 0.651611 |

The NVIDIA NVFP4 matrix itself has relative error 0.090372 and 0.091717
against ZAI FP8 in those two samples. Direct NVIDIA NVFP4 therefore retains
useful precision, while another conversion to the current integer format
increased error in both samples. This does not establish whole-model quality;
the next useful check is an output evaluation of a precision-preserving NVFP4
path, subject to the 310P memory limit.

## Sparse kpool bring-up on 310P

`serve-kpool.sh` serves the same W4-through-32/W2-late checkpoint with GLM's
four-token pooled indexer and native paged latent attention. The indexer
compresses completed pools with gated APE weighting and a Hadamard transform,
keeps the unfinished pool as a causal tail, and selects up to 512 completed
pools for each query. The 310P cache stores the compressed keys in FP16 because
the device's indexed write does not support BF16 at the serving write size.

The candidate checkout on Threadripper is
`/srv/ai/src/glm-kpool-sparse-20260930`. The persistent `glm-kpool-sparse`
tmux session serves loopback port 8003; `glm-profile-forward-8001` exposes it
at `http://192.168.53.187:8001/v1`. The served model ID is
`glm53-flash-ascend-profile`. Kilo on this workstation uses that LAN endpoint.

The server is configured for a 16,384-token maximum context and four scheduled
sequences. At startup it reported 18,378 total NPU KV tokens, or 1.12 times one
16,384-token request. Four simultaneous full-length requests therefore do not
fit. System-memory KV offload is not enabled for this GLM serving path.

Validation on 30 September 2026:

| Check | Result |
| --- | --- |
| Targeted unit tests | 44 passed |
| Single-device NPU probe | Two requests, cross-chunk pool completion, FP16 1,024-element cache write, and 26-token selection passed |
| HTTP short request | 22 prompt + 4 completion tokens, HTTP 200 |
| HTTP continued prefill | 3,020 prompt + 4 completion tokens, HTTP 200; prefill 109.7 s (27.5 tokens/s effective), decode about 0.8 tokens/s |
| Four-request smoke | Four 25-token prompts and 16 generated tokens each completed; aggregate decode 1.57 tokens/s, peak running requests 4 |
| Longer sparse prefill | 8,020 prompt + 4 completion tokens, HTTP 200; prefill 315.8 s (25.4 tokens/s effective), decode about 0.8 tokens/s |

The 3,020-token request crosses the 2,048-token sparse selection threshold.
These four-token completions establish that the server runs without an NPU
fault; they do not establish response quality or correctness over the full
16,384-token configured context. The current checkpoint still needs a useful
output-quality evaluation, especially for its W2 expert layers.
The short four-request smoke used fewer output tokens than the earlier
32-token benchmarks, so its throughput should not be treated as a controlled
before-and-after comparison.

## Packed weight layout experiment

The grouped kernel reads 16 short row slices per 16×128 packed tile. An
isolated candidate rearranged the code bank into contiguous 16×128 tiles to
replace those row copies with one transfer. Its object differs from the live
kernel (SHA-256 prefixes `89c5d164` and `237bbdb` respectively). All four
W2/W4 benchmark outputs were bit-for-bit identical. Twenty-run medians in
separate processes showed no useful speed change:

| Projection | Routed rows | Canonical codes | Tiled codes |
| --- | ---: | ---: | ---: |
| W4 gate/up | 8 | 3.4318 ms | 3.4373 ms |
| W2 down | 8 | 3.6480 ms | 3.6536 ms |
| W4 gate/up | 32 | 13.4726 ms | 13.4683 ms |
| W2 down | 32 | 14.3911 ms | 14.3924 ms |

The tiled candidate was not deployed. The benchmark client is
`bench-tiled-candidate.py`; its isolated build is under
`/srv/ai/src/glm-tiled-groups-20260930` on Threadripper.

## NZ-ordered packed expert codes

A second isolated prototype repacked each 16×128 logical code tile into the
Cube's final NZ order. It keeps the same W2/W4 bit width and byte count. The
kernel then reads one contiguous packed tile and decodes directly into its NZ
workspace, omitting the gather and eight 16×16 transposes. The candidate
object has SHA-256 prefix `5257ee74`; all four benchmark outputs were exactly
equal to the canonical-code operator, with zero differing FP16 elements.

| Projection | Routed rows | Canonical codes | NZ-packed codes | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| W4 gate/up | 8 | 3.5075 ms | 1.0825 ms | 69.1% |
| W2 down | 8 | 3.7180 ms | 1.3396 ms | 64.0% |
| W4 gate/up | 32 | 13.5800 ms | 4.1886 ms | 69.2% |
| W2 down | 32 | 14.5898 ms | 5.0953 ms | 65.1% |

The prototype used separate Python processes and the same random seed,
inputs, scales, and routes for each layout. `bench-nzpacked-candidate.py`
contains the CPU repacker and isolated benchmark. These are projection times;
the integrated full-model decode measurements appear below.

The integrated 310P operator accepts the canonical `uint8` bank and the
NZ-packed `int8` byte view as separate input variants. The GLM loader enables
the NZ layout through the model config override
`"ascend_glm_nz_packed_codes": true`; the packed bank keeps its original byte
count. The build on Threadripper is isolated at
`/srv/ai/src/glm-nzpacked-20260930/opp-nzpacked`. Seven 310P grouped-operator
tests and eight focused host bank tests passed. A second isolated A/B using
the integrated package reproduced exact FP16 parity in all four shapes.

With the candidate serving from `serve-nzpacked.sh`, the matching 25-token
prompt / 32-token completion benchmark measured:

| Streams | Previous decode | NZ-packed decode | Improvement |
| --- | ---: | ---: | ---: |
| 1 | 0.912 tok/s | 1.386 tok/s | 52.1% |
| 4 aggregate | 1.822 tok/s | 2.946 tok/s | 61.7% |

Exact timings and concurrency samples are in `nzpacked-s1.json` and
`nzpacked-s4.json`. The launcher retains the 16,384-token maximum context,
the existing sparse kpool attention, and the port-8001 LAN forward.

The NZ-packed server also completed a 3,020-token continued prefill in
91.45 seconds (33.0 effective prompt tokens/s), compared with 109.7 seconds
(27.5 tokens/s) on the preceding kpool build. Its four-token decode ran at
about 1.4 tokens/s. The 16,384-token configured maximum has not yet been
validated end to end.

## Fused 310P mHC Sinkhorn

After the packed-code change, the sampled Python hot path shifted to mHC.
The earlier four-stream NPU trace contained 123,456 `ReduceSum` and 56,832
`Div` tasks in 16 profiled iterations, consistent with the eager 20-round
Sinkhorn normalization over each 4×4 hyper-connection matrix. Reducing the
round count is unsafe for peaked inputs: a 10-versus-20-round random-matrix
check showed differences above 0.01 at unit logit scale. The new
`MhcSinkhornV310` operator keeps all 20 rounds and runs them in one launch.

The NPU kernel uses one core per input matrix group and reuses a 16-element
UB tile. A multirow parity test caught a missing post-write barrier before
deployment; the corrected binary (SHA-256 prefix `061651a89f32`) passed
random 4×4 cases from one to 512 input rows and the complete mHC pre path.
The one-token latency median was 0.0646 ms fused versus 1.5635 ms eager;
the 512-token median was 0.3587 ms versus 1.4189 ms. These are isolated
operator measurements with NPU synchronization after each call.

Rebuilding the Python extension exposed a separate cache-write boundary:
vLLM supplies `int32` slot IDs on this path, while the native MLA cache
writer reads `int64`. The Python dispatcher now converts the slots, and a
separate `MlaCacheWriteV310` package is installed in the isolated candidate.
Without that package, the previous server used the PyTorch indexing fallback.
Two hardware tests cover int32 and int64 slots, including a negative padding
slot. The fused Sinkhorn package and MLA writer also passed together under
the exact vendor search order used by serving.

`serve-sinkhorn.sh` enables this candidate through the top-level model config
override `ascend_glm_fused_sinkhorn`; the previous eager path remains the
default. The candidate retains the NZ-packed expert bank and 16,384-token
configured context. With the same 25-token prompts and 32-token completions:

| Streams | NZ-packed decode | Sinkhorn + MLA writer decode | Improvement |
| --- | ---: | ---: | ---: |
| 1 | 1.386 tok/s | 1.743 tok/s | 25.7% |
| 4 aggregate | 2.946 tok/s | 3.269 tok/s | 11.0% |

The end-to-end gain includes both the mHC fusion and newly active MLA writer;
it is not an isolated estimate of either component. Exact timings are in
`sinkhorn-s1.json` and `sinkhorn-s4.json`. Two deterministic 16-token prompts
produced the same 32 greedy tokens before and after; the largest chosen-token
logprob change was 0.117. This smoke check does not establish model quality,
which still needs evaluation of the W2 expert layers.

The serving candidate also completed a 3,020-token continued prefill in
94.17 seconds client elapsed, with 92.35 seconds in the server's prefill phase
(32.7 effective prompt tokens/s). This is effectively unchanged from the
NZ-packed candidate's 33.0 tokens/s. Its four-token continuation ran at about
1.7 tokens/s. The configured 16,384-token maximum and four concurrent full
context windows remain unverified; the current NPU KV allocation is only
about 18,378 tokens total across requests.

## FP16 mHC state candidate

The checkpoint's expert projections already use FP16 activation math, but
the mHC residual and mixing path rounds FP32 intermediates to BF16-like
precision for reference-model compatibility. On 310P, the bit-trick rounding
uses several NPU tasks per tensor. A model override,
`"ascend_glm_mhc_fp16_state": true`, instead rounds those intermediates
through native FP16 and carries the result back in FP32. The default remains
the reference-compatible BF16-like path. The FP16 cast has a narrower exponent
range and changes numerical results; it is an explicit speed/quality trial.

For a synthetic 16,384-element input, the median synchronized rounding time
was 0.0772 ms with the FP16 round trip versus 0.2233 ms with the bit trick.
Only 12.3% of values rounded to identical FP32 results, so token-level checks
matter more than this operator benchmark. The matched serving benchmark gave:

| Streams | BF16-like state decode | FP16 state decode | Improvement |
| --- | ---: | ---: | ---: |
| 1 | 1.743 tok/s | 2.043 tok/s | 17.2% |
| 4 aggregate | 3.269 tok/s | 4.155 tok/s | 27.1% |

The two deterministic 16-token prompts emitted the same tokens in both modes;
the maximum chosen-token logprob difference was 0.043. This is a smoke test,
not a quality evaluation. The FP16 candidate also completed the same
3,020-token context probe in 67.62 seconds (server prefill 65.97 seconds,
45.8 effective prompt tokens/s), compared with 94.17 seconds and 32.7
effective prompt tokens/s for the previous build. Its four-token continuation
ran at about 1.9 tokens/s. Evidence is in `fp16-mhc-s1.json`,
`fp16-mhc-s4.json`, `fp16-baseline-output.json`, `fp16-candidate-output.json`,
and `fp16-mhc-3020-context.json`. The FP16 candidate serves through
`serve-fp16-mhc.sh` and the existing Threadripper port-8001 forward.

An isolated follow-up checked whether running the mHC hyper-connection
projection itself in FP16 would help. For a 16,384-wide input and 24 output
channels, the one-row median was 0.1313 ms in FP32, 0.1645 ms with FP16
activation and weight casts per call, and 0.1503 ms with a cached FP16 weight.
The four-row case was likewise 0.1270, 0.1608, and 0.1418 ms. That projection
change was not promoted; it is slower at the decode shapes and changes the
output by up to 0.12 in the synthetic check.
