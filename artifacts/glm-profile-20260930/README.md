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
