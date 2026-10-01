# GLM post-fusion 310P profile (30 September 2026)

The measured server used the checkpoint, TP4 launch, 16,384-token context,
NZ-packed W2/W4 expert bank, fused mHC Sinkhorn, MLA writer, and FP16 mHC state
from `../serve-fp16-mhc.sh`. A separate launch added a torch NPU profiler with
16 active iterations, zero delay, and output under
`/srv/ai/src/glm-round-20260930/glm-profile-codex-20260930/npu` on Threadripper.
The same server stayed loaded for all four client runs. Each client stream used
a 25-token prompt and 32 output tokens with EOS ignored.

| Run | Streams | Unprofiled decode | Profiled decode |
| --- | ---: | ---: | ---: |
| Matched short request | 1 | 2.049 tok/s | 1.855 tok/s |
| Matched short request | 4 | 4.023 tok/s aggregate | 3.777 tok/s aggregate |

Exact client timestamps are in `baseline-s1.json`, `baseline-s4.json`,
`profiled-s1.json`, and `profiled-s4.json`. The four-stream baseline reached
four running requests. These short runs reproduce the earlier 2.043 and
4.155 tok/s results within run-to-run variation. Profiled rates include
tracing overhead and are not serving baselines.

The Ascend parser completed four of eight rank captures before NPU use was
deferred for Qwen tests: ranks 1 and 3 of the one-stream capture, and ranks 2
and 3 of the four-stream capture. The raw captures for all eight ranks remain
at the Threadripper path above. The parser and GLM server were stopped; all
four NPUs reported no running processes. No post-trace model change or NPU
test was made.

| Capture | Rank | Task span | Grouped expert tasks | Collective tasks | Collective wait | Collective transit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 stream | 1 | 10.50 s | 3.45 s | 2.81 s | 2.67 s | 21 ms |
| 1 stream | 3 | 10.50 s | 3.67 s | 2.31 s | 2.17 s | 22 ms |
| 4 streams | 2 | 19.99 s | 10.16 s | 3.89 s | 3.71 s | 46 ms |
| 4 streams | 3 | 19.99 s | 10.07 s | 3.71 s | 3.53 s | 57 ms |

These sums are task attribution; task overlap and rank synchronization mean
they cannot be added to predict end-to-end latency. The four-stream capture
includes three prefill iterations with 200, 208, and 416 routed rows. The
one-stream capture includes one 200-row prefill iteration. Filtering each
rank's timeline after the last prefill grouped task gives this approximate
decode-only breakdown:

| Capture / rank | Decode iterations | Remaining task span | Grouped expert tasks | Collectives | Large cache slices |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 stream / rank 1 | 15 | 8.01 s | 1.93 s (129 ms/step) | 2.45 s (163 ms/step) | 0.44 s |
| 4 streams / rank 3 | 13 | 11.67 s | 5.11 s (393 ms/step) | 2.81 s (216 ms/step) | 0.39 s |

The grouped operator's decode activation shape changes from eight to 32
routed rows. On rank 3, its median task grows from about 1.0 to 3.0 ms.
Four-stream grouped work consumes roughly 44% of the post-prefill device
task span, so the packed expert projection remains the first kernel target
at useful concurrency. The 1,472 collective calls per capture mostly wait
for rank arrival; their data transit is tens of milliseconds. A full-rank
arrival-skew analysis is still needed before attributing those waits to an
individual rank or expert schedule.

The trace also contains 330 slow `aclnnContiguous_SliceAiCore_Slice` tasks on
rank 1, with input shape `[342,64,384,16]` and about 1.34 ms median each.
They total 0.44 s in the decode portion (about 30 ms/step) and deserve a
separate cache-layout investigation. The grouped projection and collective
wait are larger targets. The previous resident-L1 grouped candidate was
slower in isolated tests and remains disabled.

## Source follow-up while the NPUs are reserved

The grouped task maps to the 310P W2/W4 grouped Cube operator in
`csrc/gmm/w2_grouped_blocked_dequant_matmul_v310`. Its kernel reads the
device-resident cumulative expert ends, scans every local expert, decodes
packed weight tiles into its reusable NZ workspace, and runs the Cube matmul.
The trace measures the complete grouped task, so it does not separate the
expert scan, dequantization, workspace traffic, and Cube math. Any claimed
speedup from changing one of those stages needs an isolated operator trace.

The 64-to-32-channel slices match two 512-wide views of a physically padded
1024-wide NZ page, once per key and value view in each of 11 sparse layers.
The 310P decode path passes those views to `npu_qsa_sparse_attention_310`;
the QSA kernel currently computes the page address from the *logical* cache
channel count. A direct use of the padded backing therefore needs a physical
page stride in the operator contract and tiling data, plus a logical head
count independent of that stride. Merely dropping `contiguous` or passing
the full page would address the wrong rows. The trace shape and call path
support this explanation, but the exact caller of CANN's copy still needs a
runtime stack trace or targeted operator capture when hardware is free. A
read-only inspection of rank 1's saved `trace_view.json` found a pair of
1.33 ms and 1.37 ms slice tasks immediately after `AscendCL@aclnnContiguous`;
no `aten::contiguous` event was nearby. This narrows the copy to CANN's
contiguity handling, but does not identify which cache view triggered it.

The merged local model runner also referred to missing padded-layout and
GLM view helpers and used undefined cache-layout locals. The source repair
restores the cache capability helpers, initializes the DeepSeek-V4 stride
mapping, and keeps GLM on its existing page-strided reshape path. The V2
cache-spec conversion now preserves GLM's model marker, physical stride
marker, and compression ratio; its historical tail view is limited to the
historical tail role. These are startup/correctness repairs with no measured
decode-speed result.

When NPU testing resumes, first complete the other four rank parses. Then
benchmark a new grouped-projection candidate at eight and 32 routed rows,
covering W4 gate/up and W2 down, before a full model reload. The candidate
must pass exact FP16 parity, including empty local groups and repeated
experts; follow with matched end-to-end one- and four-stream runs. A direct
small-batch packed-code projection is a plausible prototype because each
active expert receives few rows, but this trace does not establish that it
will beat the current Cube path.

## Follow-up before the next GLM reload

An isolated NZ-packed grouped build replaced the 72-expert linear scan with a
binary search for decode rows. Four hardware parity cases passed. Its isolated
operator medians did not improve on the known-good build, so the source was
returned to the linear scan and the candidate package was not used for serving.

| Projection | Rows | Known-good | Binary-scan candidate |
| --- | ---: | ---: | ---: |
| W4 gate/up | 8 | 1.0800 ms | 1.1207 ms |
| W2 down | 8 | 1.3058 ms | 1.3112 ms |
| W4 gate/up | 32 | 4.0421 ms | 4.0533 ms |
| W2 down | 32 | 4.9823 ms | 4.9607 ms |

A direct QSA probe with the trace's 342-page geometry initially measured
2.90 ms per call for the logical `[342,32,384,16]` view backed by 64
physical channels. A contiguous 64-channel control took 0.065 ms. That
control had a different logical KV-head count, so it isolated copy cost but
was not equivalent attention. The source now passes QSA a contiguous physical
page and an explicit logical KV-head count; the kernel uses its physical page
stride for addressing. The compact live-KDA planner remains a separate route
to avoid padding the MLA page in the first place.

## Physical-page QSA and four-rank integration

The isolated native QSA build passed two Ascend 310P parity tests, including a
40-token case across two pages with exact FP16 output. With 342 physical pages
and the same logical KV-head count in both calls, the strided view measured
2.8705 ms median and the contiguous physical-page call measured 0.0615 ms
median. `benchmark-qsa-physical-page.py` and `qsa-physical-page-342.json`
record the probe. This is an isolated operator result, not a serving rate.

The first full TP4 load of the isolated package failed before serving because
vLLM's cross-rank shrinker omitted GLM's fixed live-KDA memory reserve and
returned different KV block counts. Re-planning each worker with its fixed
reserve plus a common variable-block budget fixed startup. A second TP4 load
served `/v1/models` at port 8003 and reported 37,780 tokens of NPU KV cache.

Its first generation request then exited with an AICPU `index 9 is out of
bounds for dimension 0 with size 4` error, followed by an AICore exception in
`RecurrentGatedDeltaRuleV310`. The isolated package had an older 310P runner
condition that enabled four-request compact Mamba state for Qwen but omitted
GLM. The planner allocated four live KDA state slots while the runner left
scheduler block IDs 9–11 unmapped. The candidate package now enables GLM
remapping and checks this invariant before allocation; the repository runner
has the same condition and early check. Twenty-two CPU planner and compact
state tests passed. A later full TP4 serve on port 8003 completed generation;
its matched 25-prompt-token, 32-output-token runs measured 2.212 tok/s at one
stream and 4.754 tok/s aggregate at four streams. The corresponding files are
`compact-kda-s1.json` and `compact-kda-s4.json`. The 4-stream gain over the
earlier 4.023 tok/s baseline combines the compact KDA and physical-page QSA
changes, so it cannot be attributed to either change alone.

A separate FULL_DECODE_ONLY graph trial failed during capture in the GLM
K-pool indexer's `_write_pools`: boolean advanced indexing on `valid_state`
called `aclnnNonzeroV2` while the stream was captured. The later `507015`
stream synchronization errors were consequences of that failure. The same
function also selects completed pools with a device-side boolean mask, so a
single indexing-site edit would not establish graph compatibility. No graph
throughput result exists.

## Port 8001 serving check

The eager TP4 server was reloaded on `0.0.0.0:8001` with the same checkpoint,
context limit, and 0.965 NPU memory utilization. A 26-token arithmetic prompt
completed without a device fault. It answered 42 but exposed a stray
`</think>` marker in the content. Matched 25-prompt-token, 32-output-token
runs measured 2.236 tok/s at one stream and 4.476 tok/s aggregate at four
streams; see `compact-kda-port8001-s1.json` and
`compact-kda-port8001-s4.json`. This is within the variation of the earlier
port 8003 run. The four-stream run reached four active requests with none
waiting. A 2,066-token prompt completed its prefill in 45.0 seconds and
decoded 16 tokens in 7.0 seconds; see `context-port8001-2k.json`.
An 8,232-token prompt completed prefill in 210.3 seconds (39.1 prompt tok/s)
and decoded 32 tokens in 14.4 seconds (2.16 tok/s); see
`context-port8001-8k.json`. The later 512-token prefill chunks were about
13–14 seconds each. Decode latency did not grow materially between the short
and 8K requests, while long-context prefill became much slower.

These requests passed `chat_template_kwargs={"enable_thinking": false}`.
Inspection of the checkpoint's `chat_template.jinja` shows that it does not
read this key: it always starts generation with `<think>` and accepts
`reasoning_effort` instead. The server was launched without a reasoning
parser, so response `content` mixed reasoning with the final answer. The
8K request's first 32 tokens were off topic, but they were not a completed
answer; this evidence does not establish long-context answer quality. A
short arithmetic response was correct yet exposed `</think>` in content.

`serve-compact-kda-8001.sh` records the next launch configuration with
`--reasoning-parser glm47`; `probe-coherence.py` records reasoning and final
content separately and uses the supported `reasoning_effort` template key.
Neither has been run on NPU. The source batch builds K-pool indices for every
short-context query row at once when all completed pools fit the 512-pool
budget, requests FP32 directly for the 310P expert router, and uses an op
scalar for the routed/shared combine instead of allocating a device tensor
for the multiplier. CPU K-pool parity and MoE tests pass; NPU correctness and
speed remain to be measured. NPU testing was deferred before these source
changes were deployed. The eager server subsequently received a shutdown
signal and port 8001 is down. Its final log shows orderly teardown, not a new
AICore exception.

## Next decode optimization batch

The grouped W2/W4 operator owns one Cube launch per projection across all
experts. A core currently scans cumulative group ends in expert order and
dequantizes each active expert's packed weight tiles into an NZ workspace.
The prior binary search reduced scan work but changed the 8/32-row isolated
medians by less than noise; a resident-L1 variant was slower. The next kernel
candidate should target *work per active expert*: cache a packed tile in L1
only when more than one routed row shares it, and keep the current workspace
path for singleton groups. That needs a separate isolated kernel package and
parity cases for W4 gate/up, W2 down, zero local routes, and repeated experts.
The existing known-good kernel remains the serving default until the isolated
8/32-row medians beat it and matched one-/four-stream runs confirm the gain.

Before that kernel trial, a single parser-enabled serve should check the
completed final answer at short and 8K contexts, followed by K-pool index
parity, router IDs/weights, and combine output against this CPU-tested batch.
The saved 8K response stopped after 32 reasoning tokens, so it cannot serve
as an answer-quality verdict. A model reload or new throughput claim awaits
NPU availability; this batch has not been deployed to Threadripper.
