# Qwen3.8 Flash Next grouped-MTP result (2026-09-30)

This pass moves the MTP routed experts from host-dispatched W8A16 expert
matmuls to an opt-in packed W8A8 grouped path. Selection is explicit through
`text_config.ascend_expert_quantization.mtp_expert_execution=w8a8_grouped`;
the default remains `w8a16_routed`.

The candidate ran the real checkpoint on two Atlas 300I Duo cards (four Ascend
310P3 devices), TP4/EP, 262,144 configured context, MTP2, full decode graphs,
`max_num_seqs=4`, and `max_num_batched_tokens=512`. The model name and port
remained `qwen38-w4-pipeline-c1-dispatch` and 8001.

| Workload | Retained backend | Grouped MTP | Change |
| --- | ---: | ---: | ---: |
| c1 decode median, five 512-token runs | 29.8610 tok/s | 29.7374 tok/s | -0.41% |
| c1 decode peak | 30.3191 tok/s | 34.3382 tok/s | +13.26% |
| c4 aggregate median, three 4x1024-token runs | 59.5233 tok/s | 60.8789 tok/s | +2.28% |

The c4 runs were 61.3700, 60.8789, and 59.4403 aggregate decode tok/s. All
four requests ran concurrently with no queued requests. The c1 mean was
30.8504 tok/s; prompt-dependent MTP acceptance had a 70.75% median. A smoke
request correctly answered the “all but 9” sheep question. The earlier fixed
228-question quality gate was not rerun for this isolated grouped-MTP change.

The first device attempt exposed two CANN requirements now covered by tests:
grouped weight scales must be FP32, and INT8 expert banks must be converted
once after loading from logical `[expert, N, K]` tensors to FRACTAL_NZ. Focused
host coverage passed 71 tests with one skipped test, and the real service
completed model loading, memory profiling, graph warmup, coherent generation,
and the c1/c4 runs without runtime errors.

## Multi-request QSA and cold-prefill follow-up

The next measured pass retained the stable served model ID
`qwen38-w4-pipeline-c1-dispatch` on port 8001. Before the c3 failure, the
multi-request QSA path measured:

| Workload | Result |
| --- | ---: |
| Short c1 decode | 31.1801 tok/s |
| Short c2 aggregate decode | 44.2890 tok/s |
| 40K warm-prefix c2 aggregate decode | 30.7617 tok/s |
| 40K cold TTFT | 131.905 s |

The first c3 run failed with `decode_group_list does not cover all query
groups`. Each request scheduled three verification tokens under MTP2. With
three requests and two KV heads, the grouped attention operation needed 18
boundaries, but the buffer held only the single-request maximum of 16. The
buffer now scales with the scheduler's `max_num_seqs`, and host query lengths
split packed multi-request QSA selection and KV gathering against the correct
block-table row. Regression tests cover the capacity calculation, packed
request validation, multi-request result parity, and changing-input graph
replay.

Cold prefill was still constrained by the original 512-token/5,120-route
grouped-MoE chunk. The native-INT4 pack and grouped-matmul kernels grid-stride
bounded row tiles and now accept a 2,048-token/20,480-route chunk, which
permits a future native-W4 service launch with `--max-num-batched-tokens
2048`. The established W4A16 workspace remains at 512 tokens. Multi-request
prefill can also use the request-aware QSA matrix path instead of falling back
to scalar sparse attention.

The 2K cold-prefill configuration subsequently passed the 20,480-route
operator boundary and full-service checks. Three distinct 40,028-token cold
prompts measured 121.3474, 120.5582, and 119.9475 seconds TTFT, with subsequent
decode at 30.15, 32.10, and 32.38 tok/s. The best result is only 5.8% below the
127.3949-second 512-token baseline, so larger MoE chunks are stable but do not
explain the roughly 10x cold/warm gap.

Stage timing localized most cold time to the twelve QSA layers. At a 2,048
token chunk, attention costs about 246 ms per QSA layer and index selection
about 62 ms per QSA layer. Within each 64-query attention tile, key gather is
about 2.68 ms, QK is 1.02 ms, softmax is 0.77 ms, value gather is 2.64 ms, and
PV is 1.07 ms. Across roughly twenty chunks, QSA attention and selection
account for about 74 seconds of the 120-second cold prompt.

The offline implementation now reuses one max-tile NZ key/value workspace per
QSA invocation and shares two lazily-created gather streams across all QSA
layers. Key and value gathers run in parallel. The main stream joins the key
stream immediately before QK and joins the value stream immediately before PV,
allowing the value gather to overlap QK and softmax. A main-stream event before
each tile prevents the side streams from overwriting the reusable buffers
until the preceding PV consumer completes. Decode and graph replay retain the
single-stream grouped-matmul path.

A fused QSA kernel was also implemented and passed the focused numerical
checks. It gathers selected K/V from paged device memory into on-chip storage,
then performs QK, softmax, and PV without writing the selected tensors to NPU
global memory and reading them back. This removes a device-memory round trip;
there was no host-RAM transfer in the established gather path. The fused
kernel remains experimental because its 64-query and 2,048-query timings were
about 10.04 ms and 311.7 ms, versus 7.91 ms and 237.8 ms for the established
matrix path. A 128-query tile and double-buffer variant also regressed.

Host compilation, focused lint, and isolated stream-owner/request-boundary
checks pass. The full pytest harness on this workstation cannot collect
because its Python environment lacks `fla_npu`; NPU validation of the new
parallel-gather path is deferred at the user's request. The NPU nightly test
now covers the parallel stream path and a 65-token partial final tile for the
next hardware validation pass.
