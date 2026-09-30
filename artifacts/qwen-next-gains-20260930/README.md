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
