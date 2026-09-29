# Native W4A8 c1 weight-pipeline result

The retained TP4/EP native-INT4 build streams sparse expert weights only for
the qualified c1 decode shape (`rows <= 30`, `N = 320`). Higher-concurrency
shapes instantiate the original resident-weight schedule at compile time. This
split preserves aggregate throughput while reducing single-stream latency.

## End-to-end result

The baseline and candidate used the same two Atlas 300I Duo cards, real
Qwen3.8 Flash Next checkpoint, MTP2, full decode graphs, 262,144-token model
limit, four-sequence scheduler limit, and CPU affinity. Warmup runs are not
included in the medians.

| Workload | Resident baseline | c1 pipeline | Change |
| --- | ---: | ---: | ---: |
| c1 decode, median | 28.1735 tok/s | **29.8610 tok/s** | **+5.99%** |
| c1 decode, peak | — | **30.3191 tok/s** | — |
| c4 aggregate decode, median | 59.6759 tok/s | **59.5233 tok/s** | -0.26% |

Candidate c1 runs were 29.9512, 30.3191, 29.8610, 29.7040, and 29.6911
tok/s. Candidate c4 aggregate runs were 62.1914, 59.5233, and 58.2283 tok/s.
The c4 median difference is within observed run-to-run variance.

The fixed 228-question zero-shot quality sample scored **206/228 with zero
invalid answers**. This is a fixed sampled gate, not an official full-dataset
MMLU score. Across the complete validation service run, interval metrics
reported 7,412 accepted speculative tokens from 9,078 drafted tokens
(81.65%).

Five real-NPU changing-input graph tests passed: c1 gate and down projections,
120-row c4 fallback for both `N = 1280` and `N = 2560`, and fused down with
changing inputs, routes, and weights. The focused host tests cover c1
eligibility, c4 sparse-many-expert fallback, packed layout, and event ordering.

## Why compile-time dispatch matters

An unrestricted streamed-weight schedule made c1 faster but reduced c4 to
about 39 tok/s. A runtime `rows <= 30` guard did not recover c4 because the
pipeline state and instructions were still compiled into the shared schedule
specialization. Separating the c1 and c2-c4 template instantiations removed
that pressure: c1 uses the streamed-weight pipeline and c2-c4 compile without
it.

The retained service uses:

- runtime: `/srv/ai/src/native-int4-w4a8.KiuhBN/runtime-w4-weight-pipeline-c1-dispatch`
- custom operators: `/srv/ai/src/w4-weight-pipeline-c1-dispatch-20260929/vllm_ascend/_cann_ops_custom/vendors/custom_transformer`
- fallback operator package: `/srv/ai/src/native-int4-w4a8.KiuhBN/opp-retained-good-20260928`
- served model: `qwen38-w4-pipeline-c1-dispatch`

The host launchers `~/start_qwen38_flashnext_mtp_graph.sh` and
`~/start_qwen38_pass4_unified_test.sh` default to this exact configuration.
The implementation is commit `456e1ebcd` on `main`.
