# GLM 5.3 Flash Performance and Context Plan for Four Ascend 310P Chips

- **Status:** Draft for implementation, 1 October 2026
- **Deployment:** Four 48 GiB Ascend 310P chips on two cards, tensor and expert parallel size four
- **Working checkpoint:** `GLM-5.3-Flash-W4through32-noclip-310p`, with mixed W2/W4 routed experts
- **Hardware status:** NPU testing is deferred until the user releases the devices.

## Decision and outcome

The last measured eager GLM build proves that the checkpoint can load and generate, but it is not yet a usable long-context service. A matched short-prompt run delivered 2.236 generated tokens/s for one request and 4.476 tokens/s aggregate for four; an 8,232-token prompt took 210.3 seconds to prefill. The 32-token long-context response ended during reasoning, so it does not establish answer quality. The latest CPU-tested K-pool and MoE source changes have not been served on the NPUs.

This plan makes generation speed the first optimization target while treating final-answer quality, prefill time, context capacity, memory, and crash-free operation as independent release gates. Preserve GLM's packed W2/W4 experts, KDA equations, latent MLA cache, kpool selection, and mHC math. Borrow the scheduling, graph, and tile-design techniques that worked for Qwen where the GLM operator contract permits them. Do not relabel the model or convert all its experts to Qwen W4 merely to fit a Qwen kernel.

The [post-fusion trace and serving report](../../../../artifacts/glm-profile-20260930/post-fusion-20260930/README.md), [offline decode design](../../../../tools/glm_w2/offline_performance_design.md), [context offload plan](../../../../tools/glm_w2/context_offload_plan.md), and [quantization quality plan](../../../../artifacts/glm-w2-quant-quality-plan.md) are the evidence base. Older bring-up documents describe earlier code states; use their measurements only with their recorded launch configurations.

## Product requirements and measured baseline

The product is one text-serving endpoint on port 8001 with the real GLM checkpoint and four-chip parallelism. It must return completed final answers through the GLM reasoning parser and accept the configured tool-call parser. Vision, a new quantization release, and public checkpoint publication are outside this performance pass. The model's advertised 1,048,576-position limit is an architecture property, not a validated serving claim.

| Measure | Last measured or inspected state | Next required evidence |
| --- | --- | --- |
| One-request decode | 2.236 generated tokens/s, 25-token prompt, 32-token completion, eager TP4 | Repeat with at least 256 generated tokens and parser-enabled launch |
| Four-request decode | 4.476 generated tokens/s aggregate, four active requests, same short workload | Record aggregate and per-request rates, fairness, and full output |
| 8K prefill | 8,232 input tokens in 210.3 s, then 2.16 decode tokens/s | Separate model prefill, host transfer, and API scheduling time |
| Output quality | Short arithmetic answer was correct but leaked `</think>`; 8K sample stopped during reasoning | Completed final answers with `glm47` parser and enough output budget |
| Context | 16,384 configured; an opt-in host MLA history path exists with CPU coverage | Real-weight equivalence, capacity, and latency at 16K, 32K, 128K, then concurrent 256K |
| Stability | A prior compact-state build crashed the KDA kernel after an invalid state index; subsequent eager TP4 served | Recheck state mapping and long-running mixed traffic on the exact promoted build |
| Decode trace | Grouped expert tasks took about 129 ms/step on one captured rank at one stream and 393 ms/step on one rank at four streams; collective wait was mostly rank arrival | Complete all four-rank attribution on a matched, current build |

These are measurements of specific prior builds, not speed estimates for untested source changes. The 32-token rates are useful for regression triage but too short to establish sustained generation speed. The unprofiled, parser-free run and a parser-enabled future run must be labeled separately.

## Proposed acceptance targets

Targets are product decisions, not observed results. The first two gates establish whether an optimization is worth another model reload; only the release gate makes the endpoint a recommended GLM service. Compare each candidate with a contemporaneous known-good server using the same checkpoint, context, request mix, warmup, and measurement script.

| Gate | One-request decode | Four-request aggregate decode | Prefill and context | Quality and stability |
| --- | ---: | ---: | --- | --- |
| A: correct baseline | No numeric speed target | No numeric speed target | 8K and 16K complete | Parser-separated final answers; fixed prompts and low-level parity pass |
| B: useful experiment | At least 5 tok/s | At least 10 tok/s | 8K prefill at most 120 s; one 32K request completes | No answer regression against gate A; no AICore faults |
| C: useful service | At least 10 tok/s | At least 20 tok/s | 8K prefill at most 60 s; four 32K windows complete | Full quality suite, memory headroom, and extended mixed-load run pass |
| Release target | At least 15 tok/s | At least 30 tok/s | 8K prefill at most 45 s; four 128K windows with host history | Same quality and stability gates on the exact served build |
| Stretch target | 30 tok/s | 60 tok/s | Four 256K windows without truncation or unsafe memory pressure | Explicitly measured, not inferred from Qwen's result |

Decode rates count all generated tokens, including reasoning tokens, from the first to last streamed token. Report time to first token, total completion time, output token count, and per-request p50/p95 as well. A short arithmetic prompt cannot satisfy a quality gate by itself. If hardware measurements show that one target is physically untenable, retain the result and revise the target with a measured bottleneck and a capacity model rather than quietly lowering it.

Use the same workload ladder at each gate: a 25-token prompt with 256 generated tokens at one and four streams; an 8K prompt with 256 generated tokens; a 32K retrieval prompt; and the current gate's four concurrent context windows. Use fixed seeds and record any early EOS. A completion shorter than the planned output length is reported separately, not silently treated as a faster decode run. Keep a short 32-token case only for rapid fault detection.

## Correctness and quality contract

Before performance comparisons, run the recorded [coherence probe](../../../../artifacts/glm-profile-20260930/post-fusion-20260930/probe-coherence.py) against a parser-enabled server. It uses the checkpoint's supported `reasoning_effort` template field. Its exact-answer and marker checks establish that a response reached final content instead of merely beginning a plausible reasoning trace. Expand it into a fixed suite of at least 20 deterministic arithmetic, instruction, code, and retrieval prompts, plus retrieval at 8K, 32K, and each promoted context tier. Record prompt tokens, output budget, finish reason, reasoning, final content, and reference answer or scoring rule.

For a kernel or graph change with unchanged weights, compare its operator output against the current quantized GLM implementation at the same inputs and states. Require exact packed-code decoding and FP16 parity where the operator is intended to be identical; where reduction order changes, set per-operator error bounds before measuring and inspect router IDs, selected kpool groups, recurrent state, and final logits. For a deliberate quantization change, use the fixed answer suite and layerwise comparison with the higher-precision reference. Preserve headwise ReLU in GLM kpool scoring; moving it after the head sum changes the selected history.

At every promoted gate, final content must have no raw `<think>` markers, and the server must complete the requested answer rather than exhaust a too-small output budget during reasoning. Test tool calls separately with `--enable-auto-tool-choice` and the configured `--tool-call-parser`; a request that merely avoids an HTTP error is not a successful tool call. Preserve the exact tokenizer, chat template, parser, and checkpoint revision in each result.

## Memory and context contract

Budget each chip separately. The known mixed-quant checkpoint was reported as 151.58 GiB on disk and about 35.6075 GB of loaded weights per rank in one profile. Those numbers need reconciliation with a fresh allocator breakdown because on-disk bytes, live tensor bytes, allocator reservation, and CANN workspaces differ. At the current `gpu_memory_utilization=0.965`, record available and peak bytes for weights, KDA state, MLA history, compressed indexer, hot pages, temporary expert tiles, graph captures, HCCL, and fragmentation. A release configuration must keep enough uncommitted HBM for its declared concurrency; startup must fail clearly when it cannot.

The [host MLA path](../../../../tools/glm_w2/context_offload_plan.md) is an opt-in correctness implementation: it writes history to host RAM and synchronously stages selected pages back to the NPU. It has CPU tests but no NPU output or throughput qualification. Its current behavior can make decode slower even while increasing capacity. First prove output equivalence to resident MLA at 16K. Then add a bounded resident hot-page cache, page reuse across decode steps, batched asynchronous transfers, and metrics for transferred bytes, hit rate, transfer overlap, NPU peak, and host RSS. The compressed kpool indexer and live KDA state remain on the NPU. Validate 32K and 128K before trying four 256K requests.

Keep the present W2/W4 expert precision map unless quality measurements justify a different one. Full W4 repacking consumes much more HBM and takes capacity away from context. Do not expand GLM's latent MLA history into full per-head K/V merely to call Qwen's sparse kernel. Reuse Qwen's four-token group scheduling where it matches GLM's semantics, and keep GLM's kpool gate, absolute-position embedding, causal tail, and latent cache.

## Work packages in priority order

Each package has a measurable hypothesis and a stop rule. A source change stays an isolated candidate until parity and matched serving measurements pass. Avoid a full model restart for an unmeasured kernel idea.

### 1. Establish the exact baseline and finish the trace

Run the parser-enabled launch in `artifacts/glm-profile-20260930/post-fusion-20260930/serve-compact-kda-8001.sh` only when the NPUs are released. Confirm the latest CPU-tested kpool, router, and combine changes on device. Capture one and four streams, 256-token completions, 8K prefill, peak HBM, and all four rank timelines. Identify the per-step critical path, not just the sum of kernel durations; separate prefill from decode and collective transit from rank-arrival wait. Reproduce the output suite before modifying the math.

**Exit:** A versioned artifact contains launch arguments, code and binary hashes, checkpoint revision, prompt and response records, per-rank timeline, and the measurements in the baseline table. If this build faults, repair and validate the first failing kernel/state transition before any speed experiment.

### 2. Reduce packed expert projection time

The current [grouped W2/W4 operator](../../../../vllm_ascend/_310p/quantization/methods/w2_dynamic.py) keeps codes resident and groups routes on device, but its decode trace is the leading measured device target. The existing direct-to-L1 candidate passed parity and was 10–29% slower on decode-shaped workloads; binary search over the 72 local expert boundaries did not improve isolated medians. Do not repeat those designs unchanged.

Prototype one fixed-shape decode schedule that visits active expert groups, tiles W2/W4 decode and Cube work to avoid repeated full-tile traffic, and fuses gate/up when that saves a launch. Treat singleton and repeated-expert groups separately. Test zero local routes, peer-owned routes, repeated IDs, W4 gate/up, W2 down, and both 8 and 32 routed rows. Preserve the current FP16 rounding contract. The Qwen W4 kernel's input-width limit is 2560, while GLM needs 4096; extending its buffer and tiling design is an option, but copying its on-disk W4 format is not a requirement.

**Promotion rule:** Byte and output parity first; then at least a reproducible 15% reduction in the relevant isolated decode projection under a clean build with verified binary hash; then at least a 10% matched end-to-end decode improvement with no quality or memory regression. If isolated speed improves but end-to-end speed does not, inspect the new critical path and stop promotion.

### 3. Remove per-token launch and allocation overhead

Audit GLM KDA projection, three short convolutions, bounded per-channel decay, recurrent update, output normalization, mHC mixing, MoE route sorting, shared expert, and final combine. Keep the already fused mHC Sinkhorn and merged KDA projection as baselines. A graph is attractive only after the eager model is coherent and its cache metadata is fixed. The prior FULL_DECODE_ONLY trial failed at boolean advanced indexing in the kpool writer during capture; the subsequent 507015 synchronizations were downstream errors. Replace capture-hostile dynamic indexing with bounded, device-side metadata and validate state replay, preemption, and multi-request padding before enabling decode graphs or MTP.

**Promotion rule:** Exact state and output equivalence for eager versus replay, followed by sustained one- and four-stream improvement. Disable any graph path that changes selected history, leaks stale recurrent state, or adds enough memory to defeat the context gate.

### 4. Make kpool and MLA scale with context

Retain GLM's 2048-token sparse budget and four-token compression groups. Fuse tiled score calculation only if a current trace shows it on the critical path; preserve per-head dot product, ReLU, weighting, and reduction order. The 342-page physical-stride QSA probe reduced an isolated copy-heavy call from about 2.87 ms to 0.062 ms, but the subsequent model run improved only modestly. Maintain physical-stride addressing and measure the complete prefill and decode path. Investigate the 8K prefill's roughly 13–14 seconds per 512-token chunk before prioritizing another decode-only cache micro-optimization.

**Promotion rule:** Selection parity at boundary positions and page reuse, no excess NPU history allocation, plus an end-to-end prefill or long-context decode gain on the same request shape. Report whether time moved into host staging or another layer.

### 5. Qualify speculative decoding only after the base path

GLM provides one MTP layer. Test MTP-1 only after eager quality and decode graph replay are stable. Measure accepted draft tokens, verifier cost, total generated tokens/s, memory, and output parity under deterministic sampling. Acceptance rate alone is not a speed result. Keep MTP disabled if it degrades the one- or four-stream serving gates.

## Experiment protocol and decision record

1. Keep a known-good launch and OPP package. Build a candidate in an isolated directory, rebuild changed native objects cleanly, and record source and binary SHA-256 values. The earlier stale-object incident produced misleading parity and timing, so source diff alone cannot establish which kernel ran.
2. Use a single operator harness before a full checkpoint reload. Compare identical tensors, warmups, synchronization, route distribution, and active quant widths. Measure at least the 8-row and 32-row decode cases and a prefill case; include repeat variance and workspace bytes.
3. Promote one candidate at a time to the real model. Run the same prompts and server flags for baseline and candidate, including `--max-model-len`, `--max-num-seqs`, chunk size, quant map, memory utilization, parsers, and graph mode. Do not compare profiler-active rates with unprofiled rates.
4. Save `tokens/s` together with time to first token, 8K and long-context prefill, final-answer scoring, HBM allocated/reserved/peak, host RSS, fault logs, and per-rank activity. Compute aggregate and per-request rates from actual streamed token timestamps.
5. Record **adopt**, **revise**, or **reject** for each candidate with the measured delta and the next bottleneck. Reject a faster isolated operator when it changes answers, reduces context capacity, or fails to improve serving.

## Delivery sequence and dependencies

| Phase | Depends on | Deliverable |
| --- | --- | --- |
| Baseline and quality | NPU availability | Parser-enabled, full-answer baseline; complete four-rank trace; memory ledger |
| Expert kernel candidate | CPU parity harness and baseline trace | Isolated W2/W4 package with 8/32-row and prefill results |
| Decode integration | Expert parity and clean binary identity | Matched one- and four-request serving comparison |
| Graph and MTP | Eager coherence, stable kpool metadata, recurrent state replay | Separate graph and MTP results, each with parity and memory |
| Context service | Resident 16K equivalence | Host-history hot-page policy, 32K and 128K qualification |
| Release decision | Quality, speed, context, and stability gates | Reproducible launch, checkpoint manifest, result bundle, rollback command |

Source and CPU tests may proceed while the NPUs are reserved. NPU-dependent phases wait for explicit availability. The first return to hardware should validate the current code and collect the missing evidence; it should not bundle another untested kernel, graph, and quantization change into one model restart.

### Immediate source work while hardware is reserved

1. Extend the existing coherence probe into the fixed answer suite and add a workload runner that emits one machine-readable record per request. Reuse the current 8K prompt and existing serving artifacts; include parser fields and full token timing so the first NPU run can be evaluated without editing the client.
2. Add CPU tests for the kpool writer's graph-safe metadata path, covering partial pools, multiple requests, cache-block reuse, padded decode slots, and zero-work rows. Replace capture-hostile indexing only after these reference cases define the expected outputs.
3. Prepare a single expert-kernel candidate with explicit 4096-input tiling, W2/W4 packed-code contracts, and an isolated parity/latency harness. Record the expected workspace and tile traffic in advance. Leave the serving operator selection on the known-good path until hardware measurements meet its promotion rule.

## Implementation constraints

Keep model-specific behavior in the existing GLM adapter and custom operators. New environment controls belong in `vllm_ascend/envs.py` with documented defaults and review. New model runner behavior or patches need an architectural review and a focused regression test. Avoid device `.item()` and host synchronization in generation hot paths. Add unit tests for every new feature and regression; run NPU parity and end-to-end tests when use is authorized. Record any numerical approximation explicitly and require the quality gate before promotion. Use signed Conventional Commits for implementation changes and run `bash format.sh ci` before pushing.
