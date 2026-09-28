# Native INT4 performance checklist, reordered by measured ceiling

This checklist prioritizes end-to-end decode throughput. Operator latency is
only promoted when repeated real-model tok/s improves without a material
accuracy, MTP-acceptance, graph-replay, or context-capacity regression.

## Active two-card split

| Card | Candidate | Primary question | Promotion gate |
| --- | --- | --- | --- |
| NPUs 0,1 | replicated shared expert, then auxiliary-stream overlap | Can shared FP16 compute overlap the routed TP all-reduce and reduce rank-arrival skew? | c1/c3/c4 tok/s, graph capture/replay, fixed 228 questions, MTP acceptance |
| NPUs 2,3 | HCCL task queue, then AIV only if task queue helps | Can runtime scheduling reduce collective wait without changing model math? | normalized c1/c3/c4 tok/s and retained context capacity |

Checkpoint loading is the shared host bottleneck. The two services may load in
parallel, but throughput runs start only after both loads finish so storage and
CPU activity do not contaminate the result.

## Priority order

1. **Replicate and overlap the shared expert.** The current profile spends
   41--49% of rank-0 time in collectives, with 90--96% of collective elapsed
   time waiting for ranks to arrive. This change attacks both exposed shared
   MLP work and arrival skew. At TP4 it costs 337.5 MiB/NPU and about 28.8K raw
   QSA-token capacity. The serial replicated mode isolates numerical/loading
   effects before the overlap mode is measured.
2. **Task-queue/AIV runtime scheduling.** `TASK_QUEUE_ENABLE=2` already showed
   a 0.70 GiB/NPU allocation cost: the same TP2 service fell from a 65,536-token
   launch to an estimated 61,824-token ceiling. It must produce a meaningful
   normalized tok/s gain to justify that loss.
3. **Grouped W8A16 MTP expert dispatch.** The eager draft path still performs a
   host-visible expert count and small per-expert matmuls. It is a plausible
   source of rank skew and affects every speculative draft pass, but needs a
   dedicated 310P grouped W8A16 operator rather than an unsupported API call.
4. **Fuse the PLE four-tap convolution path.** The static depthwise filter
   layout conversion costs about 2.31 ms/iteration. A dedicated kernel can
   combine padding, convolution, SiLU, and residual work.
5. **Preformat remaining static layouts.** PLE projection is about
   0.83 ms/iteration; RoPE/GDN cleanups together are below 0.6 ms/iteration.
   These are useful after the structural experiments.
6. **Host-paged old KV for capacity.** This is a separate long-context mode.
   It must measure sparse-attention selected-token page faults, tok/s, and
   accuracy; it should not be mixed into decode-kernel promotion results.

## Route-128 result

Extending routed native INT4 from 80 to 128 rows makes the isolated 90-route
projection 1.132x faster and the 120-route projection 1.102x faster, removing
roughly 7.5--8.5 ms across 48 layers. The NPU operator suite passed 132 tests,
including changing-input and changing-ID graph replay at 120/128 rows. The
fixed real-model quality gate scored 204/228 with zero invalid answers, within
the accepted historical 203--206 range.

The initial full-model result looked 2--3% faster across physical cards, but
card 0 was 3.6% faster in the c1 normalization. After that correction, c3/c4
are approximately flat to 1.4% slower. Keep the change while larger structural
work is evaluated because its regression is small and its device work is
strictly lower, but do not claim an end-to-end tok/s win from it.

## Required final gates

- repeated c1/c3/c4 decode with identical token counts and worker-only CPU
  affinity;
- fixed 228-question zero-shot sample with dataset hash
  `de4f0a3b2a14bab908865733f639b66428116eee33f2958495814920745cd27b`;
- MTP accepted/drafted counts and acceptance by position;
- changing-input and changing-route-ID ACLGraph replay;
- short and long-context real-model generation;
- capacity accounting for every mode that reserves additional device memory.
