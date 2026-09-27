# W4A16 310P hardware smoke

## Scope and isolation

Maintenance-window validation on 2026-09-26 EDT (2026-09-27 in the host logs).
The W8 server was gracefully stopped for this test. Its model directory, Python
environment, source checkout, and home-directory launcher were not modified.
W4 uses its separate checkpoint, source checkout, a private copy of the pinned
Python environment, and loopback port 8002. No package upgrades or checkpoint
re-quantization were performed.

This is the `eager_dequant` reference backend, not a packed W4 GEMM kernel.
No model runtime changes were required for the three completed responses.

## Reproducibility

- Hardware: four Ascend 310P3 logical devices on two Atlas 300I Duo cards.
- Python 3.12.13; torch 2.13.0+cpu; torch_npu 2.13.0.rc1; CANN 9.1.0.
- vLLM: `3ab5dda29acabea01f6a63d0806bdbbb4a27bde5`.
- Plugin: `ce1862e52947870f06e28f0667662298b6391a83`.
- Model: `/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i`.
- Source: `/srv/ai/src/qwen38-w4-ce1862e52`.
- Private environment: `/srv/ai/venvs/qwen38-w4-test-ce1862`.
- Logs and launch/test scripts: `/srv/ai/src/qwen38-w4-hardware-20260927`.

The existing 310P extension and `_cann_ops_custom` assets were copied into the
isolated source; they were not rebuilt. The extension SHA256 is
`f079d9a462a9ea9ffe6a47a68e7bdee2d771f27e6172550ba404c8331f98b943`.
The test explicitly puts the isolated plugin and pinned vLLM source on
`PYTHONPATH`; import provenance was printed by the operator smoke. This avoids
modifying the production editable installation.

Launch flags:

```bash
python -m vllm.entrypoints.cli.main serve \
  /srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --served-model-name qwen38-w4-experimental --host 127.0.0.1 --port 8002 \
  --dtype float16 --tensor-parallel-size 4 --enforce-eager \
  --no-async-scheduling --disable-custom-all-reduce \
  --max-model-len 8192 --max-num-batched-tokens 512 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 --language-model-only \
  --enable-expert-parallel --enable-ep-weight-filter \
  --reasoning-parser qwen3 --mamba-cache-mode align --enable-chunked-prefill \
  --limit-mm-per-prompt '{"image":0,"video":0}'
```

Source the pinned CANN and ATB setup first; use `SOC_VERSION=ascend310p1`,
`VLLM_ASCEND_ENABLE_310P=1`, `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`,
`TASK_QUEUE_ENABLE=1`, and `OMP_NUM_THREADS=1`. The test has no MTP, graph,
generic `--quantization`, or new runtime environment-variable settings.
The memory utilization is 0.90, below the established 0.965 ceiling. A watchdog
terminates only this test server at 96°C.

## Operator and loading results

- All 256 possible packed bytes unpacked exactly against the CPU result.
- Three real layer-0/expert-0 projections (gate, up, down) dequantized exactly
  against the CPU result. FP16 linear-output maximum absolute errors were
  0.00024414, 0, and 0.00048828 respectively.
- Five synchronized warm iterations measured 4.222 / 4.118 / 4.118 ms for
  dequantization plus matmul per projection. These are microbenchmarks, not
  whole-model throughput or a fused-kernel measurement.
- All 1,610 checkpoint shards loaded; expert ownership was 128/512 per rank.
  Rank-0 weight loading took 90.97 seconds; runner initialization about 97 s.
- Each rank reported **18.6917 GiB** model-load memory. The log labels this
  `GB`, but the implementation divides bytes by `2**30`.
- Profiling reported 0.39 GiB peak activation and no graph memory. Post-warmup
  torch allocation was 23.08 GiB/rank. This is an 8k, single-request profile,
  not a measured long-context or concurrent-window limit.
- The earlier W8 + MTP service reported 33.0402 GiB model memory/rank. Its
  MTP and runtime settings differ, so the entire difference must not be
  attributed to W4. The audited routed-expert payload saving alone is
  13.5791 GiB/rank at TP4.

The generic loader prints that no quantization signature was detected and the
model will load as float. Here, generic quantization is deliberately absent;
the model-specific W4 metadata selects packed INT8-byte expert banks. The
strict W4 loader and its complete local-expert inventory checks completed.

## HTTP results

Requests use temperature 0, seed 1024, and
`chat_template_kwargs={"enable_thinking": false}`. A pass requires the complete
expected answer, `finish_reason=stop`, and the stream's `[DONE]` marker.

All three requests passed with correct completed answers. Token usage includes
the terminating token.

| Case | Prompt / completion tokens | TTFT (s) | Total (s) | Decode (tok/s) |
| --- | --- | --- | --- | --- |
| `17 * 19` → `323` | 28 / 4 | 30.319 | 43.387 | 0.2296 |
| Comma-separated integers 1–50 | 31 / 190 | 31.560 | 857.620 | 0.2288 |
| Python list comprehension → `[0, 4, 16]` | 46 / 11 | 39.115 | 82.763 | 0.2291 |

The complete streams and pass marker are in `http-smoke-r1.jsonl`. No worker
exception occurred; maximum observed temperature was 78°C. The W4 server then
exited after a targeted SIGTERM, and all four workers released their devices.

For reported decode throughput use
`(completion_tokens - 1) / (elapsed_s - ttft_s)`. The initial client log used
the last visible delta timestamp as the end but counted the terminating token;
that raw `decode_tokens_per_second` field is optimistic and must be corrected
from `usage`, `elapsed_s`, and `ttft_s`. End-to-end throughput includes prefill.
This client-side decode estimate assumes one token before the first visible
delta; grouped speculative output can make it less accurate for short answers.
Longer fixed-length comparisons and engine timing are required for performance
acceptance.

## Regression checks

- CPU packing/export tests: **21 passed**.
- New 310P regression: **7 passed**, covering all byte encodings and both real
  projection geometries at 1, 7 and 64 tokens. CPU/NPU dequantization is exact;
  FP16 projections pass tolerances and the expert banks stay byte-packed.

```bash
python -m pytest --noconftest -q \
  tests/e2e/nightly/310p/single_node/ops/test_qwen4exp_w4_310.py
```

Run with the isolated environment and source setup above. The hardware test log
is `npu-tests-r1.log` (50.69 seconds; dependency deprecation warnings only).

The required `bash format.sh ci` ran in a disposable checkout to avoid changing
unrelated files. It failed on existing repository-wide lint issues, including
`csrc/torch_binding_meta.cpp:655`. Changed-file manual hooks pass with only the
global `check-symbolic-meta` hook skipped. Logs are local at
`/tmp/qwen38-w4-hardware-format.fFNA9p/format-ci.log` and `scoped-lint.log`.

## Restored W8 baseline

The original home launcher was restarted unchanged on port 8001, retaining
160,000 configured context, MTP k=1, decode graphs and utilization 0.965.
Its SHA256 remains
`ae81d75eec7516fc6f455a3a071e52410ba6c477c34e0114df647f4c125c77df`.
All three matching smoke requests passed. The service remains running in tmux
`qwen38-w8-restored`, API PID 1588870; `/health` and `/v1/models` returned 200
after benchmarking. The restored startup/request log snapshot is
`w8-restored-server-r1.log`; graph capture completed and no worker errors were
found. No new W8 performance code was introduced.

Six sustained runs used three distinct coding/service-analysis prompts, each
at short and approximately 23.4k-token context, with 512 forced completion
tokens, temperature 0, seed 42, reasoning disabled, and k=1 speculation.
The long prefix cache was warmed before measurement. These length-limited
runs measure throughput, **not completed coding-task correctness**. They are
three different prompts per context, not three statistical repeats of one
prompt.

| W8 workload | Server decode range (tok/s) | Median (tok/s) | Draft acceptance range |
| --- | --- | --- | --- |
| 32–49 prompt tokens, 512 output tokens | 18.515–20.282 | 19.073 | 78.4–93.9% |
| 23,407–23,424 prompt tokens, 512 output tokens | 18.068–19.223 | 18.091 | 81.6–91.8% |

Server timing uses 511 token gaps and matches client estimates closely. The
first uncached long-prefix warmup had TTFT 45.476 s; measured cached long
requests had TTFT 0.897–1.223 s. The prefix cache does not remove the decoded
tokens' approximately 23.4k context.

Raw streams are `baseline-w8-short-r1.jsonl` and `baseline-w8-long-r1.jsonl` in
the remote evidence directory; the original benchmark module and prefix are
under `/srv/ai/src/qwen38-decode-study-20260926`. The prefix SHA256 is
`804bfae0b86d5238706bc8823f23b5d74039d952b170efc8a3336af2577e57f4`.
[HARDWARE_RESULTS.json](HARDWARE_RESULTS.json) records exact timings, request
IDs, usage, acceptance, source/launcher hashes and the normalized W4 results.

W4 has not run those sustained 512-token/23.4k-context workloads: it remains an
8k eager smoke candidate. Thus the data establish the comparison target, not
a matched W4/W8 production-speed claim. The active performance goal is to
match and exceed measured W8, retaining packed memory and correct output; the
historical 15 tok/s figure is not the only acceptance threshold.

## Limits and next gate

This backend spends about four seconds per decode token in the first request.
Its selected-expert dequantization and Python/host routing are unsuitable for
the 15 tok/s production target. A 310P groupwise packed-W4 matmul and
device-side expert dispatch are the next performance work; keeping an expanded
FP16 or INT8 shadow bank is not an acceptable way to claim W4 memory savings.

Short smoke answers are not a coding-quality/perplexity evaluation. W4 MTP,
ACLGraph, longer context, concurrent windows, other TP sizes, and multimodal
requests remain unvalidated. Startup cache-capacity estimates are not evidence
for those features.
