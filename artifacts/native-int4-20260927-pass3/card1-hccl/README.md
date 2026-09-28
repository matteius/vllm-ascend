# Card1 TP2 HCCL/runtime sweep

This directory contains the isolated physical-card `16409` (`ASCEND_RT_VISIBLE_DEVICES=2,3`) measurements. All successful throughput runs use real weights, the native `cube_310_int4_a8` backend, TP2+EP, MTP2, `gpu_memory_utilization=0.965`, `VLLM_ASCEND_KV_CACHE_FRACTION=0.80`, `max_num_seqs=4`, and 512 generated tokens per stream.

## Completed throughput

| Runtime | Affinity | c1 decode | c3 aggregate median | c4 aggregate median |
| --- | --- | ---: | ---: | ---: |
| `TASK_QUEUE_ENABLE=1`, AIV unset | workers `16-21` / `24-29`; API and Engine unrestricted | 19.447 tok/s | 32.904 tok/s | 31.720 tok/s |
| Same runtime | API `22-23`; Engine `30-31`; same worker groups | not repeated | 32.674 tok/s | 30.423 tok/s |

Strict two-core API/two-core Engine isolation changed c3 by -0.70% and c4 by -4.09%. It was abandoned. The c3 baseline runs were 28.486, 33.597, and 32.904 tok/s; the first run is a visible cold outlier, so medians are used. Baseline c4 runs were 32.040, 31.625, and 31.720 tok/s.

The concurrent card0 structural experiment reported 20.142 tok/s at c1, versus this card's 19.447 tok/s. Candidate gains should therefore be compared with the baseline on the same physical card rather than directly comparing raw card0 and card1 rates.

## Context and memory effects

With task queue mode 1, the 80K/maxseq4 launch reported 5.73 GiB available KV memory but needed 6.46 GiB; the planner estimated a 70,656-token maximum. The otherwise matched 65,536-token service started successfully.

`TASK_QUEUE_ENABLE=2` reduced available KV memory to 5.03 GiB in the pre-reboot attempt. A 65,536-token launch needed 5.31 GiB and failed, with a planner estimate of 61,824 tokens. Against the contemporaneous TQ1 profile, this was about 0.70 GiB less KV headroom per NPU. This delta is approximate because subsequent post-reboot profiling was asymmetric.

The first TQ2@60K attempt was stopped after one heavy shard to serialize checkpoint I/O. A later concurrent attempt loaded all 48 large expert shards in 7:25, then entered severe major-page-fault thrash while two TP2 services loaded a 169.19-GiB checkpoint on a 121-GiB host. After reboot, an isolated TQ2@60K load completed in 504–506 seconds but still failed capacity: the limiting rank exposed 4.67 GiB, below the required 4.88 GiB, and the planner estimated a 57,344-token maximum. Rank0 separately logged 7.30 GiB; that non-limiting value must not be used as service capacity. A 56K relaunch was canceled by the user at 3/1,610 shards before throughput validation. TQ2 is deferred for a later two-card study; there is no TQ2 tok/s result.

The three `serve-card1-hccl-*.sh` files preserve the 65K baseline, 60K TQ2, and canceled 56K TQ2 launch declarations. Environment snapshots, compressed server logs, affinity records, and every raw benchmark JSON are beside this report. `summarize.py` deterministically regenerates `summary.json`.
