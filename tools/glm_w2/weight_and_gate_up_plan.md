# GLM routed-weight budget and gate/up fusion

The Threadripper checkpoint `GLM-5.3-Flash-W4through32-noclip-310p` has overlay
shards. Counting every tensor in every shard overstates the active checkpoint:
the `model.safetensors.index.json` weight map selects **141.282 GiB**, while the
index's stale `metadata.total_size` reports 151.58 GiB. The selected tensors
contain 121.500 GiB of routed-expert codes and 1.107 GiB of their scales,
excluding 1.714 GiB of unwired MTP expert tensors. At four-way expert
parallelism, the routed bank is about **30.652 GiB per rank** before allocator
effects. The prior server reported 35.6075 GB loaded weights per rank. Thus
the packed routed bank is the main HBM pressure; trimming small norms or
router tensors cannot recover several GiB.

The resident packed bank now stores gate and up rows in one allocation. The
grouped projection reads both as one matrix and splits the FP16 output into
gate/up views before SwiGLU. Per-expert and per-projection views remain for the
eager path. This preserves the packed W2/W4 byte count and scale dtype while
removing one grouped operator launch per routed MoE layer. A width-changing
overlay rebuilds both views together to avoid mixing old W2 and new W4 rows.

CPU tests check shared storage, overlay replacement, and exact fused-versus-
separate output in a grouped projection harness. The actual 310P operator and
full-model speed have **not** been tested while the NPU is reserved for another
experiment. Before selecting this as a serving optimization, compare the same
model session with and without fused gate/up banks: projection latency at one
and four streams, decode tokens/s, peak HBM, and a greedy output/logprob sample.
The fusion should save launch overhead but cannot recover meaningful HBM; the
existing selected-expert host offload or a fidelity-preserving expert format
change is needed to free multiple GiB for full-W4 weights or device KV cache.
