# GLM-5.3-Flash 310P context offload plan

## Current constraints

The W2 deployment uses 11 sparse MLA layers and 34 KDA layers, tensor
parallel size 2, a 512-token logical KV block, a 512-wide FP16 MLA latent,
and one FP16 128-wide index key per four tokens. The checkpoint advertises
1,048,576 positions. The last profiled serve allocated about 3.19 GiB of KV
memory per rank and admitted 16,384 tokens with four request slots. These
numbers describe that serve configuration, not a validated capacity limit.

The full MLA history costs 11 KiB per token per rank. Four 256K-token
contexts therefore need about 11 GiB of MLA history per rank. The compressed
indexer history costs 704 bytes per token, or 704 MiB for the same four
contexts. KDA recurrent state and the incomplete kpool tail need only current
state per live request. The compact KDA and bounded score changes in this
branch address those two sources of unnecessary NPU memory use.

The current GLM planner still allocates MLA and indexer tensors using one
global scheduler block-ID range. Even if a state group's physical pages are
small, its virtual IDs increase the size of the large MLA allocation. A host
tier must split physical page addressing from scheduler IDs; replacing the
MLA tensor with a CPU tensor alone cannot make the existing attention kernel
read it.

## Implementation sequence

1. Keep the compressed indexer and live KDA state on NPU. Put the full MLA
   latent history in a bounded pinned host pool. Account for host pages and
   NPU pages separately in the KV planner, including the null block and
   in-flight prefill blocks.
2. Add a request-stable mapping from scheduler block IDs to host MLA pages.
   Release it on request completion and preemption. Keep current prefill
   writes causal and copy complete latent pages to host in batches.
3. Allocate a bounded NPU hot cache for the current prefill chunk and sparse
   selections. After kpool top-k, deduplicate selected MLA page IDs, gather
   them from host in one batched transfer, and remap token IDs to hot slots
   before sparse MLA attention. Join the transfer stream once per step.
4. Validate the 310P sparse MLA kernel's hot-cache addressing. If it assumes
   global page IDs, add a focused 310P gather/remap kernel; keep the model
   and cache-manager interfaces generic.
5. Start with no prefix caching, no MTP, TP=2, and a fixed maximum of four
   live requests. Add prefix reuse and speculative decoding after page
   identity and state restoration are verified.

## Gates before serving longer contexts

- CPU: scheduler admission for four long requests, block-ID reuse, request
  completion/preemption, and exact selected-token equivalence against the
  NPU-resident reference on small tensors.
- NPU, when available: real-weight output equivalence at 16K, then 32K/128K,
  then four 256K windows; record host transfer bytes, hot-cache hit rate,
  prefill time, decode tokens/s, NPU peak memory, and host RSS.
- Reject any context length for which the host pool, compressed indexer,
  transfer buffer, or hot cache cannot be reserved before startup.
