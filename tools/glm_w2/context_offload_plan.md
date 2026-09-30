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

The scheduler still allocates one global block-ID range. In host mode the
planner sizes the device pool from compressed indexer pages and reserves a
separate fixed MLA hot cache; the host MLA history uses the full logical ID
range. The QSA block table is remapped to hot 32-token pages for each call.

## Implemented opt-in path

Set `VLLM_ASCEND_310P_ENABLE_MLA=1` and
`VLLM_ASCEND_310P_GLM_HOST_KV=1` before starting the 310P worker. The
second flag defaults to off. Use an explicit `--max-model-len` and
`--max-num-seqs` so the planner can reserve the full logical block pool.
Pass `--enforce-eager` because host transfers are not graph-safe yet.
The planner checks the estimated host allocation against available RAM and
reserves an NPU hot cache large enough for the worst sparse selection of
every live request. The NPU still holds the compressed kpool indexer and
live KDA state. It caps logical pages at the configured maximum request
concurrency, so surplus device memory does not inflate the host allocation.

Each MLA layer stores its normalized 512-wide FP16 rows under the original
scheduler block IDs in host RAM. Before QSA attention, it copies selected
32-token pages to its NPU hot cache and remaps the QSA block table. Decode
stages one combined batch. Continued prefill batches consecutive queries
until their selected-page union reaches the fixed hot cache. Fresh short
prefill still uses the native flash path. Scheduler block reuse is safe
because all visible rows are overwritten before they are selected.

This first implementation is a correctness path: it synchronizes device to
host for writes and kpool selections and recopies selected pages at each
attention call. Decode throughput and long-prefill latency are expected to
be substantially worse than the all-NPU 16K baseline until page residency
and batched asynchronous copies are added. It has
CPU unit coverage only; NPU startup and output equivalence remain untested
because the current live NPU experiment is reserved for the user.

The opt-in path requires FP16 MLA cache, 310P, DCP=1, PCP=1, no prefix caching,
and no speculative decoding. Pinning host pages is attempted; pageable host
memory is used if that allocation fails. TP ranks maintain separate host
copies, and the planner reserves RAM for all local TP ranks.

## Remaining performance work

1. Validate the 310P QSA kernel's remapped 32-token page addressing against
   the existing all-NPU output at 16K, followed by 32K and 128K.
2. Keep recently selected host pages resident on NPU, avoid repeated D2H
   metadata transfers, and batch asynchronous H2D page copies.
3. Profile continued-prefill grouping and tune segment boundaries against
   QSA launch overhead and host transfer volume.
4. Measure four concurrent 256K requests on two cards. Only then consider
   prefix reuse and speculative decoding with explicit host-state restore.

## Gates before serving longer contexts

- CPU: scheduler admission for four long requests, block-ID reuse, request
  completion/preemption, and exact selected-page equivalence against the
  resident reference on small tensors. Selected-page and reuse unit tests
  pass; full scheduler admission is pending.
- NPU, when available: real-weight output equivalence at 16K, then 32K/128K,
  then four 256K windows; record host transfer bytes, hot-cache hit rate,
  prefill time, decode tokens/s, NPU peak memory, and host RSS.
- Reject any context length for which the host pool, compressed indexer,
  transfer buffer, or hot cache cannot be reserved before startup.
