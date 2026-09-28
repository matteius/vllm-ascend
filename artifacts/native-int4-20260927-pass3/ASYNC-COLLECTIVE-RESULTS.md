# Native INT4 asynchronous collective study

## Dataflow candidates

`replicated_deferred` starts the routed all-reduce on a model-owned HCCL
stream, computes a replicated shared expert on the main stream, and waits for
the reduction only before adding both outputs. Cross-stream tensor lifetimes
are protected with `record_stream`.

`tp_sharded_overlap` keeps the checkpoint's shared-expert weights TP-sharded.
For static decode shapes with exactly one row, it computes the local shared
partial on the existing shared-expert stream while routed experts run on the
main stream. It joins with an event, adds both local partials, and issues the
original single combined all-reduce:

```text
all_reduce(routed_local + shared_local)
```

Larger shapes retain sequential TP-sharded execution because concurrent GEMMs
contend for Cube resources on 310P.

## TP4 measurements

All measurements use four NPUs, MTP2, route-cache maximum 80, max sequence
count four, a 262,144-token context limit, and 0.965 device-memory utilization.

| Shared-expert execution | c1 tok/s | c3 tok/s | c4 tok/s | Result |
|---|---:|---:|---:|---|
| `tp_sharded` proven baseline | 24.690 | 39.07 | 50.07 | Reference |
| `replicated_deferred`, default-priority HCCL stream | 24.552 | 32.607 | 42.451 | Saturation regression |
| `tp_sharded_overlap`, unrestricted shapes | 25.584 | 23.587 | 45.591 | Helps c1; severe c3 contention |
| `tp_sharded_overlap`, at most two rows | — | 38.845 | 31.978 | MTP emits contended two-row calls at c4 |
| `tp_sharded_overlap`, one row | 24.855 | — | 48.348 median | Final opt-in candidate |

A high-priority (`priority=-1`) HCCL stream is invalid under 310P ACL graph
capture. All four ranks aborted in the first decode graph with HCCL watchdog
error `ERR02005`; the implementation therefore uses default stream priority.

The final candidate gates TP-sharded overlap to one-row static graph shapes.
It is opt-in through quantization metadata, and it does not change
activation precision, collective count, or the default W8/W4 execution paths.
Three repeated four-request batches with distinct inputs produced exact stable
outputs across graph replays.
