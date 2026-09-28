# Cast, layout, and copy audit

The four-rank c3 profile identifies grouped expert routing as the largest
removable conversion chain. Across 48 layers, its equality-matrix count path
launches 48 FP32-to-INT64 AI-CPU casts and 48 cumsums per decode iteration.
Those operations account for about 6.66 ms and 3.46 ms respectively. The two
route sorts per layer add about 1.32 ms, and seven gathers add about 0.85 ms.

The first optimization is therefore to keep the 90-route and 120-route c3/c4
shapes on the native routed kernel. Raising its bounded capacity from 80 to 128
eliminates the grouped descriptor rather than optimizing each descriptor
operation separately. A direct one-layer ACL-graph replay benchmark measured:

| Routes | Grouped | Routed-128 | Speedup |
|---:|---:|---:|---:|
| 90 | 1.513 ms | 1.336 ms | 1.132x |
| 120 | 1.690 ms | 1.533 ms | 1.102x |

The outputs matched bit-for-bit. The focused hardware gate passed 50 tests at
90, 120, and 128 routes, including changing expert IDs and graph replay. The
complete operator suite passed 132 tests.

Remaining candidates, ordered by measured ceiling:

1. Replace the grouped equality matrix, sort, count, and cumsum with one
   fixed-shape INT32 routing operator for shapes above the routed bound.
2. Cache or fuse the PLE four-tap depthwise convolution layout conversion,
   measured at about 2.31 ms per decode iteration.
3. Preformat the static PLE projection weight, measured at about 0.83 ms.
4. Group the MTP W8A16 expert dispatch. Its per-expert transpose/TransData work
   costs about 1.45 ms and varies by rank, so it can amplify HCCL arrival skew.
5. Route bounded positions through INT32 before FP32 (about 0.39 ms) and cache
   the transposed GDN convolution filter (about 0.20 ms).

The fused native activation pack is already small at about 0.98 ms per decode
iteration. Dynamic ND-to-NZ activation conversions remain necessary unless an
upstream operator emits NZ directly. Graph-break buffer copies in PLE and MTP
must continue overwriting their stable graph-pool buffers on every replay.
