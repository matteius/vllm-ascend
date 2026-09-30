# GLM grouped W2/W4 resident-L1 candidate

The experimental grouped 310P path keeps one 32-output-channel decoded NZ tile
in L1 across all rows of an expert group. It copies the decoded FP16 tile from
UB to L1, stages activations through two L1/L0 buffers, and sends both operands
to Cube. This removes the decoded-weight GM write and read. The normal grouped
call uses the faster GM-workspace path; the single-expert operator is unchanged.

For K=4096 the decoded B tile uses 256 KiB of L1; two 128x128 FP16 activation
stages use another 64 KiB. K=2048 uses half as much B space. The tested L1
package is installed separately on Threadripper at
`/srv/ai/src/glm-l1-candidate-20260930/opp-l1`. That package has L1 enabled;
the repository's normal grouped entry point leaves it disabled.

The initial parity command failed during CANN setup because it overwrote
`PYTHONPATH`, hiding `tbe`. Later rebuilds initially reused stale 310P kernel
objects; regenerating the kernel targets was necessary after changing the
header. The corrected package passed all 15 grouped tests and ten repeated
W4 calls, including the first invocation. In the four decode-shaped A/B cases
and six dense cases, baseline and candidate FP16 outputs were bit-for-bit identical.

| Quant / workload | GM baseline | L1 candidate | L1 slowdown |
| --- | ---: | ---: | ---: |
| W4 / 8 routed rows | 1.077 ms | 1.206 ms | 12% |
| W2 / 8 routed rows | 1.294 ms | 1.420 ms | 10% |
| W4 / 32 routed rows | 4.054 ms | 5.214 ms | 29% |
| W2 / 32 routed rows | 4.951 ms | 5.787 ms | 17% |
| W4 / 64 rows, one expert | 0.672 ms | 1.005 ms | 50% |
| W2 / 64 rows, one expert | 0.782 ms | 1.117 ms | 43% |
| W4 / 128 rows, one expert | 0.802 ms | 1.552 ms | 93% |
| W2 / 128 rows, one expert | 0.922 ms | 1.677 ms | 82% |

The decode-shaped measurements use 72 experts with four rows in each active
group. Every result is the median of 20 synchronized operator calls after four
warmups, measured in separate Python processes. Denser measurements use one
expert and three warmups. Removing a full pipeline barrier made no meaningful
speed difference. The 32-channel Cube schedule and extra L1 staging cost more
than the decoded-weight GM round trip saves in these workloads. Keep the
current serving OPP and grouped GM path for now.

An additional 129-row test exposed incorrect output from the existing GM
path. The L1 package passed a CPU-reference case at a smaller geometry, but
its full GLM geometry has not been independently checked against CPU. The
normal entry point therefore does not select L1 as a large-group fallback yet.
