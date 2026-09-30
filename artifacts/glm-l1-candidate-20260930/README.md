# GLM grouped W2/W4 resident-L1 candidate

The grouped 310P kernel now keeps one 32-output-channel decoded NZ tile in L1
across all rows of an expert group. It copies the decoded FP16 tile from UB to
L1, stages activations through two L1/L0 buffers, and sends both operands to
Cube. This removes the decoded-weight GM write and read used by the NZ-packed
grouped path. Canonical `uint8` grouped inputs and the single-expert operator
retain their existing GM-workspace path.

For K=4096 the decoded B tile uses 256 KiB of L1; two 128x128 FP16 activation
stages use another 64 KiB. K=2048 uses half as much B space. The candidate
compiles for 310P and is installed separately on Threadripper at
`/srv/ai/src/glm-l1-candidate-20260930/opp-l1`.

Hardware parity and speed remain unmeasured. The initial parity command
failed during `torch_npu.npu.set_compile_mode` because its `PYTHONPATH`
overwrote CANN's Python paths, hiding `tbe`; it did not reach any kernel case.
NPU testing is deferred while another user experiment is queued.

When NPU use is authorized again, run the existing grouped parity test with
the candidate vendor first in `ASCEND_CUSTOM_OPP_PATH`, preserving the CANN
environment's Python path. The expanded NZ-packed test covers W2 and W4 at
one, sixteen, and 129 rows. Benchmark in separate Python processes using
`artifacts/glm-profile-20260930/bench-nzpacked-candidate.py --nzpacked`:
once with the prior vendor and once with the candidate vendor. The output
files contain deterministic tensors for parity and operator timings.

Do not switch the serving OPP path based on this build alone. Compare FP16
outputs and per-projection latency first, followed by integrated decode and
prefill throughput at the existing model settings.
