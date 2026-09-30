# Upstream vLLM Ascend merge review (2026-09-30)

`vllm-project/vllm-ascend` main was fetched at `62e05feb3db521230c27714ad4347bd0d9d38f1a`. Its merge base with the local branch is `7ab47e73d005756963e3c6dd0e491296401b1f19`: 502 upstream commits and 331 local commits diverged from that base. The merge used `-X ours` for overlapping hunks. The local `glm5next` package, `attention/indexer_kpool.py`, and `core/kv_cache_interface.py` were retained as whole files because an automatic blend changed the KPool call signature while leaving our 310P caller in place. In particular, it produced duplicate `positions` arguments and removed local cache fields. These files must be ported as an interface unit if we move to upstream's new KPool backend.

## Performance changes to transfer to the 310P GLM path

| Upstream commit | Useful idea | Local disposition |
| --- | --- | --- |
| `7c227e9bc` | Skip incomplete/invalid KPool compression; fuse score sanitation, index expansion, and causal-tail packing; avoid unused gated-norm statistics. Upstream reports about 7% TPOT improvement on A3 W8A8 at C1/C8. | Triton A3/A5 implementation does not run in our 310P AscendC sparse indexer. Port the early-exit and fused output logic only after measuring our indexer share of decode time. |
| `e152f7f2d` | Write KDA decode output directly into the captured destination, including padding, instead of zeroing then copying. | Our 310P KDA uses a different recurrence kernel and already has a fused MLA writer. Check the remaining KDA output copy in a fresh profile before adapting. |
| `341982c83` | Reduce convolution-state copy work for strided caches. | Triton kernel targets a different cache layout. Compare our 310P copy traffic first. |
| `547fde3ab` | Keep FP16 MoE gate projection in NZ instead of casting activation/weight to FP32. | The upstream 310P fused-MoE runner receives this change, but our GLM W2 route uses the external `GateLinear` and custom expert method. Test router top-k parity before changing its precision. |
| `8d3eb1461` | Share compatible Mamba state slots across cache groups to restore context capacity. | Merged into the generic 310P runners; our GLM host-context path has separate cache plumbing. Check capacity and aliasing before enabling it for GLM. |

## Validation and deployment boundary

- The merge is source-only. It did not change the running Threadripper server or use the NPU.
- The local workstation has `vllm 0.22.0` and no `torch_npu`; upstream's newer APIs require a matching runtime. CPU GLM tests that avoid those imports passed (156 cases). Full collection and hardware performance/parity are pending on a compatible environment.
- Upstream's release marker is vLLM `v0.30.0`, but `Dockerfile.310p` still overrides its vLLM build arguments with the local `opensensor/vllm:qwen4exp-310p-dispatch-rope` fork. The fork is described there as based on an older upstream commit. Do not rebuild or restart the existing GLM service from this merged source until the fork is rebased or the required Qwen changes are carried onto the matching vLLM release and tested.
- A Python source compilation pass covers all changed `.py` files. This catches syntax errors, but it cannot validate imports or AscendC kernels.
- Before deploying: align the upstream vLLM/CANN dependencies in an isolated environment, run GLM W2 real-weight smoke and 310P KPool parity, then compare s=1 and s=4 decode/TTFT against the pre-merge profile. Keep the existing service until these gates pass.
