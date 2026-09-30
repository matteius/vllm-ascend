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

## GLM KDA and GDN overlap audit

These upstream changes landed while the local GLM 310P path was being optimized. A merged commit is not necessarily an active kernel in our serving path. The current launcher, `artifacts/glm-profile-20260930/serve-fp16-mhc.sh`, runs `Glm5NextW2ForCausalLM` with `--enforce-eager` and no speculative configuration. Our W2 adapter calls `glm5next_w2/kda_310.py`, `npu_causal_conv1d_310`, `npu_recurrent_gated_delta_rule_310`, and `chunk_kda_fwd` rather than upstream's Triton GLM KDA wrapper.

| Upstream change | What it actually changes | Effect on current GLM 310P server | Action |
| --- | --- | --- | --- |
| `e152f7f2d` KDA decode writeback | The upstream Triton recurrent wrapper writes directly into the graph output buffer on pure decode, avoiding a zero plus copy. | Not active: we retained the local GLM package. Our recurrent C++ binding allocates an output tensor, and the local KDA integration can still copy outputs into the caller's buffer. | Profile the post-expert-optimization decode trace. If that copy is material, add an optional output tensor to the 310P op and test padded, mixed and speculative batches. |
| `341982c83` convolution state copies | The upstream Triton helper packs strided state rows by row/channel tiles, then writes them back after convolution. | Not active: our `npu_causal_conv1d_310` receives the persistent `conv_state` directly. There is no matching Python pack/writeback pair in `kda_310.py`. | Inspect the native operator's cache traffic before adapting the Triton algorithm; avoid introducing a redundant staging copy. |
| `5a84871b2` GLM MTP graph | Gives multi-KV draft steps stable per-step slot buffers and graph-capture metadata; upstream reports A3 W8A8 MTP graph gains. | Incomplete for us: generic proposer code merged, but the local KPool metadata builder was retained and the live server is eager without MTP. | Treat graph MTP as a separate integration after the vLLM version alignment and a correct 310P KPool parity test. |
| `7e2c563f5` FLA GDN prefill | Adds an external `fla_npu` chunk GDN path selected by A2/A3/A5 hardware capabilities. | No direct GLM gain: 310P lacks `FLA_GDN_PREFILL`, and GLM KDA uses per-channel bounded decay rather than the generic GDN gate contract. | Do not substitute this kernel for GLM KDA without deriving and validating the different gate math. |
| `f2d529279` gate-transpose reuse | Reuses one `[B,H,T]` contiguous cumulative-gate layout across the Triton FLA GDN chunk kernels. | No direct GLM gain: our 310P GLM prefill calls its AscendC KDA chunk op with `raw_gate` in BSND layout. | Search for duplicate gate layout conversion in a new 310P KDA prefill trace before porting the idea. |

The recorded pre-optimization decode trace assigned roughly 14–15 seconds of a 29.8-second 16-iteration one-stream span to grouped expert projections. The NZ-packed code layout greatly reduced that isolated projection cost, but there is no matching post-fusion operator attribution yet. A fresh trace is needed before ranking KDA writeback or graph work above expert dequantization. The newest equal-scale NZ multiply change (`3afc45460`) is mathematically exact but remains unmeasured on NPU.

## Validation and deployment boundary

- The merge is source-only. It did not change the running Threadripper server or use the NPU.
- The local workstation has `vllm 0.22.0` and no `torch_npu`; upstream's newer APIs require a matching runtime. CPU GLM tests that avoid those imports passed (156 cases). Full collection and hardware performance/parity are pending on a compatible environment.
- Upstream's release marker is vLLM `v0.30.0`, but `Dockerfile.310p` still overrides its vLLM build arguments with the local `opensensor/vllm:qwen4exp-310p-dispatch-rope` fork. The fork is described there as based on an older upstream commit. Do not rebuild or restart the existing GLM service from this merged source until the fork is rebased or the required Qwen changes are carried onto the matching vLLM release and tested.
- A Python source compilation pass covers all changed `.py` files. This catches syntax errors, but it cannot validate imports or AscendC kernels.
- Before deploying: align the upstream vLLM/CANN dependencies in an isolated environment, run GLM W2 real-weight smoke and 310P KPool parity, then compare s=1 and s=4 decode/TTFT against the pre-merge profile. Keep the existing service until these gates pass.
