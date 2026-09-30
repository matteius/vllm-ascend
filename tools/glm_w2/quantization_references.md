# GLM-5.3-Flash quantization references for 310P

The active text checkpoint uses W4 for routed-expert layers 3–32 and W2 for
33–44. Its indexed routed codes and scales total 122.607 GiB, about 30.652
GiB per rank under four-way expert parallelism. One W4-to-W2 layer change
saves 0.421875 GiB per rank. The current W2/W4 kernel accepts only those two
widths; each layer's local expert bank has one common code width. See
[`weight_and_gate_up_plan.md`](weight_and_gate_up_plan.md) for the indexed
weight accounting.

| Reference | Relevant organization | Published evidence and limits |
| --- | --- | --- |
| [EXL3 2.25 bpw](https://huggingface.co/r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121) | Its [tensor manifest](https://huggingface.co/r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121/blob/main/quantization_config.json) assigns 30 routed layers 2 bits and 12 layers 3 bits; attention and shared experts are mostly 4 bits, head 6 bits. | 98.5 GB artifact; author reports 174/200 on Q200v2 in ExLlamaV3. This is not a matched comparison with our runtime or source quantizer. |
| [GSQ-RCO GGUF](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF) | Allocates Q2/Q3/Q4 by tensor under exact 3.0- and 3.5-bpw budgets. | 117.48/137.07 GB text GGUFs; MMLU-Pro 60.00%/60.55% versus 61.95% for its Q8 reference, on 2,000 no-reasoning questions. The author's teacher-KL search kept its initial allocation. |
| [Canada GPTQ W4A16](https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP) | Group-128 symmetric INT4 on all 36,288 trunk expert matrices; attention, routers, shared experts, mHC, head and MTP retain source precision. | Calibrated on 256 × 4,096-token in-distribution samples. A method/quality reference, but its group scales and full W4 bank need their own 310P fit calculation. |
| [Intel AutoRound W4A16](https://huggingface.co/Intel/GLM-5.3-Flash-W4A16-AutoRound) | Group-128 INT4 with learned rounding; excludes attention, indexer, router, head, shared expert and mHC from quantization. | Card reports 0.8386 average over four tasks versus 0.8399 BF16, using its own evaluation recipe. |
| [NVIDIA NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) | FP4 E2M1 codes with fine per-16 scales and selected FP8 KV. | Published benchmark scores close to BF16. Our direct NVFP4 conversion retained its codes but the per-16 scale grid was too large for full 310P residency as implemented. |
| [Vontra MLX Q2](https://huggingface.co/Vontra/GLM-5.3-Flash-MLX-2bit-MTP) | Q2 trunk experts, Q8 non-expert language projections and indexer, Q4 MTP, BF16 embeddings/head/vision. | 111.318 GB artifact; author reports generation and MTP-parity checks. No matched broad quality comparison is published on the card. |

The EXL3 manifest gives the most actionable layer map for a first sensitivity
experiment. Its 3-bit routed layers are **3–5, 14, 24, 33 and 39–44**; all
other trunk routed layers are 2-bit. This raises precision on the last six
layers where our checkpoint uses W2. The map is a candidate, not a transferable
accuracy claim: EXL3 uses a trellis codebook, output scales and a different
runtime. Its routed-expert stored payload is about 87.4 GB versus about 131.6
GB for our selected routed payload, a theoretical difference near 10.3 GiB
per rank. Reproducing that format requires new 310P packing and decode math.

The near-term compatible experiment is a fixed-memory W2/W4 reassignment:
measure each layer's effect on teacher-logit KL, chosen-token rank, later-layer
activation cosine and representative generation quality; preserve early layers
and layers 19–26 unless direct evidence supports lowering them; test W4 on
late sensitive layers before changing the total W4 count. A matched prompt,
runtime, context and sampling recipe is required. The 310P is reserved for a
separate experiment, so this work remains a reference and offline plan.

Generation speed is the present priority. The grouped packed expert kernel is
the measured decode hotspot; quantization changes are deferred until a stable
performance baseline and kernel parity are available.
