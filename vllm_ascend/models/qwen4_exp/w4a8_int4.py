# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental W4A8: two signed INT4 Cube products, never FP16 weight GEMM.

Activation quantization is per row and per 128-element weight group. It is a
NEW approximation relative to W4A16 and requires real-model accuracy gates.
The decomposition itself is exact: a8 = lo + 16*hi + 8, with lo,hi in [-8,7].
Thus dot(a8,q-z) = dot(lo,q) + 16*dot(hi,q) + 8*sum(q) - z*sum(a8).
"""

import os

import torch

GROUP_SIZE = 128
N_TILE = 16
INT4_FRACTAL_K = 64
INT8_MAX = 127
INT4_BIAS = 8
INT4_RADIX = 16
NATIVE_INT4_BACKEND = "cube_310_int4_a8"
MAX_NATIVE_WEIGHT_PACK_THREADS = 6


def pack_activation_device(inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Fused NPU quantization/packing with aligned broadcast metadata.

    The reference quantizer remains available for independent numerical gates.
    Missing kernels fail explicitly; activation precision never changes here.
    """
    if inputs.device.type != "npu":
        raise ValueError("native INT4 activation packing requires NPU inputs")
    if not hasattr(torch.ops._C_ascend, "npu_qwen_w4_a8_pack_310"):
        raise RuntimeError("native INT4 requires the rebuilt activation pack operator")
    return torch.ops._C_ascend.npu_qwen_w4_a8_pack_310(inputs)


def swiglu_pack_activation_device(gate_up: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Fuse FP32 SwiGLU, FP16 rounding, and native-INT4 activation packing."""
    if gate_up.device.type != "npu":
        raise ValueError("native INT4 SwiGLU packing requires NPU inputs")
    if not hasattr(torch.ops._C_ascend, "npu_qwen_w4_a8_swiglu_pack_310"):
        raise RuntimeError("native INT4 requires the rebuilt SwiGLU pack operator")
    return torch.ops._C_ascend.npu_qwen_w4_a8_swiglu_pack_310(gate_up)


def pack_nibbles(values: torch.Tensor) -> torch.Tensor:
    return ((values[..., ::2] & 15) | ((values[..., 1::2] & 15) << 4)).to(torch.int8)


def pack_float_nibbles(values: torch.Tensor) -> torch.Tensor:
    """Exact small-integer FP32 arithmetic avoids 310P INT16 bitwise AiCPU casts."""
    unsigned = torch.where(values < 0, values + INT4_RADIX, values)
    packed = unsigned[..., ::2] + INT4_RADIX * unsigned[..., 1::2]
    signed = torch.where(packed >= 128, packed - 256, packed)
    return signed.to(torch.int8)


def quantize_activation_limbs(inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Device-only, shape-static preparation; no observed-value host branches."""
    if inputs.ndim != 2 or inputs.shape[1] % GROUP_SIZE or inputs.dtype != torch.float16:
        raise ValueError("W4A8 inputs must be FP16 [rows,K] with K divisible by 128")
    rows, width = inputs.shape
    groups = inputs.float().reshape(rows, width // GROUP_SIZE, GROUP_SIZE)
    maximum = groups.abs().amax(-1)
    scales = torch.where(maximum == 0, torch.ones_like(maximum), maximum / INT8_MAX)
    quant = (groups / scales.unsqueeze(-1)).round().clamp(-INT8_MAX, INT8_MAX)
    high = torch.floor(quant / INT4_RADIX)
    low = quant - INT4_RADIX * high - INT4_BIAS
    return (
        pack_float_nibbles(low).reshape(rows, width // 2).contiguous(),
        pack_float_nibbles(high).reshape(rows, width // 2).contiguous(),
        scales.contiguous(),
        quant.sum(-1).contiguous(),
    )


def pack_native_weight(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Losslessly store signed nibbles as [N/16,G,2,16,32] Cube B tiles."""
    outputs, packed_k = tensor.shape
    if tensor.dtype != torch.int8 or outputs % N_TILE or packed_k % (GROUP_SIZE // 2):
        raise ValueError("native INT4 weight must be INT8 bytes [N,K/2], N%16=K%128=0")

    # Inference deliberately runs with one OpenMP thread per TP worker, but
    # host-side checkpoint packing is a large elementwise transform. Use the
    # worker's assigned CPU set while packing, then restore the inference
    # setting. This avoids adding loader threads that would read peer experts
    # or retain multiple checkpoint shards in RAM.
    affinity_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    pack_threads = min(MAX_NATIVE_WEIGHT_PACK_THREADS, affinity_count)
    previous_threads = torch.get_num_threads()
    change_threads = pack_threads > previous_threads
    if change_threads:
        torch.set_num_threads(pack_threads)
    try:
        groups = packed_k // (GROUP_SIZE // 2)
        nibbles = torch.stack((tensor & 15, (tensor >> 4) & 15), -1).flatten(-2)
        signed = torch.where(nibbles >= INT4_BIAS, nibbles - INT4_RADIX, nibbles)
        sums = signed.reshape(outputs, groups, GROUP_SIZE).sum(-1).half()
        packed = tensor.reshape(outputs // N_TILE, N_TILE, groups, 2, INT4_FRACTAL_K // 2)
        return packed.permute(0, 2, 3, 1, 4).contiguous().view_as(tensor), pack_native_metadata(sums)
    finally:
        if change_threads:
            torch.set_num_threads(previous_threads)


def pack_native_metadata(tensor: torch.Tensor) -> torch.Tensor:
    # Offsets and sums fit FP16 exactly. FP16 metadata allows aligned 32-byte
    # copies of 16 channels, including the last group, without GM overreads.
    return tensor.reshape(tensor.shape[0] // N_TILE, N_TILE, -1).transpose(1, 2).contiguous().view_as(tensor)
