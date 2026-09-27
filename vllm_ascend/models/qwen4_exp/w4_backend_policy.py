# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve W4 compute policy before weight layout selection and graph capture."""

from dataclasses import dataclass

import torch
from vllm.logger import logger

from .w4a8_int4 import NATIVE_INT4_BACKEND

NATIVE_OPERATORS = frozenset({"npu_qwen_w4_a8_pack_310", "npu_qwen_w4_a8_int4_matmul_310"})
ACTIVATION_POLICIES = ("float16", "int8_per_group")
MIN_NATIVE_K = 256
MAX_NATIVE_K = 2560
NATIVE_GROUP_SIZE = 128


@dataclass(frozen=True)
class BackendSelection:
    backend: str
    activation_quantization: str
    reason: str


def resolve_w4_backend(
    metadata: dict,
    *,
    hidden_size: int,
    intermediate_size: int,
    device_name: str,
    available_ops: frozenset[str],
    native_qualified: bool = False,
) -> BackendSelection:
    """Explicit overrides fail closed; auto retains FP16 until native qualifies.

    ``int8_per_group`` permits quantization; ``float16`` (also the legacy
    default) forbids it. Qualification requires model accuracy, MTP acceptance,
    changing-input graph replay and end-to-end performance evidence. It is
    supplied by deployment code, never inferred from a model name.
    """
    if metadata.get("bits") != 4:
        raise ValueError("W4 backend selection requires four-bit weight metadata")
    requested = metadata.get("backend")
    activation = metadata.get("activation_quantization", "float16")
    if activation not in ACTIVATION_POLICIES:
        raise ValueError(f"activation_quantization must be one of {ACTIVATION_POLICIES}")
    shape_supported = metadata.get("group_size") == NATIVE_GROUP_SIZE and all(
        MIN_NATIVE_K <= width <= MAX_NATIVE_K and width % NATIVE_GROUP_SIZE == 0
        for width in (hidden_size, intermediate_size)
    )
    hardware_supported = "310P" in device_name.upper()
    native_available = shape_supported and hardware_supported and available_ops >= NATIVE_OPERATORS
    if requested == NATIVE_INT4_BACKEND:
        if activation != "int8_per_group":
            raise ValueError("native INT4 requires explicit activation_quantization=int8_per_group permission")
        if not native_available:
            raise ValueError("native INT4 requires Ascend 310P, supported G128 shapes, and both rebuilt operators")
        return BackendSelection(requested, "int8_per_group", "explicit experimental native INT4 override")
    if requested != "auto":
        return BackendSelection(requested, "float16", "explicit W4A16 backend")
    if activation == "int8_per_group" and native_available and native_qualified:
        return BackendSelection(NATIVE_INT4_BACKEND, activation, "qualified native INT4 on supported hardware/shapes")
    if shape_supported and hardware_supported and "npu_qwen_w4_grouped_matmul_310" in available_ops:
        return BackendSelection(
            "cube_310_grouped", "float16", "W4A16; native INT4 is forbidden, unavailable or unqualified"
        )
    return BackendSelection("eager_dequant", "float16", "W4A16 reference for unsupported Cube hardware/shapes")


def configure_w4_backend(config: object) -> BackendSelection | None:
    """Freeze deployment selection before modules allocate packed weight banks.

    W8 configurations have no packed-W4 metadata and never enter this path.
    Native remains experimental, so automatic promotion is disabled until the
    recorded real-model gates pass. The per-model backend is the explicit
    override; no environment variable or global dtype switch is involved.
    """
    metadata = getattr(config, "ascend_expert_quantization", None)
    if not isinstance(metadata, dict) or metadata.get("backend") not in ("auto", NATIVE_INT4_BACKEND):
        return None
    device_name = torch.npu.get_device_name() if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"
    names = NATIVE_OPERATORS | {"npu_qwen_w4_grouped_matmul_310"}
    available_ops = frozenset()
    if device_name != "cpu":
        # Worker-local lazy registration: the device is already selected here,
        # but no module has yet chosen a weight layout or captured a graph.
        from vllm_ascend.utils import enable_custom_op

        if enable_custom_op():
            available_ops = frozenset(name for name in names if hasattr(torch.ops._C_ascend, name))
    selection = resolve_w4_backend(
        metadata,
        hidden_size=int(config.hidden_size),
        intermediate_size=int(config.moe_intermediate_size),
        device_name=device_name,
        available_ops=available_ops,
    )
    config.ascend_expert_quantization = {**metadata, "backend": selection.backend}
    logger.info(
        "W4 backend=%s activation_quantization=%s: %s",
        selection.backend,
        selection.activation_quantization,
        selection.reason,
    )
    return selection
