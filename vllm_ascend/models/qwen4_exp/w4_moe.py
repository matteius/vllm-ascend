# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in packed W4A16 storage with reference and 310P Cube projections.

Only selected experts are dequantized; no persistent FP16 bank or W8 shadow
is created. The routed Cube backend keeps bounded decode routing on device;
the reference and group-only backends require eager execution.
"""

from __future__ import annotations

import regex as re
import torch
import torch.nn.functional as F
from torch import nn

from .dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY, Qwen4ExpDtypePolicy
from .moe import route_topk
from .weight_mapping import local_expert_range

FORMAT = "qwen4exp_w4a16_group_v1"
EXPERT_NAME = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale|weight_offset)$"
)
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
KINDS = ("weight", "weight_scale", "weight_offset")
MAX_EAGER_TOKENS = 256
MAX_CUBE_TOKENS = 128
CUBE_GROUP_SIZE = 128
CUBE_MAX_INPUTS = 2560
CUBE_TILE_OUTPUTS = 32
CUBE_ZERO_POINT_BIAS = 8
CUBE_FRACTAL_SIZE = 16
CUBE_BACKENDS = ("cube_310", "cube_310_tiled", "cube_310_routed")
CUBE_TILED_BACKENDS = ("cube_310_tiled", "cube_310_routed")
MAX_CUBE_ROUTES = 80


def w4_config(config: object) -> dict | None:
    metadata = getattr(config, "ascend_expert_quantization", None)
    if metadata is None:
        return None
    expected = {
        "format": FORMAT,
        "bits": 4,
        "symmetric": False,
        "packing": "signed_int4_low_nibble_first_in_axis",
        "scale_dtype": "float16",
        "offset_dtype": "int8",
    }
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("unsupported Qwen4Exp expert quantization metadata; refusing W8 fallback")
    backend = metadata.get("backend")
    if backend not in ("eager_dequant", *CUBE_BACKENDS):
        raise ValueError("W4 backend must be eager_dequant, cube_310, cube_310_tiled or cube_310_routed")
    group = metadata.get("group_size")
    if type(group) is not int or group <= 0 or group % 2:
        raise ValueError("W4 group_size must be a positive even integer")
    for field in ("hidden_size", "moe_intermediate_size"):
        if int(getattr(config, field)) % group:
            raise ValueError(f"W4 group_size must divide {field}")
        if backend in CUBE_BACKENDS and not 256 <= int(getattr(config, field)) <= CUBE_MAX_INPUTS:
            raise ValueError(f"W4 cube_310 requires 256 <= {field} <= {CUBE_MAX_INPUTS}")
    if backend in CUBE_BACKENDS and group != CUBE_GROUP_SIZE:
        raise ValueError("W4 cube_310 requires group_size=128")
    return metadata


def require_eager_w4(model_config: object, config: object) -> None:
    metadata = w4_config(config)
    if metadata is not None and metadata["backend"] != "cube_310_routed":
        if not getattr(model_config, "enforce_eager", False):
            raise ValueError("Qwen4Exp W4 host routing requires --enforce-eager; use cube_310_routed for decode graphs")


def unpack_signed_int4(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.int8:
        raise ValueError("W4 packed storage must be int8")
    # Signed shifts are deliberate; mask recovers the high two's-complement nibble.
    nibbles = torch.stack((packed & 15, (packed >> 4) & 15), dim=-1).flatten(-2)
    return torch.where(nibbles >= 8, nibbles - 16, nibbles)


def dequantize(packed: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, group_size: int) -> torch.Tensor:
    dtype = ASCEND_QWEN4EXP_DTYPE_POLICY.accumulation_dtype
    quant = unpack_signed_int4(packed).to(dtype).reshape(*scale.shape, group_size)
    return ((quant - offset.to(dtype).unsqueeze(-1)) * scale.to(dtype).unsqueeze(-1)).flatten(-2)


def pack_cube_tiles(tensor: torch.Tensor, kind: str) -> torch.Tensor:
    """Lossless one-expert load-time encoding; preserve shape/dtype/bytes.

    Codes: [N/32, K/128, 8, 16, 16], with the low/high nibbles holding
    output channels n and n+16. Each unpacked plane is already Cube NZ.
    Metadata: [N/32, K/128, 32]. Bias both codes and offsets by eight, so subtraction
    gives exactly the original signed q-offset without GPU sign extension.
    Public tensor shapes remain canonical for loader validation; the tiled
    operator must be selected explicitly to interpret their physical layout.
    """
    if tensor.ndim != 2 or tensor.shape[0] % CUBE_TILE_OUTPUTS:
        raise ValueError("W4 tile packing requires a matrix with N divisible by 32")
    rows = tensor.shape[0] // CUBE_TILE_OUTPUTS
    if kind == "weight":
        packed_group = CUBE_GROUP_SIZE // 2
        if tensor.shape[1] % packed_group:
            raise ValueError("W4 tile packing requires K divisible by 128")
        biased = unpack_signed_int4(tensor) + CUBE_ZERO_POINT_BIAS
        planes = biased.reshape(rows, 2, CUBE_FRACTAL_SIZE, -1, CUBE_FRACTAL_SIZE).permute(0, 3, 4, 1, 2)
        packed = planes[..., 0, :] | (planes[..., 1, :] << 4)
    elif kind in ("weight_scale", "weight_offset"):
        values = tensor + CUBE_ZERO_POINT_BIAS if kind == "weight_offset" else tensor
        packed = values.reshape(rows, CUBE_TILE_OUTPUTS, -1).transpose(1, 2)
    else:
        raise ValueError(f"unknown W4 projection field: {kind}")
    return packed.contiguous().view_as(tensor)


class PackedExpertBank(nn.Module):
    def __init__(
        self, experts: int, outputs: int, inputs: int, group_size: int, *, backend: str = "eager_dequant"
    ) -> None:
        super().__init__()
        if backend not in ("eager_dequant", *CUBE_BACKENDS):
            raise ValueError("unsupported W4 projection backend")
        self.group_size = group_size
        self.backend = backend
        dtype = ASCEND_QWEN4EXP_DTYPE_POLICY.main_dtype
        self.weight = nn.Parameter(torch.zeros(experts, outputs, inputs // 2, dtype=torch.int8), requires_grad=False)
        self.weight_scale = nn.Parameter(
            torch.zeros(experts, outputs, inputs // group_size, dtype=dtype), requires_grad=False
        )
        self.weight_offset = nn.Parameter(torch.zeros(self.weight_scale.shape, dtype=torch.int8), requires_grad=False)

    def linear(self, inputs: torch.Tensor, expert: int) -> torch.Tensor:
        if self.backend in CUBE_BACKENDS:
            if inputs.device.type != "npu" or self.group_size != CUBE_GROUP_SIZE:
                raise ValueError("W4 cube_310 requires NPU inputs and group_size=128")
            if not hasattr(torch.ops._C_ascend, "npu_qwen_w4_group_matmul_310"):
                raise RuntimeError("W4 cube_310 requires the rebuilt Qwen W4 custom operator; refusing silent fallback")
            return torch.ops._C_ascend.npu_qwen_w4_group_matmul_310(
                inputs,
                self.weight[expert],
                self.weight_scale[expert],
                self.weight_offset[expert],
                self.backend in CUBE_TILED_BACKENDS,
            )
        weight = dequantize(self.weight[expert], self.weight_scale[expert], self.weight_offset[expert], self.group_size)
        return F.linear(inputs, weight.to(inputs.dtype))

    def routed_linear(self, inputs: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        if self.backend != "cube_310_routed" or inputs.device.type != "npu":
            raise ValueError("W4 routed projection requires NPU inputs and cube_310_routed")
        if not hasattr(torch.ops._C_ascend, "npu_qwen_w4_routed_matmul_310"):
            raise RuntimeError("W4 routed projection requires the rebuilt custom operator; refusing silent fallback")
        return torch.ops._C_ascend.npu_qwen_w4_routed_matmul_310(
            inputs, self.weight, self.weight_scale, self.weight_offset, expert_ids
        )


class W4SparseMoE(nn.Module):
    """Same router/shared/TP contract as W8, but separate packed weight banks."""

    def __init__(self, *, config: object, dtype_policy: Qwen4ExpDtypePolicy, expert_sharding=(0, 1)) -> None:
        super().__init__()
        metadata = w4_config(config)
        if metadata is None:
            raise ValueError("W4SparseMoE requires explicit checkpoint metadata")
        hidden, intermediate = int(config.hidden_size), int(config.moe_intermediate_size)
        self.max_chunk_tokens = MAX_CUBE_TOKENS if metadata["backend"] in CUBE_BACKENDS else MAX_EAGER_TOKENS
        self.device_routing = metadata["backend"] == "cube_310_routed"
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.expert_tp_rank, self.expert_tp_size = expert_sharding
        if (
            self.expert_tp_size < 1
            or not 0 <= self.expert_tp_rank < self.expert_tp_size
            or self.num_experts < self.expert_tp_size
        ):
            raise ValueError("invalid W4 expert TP ownership")
        self.expert_offset, stop = local_expert_range(self.num_experts, self.expert_tp_size, self.expert_tp_rank)
        self.num_local_experts = stop - self.expert_offset
        self.params_dtype, self.compute_dtype = dtype_policy.main_dtype, dtype_policy.accumulation_dtype
        self.renormalize = bool(getattr(config, "norm_topk_prob", True))
        self.routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.0) or 1.0)
        self.gate = nn.Parameter(torch.zeros(self.num_experts, hidden, dtype=self.params_dtype))
        self.projections = nn.ModuleDict(
            {
                name: PackedExpertBank(
                    self.num_local_experts,
                    hidden if name == "down_proj" else intermediate,
                    intermediate if name == "down_proj" else hidden,
                    metadata["group_size"],
                    backend=metadata["backend"],
                )
                for name in PROJECTIONS
            }
        )
        shared = int(getattr(config, "shared_expert_intermediate_size", 0) or 0)
        self.has_shared_expert = shared > 0
        self.local_shared_inter = shared // self.expert_tp_size
        if shared % self.expert_tp_size:
            raise ValueError("shared expert intermediate dimension must divide TP")
        if self.has_shared_expert:
            self.shared_gate_up = nn.Parameter(
                torch.zeros(2 * self.local_shared_inter, hidden, dtype=self.params_dtype)
            )
            self.shared_down = nn.Parameter(torch.zeros(hidden, self.local_shared_inter, dtype=self.params_dtype))
            self.shared_expert_gate = nn.Parameter(torch.zeros(1, hidden, dtype=self.params_dtype))
        self._tp_reduce = None
        if self.expert_tp_size > 1:
            from vllm.distributed import tensor_model_parallel_all_reduce

            self._tp_reduce = tensor_model_parallel_all_reduce

    def load_projection(self, expert: int, projection: str, kind: str, tensor: torch.Tensor) -> str | None:
        if not 0 <= expert < self.num_experts:
            raise ValueError(f"W4 expert id out of range: {expert}")
        local = expert - self.expert_offset
        if not 0 <= local < self.num_local_experts:
            return None
        target = getattr(self.projections[projection], kind)[local]
        if tensor.dtype != target.dtype or tuple(tensor.shape) != tuple(target.shape):
            raise ValueError(
                f"W4 {projection}.{kind}: expected {target.dtype} {tuple(target.shape)}, "
                f"got {tensor.dtype} {tuple(tensor.shape)}"
            )
        if kind != "weight":
            if not torch.isfinite(tensor).all() or (kind == "weight_scale" and not (tensor > 0).all()):
                raise ValueError("W4 scales must be positive and quantization parameters finite")
            if kind == "weight_offset" and not ((tensor >= -8) & (tensor <= 7)).all():
                raise ValueError("W4 offsets must be signed-int4 zero points")
        with torch.no_grad():
            if self.projections[projection].backend in CUBE_TILED_BACKENDS:
                tensor = pack_cube_tiles(tensor, kind)
            target.copy_(tensor)
        return f"projections.{projection}.{kind}"

    def forward(self, block_input: torch.Tensor) -> torch.Tensor:
        weights, ids = route_topk(
            F.linear(block_input, self.gate),
            self.top_k,
            renormalize=self.renormalize,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        if self.device_routing and block_input.shape[0] * self.top_k <= MAX_CUBE_ROUTES:
            result = self._forward_routed(block_input, weights, ids)
        else:
            if self.device_routing and torch.npu.is_current_stream_capturing():
                raise RuntimeError(f"W4 decode graph exceeds {MAX_CUBE_ROUTES} routes; reduce capture sizes")
            result = self._forward_host_routed(block_input, weights, ids)
        if self.has_shared_expert:
            # Match production W8's NPU projection policy. Converting NZ FP16
            # weights to FP32 on every call both copies weights and selects an
            # aclop Cast that cannot be captured. Keep the CPU/eager reference
            # unchanged; the routed backend uses its resident FP16 weights.
            operand_dtype = (
                self.params_dtype
                if block_input.device.type == "npu" and self.projections["gate_proj"].backend == "cube_310_routed"
                else self.compute_dtype
            )
            inputs = block_input.to(operand_dtype)
            gate, up = F.linear(inputs, self.shared_gate_up.to(operand_dtype)).chunk(2, -1)
            shared = F.linear(F.silu(gate) * up, self.shared_down.to(operand_dtype))
            result += shared * torch.sigmoid(F.linear(inputs, self.shared_expert_gate.to(operand_dtype)))
        if self.expert_tp_size > 1:
            if self._tp_reduce is None:
                raise RuntimeError("W4 expert TP requires all-reduce")
            result = self._tp_reduce(result)
        return result.to(self.params_dtype)

    def _forward_routed(self, block_input: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        # Static route slots, dynamic device expert IDs. Peer routes produce
        # exact zero rows and must also overwrite rows on every graph replay.
        tokens, hidden = block_input.shape
        inputs = block_input[:, None, :].expand(-1, self.top_k, -1).reshape(-1, hidden).contiguous()
        local_ids = (ids - self.expert_offset).to(torch.int32).flatten().contiguous()
        gate = self.projections["gate_proj"].routed_linear(inputs, local_ids).to(self.compute_dtype)
        up = self.projections["up_proj"].routed_linear(inputs, local_ids).to(self.compute_dtype)
        activation = (F.silu(gate) * up).to(self.params_dtype)
        output = self.projections["down_proj"].routed_linear(activation, local_ids).to(self.compute_dtype)
        return (output.reshape(tokens, self.top_k, hidden) * weights.unsqueeze(-1)).sum(dim=1)

    def _forward_host_routed(self, block_input: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(block_input, dtype=self.compute_dtype)
        # One routing sync per bounded token chunk. Intentionally eager-only.
        for start in range(0, block_input.shape[0], self.max_chunk_tokens):
            routes: dict[int, list[tuple[int, int]]] = {}
            for token, token_ids in enumerate(ids[start : start + self.max_chunk_tokens].cpu().tolist()):
                for slot, global_id in enumerate(token_ids):
                    local = global_id - self.expert_offset
                    if 0 <= local < self.num_local_experts:
                        routes.setdefault(local, []).append((start + token, slot))
            for expert, selected in routes.items():
                indices = torch.tensor(selected, dtype=torch.long, device=block_input.device)
                tokens, slots = indices.unbind(-1)
                inputs = block_input.index_select(0, tokens)
                gate = self.projections["gate_proj"].linear(inputs, expert).to(self.compute_dtype)
                up = self.projections["up_proj"].linear(inputs, expert).to(self.compute_dtype)
                activation = (F.silu(gate) * up).to(self.params_dtype)
                output = self.projections["down_proj"].linear(activation, expert).to(self.compute_dtype)
                result.index_add_(0, tokens, output * weights[tokens, slots].unsqueeze(-1))
        return result


def validate_w4_inventory(layers: nn.ModuleList, names: set[str]) -> None:
    expected = {
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.{kind}"
        for layer, module in enumerate(layers)
        if isinstance(module.mlp, W4SparseMoE)
        for expert in range(module.mlp.expert_offset, module.mlp.expert_offset + module.mlp.num_local_experts)
        for projection in PROJECTIONS
        for kind in KINDS
    }
    if names != expected:
        raise ValueError(
            f"incomplete W4 expert checkpoint: missing={len(expected - names)}, extra={len(names - expected)}"
        )
