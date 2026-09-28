#!/usr/bin/env python3
"""Validate and time preformatted Qwen4Exp MTP W8A16 weights on 310P."""

from __future__ import annotations

import json
import statistics
import time
from types import SimpleNamespace

import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.dtype_policy import Qwen4ExpDtypePolicy
from vllm_ascend.models.qwen4_exp.moe import _w8a16_linear_npu
from vllm_ascend.models.qwen4_exp.mtp import _MTPFP16MoE
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


def synchronize() -> None:
    torch_npu.npu.synchronize()


def timed_ms(fn, *, blocks: int = 9, iterations: int = 50) -> tuple[float, list[float]]:
    for _ in range(10):
        fn()
    synchronize()
    trials = []
    for _ in range(blocks):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        synchronize()
        trials.append((time.perf_counter() - start) * 1000 / iterations)
    return statistics.median(trials), trials


def projection_case(name: str, rows: int, k: int, n: int, seed: int) -> dict[str, object]:
    torch.manual_seed(seed)
    device = torch.device("npu:0")
    x = (torch.randn(rows, k, dtype=torch.float16, device=device) * 0.05).contiguous()
    weight_nd = torch.randint(-127, 128, (k, n), dtype=torch.int8, device=device)
    scale = (torch.rand(n, dtype=torch.float16, device=device) * 0.001 + 0.0001).contiguous()
    weight_nz = torch_npu.npu_format_cast(weight_nd, ACL_FORMAT_FRACTAL_NZ)

    expected = _w8a16_linear_npu(x, weight_nd, scale)
    actual = _w8a16_linear_npu(x, weight_nz, scale)
    synchronize()
    exact = torch.equal(expected, actual)
    max_abs = float((expected - actual).abs().max().cpu())
    nd_ms, nd_trials = timed_ms(lambda: _w8a16_linear_npu(x, weight_nd, scale))
    nz_ms, nz_trials = timed_ms(lambda: _w8a16_linear_npu(x, weight_nz, scale))
    return {
        "name": name,
        "shape": {"rows": rows, "k": k, "n": n},
        "format": {
            "nd": int(torch_npu.get_npu_format(weight_nd)),
            "nz": int(torch_npu.get_npu_format(weight_nz)),
        },
        "exact": exact,
        "max_abs": max_abs,
        "nd_median_ms": nd_ms,
        "nz_median_ms": nz_ms,
        "speedup": nd_ms / nz_ms,
        "nd_trials_ms": nd_trials,
        "nz_trials_ms": nz_trials,
    }


def lifecycle_case() -> dict[str, object]:
    config = SimpleNamespace(
        num_experts=4,
        num_experts_per_tok=2,
        hidden_size=256,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=0,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )
    policy = Qwen4ExpDtypePolicy.for_310p()
    bank = _MTPFP16MoE(config, policy, (0, 1), quantize_experts=True).to("npu:0")
    with torch.no_grad():
        for projection in ("gate_up_proj", "down_proj"):
            for weight in getattr(bank, projection):
                weight.copy_(torch.randint(-127, 128, weight.shape, dtype=torch.int8, device=weight.device))
            for scale in getattr(bank, projection + "_scale"):
                scale.copy_(torch.rand_like(scale) * 0.001 + 0.0001)

    x = torch.randn(3, config.hidden_size, dtype=torch.float16, device="npu:0") * 0.05
    weight_nd = bank.gate_up_proj[0].detach().clone()
    scale = bank.gate_up_proj_scale[0]
    expected = _w8a16_linear_npu(x, weight_nd, scale)
    formats_before = [
        int(torch_npu.get_npu_format(weight))
        for projection in (bank.gate_up_proj, bank.down_proj)
        for weight in projection
    ]
    bank.process_weights_after_loading()
    formats_after = [
        int(torch_npu.get_npu_format(weight))
        for projection in (bank.gate_up_proj, bank.down_proj)
        for weight in projection
    ]
    actual = bank._expert_linear(x, "gate_up_proj", 0)
    synchronize()
    return {
        "formats_before": formats_before,
        "formats_after": formats_after,
        "all_nz": all(value == ACL_FORMAT_FRACTAL_NZ for value in formats_after),
        "exact": torch.equal(expected, actual),
        "max_abs": float((expected - actual).abs().max().cpu()),
    }


def main() -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires Ascend 310P")
    torch_npu.npu.set_device(0)
    result = {
        "device": torch_npu.npu.get_device_name(0),
        "lifecycle": lifecycle_case(),
        "projections": [
            projection_case("gate_up_r1", 1, 2560, 1280, 3101),
            projection_case("gate_up_r4", 4, 2560, 1280, 3104),
            projection_case("down_r1", 1, 640, 2560, 3201),
            projection_case("down_r4", 4, 640, 2560, 3204),
        ],
    }
    result["passed"] = (
        result["lifecycle"]["all_nz"]
        and result["lifecycle"]["exact"]
        and all(case["exact"] for case in result["projections"])
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
