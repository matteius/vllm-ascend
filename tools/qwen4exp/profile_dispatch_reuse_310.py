# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare exact inverse-permutation and identity-scale removal on Ascend 310P."""

import argparse
import importlib
import json
from pathlib import Path

import torch

from tools.qwen4exp.profile_projection_dtypes_310 import timed
from vllm_ascend.models.qwen4_exp.moe import route_topk


def make_cases(order, logits, npu=None):
    def old_router():
        weights, _ = logits.float().softmax(-1).topk(10, dim=-1)
        return weights / weights.sum(-1, keepdim=True) * 1.0

    cases = {
        "inverse_sort": lambda: torch.argsort(order.float()),
        "inverse_sort_via_int32": lambda: torch.argsort(order.to(torch.int32).float()),
        "inverse_scatter": lambda: torch.empty_like(order).scatter_(
            0, order, torch.arange(order.numel(), dtype=order.dtype, device=order.device)
        ),
        "inverse_scatter_int32": lambda: (
            torch.empty_like(order, dtype=torch.int32)
            .scatter_(0, order, torch.arange(order.numel(), dtype=torch.int32, device=order.device))
            .to(order.dtype)
        ),
        "inverse_scatter_float32": lambda: (
            torch.empty_like(order, dtype=torch.float32)
            .scatter_(0, order, torch.arange(order.numel(), dtype=torch.float32, device=order.device))
            .to(order.dtype)
        ),
        "inverse_index_copy": lambda: torch.empty_like(order).index_copy_(
            0, order, torch.arange(order.numel(), dtype=order.dtype, device=order.device)
        ),
        "inverse_index_copy_int32": lambda: (
            torch.empty_like(order, dtype=torch.int32)
            .index_copy_(0, order, torch.arange(order.numel(), dtype=torch.int32, device=order.device))
            .to(order.dtype)
        ),
        "inverse_index_copy_all_int32": lambda: (
            torch.empty_like(order, dtype=torch.int32)
            .index_copy_(0, order.to(torch.int32), torch.arange(order.numel(), dtype=torch.int32, device=order.device))
            .to(order.dtype)
        ),
        "inverse_index_copy_float32_via_int32": lambda: (
            torch.empty_like(order, dtype=torch.float32)
            .index_copy_(0, order, torch.arange(order.numel(), dtype=torch.float32, device=order.device))
            .to(torch.int32)
            .to(order.dtype)
        ),
        "router_identity_multiply": old_router,
        "router_without_identity": lambda: route_topk(logits, 10)[0],
    }
    if npu is not None:
        # The 310P indexed-update primitive supports FP32 values. Positions are
        # exact in FP32 for these bounded benchmark shapes (all below 2**24).
        cases["inverse_scatter_nd_float32"] = lambda: npu.npu_scatter_nd_update_(
            torch.empty_like(order, dtype=torch.float32),
            order.to(torch.int32).unsqueeze(1),
            torch.arange(order.numel(), dtype=torch.float32, device=order.device),
        ).to(order.dtype)
        cases["inverse_scatter_nd_float32_via_int32"] = lambda: (
            npu.npu_scatter_nd_update_(
                torch.empty_like(order, dtype=torch.float32),
                order.to(torch.int32).unsqueeze(1),
                torch.arange(order.numel(), dtype=torch.float32, device=order.device),
            )
            .to(torch.int32)
            .to(order.dtype)
        )
    return cases


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--cases", nargs="+", help="limit profiling to named cases")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh evidence path")
    if args.trace_dir and args.trace_dir.exists():
        parser.error("choose a fresh trace directory")
    npu = importlib.import_module("torch_npu")
    if not npu.npu.is_available() or "310" not in npu.npu.get_device_name(0):
        raise RuntimeError("requires Ascend 310P")
    torch.npu.set_device(0)
    npu.npu.set_compile_mode(jit_compile=False)
    generator = torch.Generator().manual_seed(73)
    with args.output.open("x") as output:
        for tokens in (1, 3, 9, 16, 32, 64, 128, 512):
            routes = tokens * 10
            order = torch.randperm(routes, generator=generator).npu()
            logits = torch.randn(tokens, 512, generator=generator).half().npu()

            cases = make_cases(order, logits, npu)
            if args.cases:
                unknown = set(args.cases) - cases.keys()
                if unknown:
                    parser.error(f"unknown cases: {sorted(unknown)}")
                cases = {name: cases[name] for name in args.cases}
            expected_inverse = torch.argsort(order.cpu()).npu()
            for name, function in cases.items():
                if name.startswith("inverse_"):
                    torch.testing.assert_close(expected_inverse, function(), rtol=0, atol=0)
            if "router_identity_multiply" in cases and "router_without_identity" in cases:
                torch.testing.assert_close(
                    cases["router_identity_multiply"](), cases["router_without_identity"](), rtol=0, atol=0
                )
            for name, function in cases.items():
                stream = torch.npu.Stream()
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    for _ in range(3):
                        function()
                torch.npu.current_stream().wait_stream(stream)
                graph = torch.npu.NPUGraph()
                unroll = 10
                with torch.npu.graph(graph, stream=stream):
                    for _ in range(unroll):
                        captured = function()
                graph.replay()
                torch.testing.assert_close(captured.cpu(), function().cpu(), rtol=0, atol=0)
                record = {
                    "tokens": tokens,
                    "routes": routes,
                    "case": name,
                    "graph": timed(graph.replay, 30, 5, unroll),
                    "scope": "routing_microbenchmark_not_model_throughput",
                }
                output.write(json.dumps(record) + "\n")
                output.flush()
                print(json.dumps(record), flush=True)
                del graph, captured
            if args.trace_dir and tokens in (3, 512):
                with npu.profiler.profile(
                    activities=[npu.profiler.ProfilerActivity.CPU, npu.profiler.ProfilerActivity.NPU],
                    record_shapes=True,
                    on_trace_ready=npu.profiler.tensorboard_trace_handler(str(args.trace_dir / f"t{tokens}")),
                ):
                    for name, function in cases.items():
                        with torch.profiler.record_function(name):
                            function()
                            torch.npu.synchronize()


if __name__ == "__main__":
    main()
