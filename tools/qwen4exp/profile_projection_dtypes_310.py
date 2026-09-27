# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched FP16/INT8/W4 projection diagnostics, NOT a hardware peak-TOPS test.

All prepared projections compute the same integer-valued matrices and scaling.
Stock INT8 runs a whole-K GEMM; experimental INT4 preserves the model's per-128
group epilogue and uses two products. Comparing them measures implementations,
not an intrinsic ranking of arithmetic types. No checkpoint or server changes.
Stop concurrent inference/host profiling before collecting clean timings.
"""

import argparse
import importlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm_ascend.models.qwen4_exp.w4_moe import pack_cube_tiles
from vllm_ascend.models.qwen4_exp.w4a8_int4 import (
    GROUP_SIZE,
    pack_native_metadata,
    pack_native_weight,
    pack_nibbles,
    quantize_activation_limbs,
)

ACTIVATION_SCALE = 1 / 128
WEIGHT_SCALE = 1 / 32
NZ_FORMAT = 29
PROJECTIONS = {"gate_up": (2560, 1280), "down": (640, 2560)}


def matched_inputs(rows, width, outputs, seed=1024):
    if rows <= 0 or width % GROUP_SIZE or outputs % GROUP_SIZE:
        raise ValueError("positive rows and group-aligned projection dimensions required")
    generator = torch.Generator().manual_seed(seed)
    qa = torch.randint(-127, 128, (rows, width), dtype=torch.int16, generator=generator)
    # Per-row and per-group dynamic quantization both have an exact 1/128 scale.
    qa[:, ::GROUP_SIZE] = 127
    qw = torch.randint(-8, 8, (outputs, width), dtype=torch.int8, generator=generator)
    x = (qa.float() * ACTIVATION_SCALE).half()
    weight = (qw.float() * WEIGHT_SCALE).half()
    scales = torch.full((outputs, width // GROUP_SIZE), WEIGHT_SCALE, dtype=torch.float16)
    offsets = torch.zeros_like(scales, dtype=torch.int8)
    return x, weight, qa.to(torch.int8), qw, pack_nibbles(qw), scales, offsets


def timed(function, iterations, repeats, operations_per_call=1):
    for _ in range(3):
        function()
    torch.npu.synchronize()
    wall, events = [], []
    for _ in range(repeats):
        start_event, end_event = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start = time.perf_counter()
        start_event.record()
        for _ in range(iterations):
            function()
        end_event.record()
        end_event.synchronize()
        wall.append((time.perf_counter() - start) * 1000 / (iterations * operations_per_call))
        events.append(start_event.elapsed_time(end_event) / (iterations * operations_per_call))
    return {
        "wall_median_ms": statistics.median(wall),
        "event_median_ms": statistics.median(events),
        "wall_trials_ms": wall,
        "event_trials_ms": events,
    }


def make_cases(data, npu):
    x_cpu, weight_cpu, qa_cpu, qw_cpu, packed, scales, offsets = data
    rows = x_cpu.shape[0]
    x, qa = x_cpu.npu(), qa_cpu.npu()
    weight_nz = npu.npu_format_cast(weight_cpu.npu(), NZ_FORMAT)
    int8_nz = npu.npu_format_cast(qw_cpu.npu(), NZ_FORMAT)
    if any(npu.get_npu_format(weight) != NZ_FORMAT for weight in (weight_nz, int8_nz)):
        raise RuntimeError("benchmark requires actual NZ weights; refusing hidden layout conversion")
    # 310P expects transposed preformatted [N,K] weights. Preconvert scale for graphs.
    quant_scale = npu.npu_trans_quant_param(
        torch.full((qw_cpu.shape[0],), ACTIVATION_SCALE * WEIGHT_SCALE, device=x.device), None
    )
    tiled = [
        pack_cube_tiles(value, kind).unsqueeze(0).npu()
        for value, kind in zip((packed, scales, offsets), ("weight", "weight_scale", "weight_offset"))
    ]
    native_weight, sums = pack_native_weight(packed)
    native = [
        native_weight.unsqueeze(0).npu(),
        pack_native_metadata(scales).unsqueeze(0).npu(),
        pack_native_metadata(offsets).half().unsqueeze(0).npu(),
        sums.unsqueeze(0).npu(),
    ]
    ends = torch.tensor([rows], dtype=torch.int64, device=x.device)
    limbs = quantize_activation_limbs(x)

    def int8_with_quant():
        quantized, _ = npu.npu_dynamic_quant(x)
        # Only valid as matched-data diagnostic: every row's scale is 1/128.
        return npu.npu_quant_matmul(quantized, int8_nz.t(), quant_scale, output_dtype=torch.float16)

    cases = {
        "fp16_preformatted": lambda: F.linear(x, weight_nz),
        "int8_preformatted": lambda: npu.npu_quant_matmul(qa, int8_nz.t(), quant_scale, output_dtype=torch.float16),
        "w4a16_packed_grouped": lambda: torch.ops._C_ascend.npu_qwen_w4_grouped_matmul_310(x, *tiled, ends),
        "w4a8_native_prepared": lambda: torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310(*limbs, *native, ends),
        "int8_activation_quant_only": lambda: npu.npu_dynamic_quant(x),
        "int4_activation_limbs_only": lambda: quantize_activation_limbs(x),
        "int8_with_activation_quant": int8_with_quant,
        "w4a8_native_with_activation_quant": lambda: torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310(
            *quantize_activation_limbs(x), *native, ends
        ),
    }
    if rows <= 80:
        expert_ids = torch.zeros(rows, dtype=torch.int32, device=x.device)
        cases["w4a16_packed_routed"] = lambda: torch.ops._C_ascend.npu_qwen_w4_routed_matmul_310(x, *tiled, expert_ids)
    return cases


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 3, 6, 16, 32, 128, 512])
    parser.add_argument("--projections", choices=list(PROJECTIONS), nargs="+", default=list(PROJECTIONS))
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--graph-unroll", type=int, default=10, help="calls per graph, amortizes Python replay dispatch"
    )
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.trace_dir and args.trace_dir.exists()):
        parser.error("choose fresh evidence paths")
    if min(args.rows) <= 0 or max(args.rows) > 512 or min(args.iterations, args.repeats, args.graph_unroll) <= 0:
        parser.error("rows must be 1..512 and timings positive")
    npu = importlib.import_module("torch_npu")
    if not npu.npu.is_available() or "310" not in npu.npu.get_device_name(0):
        raise RuntimeError("requires Ascend 310P")
    torch.npu.set_device(0)
    npu.npu.set_compile_mode(jit_compile=False)
    importlib.import_module("vllm_ascend.utils").enable_custom_op()
    with args.output.open("x") as file:
        for projection in args.projections:
            width, outputs = PROJECTIONS[projection]
            for rows in args.rows:
                data = matched_inputs(rows, width, outputs)
                expected = F.linear(data[0].float(), data[1].float()).half()
                cases = make_cases(data, npu)
                for name, function in cases.items():
                    actual = function()
                    if "only" not in name:
                        torch.testing.assert_close(actual.cpu(), expected, rtol=0.002, atol=0.002)
                    eager = timed(function, args.iterations, args.repeats)
                    stream = torch.npu.Stream()
                    stream.wait_stream(torch.npu.current_stream())
                    with torch.npu.stream(stream):
                        for _ in range(3):
                            function()
                    torch.npu.current_stream().wait_stream(stream)
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        for _ in range(args.graph_unroll):
                            captured = function()
                    graph.replay()
                    torch.npu.synchronize()
                    if "only" not in name:
                        torch.testing.assert_close(captured.cpu(), expected, rtol=0.002, atol=0.002)
                    replay = timed(graph.replay, args.iterations, args.repeats, args.graph_unroll)
                    record = {
                        "projection": projection,
                        "rows": rows,
                        "k": width,
                        "n": outputs,
                        "case": name,
                        "matched_arithmetic": True,
                        "eager": eager,
                        "graph": replay,
                        "calls_per_graph": args.graph_unroll,
                        "storage_bytes": {
                            "fp16": width * outputs * 2,
                            "int8": width * outputs,
                            "w4_with_metadata": width * outputs // 2 + outputs * (width // 128) * 3,
                            "native_w4_with_metadata": width * outputs // 2 + outputs * (width // 128) * 6,
                        },
                        "scope": "one_expert_synthetic_no_routing_no_collective_not_model_tps",
                    }
                    line = json.dumps(record)
                    print(line, flush=True)
                    file.write(line + "\n")
                    file.flush()
                    del graph, captured, actual
                if args.trace_dir:
                    trace_path = args.trace_dir / f"{projection}-m{rows}"
                    with npu.profiler.profile(
                        activities=[npu.profiler.ProfilerActivity.CPU, npu.profiler.ProfilerActivity.NPU],
                        record_shapes=True,
                        on_trace_ready=npu.profiler.tensorboard_trace_handler(str(trace_path)),
                    ):
                        for name, function in cases.items():
                            with torch.profiler.record_function(f"dtype::{name}"):
                                for _ in range(3):
                                    function()
                                torch.npu.synchronize()
                del cases
                torch.npu.empty_cache()


if __name__ == "__main__":
    main()
