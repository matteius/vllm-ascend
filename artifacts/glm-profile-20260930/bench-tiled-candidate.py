"""Compare contiguous-tile packed expert codes with the canonical layout.

Run in separate processes with baseline and candidate OPP paths. The custom
operator name is the same; --tiled tells this client how to lay out its input.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

EXPERTS = 72
ACTIVE_EXPERTS = {8: (0, 17), 32: (0, 8, 17, 26, 35, 44, 53, 71)}
OUTPUT_TILE = 16
INPUT_TILE = 128


def tiled_codes(codes: torch.Tensor, k: int, bits: int) -> torch.Tensor:
    """Reorder bytes to [expert, N/16, K/128, 16, 128/codes_per_byte]."""
    experts, n, packed_k = codes.shape
    codes_per_byte = 8 // bits
    if n % OUTPUT_TILE or k % INPUT_TILE or packed_k != k // codes_per_byte:
        raise ValueError("packed code geometry does not fit the tiled kernel")
    return (
        codes.view(experts, n // OUTPUT_TILE, OUTPUT_TILE, k // INPUT_TILE, INPUT_TILE // codes_per_byte)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(experts, n, packed_k)
    )


def group_ends(rows: int) -> torch.Tensor:
    active = set(ACTIVE_EXPERTS[rows])
    end = 0
    ends = []
    for expert in range(EXPERTS):
        if expert in active:
            end += 4
        ends.append(end)
    return torch.tensor(ends, dtype=torch.int64, device="npu")


def run_case(bits: int, rows: int, n: int, k: int, *, tiled: bool) -> tuple[dict, torch.Tensor]:
    torch.manual_seed(20260930 + bits + rows)
    codes = torch.randint(0, 256, (EXPERTS, n, k // (8 // bits)), dtype=torch.uint8)
    if tiled:
        codes = tiled_codes(codes, k, bits)
    codes = codes.npu()
    scales = (torch.rand(EXPERTS, n // 32, k // 32) * 0.02 + 0.005).npu()
    inputs = torch.randn(rows, k).half().npu()
    ends = group_ends(rows)
    op = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310
    for _ in range(4):
        output = op(inputs, codes, scales, ends)
    torch.npu.synchronize()
    timings = []
    for _ in range(20):
        start = time.perf_counter()
        output = op(inputs, codes, scales, ends)
        torch.npu.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    if not torch.isfinite(output).all().cpu().item():
        raise RuntimeError("candidate output contains non-finite values")
    return (
        {
            "bits": bits,
            "rows": rows,
            "n": n,
            "k": k,
            "median_ms": statistics.median(timings),
            "minimum_ms": min(timings),
        },
        output.cpu(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    summary = []
    outputs = {}
    for rows in (8, 32):
        for bits, n, k in ((4, 2048, 4096), (2, 4096, 2048)):
            record, output = run_case(bits, rows, n, k, tiled=args.tiled)
            summary.append(record)
            outputs[f"{bits}_{rows}_{n}_{k}"] = output
            print(json.dumps(record), flush=True)
    torch.save({"summary": summary, "outputs": outputs}, args.out)


if __name__ == "__main__":
    main()
