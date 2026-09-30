"""Compare the grouped 310P projection at GLM decode shapes without model load."""

import os
import statistics
import time

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

EXPERTS = 72
ACTIVE_EXPERTS = {
    8: (0, 17),
    32: (0, 8, 17, 26, 35, 44, 53, 71),
}


def group_ends(rows: int) -> torch.Tensor:
    active = set(ACTIVE_EXPERTS[rows])
    end = 0
    ends = []
    for expert in range(EXPERTS):
        if expert in active:
            end += 4
        ends.append(end)
    assert end == rows
    return torch.tensor(ends, dtype=torch.int64, device="npu")


def run_case(bits: int, rows: int, n: int, k: int) -> None:
    codes_per_byte = 8 // bits
    codes = torch.full((EXPERTS, n, k // codes_per_byte), 0x55, dtype=torch.uint8, device="npu")
    scales = torch.full((EXPERTS, n // 32, k // 32), 0.01, dtype=torch.float32, device="npu")
    inputs = torch.randn(rows, k, dtype=torch.float16, device="npu")
    ends = group_ends(rows)
    op = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310
    for _ in range(4):
        output = op(inputs, codes, scales, ends)
    torch.npu.synchronize()
    timings = []
    for _ in range(30):
        start = time.perf_counter()
        output = op(inputs, codes, scales, ends)
        torch.npu.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    assert torch.isfinite(output).all()
    print(
        f"bits={bits} rows={rows} n={n} k={k} "
        f"median_ms={statistics.median(timings):.4f} "
        f"mean_ms={statistics.mean(timings):.4f} "
        f"min_ms={min(timings):.4f} max_ms={max(timings):.4f}",
        flush=True,
    )


def main() -> None:
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    print("opp", os.environ["ASCEND_CUSTOM_OPP_PATH"].split(":")[0], flush=True)
    for rows in (8, 32):
        run_case(4, rows, 2048, 4096)
        run_case(2, rows, 4096, 2048)


if __name__ == "__main__":
    main()
