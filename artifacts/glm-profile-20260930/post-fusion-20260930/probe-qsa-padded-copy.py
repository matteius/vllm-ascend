"""Measure the 310P QSA penalty for a page-strided cache view.

The full-page control changes QSA's logical KV-head count, so compare timing
only. It is deliberately not a correctness candidate.
"""

import argparse
import json
import statistics
import time

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=342)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    device = f"npu:{args.device}"
    query = torch.zeros((1, 32, 512), dtype=torch.float16, device=device)
    backing = torch.zeros((args.blocks, 64, 384, 16), dtype=torch.float16, device=device)
    logical_view = backing[:, :32]
    groups = torch.zeros((1, 1), dtype=torch.int32, device=device)
    sequence_lengths = torch.ones((1,), dtype=torch.int32, device=device)
    tail_starts = torch.zeros_like(sequence_lengths)
    tail_counts = torch.full_like(sequence_lengths, -1)
    block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    query_start_loc = torch.tensor((0, 1), dtype=torch.int32, device=device)
    op = torch.ops._C_ascend.npu_qsa_sparse_attention_310

    def measure(cache: torch.Tensor) -> list[float]:
        def call() -> torch.Tensor:
            return op(
                query,
                cache,
                cache,
                groups,
                sequence_lengths,
                tail_starts,
                tail_counts,
                block_table,
                query_start_loc,
                512**-0.5,
                4,
            )

        for _ in range(3):
            call()
        torch.npu.synchronize()
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            call()
            torch.npu.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        return times

    strided = measure(logical_view)
    physical = measure(backing)
    print(
        json.dumps(
            {
                "logical_view_shape": list(logical_view.shape),
                "logical_view_stride": list(logical_view.stride()),
                "physical_shape": list(backing.shape),
                "strided_median_ms": statistics.median(strided),
                "physical_median_ms": statistics.median(physical),
                "note": "The physical control changes the logical KV-head count; timings only.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
