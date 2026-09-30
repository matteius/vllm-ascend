"""Compare equivalent logical and physical NZ pages for GLM QSA decode."""

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
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    device = f"npu:{args.device}"
    query = torch.randn((1, 32, 512), dtype=torch.float16, device=device)
    backing = torch.zeros((args.blocks, 64, 384, 16), dtype=torch.float16, device=device)
    backing[0, :32, 0] = torch.randn((32, 16), dtype=torch.float16, device=device)
    logical_view = backing[:, :32]
    group_indices = torch.zeros((1, 1), dtype=torch.int32, device=device)
    sequence_lengths = torch.ones((1,), dtype=torch.int32, device=device)
    tail_starts = torch.zeros_like(sequence_lengths)
    tail_counts = torch.full_like(sequence_lengths, -1)
    block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    query_start_loc = torch.tensor((0, 1), dtype=torch.int32, device=device)
    op = torch.ops._C_ascend.npu_qsa_sparse_attention_310

    def call(cache: torch.Tensor, logical_kv_heads: int) -> torch.Tensor:
        return op(
            query,
            cache,
            cache,
            group_indices,
            sequence_lengths,
            tail_starts,
            tail_counts,
            block_table,
            query_start_loc,
            512**-0.5,
            4,
            logical_kv_heads,
        )

    reference = call(logical_view, 0)
    physical = call(backing, 1)
    torch.testing.assert_close(physical.cpu(), reference.cpu(), rtol=0, atol=0)

    def measure(cache: torch.Tensor, logical_kv_heads: int) -> list[float]:
        for _ in range(3):
            call(cache, logical_kv_heads)
        torch.npu.synchronize()
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            call(cache, logical_kv_heads)
            torch.npu.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        return times

    strided = measure(logical_view, 0)
    contiguous = measure(backing, 1)
    print(
        json.dumps(
            {
                "blocks": args.blocks,
                "logical_shape": list(logical_view.shape),
                "logical_stride": list(logical_view.stride()),
                "physical_shape": list(backing.shape),
                "parity": "exact FP16",
                "strided_median_ms": statistics.median(strided),
                "physical_median_ms": statistics.median(contiguous),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
