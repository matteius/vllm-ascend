#!/usr/bin/env python3
"""Probe the 310P paged-latent kernel as a continued-MLA-prefill fallback."""

from __future__ import annotations

import time

import torch
import torch_npu

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, enable_custom_op

BLOCK_SIZE = 32
HEAD_DIM = 512
NUM_HEADS = 64
SCALE = 256**-0.5


def run_case(prefix_tokens: int, query_tokens: int, check_reference: bool) -> None:
    token_capacity = prefix_tokens + query_tokens
    blocks = (token_capacity + BLOCK_SIZE - 1) // BLOCK_SIZE
    cache_cpu = torch.randn(blocks, HEAD_DIM // 16, BLOCK_SIZE, 16, dtype=torch.float16) * 0.1
    query_cpu = torch.randn(query_tokens, NUM_HEADS, HEAD_DIM, dtype=torch.float16) * 0.1
    cache = torch_npu.npu_format_cast(cache_cpu.npu(), ACL_FORMAT_FRACTAL_NZ)
    query = query_cpu.npu()
    lengths = torch.arange(prefix_tokens + 1, token_capacity + 1, dtype=torch.int32, device=query.device)
    group_indices = torch.zeros((query_tokens, 1), dtype=torch.int32, device=query.device)
    tail_starts = torch.zeros(query_tokens, dtype=torch.int32, device=query.device)
    tail_counts = torch.full((query_tokens,), -1, dtype=torch.int32, device=query.device)
    block_table = torch.arange(blocks, dtype=torch.int32, device=query.device).unsqueeze(0)
    query_start_loc = torch.tensor([0, query_tokens], dtype=torch.int32, device=query.device)
    op = torch.ops._C_ascend.npu_qsa_sparse_attention_310

    def run() -> torch.Tensor:
        return op(
            query,
            cache,
            cache,
            group_indices,
            lengths,
            tail_starts,
            tail_counts,
            block_table,
            query_start_loc,
            SCALE,
            4,
        )

    run()
    torch.npu.synchronize()
    started = time.perf_counter()
    output = run()
    torch.npu.synchronize()
    elapsed = time.perf_counter() - started
    print(f"prefix={prefix_tokens} query={query_tokens} elapsed={elapsed:.4f}s", flush=True)
    if check_reference:
        rows = cache_cpu.permute(0, 2, 1, 3).reshape(blocks * BLOCK_SIZE, HEAD_DIM).float()
        expected = []
        for token in range(query_tokens):
            visible = prefix_tokens + token + 1
            scores = torch.matmul(query_cpu[token].float(), rows[:visible].T) * SCALE
            expected.append(torch.matmul(torch.softmax(scores, dim=-1), rows[:visible]))
        reference = torch.stack(expected).half()
        torch.testing.assert_close(output.cpu(), reference, rtol=1e-2, atol=2e-3)
        print("reference parity: pass", flush=True)


if __name__ == "__main__":
    torch.manual_seed(310)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    run_case(prefix_tokens=16, query_tokens=8, check_reference=True)
    run_case(prefix_tokens=512, query_tokens=16, check_reference=False)
    run_case(prefix_tokens=512, query_tokens=128, check_reference=False)
    run_case(prefix_tokens=8192, query_tokens=16, check_reference=False)
