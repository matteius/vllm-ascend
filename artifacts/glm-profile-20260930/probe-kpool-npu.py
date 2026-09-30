"""Small device gate for GLM kpool primitives on an occupied 310P."""

import time

import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.glm5next.kpool_ops import (
    compress_kpool,
    expand_kpool_groups,
    score_kpool,
    select_kpool_groups,
)


def main():
    torch.manual_seed(41)
    keys = torch.randn(1024, 4, 128, dtype=torch.bfloat16)
    gates = torch.randn(1024, 4, 128)
    ape = torch.randn(4, 128)
    query = torch.randn(1, 64, 128, dtype=torch.bfloat16)
    weights = torch.randn(1, 64, dtype=torch.bfloat16)
    positions = torch.tensor([4096], dtype=torch.int32)

    reference_keys = compress_kpool(keys, gates, ape)
    reference_logits = score_kpool(query, weights, reference_keys)
    reference_groups = select_kpool_groups(reference_logits, positions, 2048, 4)
    reference_tokens = expand_kpool_groups(reference_groups[0], reference_groups[2], reference_groups[3], 4)

    device = torch.device("npu:0")
    keys_npu = keys.to(device)
    gates_npu = gates.to(device)
    ape_npu = ape.to(device)
    query_npu = query.to(device)
    weights_npu = weights.to(device)
    positions_npu = positions.to(device)
    elapsed = []
    for _ in range(2):
        start = time.perf_counter()
        pooled = compress_kpool(keys_npu, gates_npu, ape_npu)
        logits = score_kpool(query_npu, weights_npu, pooled)
        groups = select_kpool_groups(logits, positions_npu, 2048, 4)
        tokens = expand_kpool_groups(groups[0], groups[2], groups[3], 4)
        torch.npu.synchronize()
        elapsed.append(round(time.perf_counter() - start, 4))
    torch.testing.assert_close(pooled.cpu(), reference_keys, atol=0.04, rtol=0.04)
    torch.testing.assert_close(logits.cpu(), reference_logits, atol=0.2, rtol=0.03)
    assert torch.equal(tokens.cpu(), reference_tokens)
    print({"device": str(device), "pools": 1024, "run_seconds": elapsed})


if __name__ == "__main__":
    main()
