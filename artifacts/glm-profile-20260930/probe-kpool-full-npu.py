"""Exercise GLM kpool cache writes and selection on one 310P device."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.glm5next.kpool_ops import compress_kpool, select_kpool_groups
from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import SparseAttnIndexerKpool


def main() -> None:
    device = torch.device("npu:0")
    keys = torch.ones(8, 128, dtype=torch.bfloat16)
    keys[4:] = 3
    keys = keys.to(device)
    gates = torch.zeros(8, 128, dtype=torch.float32, device=device)
    ape = torch.zeros(4, 128, dtype=torch.float32, device=device)
    page_elements = 4 * 128
    page_stride = page_elements + 64
    raw_key_cache = torch.zeros(2 * page_stride + 16, dtype=torch.float16, device=device)
    key_cache = torch.as_strided(
        raw_key_cache,
        size=(2, 4, 1, 128),
        stride=(page_stride, 128, 128, 1),
        storage_offset=8,
    )
    state_cache = torch.zeros(2, 4, 256, dtype=torch.float32, device=device)
    indexer = SparseAttnIndexerKpool.__new__(SparseAttnIndexerKpool)
    torch.nn.Module.__init__(indexer)
    indexer.k_cache = SimpleNamespace(kv_cache=key_cache, prefix="index")
    indexer.tail_cache = SimpleNamespace(kv_cache=state_cache, prefix="state")
    indexer.topk_tokens = 4
    indexer.head_dim = 128
    indexer.topk_indices_buffer = torch.full((8, 8), -1, dtype=torch.int32, device=device)
    indexer.skip_k_cache_insert = False
    positions = torch.arange(4, dtype=torch.int32, device=device).repeat(2)
    metadata = {
        "index": SimpleNamespace(
            num_actual_tokens=8,
            slot_mapping=torch.tensor([-1, -1, -1, 0, -1, -1, -1, 4], device=device),
            seq_lens_cpu=torch.tensor([1, 1]),
            raw_seq_lens=torch.tensor([4, 4], device=device),
            block_table=torch.tensor([[0], [1]], dtype=torch.int32, device=device),
            cum_query_lens=torch.tensor([4, 8], dtype=torch.int32, device=device),
            cum_query_lens_cpu=torch.tensor([0, 4, 8]),
        ),
        "state": SimpleNamespace(slot_mapping=torch.arange(8, device=device)),
    }
    with patch(
        "vllm_ascend.models.glm5next.sparse_attn_indexer_kpool.get_forward_context",
        return_value=SimpleNamespace(attn_metadata=metadata),
    ):
        selected = indexer.forward_oot(
            keys,
            torch.ones(8, 1, 128, dtype=torch.bfloat16, device=device),
            keys,
            torch.ones(8, 1, dtype=torch.float32, device=device),
            gate_score=gates,
            compress_ape=ape,
            index_kpool=4,
            positions=positions,
        )
    torch.npu.synchronize()
    for request in range(2):
        row = request * 4
        expected = compress_kpool(keys[row : row + 4][None], gates[row : row + 4][None], ape)
        torch.testing.assert_close(key_cache[request, 0, 0].cpu(), expected[0].half().cpu())
        torch.testing.assert_close(selected[row + 3, :4].cpu(), torch.arange(4, dtype=torch.int32))
    # Decode inside an incomplete pool: there is no compressed-key write.
    metadata["index"].num_actual_tokens = 2
    metadata["index"].slot_mapping = torch.tensor([-1, -1], device=device)
    metadata["index"].cum_query_lens = torch.tensor([1, 2], dtype=torch.int32, device=device)
    metadata["index"].cum_query_lens_cpu = torch.tensor([0, 1, 2])
    metadata["index"].raw_seq_lens = torch.tensor([5, 5], device=device)
    metadata["state"].slot_mapping = torch.tensor([0, 4], device=device)
    next_positions = torch.tensor([4, 4], dtype=torch.int32, device=device)
    with patch(
        "vllm_ascend.models.glm5next.sparse_attn_indexer_kpool.get_forward_context",
        return_value=SimpleNamespace(attn_metadata=metadata),
    ):
        selected = indexer.forward_oot(
            keys[:2],
            torch.ones(2, 1, 128, dtype=torch.bfloat16, device=device),
            keys[:2],
            torch.ones(2, 1, dtype=torch.float32, device=device),
            gate_score=gates[:2],
            compress_ape=ape,
            index_kpool=4,
            positions=next_positions,
        )
    torch.npu.synchronize()
    assert selected[0, 4].cpu().item() == selected[1, 4].cpu().item() == 4
    # Cross the 310P index-copy size that selected a BF16-only missing kernel
    # in the live server: eight FP16 pools, 1024 scalar elements.
    slots = torch.arange(8, device=device)
    rows = key_cache.storage_offset() + (slots // 4) * key_cache.stride(0) + (slots % 4) * key_cache.stride(1)
    offsets = rows[:, None] + torch.arange(128, device=device)[None, :]
    flat = key_cache.as_strided(
        (key_cache.untyped_storage().nbytes() // key_cache.element_size(),),
        (1,),
        storage_offset=0,
    )
    flat.index_copy_(0, offsets.flatten(), torch.ones(1024, dtype=torch.float16, device=device))
    torch.npu.synchronize()
    assert torch.all(key_cache[:, :, 0, :] == 1).cpu().item()
    # The first live chat prefill has seven pools and a 26-token causal query.
    # This caught an AICPU GatherElements failure with the old validity gather.
    prefill_positions = torch.arange(26, dtype=torch.int32, device=device)
    prefill_groups, _, _, _ = select_kpool_groups(
        torch.zeros(26, 7, dtype=torch.float32, device=device),
        prefill_positions,
        2048,
        4,
    )
    torch.npu.synchronize()
    assert prefill_groups[25, :6].cpu().tolist() == list(range(6))
    print({"device": str(device), "requests": 2, "decode_selected": selected[:2, :5].cpu().tolist()})


if __name__ == "__main__":
    main()
