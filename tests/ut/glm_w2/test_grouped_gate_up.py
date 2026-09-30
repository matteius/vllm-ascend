# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host checks for the resident GLM gate/up grouped projection."""

import torch

from vllm_ascend._310p.quantization.methods.w2_dynamic import (
    AscendW2DynamicFusedMoEMethod310,
    _can_use_w2_grouped_cube,
)
from vllm_ascend.models.glm5next_w2.model import _PackedW2ExpertBank


def test_fused_gate_up_matches_separate_projection_and_uses_two_calls():
    torch.manual_seed(17)
    hidden = inter = 256
    bank = _PackedW2ExpertBank(
        hidden=hidden,
        inter=inter,
        num_experts=2,
        local_expert_offset=0,
        num_local_experts=2,
        offload_to_cpu=False,
    )
    for expert_id in range(2):
        for projection in ("gate", "up", "down"):
            bank.place_resident_tensor(
                expert_id,
                f"{projection}_packed",
                torch.randint(0, 2, (inter, hidden // 2), dtype=torch.uint8),
                device="cpu",
            )
            bank.place_resident_tensor(
                expert_id,
                f"{projection}_scale",
                torch.ones(inter // 32, hidden // 32),
                device="cpu",
            )
    bank.finalize_grouped_storage()

    calls = []

    def grouped_projection(inputs, codes, scales, group_ends):
        del scales
        calls.append(codes.shape[1])
        output = torch.empty(inputs.shape[0], codes.shape[1], dtype=torch.float16)
        start = 0
        for expert_id, end in enumerate(group_ends.tolist()):
            weight = codes[expert_id].float().repeat_interleave(2, dim=-1)
            output[start:end] = (inputs[start:end].float() @ weight.T / hidden).half()
            start = end
        return output

    x = torch.randn(2, hidden, dtype=torch.float16) * 0.1
    ids = torch.tensor([[0, 1], [1, 0]])
    weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
    method = AscendW2DynamicFusedMoEMethod310()
    assert _can_use_w2_grouped_cube(grouped_projection, bank, ids.numel())
    fused = method._apply_device_grouped(grouped_projection, bank, x, weights, ids, None)
    assert calls == [2 * inter, hidden]

    fused_codes = bank.gate_up_packed_bank
    fused_scales = bank.gate_up_scale_bank
    del bank.gate_up_packed_bank
    del bank.gate_up_scale_bank
    assert not _can_use_w2_grouped_cube(grouped_projection, bank, ids.numel())
    calls.clear()
    separate = method._apply_device_grouped(grouped_projection, bank, x, weights, ids, None)
    bank.gate_up_packed_bank = fused_codes
    bank.gate_up_scale_bank = fused_scales

    assert calls == [inter, inter, hidden]
    torch.testing.assert_close(fused, separate, rtol=0, atol=0)
