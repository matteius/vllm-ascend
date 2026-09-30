# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent per-draft-step buffers for GLM KPool attention metadata."""

from dataclasses import dataclass

import torch


@dataclass
class KPoolMetadataBuffers:
    slots: torch.Tensor
    seq_lens: torch.Tensor
    query_ends: torch.Tensor
    raw_seq_lens: torch.Tensor
    positions: torch.Tensor
    block_table: torch.Tensor


class KPoolGraphBufferSets:
    """Key derived metadata by the capture-stable source slot address."""

    def __init__(self, device: torch.device, max_tokens: int, max_reqs: int, block_table_width: int) -> None:
        self.device = device
        self.max_tokens = max_tokens
        self.max_reqs = max_reqs
        self.block_table_width = block_table_width
        self._buffers: dict[int, KPoolMetadataBuffers] = {}
        self._frozen: set[int] = set()

    def get(self, source_slots: torch.Tensor) -> KPoolMetadataBuffers:
        key = source_slots.data_ptr()
        buffers = self._buffers.get(key)
        if buffers is None:
            buffers = KPoolMetadataBuffers(
                slots=torch.empty(self.max_tokens, dtype=torch.int64, device=self.device),
                seq_lens=torch.empty(self.max_reqs, dtype=torch.int32, device=self.device),
                query_ends=torch.empty(self.max_reqs, dtype=torch.int32, device=self.device),
                raw_seq_lens=torch.empty(self.max_reqs, dtype=torch.int32, device=self.device),
                positions=torch.empty(self.max_tokens, dtype=torch.int64, device=self.device),
                block_table=torch.empty(self.max_reqs, self.block_table_width, dtype=torch.int32, device=self.device),
            )
            self._buffers[key] = buffers
        return buffers

    def ensure_table_capacity(self, source_slots: torch.Tensor, num_reqs: int, width: int) -> KPoolMetadataBuffers:
        buffers = self.get(source_slots)
        if num_reqs > buffers.block_table.shape[0] or width > buffers.block_table.shape[1]:
            if source_slots.data_ptr() in self._frozen:
                raise RuntimeError("GLM KPool graph metadata exceeded its captured block-table capacity")
            buffers.block_table = torch.empty(
                max(num_reqs, buffers.block_table.shape[0]),
                max(width, buffers.block_table.shape[1]),
                dtype=torch.int32,
                device=self.device,
            )
        return buffers

    def freeze(self, source_slots: torch.Tensor) -> None:
        """Reject later pointer replacement for a captured draft step."""
        self.get(source_slots)
        self._frozen.add(source_slots.data_ptr())
