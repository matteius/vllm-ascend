# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host MLA history and bounded 310P hot-page staging for GLM-5.3-Flash.

Scheduler IDs remain the source of page identity. The compressed kpool index
continues to use those IDs on NPU; only the 512-wide MLA latent is offloaded.
This correctness-first path copies selected 32-token pages once per attention
call and leaves room for a future asynchronous transfer/cache policy.
"""

from __future__ import annotations

import numpy as np
import torch
from vllm.logger import logger

HOT_BLOCK_SIZE = 32
NZ_INNER = 16


def _cpu_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(device="cpu", non_blocking=False).numpy()


def selected_page_columns(
    group_indices: np.ndarray,
    group_counts: np.ndarray,
    tail_starts: np.ndarray,
    tail_counts: np.ndarray,
    pool_size: int,
) -> list[np.ndarray]:
    """Return the exact 32-token page columns read by each QSA query."""
    if pool_size <= 0 or HOT_BLOCK_SIZE % pool_size:
        raise ValueError("QSA pool size must divide the 32-token hot page")
    if group_indices.ndim != 2 or group_indices.shape[0] != len(group_counts):
        raise ValueError("QSA selection dimensions do not match")
    pages: list[np.ndarray] = []
    for row, count in enumerate(group_counts):
        count = int(count)
        if int(tail_counts[row]) < 0:
            # QSA's dense sentinel encodes a raw visible token length.
            columns = np.arange((count + HOT_BLOCK_SIZE - 1) // HOT_BLOCK_SIZE, dtype=np.int64)
        else:
            if count < 0 or count > group_indices.shape[1]:
                raise ValueError("QSA group count exceeds selected groups")
            groups = group_indices[row, :count].astype(np.int64, copy=False)
            if np.any(groups < 0):
                raise ValueError("QSA selected an invalid negative group")
            columns = groups * pool_size // HOT_BLOCK_SIZE
            if int(tail_counts[row]) > 0:
                columns = np.append(columns, int(tail_starts[row]) // HOT_BLOCK_SIZE)
            columns = np.unique(columns)
        pages.append(columns)
    return pages


def remap_selected_pages(
    block_table: np.ndarray,
    columns: list[np.ndarray],
    host_num_pages: int,
    hot_num_pages: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack referenced scheduler pages and remap the QSA block table."""
    if block_table.ndim != 2 or block_table.shape[0] != len(columns):
        raise ValueError("QSA block table and selections have different request counts")
    referenced: list[np.ndarray] = []
    for row, cols in enumerate(columns):
        if np.any(cols < 0) or np.any(cols >= block_table.shape[1]):
            raise ValueError("QSA selected a page outside the request block table")
        referenced.append(block_table[row, cols].astype(np.int64, copy=False))
    old_pages = np.unique(np.concatenate(referenced)) if referenced else np.empty(0, dtype=np.int64)
    if np.any(old_pages < 0) or np.any(old_pages >= host_num_pages):
        raise ValueError("QSA selected a scheduler page outside host MLA history")
    if len(old_pages) > hot_num_pages:
        raise ValueError(
            f"GLM MLA selection needs {len(old_pages)} hot pages but only {hot_num_pages} are reserved; "
            "reduce max-num-batched-tokens or max-num-seqs"
        )
    remapped = np.zeros(block_table.shape, dtype=np.int32)
    for row, cols in enumerate(columns):
        remapped[row, cols] = np.searchsorted(old_pages, block_table[row, cols]).astype(np.int32)
    return old_pages, remapped


class GlmHostKVLayer:
    """One MLA layer's host history with a temporary NPU QSA hot cache."""

    def __init__(self, logical_blocks: int, hot_cache: torch.Tensor, block_size: int, *, pin_memory: bool = True):
        if block_size % HOT_BLOCK_SIZE:
            raise ValueError("GLM host MLA requires a block size divisible by 32")
        if hot_cache.ndim != 4 or hot_cache.shape[1:] != (32, HOT_BLOCK_SIZE, NZ_INNER):
            raise ValueError("GLM hot cache must have NZ [pages, 32, 32, 16] geometry")
        self.hot_cache = hot_cache
        self.logical_blocks = logical_blocks
        self.pages_per_block = block_size // HOT_BLOCK_SIZE
        shape = (logical_blocks * self.pages_per_block, 32, HOT_BLOCK_SIZE, NZ_INNER)
        try:
            self.history = torch.empty(shape, dtype=hot_cache.dtype, device="cpu", pin_memory=pin_memory)
        except RuntimeError:
            if not pin_memory:
                raise
            logger.warning("Pinned GLM MLA host allocation failed; using pageable host history")
            self.history = torch.empty(shape, dtype=hot_cache.dtype, device="cpu")

    def write(self, rows: torch.Tensor, slots: torch.Tensor) -> None:
        """Persist the normalized latent before either decode or prefill reads it."""
        if rows.ndim != 3 or rows.shape[1:] != (32, NZ_INNER):
            raise ValueError("GLM MLA rows have unexpected NZ geometry")
        if slots.ndim != 1 or len(slots) != len(rows):
            raise ValueError("GLM MLA slots and rows have different lengths")
        slots_cpu = _cpu_array(slots).astype(np.int64, copy=False)
        valid = slots_cpu >= 0
        if not np.any(valid):
            return
        if np.any(slots_cpu[valid] >= self.history.shape[0] * HOT_BLOCK_SIZE):
            raise ValueError("GLM MLA write exceeds host history")
        rows_cpu = rows.detach().to(device="cpu", dtype=self.history.dtype, non_blocking=False)
        pages = torch.from_numpy(slots_cpu[valid] // HOT_BLOCK_SIZE)
        offsets = torch.from_numpy(slots_cpu[valid] % HOT_BLOCK_SIZE)
        self.history[pages, :, offsets, :] = rows_cpu[torch.from_numpy(valid)]

    def stage(
        self,
        block_table: torch.Tensor,
        group_indices: torch.Tensor,
        group_counts: torch.Tensor,
        tail_starts: torch.Tensor,
        tail_counts: torch.Tensor,
        pool_size: int,
    ) -> torch.Tensor:
        """Copy only pages read by this call and return its remapped table."""
        cpu_table = _cpu_array(block_table).astype(np.int64, copy=False)
        columns = selected_page_columns(
            _cpu_array(group_indices),
            _cpu_array(group_counts),
            _cpu_array(tail_starts),
            _cpu_array(tail_counts),
            pool_size,
        )
        return self._stage_columns(cpu_table, columns)

    def _stage_columns(self, cpu_table: np.ndarray, columns: list[np.ndarray]) -> torch.Tensor:
        old_pages, remapped = remap_selected_pages(cpu_table, columns, self.history.shape[0], self.hot_cache.shape[0])
        if len(old_pages):
            page_tensor = torch.from_numpy(old_pages)
            selected = self.history.index_select(0, page_tensor)
            self.hot_cache[: len(old_pages)].copy_(selected, non_blocking=False)
        return torch.from_numpy(remapped).to(device=self.hot_cache.device, non_blocking=False)

    def prefill_segments(
        self,
        block_table: torch.Tensor,
        group_indices: torch.Tensor,
        group_counts: torch.Tensor,
        tail_starts: torch.Tensor,
        tail_counts: torch.Tensor,
        boundaries: list[int],
        pool_size: int,
    ) -> list[tuple[int, int, int]]:
        """Greedily batch queries whose union fits the hot cache."""
        table = _cpu_array(block_table)
        columns = selected_page_columns(
            _cpu_array(group_indices),
            _cpu_array(group_counts),
            _cpu_array(tail_starts),
            _cpu_array(tail_counts),
            pool_size,
        )
        if len(boundaries) != table.shape[0] or (boundaries and boundaries[-1] != len(columns)):
            raise ValueError("GLM prefill boundaries do not match QSA requests")
        segments: list[tuple[int, int, int]] = []
        start = 0
        for request, end in enumerate(boundaries):
            if end < start:
                raise ValueError("GLM prefill boundaries are not monotonic")
            segment_start = start
            selected: set[int] = set()
            for token in range(start, end):
                cols = columns[token]
                if np.any(cols >= table.shape[1]):
                    raise ValueError("GLM prefill selection exceeds the block table")
                pages = set(int(page) for page in table[request, cols])
                if len(pages) > self.hot_cache.shape[0]:
                    raise ValueError("One GLM prefill query exceeds the reserved hot cache")
                if len(selected | pages) > self.hot_cache.shape[0]:
                    segments.append((segment_start, token, request))
                    segment_start = token
                    selected = pages
                else:
                    selected.update(pages)
            if segment_start < end:
                segments.append((segment_start, end, request))
            start = end
        return segments

    def stage_prefill(
        self,
        block_table_row: torch.Tensor,
        group_indices: torch.Tensor,
        group_counts: torch.Tensor,
        tail_starts: torch.Tensor,
        tail_counts: torch.Tensor,
        pool_size: int,
    ) -> torch.Tensor:
        """Stage the union of a single request's consecutive prefill queries."""
        columns = selected_page_columns(
            _cpu_array(group_indices),
            _cpu_array(group_counts),
            _cpu_array(tail_starts),
            _cpu_array(tail_counts),
            pool_size,
        )
        union = np.unique(np.concatenate(columns)) if columns else np.empty(0, dtype=np.int64)
        return self._stage_columns(_cpu_array(block_table_row), [union])
