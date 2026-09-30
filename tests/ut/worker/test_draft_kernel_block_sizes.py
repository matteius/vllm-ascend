# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm_ascend.worker.model_runner_v1 import _select_draft_kernel_block_sizes


def test_select_draft_kernel_block_sizes_flattens_310p_groups() -> None:
    assert _select_draft_kernel_block_sizes([[64], [0], [64]]) == [64, 0, 64]
    assert _select_draft_kernel_block_sizes([64, 0, 64]) == [64, 0, 64]
