# SPDX-License-Identifier: Apache-2.0
"""310P MLA cache write with scheduler int32 and explicit int64 slots."""

import pytest
import torch
import torch_npu

import vllm_ascend.ops  # noqa: F401
from vllm_ascend._310p.attention.mla_v1_310 import _write_nz_latent_cache
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(autouse=True, scope="module")
def require_kernel():
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    assert hasattr(torch.ops._C_ascend, "mla_cache_write_310")


@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_mla_cache_write_accepts_scheduler_slots(slot_dtype: torch.dtype):
    cache = torch.zeros(3, 2, 4, 16, dtype=torch.float16, device="npu")
    rows = torch.arange(4 * 32, dtype=torch.float16, device="npu").view(4, 2, 16)
    slots = torch.tensor([0, 7, -1, 9], dtype=slot_dtype, device="npu")

    _write_nz_latent_cache(cache, rows, slots)
    got = cache.cpu()
    expected_rows = rows.cpu()
    torch.testing.assert_close(got[0, :, 0], expected_rows[0])
    torch.testing.assert_close(got[1, :, 3], expected_rows[1])
    torch.testing.assert_close(got[2, :, 1], expected_rows[3])
    assert torch.count_nonzero(got) == torch.count_nonzero(expected_rows[[0, 1, 3]])
