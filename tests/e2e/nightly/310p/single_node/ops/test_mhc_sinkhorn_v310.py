# SPDX-License-Identifier: Apache-2.0
"""Parity and latency of the 310P four-stream mHC Sinkhorn kernel."""

import statistics
import time
from types import SimpleNamespace

import pytest
import torch
import torch_npu
from vllm.model_executor.layers.mhc import MHCPreOp

import vllm_ascend.patch.worker.patch_mhc_norm  # noqa: F401
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(autouse=True, scope="module")
def require_kernel():
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    assert hasattr(torch.ops._C_ascend, "mhc_sinkhorn_310")


def _reference(logits: torch.Tensor, iterations: int, epsilon: float) -> torch.Tensor:
    mix = torch.softmax(logits, dim=-1) + epsilon
    mix = mix / (mix.sum(dim=-2, keepdim=True) + epsilon)
    for _ in range(iterations - 1):
        mix = mix / (mix.sum(dim=-1, keepdim=True) + epsilon)
        mix = mix / (mix.sum(dim=-2, keepdim=True) + epsilon)
    return mix


@pytest.mark.parametrize(
    "rows,scale,iterations", [(1, 0.2, 20), (4, 1.0, 20), (17, 4.0, 20), (512, 2.0, 20), (2, 2.0, 1)]
)
def test_mhc_sinkhorn_matches_torch(rows: int, scale: float, iterations: int):
    torch.manual_seed(rows + iterations)
    logits = (torch.randn(rows, 4, 4) * scale).npu()
    epsilon = 1e-6
    expected = _reference(logits, iterations, epsilon)
    actual = torch.ops._C_ascend.mhc_sinkhorn_310(logits, iterations, epsilon)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-4, atol=1e-5)


def test_mhc_sinkhorn_rejects_invalid_geometry():
    with pytest.raises(RuntimeError, match=r"\[tokens,4,4\]"):
        torch.ops._C_ascend.mhc_sinkhorn_310(torch.randn(1, 3, 3).npu())


def test_mhc_pre_matches_eager_path():
    torch.manual_seed(93)
    streams, hidden = 4, 256
    mix_width = 2 * streams + streams * streams
    residual = torch.randn(2, streams, hidden, dtype=torch.float32, device="npu")
    fn = torch.randn(mix_width, streams * hidden, dtype=torch.float32, device="npu") * 0.01
    hc_scale = torch.tensor([0.05, 0.05, 0.05], device="npu")
    hc_base = torch.zeros(mix_width, dtype=torch.float32, device="npu")
    norm_weight = torch.ones(hidden, dtype=torch.float16, device="npu")
    op = SimpleNamespace(use_310p_sinkhorn=False)
    args = (residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 2.0, 20)
    baseline = MHCPreOp.forward_native(op, *args, norm_weight=norm_weight, norm_eps=1e-6)
    op.use_310p_sinkhorn = True
    fused = MHCPreOp.forward_native(op, *args, norm_weight=norm_weight, norm_eps=1e-6)
    for expected, actual in zip(baseline, fused):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("rows", [1, 512])
def test_mhc_sinkhorn_latency(rows: int):
    logits = torch.randn(rows, 4, 4, dtype=torch.float32, device="npu")
    for _ in range(5):
        torch.ops._C_ascend.mhc_sinkhorn_310(logits)
        _reference(logits, 20, 1e-6)
    torch.npu.synchronize()
    timings = {}
    for label, fn in (
        ("fused", lambda: torch.ops._C_ascend.mhc_sinkhorn_310(logits)),
        ("eager", lambda: _reference(logits, 20, 1e-6)),
    ):
        samples = []
        for _ in range(20):
            start = time.perf_counter()
            fn()
            torch.npu.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        timings[label] = statistics.median(samples)
    print(f"mHC sinkhorn rows={rows} fused={timings['fused']:.4f} ms eager={timings['eager']:.4f} ms")
    assert timings["fused"] < timings["eager"]
