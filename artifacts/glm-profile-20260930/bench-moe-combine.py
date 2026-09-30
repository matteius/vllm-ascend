"""Measure the GLM routed/shared combine at one decode token on 310P."""

import statistics
import time

import torch
import torch_npu  # noqa: F401


def current(routed: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    scale = torch.as_tensor(2.5, dtype=routed.dtype, device=routed.device)
    return torch.addcmul(shared, routed, scale)


def candidate(routed: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    return torch.add(shared, routed, alpha=2.5)


def time_variant(fn, routed, shared, runs=80):
    for _ in range(5):
        fn(routed, shared)
    torch.npu.synchronize()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = fn(routed, shared)
        torch.npu.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return result, statistics.median(times), statistics.mean(times)


def main():
    torch.npu.set_device(0)
    torch.manual_seed(19)
    routed = torch.randn(1, 4096, dtype=torch.float16, device="npu").float()
    shared = torch.randn(1, 4096, dtype=torch.float16, device="npu").float()
    baseline, base_median, base_mean = time_variant(current, routed, shared)
    updated, new_median, new_mean = time_variant(candidate, routed, shared)
    error = (baseline - updated).abs().max().item()
    print(
        f"old_median_ms={base_median:.4f} old_mean_ms={base_mean:.4f} "
        f"new_median_ms={new_median:.4f} new_mean_ms={new_mean:.4f} "
        f"max_abs_error={error:.8g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
