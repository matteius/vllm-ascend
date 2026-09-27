# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare old/new eager greedy rejection on idle 310P, not model throughput."""

import argparse
import hashlib
import importlib.util
import json
import sys
from functools import partial
from pathlib import Path

import torch
import torch_npu

from tools.qwen4exp.benchmark_w4_projection_310 import timed_ms
from vllm_ascend.sample import rejection_sampler, uniform_greedy_rejection


def _load_reference(path):
    spec = importlib.util.spec_from_file_location("rejection_reference", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load reference: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.rejection_greedy_sample_pytorch


def _run(function, result, cumulative, drafts, targets, bonus, width, batch_size):
    result.fill_(-1)
    function(result, cumulative, drafts, targets, bonus, [width] * batch_size, width)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or args.iterations <= 0 or args.repeats <= 0:
        parser.error("use a new output path and positive iterations/repeats")
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    reference = _load_reference(args.reference_module)
    candidate = rejection_sampler.rejection_greedy_sample_pytorch
    hashes = {
        "reference_sha256": hashlib.sha256(args.reference_module.read_bytes()).hexdigest(),
        "dispatcher_sha256": hashlib.sha256(Path(rejection_sampler.__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(Path(uniform_greedy_rejection.__file__).read_bytes()).hexdigest(),
    }
    with args.output.open("x") as output:
        for width in (2, 4, 8):
            for batch_size in (1, 4, 16):
                for accept_all in (False, True):
                    targets = torch.arange(batch_size * width, dtype=torch.int64).npu()
                    drafts = targets.clone()
                    if not accept_all:
                        drafts.reshape(batch_size, width)[:, width // 2] += 100
                    # The old indexed bonus assignment requires matching the
                    # output dtype, as the real sampler's bonus tokens do.
                    bonus = torch.arange(batch_size, dtype=torch.int32).npu().reshape(-1, 1) + 1000
                    cumulative = torch.arange(1, batch_size + 1, dtype=torch.int32).npu() * width
                    values = [torch.empty(batch_size, width + 1, dtype=torch.int32, device="npu") for _ in range(2)]

                    runs = [
                        partial(_run, function, value, cumulative, drafts, targets, bonus, width, batch_size)
                        for function, value in zip((reference, candidate), values)
                    ]
                    runs[0]()
                    runs[1]()
                    torch.testing.assert_close(values[0].cpu(), values[1].cpu(), rtol=0, atol=0)
                    baseline = timed_ms(runs[0], args.iterations, args.repeats)
                    improved = timed_ms(runs[1], args.iterations, args.repeats)
                    record = {
                        "batch_size": batch_size,
                        "draft_tokens": width,
                        "accept_all": accept_all,
                        "all_outputs_exact": True,
                        "reference_eager": baseline,
                        "candidate_eager": improved,
                        "iterations": args.iterations,
                        "repeats": args.repeats,
                        "not_model_throughput": True,
                        **hashes,
                    }
                    line = json.dumps(record)
                    print(line, flush=True)
                    output.write(line + "\n")
                    output.flush()


if __name__ == "__main__":
    main()
