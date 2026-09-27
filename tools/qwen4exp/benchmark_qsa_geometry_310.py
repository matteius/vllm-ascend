# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched eager QSA selection timings for two source snapshots on idle 310P.

This measures the selection callback between breakable graph segments, not
whole-model throughput. Both snapshots must export the same selection API.
Use external temperature supervision and do not run alongside serving work.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch_npu

from tools.qwen4exp.benchmark_w4_projection_310 import timed_ms
from vllm_ascend.utils import enable_custom_op


def _load_snapshot(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import snapshot: {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve annotations using their defining module.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-module", type=Path, required=True)
    parser.add_argument("--candidate-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output path")
    if args.iterations <= 0 or args.repeats <= 0:
        parser.error("iterations and repeats must be positive")
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        raise RuntimeError("requires Ascend 310P")
    snapshots = [
        _load_snapshot(args.reference_module, "qsa_geometry_reference"),
        _load_snapshot(args.candidate_module, "qsa_geometry_candidate"),
    ]
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.reference_module, args.candidate_module)]
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    generator = torch.Generator().manual_seed(1024)
    with args.output.open("x") as output:
        for capacity in (512, 2048, 5856, 8192):
            pages = capacity // 16
            for tokens in (1, 2, 3, 5, 8):
                query = torch.randn((tokens, 4, 128), generator=generator).half().npu()
                cache = torch.randn((pages + 7, 19, 128), generator=generator).half().npu()
                table = torch.randperm(pages + 7, generator=generator)[:pages].int().unsqueeze(0).npu()
                boundaries = torch.tensor([0, tokens], dtype=torch.int32, device="npu:0")
                positions = torch.arange(capacity * 4 - tokens - 1, capacity * 4 - 1, dtype=torch.int32).npu()

                def select(module, query=query, cache=cache, table=table, boundaries=boundaries, positions=positions):
                    return module.qsa_indexer_select_groups_310(
                        query,
                        cache,
                        table,
                        boundaries,
                        positions,
                        compress_ratio=4,
                        token_topk=2048,
                        max_matmul_decode_tokens=8,
                    )

                reference, candidate = [select(module) for module in snapshots]
                for field in reference.__dataclass_fields__:
                    torch.testing.assert_close(
                        getattr(candidate, field).cpu(), getattr(reference, field).cpu(), rtol=0, atol=0
                    )
                baseline = timed_ms(lambda: select(snapshots[0]), args.iterations, args.repeats)
                candidate_timing = timed_ms(lambda: select(snapshots[1]), args.iterations, args.repeats)
                record = {
                    "query_tokens": tokens,
                    "capacity_groups": capacity,
                    "reference_sha256": hashes[0],
                    "candidate_sha256": hashes[1],
                    "reference_selection": baseline,
                    "candidate_selection": candidate_timing,
                    "all_selection_fields_exact": True,
                    "execution": "eager_between_breakable_graph_segments",
                    "iterations": args.iterations,
                    "repeats": args.repeats,
                    "not_model_throughput": True,
                }
                line = json.dumps(record)
                print(line, flush=True)
                output.write(line + "\n")
                output.flush()


if __name__ == "__main__":
    main()
