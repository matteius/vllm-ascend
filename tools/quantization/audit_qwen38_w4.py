#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit a completed W4 artifact offline; sample reconstruction is not an accuracy evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import regex as re
import torch
import torch.nn.functional as F
from safetensors import safe_open

EXPERT = re.compile(
    r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale|weight_offset)$"
)


def audit(path: Path, *, sample_layers: list[int], sample_experts: list[int]) -> dict:
    """Check the complete header inventory, then compare selected real experts with BF16 sources."""
    journal = json.loads((path / "build-journal.json").read_text())
    if not journal["complete"]:
        raise ValueError("build is incomplete")
    config = json.loads((path / "config.json").read_text())["text_config"]
    group = config["ascend_expert_quantization"]["group_size"]
    index = json.loads((path / "model.safetensors.index.json").read_text())
    source = Path(journal["identity"]["source"])
    original = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    source_headers = {}
    for filename in set(original.values()):
        with safe_open(source / filename, framework="pt", device="cpu") as handle:
            for name in list(handle.keys()):
                view = handle.get_slice(name)
                source_headers[name] = (view.get_shape(), view.get_dtype())
    files = defaultdict(set)
    for name, filename in index["weight_map"].items():
        if Path(filename).name != filename:
            raise ValueError("unsafe shard path")
        files[filename].add(name)
    counts = defaultdict(int)
    payload_bytes = defaultdict(int)
    expert_keys = set()
    for filename, expected_keys in files.items():
        with safe_open(path / filename, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != expected_keys:
                raise ValueError(f"index/header mismatch in {filename}")
            for name in expected_keys:
                view = handle.get_slice(name)
                shape, dtype = view.get_shape(), view.get_dtype()
                match = EXPERT.fullmatch(name)
                if match:
                    layer, expert, projection, kind = match.groups()
                    if int(layer) >= config["num_hidden_layers"] or int(expert) >= config["num_experts"]:
                        raise ValueError(f"expert outside model geometry: {name}")
                    out, inputs = (
                        (config["hidden_size"], config["moe_intermediate_size"])
                        if projection == "down_proj"
                        else (config["moe_intermediate_size"], config["hidden_size"])
                    )
                    expected = [out, inputs // (2 if kind == "weight" else group)]
                    if shape != expected or dtype != ("F16" if kind == "weight_scale" else "I8"):
                        raise ValueError(f"invalid packed tensor {name}: {shape} {dtype}")
                    expert_keys.add(name)
                    category = "routed_experts"
                else:
                    if name not in original:
                        raise ValueError(f"unexpected non-expert tensor: {name}")
                    original_shape, original_dtype = source_headers[name]
                    expected_dtype = "F16" if original_dtype == "BF16" else original_dtype
                    if shape != original_shape or dtype != expected_dtype:
                        raise ValueError(f"untouched tensor shape/dtype mismatch: {name}")
                    category = "ple" if ".ngram_embedding.shard_" in name else "other_float"
                size = (
                    torch.tensor(shape).prod().item()
                    * {"I8": 1, "U8": 1, "F16": 2, "BF16": 2, "F32": 4, "I64": 8, "I32": 4}[dtype]
                )
                counts[category] += 1
                payload_bytes[category] += size
    expected_count = config["num_hidden_layers"] * config["num_experts"] * 9
    if len(expert_keys) != expected_count:
        raise ValueError("incomplete expert inventory")
    untouched = {
        name
        for name in original
        if not (
            name.startswith("model.language_model.layers.")
            and name.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj"))
        )
    }
    if set(index["weight_map"]) - expert_keys != untouched:
        raise ValueError("incomplete untouched tensor inventory")
    if sum(payload_bytes.values()) != index["metadata"]["total_size"]:
        raise ValueError("index byte count mismatch")
    samples = []
    generator = torch.Generator().manual_seed(1024)
    for layer in sample_layers:
        for expert in sample_experts:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = f"model.language_model.layers.{layer}.mlp.experts"
                key = f"{prefix}.{expert}.{projection}"
                with safe_open(path / index["weight_map"][key + ".weight"], framework="pt", device="cpu") as handle:
                    packed = handle.get_tensor(key + ".weight").to(torch.int16)
                    scale = handle.get_tensor(key + ".weight_scale").float()
                    offset = handle.get_tensor(key + ".weight_offset").float()
                    q = torch.stack((packed & 15, (packed >> 4) & 15), -1).flatten(-2)
                    q = torch.where(q >= 8, q - 16, q).float()
                    reconstructed = ((q.reshape(*scale.shape, group) - offset[..., None]) * scale[..., None]).flatten(
                        -2
                    )
                source_key = prefix + (".down_proj" if projection == "down_proj" else ".gate_up_proj")
                with safe_open(source / original[source_key], framework="pt", device="cpu") as handle:
                    source_view = handle.get_slice(source_key)
                    if projection == "down_proj":
                        reference = source_view[expert, :, :].float()
                    else:
                        start = 0 if projection == "gate_proj" else config["moe_intermediate_size"]
                        reference = source_view[expert, start : start + config["moe_intermediate_size"], :].float()
                inputs = torch.randn(8, reference.shape[1], generator=generator)
                output, expected = F.linear(inputs, reconstructed), F.linear(inputs, reference)
                samples.append(
                    {
                        "projection": key,
                        "relative_weight_rmse": ((reference - reconstructed).square().sum() / reference.square().sum())
                        .sqrt()
                        .item(),
                        "weight_cosine": F.cosine_similarity(
                            reference.flatten(), reconstructed.flatten(), dim=0
                        ).item(),
                        "synthetic_projection_relative_rmse": (
                            (output - expected).square().sum() / expected.square().sum()
                        )
                        .sqrt()
                        .item(),
                        "finite": bool(torch.isfinite(reconstructed).all() and torch.isfinite(output).all()),
                    }
                )
    if not all(sample["finite"] for sample in samples):
        raise ValueError("nonfinite reconstruction")
    return {
        "complete": True,
        "header_inventory_valid": True,
        "shards": len(files),
        "tensor_counts": dict(counts),
        "tensor_bytes": dict(payload_bytes),
        "tensor_gib": {key: value / 1024**3 for key, value in payload_bytes.items()},
        "samples": samples,
        "full_model_accuracy_validated": False,
        "npu_execution_validated": False,
        "note": (
            "Weight reconstruction and Gaussian-input projection diagnostics only; "
            "not perplexity, task accuracy, or throughput."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 12, 24, 36, 47])
    parser.add_argument("--experts", type=int, nargs="+", default=[0, 255, 511])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    with torch.inference_mode():
        result = audit(args.checkpoint, sample_layers=args.layers, sample_experts=args.experts)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
