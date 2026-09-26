#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline, resumable Qwen4Exp routed-expert W4 export using ModelSlim IR.

This is per-group asymmetric min/max RTN, NOT calibrated GPTQ/AWQ. The input
must be the original floating-point checkpoint. No model or accelerator is
initialized. A completed index is published only after every shard is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import regex as re
import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT = "qwen4exp_w4a16_group_v1"
DEFAULT_GROUP_SIZE = 128
BANK = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")


def quantize_projection(weight: torch.Tensor, group_size: int) -> dict[str, torch.Tensor]:
    # Lazy dependency: --help and inspection do not require ModelSlim installed.
    from msmodelslim.format.common.pack import _pack_int4
    from msmodelslim.ir.api import calculate_qparam, quantize
    from msmodelslim.ir.qal import QDType, QScope, QStorage

    if weight.ndim != 2 or not weight.is_floating_point() or weight.shape[1] % group_size:
        raise ValueError("expected floating [out, in] weights with in divisible by group_size")
    weight = weight.float()
    if not torch.isfinite(weight).all():
        raise ValueError("nonfinite source weight")
    groups = weight.reshape(weight.shape[0], -1, group_size)
    params = calculate_qparam(groups.amin(-1), groups.amax(-1), QDType.INT4, QScope.PER_GROUP, False)
    params.ext["group_size"] = group_size
    # ModelSlim's per-group IR quantizer takes [in, out], grouping on in.
    quant = quantize(QStorage(QDType.FLOAT, weight.t()), params).value.t().contiguous()
    scale = params.ext["scale"].to(torch.float16)
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("quantization scale cannot be stored as finite positive FP16")
    return {
        "weight": _pack_int4(quant).contiguous(),
        "weight_scale": scale.contiguous(),
        "weight_offset": params.ext["offset"].to(torch.int8).contiguous(),
    }


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(data)
    return result.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def convert(source: Path, destination: Path, *, group_size: int = DEFAULT_GROUP_SIZE, resume: bool = False) -> dict:
    import msmodelslim

    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("destination must be separate from the source")
    if group_size <= 0 or group_size % 2:
        raise ValueError("group_size must be a positive even integer")
    config = json.loads((source / "config.json").read_text())
    text_config = config["text_config"]
    if (
        config.get("model_type") != "qwen4_exp"
        or any(
            key in cfg for cfg in (config, text_config) for key in ("quantization_config", "ascend_expert_quantization")
        )
        or (source / "quant_model_description.json").exists()
    ):
        raise ValueError("requires the original unquantized qwen4_exp checkpoint")
    index_path = source / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    hidden, intermediate = text_config["hidden_size"], text_config["moe_intermediate_size"]
    layers, experts = text_config["num_hidden_layers"], text_config["num_experts"]
    if hidden % group_size or intermediate % group_size:
        raise ValueError("group_size must divide hidden and expert intermediate dimensions")
    for filename in set(weight_map.values()):
        if Path(filename).name != filename or not (source / filename).is_file():
            raise ValueError(f"missing or unsafe source shard: {filename}")
    # Header-only preflight: reject incomplete banks before producing output.
    for layer in range(layers):
        for projection, shape in (
            ("gate_up_proj", [experts, 2 * intermediate, hidden]),
            ("down_proj", [experts, hidden, intermediate]),
        ):
            name = f"model.language_model.layers.{layer}.mlp.experts.{projection}"
            with safe_open(source / weight_map[name], framework="pt", device="cpu") as handle:
                view = handle.get_slice(name)
                if view.get_shape() != shape or view.get_dtype() not in ("BF16", "F16", "F32"):
                    raise ValueError(f"invalid floating expert bank: {name}")
    slim_root = Path(msmodelslim.__file__).resolve().parents[1]
    revision = subprocess.run(["git", "-C", str(slim_root), "rev-parse", "HEAD"], capture_output=True, text=True)
    identity = {
        "format": FORMAT,
        "source": str(source),
        "source_index_sha256": digest(index_path),
        "source_config_sha256": digest(source / "config.json"),
        "group_size": group_size,
        "method": "ModelSlim IR asymmetric per-group minmax RTN",
        "calibrated": False,
        "scale_dtype": "float16",
        "offset_dtype": "int8",
        "modelslim_revision": revision.stdout.strip() if revision.returncode == 0 else "unknown",
        "torch_version": torch.__version__,
        "source_shards": {
            name: {"size": (source / name).stat().st_size, "mtime_ns": (source / name).stat().st_mtime_ns}
            for name in sorted(set(weight_map.values()))
        },
    }
    journal_path = destination / "build-journal.json"
    if destination.exists() and any(destination.iterdir()):
        if not resume or not journal_path.is_file():
            raise ValueError("requires an empty destination (or --resume with matching journal)")
        journal = json.loads(journal_path.read_text())
        if journal["identity"] != identity:
            raise ValueError("resume source, tooling, or quantization settings changed")
    else:
        destination.mkdir(parents=True, exist_ok=True)
        journal = {"identity": identity, "shards": {}, "complete": False}
        write_json(journal_path, journal)
    receipts = destination / "build-receipts"
    receipts.mkdir(exist_ok=True)

    def done(filename: str) -> bool:
        record = journal["shards"].get(filename)
        if record is None:
            return False
        if digest(destination / filename) != record["sha256"]:
            raise ValueError(f"corrupt completed shard: {filename}")
        return True

    def save(filename: str, tensors: dict[str, torch.Tensor]) -> None:
        temporary = destination / (filename + ".tmp")
        save_file(tensors, str(temporary), metadata={"format": "pt", "expert_quantization": FORMAT})
        temporary.replace(destination / filename)
        record = {
            "sha256": digest(destination / filename),
            "tensors": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "bytes": value.numel() * value.element_size(),
                }
                for name, value in tensors.items()
            },
        }
        write_json(receipts / (filename + ".json"), record)
        journal["shards"][filename] = {"sha256": record["sha256"]}
        write_json(journal_path, journal)
        print(
            json.dumps({"saved": filename, "completed_shards": len(journal["shards"]), "time": time.time()}), flush=True
        )

    for layer in range(layers):
        filename = f"experts-{layer:03d}.safetensors"
        if done(filename):
            continue
        pending = {}
        prefix = f"model.language_model.layers.{layer}.mlp.experts"
        for bank_name, projections in (("gate_up_proj", ("gate_proj", "up_proj")), ("down_proj", ("down_proj",))):
            name = f"{prefix}.{bank_name}"
            with safe_open(source / weight_map[name], framework="pt", device="cpu") as handle:
                view = handle.get_slice(name)
                for expert in range(experts):
                    for part, projection in enumerate(projections):
                        weight = (
                            view[expert, part * intermediate : (part + 1) * intermediate, :]
                            if bank_name == "gate_up_proj"
                            else view[expert, :, :]
                        )
                        for kind, tensor in quantize_projection(weight, group_size).items():
                            pending[f"{prefix}.{expert}.{projection}.{kind}"] = tensor
        save(filename, pending)
        del pending

    # Keep PLE's existing split tensors; never materialize the entire host table.
    # One untouched source tensor per output shard also keeps MTP namespaces intact.
    for position, name in enumerate(sorted(weight_map)):
        if BANK.fullmatch(name):
            continue
        filename = f"float-{position:05d}.safetensors"
        if done(filename):
            continue
        with safe_open(source / weight_map[name], framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name)
            if tensor.is_floating_point():
                tensor = tensor.to(torch.float32 if tensor.dtype == torch.float32 else torch.float16)
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"nonfinite floating export: {name}")
            save(filename, {name: tensor.contiguous()})
        del tensor

    output_map, total_bytes = {}, 0
    for filename in journal["shards"]:
        record = json.loads((receipts / (filename + ".json")).read_text())
        for name, tensor in record["tensors"].items():
            if name in output_map:
                raise ValueError(f"duplicate tensor: {name}")
            output_map[name] = filename
            total_bytes += tensor["bytes"]
    expected_quantized = layers * experts * 3 * 3
    if (
        sum(".mlp.experts." in name and name.startswith("model.language_model.layers.") for name in output_map)
        != expected_quantized
    ):
        raise ValueError("incomplete quantized expert inventory")
    for path in source.iterdir():
        if path.is_file() and (
            path.name.startswith("tokenizer")
            or path.name
            in (
                "generation_config.json",
                "chat_template.jinja",
                "preprocessor_config.json",
                "video_preprocessor_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "vocab.json",
                "merges.txt",
                "LICENSE",
            )
        ):
            shutil.copyfile(path, destination / path.name)
    text_config["dtype"] = "float16"
    text_config["ascend_expert_quantization"] = {
        "format": FORMAT,
        "bits": 4,
        "group_size": group_size,
        "symmetric": False,
        "packing": "signed_int4_low_nibble_first_in_axis",
        "backend": "eager_dequant",
        "scale_dtype": "float16",
        "offset_dtype": "int8",
    }
    write_json(destination / "config.json", config)
    write_json(destination / "quantization_provenance.json", identity)
    write_json(
        destination / "model.safetensors.index.json",
        {"metadata": {"total_size": total_bytes}, "weight_map": output_map},
    )
    journal["complete"] = True
    journal["total_bytes"] = total_bytes
    write_json(journal_path, journal)
    return {"total_bytes": total_bytes, "shards": len(journal["shards"]), "quantized_projections": layers * experts * 3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    with torch.inference_mode():
        print(json.dumps(convert(args.source, args.output, group_size=args.group_size, resume=args.resume)), flush=True)


if __name__ == "__main__":
    main()
