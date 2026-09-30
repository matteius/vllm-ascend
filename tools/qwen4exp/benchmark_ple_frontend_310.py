# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure Qwen4Exp PLE hashing, host gather, H2D, and projection on 310P.

The benchmark uses the checkpoint's real n-gram table rows and PLE projection
weights. It compares the current device-hash/D2H frontend with host hashing from
the runner's existing CPU mirrors. Run with the inference server stopped and
save every result to a new output path.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torch_npu
from safetensors import safe_open

from vllm_ascend.models.qwen4_exp.lm_head_w8a8 import quantize_lm_head_weight
from vllm_ascend.models.qwen4_exp.ngram_embedding import (
    DEFAULT_SAFETENSORS_INDEX,
    AscendPLELazyShardEmbeddingMethod,
    AscendQwen4ExpNGramEmbedding,
)
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, maybe_trans_nz

PLE_PREFIX = "model.language_model.layers.1.ple"


def timed_host_ms(function, iterations: int, repeats: int) -> dict[str, object]:
    for _ in range(3):
        function()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(iterations):
            function()
        times.append((time.perf_counter() - started) * 1000 / iterations)
    return {"median_ms": statistics.median(times), "trials_ms": times}


def timed_device_ms(function, iterations: int, repeats: int) -> dict[str, object]:
    for _ in range(3):
        function()
    torch.npu.synchronize()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(iterations):
            function()
        torch.npu.synchronize()
        times.append((time.perf_counter() - started) * 1000 / iterations)
    return {"median_ms": statistics.median(times), "trials_ms": times}


def load_tensor(model: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(model / weight_map[name], framework="pt", device="cpu") as shard:
        return shard.get_tensor(name)


def main() -> None:
    """Load real PLE tensors and record each frontend stage independently."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--label", default="unlabeled")
    args = parser.parse_args()
    if args.output.exists() or min(*args.tokens, args.iterations, args.repeats) < 1:
        parser.error("use a new output path and positive token/iteration/repeat counts")
    if not torch_npu.npu.is_available() or "310P" not in torch_npu.npu.get_device_name(0).upper():
        raise RuntimeError("requires Ascend 310P")

    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    raw_config = json.loads((args.model / "config.json").read_text())
    config = SimpleNamespace(**raw_config.get("text_config", raw_config))
    index_filename = (
        "model.safetensors.index.json"
        if (args.model / "model.safetensors.index.json").exists()
        else DEFAULT_SAFETENSORS_INDEX
    )
    weight_map = json.loads((args.model / index_filename).read_text())["weight_map"]
    host_hasher = AscendQwen4ExpNGramEmbedding(config=config, ple_dense_layer_id=0)
    device_hasher = AscendQwen4ExpNGramEmbedding(config=config, ple_dense_layer_id=0).npu()
    table = AscendPLELazyShardEmbeddingMethod(
        host_hasher.padded_vocab_size,
        host_hasher.head_dim,
        checkpoint_dir=args.model,
        index_filename=index_filename,
        split_ngram_parts=host_hasher.split_ngram_parts,
    )

    key_weight = load_tensor(args.model, weight_map, f"{PLE_PREFIX}.key_proj.weight")
    value_weight = load_tensor(args.model, weight_map, f"{PLE_PREFIX}.value_proj.weight")
    projection_weight = torch.cat((key_weight, value_weight), dim=0).half().npu()
    projection_weight = torch_npu.npu_format_cast(projection_weight, ACL_FORMAT_FRACTAL_NZ)
    quantized_weight, quantized_weight_scale = quantize_lm_head_weight(torch.cat((key_weight, value_weight), dim=0))
    quantized_weight = maybe_trans_nz(quantized_weight.npu()).transpose(0, 1)
    quantized_weight_scale = quantized_weight_scale.npu()
    generator = torch.Generator().manual_seed(31038)

    with args.output.open("x") as output_file:
        for num_tokens in args.tokens:
            input_ids_cpu = torch.randint(
                0,
                int(config.vocab_size),
                (num_tokens,),
                dtype=torch.int64,
                generator=generator,
            )
            query_start_cpu = torch.tensor([0, num_tokens], dtype=torch.int64)
            context_cpu = torch.full((1, int(config.ngram_size) - 1), int(config.eos_token_id), dtype=torch.int64)
            input_ids_npu = input_ids_cpu.npu()
            query_start_npu = query_start_cpu.npu()
            context_npu = context_cpu.npu()

            def hash_host(
                input_ids: torch.Tensor = input_ids_cpu,
                query_start: torch.Tensor = query_start_cpu,
                context: torch.Tensor = context_cpu,
            ) -> torch.Tensor:
                return host_hasher.compute_ngram_ids(input_ids, query_start, context)

            def hash_device_to_host(
                input_ids: torch.Tensor = input_ids_npu,
                query_start: torch.Tensor = query_start_npu,
                context: torch.Tensor = context_npu,
            ) -> torch.Tensor:
                return device_hasher.compute_ngram_ids(input_ids, query_start, context).cpu()

            host_ids = hash_host()
            device_ids = hash_device_to_host()
            torch.testing.assert_close(device_ids, host_ids, rtol=0, atol=0)
            rows_cpu = table.gather_rows(host_ids)
            embeddings_cpu = rows_cpu.reshape(num_tokens, -1)
            embeddings_npu = embeddings_cpu.npu()
            expected_projection = F.linear(embeddings_npu, projection_weight)
            torch.npu.synchronize()

            def gather_host(ids: torch.Tensor = host_ids) -> torch.Tensor:
                return table.gather_rows(ids)

            def copy_h2d(embeddings: torch.Tensor = embeddings_cpu) -> torch.Tensor:
                return embeddings.npu()

            def project(
                embeddings: torch.Tensor = embeddings_npu,
                weight: torch.Tensor = projection_weight,
            ) -> torch.Tensor:
                return F.linear(embeddings, weight)

            def project_w8a8(
                embeddings: torch.Tensor = embeddings_npu,
                weight: torch.Tensor = quantized_weight,
                weight_scale: torch.Tensor = quantized_weight_scale,
            ) -> torch.Tensor:
                quantized_embeddings, input_scale = torch_npu.npu_dynamic_quant(embeddings)
                return torch_npu.npu_quant_matmul(
                    quantized_embeddings,
                    weight,
                    weight_scale,
                    pertoken_scale=input_scale,
                    output_dtype=embeddings.dtype,
                )

            def old_frontend(
                input_ids: torch.Tensor = input_ids_npu,
                query_start: torch.Tensor = query_start_npu,
                context: torch.Tensor = context_npu,
                tokens: int = num_tokens,
                weight: torch.Tensor = projection_weight,
            ) -> torch.Tensor:
                ids = device_hasher.compute_ngram_ids(input_ids, query_start, context).cpu()
                embeddings = table.gather_rows(ids).reshape(tokens, -1).npu()
                return F.linear(embeddings, weight)

            def host_hash_frontend(
                input_ids: torch.Tensor = input_ids_cpu,
                query_start: torch.Tensor = query_start_cpu,
                context: torch.Tensor = context_cpu,
                tokens: int = num_tokens,
                weight: torch.Tensor = projection_weight,
            ) -> torch.Tensor:
                ids = host_hasher.compute_ngram_ids(input_ids, query_start, context)
                embeddings = table.gather_rows(ids).reshape(tokens, -1).npu()
                return F.linear(embeddings, weight)

            actual_projection = host_hash_frontend()
            torch.testing.assert_close(actual_projection.cpu(), expected_projection.cpu(), rtol=0, atol=0)
            w8a8_projection = project_w8a8()
            projection_error = (w8a8_projection.float() - expected_projection.float()).cpu()
            expected_projection_cpu = expected_projection.float().cpu()
            record = {
                "scope": "qwen4exp_ple_frontend_not_whole_model",
                "label": args.label,
                "tokens": num_tokens,
                "ngram_heads": host_hasher.ngram_heads,
                "head_dim": host_hasher.head_dim,
                "projection_weight_shape": list(projection_weight.shape),
                "device_hash_and_d2h": timed_device_ms(hash_device_to_host, args.iterations, args.repeats),
                "host_hash": timed_host_ms(hash_host, args.iterations, args.repeats),
                "lazy_gather_hot": timed_host_ms(gather_host, args.iterations, args.repeats),
                "pageable_h2d": timed_device_ms(copy_h2d, args.iterations, args.repeats),
                "projection": timed_device_ms(project, args.iterations, args.repeats),
                "dynamic_w8a8_projection": timed_device_ms(project_w8a8, args.iterations, args.repeats),
                "dynamic_w8a8_projection_error": {
                    "relative_l2": float(projection_error.norm() / expected_projection_cpu.norm()),
                    "max_abs": float(projection_error.abs().max()),
                    "cosine": float(
                        F.cosine_similarity(
                            w8a8_projection.float().reshape(-1),
                            expected_projection.float().reshape(-1),
                            dim=0,
                        ).cpu()
                    ),
                },
                "old_frontend": timed_device_ms(old_frontend, args.iterations, args.repeats),
                "host_hash_frontend": timed_device_ms(host_hash_frontend, args.iterations, args.repeats),
                "iterations": args.iterations,
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
            }
            line = json.dumps(record)
            print(line, flush=True)
            output_file.write(line + "\n")
            output_file.flush()

    table.close()


if __name__ == "__main__":
    main()
