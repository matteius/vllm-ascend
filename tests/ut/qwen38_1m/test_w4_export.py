# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelSlim integration tests, runnable without importing vLLM or using NPUs."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

_SCRIPT = Path(__file__).parents[3] / "tools/quantization/qwen38_modelslim_w4.py"
_SPEC = importlib.util.spec_from_file_location("qwen38_w4_export", _SCRIPT)
export = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export)
_AUDIT_SPEC = importlib.util.spec_from_file_location("qwen38_w4_audit", _SCRIPT.with_name("audit_qwen38_w4.py"))
audit_module = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(audit_module)


@pytest.fixture
def checkpoint(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "model_type": "qwen4_exp",
        "text_config": {"hidden_size": 16, "moe_intermediate_size": 8, "num_experts": 2, "num_hidden_layers": 1},
    }
    export.write_json(source / "config.json", config)
    tensors = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 16, 16).bfloat16(),
        "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(2, 16, 8).bfloat16(),
        "mtp.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 16, 16).bfloat16(),
        "model.language_model.layers.0.mlp.gate.weight": torch.randn(2, 16).bfloat16(),
    }
    save_file(tensors, str(source / "model.safetensors"))
    export.write_json(
        source / "model.safetensors.index.json", {"weight_map": dict.fromkeys(tensors, "model.safetensors")}
    )
    return source, tensors


def test_modelslim_roundtrip_and_signed_nibbles():
    pytest.importorskip("msmodelslim")
    weight = torch.arange(-8, 8).float().repeat(2, 1)
    result = export.quantize_projection(weight, 16)
    assert result["weight"].dtype == torch.int8
    packed = result["weight"].to(torch.int16)
    q = torch.stack((packed & 15, (packed >> 4) & 15), -1).reshape_as(weight)
    q = torch.where(q >= 8, q - 16, q)
    actual = (q - result["weight_offset"]) * result["weight_scale"].float()
    torch.testing.assert_close(actual, weight)


def test_export_complete_resume_preserves_mtp(checkpoint, tmp_path):
    pytest.importorskip("msmodelslim")
    source, tensors = checkpoint
    dest = tmp_path / "output"
    before = export.digest(source / "model.safetensors")
    result = export.convert(source, dest, group_size=8)
    assert result["quantized_projections"] == 6
    index = json.loads((dest / "model.safetensors.index.json").read_text())["weight_map"]
    assert len(index) == 20
    for name in tensors:
        if name in index:
            with safe_open(dest / index[name], framework="pt") as handle:
                torch.testing.assert_close(handle.get_tensor(name), tensors[name].half())
    assert export.digest(source / "model.safetensors") == before
    hashes = {p.name: export.digest(p) for p in dest.glob("*.safetensors")}
    assert export.convert(source, dest, group_size=8, resume=True) == result
    assert hashes == {p.name: export.digest(p) for p in dest.glob("*.safetensors")}
    assert json.loads((dest / "build-journal.json").read_text())["complete"]
    report = audit_module.audit(dest, sample_layers=[0], sample_experts=[0, 1])
    assert report["header_inventory_valid"]
    assert report["tensor_counts"]["routed_experts"] == 18
    assert len(report["samples"]) == 6
    assert not report["full_model_accuracy_validated"]
    with pytest.raises(ValueError, match="empty destination"):
        export.convert(source, dest, group_size=8)
    with pytest.raises(ValueError, match="changed"):
        export.convert(source, dest, group_size=4, resume=True)


def test_resume_detects_corruption(checkpoint, tmp_path):
    pytest.importorskip("msmodelslim")
    source, _ = checkpoint
    dest = tmp_path / "output"
    export.convert(source, dest, group_size=8)
    shard = dest / "experts-000.safetensors"
    with shard.open("r+b") as handle:
        handle.seek(-1, 2)
        damaged = bytes([handle.read(1)[0] ^ 1])
        handle.seek(-1, 2)
        handle.write(damaged)
    with pytest.raises(ValueError, match="corrupt"):
        export.convert(source, dest, group_size=8, resume=True)


def test_export_rejects_source_overlap(checkpoint):
    pytest.importorskip("msmodelslim")
    source, _ = checkpoint
    with pytest.raises(ValueError, match="separate"):
        export.convert(source, source, group_size=8)


@pytest.mark.parametrize("weight", [torch.ones(2, 8, dtype=torch.int8), torch.full((2, 8), float("nan"))])
def test_export_rejects_nonfloating_or_nonfinite(weight):
    pytest.importorskip("msmodelslim")
    with pytest.raises(ValueError):
        export.quantize_projection(weight, 8)
