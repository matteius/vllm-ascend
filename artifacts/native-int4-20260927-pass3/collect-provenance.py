"""Record hashes for immutable native-INT4 packages on the NPU host."""

import hashlib
import json
from pathlib import Path

ROOT = Path("/srv/ai/src/native-int4-w4a8.KiuhBN")
RESULTS = ROOT / "pass3"
VARIANTS = ("live-rows", "paired-mmad", "broadcast", "route-cache")
OPERATOR = "qwen_w4_a8_int4_matmul_v310"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    variants = {}
    for label in VARIANTS:
        package = ROOT / f"ops-pass3-{label}"
        binaries = list(package.glob(f"vendors/*/op_impl/ai_core/tbe/kernel/ascend310p/{OPERATOR}/*.o"))
        binaries += list(package.glob("vendors/*/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/*.so"))
        assert binaries, label
        source = RESULTS / f"source-{label}"
        variants[label] = {
            "operator_sha256": {str(path.relative_to(package)): digest(path) for path in sorted(binaries)},
            "source_sha256": {path.name: digest(path) for path in sorted(source.glob("*")) if path.is_file()},
        }
    runtime = ROOT / "runtime"
    final_runtime = ROOT / "runtime-qsa-combined"
    sources = (
        "vllm_ascend/models/qwen4_exp/w4_moe.py",
        "csrc/torch_binding_meta.cpp",
        f"csrc/gmm/{OPERATOR}/qwen_w4_a8_int4_matmul_310_torch_adpt.h",
    )
    result = {
        "base_commit": "7a04af2e9",
        "variants": variants,
        "runtime_source_sha256": {name: digest(runtime / name) for name in sources},
        "final_runtime_source_sha256": {
            name: digest(final_runtime / name)
            for name in (
                "vllm_ascend/models/qwen4_exp/model.py",
                "vllm_ascend/models/qwen4_exp/mtp.py",
                "vllm_ascend/models/qwen4_exp/w4_moe.py",
            )
        },
        "binding_sha256": {path.name: digest(path) for path in runtime.glob("vllm_ascend/vllm_ascend_C*.so")},
        "host_tiling_source_sha256": digest(ROOT / "csrc/gmm" / OPERATOR / "op_host" / f"{OPERATOR}_tiling.cpp"),
    }
    (RESULTS / "provenance.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"variants": list(variants), "binding": list(result["binding_sha256"])}, indent=2))


if __name__ == "__main__":
    main()
