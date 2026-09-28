"""Verify the retained service and package without exposing environment secrets."""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import psutil
import regex as re

ROOT = Path("/srv/ai/src/native-int4-w4a8.KiuhBN")
RESULTS = ROOT / "pass3"


def main():
    label = sys.argv[1]
    assert re.fullmatch(r"[a-z0-9-]+", label)
    api = psutil.Process(int((RESULTS / f"server-{label}.pid").read_text()))
    command = api.cmdline()
    assert "/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i" in command
    assert command[command.index("--port") + 1] == "8001"
    config = json.loads(command[command.index("--hf-overrides") + 1])
    policy = config["text_config"]["ascend_expert_quantization"]
    assert policy["backend"] == "cube_310_int4_a8"
    assert policy["activation_quantization"] == "int8_per_group"
    package = ROOT / f"ops-pass3-{label}"
    vendor = package / "vendors/native_int4_w4a8_transformer"
    assert api.environ()["ASCEND_CUSTOM_OPP_PATH"].split(":")[0] == str(vendor)
    provenance = json.loads((RESULTS / "provenance.json").read_text())
    for path, expected in provenance["variants"][label]["operator_sha256"].items():
        assert hashlib.sha256((package / path).read_bytes()).hexdigest() == expected
    with urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=10) as response:
        assert response.status == 200
    with urllib.request.urlopen("http://127.0.0.1:8001/v1/models", timeout=10) as response:
        models = json.load(response)
    assert any(model["id"] == "qwen38-w4-experimental" for model in models["data"])
    workers = []
    for child in api.children(recursive=True):
        if any("VLLM::Worker_TP" in arg for arg in child.cmdline()):
            workers.append({"pid": child.pid, "cpu_affinity": child.cpu_affinity()})
    assert len(workers) == 4
    result = {
        "api_pid": api.pid,
        "health": "passed",
        "model": "qwen38-w4-experimental",
        "port": 8001,
        "backend": policy["backend"],
        "activation_quantization": policy["activation_quantization"],
        "operator_package": str(package),
        "operator_hashes_match_tested_package": True,
        "workers": workers,
        "left_running_per_user_request": True,
    }
    (RESULTS / "service-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
