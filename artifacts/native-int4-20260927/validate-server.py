"""Paired real-weight validation of one explicitly identified task server."""

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil
import regex as re

ROOT = Path("/srv/ai/src/native-int4-w4a8.KiuhBN")
EVIDENCE = Path("/srv/ai/src/qwen38-w4-hardware-20260927")
STUDY = Path("/srv/ai/src/qwen38-decode-study-20260926")
PYTHON = "/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python"


def main():
    """Verify process identity, pin workers, then run quality gates before timing."""
    label = sys.argv[1]
    assert label in ("baseline", "native", "affine")
    api = psutil.Process(int((ROOT / f"server-{label}.pid").read_text()))
    checkpoint = "/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i"
    launch_deadline = time.monotonic() + 30
    while checkpoint not in api.cmdline():
        if not api.is_running() or api.status() == psutil.STATUS_ZOMBIE or time.monotonic() > launch_deadline:
            raise RuntimeError("task process did not exec the expected model server")
        time.sleep(1)
    deadline = time.monotonic() + 2400
    while True:
        if not api.is_running() or api.status() == psutil.STATUS_ZOMBIE:
            raise RuntimeError("task API exited")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=5) as response:
                if response.status == 200:
                    break
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError("candidate not ready")
            time.sleep(5)
    workers = {}
    for process in api.children(recursive=True):
        match = re.match(r"VLLM::Worker_TP(\d+)_EP\d+", " ".join(process.cmdline()))
        if match:
            workers[int(match[1])] = process
    assert sorted(workers) == [0, 1, 2, 3]
    engine = workers[0].ppid()
    assert all(w.ppid() == engine for w in workers.values())
    with (ROOT / f"{label}-affinity.json").open("x") as output:
        subprocess.run(
            [
                PYTHON,
                str(EVIDENCE / "qwen38-w4-kilo-affinity-r8.py"),
                "--api-pid",
                str(api.pid),
                "--engine-pid",
                str(engine),
                "--workers",
                *[str(workers[r].pid) for r in range(4)],
            ],
            stdout=output,
            check=True,
        )
    with (ROOT / f"{label}-smoke.jsonl").open("x") as output:
        subprocess.run(
            [PYTHON, str(EVIDENCE / "qwen38-w4-http-smoke.py"), str(api.pid), "http://127.0.0.1:8001"],
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("REAL_WEIGHT_SMOKE_PASS", label, flush=True)
    subprocess.run([PYTHON, str(ROOT / "dual-count.py"), str(ROOT / f"{label}-dual-count.jsonl")], check=True)
    subprocess.run(
        [
            "bash",
            str(ROOT / "run-python.sh"),
            "-m",
            "tools.qwen4exp.evaluate_w4a8_http",
            "--base-url",
            "http://127.0.0.1:8001",
            "--dataset",
            str(ROOT / "native-int4-mmlu-228.jsonl"),
            "--output",
            str(ROOT / f"{label}-mmlu.jsonl"),
            "--label",
            label,
        ],
        check=True,
    )
    if label == "affine":
        subprocess.run(
            [
                "bash",
                str(ROOT / "run-python-affine-r2.sh"),
                "-m",
                "tools.qwen4exp.evaluate_w4a8_http",
                "--base-url",
                "http://127.0.0.1:8001",
                "--dataset",
                str(ROOT / "native-int4-mmlu-heldout-228.jsonl"),
                "--output",
                str(ROOT / "affine-heldout.jsonl"),
                "--label",
                "affine-heldout",
            ],
            check=True,
        )
    for context, extra in [("short", []), ("long", ["--prompt-prefix-file", str(STUDY / "long-prefix.txt")])]:
        subprocess.run(
            [
                PYTHON,
                "-m",
                "tools.qwen38_decode_study.benchmark",
                "--base-url",
                "http://127.0.0.1:8001",
                "--output",
                str(ROOT / f"{label}-{context}.jsonl"),
                "--lengths",
                "512",
                "--repeats",
                "3",
                *extra,
            ],
            cwd=STUDY / "study-tools",
            check=True,
        )
    print(json.dumps({"event": "PAIRED_MODEL_GATE_COMPLETE", "label": label, "api_pid": api.pid}), flush=True)


if __name__ == "__main__":
    main()
