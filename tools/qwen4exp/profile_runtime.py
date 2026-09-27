# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in NPU timeline + multi-process Python sampling, or cProfile call graphs.

Capture is a dry run unless --execute is supplied. This tool never starts,
stops, reconfigures, or restarts vLLM. Run CPU sampling and NPU tracing separately
for clean attribution. Profiled requests are not production throughput numbers.
"""

import argparse
import contextlib
import hashlib
import importlib
import json
import pstats
import shutil
import signal
import subprocess
import urllib.request
from pathlib import Path


def profiler_config(trace_dir: str, phase: str, steps: int = 4, delay: int | None = None) -> dict:
    if phase not in ("cold-prefill", "warm-prefill", "decode") or steps <= 0 or (delay is not None and delay < 0):
        raise ValueError("invalid profiling phase, steps or delay")
    return {
        "profiler": "torch",
        "torch_profiler_dir": trace_dir,
        "torch_profiler_with_stack": False,
        "torch_profiler_with_memory": False,
        "ignore_frontend": True,
        # Decode delay is only appropriate after priming the requested prefix.
        "delay_iterations": (4 if phase == "decode" else 0) if delay is None else delay,
        "max_iterations": steps,
    }


def profile_rpc(base_url: str, action: str) -> None:
    if action not in ("start_profile", "stop_profile"):
        raise ValueError("unsupported profiler action")
    request = urllib.request.Request(base_url.rstrip("/") + "/" + action, data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError(f"{action} returned HTTP {response.status}")


@contextlib.contextmanager
def npu_capture(base_url: str):
    profile_rpc(base_url, "start_profile")
    try:
        yield
    finally:
        # Stop on client failure too; never leave the worker recorder running.
        profile_rpc(base_url, "stop_profile")


def pyspy_command(pid: int, duration: int, output: Path) -> list[str]:
    if pid <= 0 or duration <= 0:
        raise ValueError("pid and duration must be positive")
    return [
        "py-spy",
        "record",
        "--pid",
        str(pid),
        "--subprocesses",
        "--threads",
        "--duration",
        str(duration),
        "--format",
        "speedscope",
        "--output",
        str(output),
    ]


def call_graph(stats: pstats.Stats, limit: int = 60) -> str:
    """DOT call graph: node self/cumulative wall time includes NPU waits."""
    if limit <= 0:
        raise ValueError("call graph node limit must be positive")
    ranked = sorted(stats.stats, key=lambda key: stats.stats[key][3], reverse=True)[:limit]
    ids = {key: index for index, key in enumerate(ranked)}
    lines = ["digraph python_calls {", "rankdir=LR;", "node [shape=box];"]
    for key in ranked:
        primitive, calls, self_time, cumulative, callers = stats.stats[key]
        label = (
            f"{Path(key[0]).name}:{key[1]} {key[2]}\ncalls={calls} self={self_time:.6f}s cumulative={cumulative:.6f}s"
        )
        lines.append(f"n{ids[key]} [label={json.dumps(label)}];")
        for caller in callers:
            if caller in ids:
                lines.append(f"n{ids[caller]} -> n{ids[key]};")
    return "\n".join([*lines, "}"]) + "\n"


def export_cprofile(path: Path, output: Path, limit: int = 60) -> None:
    # pstats uses pickle: only read trusted, locally generated profiles.
    stats = pstats.Stats(str(path))
    with output.open("x") as file:
        file.write(call_graph(stats, limit))


def analyse_npu(trace_root: Path, processes: int = 1) -> list[Path]:
    """Parse daemon-worker captures offline; no model load or profiling RPC.

    torch_npu cannot parse in vLLM daemon workers. Its standalone parser writes
    ASCEND_PROFILER_OUTPUT next to each raw FRAMEWORK capture. Preserve earlier
    parsed evidence instead of asking it to overwrite the same output folder.
    """
    if processes <= 0:
        raise ValueError("parser process count must be positive")
    if not trace_root.is_dir():
        raise ValueError("trace root must be an existing directory")
    captures = [trace_root] if trace_root.name.endswith("_ascend_pt") else sorted(trace_root.glob("*_ascend_pt"))
    if not captures or any(not (path / "FRAMEWORK").is_dir() for path in captures):
        raise ValueError("no complete raw Ascend framework captures found")
    outputs = [path / "ASCEND_PROFILER_OUTPUT" for path in captures]
    if any(path.exists() for path in outputs):
        raise FileExistsError("parsed output already exists; preserve it and select an unparsed capture")
    # Optional hardware dependency: config/capture/cProfile remain stdlib-only.
    profiler = importlib.import_module("torch_npu.profiler.profiler")
    profiler.analyse(str(trace_root), max_process_number=processes)
    details = [path / "kernel_details.csv" for path in outputs]
    if any(not path.is_file() for path in details):
        raise RuntimeError("NPU parser did not produce all kernel_details.csv files; inspect parser diagnostics")
    return details


def capture(args) -> None:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("provide the benchmark client command after --")
    config = profiler_config(args.trace_dir, args.phase, args.steps, args.delay)
    print(
        json.dumps(
            {"dry_run": not args.execute, "profiler_config": config, "requires_server_started_with_this_config": True},
            indent=2,
        )
    )
    if not args.execute:
        return
    if args.output.exists():
        raise FileExistsError("choose a new output directory; earlier evidence is preserved")
    if args.cpu_pid and shutil.which("py-spy") is None:
        raise RuntimeError("py-spy is not installed; no dependencies are installed automatically")
    args.output.mkdir(parents=True)
    sampler = None
    sampler_log = None
    report = {
        "phase": args.phase,
        "profiled_not_benchmark_baseline": True,
        "profiler_config": config,
        "npu_trace_requested": not args.cpu_only,
        "client_command_sha256": hashlib.sha256(json.dumps(command).encode()).hexdigest(),
        "cpu_pid": args.cpu_pid,
        "success": False,
    }
    try:
        if args.cpu_pid:
            sampler_log = (args.output / "py-spy.log").open("x")
            sampler = subprocess.Popen(
                pyspy_command(args.cpu_pid, args.seconds, args.output / "python.speedscope.json"),
                stdout=sampler_log,
                stderr=subprocess.STDOUT,
            )
        context = contextlib.nullcontext() if args.cpu_only else npu_capture(args.base_url)
        with context:
            subprocess.run(command, check=True, timeout=args.timeout)
        report["success"] = True
    finally:
        if sampler is not None:
            if sampler.poll() is None:
                sampler.send_signal(signal.SIGINT)
            try:
                sampler.wait(timeout=15)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait(timeout=15)
            report["py_spy_exit_code"] = sampler.returncode
            report["cpu_samples_present"] = (args.output / "python.speedscope.json").exists()
            if not report["cpu_samples_present"]:
                report["success"] = False
        if sampler_log is not None:
            sampler_log.close()
        with (args.output / "manifest.json").open("x") as file:
            json.dump(report, file, indent=2)
    if not report["success"]:
        raise RuntimeError("capture incomplete; inspect manifest.json and py-spy.log")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    config = sub.add_parser("config", help="print --profiler-config for the next server launch")
    config.add_argument("--trace-dir", required=True)
    config.add_argument("--phase", choices=("cold-prefill", "warm-prefill", "decode"), required=True)
    config.add_argument("--steps", type=int, default=4)
    config.add_argument("--delay", type=int)
    cap = sub.add_parser("capture", help="profile one explicit benchmark command")
    cap.add_argument("--base-url", default="http://127.0.0.1:8001")
    cap.add_argument("--phase", choices=("cold-prefill", "warm-prefill", "decode"), required=True)
    cap.add_argument("--trace-dir", required=True)
    cap.add_argument("--steps", type=int, default=4)
    cap.add_argument("--delay", type=int)
    cap.add_argument("--output", type=Path, required=True)
    cap.add_argument("--cpu-pid", type=int, help="API server PID; py-spy includes its worker descendants")
    cap.add_argument("--cpu-only", action="store_true", help="sample Python without NPU profiling RPCs")
    cap.add_argument("--seconds", type=int, default=60)
    cap.add_argument("--timeout", type=int, default=900)
    cap.add_argument("--execute", action="store_true")
    cap.add_argument("command", nargs=argparse.REMAINDER)
    analyse = sub.add_parser("analyse", help="parse raw daemon-worker NPU captures offline (requires torch_npu)")
    analyse.add_argument("trace_root", type=Path)
    analyse.add_argument("--processes", type=int, default=1, help="limit host parser concurrency; default 1")
    graph = sub.add_parser("cprofile-dot", help="convert a trusted cProfile file to Graphviz DOT")
    graph.add_argument("profile", type=Path)
    graph.add_argument("--output", type=Path, required=True)
    graph.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()
    if args.action == "config":
        print(json.dumps(profiler_config(args.trace_dir, args.phase, args.steps, args.delay)))
    elif args.action == "capture":
        if args.cpu_only and not args.cpu_pid:
            parser.error("--cpu-only requires --cpu-pid")
        if args.seconds <= 0 or args.timeout <= 0:
            parser.error("duration and timeout must be positive")
        capture(args)
    elif args.action == "analyse":
        print(json.dumps({"kernel_details": [str(path) for path in analyse_npu(args.trace_root, args.processes)]}))
    else:
        export_cprofile(args.profile, args.output, args.limit)


if __name__ == "__main__":
    main()
