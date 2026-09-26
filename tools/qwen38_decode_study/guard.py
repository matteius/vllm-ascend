# SPDX-License-Identifier: Apache-2.0
"""Refuse a candidate launch while another vLLM server/worker exists."""

from pathlib import Path


def active_servers(proc: Path = Path("/proc")) -> list[int]:
    pids = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = entry.joinpath("cmdline").read_bytes().split(b"\0")
            comm = entry.joinpath("comm").read_text().strip()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if comm.startswith("VLLM::") or (
            b"serve" in args and any(Path(arg.decode(errors="replace")).name == "vllm" for arg in args if arg)
        ):
            pids.append(int(entry.name))
    return sorted(pids)


if __name__ == "__main__":
    existing = active_servers()
    if existing:
        raise SystemExit(f"Refusing candidate launch: vLLM processes still exist: {existing}. No process was stopped.")
