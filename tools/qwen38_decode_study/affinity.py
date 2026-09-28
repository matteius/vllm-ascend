# SPDX-License-Identifier: Apache-2.0
"""Plan physical-core pools and bind only the calling worker's threads."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from contextlib import suppress
from pathlib import Path


def cpu_topology(root: Path = Path("/sys/devices/system/cpu")) -> dict[int, tuple[int, int, int]]:
    """Read (package, core, last-level-cache) identity without touching affinity."""
    result = {}
    for directory in root.glob("cpu[0-9]*"):
        cpu = int(directory.name[3:])
        topology = directory / "topology"
        if not topology.is_dir():
            continue
        package = int((topology / "physical_package_id").read_text())
        core = int((topology / "core_id").read_text())
        caches = []
        for cache in (directory / "cache").glob("index*"):
            if (cache / "id").exists():
                caches.append((int((cache / "level").read_text()), int((cache / "id").read_text())))
        result[cpu] = (package, core, max(caches)[1] if caches else core)
    return result


def validate_groups(groups, allowed: set[int], topology: dict[int, tuple[int, int, int]]) -> None:
    if not groups or any(not group for group in groups):
        raise ValueError("Every rank needs a non-empty CPU group")
    used_cpus: set[int] = set()
    used_cores: set[tuple[int, int]] = set()
    for group in groups:
        cores = set()
        for cpu in group:
            if type(cpu) is not int or cpu not in allowed or cpu not in topology:
                raise ValueError(f"CPU {cpu!r} is unavailable in this process's cpuset/topology")
            if cpu in used_cpus:
                raise ValueError(f"CPU {cpu} appears more than once")
            used_cpus.add(cpu)
            core = topology[cpu][:2]
            if core in used_cores:
                raise ValueError(f"Physical core {core} is shared by different ranks")
            cores.add(core)
        used_cores.update(cores)


def plan_groups(topology, allowed: set[int], ranks: int, cores_per_rank: int) -> list[list[int]]:
    if ranks < 1 or cores_per_rank < 1:
        raise ValueError("ranks and cores_per_rank must be positive")
    packages = {topology[cpu][0] for cpu in allowed if cpu in topology}
    if len(packages) != 1:
        raise ValueError("Automatic planning requires one socket; supply a reviewed plan for other hosts")
    physical = defaultdict(list)
    for cpu in sorted(allowed):
        if cpu in topology:
            physical[topology[cpu][:2]].append(cpu)
    representatives = sorted((min(cpus) for cpus in physical.values()), key=lambda cpu: topology[cpu][::2] + (cpu,))
    if len(representatives) < ranks * cores_per_rank:
        raise ValueError("Not enough physical cores for disjoint worker pools")
    # Divide the socket evenly, keeping each worker near its local LLCs and
    # leaving the remainder of each slice for scheduler/API/OS work.
    groups = []
    for rank in range(ranks):
        start = rank * len(representatives) // ranks
        groups.append(representatives[start : start + cores_per_rank])
    validate_groups(groups, allowed, topology)
    return groups


def bind_worker(rank: int, groups) -> None:
    """Opt-in startup binding, with rollback if an existing thread cannot bind.

    Never accepts a PID: this cannot be used to rebind an unrelated live server.
    Threads created afterward inherit their creating thread's affinity.
    """
    if not 0 <= rank < len(groups):
        raise ValueError(f"Worker rank {rank} has no CPU pool")
    validate_groups(groups, set(os.sched_getaffinity(0)), cpu_topology())
    previous = {}
    try:
        for task in Path("/proc/self/task").iterdir():
            tid = int(task.name)
            try:
                previous[tid] = os.sched_getaffinity(tid)
                os.sched_setaffinity(tid, set(groups[rank]))
            except ProcessLookupError:
                previous.pop(tid, None)
    except OSError:
        for tid, mask in previous.items():
            with suppress(ProcessLookupError):
                os.sched_setaffinity(tid, mask)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--cores-per-rank", type=int, default=6)
    args = parser.parse_args()
    topology = cpu_topology()
    allowed = set(os.sched_getaffinity(0))
    groups = plan_groups(topology, allowed, args.ranks, args.cores_per_rank)
    print(json.dumps({"groups": groups, "topology": topology, "allowed": sorted(allowed)}, indent=2))


if __name__ == "__main__":
    main()
