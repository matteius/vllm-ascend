"""Pin the verified dual-NPU workers to separate CPU groups."""

import json
import os
import re
import sys
from contextlib import suppress
from pathlib import Path

import psutil

RESULTS = Path("/srv/ai/src/native-int4-w4a8.KiuhBN/pass3/single-card")
GROUPS_BY_DEVICES = {
    "0,1": (set(range(0, 6)), set(range(8, 14))),
    "2,3": (set(range(16, 22)), set(range(24, 30))),
}


def main():
    label = sys.argv[1]
    api = psutil.Process(int((RESULTS / f"server-{label}.pid").read_text()))
    command = api.cmdline()
    port = command[command.index("--port") + 1]
    assert port in ("8002", "8003")
    visible_devices = api.environ()["ASCEND_RT_VISIBLE_DEVICES"]
    groups = GROUPS_BY_DEVICES[visible_devices]
    workers = {}
    for child in api.children(recursive=True):
        match = re.match(r"VLLM::Worker_TP(\d+)_EP\d+", " ".join(child.cmdline()))
        if match:
            workers[int(match[1])] = child
    assert sorted(workers) == [0, 1]
    previous = {}
    try:
        for rank, group in enumerate(groups):
            for task in (Path(f"/proc/{workers[rank].pid}/task")).iterdir():
                tid = int(task.name)
                with suppress(ProcessLookupError):
                    previous[tid] = os.sched_getaffinity(tid)
                    assert group <= previous[tid]
                    os.sched_setaffinity(tid, group)
        result = {
            "api_pid": api.pid,
            "card_visible_devices": visible_devices,
            "port": int(port),
            "workers": {rank: {"pid": workers[rank].pid, "cpu_group": sorted(groups[rank])} for rank in workers},
        }
        (RESULTS / f"{label}-affinity.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    except BaseException:
        for tid, mask in previous.items():
            with suppress(ProcessLookupError):
                os.sched_setaffinity(tid, mask)
        raise


if __name__ == "__main__":
    main()
