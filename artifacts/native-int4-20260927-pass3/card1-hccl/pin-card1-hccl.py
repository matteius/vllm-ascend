import json
import os
import sys
import time
from contextlib import suppress
from pathlib import Path

import psutil
import regex as re

root = Path("/srv/ai/src/native-int4-w4a8.KiuhBN/pass3/card1-hccl")
label, mode = sys.argv[1:]
api = psutil.Process(int((root / f"{label}.pid").read_text()))
children = api.children(recursive=True)
engine = next(p for p in children if "EngineCore" in " ".join(p.cmdline()))
workers = {}
for p in children:
    m = re.search(r"Worker_TP(\d+)", " ".join(p.cmdline()))
    if m:
        workers[int(m.group(1))] = p
assert sorted(workers) == [0, 1], workers
groups = {workers[0].pid: set(range(16, 22)), workers[1].pid: set(range(24, 30))}
if mode == "isolated":
    groups[api.pid] = {22, 23}
    groups[engine.pid] = {30, 31}
elif mode != "workers":
    raise ValueError(mode)
result = {
    "label": label,
    "mode": mode,
    "api_pid": api.pid,
    "engine_pid": engine.pid,
    "workers": {str(k): v.pid for k, v in workers.items()},
    "groups": {str(k): sorted(v) for k, v in groups.items()},
    "time": time.time(),
}
for pid, cpus in groups.items():
    for task in Path(f"/proc/{pid}/task").iterdir():
        with suppress(ProcessLookupError):
            os.sched_setaffinity(int(task.name), cpus)
(root / f"{label}-affinity-{mode}.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
