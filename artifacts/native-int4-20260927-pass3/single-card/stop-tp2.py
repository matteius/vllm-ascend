"""Stop only a verified single-card W4 experiment and its captured workers."""

import sys
from contextlib import suppress
from pathlib import Path

import psutil

RESULTS = Path("/srv/ai/src/native-int4-w4a8.KiuhBN/pass3/single-card")


def main():
    label = sys.argv[1]
    api = psutil.Process(int((RESULTS / f"server-{label}.pid").read_text()))
    command = api.cmdline()
    assert "/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i" in command
    assert command[command.index("--port") + 1] in ("8002", "8003")
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert api.environ()["ASCEND_RT_VISIBLE_DEVICES"] in ("0,1", "2,3")
    children = api.children(recursive=True)
    api.terminate()
    psutil.wait_procs([api], timeout=10)
    for child in children:
        if child.is_running():
            with suppress(psutil.NoSuchProcess):
                child.terminate()
    _, alive = psutil.wait_procs([api, *children], timeout=10)
    for child in alive:
        if child.status() != psutil.STATUS_ZOMBIE:
            child.kill()
    _, alive = psutil.wait_procs(alive, timeout=5)
    assert all(child.status() == psutil.STATUS_ZOMBIE for child in alive)
    print("SINGLE_CARD_EXPERIMENT_STOPPED", api.pid)


if __name__ == "__main__":
    main()
