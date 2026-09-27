"""Stop only the explicitly identified experimental API and its captured tree."""

import argparse
from contextlib import suppress
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pid_file", type=Path)
    args = parser.parse_args()
    api = psutil.Process(int(args.pid_file.read_text()))
    argv = api.cmdline()
    assert "/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i" in argv
    assert argv[argv.index("--port") + 1] == "8001"
    captured = api.children(recursive=True)
    api.terminate()
    psutil.wait_procs([api], timeout=10)
    # psutil Process objects retain creation time and reject reused PIDs.
    for process in captured:
        if process.is_running():
            with suppress(psutil.NoSuchProcess):
                process.terminate()
    _, alive = psutil.wait_procs([api, *captured], timeout=10)
    for process in alive:
        if process.status() != psutil.STATUS_ZOMBIE:
            process.kill()
    _, alive = psutil.wait_procs(alive, timeout=5)
    assert all(process.status() == psutil.STATUS_ZOMBIE for process in alive)
    print("EXPERIMENT_STOPPED", api.pid)


if __name__ == "__main__":
    main()
