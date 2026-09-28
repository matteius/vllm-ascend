"""Launch one isolated physical dual-NPU card for a bounded context trial."""

import re
import subprocess
import sys
from pathlib import Path

RESULTS = Path("/srv/ai/src/native-int4-w4a8.KiuhBN/pass3/single-card")


def main():
    label, max_len, mode, *options = sys.argv[1:]
    assert re.fullmatch(r"[a-z0-9-]+", label)
    assert max_len.isdecimal() and 1 <= int(max_len) <= 262144
    assert mode in ("mtp1", "mtp2", "mtp3", "plain", "mtp2-noprefix", "plain-noprefix")
    assert len(options) in (0, 2, 4, 5)
    memory = options[:2] if options else ["0.94", "0.80"]
    devices, port = options[2:] if len(options) == 4 else ("0,1", "8002")
    runtime = None
    if len(options) == 5:
        devices, port, runtime = options[2:]
    if memory:
        assert 0.8 <= float(memory[0]) < 1.0
        assert 0.8 <= float(memory[1]) <= 1.0
    assert re.fullmatch(r"[0-9]+,[0-9]+", devices)
    assert port.isdecimal() and 1024 <= int(port) <= 65535
    if runtime is not None:
        runtime_path = Path(runtime)
        assert runtime_path.is_absolute() and (runtime_path / "vllm_ascend").is_dir()
    command = ["bash", str(RESULTS / "serve-tp2.sh"), max_len, mode, *memory, devices, port, label]
    if runtime is not None:
        command.append(runtime)
    with (RESULTS / f"server-{label}.pid").open("x") as pid_file:
        with (RESULTS / f"server-{label}.log").open("xb") as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_file.write(f"{process.pid}\n")
    print(process.pid)


if __name__ == "__main__":
    main()
