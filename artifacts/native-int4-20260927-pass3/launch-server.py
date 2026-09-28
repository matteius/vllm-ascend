"""Launch an explicit native variant without replacing previous evidence."""

import subprocess
import sys
from pathlib import Path

import regex as re

ROOT = Path("/srv/ai/src/native-int4-w4a8.KiuhBN")
RESULTS = ROOT / "pass3"


def main():
    assert len(sys.argv) in (2, 3, 4)
    label = sys.argv[1]
    runtime = Path(sys.argv[2]) if len(sys.argv) >= 3 else ROOT / "runtime"
    max_num_seqs = sys.argv[3] if len(sys.argv) == 4 else "2"
    assert re.fullmatch(r"[a-z0-9-]+", label)
    assert max_num_seqs.isdecimal() and 1 <= int(max_num_seqs) <= 4
    assert (ROOT / f"ops-pass3-{label}").is_dir()
    assert runtime.is_absolute() and (runtime / "vllm_ascend").is_dir()
    # Exclusive files prevent accidental replacement of a running experiment.
    with (RESULTS / f"server-{label}.pid").open("x") as pid_file:
        with (RESULTS / f"server-{label}.log").open("xb") as log:
            process = subprocess.Popen(
                ["bash", str(RESULTS / "serve.sh"), label, str(runtime), max_num_seqs],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_file.write(str(process.pid) + "\n")
    print(process.pid)


if __name__ == "__main__":
    main()
