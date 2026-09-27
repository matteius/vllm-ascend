"""Launch an explicit native variant without replacing previous evidence."""

import subprocess
import sys
from pathlib import Path

import regex as re

ROOT = Path("/srv/ai/src/native-int4-w4a8.KiuhBN")
RESULTS = ROOT / "pass2"


def main():
    label = sys.argv[1]
    assert re.fullmatch(r"[a-z0-9-]+", label)
    assert (ROOT / f"ops-pass2-{label}").is_dir()
    # Exclusive files prevent accidental replacement of a running experiment.
    with (RESULTS / f"server-{label}.pid").open("x") as pid_file:
        with (RESULTS / f"server-{label}.log").open("xb") as log:
            process = subprocess.Popen(
                ["bash", str(RESULTS / "serve.sh"), label],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_file.write(str(process.pid) + "\n")
    print(process.pid)


if __name__ == "__main__":
    main()
