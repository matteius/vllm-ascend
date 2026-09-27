"""Recreate the fixed sample from the author's archive without extracting files."""

import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("mmlu-manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert hashlib.sha256(args.archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
    with tarfile.open(args.archive) as archive, args.output.open("x") as output:
        members = {Path(m.name).name: m for m in archive.getmembers() if m.isfile() and "/test/" in m.name}
        for sample in manifest["samples"]:
            subject, row_number = sample["id"].rsplit(":", 1)
            with archive.extractfile(members[f"{subject}_test.csv"]) as content:
                rows = list(csv.reader(io.TextIOWrapper(content, encoding="utf-8")))
            row = rows[int(row_number)]
            prompt = (
                row[0]
                + "\n"
                + "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", row[1:5]))
                + "\nAnswer:"
            )
            assert hashlib.sha256(prompt.encode()).hexdigest() == sample["prompt_sha256"]
            output.write(
                json.dumps({"id": sample["id"], "subject": subject, "prompt": prompt, "answer": row[5]}) + "\n"
            )


if __name__ == "__main__":
    main()
