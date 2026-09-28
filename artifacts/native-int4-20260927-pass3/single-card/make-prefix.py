"""Build a reproducible near-limit text prefix, measured by the model tokenizer."""

import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

MODEL = Path("/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i")
SOURCE = Path("/srv/ai/src/qwen38-decode-study-20260926/long-prefix.txt")
RESULTS = Path("/srv/ai/src/native-int4-w4a8.KiuhBN/pass3/single-card")


def main():
    target = int(sys.argv[1])
    assert 1000 <= target <= 260000
    output = RESULTS / f"prefix-{target}.txt"
    assert not output.exists()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    base = SOURCE.read_text()
    source_tokens = len(tokenizer.encode(base, add_special_tokens=False))
    full = (base + "\n\n") * (target // source_tokens + 2)
    lo, hi = 0, len(full)
    while lo < hi:
        middle = (lo + hi + 1) // 2
        tokens = len(tokenizer.encode(full[:middle], add_special_tokens=False))
        if tokens <= target:
            lo = middle
        else:
            hi = middle - 1
    prefix = full[:lo]
    actual = len(tokenizer.encode(prefix, add_special_tokens=False))
    assert actual <= target and target - actual < 16
    output.write_text(prefix)
    record = {
        "target_prefix_tokens": target,
        "actual_prefix_tokens": actual,
        "utf8_bytes": len(prefix.encode()),
        "sha256": hashlib.sha256(prefix.encode()).hexdigest(),
        "source_sha256": hashlib.sha256(base.encode()).hexdigest(),
    }
    (RESULTS / f"prefix-{target}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
