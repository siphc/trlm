"""Files the cloud instance needs for studies A and B (decision 12), with sha256 hashes so the instance can prove it
has exactly the laptop's data. The test split is never copied.

    .venv/bin/python scripts/aws/manifest.py make     # on the laptop: writes scripts/aws/manifest.json
    .venv/bin/python scripts/aws/manifest.py check    # on the instance: verifies every listed file
    .venv/bin/python scripts/aws/manifest.py list     # paths only (for rsync --files-from)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "scripts", "aws", "manifest.json")
FILES = ["data/tokens/train.bin", "data/tokens/dev.bin", "data/eval/dev_fast.npy", "data/eval/dev_full.npy",
         "data/eval/dev_full_window_idx.npy", "data/prep_meta.json", "data/tokenizer.json",
         "data/splits/dev_indices.npy", "data/baselines/bigram_logprobs.npy", "data/fixed_votes/tables.npz",
         "analysis/out/candidates/qk/N_1.npy"] + \
        [f"models/final/attn_s{k}/{f}" for k in range(5) for f in ("final.pt", "result.json")]
FORBIDDEN = ("test.bin", "test_full")                  # the test split stays on the laptop (AGENTS.md section 9)


def sha(path):
    h = hashlib.sha256()
    with open(os.path.join(REPO, path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main(cmd):
    assert not any(x in f for f in FILES for x in FORBIDDEN)
    if cmd == "list":
        print("\n".join(FILES))
    elif cmd == "make":
        with open(OUT, "w") as f:
            json.dump({p: sha(p) for p in FILES}, f, indent=1)
        print(f"wrote {OUT}")
    elif cmd == "check":
        want = json.load(open(OUT))
        bad = [p for p, h in want.items() if not os.path.exists(os.path.join(REPO, p)) or sha(p) != h]
        print("manifest OK" if not bad else f"MISMATCH: {bad}")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "check")
