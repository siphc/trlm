"""Held-out windows for claim tests: 200 dev windows NOT in dev_full (so not in dev_fast either).
READ-ONLY for replacement agents. Usage: python -m harness.build_claim_windows"""
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 200
SEED = 7


def main() -> None:
    d = os.path.join(REPO, "data")
    s = np.memmap(os.path.join(d, "tokens", "dev.bin"), dtype=np.uint16, mode="r")
    n = (len(s) - 1) // 256
    used = set(np.load(os.path.join(d, "eval", "dev_full_window_idx.npy")).tolist())
    free = np.array([i for i in range(n) if i not in used])
    pick = np.random.default_rng(SEED).choice(free, size=N, replace=False)
    w = np.stack([np.asarray(s[i * 256 : i * 256 + 257]) for i in pick]).astype(np.uint16)
    out = os.path.join(d, "eval", "claims.npy")
    assert not os.path.exists(out), "claims.npy already exists; evaluation sets are immutable"
    np.save(out, w)
    np.save(os.path.join(d, "eval", "claims_window_idx.npy"), pick)
    print(f"claims: {w.shape}")


if __name__ == "__main__":
    main()
