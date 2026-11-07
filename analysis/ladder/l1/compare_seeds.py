"""Do the L1 account's tables mean the same thing on every seed? For each seed: the singular values of the score
table, the symbols at each end of each key scale, the query loadings of the space and of the letters, and the
attention entropy for space and letter queries on dev_full.

    .venv/bin/python -m analysis.ladder.l1.compare_seeds --arm ln
"""
from __future__ import annotations

import argparse

import numpy as np

from analysis.ladder.common import load, windows
from analysis.ladder.dump import attention, symbol_names
from analysis.ladder.l1.build_account import frequencies, tables


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ln")
    a = ap.parse_args()
    names = symbol_names()
    f = frequencies()
    common = f > 3e-3
    X = windows("L1", "dev_full")[:300, :-1].astype(np.int64)
    sp = names.index("␣")
    lower = np.array([len(c) == 1 and c.islower() for c in names])
    upper = np.array([len(c) == 1 and c.isupper() for c in names])
    for s in range(5):
        m = load("L1", a.arm, s).double()
        t = tables(m)
        lam, kap = t["LAM"], t["KAP"]
        mag = np.sqrt((lam ** 2 * f[:, None]).sum(0))
        print(f"## seed {s}: component strength (freq-weighted rms of LAM) {np.round(mag, 1).tolist()}")
        for r in range(kap.shape[1]):
            o = [i for i in np.argsort(kap[:, r]) if common[i]]
            print(f"  scale {r + 1}: low end {' '.join(names[i] for i in o[:8])} | high end "
                  f"{' '.join(names[i] for i in o[-8:])}")
            print(f"     KAP: space {kap[sp, r]:+.2f}, capitals mean {kap[upper & common, r].mean():+.2f}, lowercase "
                  f"mean {kap[lower & common, r].mean():+.2f};  LAM: space {lam[sp, r]:+.1f}, lowercase mean "
                  f"{np.average(lam[lower, r], weights=f[lower]):+.1f}")
        A = attention(m, X, 0)
        H = -(A * np.log(A + 1e-30)).sum(-1)[:, 16:]
        q = X[:, 16:]
        print(f"  attention entropy (nats): space queries {H[q == sp].mean():.2f}, lowercase queries "
              f"{H[lower[q]].mean():.2f}, capital queries {H[upper[q]].mean():.2f}")


if __name__ == "__main__":
    main()
