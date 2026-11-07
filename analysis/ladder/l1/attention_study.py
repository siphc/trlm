"""How much of the L1 head's context effect do candidate table descriptions explain? (decision 22)

For each seed, the head's write is computed from per-symbol tables (build_account_v1.tables: the score is
LAM·KAP over N_COMP components, plus the rank-1 window-start term ALPHA·RHO·BETA; the write is WO·VALUE). It is
scored by context faithfulness, 1 - KL(model || head patched) / KL(model || head -> its mean for the current
symbol), and by the usual faithfulness against resample ablation.

Variants: the components kept (1, 2, 3), with and without the position term, and symbol classes (k-means on the
query loadings, the key loadings and the writes, each separately; frequency weighted).

    .venv/bin/python -m analysis.ladder.l1.attention_study --arm ln
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from analysis.ladder.common import kl_pos, load, windows
from analysis.ladder.compress_probe import kmeans
from analysis.ladder.l1.build_account_v1 import frequencies, tables
from analysis.ladder.scrub import finish, mean_for_current_symbol, parts, scrubbed


def head_write(X, lam, kap, write, pos=None):
    s = np.einsum("ntr,nsr->nts", lam[X], kap[X])
    if pos is not None:
        alpha, rho, beta = pos
        T = X.shape[1]
        s = s + alpha[X][:, :, None] * rho[None, None, :T] * beta[X][:, None, :]
    T = X.shape[1]
    s = np.where(np.tri(T, dtype=bool), s, -np.inf)
    w = np.exp(s - s.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    return np.einsum("nts,nsd->ntd", w, write[X])


def classes(M, f, k):
    a = kmeans(M, np.sqrt(f), k)
    C = np.stack([np.average(M[a == j], 0, f[a == j]) if (a == j).any() else np.zeros(M.shape[1]) for j in range(k)])
    return C[a]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ln")
    ap.add_argument("--windows", type=int, default=1000)
    a = ap.parse_args()
    f = frequencies()
    X = windows("L1", "dev_full")[:a.windows, :-1].astype(np.int64)
    out = {}
    for seed in range(5):
        m = load("L1", a.arm, seed).double()
        x0, z = parts(m, X)
        lp = finish(m, x0, z)
        zm = z.copy()
        zm[:, :, 0] = mean_for_current_symbol(X, z[:, :, 0], lp.shape[-1])
        k_mean = kl_pos(lp, finish(m, x0, zm)).mean()
        k_res = kl_pos(lp, scrubbed(m, x0, z, {"head0": None}, np.random.default_rng(0))).mean()
        t = tables(m)
        write = t["VALUE"] @ t["HEAD_OUT"].T
        pos = (t["ALPHA"], t["RHO"], t["BETA"])
        zero = np.zeros_like(z)
        row = {"kl_mean_for_symbol": float(k_mean), "kl_resample": float(k_res)}

        def score(name, w):
            k = kl_pos(lp, finish(m, x0, zero, w)).mean()
            row[name] = {"context_faith": round(float(1 - k / k_mean), 4), "faith": round(float(1 - k / k_res), 4)}

        for r in (1, 2, 3):
            score(f"{r} comp", head_write(X, t["LAM"][:, :r], t["KAP"][:, :r], write))
            score(f"{r} comp + position", head_write(X, t["LAM"][:, :r], t["KAP"][:, :r], write, pos))
        for k in (8, 16, 32):
            lam_c = classes(t["LAM"], f, k)
            kap_c = classes(t["KAP"], f, k)
            wr_c = classes(write, f, k)
            score(f"classes {k} (query, key, write) + position", head_write(X, lam_c, kap_c, wr_c, pos))
            score(f"classes {k} key only + position", head_write(X, t["LAM"], kap_c, write, pos))
        out[f"s{seed}"] = row
        print(f"s{seed}: " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
