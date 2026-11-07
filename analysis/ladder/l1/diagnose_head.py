"""Which simplification of the L1 head fails on a seed: replacement-test faithfulness of the head computed
(a) exactly, (b) with mean positions everywhere except the keys, (c) content only, (d) with the account's score
(rank 2 content + rank-1 position term), and the singular values of the score table and the position term.

    .venv/bin/python -m analysis.ladder.l1.diagnose_head --arm ln --seed 1
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from analysis.ladder.common import kl_pos, load, windows
from analysis.ladder.l1.build_account import frequencies, ln, tables
from analysis.ladder.scrub import finish, parts, scrubbed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ln")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    m = load("L1", a.arm, a.seed).double()
    X = windows("L1", "dev_full")[:, :-1].astype(np.int64)
    n, T = X.shape
    x0, z = parts(m, X)
    lp = finish(m, x0, z)
    S = {k: v.numpy() for k, v in m.state_dict().items()}
    E, P = S["wte.weight"], S["wpe.weight"]
    g1 = S.get("blocks.0.ln1.weight")
    norm = (lambda x: ln(x, g1)) if g1 is not None else (lambda x: x)
    Wq, Wk, Wv = np.split(S["blocks.0.attn.qkv.weight"], 3, 0)
    dh = m.cfg.d_head
    mask = np.tril(np.ones((T, T), bool))
    rng = np.random.default_rng(0)
    kr = kl_pos(lp, scrubbed(m, x0, z, {"head0": None}, rng)).mean()

    def faith(v):
        return float(1 - kl_pos(lp, scrubbed(m, x0, z, {"head0": ("value", v)}, rng)).mean() / kr)

    def attend(s, vals):
        s = np.where(mask, s, -np.inf)
        w = np.exp(s - s.max(-1, keepdims=True))
        w /= w.sum(-1, keepdims=True)
        return np.einsum("nts,nsd->ntd", w, vals)

    pm = P[1:].mean(0)
    xm = norm(E[X] + pm)
    xk = norm(E[X] + P[None, :T])
    out = {}
    out["exact"] = faith(attend(np.einsum("ntd,nsd->nts", xk @ Wq.T, xk @ Wk.T) / np.sqrt(dh), xk @ Wv.T))
    out["mean positions except keys"] = faith(attend(np.einsum("ntd,nsd->nts", xm @ Wq.T, xk @ Wk.T) / np.sqrt(dh),
                                                     xm @ Wv.T))
    out["content only"] = faith(attend(np.einsum("ntd,nsd->nts", xm @ Wq.T, xm @ Wk.T) / np.sqrt(dh), xm @ Wv.T))
    t = tables(m)
    sc = np.einsum("ntr,nsr->nts", t["LAM"][X], t["KAP"][X])
    out[f"rank-{t['LAM'].shape[1]} content"] = faith(attend(sc, t["VALUE"][X]))
    sc = sc + t["ALPHA"][X[:, :, None]] * t["RHO"][None, None, :T] * t["BETA"][X[:, None, :]]
    out["account (content components + rank-1 position)"] = faith(attend(sc, t["VALUE"][X]))
    # exact content score + exact key-position delta with mean-position queries and values = "mean positions except keys"
    f = frequencies()
    u = norm(E + pm)
    score = (u @ Wq.T) @ (u @ Wk.T).T / np.sqrt(dh)
    score = score - (score @ f)[:, None]
    sf = np.sqrt(f)
    out["score table weighted singular values"] = np.round(np.linalg.svd(sf[:, None] * score * sf[None],
                                                                         compute_uv=False)[:4], 3).tolist()
    ub = norm(E[None] + P[:, None])
    D = np.einsum("ad,jbd->ajb", u @ Wq.T, ub @ Wk.T - (u @ Wk.T)[None]) / np.sqrt(dh)
    Mq = (sf[:, None, None] * D * sf[None, None, :]).reshape(len(E), -1)
    out["position term singular values over query"] = np.round(np.linalg.svd(Mq, compute_uv=False)[:4], 3).tolist()
    Mj = (sf[:, None, None] * D * sf[None, None, :]).transpose(1, 0, 2).reshape(T, -1)
    out["position term singular values over key position"] = np.round(np.linalg.svd(Mj, compute_uv=False)[:4],
                                                                       3).tolist()
    out["resample KL of the head"] = float(kr)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
