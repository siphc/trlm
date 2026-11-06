"""EXPLORATORY (logs/prereg_fixed_votes.md, "Exploratory": usefulness relative to the model's own prediction instead of
the bigram; decision 13). Does the look-follows-write law describe more of a fixed-vote model's attention when the
baseline is what the model already predicts (explaining away by the other heads), as in the gradient late in training?

For head h with table T_h, distance d, key token k, query token q (aggregated over a slice of the TRAIN stream):

    S(k, q)        = sum over positions of T_h(k, y)                     (y the actual next token)
    bigram         U = (S - N * sum_v P1(v | q) T_h(k, v)) / (N + 20)    (the pre-registered usefulness, on this slice)
    model          U = (S - sum over positions of sum_v p(v) T_h(k, v)) / (N + 20)      p: the model's prediction
    minus_h        as model, with p the model's prediction without head h's vote (what the OTHER heads and the
                   bigram already predict: the explaining-away baseline)

Then, per head, goal 1's base rule plus each usefulness (c1 at d = 1, c2 at d = 2..7, no band; features beta_h * U)
is fitted to the head's pattern (train_windows(32, 2024)) and substituted in the exact model on train_windows(32,
2025), as analysis/fixed_votes/law.py does; and the pure-pair correlation at d = 1 (as law.py). Reference arms: the
full-stream bigram usefulness without band (own_nb), and from law.json the pre-registered own arm (with band) and
the token-token ceilings tt and tt_self.

    OMP_NUM_THREADS=4 nice -n 20 .venv/bin/python -m analysis.fixed_votes.explain_away models/fixed_votes/base_s0/final.pt
        [--tokens 10000000]

Writes analysis/out/fixed_votes/<name>/explain_away.json and the tables as explain_<kind>_h<h>_d<d>.npy (float32).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from analysis.fixed_votes import law as FL
from analysis.look_write import common as C
from analysis.look_write.fit_arms import Rule, double_centre, fit_rule, pair_features, wcorr

KINDS = ("bigram", "model", "minus_h")
T_WIN = 256


def accumulate(m, corpus, n_tokens, bs=32, log=print):
    """Sums S, the model baselines and N over non-overlapping windows of the stream's first n_tokens tokens."""
    H, V, D = m.cfg.n_heads, C.V, C.D_MAX
    n_win = n_tokens // T_WIN
    S = torch.zeros(H, D, V * V)
    Bm = torch.zeros(H, D, V * V)
    Bx = torch.zeros(H, D, V * V)
    N = torch.zeros(D, V * V)
    tabs = m.tables
    t0 = time.time()
    with torch.no_grad():
        for b0 in range(0, n_win, bs):
            nb = min(bs, n_win - b0)
            w = np.asarray(corpus[b0 * T_WIN:(b0 + nb) * T_WIN + 1], np.int64)
            x = torch.as_tensor(np.stack([w[i * T_WIN:(i + 1) * T_WIN] for i in range(nb)]))
            y = torch.as_tensor(np.stack([w[i * T_WIN + 1:(i + 1) * T_WIN + 1] for i in range(nb)]))
            a = m.patterns(x)
            logits = m.beta_follow * m.follow[x]
            votes = []
            for h in range(H):
                votes.append(m.beta[h] * (a[:, h] @ tabs[h][x]))
                logits = logits + votes[-1]
            p = torch.softmax(logits, -1)
            for d in range(1, D + 1):
                k, q, yy = x[:, :-d], x[:, d:], y[:, d:]
                idx = (k * V + q).reshape(-1)
                N[d - 1].index_add_(0, idx, torch.ones(idx.numel()))
                pd = p[:, d:]
                for h in range(H):
                    R = tabs[h][k]                                           # (nb, T - d, V): T_h(k, .)
                    S[h, d - 1].index_add_(0, idx, R.gather(-1, yy[..., None]).reshape(-1))
                    Bm[h, d - 1].index_add_(0, idx, (R * pd).sum(-1).reshape(-1))
                    pm = torch.softmax(logits[:, d:] - votes[h][:, d:], -1)
                    Bx[h, d - 1].index_add_(0, idx, (R * pm).sum(-1).reshape(-1))
            if (b0 // bs) % 50 == 0:
                done = b0 + nb
                log(f"  {done}/{n_win} windows, {time.time() - t0:.0f}s (eta {(time.time() - t0) / done * (n_win - done):.0f}s)")
    return S.numpy(), Bm.numpy(), Bx.numpy(), N.numpy()


def run(path, n_tokens, log=print):
    from harness.common import load_corpus
    from src.train.train import load_model

    C.set_threads()
    m = load_model(path)
    for p in m.parameters():
        p.requires_grad_(False)
    names, H, V = m.cfg.names, m.cfg.n_heads, C.V
    beta = [float(b) for b in m.beta]
    out_dir = os.path.join(FL.OUT, FL.name_of(path))
    os.makedirs(out_dir, exist_ok=True)
    log(f"{path}: accumulating over {n_tokens:,} tokens")
    S, Bm, Bx, N = accumulate(m, load_corpus(), n_tokens, log=log)
    P1 = torch.exp(m.follow.double()).numpy()                                # [q, v]
    U = {kind: {h: {} for h in range(H)} for kind in KINDS}
    for h in range(H):
        Bbig = (m.tables[h].double().numpy() @ P1.T).reshape(-1)             # [k, q] flattened
        for d in range(1, C.D_MAX + 1):
            n = N[d - 1].astype(np.float64)
            for kind, base in (("bigram", n * Bbig), ("model", Bm[h, d - 1]), ("minus_h", Bx[h, d - 1])):
                u = ((S[h, d - 1] - base) / (n + C.SHRINK)).reshape(V, V).astype(np.float32)
                U[kind][h][d] = u
                np.save(os.path.join(out_dir, f"explain_{kind}_h{h}_d{d}.npy"), u)
    del S, Bm, Bx
    law = json.load(open(os.path.join(out_dir, "law.json")))
    shapes = C.token_shapes()
    fit_w, ev_w = C.train_windows(C.FIT_SEED), C.train_windows(C.EVAL_SEED)
    A_fit = FL.patterns(m, fit_w)
    lp_ref = FL.logprobs(m, ev_w)
    rf, re = Rule(fit_w, shapes), Rule(ev_w, shapes)
    Y = C.pure_pair_scores(FL.fv_weights(m))
    N1 = np.load(os.path.join(C.REPO, "analysis", "out", "candidates", "qk", "N_1.npy")).astype(np.float64)
    kidx, qidx = np.nonzero(N1 >= C.MIN_COUNT)
    wts = N1[kidx, qidx]
    res = {"model": path, "tokens": n_tokens, "tables": names, "beta": beta, "heads": {}}
    for h in range(H):
        t0 = time.time()
        full, _ = FL.usefulness_tables(names[h])
        feats = {"base": {}}
        sources = {"own_nb": full, **{k: U[k][h] for k in KINDS}}
        for arm, tabs in sources.items():
            feats[arm] = tabs
        hr = {"table": names[h], "arms": {}, "pure_pair_corr": {},
              "from_law_json": {a: law["heads"][str(h)]["arms"][a]["gain"] for a in ("own", "tt", "tt_self")}}
        A_f = torch.as_tensor(A_fit[:, h].astype(np.float64))
        for arm, tabs in feats.items():
            if tabs:
                f1, f2 = pair_features(fit_w, {d: beta[h] * t for d, t in tabs.items()})
                e1, e2 = pair_features(ev_w, {d: beta[h] * t for d, t in tabs.items()})
                ff = {"c1": torch.as_tensor(f1), "c2": torch.as_tensor(f2)}
                fe = {"c1": torch.as_tensor(e1), "c2": torch.as_tensor(e2)}
            else:
                ff, fe = {}, {}
            theta, loss, _ = fit_rule(A_f, rf, ff)
            with torch.no_grad():
                pat = torch.exp(re.logpattern(theta, fe)).numpy()
            hr["arms"][arm] = {"kl": C.kl(lp_ref, FL.logprobs(m, ev_w, h, pat)), "pattern_ce_fit": loss,
                               "coef": {k: float(v) for k, v in theta.items() if k not in ("beta", "sk")}}
        base = hr["arms"]["base"]["kl"]
        for arm in hr["arms"]:
            hr["arms"][arm]["gain"] = base - hr["arms"][arm]["kl"]
        ry = double_centre(Y[h], kidx, qidx, wts)[0]
        for arm, tabs in sources.items():
            hr["pure_pair_corr"][arm] = wcorr(ry, double_centre(tabs[1], kidx, qidx, wts)[0], wts)
        res["heads"][h] = hr
        log(f"  head {h} ({names[h]}): gains " + ", ".join(f"{a} {v['gain']:+.5f}" for a, v in hr["arms"].items()
                                                         if a != "base")
            + f"; law.json own {hr['from_law_json']['own']:+.5f} tt {hr['from_law_json']['tt']:+.5f}; pure-pair "
            + ", ".join(f"{a} {v:+.3f}" for a, v in hr["pure_pair_corr"].items()) + f" [{time.time() - t0:.0f}s]")
    with open(os.path.join(out_dir, "explain_away.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--tokens", type=int, default=10_000_000)
    a = ap.parse_args()
    for p in a.paths:
        run(p, a.tokens, log=lambda s: print(s, flush=True))
