"""A final-normalisation rule for the program (goal 2; analysis of a candidate mechanism).

The model's final LayerNorm divides the residual stream by its size, so one large write (e.g. head 1 attending to a
single token) cannot dominate unchecked. The program adds votes with no such limit. Candidate rule:

    score_i = S * sum_m beta_m V_m,i / sqrt(1 + sum_m c_m n_m,i^2)

V_m,i: mechanism m's votes at position i (as in the program, strengths included); n_m,i: their spread over the
vocabulary (unigram-weighted standard deviation); beta_m, c_m >= 0, S: named scalars (one each per mechanism).
Compared, on held-out windows, with the same mechanisms refitted without normalisation (beta_m only).

    .venv/bin/python -m analysis.goal2.ln_rule [--fit 16 --eval 32]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from analysis.fitlib import train_windows
from analysis.goal2.bottleneck import OUT
from analysis.goal2.budget import program_parts
from analysis.mech_eval import EVAL_SEED, model_logprobs, program

V = 2000
MECHS = {"FOLLOW": ["follow"], "LOCAL0": ["local0.self", "local0.earlier"], "LOCAL2": ["local2.self", "local2.earlier"],
         "RECALL": ["recall.self", "recall.earlier"], "PHRASE": ["phrase"]}


def stack(P, W):
    out = []
    for w in W:
        pp = program_parts(P, np.asarray(w[:256], np.int64))
        out.append(np.stack([sum(pp[k] for k in ks) for ks in MECHS.values()], -1).astype(np.float32))
    return torch.as_tensor(np.concatenate(out))                    # (N, V, M)


def norms(X, pu):
    mu = (X * pu[None, :, None]).sum(1, keepdim=True)
    return torch.sqrt(((X - mu) ** 2 * pu[None, :, None]).sum(1))  # (N, M)


def score(X, nrm, beta, logc, logS, normalise):
    s = X @ beta
    if normalise:
        s = s * torch.exp(logS) / torch.sqrt(1 + (torch.exp(logc)[None] * nrm ** 2).sum(-1, keepdim=True))
    return s


def fit(X, nrm, Pm, normalise, iters=200):
    M = X.shape[-1]
    beta = torch.ones(M, requires_grad=True)
    logc = torch.full((M,), -4.0, requires_grad=True)
    logS = torch.zeros(1, requires_grad=True)
    params = [beta, logc, logS] if normalise else [beta]
    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = 0.0
        for c in range(0, len(X), 512):
            sc = score(X[c:c + 512], nrm[c:c + 512], beta, logc, logS, normalise)
            l = -(Pm[c:c + 512] * torch.log_softmax(sc, -1)).sum() / len(X)
            l.backward()
            loss += float(l)
        return torch.tensor(loss)

    opt.step(closure)
    return beta.detach(), logc.detach(), logS.detach()


def evaluate(X, nrm, LP, params, normalise):
    beta, logc, logS = params
    kls = []
    with torch.no_grad():
        for c in range(0, len(X), 512):
            lq = torch.log_softmax(score(X[c:c + 512], nrm[c:c + 512], beta, logc, logS, normalise), -1)
            p = LP[c:c + 512].exp()
            kls.append((p * (LP[c:c + 512] - lq)).sum(-1))
    return torch.cat(kls).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=int, default=16)
    ap.add_argument("--eval", type=int, default=32)
    args = ap.parse_args()
    torch.set_num_threads(4)
    P = program()
    pu = torch.as_tensor(P.STATS.unigram / P.STATS.unigram.sum(), dtype=torch.float32)
    Wf, We = train_windows(args.fit, seed=2024), train_windows(args.eval, seed=EVAL_SEED)
    Xf, Xe = stack(P, Wf), stack(P, We)
    nf, ne = norms(Xf, pu), norms(Xe, pu)
    Pf = torch.as_tensor(np.exp(model_logprobs(Wf)).reshape(-1, V), dtype=torch.float32)
    LPe = torch.as_tensor(model_logprobs(We).reshape(-1, V), dtype=torch.float32)
    res = {"fit_windows": args.fit, "eval_windows": args.eval, "mechanisms": list(MECHS)}
    kl = {}
    for normalise in (False, True):
        prm = fit(Xf, nf, Pf, normalise)
        k = evaluate(Xe, ne, LPe, prm, normalise)
        name = "normalised" if normalise else "plain refit"
        kl[name] = k.reshape(args.eval, 256).mean(1)
        res[name] = {"kl": float(k.mean()), "beta": prm[0].tolist(), "c": torch.exp(prm[1]).tolist(),
                     "S": float(torch.exp(prm[2]))}
        print(name, res[name], flush=True)
    d = kl["normalised"] - kl["plain refit"]
    res["paired_diff"], res["paired_se"] = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    print(json.dumps(res, indent=1))
    with open(os.path.join(OUT, "ln_rule.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
