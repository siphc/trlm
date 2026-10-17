"""vote_fit with saturating mechanisms (goal 2): every vote strength refitted, plus a final-normalisation term for
chosen mechanisms, compared paired on held-out TRAIN windows with the plain refit.

    score_i = sum_{k not in G} beta_k F_k,i + sum_{k in G} beta_k F_k,i / sqrt(1 + c * n_i^2)
    n_i = unigram-weighted spread over the vocabulary of sum_{k in G} beta_k F_k,i

G = the saturating mechanism's directions (default: RECALL, head1.*). The candidate file (analysis/vote_fit.py
format) can add or replace directions.

    .venv/bin/python -m analysis.goal2.vote_fit_sat [candidate.py] [--sat head1] [--fit 16 --eval 32]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from analysis import vote_fit
from analysis.fitlib import train_windows
from analysis.goal2.bottleneck import OUT
from analysis.mech_eval import EVAL_SEED, model_logprobs, program

V = 2000


MODE = {"m": "group"}


def score(X, beta, gmask, logc, pu, sat):
    if not sat:
        return X @ beta
    plain = X[..., ~gmask] @ beta[~gmask]
    g = X[..., gmask] @ beta[gmask]
    mu = (g * pu).sum(-1, keepdim=True)
    n2 = (((g - mu) ** 2) * pu).sum(-1, keepdim=True)
    if MODE["m"] == "everything":                  # the whole score shrinks when the group writes a lot
        return (plain + g) / torch.sqrt(1 + torch.exp(logc) * n2)
    return plain + g / torch.sqrt(1 + torch.exp(logc) * n2)


def run_fit(X, Pm, beta0, gmask, pu, sat, iters=200):
    beta = torch.tensor(beta0, dtype=torch.float32, requires_grad=True)
    logc = torch.tensor(-3.0, requires_grad=True)
    opt = torch.optim.LBFGS([beta, logc] if sat else [beta], lr=1, max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-10, history_size=50)

    def closure():
        opt.zero_grad()
        loss = 0.0
        for c in range(0, len(X), 256):
            sc = score(X[c:c + 256].float(), beta, gmask, logc, pu, sat)
            l = -(Pm[c:c + 256] * torch.log_softmax(sc, -1)).sum() / len(X)
            l.backward()
            loss += float(l)
        return torch.tensor(loss)

    opt.step(closure)
    return beta.detach(), logc.detach()


def per_window_kl(X, LP, beta, gmask, logc, pu, sat, n_win):
    out = []
    with torch.no_grad():
        for c in range(0, len(X), 256):
            lq = torch.log_softmax(score(X[c:c + 256].float(), beta, gmask, logc, pu, sat), -1)
            lp = LP[c:c + 256]
            out.append((lp.exp() * (lp - lq)).sum(-1))
    return torch.cat(out).numpy().reshape(n_win, -1).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", nargs="?")
    ap.add_argument("--sat", default="head1", help="prefix of the saturating mechanism's directions")
    ap.add_argument("--fit", type=int, default=16)
    ap.add_argument("--eval", type=int, default=32)
    ap.add_argument("--tag", default="")
    ap.add_argument("--mode", default="group", choices=["group", "everything"])
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("VOTE_FIT_THREADS", "4")))
    MODE["m"] = args.mode
    P = program()
    pu = torch.as_tensor(P.STATS.unigram / P.STATS.unigram.sum(), dtype=torch.float32)
    cand = vote_fit.load_candidate(args.candidate) if args.candidate else None
    Wf, We = train_windows(args.fit, seed=vote_fit.FIT_SEED), train_windows(args.eval, seed=EVAL_SEED)
    Pf = torch.as_tensor(np.exp(model_logprobs(Wf)).reshape(-1, V), dtype=torch.float32)
    LPe = torch.as_tensor(model_logprobs(We).reshape(-1, V), dtype=torch.float32)
    cname = getattr(cand, "NAME", None)
    res = {"candidate_name": cname, "sat": args.sat, "fit": args.fit, "eval": args.eval}
    kl = {}
    for arm, c in (("base", None), ("candidate", cand)) if cand else (("base", None),):
        Xf, keys, b0 = vote_fit.stacked(P, Wf, c)
        Xe, _, _ = vote_fit.stacked(P, We, c)
        Xf, Xe = torch.from_numpy(Xf), torch.from_numpy(Xe)
        prefixes = tuple(x + "." for x in args.sat.split(","))
        gmask = torch.tensor([args.sat == "all" or k.startswith(prefixes) for k in keys])
        beta0 = [b0[k] for k in keys]
        for sat in (False, True):
            beta, logc = run_fit(Xf, Pf, beta0, gmask, pu, sat)
            name = f"{arm}{' + saturation' if sat else ''}"
            kl[name] = per_window_kl(Xe, LPe, beta, gmask, logc, pu, sat, args.eval)
            res[name] = {"kl": float(kl[name].mean()), "c": float(torch.exp(logc)) if sat else None,
                         "strengths": {k: round(float(b), 4) for k, b in zip(keys, beta)}}
            print(name, res[name]["kl"], res[name]["c"], flush=True)
        del Xf, Xe
    for a, b in (("base + saturation", "base"), ("candidate", "base"), ("candidate + saturation", "base")):
        if a in kl:
            d = kl[a] - kl[b]
            res[f"{a} vs {b}"] = {"diff": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}
    print(json.dumps({k: v for k, v in res.items() if " vs " in k}, indent=1))
    with open(os.path.join(OUT, f"vfsat_{cname or 'base'}_{args.sat.replace(',', '+')}{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
