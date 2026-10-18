"""Softmax-bottleneck test (analysis only).

The model's logits are W_E . LN(h): every next-token log-probability vector it can output lies in the span of its
128-dimensional token embedding (plus nothing else: no output bias). The model is trained to be the cross-entropy
best fit to the stories inside that family. The program's distribution q has no such limit. Question: does the
M-projection of q onto the family {softmax(B y)}, y in R^k,

    q' = argmin_{p in family} KL(q || p)      (moment matching: B^T p = B^T q)

move the program towards the model? With B = the model's W_E this is an oracle (insight only). With B built from
corpus statistics it would be a mechanism the program could state.

    OMP_NUM_THREADS=4 .venv/bin/python -m analysis.program_gap.bottleneck [--n 16] [--bases model,bigram_pca,...]
Writes analysis/out/program_gap/bottleneck.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from analysis.fitlib import REPO, train_windows
from analysis.mech_eval import EVAL_SEED, model_logprobs, program

OUT = os.path.join(REPO, "analysis", "out", "program_gap")
V = 2000


def project(logq, B, iters=200):
    """M-projection of each row of logq (N, V) onto softmax(B y), B (V, k). Returns log-probs (N, V) and the
    final max gradient norm. Convex; L-BFGS on the summed objective (rows are independent)."""
    Bt = torch.as_tensor(B, dtype=torch.float64)
    lq = torch.as_tensor(logq, dtype=torch.float64)
    q = lq.exp()
    # init: least squares of the centred log q on B
    c = lq - lq.mean(1, keepdim=True)
    Bc = Bt - Bt.mean(0, keepdim=True)
    y0 = torch.linalg.lstsq(Bc, c.T).solution.T                    # (N, k)
    y = y0.clone().requires_grad_(True)
    target = q @ Bt                                                # (N, k) moments
    opt = torch.optim.LBFGS([y], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        z = y @ Bt.T
        f = (torch.logsumexp(z, 1) - (target * y).sum(1)).sum()
        f.backward()
        return f

    opt.step(closure)
    with torch.no_grad():
        z = y @ Bt.T
        lp = z - torch.logsumexp(z, 1, keepdim=True)
        g = (lp.exp() @ Bt - target).abs().max()
    return lp.numpy(), float(g)


def kl_rows(lp, lq):
    p = np.exp(lp)
    return (p * (lp - lq)).sum(-1)


def corpus_bases(s, k):
    """Candidate k-dimensional output bases from corpus statistics (columns span a subspace of R^V)."""
    p = s.unigram / s.unigram.sum()
    w = np.sqrt(p)[:, None]
    out = {}

    def pca(M, name):
        Mc = M - M.mean(1, keepdims=True)                          # per-row centring (softmax shift)
        mu = (p[:, None] * Mc).sum(0)
        U, S, Vt = np.linalg.svd(w * (Mc - mu), full_matrices=False)
        out[name] = np.concatenate([mu[:, None], Vt[: k - 1].T], 1)

    pca(np.asarray(s.log_follow, np.float64), "bigram_pca")
    pca(np.asarray(s.pmi[1], np.float64), "pmi1_pca")
    stack = np.concatenate([np.asarray(s.log_follow, np.float64)] +
                           [np.asarray(s.pmi[d], np.float64) for d in (2, 3, 4)], 0)
    p4 = np.tile(p, 4) / 4
    Mc = stack - stack.mean(1, keepdims=True)
    mu = (p4[:, None] * Mc).sum(0)
    _, _, Vt = np.linalg.svd(np.sqrt(p4)[:, None] * (Mc - mu), full_matrices=False)
    out["skip1to4_pca"] = np.concatenate([mu[:, None], Vt[: k - 1].T], 1)
    rng = np.random.default_rng(0)
    out["random"] = rng.standard_normal((V, k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--k", type=int, default=128)
    args = ap.parse_args()
    torch.set_num_threads(4)
    os.makedirs(OUT, exist_ok=True)
    P = program()
    W = train_windows(args.n, seed=EVAL_SEED)
    t0 = time.time()
    lp = model_logprobs(W).reshape(-1, V)
    lq = np.concatenate([P.predict(np.asarray(w[:256], np.int64)) for w in W])
    print(f"program on {args.n} windows: {time.time() - t0:.0f}s", flush=True)
    from harness.common import load_model

    E = load_model("attn_s0").wte.weight.detach().double().numpy()
    bases = {"model_WE": E, **corpus_bases(P.STATS, args.k)}
    res = {"n_windows": args.n, "k": args.k, "program": float(kl_rows(lp, lq).mean())}
    # sanity: the model's own distribution is in the W_E family
    lpp, g = project(lp[:512], E)
    res["sanity_model_projected_kl"] = float(kl_rows(lp[:512], lpp).mean())
    print(res, flush=True)
    for name, B in bases.items():
        t0 = time.time()
        lqp, g = project(lq, B)
        r = {"kl": float(kl_rows(lp, lqp).mean()), "max_grad": g,
             "kl_to_program": float(kl_rows(lq, lqp).mean()), "seconds": time.time() - t0}
        if name != "model_WE":
            # overlap of the basis span with the model's embedding span (mean squared canonical correlation)
            Qa, _ = np.linalg.qr(E - E.mean(0))
            Qb, _ = np.linalg.qr(B - B.mean(0))
            r["span_overlap_with_WE"] = float((np.linalg.svd(Qa.T @ Qb, compute_uv=False) ** 2).mean())
        res[name] = r
        print(name, r, flush=True)
    with open(os.path.join(OUT, "bottleneck.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
