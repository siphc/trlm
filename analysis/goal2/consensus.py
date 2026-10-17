"""Consensus failures: where all five seeds agree but the program does not (goal 2, analysis only; TRAIN windows).

Per position: ENS = probability average of attn_s0..s4; spread = mean_k KL(ENS || seed k) (how much seeds disagree);
miss = KL(ENS || program). Positions with high miss and low spread are behaviour every seed learned that the
program lacks. Reports the share of the program's miss at low-spread positions, a breakdown by simple context
classes, and printed examples.

    .venv/bin/python -m analysis.goal2.consensus [--n 64] [--examples 40]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.fitlib import train_windows
from analysis.goal2.bottleneck import OUT, kl_rows
from analysis.mech_eval import EVAL_SEED, program
from harness.common import load_model, model_logprobs

V = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--examples", type=int, default=40)
    args = ap.parse_args()
    from program.vocab import TOKENS

    W = train_windows(args.n, seed=EVAL_SEED)
    seeds = [f"attn_s{k}" for k in range(5)]
    lps = np.stack([model_logprobs(load_model(m), W).reshape(-1, V) for m in seeds])
    P = program()
    s = P.STATS
    lq = np.concatenate([P.predict(np.asarray(w[:256], np.int64)) for w in W])
    ens = np.log(np.exp(lps).mean(0))
    spread = np.mean([kl_rows(ens, l) for l in lps], 0)
    miss = kl_rows(ens, lq)
    toks = np.asarray(W[:, :256], np.int64).reshape(-1)
    nxt = np.asarray(W[:, 1:257], np.int64).reshape(-1)
    pos = np.tile(np.arange(256), args.n)
    q = np.quantile(spread, [0.25, 0.5, 0.75])
    res = {"n_windows": args.n, "miss_mean": float(miss.mean()), "spread_mean": float(spread.mean()),
           "corr(miss, spread)": float(np.corrcoef(miss, spread)[0, 1]),
           "miss by spread quartile": [float(miss[(spread >= lo) & (spread < hi)].mean()) for lo, hi in
                                       zip([-1, *q], [*q, 99])],
           "share of miss at low-spread positions (spread below median)": float(miss[spread < q[1]].sum() / miss.sum())}
    # context classes
    ens_top = ens.argmax(-1)
    cls = {
        "ensemble's top token appeared earlier in the window": np.array(
            [ens_top[i] in set(toks[i - p: i + 1]) for i, p in enumerate(pos)]),
        "ensemble's top token is a word piece": s.piece[ens_top] > 0.5,
        "current token is a word piece": s.piece[toks] > 0.5,
        "ensemble is confident (top prob > 0.5)": np.exp(ens.max(-1)) > 0.5,
    }
    res["classes"] = {k: {"share_positions": float(m.mean()), "miss": float(miss[m].mean()),
                          "share_of_miss": float(miss[m].sum() / miss.sum()), "spread": float(spread[m].mean())}
                      for k, m in cls.items()}
    # confident consensus: ensemble top prob > 0.5 and low spread; what does the program give the top token?
    conf = (np.exp(ens.max(-1)) > 0.5) & (spread < q[1])
    pt = np.exp(lq[np.arange(len(lq)), ens_top])
    res["confident consensus"] = {"share_positions": float(conf.mean()), "share_of_miss": float(miss[conf].sum() / miss.sum()),
                                  "ensemble top prob": float(np.exp(ens.max(-1))[conf].mean()),
                                  "program prob on it": float(pt[conf].mean()),
                                  "of these, top token appeared earlier": float(cls["ensemble's top token appeared earlier in the window"][conf].mean()),
                                  "of these, top token is a word piece": float(cls["ensemble's top token is a word piece"][conf].mean())}
    print(json.dumps(res, indent=1))
    with open(os.path.join(OUT, "consensus.json"), "w") as f:
        json.dump(res, f, indent=1)
    score = miss - 3 * spread
    for i in np.argsort(-score)[: args.examples]:
        w, p = divmod(i, 256)
        ctx = "".join(TOKENS[x] for x in toks[i - min(p, 14): i + 1]).replace("\n", "/")
        e = np.argsort(-ens[i])[:4]
        g = np.argsort(-lq[i])[:4]
        print(f"miss {miss[i]:.2f} spread {spread[i]:.2f} ...{ctx!r} -> {TOKENS[nxt[i]]!r}\n"
              f"   seeds: " + " ".join(f"{TOKENS[v]!r}:{np.exp(ens[i, v]):.2f}" for v in e)
              + "\n   prog:  " + " ".join(f"{TOKENS[v]!r}:{np.exp(lq[i, v]):.2f}" for v in g))


if __name__ == "__main__":
    main()
