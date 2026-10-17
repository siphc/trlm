"""Where does the program disagree with the model? (goal 2, analysis only; TRAIN windows)

Per position: KL(model || program), entropies, whether the model's top token is a word piece (the model expects the
current word to continue), the current token's shape, and the split of the KL into the model's top tokens and the
tail. Also a per-position temperature oracle (best single scale of the program's log-probs per position) and the
best global temperature.

    .venv/bin/python -m analysis.goal2.diagnose [--n 32]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.fitlib import train_windows
from analysis.goal2.bottleneck import OUT, kl_rows
from analysis.mech_eval import EVAL_SEED, model_logprobs, program

V = 2000


def entropy(lp):
    return -(np.exp(lp) * lp).sum(-1)


def scaled(lq, t):
    z = lq * t
    return z - np.logaddexp.reduce(z, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()
    P = program()
    s = P.STATS
    W = train_windows(args.n, seed=EVAL_SEED)
    lp = model_logprobs(W)
    toks = np.asarray(W[:, :256], np.int64)
    lq = np.stack([P.predict(t) for t in toks])
    lp, lq = lp.reshape(-1, V), lq.reshape(-1, V)
    t = toks.reshape(-1)
    kl = kl_rows(lp, lq)
    Hm, Hq = entropy(lp), entropy(lq)
    top_m = lp.argmax(-1)
    top_q = lq.argmax(-1)
    piece_next = s.piece[top_m] > 0.5
    shape = s.shape[t]
    res = {"n_windows": args.n, "kl": float(kl.mean()), "top1_agree": float((top_m == top_q).mean())}
    srt = np.sort(kl)[::-1]
    res["kl_share_top10pct_positions"] = float(srt[: len(srt) // 10].sum() / srt.sum())
    res["kl_quantiles"] = {q: float(np.quantile(kl, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)}
    res["entropy"] = {"model": float(Hm.mean()), "program": float(Hq.mean()),
                      "corr_kl_vs_Hq_minus_Hm": float(np.corrcoef(kl, Hq - Hm)[0, 1]),
                      "share_positions_program_flatter": float((Hq > Hm).mean())}
    # by class
    def by(mask):
        return {"share_positions": float(mask.mean()), "kl_mean": float(kl[mask].mean()),
                "share_kl": float(kl[mask].sum() / kl.sum()), "H_model": float(Hm[mask].mean()),
                "H_program": float(Hq[mask].mean()), "top1": float((top_m == top_q)[mask].mean())}
    res["by_model_expects_piece"] = {"yes": by(piece_next), "no": by(~piece_next)}
    res["by_current_shape"] = {name: by(shape == k) for k, name in
                               enumerate(["word start", "continuation", "other", "end of story"]) if (shape == k).any()}
    # KL split: model's top-10 tokens vs the rest (sum of p log p/q terms; both parts can be negative)
    p = np.exp(lp)
    terms = p * (lp - lq)
    idx = np.argsort(-lp, axis=1)[:, :10]
    top_part = np.take_along_axis(terms, idx, 1).sum(1)
    res["kl_split"] = {"model_top10_terms": float(top_part.mean()), "rest": float((kl - top_part).mean()),
                       "model_mass_top10": float(np.take_along_axis(p, idx, 1).sum(1).mean()),
                       "program_mass_on_model_top10": float(np.take_along_axis(np.exp(lq), idx, 1).sum(1).mean())}
    # temperature oracles
    grid = np.linspace(0.6, 1.6, 21)
    per = np.stack([kl_rows(lp, scaled(lq, g)) for g in grid])          # (G, N)
    res["temperature"] = {"best_global": float(grid[per.mean(1).argmin()]), "kl_best_global": float(per.mean(1).min()),
                          "kl_per_position_oracle": float(per.min(0).mean())}
    print(json.dumps(res, indent=1))
    with open(os.path.join(OUT, "diagnose.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
