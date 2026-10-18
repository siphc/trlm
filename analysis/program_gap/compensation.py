"""Does the model's current-token table compensate for what the context usually adds? (analysis only)

For each current token s (>= MIN positions in the TRAIN windows):
  B(s, .)      the model's current-token logit part (direct + every head's current-token part), centred, averaged
  Cm(s, .)     the model's context part (every head's earlier-token part), centred, averaged over the same positions
  Cp(s, .)     the program's context votes (LOCAL-0/2 earlier votes, RECALL earlier votes, PHRASE), same
Regress B on log P1, log P and Cm or Cp (weighted by the model's mean next-token mass). Hypothesis: B ~ a log P1
- b C(s, .) with b > 0 (the current token supplies what the context does not already supply).

    .venv/bin/python -m analysis.program_gap.compensation [--n 128]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from analysis.fitlib import train_windows
from analysis.program_gap.bottleneck import OUT
from analysis.program_gap.budget import model_parts, program_parts
from analysis.mech_eval import program

V = 2000
MIN = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--model", default="attn_s0")
    args = ap.parse_args()
    from harness.common import load_model

    torch.set_num_threads(4)
    model = load_model(args.model)
    P = program()
    s = P.STATS
    W = train_windows(args.n, seed=7)
    accB, accC, accP, cnt, pm = np.zeros((V, V)), np.zeros((V, V)), np.zeros((V, V)), np.zeros(V), np.zeros((V, V))
    for w in W:
        t = np.asarray(w[:256], np.int64)
        mp, logits = model_parts(model, t)
        pp = program_parts(P, t)
        Bp = mp["direct"] + sum(mp[f"self{h}"] for h in range(4))
        Cm = sum(mp[f"earlier{h}"] for h in range(4))
        Cp = pp["local0.earlier"] + pp["local2.earlier"] + pp["recall.earlier"] + pp["phrase"]
        for acc, X in ((accB, Bp), (accC, Cm), (accP, Cp)):
            np.add.at(acc, t, X - X.mean(-1, keepdims=True))
        np.add.at(cnt, t, 1)
        z = logits - logits.max(-1, keepdims=True)
        np.add.at(pm, t, np.exp(z) / np.exp(z).sum(-1, keepdims=True))
    keep = np.nonzero(cnt >= MIN)[0]
    n = cnt[keep, None]
    B, Cm, Cp, wm = accB[keep] / n, accC[keep] / n, accP[keep] / n, pm[keep] / n
    Wt = wm * n
    Wt = Wt / Wt.sum()

    def cen(x):
        x = np.broadcast_to(x, B.shape)
        return x - (Wt * x).sum(1, keepdims=True) / Wt.sum(1, keepdims=True)

    y = cen(B)
    sst = (Wt * y * y).sum()
    lp1 = np.asarray(s.log_follow, np.float64)[keep]
    lu = np.log(s.p_uni)[None, :]

    def r2(F):
        X = np.stack([cen(f) for f in F.values()], -1)
        xtx = np.einsum("svk,sv,svl->kl", X, Wt, X)
        xty = np.einsum("svk,sv,sv->k", X, Wt, y)
        b = np.linalg.solve(xtx + 1e-9 * np.eye(len(F)), xty)
        return round(float(1 - (sst - 2 * b @ xty + b @ xtx @ b) / sst), 3), dict(zip(F, np.round(b, 3).tolist()))

    res = {"model": args.model, "n_windows": args.n, "n_tokens": int(len(keep)),
           "corpus": r2({"log_P1": lp1, "log_P": lu}),
           "corpus + model context mean (oracle)": r2({"log_P1": lp1, "log_P": lu, "Cm": Cm}),
           "corpus + program context mean": r2({"log_P1": lp1, "log_P": lu, "Cp": Cp}),
           "model context mean alone": r2({"Cm": Cm}),
           "corr(Cm, Cp) weighted": None}
    a, b = cen(Cm), cen(Cp)
    res["corr(Cm, Cp) weighted"] = round(float((Wt * a * b).sum() / np.sqrt((Wt * a * a).sum() * (Wt * b * b).sum())), 3)
    print(json.dumps(res, indent=1))
    with open(os.path.join(OUT, f"compensation_{args.model}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
