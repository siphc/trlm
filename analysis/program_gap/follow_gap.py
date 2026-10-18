"""How far is the program's current-token prediction from the model's? (analysis only; TRAIN windows)

The model's logits = E_i (every head's earlier-token part, exact) + S_i (direct path + every head's current-token
part, exact). Keep E_i exact and replace S_i by a linear combination of candidate directions, fitted (softmax
regression with offset E_i) on fit windows, evaluated on held-out windows:

  program: the program's current-token directions (FOLLOW bigram and unigram, LOCAL-0/LOCAL-2/RECALL self votes)
  oracle table: the model's own S averaged over the positions of each current token (in-sample; insight only),
                alone and with the program's directions
  oracle table x self weights: the same table plus the table scaled by each head's actual self weight
                (does the context-dependence of S matter?)

    .venv/bin/python -m analysis.program_gap.follow_gap [--fit 16 --eval 16]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from analysis.fitlib import train_windows
from analysis.program_gap.bottleneck import OUT, kl_rows
from analysis.program_gap.budget import model_parts, program_parts
from analysis.mech_eval import program

V = 2000


def collect(model, P, W):
    E, S, F, T, LP = [], [], [], [], []
    for w in W:
        t = np.asarray(w[:256], np.int64)
        mp, logits = model_parts(model, t)
        pp = program_parts(P, t)
        E.append(sum(mp[f"earlier{h}"] for h in range(4)).astype(np.float32))
        S.append((mp["direct"] + sum(mp[f"self{h}"] for h in range(4))).astype(np.float32))
        F.append(np.stack([pp[k] for k in ("follow", "local0.self", "local2.self", "recall.self")], -1).astype(np.float32))
        T.append(t)
        z = logits - logits.max(-1, keepdims=True)
        LP.append((z - np.log(np.exp(z).sum(-1, keepdims=True))).astype(np.float32))
    return E, S, F, T, LP


def fit_eval(off_f, X_f, P_f, off_e, X_e, LP_e, iters=150):
    Xf, Of, Pf = map(torch.as_tensor, (X_f, off_f, P_f))
    beta = torch.zeros(Xf.shape[-1], dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        beta[:] = 1.0 / Xf.shape[-1]
    opt = torch.optim.LBFGS([beta], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = 0.0
        for c in range(0, len(Xf), 512):
            sc = Of[c:c + 512] + Xf[c:c + 512] @ beta
            l = -(Pf[c:c + 512] * torch.log_softmax(sc, -1)).sum() / len(Xf)
            l.backward()
            loss += float(l)
        return torch.tensor(loss)

    opt.step(closure)
    b = beta.detach()
    kl = []
    with torch.no_grad():
        for c in range(0, len(X_e), 512):
            sc = torch.as_tensor(off_e[c:c + 512]) + torch.as_tensor(X_e[c:c + 512]) @ b
            lq = torch.log_softmax(sc, -1).numpy()
            kl.append(kl_rows(LP_e[c:c + 512].astype(np.float64), lq.astype(np.float64)))
    return float(np.concatenate(kl).mean()), b.numpy().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=int, default=16)
    ap.add_argument("--eval", type=int, default=16)
    args = ap.parse_args()
    from harness.common import load_model

    torch.set_num_threads(4)
    model = load_model("attn_s0")
    P = program()
    Ef, Sf, Ff, Tf, LPf = collect(model, P, train_windows(args.fit, seed=2024))
    Ee, Se, Fe, Te, LPe = collect(model, P, train_windows(args.eval, seed=2000))
    # oracle context-free table: mean centred S per current token over fit+eval positions
    acc, cnt = np.zeros((V, V)), np.zeros(V)
    for S, t in zip(Sf + Se, Tf + Te):
        np.add.at(acc, t, S - S.mean(-1, keepdims=True))
        np.add.at(cnt, t, 1)
    table = (acc / np.maximum(cnt, 1)[:, None]).astype(np.float32)
    cat = lambda xs: np.concatenate(xs)
    off_f, off_e = cat(Ef), cat(Ee)
    Pf = np.exp(cat(LPf))
    LP_e = cat(LPe)
    tf, te = cat(Tf), cat(Te)
    res = {"fit_windows": args.fit, "eval_windows": args.eval,
           "earlier parts only (S removed)": float(kl_rows(LP_e.astype(np.float64),
                                                            (off_e - np.logaddexp.reduce(off_e, -1, keepdims=True)).astype(np.float64)).mean()),
           "program as is (for reference)": None}
    arms = {"program current-token directions": (cat(Ff), cat(Fe)),
            "oracle table": (table[tf][..., None], table[te][..., None]),
            "oracle table + program directions": (np.concatenate([table[tf][..., None], cat(Ff)], -1),
                                                  np.concatenate([table[te][..., None], cat(Fe)], -1))}
    for name, (Xf, Xe) in arms.items():
        kl, b = fit_eval(off_f, Xf, Pf, off_e, Xe, LP_e)
        res[name] = {"kl": kl, "strengths": b}
        print(name, res[name], flush=True)
    res["exact S (check, should be 0)"] = float(kl_rows(LP_e.astype(np.float64), (lambda z: z - np.logaddexp.reduce(z, -1, keepdims=True))((off_e + cat(Se)).astype(np.float64))).mean())
    print(json.dumps(res, indent=1))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "follow_gap.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
