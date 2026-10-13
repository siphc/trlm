"""Fit B_WORD for head 1's attention term  B_WORD * gate(t_i) * max(0, PMI_1(t_i, t_j))  (j < i) to head 1's OWN
attention (32 fit windows, seed 2024; pattern KL on 32 eval windows, seed 2000). gate = cont(t_i) (chance the next
token is a word piece, corpus), the onset indicator, or startcont = (1 - piece(t_i)) * cont(t_i). Also with an induction term gated the same way."""
import json
import os

import numpy as np
import torch

from analysis.fitlib import train_windows
from analysis.investigations.word import wordstats as WS
from analysis.investigations.word.h1_attn import collect, pattern_kl, FIT_SEED
from analysis.mech_eval import EVAL_SEED, program
from harness.common import load_model

P = program()
PIECE = WS.piece_flags()
model = load_model("attn_s0")
Wf, We = train_windows(32, seed=FIT_SEED), train_windows(32, seed=EVAL_SEED)
fitd, evald = collect(P, model, Wf, PIECE), collect(P, model, We, PIECE)
cont = np.asarray(P.STATS.cont)


def gates(W, data, kind):
    out = []
    for w, (A, base, succ, ind, cl) in zip(W, data):
        t = np.asarray(w[:256])
        if kind == "cont":
            g = cont[t]
        elif kind == "startcont":
            g = cont[t] * (1 - np.asarray(P.STATS.piece)[t])
        else:
            g = np.array([c == "onset" for c in cl], float)
        out.append(g[:, None])
    return out


def fit(data, G, use_ind):
    b = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    mask = torch.as_tensor(np.tril(np.ones((256, 256), bool)))
    tens = [(torch.as_tensor(A), torch.as_tensor(base), torch.as_tensor(succ * g), torch.as_tensor(ind * g))
            for (A, base, succ, ind, cl), g in zip(data, G)]
    opt = torch.optim.LBFGS([b], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = 0.0
        for A, base, s, i in tens:
            sc = base + b[0] * s + (b[1] * i if use_ind else 0)
            sc = sc.masked_fill(~mask, float("-inf"))
            loss = loss - (A * torch.log_softmax(sc, 1).masked_fill(~mask, 0)).sum() / (256 * len(tens))
        loss.backward()
        return loss

    opt.step(closure)
    return b.detach().numpy() * np.array([1, 1 if use_ind else 0])


res = {"base": pattern_kl(evald, 0, 0)}
print("base", {k: round(v, 3) for k, v in sorted(res["base"].items())})
for kind in ("cont", "onset", "startcont"):
    Gf, Ge = gates(Wf, fitd, kind), gates(We, evald, kind)
    for use_ind in (False, True):
        b = fit(fitd, Gf, use_ind)
        ev = [(A, base + b[0] * succ * g + b[1] * ind * g, succ, ind, cl) for (A, base, succ, ind, cl), g in zip(evald, Ge)]
        k = pattern_kl(ev, 0, 0)
        res[f"{kind}{'+ind' if use_ind else ''}"] = {"b": b.tolist(), "kl": k}
        print(kind, "+ind" if use_ind else "", np.round(b, 3), {c: round(v, 3) for c, v in sorted(k.items())}, flush=True)
with open(os.path.join(WS.OUT, "h1_fit_word.json"), "w") as f:
    json.dump(res, f, indent=1)
