"""Error budget of the program by part (goal 2, analysis only; TRAIN windows).

The model's logits split exactly into additive parts once the final LayerNorm's per-position scale sigma_i is fixed
at its actual value: logits_i = sum_c (g * (c_i - mean c_i) / sigma_i) @ W_E^T over c = direct path (token +
position embedding) and, per head h, its current-token part a_ii O_h v_i and its earlier-token part.
The program's score splits into its mechanisms. Groups that should correspond:

  FOLLOW   model: direct + every head's current-token part   program: FOLLOW + LOCAL-0/LOCAL-2/RECALL current-token
  LOCAL    model: heads 0, 2, 3 earlier-token parts            program: LOCAL-0/LOCAL-2 earlier votes + PHRASE
  RECALL   model: head 1 earlier-token part                    program: RECALL earlier votes

Hybrid_g = model logits - model part g + program part g. KL(model || hybrid_g) is the cost of the program's error in
group g with everything else exact. Also: each model head's earlier part zeroed / replaced by its program part.

    .venv/bin/python -m analysis.goal2.budget [--n 16]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from analysis.fitlib import train_windows
from analysis.goal2.bottleneck import OUT, kl_rows
from analysis.mech_eval import EVAL_SEED, program

V = 2000


def model_parts(model, toks):
    """(T, V) logit parts for one window: direct, self[h], earlier[h]; and the exact logits."""
    from harness.common import _attention_parts

    x = torch.as_tensor(toks[None], dtype=torch.int64)
    with torch.no_grad():
        x0, z, sp = _attention_parts(model, x)
        Wo = model.attn_out_weight if hasattr(model, "attn_out_weight") else model.blocks[0].attn.out.weight
        H, Dh = z.shape[2], z.shape[3]
        parts = {"direct": x0[0].double()}
        for h in range(H):
            Woh = Wo[:, h * Dh:(h + 1) * Dh].double()
            parts[f"self{h}"] = sp[0, :, h].double() @ Woh.T
            parts[f"earlier{h}"] = (z[0, :, h] - sp[0, :, h]).double() @ Woh.T
        total = sum(parts.values())
        mu = total.mean(-1, keepdim=True)
        sigma = torch.sqrt(((total - mu) ** 2).mean(-1, keepdim=True) + model.ln_f.eps)
        g = model.ln_f.weight.double()
        E = model.wte.weight.double()
        out = {k: (((c - c.mean(-1, keepdim=True)) / sigma * g) @ E.T).numpy() for k, c in parts.items()}
        logits = model(x)[0].double().numpy()
    return out, logits


def program_parts(P, t):
    from program import follow, head0, head1, head2, phrase
    from program.readers import skipgram_votes

    s = P.STATS
    n = len(t)
    out = {"follow": follow.votes(t, s)}
    for name, mod in (("local0", head0), ("local2", head2)):
        a = mod.attention(t, s)
        out[f"{name}.self"] = skipgram_votes(a, t, s, mod.STRENGTH_SELF, 0.0, 0.0)
        out[f"{name}.earlier"] = mod.earlier_votes(a, t, s)
    a1 = head1.attention(t, s)
    rv = head1.recalled_votes(t, s)
    out["recall.self"] = np.diagonal(a1)[:, None] * rv
    out["recall.earlier"] = (a1 * (1 - np.eye(n))) @ rv
    out["phrase"] = phrase.votes(t, s)
    return out


def lsm(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()
    from harness.common import load_model

    torch.set_num_threads(3)
    model = load_model("attn_s0")
    P = program()
    W = train_windows(args.n, seed=EVAL_SEED)
    groups = {"FOLLOW": (["direct", "self0", "self1", "self2", "self3"], ["follow", "local0.self", "local2.self",
                                                                           "recall.self"]),
              "LOCAL": (["earlier0", "earlier2", "earlier3"], ["local0.earlier", "local2.earlier", "phrase"]),
              "LOCAL_no_phrase": (["earlier0", "earlier2", "earlier3"], ["local0.earlier", "local2.earlier"]),
              "RECALL": (["earlier1"], ["recall.earlier"])}
    MP, PP, LP = [], [], []
    for w in W:
        t = np.asarray(w[:256], np.int64)
        mp, logits = model_parts(model, t)
        MP.append({k: v.astype(np.float32) for k, v in mp.items()})
        PP.append({k: v.astype(np.float32) for k, v in program_parts(P, t).items()})
        LP.append(lsm(logits))
    # constant (position-independent) part of every part goes to one BIAS group per side
    mbar = {k: np.mean([m[k].mean(0) for m in MP], 0) for k in MP[0]}
    pbar = {k: np.mean([p[k].mean(0) for p in PP], 0) for k in PP[0]}
    acc = {}

    def add(key, val):
        acc.setdefault(key, []).append(val)

    for mp, pp, lp in zip(MP, PP, LP):
        L = sum(mp.values())
        Q = sum(pp.values())
        add("program", kl_rows(lp, lsm(Q)).mean())
        mc = {k: mp[k] - mbar[k] for k in mp}
        pc = {k: pp[k] - pbar[k] for k in pp}
        mb, pb = sum(mbar.values()), sum(pbar.values())

        def hybrid(gs, bias=False):
            H = L.astype(np.float64)
            for g in gs:
                H = H - sum(mc[c] for c in groups[g][0]) + sum(pc[c] for c in groups[g][1])
            if bias:
                H = H - mb + pb
            return kl_rows(lp, lsm(H)).mean()

        add("hybrid BIAS only", hybrid([], bias=True))
        for g in groups:
            add(f"hybrid {g} (centred)", hybrid([g]))
        add("hybrid LOCAL+RECALL (centred)", hybrid(["LOCAL", "RECALL"]))
        add("hybrid FOLLOW+LOCAL (centred)", hybrid(["FOLLOW", "LOCAL"]))
        add("hybrid FOLLOW+RECALL (centred)", hybrid(["FOLLOW", "RECALL"]))
        add("hybrid all (centred) + BIAS", hybrid(["FOLLOW", "LOCAL", "RECALL"], bias=True))
        for c in ["earlier0", "earlier1", "earlier2", "earlier3"]:
            add(f"model with {c} at its mean", kl_rows(lp, lsm(L - mc[c])).mean())
        for c in ["self0", "self1", "self2", "self3"]:
            add(f"model with {c} at its mean", kl_rows(lp, lsm(L - mc[c])).mean())
    res = {k: {"kl": float(np.mean(v)), "se": float(np.std(v, ddof=1) / np.sqrt(len(v)))} for k, v in acc.items()}
    res["n_windows"] = args.n
    print(json.dumps(res, indent=1))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "budget.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
