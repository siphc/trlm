"""Evaluate the mechanism program (program/) on TRAIN windows: ledger, consistency check, claim measurements.

The program's corpus tables (program.corpus.build, ~2 min) are cached in analysis/out/program_stats.pkl for
analysis only; the program itself always rebuilds them in setup().

  python -m analysis.mech_eval ledger        # KL with all mechanisms and with each one switched off
  python -m analysis.mech_eval consistency   # program part removal vs the matching model-head ablation
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np

from analysis.fitlib import REPO, train_windows

STATS_CACHE = os.path.join(REPO, "analysis", "out", "program_stats.pkl")
EVAL_SEED = 2000            # same TRAIN windows as analysis/paired.py
HEAD_OF = {"head0": 0, "head1": 1, "head2": 2}


def stats():
    if os.path.exists(STATS_CACHE):
        with open(STATS_CACHE, "rb") as f:
            return pickle.load(f)
    from harness.common import load_corpus
    from program import corpus

    t0 = time.time()
    s = corpus.build(load_corpus())
    with open(STATS_CACHE, "wb") as f:
        pickle.dump(s, f, protocol=4)
    print(f"corpus tables built in {time.time() - t0:.0f}s -> {STATS_CACHE}", flush=True)
    return s


def program(s=None):
    import program as P

    P.STATS = s if s is not None else stats()
    return P


def model_logprobs(windows, ablate_heads=(), ablation="zero", means=None):
    from harness.common import load_model, model_logprobs as mlp

    global _MODEL
    if "_MODEL" not in globals():
        _MODEL = load_model("attn_s0")
    return mlp(_MODEL, windows, ablate_heads=tuple(ablate_heads), ablation=ablation, means=means)


def kl(p_log, q_log):
    p = np.exp(p_log)
    return np.where(p > 0, p * (p_log - q_log), 0.0).sum(-1)


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def earlier_votes(name, tokens, s):
    """The part of a head's votes that comes from tokens before the current one."""
    import program.head0, program.head1, program.head2
    from program.readers import skipgram_votes

    mod = {"head0": program.head0, "head1": program.head1, "head2": program.head2}[name]
    a = mod.attention(tokens, s)
    if name == "head1":
        a = a * (1 - np.eye(len(tokens)))
        g = mod.GROUP[tokens]
        src = (mod._per_group(mod.TOPIC)[g][:, None] * s.topic[tokens]
               + mod._per_group(mod.NEXT)[g][:, None] * s.pmi[1][tokens])
        src[np.arange(len(tokens)), tokens] += mod._per_group(mod.COPY)[g]
        return a @ src
    return skipgram_votes(a, tokens, s, 0.0, mod.STRENGTH_NEAR, mod.STRENGTH_FAR)


def ledger(n=64):
    P = program()
    W = train_windows(n, seed=EVAL_SEED)
    lp = model_logprobs(W)
    names = [m.NAME for m in P.MECHANISMS]
    per = {"all": [], **{f"without {m}": [] for m in names}, "follow only": []}
    for w, p in zip(W, lp):
        t = np.asarray(w[:256])
        per["all"].append(kl(p, P.predict(t)).mean())
        for m in names:
            per[f"without {m}"].append(kl(p, P.predict(t, off=(m,))).mean())
        per["follow only"].append(kl(p, P.predict(t, off=tuple(x for x in names if x != "follow"))).mean())
    out = {k: {"kl": float(np.mean(v)), "se": float(np.std(v, ddof=1) / np.sqrt(len(v)))} for k, v in per.items()}
    base = np.asarray(per["all"])
    for m in names:
        d = np.asarray(per[f"without {m}"]) - base
        out[f"without {m}"].update(rise=float(d.mean()), rise_se=float(d.std(ddof=1) / np.sqrt(len(d))))
    return {"n_windows": n, "seed": EVAL_SEED, "ledger": out}


def consistency(n=64):
    """For each head: how far the output moves when its earlier-token part is replaced by its mean, in the model
    (harness 'earlier' ablation) and in the program; and whether the two moves point the same way."""
    from harness.common import head_means, load_model

    P = program()
    s = P.STATS
    W = train_windows(n, seed=EVAL_SEED)
    means = head_means(load_model("attn_s0"), W)
    lp = model_logprobs(W)
    res = {}
    for name, h in HEAD_OF.items():
        lp_abl = model_logprobs(W, ablate_heads=[h], ablation="earlier", means=means)
        ev = [earlier_votes(name, np.asarray(w[:256]), s) for w in W]
        mean_ev = np.mean([e.mean(0) for e in ev], axis=0)
        shift_m, shift_p, corr, slope = [], [], [], []
        for w, p, pa, e in zip(W, lp, lp_abl, ev):
            t = np.asarray(w[:256])
            q = P.predict(t)
            qa = log_softmax(q - e + mean_ev)
            shift_m.append(kl(p, pa).mean())
            shift_p.append(kl(q, qa).mean())
            dm, dp = p - pa, q - qa
            dm = dm - dm.mean(-1, keepdims=True)
            dp = dp - dp.mean(-1, keepdims=True)
            # probability-weighted correlation over the vocabulary, per position (model's distribution as weight)
            wgt = np.exp(p)
            cm = (wgt * dm * dp).sum(-1)
            vm, vp = (wgt * dm * dm).sum(-1), (wgt * dp * dp).sum(-1)
            corr.append(np.median(cm / np.sqrt(vm * vp + 1e-30)))
            slope.append(np.median(cm / (vp + 1e-30)))
        res[name] = {"model_head": h, "model_shift": float(np.mean(shift_m)), "program_shift": float(np.mean(shift_p)),
                     "median_weighted_corr": float(np.mean(corr)), "median_slope_model_on_program": float(np.mean(slope))}
        print(name, res[name], flush=True)
    return {"n_windows": n, "seed": EVAL_SEED, "heads": res}


if __name__ == "__main__":
    what = sys.argv[1]
    out = {"ledger": ledger, "consistency": consistency}[what]()
    path = os.path.join(REPO, "analysis", "out", f"mech_{what}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
