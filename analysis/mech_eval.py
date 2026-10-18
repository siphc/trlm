"""Evaluate the mechanism program (program/) on TRAIN windows: ledger, consistency check, claim measurements.

The program's corpus tables (program.corpus.build, ~2 min) are cached in analysis/out/program_stats/ for analysis
only; the program itself always rebuilds them in setup().

  python -m analysis.mech_eval ledger        # KL with all mechanisms and with each one switched off
  python -m analysis.mech_eval consistency   # program part removal vs the matching model-head ablation
  python -m analysis.mech_eval claims        # the claim-test statistics, on TRAIN windows
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from analysis.fitlib import REPO, train_windows

STATS_DIR = os.path.join(REPO, "analysis", "out", "program_stats")   # one .npy per table, memory-mapped
EVAL_SEED = 2000            # same TRAIN windows as analysis/paired.py
HEAD_OF = {"head0": 0, "head1": 1, "head2": 2}
MODEL_ID = os.environ.get("TRLM_ANALYSIS_MODEL", "attn_s0")   # the model analysed (an id or a checkpoint path)


def stats():
    """The program's corpus tables (program.corpus.build), cached as memory-mapped .npy files so that several
    analysis processes share one copy in memory."""
    from program.corpus import Stats

    if not os.path.isdir(STATS_DIR):
        from harness.common import load_corpus
        from program import corpus

        t0 = time.time()
        s = corpus.build(load_corpus())
        os.makedirs(STATS_DIR + ".tmp", exist_ok=True)
        for k, v in vars(s).items():
            if k == "pmi":
                for d, arr in v.items():
                    np.save(os.path.join(STATS_DIR + ".tmp", f"pmi_{d}.npy"), arr)
            else:
                np.save(os.path.join(STATS_DIR + ".tmp", f"{k}.npy"), np.asarray(v))
        os.rename(STATS_DIR + ".tmp", STATS_DIR)
        print(f"corpus tables built in {time.time() - t0:.0f}s -> {STATS_DIR}", flush=True)
    s = Stats()
    s.pmi = {}
    for f in sorted(os.listdir(STATS_DIR)):
        arr = np.load(os.path.join(STATS_DIR, f), mmap_mode="r")
        name = f[:-4]
        if name.startswith("pmi_"):
            s.pmi[int(name[4:])] = arr
        elif arr.ndim == 0:
            setattr(s, name, arr.item())
        else:
            setattr(s, name, arr)
    return s


def program(s=None):
    import program as P

    P.STATS = s if s is not None else stats()
    return P


def model_logprobs(windows, ablate_heads=(), ablation="zero", means=None):
    from harness.common import load_model, model_logprobs as mlp

    global _MODEL
    if "_MODEL" not in globals():
        _MODEL = load_model(MODEL_ID)
    return mlp(_MODEL, windows, ablate_heads=tuple(ablate_heads), ablation=ablation, means=means)


def kl(p_log, q_log):
    p = np.exp(p_log)
    return np.where(p > 0, p * (p_log - q_log), 0.0).sum(-1)


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


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
    from program.tests import measure as M

    P = program()
    W = train_windows(n, seed=EVAL_SEED)
    means = head_means(load_model(MODEL_ID), W)
    lp = model_logprobs(W)
    res = {}
    for name, h in HEAD_OF.items():
        lp_abl = model_logprobs(W, ablate_heads=[h], ablation="earlier", means=means)
        res[name] = M.compare_with_head(P, W, lp, lp_abl, lambda t: M.earlier_votes(P, name, t))
        res[name]["uniform attention"] = M.compare_with_head(
            P, W, lp, lp_abl, lambda t: M.earlier_votes(P, name, t, attention=M.uniform_attention(len(t))))
        print(name, res[name], flush=True)
    return {"n_windows": n, "seed": EVAL_SEED, "heads": res}


def claims(n=64):
    """Every statistic the claim tests check, on TRAIN windows (to set their thresholds)."""
    from harness.common import head_means, load_model
    from program.tests import measure as M

    P = program()
    W = train_windows(n, seed=EVAL_SEED)
    means = head_means(load_model(MODEL_ID), W)
    lp = model_logprobs(W)
    abl = {k: model_logprobs(W, ablate_heads=list(hs), ablation="earlier", means=means)
           for k, hs in {"h0": [0], "h1": [1], "h2": [2], "h3": [3], "h02": [0, 2], "all": [0, 1, 2, 3]}.items()}
    out = {"removal_rise": M.removal_rise(P, W, lp, [m.NAME for m in P.MECHANISMS])}
    for name, h in HEAD_OF.items():
        out[name] = M.compare_with_head(P, W, lp, abl[f"h{h}"], lambda t: M.earlier_votes(P, name, t))
        out[name]["uniform"] = M.compare_with_head(
            P, W, lp, abl[f"h{h}"], lambda t: M.earlier_votes(P, name, t, attention=M.uniform_attention(len(t))))
        print(name, out[name], flush=True)
    import program.phrase as phrase
    out["phrase_agreement"] = {k: M.vote_agreement(W, lp, abl[k], lambda t: phrase.votes(t, P.STATS))
                               for k in ["h02", "h0", "h2", "h1", "h3", "all"]}
    print("phrase", out["phrase_agreement"], flush=True)
    follow_only = tuple(m.NAME for m in P.MECHANISMS if m.NAME != "follow")
    toks = [np.asarray(w[:256], dtype=np.int64) for w in W]
    q = [P.predict(t, off=follow_only) for t in toks]
    out["follow"] = {"kl_to_model": float(np.mean([kl(p, qq).mean() for p, qq in zip(lp, q)])),
                     "kl_to_context_ablated_model": float(np.mean([kl(p, qq).mean() for p, qq in zip(abl["all"], q)]))}
    print("follow", out["follow"], flush=True)
    return {"n_windows": n, "seed": EVAL_SEED, **out}


if __name__ == "__main__":
    what = sys.argv[1]
    out = {"ledger": ledger, "consistency": consistency, "claims": claims}[what]()
    path = os.path.join(REPO, "analysis", "out", f"mech_{what}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
