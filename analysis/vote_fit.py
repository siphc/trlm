"""Test a candidate mechanism against the current program, with every vote strength refitted.

Given its attention rules, the program's score is linear in its vote strengths:
score_i[v] = sum_k beta_k F_k(i, v), with one vote direction F_k per named strength (25 in v1: FOLLOW 2, LOCAL-0 3,
LOCAL-2 3, RECALL 15, PHRASE 2). Fitting the strengths to the model is therefore a convex problem (a softmax
regression on soft targets), solved here by L-BFGS. A candidate adds directions (and may remove some). Both arms
are refitted on the same fit windows and compared, paired, on held-out TRAIN windows.

A candidate is a Python file defining

    NAME = "short_name"
    HEADS = [0, 2]                 # model heads it claims to be part of (for the head-effect check)
    REPLACES = []                  # base direction names it removes (optional)
    def prepare(stats, corpus):    # optional: count extra corpus statistics once (cache under analysis/out/candidates/)
        ...
    def features(P, tokens):       # -> {direction name: (n, V) array}; P.STATS holds the program's corpus tables
        ...

Run: python -m analysis.vote_fit analysis/candidates/<file>.py [--fit 32] [--eval 64] [--no_heads]
         [--fit_seed 2024] [--eval_seed 2000] [--tag _suffix]
(When a candidate REPLACES directions of a program part, the head-effect check drops that whole part from the
'program+candidate' set, so the candidate's directions must carry the part's replaced role. Candidate directions
named *.self are current-token votes and are left out of the head-effect check, which is about earlier tokens.)
Writes analysis/out/candidates/<NAME>.json. Adoption rule (logs/decisions.md item 3): paired gain >= 0.005 and
>= 3 SE (>= 0.01 for a new component); for "explain, don't fit" the candidate should also raise the share of its
heads' effect that the program explains (R^2 below).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time

import numpy as np
import torch

from analysis.fitlib import REPO, train_windows
from analysis.mech_eval import EVAL_SEED, model_logprobs, program
from program.tests import measure as M

FIT_SEED = 2024
OUT = os.path.join(REPO, "analysis", "out", "candidates")
V = 2000


def base_features(P, t):
    """The program's vote directions (28 in v1.1), each with its strength factored out; and the strengths."""
    from program import follow, head0, head1, head2, phrase
    from program.readers import earlier_only, mass_on_tokens, skipgram_votes

    s = P.STATS
    n = len(t)
    F, beta = {}, {}
    F["follow.bigram"], beta["follow.bigram"] = np.asarray(s.log_follow[t], dtype=np.float32), follow.STRENGTH
    F["follow.unigram"], beta["follow.unigram"] = np.broadcast_to(np.asarray(s.log_uni, np.float32), (n, V)), follow.UNIGRAM
    for name, mod in (("head0", head0), ("head2", head2)):
        a = mod.attention(t, s)
        for part, unit, strength in (("self", (1, 0, 0), mod.STRENGTH_SELF), ("near", (0, 1, 0), mod.STRENGTH_NEAR),
                                     ("far", (0, 0, 1), mod.STRENGTH_FAR)):
            F[f"{name}.{part}"], beta[f"{name}.{part}"] = skipgram_votes(a, t, s, *unit), strength
        earlier = earlier_only(a)
        if name == "head0":
            F["head0.earlier_successors"], beta["head0.earlier_successors"] = (
                earlier @ np.asarray(s.pmi[1][t], np.float64), head0.EARLIER_SUCCESSORS)
            F["head0.earlier_alike"], beta["head0.earlier_alike"] = (
                earlier @ np.asarray(s.slot_affinity[t], np.float64), head0.EARLIER_ALIKE)
        else:
            F["head2.no_repeat"], beta["head2.no_repeat"] = mass_on_tokens(earlier, t), head2.NO_REPEAT
    a1 = head1.attention(t, s)
    g = head1.GROUP[t]
    topic, nxt = np.asarray(s.topic[t], np.float64), np.asarray(s.pmi[1][t], np.float64)
    for k, gname in enumerate(head1.GROUP_NAMES):
        A = a1 * (g == k)[None, :]
        copy = np.zeros((n, V))
        np.add.at(copy, (np.repeat(np.arange(n), n), np.tile(t, n)), A.reshape(-1))
        F[f"head1.copy.{gname}"], beta[f"head1.copy.{gname}"] = copy, head1.COPY[gname]
        F[f"head1.topic.{gname}"], beta[f"head1.topic.{gname}"] = A @ topic, head1.TOPIC[gname]
        F[f"head1.next.{gname}"], beta[f"head1.next.{gname}"] = A @ nxt, head1.NEXT[gname]
    tri, quad = phrase.parts(t, s)
    F["phrase.trigram"], beta["phrase.trigram"] = tri, phrase.TRIGRAM
    F["phrase.fourgram"], beta["phrase.fourgram"] = quad, phrase.FOURGRAM
    return F, beta


def load_candidate(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def all_features(P, t, cand):
    F, beta = base_features(P, t)
    for k in getattr(cand, "REPLACES", []) if cand else []:
        F.pop(k)
        beta.pop(k)
    if cand is not None:
        extra = cand.features(P, t)
        for k, v in extra.items():
            F[f"{cand.NAME}.{k}"] = v
            beta[f"{cand.NAME}.{k}"] = 0.0
    return F, beta


SATURATING = "head1."      # directions of the mechanism whose summed vote has a fixed size (program/head1.py SIZE)


def saturation_of_program():
    """The size of RECALL's vote (program/head1.py SIZE), or None if this program version has none."""
    from program import head1

    return getattr(head1, "SIZE", None)


RECALL_RMS = 3.0           # scale convention for the fixed-size mechanism's strengths (only their ratios matter)


def sat_mask(keys):
    return torch.tensor([k.startswith(SATURATING) for k in keys])


def combine(X, beta, mask=None, c=None, pu=None):
    """Score from directions X (..., K): sum_k beta_k X_k; with c, the fixed-size mechanism's part is rescaled to
    size c: c * g / spread(g), spread = unigram-weighted standard deviation over the vocabulary (program/head1.py)."""
    if c is None:
        return X @ beta
    g = X[..., mask] @ beta[mask]
    mu = (g * pu).sum(-1, keepdim=True)
    n2 = (((g - mu) ** 2) * pu).sum(-1, keepdim=True)
    return X[..., ~mask] @ beta[~mask] + c * g / torch.sqrt(n2 + 1e-18)


def fit(stack, probs, beta0, iters=200, keys=None, c0=None, pu=None):
    """L-BFGS on mean cross-entropy of the model's distribution under softmax(F beta). stack (N, V, K) float16.
    With c0 (and keys, pu), the fixed-size mechanism's size is fitted too and (beta, c) is returned; its
    strengths are then defined only up to a common factor and are rescaled to root-mean-square RECALL_RMS."""
    X = torch.from_numpy(stack)
    Pm = torch.from_numpy(probs)
    N = X.shape[0]
    beta = torch.tensor(beta0, dtype=torch.float32, requires_grad=True)
    params = [beta]
    mask = logc = put = None
    if c0 is not None:
        mask = sat_mask(keys)
        logc = torch.tensor(float(np.log(c0)), requires_grad=True)
        put = torch.as_tensor(np.asarray(pu), dtype=torch.float32)
        params.append(logc)
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-10, history_size=50)

    def closure():
        opt.zero_grad()
        total = 0.0
        for c in range(0, N, 256):
            score = combine(X[c : c + 256].float(), beta, mask, None if logc is None else torch.exp(logc), put)
            loss = -(Pm[c : c + 256] * torch.log_softmax(score, -1)).sum() / N
            loss.backward()
            total += float(loss.detach())
        return torch.tensor(total)

    opt.step(closure)
    b = beta.detach().numpy().astype(np.float64)
    if c0 is None:
        return b
    m = mask.numpy()
    b[m] *= RECALL_RMS / np.sqrt(np.mean(b[m] ** 2))
    return b, float(torch.exp(logc.detach()))


def stacked(P, windows, cand, names=None):
    rows, keys = [], None
    for w in windows:
        F, beta = all_features(P, np.asarray(w[:256], dtype=np.int64), cand)
        keys = keys or list(F)
        rows.append(np.stack([np.asarray(F[k], np.float32) for k in keys], -1).astype(np.float16))
    return np.concatenate(rows), keys, beta


def evaluate(P, windows, lp, cand, betas):
    """Per-window KL for each named strength vector in `betas` ({arm: (keys, beta) or (keys, beta, saturation)})."""
    out = {arm: [] for arm in betas}
    pu = torch.as_tensor(np.asarray(P.STATS.p_uni), dtype=torch.float64)
    for w, p in zip(windows, lp):
        t = np.asarray(w[:256], dtype=np.int64)
        F_base, _ = base_features(P, t)
        F_cand = all_features(P, t, cand)[0] if cand is not None else F_base
        for arm, spec in betas.items():
            keys, beta, c = (spec + (None,))[:3]
            F = F_cand if arm.startswith("cand") else F_base
            X = torch.as_tensor(np.stack([np.asarray(F[k], np.float64) for k in keys], -1))
            score = combine(X, torch.as_tensor(np.asarray(beta, np.float64)), sat_mask(keys) if c is not None else None,
                            c, pu).numpy()
            out[arm].append(M.kl(p, M.log_softmax(score)).mean())
    return {arm: np.asarray(v) for arm, v in out.items()}


def head_r2(P, windows, lp, cand, heads):
    """Pooled weighted R^2 of the model heads' earlier-token effect on the program parts' directions, without and
    with the candidate's directions."""
    from harness.common import head_means, load_model
    from program import phrase

    means = head_means(load_model("attn_s0"), windows)
    lpa = model_logprobs(windows, ablate_heads=heads, ablation="earlier", means=means)
    feats = []
    for w in windows:
        t = np.asarray(w[:256], dtype=np.int64)
        f = {"local0": M.earlier_votes(P, "head0", t), "local2": M.earlier_votes(P, "head2", t),
             "recall": M.earlier_votes(P, "head1", t), "phrase": phrase.votes(t, P.STATS)}
        # current-token directions (named *.self) are left out: the earlier-token ablation does not touch them
        f.update({f"cand.{k}": np.asarray(v, np.float64) for k, v in cand.features(P, t).items()
                  if not k.endswith(".self")})
        feats.append(f)
    for key in feats[0]:
        mu = np.mean([f[key].mean(0) for f in feats], axis=0)
        for f in feats:
            f[key] = f[key] - mu
    base_keys = ["local0", "local2", "recall", "phrase"]
    part_of = {"head0": "local0", "head2": "local2", "head1": "recall", "phrase": "phrase"}
    replaced = {part_of[k.split(".")[0]] for k in getattr(cand, "REPLACES", []) if k.split(".")[0] in part_of}
    cand_keys = [k for k in base_keys if k not in replaced] + [k for k in feats[0] if k.startswith("cand.")]
    res = {}
    for label, ks in (("program", base_keys), ("program+candidate", cand_keys),
                      ("candidate alone", [k for k in cand_keys if k.startswith("cand.")])):
        xtx, xty, sst = np.zeros((len(ks), len(ks))), np.zeros(len(ks)), 0.0
        for p, pa, f in zip(lp, lpa, feats):
            w = np.exp(p)
            y = p - pa
            y = y - (w * y).sum(1, keepdims=True)
            X = np.stack([f[k] for k in ks], -1)
            X = X - (w[..., None] * X).sum(1, keepdims=True)
            xtx += np.einsum("nvk,nv,nvl->kl", X, w, X)
            xty += np.einsum("nvk,nv,nv->k", X, w, y)
            sst += (w * y * y).sum()
        beta = np.linalg.lstsq(xtx, xty, rcond=None)[0]
        res[label] = float(1 - (sst - 2 * beta @ xty + beta @ xtx @ beta) / sst)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", nargs="?", help="candidate file; omit to check the base features")
    ap.add_argument("--fit", type=int, default=32)
    ap.add_argument("--eval", type=int, default=64)
    ap.add_argument("--heads_windows", type=int, default=16)
    ap.add_argument("--no_heads", action="store_true")
    ap.add_argument("--fit_seed", type=int, default=FIT_SEED)
    ap.add_argument("--eval_seed", type=int, default=EVAL_SEED)
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("VOTE_FIT_THREADS", "2")))
    t0 = time.time()
    P = program()
    cand = load_candidate(args.candidate) if args.candidate else None
    if cand is not None and hasattr(cand, "prepare"):
        from harness.common import load_corpus
        cand.prepare(P.STATS, load_corpus())
    Wf, We = train_windows(args.fit, seed=args.fit_seed), train_windows(args.eval, seed=args.eval_seed)
    lpf = model_logprobs(Wf)
    probs = np.exp(lpf).reshape(-1, V).astype(np.float32)
    sat0 = saturation_of_program()
    pu = P.STATS.p_uni
    Xb, kb, b0 = stacked(P, Wf, None)
    if sat0 is None:
        beta_base, sat_base = fit(Xb, probs, [b0[k] for k in kb]), None
    else:
        beta_base, sat_base = fit(Xb, probs, [b0[k] for k in kb], keys=kb, c0=sat0, pu=pu)
    del Xb
    betas = {"program as is": (kb, [b0[k] for k in kb], sat0), "base refit": (kb, beta_base, sat_base)}
    if cand is not None:
        Xc, kc, c0 = stacked(P, Wf, cand)
        init = [beta_base[kb.index(k)] if k in kb else 0.0 for k in kc]
        if sat0 is None:
            beta_c, sat_c = fit(Xc, probs, init), None
        else:
            beta_c, sat_c = fit(Xc, probs, init, keys=kc, c0=sat_base, pu=pu)
        del Xc
        betas["candidate refit"] = (kc, beta_c, sat_c)
    lpe = model_logprobs(We)
    kls = evaluate(P, We, lpe, cand, betas)
    res = {"candidate": getattr(cand, "NAME", None), "fit_windows": args.fit, "eval_windows": args.eval,
           "fit_seed": args.fit_seed, "eval_seed": args.eval_seed,
           "kl": {arm: float(v.mean()) for arm, v in kls.items()}}
    if cand is not None:
        d = kls["candidate refit"] - kls["base refit"]
        res["paired_diff"] = float(d.mean())
        res["paired_se"] = float(d.std(ddof=1) / np.sqrt(len(d)))
        res["candidate_strengths"] = {k: round(float(b), 4) for k, b in zip(*betas["candidate refit"][:2])
                                      if k.startswith(cand.NAME + ".")}
        res["base_strengths_change"] = {k: [round(float(beta_base[kb.index(k)]), 3), round(float(b), 3)]
                                        for k, b in zip(*betas["candidate refit"][:2]) if k in kb}
        res["saturation"] = {"base": sat_base, "candidate": betas["candidate refit"][2]}
        if not args.no_heads and getattr(cand, "HEADS", None):
            Wh = We[: args.heads_windows]
            res["head_effect_r2"] = {"heads": cand.HEADS, **head_r2(P, Wh, lpe[: args.heads_windows], cand, cand.HEADS)}
    else:
        res["base_strengths"] = {k: [round(float(b0[k]), 3), round(float(b), 3)] for k, b in zip(kb, beta_base)}
        res["saturation"] = {"program": sat0, "refit": sat_base}
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{res['candidate'] or 'base'}{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
