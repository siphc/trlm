"""Copying diagnostic (plan of 2026-11-06, step 2a; analysis only, uses attn_s0's weights as an oracle; TRAIN windows).

58% of the program's miss against the five-seed consensus is at positions whose consensus top token appeared earlier in
the window (consensus.py). Is that miss in where RECALL (model head 1) looks, or in what it writes?

Part 1, swaps inside the program (every arm's strengths refitted to attn_s0 as in
analysis/token_geometry/tables_vs_attention.py):
    A        the program (readable attention + corpus votes)
    A_att1   head 1's exact attention, RECALL's corpus votes
    A_tab1   RECALL's readable attention, head 1's exact vote table (OV_1, self and earlier parts)
    A_both1  head 1 exact (attention and table) inside the otherwise readable program
    (--arms: the same three swaps for any head h as A_att<h>, A_tab<h>, A_both<h>; A_cur: the model's direct-path
    table instead of FOLLOW's bigram)
    D        the model's own decomposition (exact attention and tables everywhere)
Each is scored per position against attn_s0 and against ENS5 (the probability average of attn_s0..s4), by class:
    copy_agree      the consensus top token appeared earlier in the window, seed spread below the median
    copy_disagree   same, spread at or above the median
    piece           the consensus top token is a word piece (and not copy)
    rest
Part 2, attention on the copy target: at copy positions, the attention each seed's RECALL head (aligned checkpoints,
analysis/readable_training/models.py) and the program's RECALL put on earlier occurrences of the consensus top token,
and how well the seeds agree on it with each other and with the program.

    OMP_NUM_THREADS=4 nice -n 10 .venv/bin/python -m analysis.program_gap.copy_diagnostic [--fit 16 --eval 32]
Writes analysis/out/program_gap/copy_diagnostic.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from analysis import vote_fit
from analysis.attention_fit import model_attention
from analysis.fitlib import REPO, train_windows
from analysis.mech_eval import EVAL_SEED, program
from analysis.program_gap.bottleneck import OUT, kl_rows
from analysis.token_geometry import tables_vs_attention as TV

V = 2000
SEEDS = [f"attn_s{k}" for k in range(5)]
ARMS = ("A", "A_att1", "A_tab1", "A_both1", "D")


def aligned(m):
    if m == "attn_s0":
        return m
    return os.path.join(REPO, "analysis", "out", "readable_training", m, "aligned.pt")


def arm_dirs(arm, P, t, att_prog, att_model, direct, ov):
    if arm in ("A", "D"):
        return TV.arm_dirs(arm, P, t, att_prog, att_model, direct, ov)
    att = dict(att_prog)
    if arm == "A_cur":                                   # the model's direct path instead of FOLLOW's bigram
        F = {k: v for k, v in TV.program_dirs(P, t, att).items() if not k.startswith("follow.")}
        F["direct"] = np.asarray(direct[t], np.float64)
        return F
    mode, h = arm[2:-1], int(arm[-1])                    # A_att<h>, A_tab<h>, A_both<h>
    if mode == "att":
        att[h] = att_model[h]
        return TV.program_dirs(P, t, att)
    prefix = {0: "local0.", 1: "head1.", 2: "local2.", 3: "local3."}[h]
    F = {k: v for k, v in TV.program_dirs(P, t, att).items() if not k.startswith(prefix)}
    a = att_prog[h] if mode == "tab" else att_model[h]
    F[f"ov{h}.self"] = np.diagonal(a)[:, None] * np.asarray(ov[h][t], np.float64)
    F[f"ov{h}.earlier"] = (a * (1 - np.eye(len(t)))) @ np.asarray(ov[h][t], np.float64)
    return F


def stack(arm, P, W, att_prog, att_model, direct, ov):
    rows, keys = [], None
    for i, w in enumerate(W):
        t = np.asarray(w[:256], np.int64)
        F = arm_dirs(arm, P, t, {h: a[i] for h, a in att_prog.items()}, {h: a[i] for h, a in att_model.items()},
                     direct, ov)
        keys = keys or list(F)
        rows.append(np.stack([np.asarray(F[k], np.float32) for k in keys], -1).astype(np.float16))
    return np.concatenate(rows), keys


def target_mass(att, toks, target):
    """(n,) attention on earlier positions j < i whose token is the target token of position i."""
    n_w, T = toks.shape
    out = np.zeros((n_w, T))
    for w in range(n_w):
        hit = toks[w][None, :] == target[w][:, None]                 # [i, j]
        out[w] = (att[w] * np.tril(hit, -1)).sum(-1)
    return out.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=int, default=16)
    ap.add_argument("--eval", type=int, default=32)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--no_mass", action="store_true", help="skip part 2")
    args = ap.parse_args()
    arms = args.arms.split(",")
    torch.set_num_threads(int(os.environ.get("VOTE_FIT_THREADS", "4")))
    t0 = time.time()
    from harness.common import load_model, model_logprobs
    from program import head0, head1, head2
    from analysis.attention_fit import fit_rule, module_copy

    P = program()
    s = P.STATS
    Wf, We = train_windows(args.fit, seed=vote_fit.FIT_SEED), train_windows(args.eval, seed=EVAL_SEED)
    toks = np.asarray(We[:, :256], np.int64)
    model = load_model("attn_s0")
    direct, ov, scale = TV.exact_tables(model, Wf)
    Af, Ae = model_attention(model, Wf), model_attention(model, We)
    h3 = module_copy(head0, tag="h3")
    fit_rule(h3, Af[:, 3], Wf, s, keep=True)

    def prog_att(W):
        out = {0: [], 1: [], 2: [], 3: []}
        for w in W:
            t = np.asarray(w[:256], np.int64)
            for h, mod in ((0, head0), (1, head1), (2, head2), (3, h3)):
                out[h].append(mod.attention(t, s))
        return {h: np.stack(v) for h, v in out.items()}

    pa_f, pa_e = prog_att(Wf), prog_att(We)
    ma_f = {h: Af[:, h].astype(np.float64) for h in range(4)}
    ma_e = {h: Ae[:, h].astype(np.float64) for h in range(4)}
    # references and classes
    lps = {m: model_logprobs(load_model(aligned(m)), We).reshape(-1, V).astype(np.float32) for m in SEEDS}
    ens = np.log(np.mean([np.exp(l.astype(np.float64)) for l in lps.values()], 0)).astype(np.float32)
    spread = np.mean([kl_rows(ens, l) for l in lps.values()], 0)
    ens_top = ens.argmax(-1)
    pos = np.tile(np.arange(256), args.eval)
    flat = toks.reshape(-1)
    copy = np.array([ens_top[i] in set(flat[i - p: i]) for i, p in enumerate(pos)])   # strictly earlier
    piece = (s.piece[ens_top] > 0.5) & ~copy
    med = np.median(spread)
    classes = {"copy_agree": copy & (spread < med), "copy_disagree": copy & (spread >= med), "piece": piece,
               "rest": ~copy & ~piece}
    res = {"fit_windows": args.fit, "eval_windows": args.eval, "spread_median": float(med),
           "class_share": {k: float(m.mean()) for k, m in classes.items()}, "arms": {}}
    # part 1
    probs = np.exp(model_logprobs(model, Wf)).reshape(-1, V).astype(np.float32)
    pu = torch.as_tensor(np.asarray(s.p_uni), dtype=torch.float32)
    per_pos = {}
    for arm in arms:
        Xf, keys = stack(arm, P, Wf, pa_f, ma_f, direct, ov)
        has_recall = any(k.startswith("head1.") for k in keys)
        b0 = TV.init_beta(keys)
        if has_recall:
            beta, size = vote_fit.fit(Xf, probs, b0, keys=keys, c0=0.5, pu=np.asarray(s.p_uni))
        else:
            beta, size = vote_fit.fit(Xf, probs, b0), None
        del Xf
        Xe, _ = stack(arm, P, We, pa_e, ma_e, direct, ov)
        X = torch.from_numpy(Xe)
        bt = torch.as_tensor(beta, dtype=torch.float32)
        mask = vote_fit.sat_mask(keys) if size is not None else None
        lq = []
        with torch.no_grad():
            for c in range(0, len(X), 256):
                lq.append(torch.log_softmax(vote_fit.combine(X[c:c + 256].float(), bt, mask, size, pu), -1).numpy())
        lq = np.concatenate(lq)
        k0, ke = kl_rows(lps["attn_s0"], lq), kl_rows(ens, lq)
        per_pos[arm] = (k0, ke)
        res["arms"][arm] = {"kl_attn_s0": float(k0.mean()), "kl_ens5": float(ke.mean()), "n_directions": len(keys),
                            "by_class": {c: {"kl_attn_s0": float(k0[m].mean()), "kl_ens5": float(ke[m].mean()),
                                             "share_of_ens5_miss": float(ke[m].sum() / ke.sum())}
                                         for c, m in classes.items()}}
        print(arm, round(float(k0.mean()), 4), round(float(ke.mean()), 4), f"[{time.time() - t0:.0f}s]", flush=True)
        del Xe, X, lq
    # differences against A, per window (paired SE), overall and per class
    nw = args.eval
    res["vs_A"] = {}
    for arm in [a for a in arms if a != "A"]:
        r = {}
        for ref, idx in (("attn_s0", 0), ("ens5", 1)):
            d = per_pos[arm][idx] - per_pos["A"][idx]
            dw = d.reshape(nw, -1).mean(1)
            r[ref] = {"all": float(d.mean()), "se": float(dw.std(ddof=1) / np.sqrt(nw)),
                      "by_class_sum_share": {c: float(d[m].sum() / len(d)) for c, m in classes.items()}}
        res["vs_A"][arm] = r
    if args.no_mass:
        res["seconds"] = time.time() - t0
        with open(os.path.join(OUT, f"copy_diagnostic{args.tag}.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res["vs_A"], indent=1))
        return
    # part 2: attention on the copy target
    tgt = ens_top.reshape(nw, 256)
    mass = {m: target_mass(model_attention(load_model(aligned(m)), We)[:, 1].astype(np.float64), toks, tgt)
            for m in SEEDS}
    mass["program"] = target_mass(pa_e[1], toks, tgt)
    cm = copy
    res["copy_target_mass"] = {m: {c: float(v[classes[c]].mean()) for c in ("copy_agree", "copy_disagree")}
                               for m, v in mass.items()}
    seed_mat = np.stack([mass[m][cm] for m in SEEDS])
    corr = np.corrcoef(np.vstack([seed_mat, mass["program"][cm][None]]))
    res["copy_target_mass_corr"] = {"seeds_pairwise_mean": float(corr[:5, :5][~np.eye(5, dtype=bool)].mean()),
                                    "program_vs_each_seed": [float(x) for x in corr[5, :5]],
                                    "program_vs_seed_mean": float(np.corrcoef(seed_mat.mean(0),
                                                                              mass["program"][cm])[0, 1])}
    res["seconds"] = time.time() - t0
    with open(os.path.join(OUT, f"copy_diagnostic{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "arms"}, indent=1))
    for arm, r in res["arms"].items():
        print(arm, json.dumps(r["by_class"]))


if __name__ == "__main__":
    main()
