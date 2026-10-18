"""Where does the program's gap come from: what the heads write (per-token tables) or where they look (attention)?
(analysis only; uses the model's weights as an oracle; TRAIN windows)

The model's logits (final LayerNorm at a typical scale) are a sum over token tables read with attention:
    logits_i = D(t_i) + sum_h sum_{j <= i} a^h_ij OV_h(t_j)
D: the direct path's table; OV_h: head h's vote table; a^h: head h's attention. The program has the same shape with
readable rules: corpus tables (FOLLOW, skip-gram, copy/topic votes, PHRASE) read with readable attention rules.
Factorial on attn_s0, every arm refitted (one strength per direction; RECALL's corpus votes keep their fixed size):

  A  readable attention + corpus tables      (the program; head 3 read with a fitted rule and skip-gram votes)
  B  readable attention + exact tables       (D and OV_h from the weights)
  C  exact attention    + corpus tables
  D  exact attention    + exact tables       (the model, up to the per-position LayerNorm scale)
  B_cur, C_cur: exact tables only for the current token (D and every head's self part), corpus earlier votes.
PHRASE (trigram/4-gram) is a direction in every arm.

    OMP_NUM_THREADS=4 .venv/bin/python -m analysis.token_geometry.tables_vs_attention [--fit 16 --eval 32]
Writes analysis/out/token_geometry/tables_vs_attention.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from analysis import vote_fit
from analysis.attention_fit import fit_rule, model_attention, module_copy
from analysis.fitlib import REPO, train_windows
from analysis.look_write import common as LW
from analysis.mech_eval import EVAL_SEED, model_logprobs, program

OUT = os.path.join(REPO, "analysis", "out", "token_geometry")
V = 2000


def exact_tables(model, windows):
    w = LW.weights(model)
    scale = LW.final_ln_scale(model, np.asarray(windows)[:, :256])
    ov = LW.ov_tables(w, scale)                                         # (H, V, V)
    x0 = w["wte"] + LW.pbar(w)[None]
    x0 = x0 - x0.mean(1, keepdims=True)
    direct = ((x0 * w["gf"]) @ w["wte"].T / scale).astype(np.float32)   # (V, V)
    return direct, ov, scale


def program_dirs(P, t, att, head3_votes=True):
    """The program's corpus-table directions, read with attention matrices att = {0: a0, 1: a1, 2: a2, 3: a3}."""
    from program import head1, phrase
    from program.readers import earlier_only, mass_on_tokens, skipgram_votes

    s = P.STATS
    n = len(t)
    F = {"follow.bigram": np.asarray(s.log_follow[t], np.float64),
         "follow.unigram": np.broadcast_to(np.asarray(s.log_uni, np.float64), (n, V))}
    for h, name in ((0, "local0"), (2, "local2"), (3, "local3")):
        if h == 3 and not head3_votes:
            continue
        a = att[h]
        for part, unit in (("self", (1, 0, 0)), ("near", (0, 1, 0)), ("far", (0, 0, 1))):
            F[f"{name}.{part}"] = skipgram_votes(a, t, s, *unit)
        e = earlier_only(a)
        if h == 0:
            F["local0.successors"] = e @ np.asarray(s.pmi[1][t], np.float64)
            F["local0.alike"] = e @ np.asarray(s.slot_affinity[t], np.float64)
        if h == 2:
            F["local2.no_repeat"] = mass_on_tokens(e, t)
    a1 = att[1]
    g = head1.GROUP[t]
    topic, nxt = np.asarray(s.topic[t], np.float64), np.asarray(s.pmi[1][t], np.float64)
    for k, gname in enumerate(head1.GROUP_NAMES):
        A = a1 * (g == k)[None, :]
        F[f"head1.copy.{gname}"] = mass_on_tokens(A, t)
        F[f"head1.topic.{gname}"] = A @ topic
        F[f"head1.next.{gname}"] = A @ nxt
    tri, quad = phrase.parts(t, s)
    F["phrase.trigram"], F["phrase.fourgram"] = tri, quad
    return F


def exact_dirs(t, att, direct, ov, earlier=True, selfp=True):
    n = len(t)
    F = {"direct": np.asarray(direct[t], np.float64)}
    eye = np.eye(n)
    for h in range(4):
        a = att[h]
        if selfp:
            F[f"ov{h}.self"] = np.diagonal(a)[:, None] * np.asarray(ov[h][t], np.float64)
        if earlier:
            F[f"ov{h}.earlier"] = (a * (1 - eye)) @ np.asarray(ov[h][t], np.float64)
    return F


def swap_dirs(arm, P, t, att_prog, att_model, direct, ov):
    """Arms 'S<unit>-<mode>' start from D (exact attention, exact tables) and replace one unit with readable parts:
    unit 0-3 = a head, 'cur' = the direct path; mode 'tab' = corpus table with the model's attention, 'att' = exact
    table with readable attention, 'joint' = corpus table with readable attention."""
    from program import phrase

    unit, mode = arm[1:].split("-")
    F = exact_dirs(t, att_model, direct, ov)
    tri, quad = phrase.parts(t, P.STATS)
    F["phrase.trigram"], F["phrase.fourgram"] = tri, quad
    if unit == "cur":
        F.pop("direct")
        G = program_dirs(P, t, att_model)
        F["follow.bigram"], F["follow.unigram"] = G["follow.bigram"], G["follow.unigram"]
        return F
    h = int(unit)
    if mode == "att":
        a = att_prog[h]
        F[f"ov{h}.self"] = np.diagonal(a)[:, None] * np.asarray(ov[h][t], np.float64)
        F[f"ov{h}.earlier"] = (a * (1 - np.eye(len(t)))) @ np.asarray(ov[h][t], np.float64)
        return F
    F.pop(f"ov{h}.self"); F.pop(f"ov{h}.earlier")
    att = dict(att_model)
    if mode == "joint":
        att[h] = att_prog[h]
    G = program_dirs(P, t, att)
    prefix = {0: "local0.", 1: "head1.", 2: "local2.", 3: "local3."}[h]
    F.update({k: v for k, v in G.items() if k.startswith(prefix)})
    return F


def arm_dirs(arm, P, t, att_prog, att_model, direct, ov):
    from program import phrase

    if arm.startswith("S"):
        return swap_dirs(arm, P, t, att_prog, att_model, direct, ov)

    att = att_prog if arm in ("A", "B", "B_cur") else att_model
    if arm in ("A", "C"):
        return program_dirs(P, t, att)
    if arm in ("B", "D"):
        F = exact_dirs(t, att, direct, ov)
        tri, quad = phrase.parts(t, P.STATS)
        F["phrase.trigram"], F["phrase.fourgram"] = tri, quad
        return F
    # *_cur: exact current-token tables, corpus earlier votes
    F = exact_dirs(t, att, direct, ov, earlier=True, selfp=True)
    for h in range(4):
        F.pop(f"ov{h}.earlier")
    G = program_dirs(P, t, att)
    for k in ("follow.bigram", "follow.unigram", "local0.self", "local2.self", "local3.self"):
        G.pop(k)
    F.update(G)
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


def init_beta(keys):
    from program import follow, head0, head1, head2, phrase

    known = {"follow.bigram": follow.STRENGTH, "follow.unigram": follow.UNIGRAM,
             "local0.self": head0.STRENGTH_SELF, "local0.near": head0.STRENGTH_NEAR, "local0.far": head0.STRENGTH_FAR,
             "local0.successors": head0.EARLIER_SUCCESSORS, "local0.alike": head0.EARLIER_ALIKE,
             "local2.self": head2.STRENGTH_SELF, "local2.near": head2.STRENGTH_NEAR, "local2.far": head2.STRENGTH_FAR,
             "local2.no_repeat": head2.NO_REPEAT, "phrase.trigram": phrase.TRIGRAM, "phrase.fourgram": phrase.FOURGRAM}
    out = []
    for k in keys:
        if k in known:
            out.append(known[k])
        elif k.startswith("head1."):
            part, g = k.split(".")[1:]
            out.append({"copy": head1.COPY, "topic": head1.TOPIC, "next": head1.NEXT}[part][g])
        elif k.startswith(("direct", "ov")):
            out.append(1.0)
        else:
            out.append(0.1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=int, default=16)
    ap.add_argument("--eval", type=int, default=32)
    ap.add_argument("--arms", default="A,B,C,D,B_cur,C_cur")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("VOTE_FIT_THREADS", "4")))
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    from harness.common import load_model
    from program import head0, head1, head2

    model = load_model("attn_s0")
    P = program()
    s = P.STATS
    Wf, We = train_windows(args.fit, seed=vote_fit.FIT_SEED), train_windows(args.eval, seed=EVAL_SEED)
    direct, ov, scale = exact_tables(model, Wf)
    Af, Ae = model_attention(model, Wf), model_attention(model, We)
    # head 3 gets a readable rule: LOCAL-0's rule refitted to head 3's own pattern
    h3 = module_copy(head0, tag="h3")
    fit_rule(h3, Af[:, 3], Wf, s, keep=True)
    from analysis.attention_fit import pattern_ce
    h3_pattern_kl = float(pattern_ce(h3, Ae[:, 3], We, s)[1])

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
    probs = np.exp(model_logprobs(Wf)).reshape(-1, V).astype(np.float32)
    lpe = torch.as_tensor(model_logprobs(We).reshape(-1, V), dtype=torch.float32)
    pu = torch.as_tensor(np.asarray(s.p_uni), dtype=torch.float32)
    res = {"fit_windows": args.fit, "eval_windows": args.eval, "final_ln_scale": scale,
           "head3_rule_pattern_kl": h3_pattern_kl, "arms": {}}
    per_window = {}
    for arm in args.arms.split(","):
        Xf, keys = stack(arm, P, Wf, pa_f, ma_f, direct, ov)
        has_recall = any(k.startswith("head1.") for k in keys)
        b0 = init_beta(keys)
        if has_recall:
            beta, size = vote_fit.fit(Xf, probs, b0, keys=keys, c0=0.5, pu=np.asarray(s.p_uni))
        else:
            beta, size = vote_fit.fit(Xf, probs, b0), None
        del Xf
        Xe, _ = stack(arm, P, We, pa_e, ma_e, direct, ov)
        X = torch.from_numpy(Xe)
        bt = torch.as_tensor(beta, dtype=torch.float32)
        mask = vote_fit.sat_mask(keys) if size is not None else None
        kls = []
        with torch.no_grad():
            for c in range(0, len(X), 256):
                lq = torch.log_softmax(vote_fit.combine(X[c:c + 256].float(), bt, mask, size, pu), -1)
                lp = lpe[c:c + 256]
                kls.append((lp.exp() * (lp - lq)).sum(-1))
        k = torch.cat(kls).numpy().reshape(args.eval, -1).mean(1)
        per_window[arm] = k
        res["arms"][arm] = {"kl": float(k.mean()), "n_directions": len(keys), "recall_size": size,
                            "strengths": {kk: round(float(b), 4) for kk, b in zip(keys, beta)}}
        print(arm, round(float(k.mean()), 4), f"[{time.time() - t0:.0f}s]", flush=True)
        del Xe, X
    pairs = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "B_cur"), ("C", "C_cur"), ("A", "D")]
    pairs += [("D", k) for k in per_window if k.startswith("S")]
    for a, b in pairs:
        if a in per_window and b in per_window:
            d = per_window[b] - per_window[a]
            res[f"{b} - {a}"] = {"diff": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}
    res["seconds"] = time.time() - t0
    with open(os.path.join(OUT, f"tables_vs_attention{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "arms"}, indent=1))


if __name__ == "__main__":
    main()
