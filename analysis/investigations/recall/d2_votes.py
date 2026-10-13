"""D2: what vote content is missing? Head 1's earlier-token effect regressed on vote directions built with the
EXACT head-1 attention (so only the vote content varies). Corpus tables: program STATS (topic 9..24, pmi[d]) and,
for exploration, the band PMI 25..64 table cached by the head-1 card (same counting rule as topic, other band).
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/recall")
import lib  # noqa: E402

from analysis.fitlib import train_windows  # noqa: E402
from analysis.mech_eval import model_logprobs, program  # noqa: E402
from harness.common import head_means, load_model  # noqa: E402
from program import head1  # noqa: E402
import h1lib  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
SEED = 3000
B25 = np.load("/Users/svi/Desktop/trlm/analysis/scratch/head1_card/PMI_25_64.npy", mmap_mode="r")
_, C64 = h1lib.classes("ATTENTION")


def feats_of(P, net, t, near_tab):
    r = net.forward(t)
    n = len(t)
    Ae = r["patterns"][1] * (1.0 - np.eye(n))
    ov = net.ov_votes(r)
    sdf = r["sdf"][:, None]
    f = {"pe." + k: v for k, v in lib.group_votes(P, t, Ae).items()}
    g = head1.GROUP[t]
    b25 = np.asarray(B25[t], np.float64)
    near = near_tab[t]
    topic = np.asarray(P.STATS.topic[t], np.float64)
    for k, gname in enumerate(head1.GROUP_NAMES):
        Ak = Ae * (g == k)[None, :]
        f["b25." + gname] = Ak @ b25
        f["near." + gname] = Ak @ near
        f["dla_g." + gname] = Ak @ ov / sdf
    c = C64[t]
    for k in range(64):
        if np.any(c == k):
            Ak = Ae * (c == k)[None, :]
            f[f"c64copy.{k}"] = lib.copy_matrix(t, Ak)
            f[f"c64topic.{k}"] = Ak @ topic
            f[f"c64b25.{k}"] = Ak @ b25
        else:
            for nm in ("c64copy", "c64topic", "c64b25"):
                f[f"{nm}.{k}"] = np.zeros((n, lib.V))
    f["dla"] = Ae @ ov / sdf
    return f


def main():
    P = program()
    net = lib.net()
    s = P.STATS
    near_tab = np.mean([np.asarray(s.pmi[d], np.float64) for d in range(2, 9)], axis=0)
    W = train_windows(N, seed=SEED)
    lp = model_logprobs(W)
    means = head_means(load_model("attn_s0"), W)
    lpa = model_logprobs(W, ablate_heads=[1], ablation="earlier", means=means)
    toks = [np.asarray(w[:256], dtype=np.int64) for w in W]
    mu = None
    for t in toks:
        f = feats_of(P, net, t, near_tab)
        m = {k: v.mean(0) / len(toks) for k, v in f.items()}
        mu = m if mu is None else {k: mu[k] + m[k] for k in m}
    G = head1.GROUP_NAMES
    pe = [f"pe.{a}.{g}" for g in G for a in ("copy", "topic", "next")]
    b25 = [f"b25.{g}" for g in G]
    near = [f"near.{g}" for g in G]
    gtn = [f"pe.{a}.{g}" for g in G for a in ("topic", "next")]
    c64c = [f"c64copy.{k}" for k in range(64)]
    c64t = [f"c64topic.{k}" for k in range(64)]
    c64b = [f"c64b25.{k}" for k in range(64)]
    sets = {"exact dla": ["dla"], "pe15": pe, "pe15+b25": pe + b25, "pe15+near": pe + near,
            "pe15+b25+near": pe + b25 + near, "c64 copy + group topic,next": c64c + gtn,
            "c64 copy,topic + group next": c64c + c64t + [f"pe.next.{g}" for g in G],
            "c64 copy,topic,b25 + group next": c64c + c64t + c64b + [f"pe.next.{g}" for g in G]}
    acc = lib.R2(sets)
    # per-group targets: how well each group's exact contribution is described by that group's corpus votes
    gsets = {}
    for g in G:
        base = [f"pe.{a}.{g}" for a in ("copy", "topic", "next")]
        gsets[g] = lib.R2({"copy,topic,next": base, "+b25": base + [f"b25.{g}"], "+near": base + [f"near.{g}"],
                           "+b25+near": base + [f"b25.{g}", f"near.{g}"]})
    for t, p, pa in zip(toks, lp, lpa):
        f = feats_of(P, net, t, near_tab)
        f = {k: v - mu[k] for k, v in f.items()}
        w = np.exp(p)
        acc.add(w, p - pa, f)
        for g in G:
            gsets[g].add(w, f["dla_g." + g], f)
    out = {"head1 effect": acc.result(), "per group (target = exact contribution of that group)":
           {g: r.result() for g, r in gsets.items()}}
    for lab, v in out["head1 effect"].items():
        print(f"{lab:40s} {v['all']['r2']}")
    for g, r in out["per group (target = exact contribution of that group)"].items():
        print(g, {lab: v["all"]["r2"] for lab, v in r.items()}, r["copy,topic,next"]["all"]["beta"])
    json.dump(out, open(f"{lib.OUT}/d2_votes.json", "w"), indent=1)


if __name__ == "__main__":
    main()
