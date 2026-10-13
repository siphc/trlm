"""D10: candidate vote tables for recalled words, screened under EXACT head-1 attention (vote content only).
  family   cosine similarity of two tokens' topic profiles (rows of the topic PMI, positive part, columns weighted by
           sqrt P(v)): words that announce the same coming text (she/her/herself/female names; I/you/my)
  near     mean within-story skip-gram PMI at distances 2..8 (program STATS)
  b25      band PMI 25..64 (cached by the head-1 card; same counting rule as topic)
  next_clean  pmi[1] with targets seen fewer than 1000 times in the corpus set to 0 (junk-free NEXT)
Directions per word group, added to the 15 RECALL directions. 24 TRAIN windows (seed 3000)."""
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/recall")
import lib  # noqa: E402
from analysis.fitlib import train_windows  # noqa: E402
from analysis.mech_eval import model_logprobs, program  # noqa: E402
from harness.common import head_means, load_model  # noqa: E402
from program import head1  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
ATT = sys.argv[2] if len(sys.argv) > 2 else "exact"


def family_table(s):
    tp = np.maximum(np.asarray(s.topic, np.float64), 0.0) * np.sqrt(np.asarray(s.p_uni, np.float64))[None, :]
    tp /= np.linalg.norm(tp, axis=1, keepdims=True) + 1e-12
    return tp @ tp.T


def main():
    P = program()
    s = P.STATS
    net = lib.net()
    fam = family_table(s)
    np.save(f"{lib.OUT}/family.npy", fam.astype(np.float32))
    near = np.mean([np.asarray(s.pmi[d], np.float64) for d in range(2, 9)], axis=0)
    b25 = np.load("/Users/svi/Desktop/trlm/analysis/scratch/head1_card/PMI_25_64.npy").astype(np.float64)
    nc = np.asarray(s.pmi[1], np.float64) * (np.asarray(s.unigram) >= 1000)[None, :]
    tabs = {"family": fam, "near": near, "b25": b25, "nextclean": nc}
    W = train_windows(N, seed=3000)
    lp = model_logprobs(W)
    means = head_means(load_model("attn_s0"), W)
    lpa = model_logprobs(W, ablate_heads=[1], ablation="earlier", means=means)
    toks = [np.asarray(w[:256], dtype=np.int64) for w in W]
    G = head1.GROUP_NAMES

    def feats(t):
        n = len(t)
        A = (net.forward(t)["patterns"][1] if ATT == "exact" else head1.attention(t, s)) * (1 - np.eye(n))
        f = {"pe." + k: v for k, v in lib.group_votes(P, t, A).items()}
        g = head1.GROUP[t]
        for k, gn in enumerate(G):
            Ak = A * (g == k)[None, :]
            for nm, tab in tabs.items():
                f[f"{nm}.{gn}"] = Ak @ tab[t]
        return f

    mu = None
    for t in toks:
        f = feats(t)
        m = {k: v.mean(0) / len(toks) for k, v in f.items()}
        mu = m if mu is None else {k: mu[k] + m[k] for k in m}
    pe = [f"pe.{a}.{g}" for g in G for a in ("copy", "topic", "next")]
    pe_nc = [f"pe.{a}.{g}" for g in G for a in ("copy", "topic")] + [f"nextclean.{g}" for g in G]
    sets = {"pe15": pe, "pe15 next->nextclean": pe_nc}
    for nm in ("family", "near", "b25", "nextclean"):
        sets[f"pe15+{nm}"] = pe + [f"{nm}.{g}" for g in G]
    sets["pe15+family+near"] = pe + [f"family.{g}" for g in G] + [f"near.{g}" for g in G]
    sets["pe15+family+near+b25"] = sets["pe15+family+near"] + [f"b25.{g}" for g in G]
    acc = lib.R2(sets)
    for t, p, pa in zip(toks, lp, lpa):
        f = feats(t)
        acc.add(np.exp(p), p - pa, {k: v - mu[k] for k, v in f.items()})
    res = acc.result()
    for lab, v in res.items():
        print(f"{ATT:6s} {lab:30s} {v['all']['r2']:.4f}  beta {v['all']['beta']}")
    json.dump(res, open(f"{lib.OUT}/d10_family_{ATT}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
