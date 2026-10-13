"""D3: screen corpus vote tables against head 1's OV rows (the logits a full attention weight on token s adds,
exact weights, mean over key positions; cached by the head-1 card as OV.npy). Insight only.

Row regression per source token s (targets v with unigram > 500, centred rows): OV[s, v] ~ sum_f c_f F_f[s, v].
Candidate tables (all counted from the TRAIN corpus):
  copy      one-hot v = s
  topic     program topic PMI (9..24 after the most recent s)
  b25       band PMI 25..64 (head-1 card cache)
  next      pmi[1] (what follows s);  prev  pmi[1].T (what precedes s)
  simL      cosine similarity of left-context PPMI vectors (words that fill the same slots as s)
  simR      cosine similarity of right-context PPMI vectors (words followed by the same tokens)
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/recall")
import lib  # noqa: E402
from analysis.mech_eval import program  # noqa: E402
from program import head1  # noqa: E402

V = 2000


def ppmi_cos(C):
    """Rows of C are context-count vectors; returns cosine similarity of their PPMI vectors (V, V)."""
    C = C.astype(np.float64)
    tot = C.sum()
    pr = C.sum(1, keepdims=True) / tot
    pc = C.sum(0, keepdims=True) / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        pm = np.log((C / tot) / (pr * pc))
    pm = np.where(np.isfinite(pm) & (pm > 0), pm, 0.0)
    nrm = np.linalg.norm(pm, axis=1, keepdims=True) + 1e-12
    X = pm / nrm
    return X @ X.T


def main():
    P = program()
    s = P.STATS
    B = np.asarray(s.bigram_counts, np.float64)        # [prev, next]
    simL = ppmi_cos(B.T)     # row v: its predecessors
    simR = ppmi_cos(B)       # row v: its successors
    np.save(f"{lib.OUT}/simL.npy", simL.astype(np.float32))
    np.save(f"{lib.OUT}/simR.npy", simR.astype(np.float32))
    OV = np.load("/Users/svi/Desktop/trlm/analysis/scratch/head1_card/OV.npy").astype(np.float64)
    topic = np.asarray(s.topic, np.float64)
    b25 = np.load("/Users/svi/Desktop/trlm/analysis/scratch/head1_card/PMI_25_64.npy").astype(np.float64)
    nxt = np.asarray(s.pmi[1], np.float64)
    uni = np.asarray(s.unigram)
    eye = np.eye(V)
    tabs = {"copy": eye, "topic": topic, "b25": b25, "next": nxt, "prev": nxt.T, "simL": simL, "simR": simR}
    srows = np.where(uni > 5000)[0]
    vcols = np.where(uni > 500)[0]
    combos = [["copy"], ["copy", "topic"], ["copy", "topic", "next"], ["copy", "topic", "b25"],
              ["copy", "topic", "prev"], ["copy", "topic", "simL"], ["copy", "topic", "simR"],
              ["copy", "topic", "simL", "simR"], ["copy", "topic", "b25", "next", "prev", "simL", "simR"],
              ["simL"], ["simR"]]
    g = head1.GROUP
    out = {}
    for cb in combos:
        r2s, coefs = [], []
        for src in srows:
            y = OV[src, vcols] - OV[src, vcols].mean()
            X = np.stack([tabs[k][src, vcols] for k in cb] + [np.ones(len(vcols))], 1)
            c, *_ = np.linalg.lstsq(X, y, rcond=None)
            r2s.append(1 - ((y - X @ c) ** 2).sum() / (y ** 2).sum())
            coefs.append(c[:-1])
        r2s = np.array(r2s)
        w = uni[srows] / uni[srows].sum()
        res = {"mean_r2": float(r2s.mean()), "freq_weighted_r2": float((r2s * w).sum()),
               "median_coef": np.round(np.median(coefs, 0), 3).tolist(),
               "by_group": {gn: float(r2s[g[srows] == k].mean()) for k, gn in enumerate(head1.GROUP_NAMES)}}
        out["+".join(cb)] = res
        print(f"{'+'.join(cb):45s} mean {res['mean_r2']:.3f} fw {res['freq_weighted_r2']:.3f} coef {res['median_coef']}",
              {k: round(v, 3) for k, v in res["by_group"].items()})
    json.dump(out, open(f"{lib.OUT}/d3_ovrows.json", "w"), indent=1)


if __name__ == "__main__":
    main()
