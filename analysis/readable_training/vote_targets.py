"""Vote targets for arm I (pre-registration: logs/prereg_readable_training.md, "Training signal").

For each constrained model head, the corpus table its vote (OV) row should correlate with, the column weights (unigram)
and the row weights (half how often the rule reads the token as an earlier token, half its unigram frequency):

  head 0 (LOCAL-0): T(s, .) = sum_{d=1..7} m0(d) STRENGTH_NEAR PMI_{d+1}(s, .) + m0(>=8) STRENGTH_FAR PMI_8(s, .)
                              + EARLIER_SUCCESSORS PMI_1(s, .) + EARLIER_ALIKE slot_affinity(s, .)
  head 2 (LOCAL-2): T(s, .) = sum_{d=1..7} m2(d) STRENGTH_NEAR PMI_{d+1}(s, .) + m2(>=8) STRENGTH_FAR PMI_8(s, .)
                              + NO_REPEAT [. = s]
  head 1 (RECALL):  T(s, .) = COPY[g] [. = s] + TOPIC[g] topic(s, .) + NEXT[g] PMI_1(s, .),  g = group of s
m_h(d): rule h's share of earlier-token attention at distance d (sum over d >= 1 is 1), on 512 pool windows. Rows are
standardised (unigram-weighted mean 0, weighted norm 1). Also reported: each target's best rank-32 share (the ceiling
for a rank-32 vote table) in the weighted metric.

    .venv/bin/python -m analysis.readable_training.vote_targets
Writes data/readable/vote_targets.npz and the summary into analysis/out/readable_training/rules.json ("vote_targets").
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from analysis.readable_training.common import REPO, RULE_HEADS, T, pool_indices, save_rules, windows_at

V = 2000
OUT = os.path.join(REPO, "data", "readable", "vote_targets.npz")


def main():
    import importlib

    from analysis.mech_eval import program

    P = program()
    s = P.STATS
    mods = {h: importlib.import_module(f"program.{h}") for h in RULE_HEADS}
    tr, _ = pool_indices()
    W = windows_at(tr[:512])
    dist_mass = {h: np.zeros(T) for h in ("head0", "head2")}
    read = {h: np.zeros(V) for h in RULE_HEADS}
    dist = np.arange(T)[:, None] - np.arange(T)[None, :]
    for w in W:
        t = np.asarray(w[:T], np.int64)
        for h in RULE_HEADS:
            a = mods[h].attention(t, s)
            e = np.tril(a, -1)                                   # earlier tokens only
            np.add.at(read[h], t, e.sum(0))
            if h in dist_mass:
                for d in range(1, T):
                    dist_mass[h][d] += np.diagonal(a, -d).sum()
    m = {h: v / v[1:].sum() for h, v in dist_mass.items()}
    pmi = {d: np.asarray(s.pmi[d], np.float64) for d in range(1, 9)}

    def local(h, extra):
        mod = mods[h]
        acc = sum(m[h][d] * mod.STRENGTH_NEAR * pmi[d + 1] for d in range(1, 8))
        acc = acc + m[h][8:].sum() * mod.STRENGTH_FAR * pmi[8]
        return acc + extra

    h0, h1, h2 = mods["head0"], mods["head1"], mods["head2"]
    T0 = local("head0", h0.EARLIER_SUCCESSORS * pmi[1] + h0.EARLIER_ALIKE * np.asarray(s.slot_affinity, np.float64))
    T2 = local("head2", h2.NO_REPEAT * np.eye(V))
    g = h1.GROUP
    T1 = (h1._per_group(h1.TOPIC)[g][:, None] * np.asarray(s.topic, np.float64)
          + h1._per_group(h1.NEXT)[g][:, None] * pmi[1])
    T1[np.arange(V), np.arange(V)] += h1._per_group(h1.COPY)[g]
    colw = s.p_uni / s.p_uni.sum()
    targets = []
    for Tb in (T0, T1, T2):
        x = Tb - (Tb @ colw)[:, None]
        x = x / np.sqrt((x * x) @ colw)[:, None]
        targets.append(x)
    targets = np.stack(targets).astype(np.float32)
    roww = np.stack([0.5 * read[h] / read[h].sum() + 0.5 * colw for h in RULE_HEADS])
    roww = roww / roww.sum(1, keepdims=True)
    # rank-32 ceiling: share of the row-weighted, column-weighted energy captured by the best rank-32 table
    ceiling = {}
    for i, h in enumerate(RULE_HEADS):
        A = np.sqrt(roww[i])[:, None] * targets[i].astype(np.float64) * np.sqrt(colw)[None, :]
        sv = np.linalg.svd(A, compute_uv=False)
        ceiling[h] = float((sv[:32] ** 2).sum() / (sv ** 2).sum())
    sha = hashlib.sha256(targets.tobytes() + roww.tobytes()).hexdigest()[:16]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, targets=targets, colw=colw.astype(np.float32), roww=roww.astype(np.float32), sha=sha,
             m0=m["head0"], m2=m["head2"])
    summary = {"sha": sha, "rank32_energy_share": ceiling,
               "distance_profile_head0_d1_to_8": [round(float(x), 4) for x in m["head0"][1:9]],
               "distance_profile_head2_d1_to_8": [round(float(x), 4) for x in m["head2"][1:9]],
               "n_windows": len(W)}
    save_rules({"vote_targets": summary})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
