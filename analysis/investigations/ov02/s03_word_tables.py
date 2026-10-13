"""s03: word-level skip-grams and slot-similarity kernels, compared with the OV maps of heads 0 and 2 (insight).

Counting rules (TRAIN stream, chunks of 25M tokens):
  unit         a token that is not a word piece starts a new unit (word, punctuation mark, end-of-story); word pieces
               (shape 'continuation': starts with a letter, no space) join the unit before them
  WC_k(s, v)   v is the first token of the k-th unit after the unit containing s (k = 1, 2, 3), no end-of-story token
               strictly between; smoothed and turned into PMI exactly like the program's skip-grams (WPMI_k)
  KPRED(s, v)  sum_a sqrt(P(prev = a | s) P(prev = a | v)): s and v are preceded by the same tokens (fill the same slot)
  KSUCC(s, v)  sum_b sqrt(P(next = b | s) P(next = b | v)): s and v are followed by the same tokens
"""
import json
import os
import time

import numpy as np

from common import OUT, V, global_r2, ov_cached, rowcorr
from analysis.mech_eval import stats
from program.corpus import BACKGROUND, CHUNK, DIRICHLET
from program.categories import CLASS_SYSTEMS
from program.vocab import ID

S_ = stats()
uni = np.asarray(S_.unigram)
pu = np.asarray(S_.p_uni)
log_uni = np.log(pu)
shape = np.asarray(S_.shape)
KMAX = 3


def word_counts():
    path = os.path.join(OUT, "wc.npy")
    if os.path.exists(path):
        return np.load(path)
    from harness.common import load_corpus
    corpus = load_corpus()
    n = len(corpus)
    counts = np.zeros((KMAX + 1, V * V), dtype=np.int64)
    margin = 400
    t0 = time.time()
    for start in range(0, n, CHUNK):
        length = min(CHUNK, n - start)
        seg = np.asarray(corpus[start: start + length + margin]).astype(np.int64)
        is_start = shape[seg] != 1
        sp = np.flatnonzero(is_start)
        u = np.cumsum(is_start) - 1                       # unit index of every position (-1 before the first start)
        ends = np.cumsum(seg == 0)
        j = np.arange(length)
        for k in range(1, KMAX + 1):
            ui = u[j] + k
            ok = (u[j] >= 0) & (ui < len(sp))
            jj = j[ok]
            tp = sp[ui[ok]]
            within = (ends[tp - 1] - ends[jj]) == 0
            jj, tp = jj[within], tp[within]
            counts[k] += np.bincount(seg[jj] * V + seg[tp], minlength=V * V)
        print(f"  chunk {start // CHUNK} {time.time() - t0:.0f}s", flush=True)
    out = counts.reshape(KMAX + 1, V, V).astype(np.int32)
    np.save(path, out)
    return out


def to_pmi(c):
    c = c.astype(np.float64)
    p = (1 - BACKGROUND) * (c + DIRICHLET * pu) / (c.sum(1, keepdims=True) + DIRICHLET) + BACKGROUND * pu
    return np.log(p) - log_uni


def kernels():
    big = np.asarray(S_.bigram_counts)
    pred = big / np.maximum(big.sum(0, keepdims=True), 1)          # P(prev = a | s) in column s
    succ = big / np.maximum(big.sum(1, keepdims=True), 1)          # P(next = b | s) in row s
    rp = np.sqrt(pred)
    rs = np.sqrt(succ)
    return rp.T @ rp, rs @ rs.T


if __name__ == "__main__":
    t0 = time.time()
    wc = word_counts()
    W = {k: to_pmi(wc[k]) for k in range(1, KMAX + 1)}
    for k in range(1, KMAX + 1):
        np.save(os.path.join(OUT, f"wpmi_{k}.npy"), W[k].astype(np.float32))
    KP, KS = kernels()
    np.save(os.path.join(OUT, "kpred.npy"), KP.astype(np.float32))
    np.save(os.path.join(OUT, "ksucc.npy"), KS.astype(np.float32))
    tcls = np.zeros(V, dtype=np.int64)
    for c, (name, toks) in enumerate(CLASS_SYSTEMS["TARGET"].items()):
        tcls[[ID[t] for t in toks]] = c
    SAMECLS = (tcls[:, None] == tcls[None, :]).astype(np.float64)
    print("tables %.0fs" % (time.time() - t0), flush=True)
    OV = ov_cached().astype(np.float64)
    pmi = {k: np.asarray(S_.pmi[k], np.float64) for k in range(1, 5)}
    EYE = np.eye(V)
    LU = np.tile(log_uni, (V, 1))
    tabs = {"PMI_1": pmi[1], "PMI_2": pmi[2], "PMI_3": pmi[3], "WPMI_1": W[1], "WPMI_2": W[2], "WPMI_3": W[3],
            "COPY": EYE, "KPRED": KP, "KSUCC": KS, "SAMECLS": SAMECLS, "LOGU": LU}
    freq = np.where(uni >= 20000)[0]
    ws_f = uni[freq] / uni[freq].sum()
    wv_p = pu / pu.sum()
    out = {}
    for h in (0, 2):
        out[f"head{h}"] = {n: [float(ws_f @ rowcorr(OV[h][freq], X[freq])), float(ws_f @ rowcorr(OV[h][freq], X[freq], wv_p))]
                           for n, X in tabs.items() if n != "LOGU"}
        print(f"head {h} rowcorr (uniform v / p_uni v):",
              {n: [round(a, 3), round(b, 3)] for n, (a, b) in out[f'head{h}'].items()}, flush=True)
    ws_all = uni / uni.sum()
    combos = [["PMI_1", "PMI_2", "PMI_3", "LOGU"], ["PMI_1", "PMI_2", "PMI_3", "LOGU", "COPY"],
              ["PMI_1", "PMI_2", "PMI_3", "LOGU", "COPY", "KPRED"], ["PMI_1", "PMI_2", "PMI_3", "LOGU", "COPY", "SAMECLS"],
              ["PMI_1", "WPMI_1", "WPMI_2", "LOGU"], ["PMI_1", "PMI_2", "PMI_3", "WPMI_1", "WPMI_2", "WPMI_3", "LOGU"],
              ["PMI_1", "PMI_2", "PMI_3", "WPMI_1", "WPMI_2", "WPMI_3", "LOGU", "COPY", "KPRED", "KSUCC", "SAMECLS"]]
    for h in (0, 2):
        for wvname, wv in (("uniform_v", np.full(V, 1.0 / V)), ("puni_v", wv_p)):
            for combo in combos:
                r2, c = global_r2(OV[h], [tabs[k] for k in combo], ws_all, wv)
                out.setdefault(f"head{h}_r2_{wvname}", {})["+".join(combo)] = [r2, [round(float(x), 3) for x in c]]
                print(f"head {h} {wvname} R2 {'+'.join(combo)}: {r2:.3f} coef {np.round(c, 2)}", flush=True)
    json.dump(out, open(os.path.join(OUT, "s03.json"), "w"), indent=1)
    print("done %.0fs" % (time.time() - t0))
