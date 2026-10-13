"""s02: is the OV of heads 0/2 a blurred (low-rank, tied-embedding) version of a corpus statistic? (insight only)

For each head h, OV_h[s, :] lies in a 32-dim subspace of logit space (the head's output directions mapped through the
final LN gain and the tied unembedding). Compare OV rows with corpus rows X[s, :] and with X projected onto
(a) the 128-dim span of the unembedding, (b) the head's own 32-dim output span, (c) the top-32 SVD of X itself.
"""
import json
import os

import numpy as np

from common import OUT, V, model_weights, ov_cached, rowcorr
from analysis.mech_eval import stats
from program.vocab import TOKENS

S_ = stats()
OV = ov_cached().astype(np.float64)
m = model_weights()
uni = np.asarray(S_.unigram)
pu = np.asarray(S_.p_uni)
pmi = {k: np.asarray(S_.pmi[k], np.float64) for k in range(1, 9)}
cs1 = np.load(os.path.join(OUT, "cs1_K30.npy")).astype(np.float64)
freq = np.where(uni >= 20000)[0]
ws = uni[freq] / uni[freq].sum()


def proj(X, U):
    """Project rows of X (over v) onto the column span of U (V x k), after centring over v."""
    Uc = U - U.mean(0, keepdims=True)
    Q, _ = np.linalg.qr(Uc)
    Xc = X - X.mean(1, keepdims=True)
    return (Xc @ Q) @ Q.T


def svd_k(X, k, rows):
    Xc = X - X.mean(1, keepdims=True)
    w = np.sqrt(uni / uni.sum())[:, None]
    U, s, Vt = np.linalg.svd(Xc * w, full_matrices=False)
    return ((U[:, :k] * s[:k]) @ Vt[:k]) / w


UE = m.WE * m.gf[None, :]
res = {}
for h in (0, 2):
    Uh = m.WE @ (m.gf[:, None] * m.WOh[h])                   # (V, 32) logit directions of head h
    r = {}
    for name, X in (("PMI_1", pmi[1]), ("PMI_2", pmi[2]), ("PMI_3", pmi[3]), ("CS1", cs1)):
        r[name] = {"raw": float(ws @ rowcorr(OV[h][freq], X[freq])),
                   "proj_unembed128": float(ws @ rowcorr(OV[h][freq], proj(X, UE)[freq])),
                   "proj_head32": float(ws @ rowcorr(OV[h][freq], proj(X, Uh)[freq])),
                   "svd32_of_X": float(ws @ rowcorr(OV[h][freq], svd_k(X, 32, freq)[freq]))}
    # how much of each OV row is inside the head-32 span: trivially 1; how much of the corpus row is inside it
    for name, X in (("PMI_1", pmi[1]), ("PMI_2", pmi[2])):
        Xc = X - X.mean(1, keepdims=True)
        r[name]["share_of_X_in_head32_span"] = float(ws @ ((proj(X, Uh)[freq] ** 2).sum(1) / (Xc[freq] ** 2).sum(1)))
        r[name]["share_of_X_in_unembed_span"] = float(ws @ ((proj(X, UE)[freq] ** 2).sum(1) / (Xc[freq] ** 2).sum(1)))
    res[f"head{h}"] = r
    print(f"head {h}:", json.dumps(r, indent=None))
json.dump(res, open(os.path.join(OUT, "s02.json"), "w"), indent=1)

# examples
for h in (0, 2):
    print(f"\n=== head {h}: top / bottom OV entries vs PMI_1 and PMI_2 top ===")
    for tok in [" the", " a", " Lily", " said", ".", " was", " to", " and", " cr", " big", "\"", " happy", " of", "Once"]:
        s = TOKENS.index(tok)
        top = np.argsort(-OV[h][s])[:8]
        bot = np.argsort(OV[h][s])[:5]
        print(f"{tok!r:9} OV+ {[TOKENS[v] for v in top]}  OV- {[TOKENS[v] for v in bot]}")
        print(f"{'':9} PMI1 {[TOKENS[v] for v in np.argsort(-pmi[1][s])[:6]]}  PMI2 {[TOKENS[v] for v in np.argsort(-pmi[2][s])[:6]]}")
