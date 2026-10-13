"""D8: which corpus pair statistic looks like head 1's token-token score term TT[q, k] (exact weights, insight only)?
Double-centred TT (remove query and key means) vs double-centred candidates, over frequent q and k, weighted by
P(q) P(k) (weighted correlation). Also R^2 of TT's interaction with several candidates together."""
import sys

import numpy as np

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/recall")
import lib  # noqa: E402
from analysis.mech_eval import program  # noqa: E402

P = program()
s = P.STATS
uni = np.asarray(s.unigram, np.float64)
TT = np.load("/Users/svi/Desktop/trlm/analysis/scratch/head1_card/TT.npy").astype(np.float64)
qs = np.argsort(-uni)[:800]
ks = np.argsort(-uni)[:1500]
wq = uni[qs] / uni[qs].sum()
wk = uni[ks] / uni[ks].sum()
W = np.outer(wq, wk)


def dc(M):
    M = M[np.ix_(qs, ks)].astype(np.float64)
    M = M - (M * wk[None, :]).sum(1, keepdims=True)
    M = M - (M * wq[:, None]).sum(0, keepdims=True)
    return M


P1 = np.exp(np.asarray(s.log_follow, np.float64))
P1 /= P1.sum(1, keepdims=True)
topic = np.asarray(s.topic, np.float64)
pv = np.asarray(s.p_uni, np.float64)
simL = np.load(f"{lib.OUT}/simL.npy").astype(np.float64)
simR = np.load(f"{lib.OUT}/simR.npy").astype(np.float64)
pmi1 = np.asarray(s.pmi[1], np.float64)
cand = {
    "class_successor": np.asarray(s.class_successor, np.float64),
    "pmi1 (k follows q)": pmi1,
    "pmi1 clipped >= -3": np.maximum(pmi1, -3),
    "ppmi1": np.maximum(pmi1, 0),
    "pmi1.T (k precedes q)": pmi1.T,
    "pmi2 (k two after q)": np.asarray(s.pmi[2], np.float64),
    "succ_sim E_v~P1(.|q) simL(v,k)": P1 @ simL,
    "simR(q,k)": simR,
    "simL(q,k)": simL,
    "topic[q,k] (k in q's topic)": topic,
    "topic[k,q]": topic.T,
    "E_v~P1(.|q) topic[k,v]": P1 @ topic.T,
    "log E_v~P1(.|q) exp topic[k,v]": np.log(P1 @ np.exp(topic).T),
}
T0 = dc(TT)
sst = (W * T0 * T0).sum()
print(f"TT interaction: weighted var {sst:.4f}; share of TT weighted variance that is interaction: "
      f"{sst / (W * (TT[np.ix_(qs, ks)] - (W * TT[np.ix_(qs, ks)]).sum()) ** 2).sum():.3f}")
X = {}
for nm, M in cand.items():
    C = dc(M)
    X[nm] = C
    r = (W * T0 * C).sum() / np.sqrt(sst * (W * C * C).sum())
    print(f"{nm:40s} corr {r:+.3f}  R2 {r * r:.3f}")
names = list(X)
A = np.stack([X[n].ravel() for n in names], 1)
w = W.ravel()
beta = np.linalg.lstsq(A * np.sqrt(w)[:, None], T0.ravel() * np.sqrt(w), rcond=None)[0]
res = T0.ravel() - A @ beta
print("all together R2", 1 - (w * res * res).sum() / sst, dict(zip(names, np.round(beta, 3))))
# effective rank of the interaction
U, S, Vt = np.linalg.svd(np.sqrt(wq)[:, None] * T0 * np.sqrt(wk)[None, :], full_matrices=False)
print("interaction singular value shares (top 10):", np.round(S[:10] ** 2 / (S ** 2).sum(), 3))
from program.vocab import TOKENS
for c in range(4):
    uq = U[:, c] / np.sqrt(wq)
    vk = Vt[c] / np.sqrt(wk)
    print(f"comp {c}: queries + {[TOKENS[qs[i]] for i in np.argsort(-uq)[:10]]}  - {[TOKENS[qs[i]] for i in np.argsort(uq)[:8]]}")
    print(f"         keys    + {[TOKENS[ks[i]] for i in np.argsort(-vk)[:12]]}  - {[TOKENS[ks[i]] for i in np.argsort(vk)[:8]]}")

# low-rank smoothing of the bigram PMI (closed form: weighted truncated SVD of a corpus table)
print("--- low-rank bigram PMI (weighted SVD over all tokens, weights sqrt P(q) sqrt P(k)) ---")
pw = np.sqrt(pv)
for base_nm, Mb in [("pmi1", pmi1), ("pmi1 clipped >= -3", np.maximum(pmi1, -3)), ("ppmi1", np.maximum(pmi1, 0))]:
    Uu, Ss, Vv = np.linalg.svd(pw[:, None] * Mb * pw[None, :], full_matrices=False)
    for r in (4, 8, 16, 32, 64, 128):
        M = (Uu[:, :r] * Ss[:r]) @ Vv[:r] / pw[:, None] / pw[None, :]
        C = dc(M)
        rr = (W * T0 * C).sum() / np.sqrt(sst * (W * C * C).sum())
        print(f"{base_nm:22s} rank {r:3d} corr {rr:+.3f} R2 {rr * rr:.3f}")
