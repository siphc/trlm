import numpy as np, json
from analysis.mech_eval import program
from harness.common import load_model
P = program(); s = P.STATS
E = load_model("attn_s0").wte.weight.detach().double().numpy()
def overlap(A, B, w=None):
    if w is not None:
        A = A * np.sqrt(w)[:, None]; B = B * np.sqrt(w)[:, None]
    Qa, _ = np.linalg.qr(A - A.mean(0)); Qb, _ = np.linalg.qr(B - B.mean(0))
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return round(float((sv ** 2).mean()), 3), round(float(sv[:32].mean()), 3)
N = np.asarray(s.bigram_counts, np.float64)
J = N / N.sum(); pr = J.sum(1); pc = J.sum(0); p = (pr + pc) / 2
C = J - np.outer(pr, pc)
S = C + C.T
res = {}
k = 128
def top_eig(M, by="abs"):
    w, V = np.linalg.eigh(M)
    idx = np.argsort(-np.abs(w)) if by == "abs" else np.argsort(-w)
    return V[:, idx[:k]]
Dm = 1 / np.sqrt(p + 1e-12)
res["S=C+C^T, |eig|"] = overlap(top_eig(S), E)
res["S, positive eig"] = overlap(top_eig(S, "pos"), E)
Sn = Dm[:, None] * S * Dm[None, :]
res["D^-1/2 S D^-1/2, |eig|"] = overlap(top_eig(Sn), E)
res["D^-1/2 S D^-1/2, positive"] = overlap(top_eig(Sn, "pos"), E)
pmi = np.asarray(s.pmi[1], np.float64)
res["sym PMI1 |eig|"] = overlap(top_eig((pmi + pmi.T) / 2), E)
U, sv, Vt = np.linalg.svd(np.sqrt(p)[:, None] * pmi * np.sqrt(p)[None, :])
res["PMI1 weighted SVD left"] = overlap(U[:, :k], E)
res["PMI1 weighted SVD right"] = overlap(Vt[:k].T, E)
res["PMI1 weighted SVD left+right (256)"] = overlap(np.concatenate([U[:, :k], Vt[:k].T], 1), E)
lp1 = np.asarray(s.log_follow, np.float64)
U2, _, Vt2 = np.linalg.svd(np.sqrt(p)[:, None] * (lp1 - lp1.mean(1, keepdims=True)))
res["logP1 weighted SVD left"] = overlap(U2[:, :k], E)
res["logP1 weighted SVD right"] = overlap(Vt2[:k].T, E)
rng = np.random.default_rng(0)
res["random 128"] = overlap(rng.standard_normal((2000, k)), E)
res["features: log p, piece, space, shape onehot"] = overlap(np.stack([np.log(s.p_uni), s.piece, s.space] + [(s.shape == j).astype(float) for j in range(4)], 1), E)
res["note"] = "values: (mean squared canonical correlation over 128 dims, mean of top-32 canonical correlations)"
# weighted by frequency
res["freq-weighted: logP1 SVD right"] = overlap(Vt2[:k].T, E, p)
res["freq-weighted: S |eig|"] = overlap(top_eig(S), E, p)
res["freq-weighted: random"] = overlap(rng.standard_normal((2000, k)), E, p)
print(json.dumps(res, indent=1))
