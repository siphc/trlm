import numpy as np, json
from analysis.mech_eval import program
P = program(); s = P.STATS
d = np.load("analysis/out/program_gap/model_table.npz")
keep, B, wm, cnt = d["keep"], d["B"], d["wmass"], d["cnt"]
lp1 = np.asarray(s.log_follow, np.float64)[keep]
lfv = np.log(s.p_uni)                              # log frequency of the candidate v
lfs = np.log(s.p_uni)[keep]                        # of the current token s
Wt = wm * cnt[:, None]; Wt = Wt / Wt.sum()
def cen(x): return x - (Wt * x).sum(1, keepdims=True) / Wt.sum(1, keepdims=True)
y = cen(B); sst = (Wt * y * y).sum()
def r2(F):
    X = np.stack([cen(np.broadcast_to(f, B.shape)) for f in F.values()], -1)
    xtx = np.einsum("svk,sv,svl->kl", X, Wt, X); xty = np.einsum("svk,sv,sv->k", X, Wt, y)
    b = np.linalg.solve(xtx + 1e-9*np.eye(len(F)), xty)
    return round(float(1 - (sst - 2 * b @ xty + b @ xtx @ b) / sst), 3), dict(zip(F, np.round(b, 3).tolist()))
base = {"lp1": lp1, "lu": np.broadcast_to(lfv, B.shape)}
res = {"base": r2(base)}
# candidate-frequency bins
edges = np.quantile(lfv, [0.2, 0.4, 0.6, 0.8, 0.9, 0.97])
binv = np.searchsorted(edges, lfv)
F = dict(base); F.update({f"lp1*vbin{k}": lp1 * (binv == k)[None, :] for k in range(len(edges) + 1)}); F.pop("lp1")
res["lp1 x candidate-frequency bins"] = r2(F)
F2 = dict(F); F2.update({f"vbin{k}": np.broadcast_to((binv == k)[None, :].astype(float), B.shape) for k in range(len(edges) + 1)})
res["+ candidate-frequency offsets"] = r2(F2)
# current-token frequency (continuous interaction)
F3 = dict(base); F3["lp1*lfs"] = lp1 * (lfs - lfs.mean())[:, None]
res["lp1 x current-token log freq"] = r2(F3)
F4 = dict(F2); F4["lp1*lfs"] = lp1 * (lfs - lfs.mean())[:, None]
res["both"] = r2(F4)
# hinge on lp1 as in follow_pwl
F5 = dict(base); F5.update({f"h{k}": np.maximum(lp1 - k, 0) for k in (-8, -6, -4, -2)})
res["lp1 hinges"] = r2(F5)
F6 = dict(F4); F6.update({f"h{k}": np.maximum(lp1 - k, 0) for k in (-8, -6, -4, -2)})
res["all"] = r2(F6)
for k, v in res.items(): print(k, v[0], v[1] if k in ("lp1 x candidate-frequency bins",) else "")
