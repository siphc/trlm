import numpy as np
from analysis.mech_eval import program
from program.vocab import TOKENS
P = program(); s = P.STATS
d = np.load("analysis/out/goal2/model_table.npz")
keep, B, wm, cnt = d["keep"], d["B"], d["wmass"], d["cnt"]
lp1 = np.asarray(s.log_follow, np.float64)[keep]; lu = np.log(s.p_uni)[None].repeat(len(keep), 0)
X = np.stack([lp1 - lp1.mean(1, keepdims=True), lu - lu.mean(1, keepdims=True)], -1)
y = B
Wt = wm * cnt[:, None]
xtx = np.einsum("svk,sv,svl->kl", X, Wt, X); xty = np.einsum("svk,sv,sv->k", X, Wt, y - (Wt*y).sum(1,keepdims=True)/Wt.sum(1,keepdims=True))
b = np.linalg.solve(xtx, xty)
R = y - X @ b
R = R - (wm * R).sum(1, keepdims=True)
for name in [" the", " a", ".", " was", " to", " she", " and", " named", " big", " very", " K", "ed", " went"]:
    if name not in TOKENS: continue
    sid = TOKENS.index(name)
    if sid not in keep: continue
    k = list(keep).index(sid)
    p1 = np.exp(lp1[k]); m = wm[k]
    top_m = np.argsort(-m)[:8]
    print(f"== {name!r} (n={int(cnt[k])})  model mean-next: " + " ".join(f"{TOKENS[v]!r}:{m[v]:.2f}/{p1[v]:.2f}" for v in top_m))
    # residual where model mass is non-trivial
    mask = m > 0.002
    idx = np.nonzero(mask)[0]
    o = idx[np.argsort(-R[k, idx])]
    print("   model > bigram fit:", " ".join(f"{TOKENS[v]!r}:{R[k,v]:+.1f}(p1 {p1[v]:.3f})" for v in o[:8]))
    print("   model < bigram fit:", " ".join(f"{TOKENS[v]!r}:{R[k,v]:+.1f}(p1 {p1[v]:.3f})" for v in o[::-1][:8]))
print("coef", b)
