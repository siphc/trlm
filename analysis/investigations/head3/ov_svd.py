"""Step 6 (insight only): the structure of head 3's OV map. U[s, v] = vote of one unit of attention on token s
(token part at the mean position, mean final-LN scale). Weighted SVD: rows weighted by attention mass received
(100 windows, seed 13), columns by the unigram. Prints the top components' source and output tokens."""
import json
import numpy as np
from h3lib import H3Run, T, V, OUT, class_ids, tok_strs
from analysis.mech_eval import stats

s = stats(); S = tok_strs()
att, att_names = class_ids("ATTENTION")
R = H3Run(); n = R.net
wts = np.load(f"{OUT}/w_sd_model.npy")
# attention mass received per token (recompute cheaply from saved distance weights is not possible; redo)
from analysis.fitlib import train_windows
mass = np.zeros(V); sdf = []
for w in train_windows(60, seed=13)[:, :T]:
    r = n.run(w); np.add.at(mass, r["t"], r["pats"][3].sum(0)); sdf.append(r["sdf"])
x0 = n.WE + n.WP[:T].mean(0)
h1, _ = n.ln(x0, n.g1)
o = (h1 @ R.Wv.T) @ R.WO.T
o = (o - o.mean(1, keepdims=True)) * n.gf
U = o @ n.WE.T / np.mean(np.concatenate(sdf))
pu = np.asarray(s.p_uni)
rw = np.sqrt(mass / mass.sum() + 1e-12)[:, None]
cw = np.sqrt(pu)[None, :]
Uc = U - (U * pu).sum(1, keepdims=True)
mean_row = (mass[:, None] * Uc).sum(0) / mass.sum()
print("share of weighted variance in the shared mean vote:",
      float(((mean_row ** 2) * pu).sum() / ((mass[:, None] * Uc ** 2 * pu).sum() / mass.sum())))
Ud = Uc - mean_row
u_, sv, vt = np.linalg.svd(rw * Ud * cw, full_matrices=False)
share = sv ** 2 / (sv ** 2).sum()
print("variance shares", np.round(share[:12], 3), "cum8", share[:8].sum().round(3), "cum32", share[:32].sum().round(3))
out = {"shares": share[:32].tolist(), "comps": []}
for c in range(8):
    src = u_[:, c] / rw[:, 0]
    tgt = vt[c] / cw[0]
    sp = [S[i] for i in np.argsort(-src * mass ** 0.5)[:12]]
    sn = [S[i] for i in np.argsort(src * mass ** 0.5)[:12]]
    tp = [S[i] for i in np.argsort(-tgt * np.sqrt(pu))[:14]]
    tn = [S[i] for i in np.argsort(tgt * np.sqrt(pu))[:14]]
    print(f"\ncomp {c} share {share[c]:.3f}\n  src+ {sp}\n  src- {sn}\n  out+ {tp}\n  out- {tn}")
    out["comps"].append({"src+": sp, "src-": sn, "out+": tp, "out-": tn})
print("\nmean vote out+", [S[i] for i in np.argsort(-mean_row * np.sqrt(pu))[:15]])
print("mean vote out-", [S[i] for i in np.argsort(mean_row * np.sqrt(pu))[:15]])
np.save(f"{OUT}/U_model.npy", U.astype(np.float32))
json.dump(out, open(f"{OUT}/ov_svd.json", "w"), indent=1)
