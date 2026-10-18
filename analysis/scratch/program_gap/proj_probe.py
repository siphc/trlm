import numpy as np, json
from analysis.mech_eval import program
from harness.common import load_model
P = program(); s = P.STATS
d = np.load("analysis/out/program_gap/model_table.npz")
keep, B, wm, cnt = d["keep"], d["B"], d["wmass"], d["cnt"]
E = load_model("attn_s0").wte.weight.detach().double().numpy()
Q, _ = np.linalg.qr(E - E.mean(0))           # span of the code (centred over tokens)
Pi = Q @ Q.T
lp1 = np.asarray(s.log_follow, np.float64); lu = np.log(s.p_uni)
T1 = lp1 - lp1.mean(1, keepdims=True)
Tu = np.broadcast_to(lu - lu.mean(), T1.shape)
feats = {"log_P1": T1[keep], "log_uni": Tu[keep],
         "out-proj log_P1": (T1 @ Pi)[keep], "in+out proj log_P1": (Pi @ T1 @ Pi)[keep],
         "in-proj log_P1": (Pi @ T1)[keep], "out-proj log_uni": (Tu @ Pi)[keep]}
Wt = wm * cnt[:, None]; Wt = Wt / Wt.sum()
def cen(x): return x - (Wt * x).sum(1, keepdims=True) / Wt.sum(1, keepdims=True)
y = cen(B); sst = (Wt * y * y).sum()
def r2(keys):
    X = np.stack([cen(feats[k]) for k in keys], -1)
    xtx = np.einsum("svk,sv,svl->kl", X, Wt, X); xty = np.einsum("svk,sv,sv->k", X, Wt, y)
    b = np.linalg.solve(xtx, xty)
    return round(float(1 - (sst - 2 * b @ xty + b @ xtx @ b) / sst), 3), [round(float(x), 3) for x in b]
out = {}
for keys in (["log_P1"], ["log_P1", "log_uni"], ["out-proj log_P1"], ["out-proj log_P1", "out-proj log_uni"],
             ["in+out proj log_P1", "out-proj log_uni"], ["in-proj log_P1", "log_uni"],
             ["log_P1", "log_uni", "in+out proj log_P1", "out-proj log_P1", "out-proj log_uni"]):
    out[" + ".join(keys)] = r2(keys)
print(json.dumps(out, indent=1))
