import numpy as np, json, torch
from analysis.goal2 import budget as B
from analysis.goal2.bottleneck import kl_rows
from analysis.fitlib import train_windows
from analysis.mech_eval import EVAL_SEED, program
from harness.common import load_model
torch.set_num_threads(3)
model = load_model("attn_s0"); P = program(); s = P.STATS
W = train_windows(16, seed=EVAL_SEED)
MP, PP, LP = [], [], []
for w in W:
    t = np.asarray(w[:256], np.int64)
    mp, logits = B.model_parts(model, t)
    MP.append(sum(mp.values()).astype(np.float64)); PP.append(sum(B.program_parts(P, t).values()).astype(np.float64)); LP.append(B.lsm(logits))
mb = np.mean([m.mean(0) for m in MP], 0); pb = np.mean([p.mean(0) for p in PP], 0)
r = {}
r["program"] = float(np.mean([kl_rows(lp, B.lsm(q)).mean() for lp, q in zip(LP, PP)]))
r["program centred + model bias"] = float(np.mean([kl_rows(lp, B.lsm(q - pb + mb)).mean() for lp, q in zip(LP, PP)]))
# best scalar mix: program + c*(mb - pb)
for c in (0.25, 0.5, 0.75, 1.0):
    r[f"program + {c}*(mb-pb)"] = float(np.mean([kl_rows(lp, B.lsm(q + c*(mb - pb))).mean() for lp, q in zip(LP, PP)]))
pu = s.unigram / s.unigram.sum()
d = mb - pb
def wc(a, b, w=pu):
    a = a - (w*a).sum(); b = b - (w*b).sum()
    return float((w*a*b).sum() / np.sqrt((w*a*a).sum()*(w*b*b).sum()))
feats = {"log_uni": np.log(s.p_uni), "piece": s.piece, "space": s.space, "cont": s.cont,
         "log_n_succ": np.log1p((s.bigram_counts > 0).sum(1)), "log_n_pred": np.log1p((s.bigram_counts > 0).sum(0)),
         "info1": s.info[0], "sentence_end": s.sentence_end}
r["corr(model bias, f) weighted by P(v)"] = {k: wc(mb, f) for k, f in feats.items()}
r["corr(model bias - program bias, f)"] = {k: wc(d, f) for k, f in feats.items()}
r["corr(model bias, program bias)"] = wc(mb, pb)
r["weighted sd: model bias, program bias, diff"] = [float(np.sqrt((pu*(x-(pu*x).sum())**2).sum())) for x in (mb, pb, d)]
r["uniform-weighted corr(mb,pb)"] = wc(mb, pb, np.ones(2000)/2000)
print(json.dumps(r, indent=1))
np.save("/tmp/mb.npy", mb); np.save("/tmp/pb.npy", pb)
