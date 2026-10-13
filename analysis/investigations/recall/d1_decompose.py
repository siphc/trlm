"""D1: is RECALL's gap in WHERE it looks (attention) or in WHAT the recalled tokens vote for (votes)?

Target: model head 1's earlier-token effect y = lp - lp_ablated (harness 'earlier' ablation).
2 x 2 decomposition with exact internals (insight only):
  attention  exact head-1 pattern A1 (earlier keys)  vs  RECALL's rule
  votes      exact OV logit votes ov_j / sdf_i        vs  RECALL's 15 corpus vote directions (per-group strengths free)
Also: exact contribution split by the attended token's word group (which groups carry the effect).
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/recall")
import lib  # noqa: E402

from analysis.fitlib import train_windows  # noqa: E402
from analysis.mech_eval import model_logprobs, program  # noqa: E402
from harness.common import head_means, load_model  # noqa: E402
from program import head1  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
SEED = 3000


def feats_of(P, net, t):
    r = net.forward(t)
    n = len(t)
    off = 1.0 - np.eye(n)
    Ae = r["patterns"][1] * off
    Ar = head1.attention(t, P.STATS) * off
    ov = net.ov_votes(r)
    sdf = r["sdf"][:, None]
    f = {"dla_ee": Ae @ ov / sdf, "dla_re": Ar @ ov / sdf, "dla_ee_nosdf": Ae @ ov / sdf.mean()}
    f["recall_v1"] = Ar @ head1.recalled_votes(t, P.STATS)
    f["recall_v1_exactattn"] = Ae @ head1.recalled_votes(t, P.STATS)
    for k, v in lib.group_votes(P, t, Ae).items():
        f["pe." + k] = v
    for k, v in lib.group_votes(P, t, Ar).items():
        f["pr." + k] = v
    g = head1.GROUP[t]
    for k, gname in enumerate(head1.GROUP_NAMES):
        f["dla_g." + gname] = (Ae * (g == k)[None, :]) @ ov / sdf
    eos = (t == 0)
    f["dla_g.EOS"] = (Ae * eos[None, :]) @ ov / sdf
    return f


def main():
    P = program()
    net = lib.net()
    W = train_windows(N, seed=SEED)
    lp = model_logprobs(W)
    means = head_means(load_model("attn_s0"), W)
    lpa = model_logprobs(W, ablate_heads=[1], ablation="earlier", means=means)
    toks = [np.asarray(w[:256], dtype=np.int64) for w in W]
    mu = None
    for t in toks:
        f = feats_of(P, net, t)
        m = {k: v.mean(0) / len(toks) for k, v in f.items()}
        mu = m if mu is None else {k: mu[k] + m[k] for k in m}
    keys = list(mu)
    pe = [k for k in keys if k.startswith("pe.")]
    pr = [k for k in keys if k.startswith("pr.")]
    dg = [k for k in keys if k.startswith("dla_g.")]
    sets = {"exact attn x exact OV": ["dla_ee"], "exact attn x exact OV, sdf by mean": ["dla_ee_nosdf"],
            "rule attn x exact OV": ["dla_re"], "exact attn x program votes (15)": pe,
            "rule attn x program votes (15)": pr, "recall v1 (1 dir)": ["recall_v1"],
            "exact attn x v1 votes (1 dir)": ["recall_v1_exactattn"],
            "exact OV split by key group (6)": dg}
    for k in dg:
        sets["only " + k] = [k]
    classes = ("all", "after sentence end", "current token is a word piece", "word onset (next is likely a piece)",
               "other")
    acc = lib.R2(sets, classes)
    shift = {c: [0.0, 0] for c in classes}
    for t, p, pa in zip(toks, lp, lpa):
        f = feats_of(P, net, t)
        f = {k: v - mu[k] for k, v in f.items()}
        cls = lib.position_classes(P, t)
        acc.add(np.exp(p), p - pa, f, cls)
        kl = (np.exp(p) * (p - pa)).sum(1)
        for c in classes:
            m = np.ones(len(t), bool) if c == "all" else cls == c
            shift[c][0] += kl[m].sum()
            shift[c][1] += m.sum()
    res = acc.result()
    tot = shift["all"][0]
    out = {"n_windows": N, "seed": SEED,
           "shift": {c: {"mean": float(v[0] / max(v[1], 1)), "share_shift": float(v[0] / tot), "n": int(v[1])} for c, v in shift.items()},
           "r2": res}
    for lab, v in res.items():
        print(f"{lab:45s}", {c: (x["r2"], x["beta"][:3] if len(x["beta"]) <= 3 else "") for c, x in v.items()})
    print(json.dumps(out["shift"], indent=0))
    json.dump(out, open(f"{lib.OUT}/d1_decompose.json", "w"), indent=1)


if __name__ == "__main__":
    main()
