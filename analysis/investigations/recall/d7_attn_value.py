"""D7: does a better-fitting attention rule explain more of head 1's effect? For each attention variant:
R^2 of head 1's earlier-token effect on (variant attention x exact OV / sdf) [attention quality, votes exact] and on
(variant attention x 15 program vote directions) [what the program can use]. 24 TRAIN windows, seed 3000."""
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/recall")
import cand_common as C  # noqa: E402
import lib  # noqa: E402
from analysis.fitlib import train_windows  # noqa: E402
from analysis.mech_eval import model_logprobs, program  # noqa: E402
from harness.common import head_means, load_model  # noqa: E402
from program import head1  # noqa: E402

VARS = sys.argv[1].split(",")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 24
EXTRA = sys.argv[3] if len(sys.argv) > 3 else ""


def feats_of(P, net, t):
    r = net.forward(t)
    n = len(t)
    off = 1.0 - np.eye(n)
    ov = net.ov_votes(r)
    sdf = r["sdf"][:, None]
    f = {}
    for v in VARS:
        A = (head1.attention(t, P.STATS) if v == "v1" else C.attention(P, t, v)) * off
        f[f"{v}.dla"] = A @ ov / sdf
        for k, x in lib.group_votes(P, t, A).items():
            f[f"{v}.p.{k}"] = x
        if "eos0" in EXTRA:       # EOS keys vote nothing: drop their contribution from the group directions
            A2 = A * (t != 0)[None, :]
            for k, x in lib.group_votes(P, t, A2).items():
                f[f"{v}.pnoeos.{k}"] = x
    Ae = r["patterns"][1] * off
    f["exact.dla"] = Ae @ ov / sdf
    return f


def main():
    P = program()
    net = lib.net()
    W = train_windows(N, seed=3000)
    lp = model_logprobs(W)
    means = head_means(load_model("attn_s0"), W)
    lpa = model_logprobs(W, ablate_heads=[1], ablation="earlier", means=means)
    toks = [np.asarray(w[:256], dtype=np.int64) for w in W]
    mu = None
    for t in toks:
        f = feats_of(P, net, t)
        m = {k: v.mean(0) / len(toks) for k, v in f.items()}
        mu = m if mu is None else {k: mu[k] + m[k] for k in m}
    sets = {"exact": ["exact.dla"]}
    for v in VARS:
        sets[f"{v} x exact OV"] = [f"{v}.dla"]
        sets[f"{v} x program votes"] = [k for k in mu if k.startswith(f"{v}.p.")]
        if "eos0" in EXTRA:
            sets[f"{v} x program votes, EOS silent"] = [k for k in mu if k.startswith(f"{v}.pnoeos.")]
    acc = lib.R2(sets)
    for t, p, pa in zip(toks, lp, lpa):
        f = feats_of(P, net, t)
        acc.add(np.exp(p), p - pa, {k: v - mu[k] for k, v in f.items()})
    res = acc.result()
    for lab, v in res.items():
        print(f"{lab:45s} {v['all']['r2']:.4f}", v["all"]["beta"] if len(v["all"]["beta"]) == 1 else "")
    old = {}
    try:
        old = json.load(open(f"{lib.OUT}/d7_attn_value.json"))
    except Exception:
        pass
    old.update(res)
    json.dump(old, open(f"{lib.OUT}/d7_attn_value.json", "w"), indent=1)


if __name__ == "__main__":
    main()
