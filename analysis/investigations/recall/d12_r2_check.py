"""D12: head-1 R^2 of the attention candidates measured on EARLIER keys only (diagonal removed, matching the
'earlier' ablation) vs with the current token included (what vote_fit's head_r2 sees for REPLACES candidates).
Free strengths per direction, like vote_fit's head_r2 (but without the other program parts)."""
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

N, SEED = int(sys.argv[1]), int(sys.argv[2])


def feats(P, t):
    n = len(t)
    f = {}
    b25 = C.b25_table(P)
    for v in ("v1", "pmi1", "keyflags_t"):
        A = head1.attention(t, P.STATS) if v == "v1" else C.attention(P, t, v)
        for lab, M in (("earlier", A * (1 - np.eye(n))), ("withself", A)):
            for k, x in lib.group_votes(P, t, M).items():
                f[f"{v}.{lab}.{k}"] = x
            for k, x in C.per_group(P, t, M, b25, "b25").items():
                f[f"{v}.{lab}.{k}"] = x
    return f


def main():
    P = program()
    W = train_windows(N, seed=SEED)
    lp = model_logprobs(W)
    means = head_means(load_model("attn_s0"), W)
    lpa = model_logprobs(W, ablate_heads=[1], ablation="earlier", means=means)
    toks = [np.asarray(w[:256], dtype=np.int64) for w in W]
    mu = None
    for t in toks:
        f = feats(P, t)
        m = {k: v.mean(0) / len(toks) for k, v in f.items()}
        mu = m if mu is None else {k: mu[k] + m[k] for k in m}
    sets = {}
    for v in ("v1", "pmi1", "keyflags_t"):
        for lab in ("earlier", "withself"):
            base = [k for k in mu if k.startswith(f"{v}.{lab}.") and ".b25." not in k]
            sets[f"{v} {lab} 15"] = base
            sets[f"{v} {lab} 15+b25"] = base + [k for k in mu if k.startswith(f"{v}.{lab}.b25.")]
    acc = lib.R2(sets)
    for t, p, pa in zip(toks, lp, lpa):
        f = feats(P, t)
        acc.add(np.exp(p), p - pa, {k: v - mu[k] for k, v in f.items()})
    res = acc.result()
    for lab, v in res.items():
        print(f"seed {SEED} n {N}: {lab:28s} {v['all']['r2']:.4f}")
    json.dump(res, open(f"{lib.OUT}/d12_r2_check_{SEED}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
