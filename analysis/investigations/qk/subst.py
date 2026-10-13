"""Pattern substitution in the EXACT model (insight): replace head h's attention by a fitted rule's pattern, keep
everything else exact (including the head's own OV votes), output KL(model || substituted) on held-out
train_windows(N, 2025). Shows whether the pair term matters when the votes are the head's own."""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

REPO = "/Users/svi/Desktop/trlm"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qklib  # noqa: E402
from attnfit import tt_feature  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D, DH, T = 128, 32, 256


def forward(m, t, sub=None):
    x0 = m.WE[t] + m.WP[:len(t)]
    h1, _ = m.ln(x0, m.g1)
    n = len(t)
    mask = np.tril(np.ones((n, n), dtype=bool))
    out = x0.copy()
    pats = []
    for h in range(4):
        q = h1 @ m.Wqkv[h * DH:(h + 1) * DH].T
        k = h1 @ m.Wqkv[D + h * DH:D + (h + 1) * DH].T
        v = h1 @ m.Wqkv[2 * D + h * DH:2 * D + (h + 1) * DH].T
        s = np.where(mask, q @ k.T / np.sqrt(DH), -np.inf)
        a = np.exp(s - s.max(1, keepdims=True))
        a /= a.sum(1, keepdims=True)
        if sub is not None and h in sub:
            a = sub[h]
        pats.append(a)
        out = out + (a @ v) @ m.WO[:, h * DH:(h + 1) * DH].T
    hf, _ = m.ln(out, m.gf)
    lg = hf @ m.WE.T
    lg = lg - lg.max(1, keepdims=True)
    return lg - np.log(np.exp(lg).sum(1, keepdims=True)), pats


def rule_pattern(head, var, t, s, fits):
    parts = var.split("+")
    pairs = [p for p in parts if p not in ("base", "key", "prog", "TT", "noinfo")]
    names, F, _ = qklib.base_features(head, t, s, "key" in parts, "noinfo" in parts)
    pn, pF = qklib.pair_features(t, s, pairs)
    names, F = names + pn, F + pF
    if "TT" in parts:
        names, F = names + ["TT[d=1]", "TT[d=2..7]"], F + list(tt_feature(head, t))
    th = fits[var]["theta"]
    return qklib.softmax_rows(sum(th[n] * f for n, f in zip(names, F)))


def main():
    from analysis.fitlib import train_windows
    from analysis.mech_eval import stats
    from analysis.tools import Model

    head = int(sys.argv[1])
    variants = sys.argv[2:]
    s = stats()
    m = Model("attn_s0")
    fits = json.load(open(os.path.join(HERE, f"attnfit_h{head}.json")))
    W = train_windows(32, seed=2025)
    res = {v: [] for v in variants}
    for w in W:
        t = np.asarray(w[:T], dtype=np.int64)
        lp, _ = forward(m, t)
        p = np.exp(lp)
        for v in variants:
            lq, _ = forward(m, t, {head: rule_pattern(head, v, t, s, fits)})
            res[v].append(float((p * (lp - lq)).sum(1).mean()))
    out = {v: {"kl": round(float(np.mean(x)), 4), "se": round(float(np.std(x, ddof=1) / np.sqrt(len(x))), 4)} for v, x in res.items()}
    base = np.asarray(res[variants[0]])
    for v in variants[1:]:
        d = np.asarray(res[v]) - base
        out[v]["diff_vs_first"] = round(float(d.mean()), 4)
        out[v]["diff_se"] = round(float(d.std(ddof=1) / np.sqrt(len(d))), 4)
    print(head, json.dumps(out), flush=True)
    path = os.path.join(HERE, "subst.json")
    allr = json.load(open(path)) if os.path.exists(path) else {}
    allr[f"h{head}"] = out
    json.dump(allr, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
