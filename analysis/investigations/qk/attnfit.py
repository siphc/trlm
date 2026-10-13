"""Fit log-linear attention rules for heads 0 and 2 to the heads' own patterns (cross-entropy, convex), with and
without pair features. Fit on train_windows(32, 2024) (the program's fit windows), report pattern KL on held-out
train_windows(16, 2025).

usage: python attnfit.py HEAD VARIANT [VARIANT ...]
  VARIANT = '+'-joined parts: base | key (head 0 key shape) | pair names (U, A, G, PMI, BACK, IG, UOVh0, UOVh2, TT)
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import torch

REPO = "/Users/svi/Desktop/trlm"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qklib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T = 256
torch.set_num_threads(2)


def model_patterns(wins, head, cache):
    if os.path.exists(cache):
        return np.load(cache)
    from analysis.tools import Model
    m = Model("attn_s0")
    out = np.stack([m.run(w[:T])["patterns"][head].astype(np.float32) for w in wins])
    np.save(cache, out)
    return out


def feats(head, wins, s, keyshape, pairs, noinfo=False):
    allF, names, th0 = [], None, None
    for w in wins:
        t = np.asarray(w[:T], dtype=np.int64)
        names, F, th0 = qklib.base_features(head, t, s, keyshape, noinfo)
        pn, pF = qklib.pair_features(t, s, [p for p in pairs if p != "TT"])
        names, F = names + pn, F + pF
        if "TT" in pairs:
            tt = tt_feature(head, t)
            names, F = names + ["TT[d=1]", "TT[d=2..7]"], F + list(tt)
        allF.append(np.stack(F).astype(np.float32))
    th0 = list(th0) + [0.0] * (len(names) - len(th0))
    return names, np.stack(allF), np.asarray(th0)


_TT = {}


def tt_feature(head, t):
    """Weights (ceiling only): exact token-token QK term I(q, k) at the typical LN1 scale, for d = 1..7."""
    if head not in _TT:
        from analysis.tools import Model
        m = Model("attn_s0")
        D, DH = 128, 32
        Ec = (m.WE - m.WE.mean(1, keepdims=True)) * m.g1
        WQ, WK = m.Wqkv[head * DH:(head + 1) * DH], m.Wqkv[D + head * DH:D + (head + 1) * DH]
        _TT[head] = (Ec @ WQ.T) @ (Ec @ WK.T).T / np.sqrt(DH) / 0.25 ** 2     # [q, k]
    I = _TT[head]
    n = len(t)
    f1, f2 = np.zeros((n, n)), np.zeros((n, n))
    for d in range(1, 8):
        i = np.arange(d, n)
        (f1 if d == 1 else f2)[i, i - d] = I[t[i], t[i - d]]
    return f1, f2


def ce(theta, F, A, mask):
    sc = torch.einsum("k,wkij->wij", theta, F).masked_fill(~mask, float("-inf"))
    lp = torch.log_softmax(sc, -1)
    return -(A * torch.where(mask, lp, torch.zeros_like(lp))).sum() / (A.shape[0] * A.shape[1])


def fit(F, A, th0, iters=300):
    F, A = torch.from_numpy(F), torch.from_numpy(A)
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    theta = torch.tensor(th0, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([theta], lr=1, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-8,
                            tolerance_change=1e-12, history_size=50)

    def closure():
        opt.zero_grad()
        loss = ce(theta, F, A, mask)
        loss.backward()
        return loss

    opt.step(closure)
    return theta.detach().numpy().astype(np.float64)


def pattern_kl(theta, F, A):
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    with torch.no_grad():
        c = float(ce(torch.tensor(theta, dtype=torch.float32), torch.from_numpy(F), torch.from_numpy(A), mask))
        Ad = torch.from_numpy(A).double()
        ent = float(-(torch.where(Ad > 0, Ad * torch.log(Ad.clamp_min(1e-30)), torch.zeros_like(Ad))).sum()
                    / (A.shape[0] * A.shape[1]))
    return c - ent


def main():
    from analysis.fitlib import train_windows
    from analysis.mech_eval import stats

    head = int(sys.argv[1])
    variants = sys.argv[2:]
    s = stats()
    Wf, We = train_windows(32, seed=2024), train_windows(16, seed=2025)
    Af = model_patterns(Wf, head, os.path.join(qklib.TABLES, f"pat_h{head}_fit32.npy"))
    Ae = model_patterns(We, head, os.path.join(qklib.TABLES, f"pat_h{head}_ev16.npy"))
    path = os.path.join(HERE, f"attnfit_h{head}.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for var in variants:
        t0 = time.time()
        parts = var.split("+")
        keyshape = "key" in parts
        pairs = [p for p in parts if p not in ("base", "key", "prog", "noinfo")]
        noinfo = "noinfo" in parts
        names, Ff, th0 = feats(head, Wf, s, keyshape, pairs, noinfo)
        if var == "prog":
            theta = th0
        else:
            theta = fit(Ff, Af, th0)
        _, Fe, _ = feats(head, We, s, keyshape, pairs, noinfo)
        r = {"pattern_kl_fit": round(pattern_kl(theta, Ff, Af), 4), "pattern_kl_eval": round(pattern_kl(theta, Fe, Ae), 4),
             "theta": {n: round(float(v), 4) for n, v in zip(names, theta)}, "seconds": round(time.time() - t0, 1)}
        res[var] = r
        print(head, var, r["pattern_kl_fit"], r["pattern_kl_eval"],
              {k: v for k, v in r["theta"].items() if "[d=" in k or "key[" in k}, flush=True)
        del Ff, Fe
        json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
