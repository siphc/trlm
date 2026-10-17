"""Check the committed 200M-token BGU tables (bgu.npz written by the MPS pass) on a random subset of slots: find every
instance of the chosen slots in the sample, recompute each instance with the independent float64 code of
verify_bgu.py (TOTAL, LTOTAL, DV, echo, lecho, shuffled, oracle), sum, and compare with the stored sums.

    .venv/bin/python analysis/out/look_write_deep/verify/verify_full_slots.py attn2L_s1 [n_per_d]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_bgu import Net, REPO, T, V, EPS  # noqa: E402

sys.path.insert(0, REPO)
torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)


def main(tag, n_per_d=12):
    from analysis.look_write import common as C1
    from src.train.data import load_stream

    s = load_stream("train")
    n_all = (len(s) - 1) // T
    fit = np.sort(np.random.default_rng(2024).choice(n_all, size=32, replace=False))
    ev = np.random.default_rng(2025).choice(n_all, size=64, replace=False)
    order = np.random.default_rng(7).permutation(n_all)
    order = order[~np.isin(order, np.concatenate([fit, ev]))]
    z = np.load(f"{REPO}/analysis/out/look_write_deep/{tag}/bgu.npz")
    keys, N = z["keys"], z["N"]
    idx = order[:int(z["n_tokens"]) // T]
    rng = np.random.default_rng(123)
    chosen = []
    for d in (1, 3, 7):
        cand = np.nonzero((keys // (V * V) == d) & (N >= 5) & (N <= 40))[0]
        chosen += list(rng.choice(cand, size=n_per_d, replace=False))
    chosen = np.array(sorted(chosen))
    look = {d: np.zeros(V * V, bool) for d in (1, 3, 7)}
    for sl in chosen:
        look[int(keys[sl] // (V * V))][keys[sl] % (V * V)] = True
    # locate the windows holding any instance of a chosen slot
    hits = []
    ar = np.arange(T)
    for c in range(0, len(idx), 20000):
        w = np.asarray(s[(idx[c:c + 20000, None] * T + ar[None]).ravel()], np.int64).reshape(-1, T)
        m = np.zeros(len(w), bool)
        for d in look:
            m |= look[d][w[:, :-d] * V + w[:, d:]].any(1)
        hits += list(idx[c:c + 20000][m])
    print(f"{tag}: {len(chosen)} slots (N {N[chosen].min():.0f}-{N[chosen].max():.0f}), {len(hits)} windows")

    net = Net(f"{REPO}/models/deep/{tag}/final.pt")
    L, H = net.L, net.H
    with torch.no_grad():
        xs = torch.as_tensor(np.stack([s[i * T:i * T + T] for i in fit]).astype(np.int64))
        hf = net.run(xs)[4]
        scale = float(torch.sqrt(hf.var(-1, unbiased=False) + EPS).mean())
    P1 = torch.as_tensor(C1.p1_table())
    obar = torch.as_tensor(z["obar"]).double()
    perm = torch.as_tensor(np.random.default_rng(0).permutation(V))
    names = ("total", "ltotal", "dv", "echo", "lecho", "shuf", "lshuf", "oqc")
    pos_of = {int(sl): n for n, sl in enumerate(chosen)}
    acc = {q: np.zeros((L, H, len(chosen))) for q in names}
    cnt = np.zeros(len(chosen))
    for c in range(0, len(hits), 16):
        w = np.stack([np.asarray(s[i * T:i * T + T + 1], np.int64) for i in hits[c:c + 16]])
        x, y = torch.as_tensor(w[:, :T]), torch.as_tensor(w[:, 1:])
        R = torch.nn.functional.one_hot(y, V).double() - P1[x]
        logits, resid, vs, scs, hL = net.run(x)
        g_tot = torch.autograd.grad((logits * R).sum(), resid, retain_graph=True)
        R_e = torch.softmax(logits.detach(), -1) - P1[x]
        g_ech = torch.autograd.grad((logits * R_e).sum(), resid, retain_graph=True)
        lin = net.lin_logits(hL, scale)
        g_lt = torch.autograd.grad((lin * R).sum(), resid, retain_graph=True)
        g_le = torch.autograd.grad((lin * R_e).sum(), resid)
        for d in look:
            bb, ii = np.nonzero(look[d][w[:, :T - d] * V + w[:, d:T]])
            ii = ii + d
            jj = ii - d
            kk = (d * V + w[bb, jj]) * V + w[bb, ii]
            sl = np.searchsorted(keys, kk)
            assert np.all(keys[sl] == kk)
            col = np.array([pos_of[int(t)] for t in sl], dtype=np.int64)
            np.add.at(cnt, col, 1)
            bt, it, jt = (torch.as_tensor(a) for a in (bb, ii, jj))
            for l in range(L):
                for h in range(H):
                    O = net.write(vs[l].detach(), l, h)[bt, jt]
                    Osh = obar[l, h][perm[x[bt, jt]]]
                    vals = {"total": (g_tot[l][bt, it] * O).sum(-1), "ltotal": (g_lt[l][bt, it] * O).sum(-1),
                            "dv": (g_lt[-1][bt, it] * O).sum(-1), "echo": (g_ech[l][bt, it] * O).sum(-1),
                            "lecho": (g_le[l][bt, it] * O).sum(-1), "shuf": (g_tot[l][bt, it] * Osh).sum(-1),
                            "lshuf": (g_lt[l][bt, it] * Osh).sum(-1),
                            "oqc": (scs[l][bt, h, it, jt] - scs[l][bt, h, it, it]).detach()}
                    for q in names:
                        np.add.at(acc[q][l, h], col, vals[q].numpy())
    print("N: max |recount - stored|", float(np.abs(cnt - N[chosen]).max()))
    for q in names:
        st = z[q][:, :, chosen]
        diff = np.abs(acc[q] - st)
        print(f"  {q:7s} max |own - stored| {diff.max():.2e}  max |stored| {np.abs(st).max():.2e}  "
              f"median rel {np.median(diff / np.maximum(np.abs(st), 1e-6)):.1e}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)
