"""Check the 200M-token sample of a full BGU pass: recount pair instances (d = 1 and d = 7) over the sample windows
(default_rng(7) order of aligned TRAIN windows, fit (2024, 32) and eval (2025, 64) windows excluded, first 781,250)
and compare with N in bgu.npz. Also: d = 1 slot coverage of pairs with N >= 20.

    .venv/bin/python analysis/out/look_write_deep/verify/verify_counts.py attn2L_s1
"""
import sys

import numpy as np

REPO = "/Users/svi/Desktop/trlm"
sys.path.insert(0, REPO)
V, T = 2000, 256


def main(tag):
    from src.train.data import load_stream

    s = load_stream("train")
    n_all = (len(s) - 1) // T
    fit = np.random.default_rng(2024).choice(n_all, size=32, replace=False)
    ev = np.random.default_rng(2025).choice(n_all, size=64, replace=False)
    order = np.random.default_rng(7).permutation(n_all)
    order = order[~np.isin(order, np.concatenate([fit, ev]))]
    z = np.load(f"{REPO}/analysis/out/look_write_deep/{tag}/bgu.npz")
    n_win = int(z["n_tokens"]) // T
    idx = order[:n_win]
    cnt = {1: np.zeros(V * V, np.int64), 7: np.zeros(V * V, np.int64)}
    ar = np.arange(T)
    for c in range(0, n_win, 20000):
        w = np.asarray(s[(idx[c:c + 20000, None] * T + ar[None]).ravel()], np.int64).reshape(-1, T)
        for d in cnt:
            cnt[d] += np.bincount((w[:, :-d] * V + w[:, d:]).ravel(), minlength=V * V)
    keys, N = z["keys"], z["N"]
    for d in cnt:
        sel = keys // (V * V) == d
        kq = keys[sel] % (V * V)
        diff = np.abs(cnt[d][kq] - N[sel])
        print(f"d={d}: slots {sel.sum()}, max |recount - N| {diff.max():.1f}, sum N {N[sel].sum():.0f}, "
              f"instances in slots {cnt[d][kq].sum()}, all instances {cnt[d].sum()}")
    sel1 = keys // (V * V) == 1
    have = np.zeros(V * V, bool)
    have[keys[sel1] % (V * V)] = True
    miss = (cnt[1] >= 20) & ~have
    print(f"d=1 pairs with sample N >= 20: {(cnt[1] >= 20).sum()}, of which not stored as slots: {miss.sum()} "
          f"(their instance share among N>=20 pairs: {cnt[1][miss].sum() / cnt[1][cnt[1] >= 20].sum():.2e})")


if __name__ == "__main__":
    main(sys.argv[1])
