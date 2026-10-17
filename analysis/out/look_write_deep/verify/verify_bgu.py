"""Independent spot check of the BGU pass (analysis/look_write_deep/pass_bgu.py) on a tiny sample.

Own float64 forward pass from the raw state dict (not common_deep.forward), own window selection, own final-LN scale,
own token-level writes, own readouts:
  TOTAL : g = dF/d resid_l,  F = sum_i <logits_i, e_y - P1(.|t_i)>  (all paths exact)
  LTOTAL: same with the final LayerNorm replaced by its linearisation at the typical scale (later blocks exact)
  DV    : gradient of the linearised readout w.r.t. the final residual only (direct path), used for every layer
  echo / lecho: R replaced by softmax(logits) - P1
Accumulates u = o_h(j) . g_i over the same windows into the slot keys written by the tiny pass, and compares.
Also: finite-difference check of u for a few instances, and the sample / exclusion logic.

    .venv/bin/python analysis/out/look_write_deep/verify/verify_bgu.py verify2L_s1_tiny 16
"""
import math
import sys

import numpy as np
import torch

REPO = "/Users/svi/Desktop/trlm"
sys.path.insert(0, REPO)
V, T, EPS, DMAX = 2000, 256, 1e-5, 7
torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)


def ln(x, g):
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + EPS) * g


class Net:
    def __init__(self, path):
        st = torch.load(path, map_location="cpu", weights_only=False)
        cfg, sd = st["model_config"], st["model"]
        self.L, self.H, self.D = cfg["n_blocks"], cfg["n_heads"], cfg["d_model"]
        self.DH = self.D // self.H
        self.sd = {k: v.double() for k, v in sd.items()}
        assert not cfg.get("mlp", False)

    def run(self, x, add=None):
        """x (B, T) long. add: (layer, b, i, vec) added to the residual after block `layer` at (b, i).
        Returns logits, resid list, v list, scores list, final-LN input."""
        sd, L, H, D, DH = self.sd, self.L, self.H, self.D, self.DH
        B, TT = x.shape
        h = sd["wte.weight"][x] + sd["wpe.weight"][:TT][None]
        if torch.is_grad_enabled():
            h = h.detach().requires_grad_(True)
        resid, vs, scs = [], [], []
        causal = torch.ones(TT, TT, dtype=torch.bool).tril()
        for l in range(L):
            xin = ln(h, sd[f"blocks.{l}.ln1.weight"])
            W = sd[f"blocks.{l}.attn.qkv.weight"]
            q = (xin @ W[:D].T).reshape(B, TT, H, DH).permute(0, 2, 1, 3)
            k = (xin @ W[D:2 * D].T).reshape(B, TT, H, DH).permute(0, 2, 1, 3)
            v = (xin @ W[2 * D:].T).reshape(B, TT, H, DH).permute(0, 2, 1, 3)
            s = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(DH)
            s = torch.where(causal, s, torch.tensor(-math.inf))
            a = torch.softmax(s, -1)
            z = torch.einsum("bhij,bhjd->bhid", a, v).permute(0, 2, 1, 3).reshape(B, TT, D)
            h = h + z @ sd[f"blocks.{l}.attn.out.weight"].T
            if add is not None and add[0] == l:
                e = torch.zeros_like(h)
                e[add[1], add[2]] = add[3]
                h = h + e
            resid.append(h)
            vs.append(v)
            scs.append(s)
        logits = ln(h, sd["ln_f.weight"]) @ sd["wte.weight"].T
        return logits, resid, vs, scs, h

    def lin_logits(self, h, scale):
        c = h - h.mean(-1, keepdim=True)
        return (c * self.sd["ln_f.weight"] / scale) @ self.sd["wte.weight"].T

    def write(self, v, l, h):
        """o_h(j) = v_h(j) W_O^h, (B, T, D)."""
        Wo = self.sd[f"blocks.{l}.attn.out.weight"][:, h * self.DH:(h + 1) * self.DH]   # (D, DH)
        return v[:, h] @ Wo.T


def main(tag, n_win):
    from analysis.look_write import common as C1          # P1 table and token stream helpers (inputs, not code
    from src.train.data import load_stream                # under test)

    stream = load_stream("train")
    n_all = (len(stream) - 1) // T
    fit_idx = np.sort(np.random.default_rng(2024).choice(n_all, size=32, replace=False))
    ev_idx = np.sort(np.random.default_rng(2025).choice(n_all, size=64, replace=False))
    # check: these are the windows analysis.fitlib.train_windows returns
    from analysis.fitlib import train_windows
    assert np.array_equal(train_windows(32, 2024), np.stack([stream[i * T:i * T + T + 1] for i in fit_idx]))
    assert np.array_equal(train_windows(64, 2025), np.stack([stream[i * T:i * T + T + 1] for i in ev_idx]))
    order = np.random.default_rng(7).permutation(n_all)
    excl = set(fit_idx.tolist()) | set(ev_idx.tolist())
    full_n = 200_000_000 // T
    sample_full = [i for i in order[:full_n + 200] if i not in excl][:full_n]
    print("excluded windows that occur in the first", full_n, "of the order:",
          sum(1 for i in order[:full_n + 200] if i in excl), "; in the sample:", len(excl & set(sample_full)))
    idx = np.array(sample_full[:n_win])

    net = Net(f"{REPO}/models/deep/{tag}/final.pt")
    L, H = net.L, net.H
    # own typical final-LN scale on the fit windows
    with torch.no_grad():
        xs = torch.as_tensor(np.stack([stream[i * T:i * T + T] for i in fit_idx]).astype(np.int64))
        hf = net.run(xs)[4]
        scale = float(torch.sqrt(hf.var(-1, unbiased=False) + EPS).mean())
    z = np.load(f"{REPO}/analysis/out/look_write_deep/{tag}/bgu.npz")
    print("scale own", scale, "stored", float(z["scale"]))
    P1 = torch.as_tensor(C1.p1_table())
    print("P1 row sums: max |1 - sum|", float((P1.sum(1) - 1).abs().max()))
    w = np.stack([np.asarray(stream[i * T:i * T + T + 1], np.int64) for i in idx])
    x, y = torch.as_tensor(w[:, :T]), torch.as_tensor(w[:, 1:])
    R = torch.nn.functional.one_hot(y, V).double() - P1[x]

    logits, resid, vs, scs, hL = net.run(x)
    g_tot = torch.autograd.grad((logits * R).sum(), resid, retain_graph=True)
    R_e = torch.softmax(logits.detach(), -1) - P1[x]
    g_ech = torch.autograd.grad((logits * R_e).sum(), resid, retain_graph=True)
    lin = net.lin_logits(hL, scale)
    g_lt = torch.autograd.grad((lin * R).sum(), resid, retain_graph=True)
    g_le = torch.autograd.grad((lin * R_e).sum(), resid, retain_graph=True)
    g_dv = g_lt[-1]                       # direct path: gradient at the final residual (no later blocks)
    # own token-level writes from the same sample windows (the pass uses its first <= 4000 sample windows)
    perm = torch.as_tensor(np.random.default_rng(0).permutation(V))
    tok = x.reshape(-1)
    cnt = torch.bincount(tok, minlength=V).double()
    obar = torch.zeros(L, H, V, net.D)
    for l in range(L):
        for h in range(H):
            o = net.write(vs[l].detach(), l, h).reshape(-1, net.D)
            S = torch.zeros(V, net.D).index_add_(0, tok, o)
            ob = S / cnt.clamp(min=1)[:, None]
            ob[cnt == 0] = S.sum(0) / cnt.sum()
            obar[l, h] = ob
    print("obar max |own - stored|:", float((obar - torch.as_tensor(z["obar"]).double()).abs().max()),
          " max |stored|:", float(np.abs(z["obar"]).max()))

    keys = z["keys"]
    names = ("total", "ltotal", "dv", "echo", "lecho", "shuf", "lshuf", "oqc")
    acc = {q: np.zeros((L, H, len(keys))) for q in names}
    Ncnt = np.zeros(len(keys))
    inst = []                              # (key, l, h, quantities) for single-instance checks
    for d in range(1, DMAX + 1):
        ii = np.arange(d, T)
        for b in range(len(w)):
            kk = (d * V + w[b, ii - d]) * V + w[b, ii]
            pos = np.searchsorted(keys, kk)
            pos = np.minimum(pos, len(keys) - 1)
            ok = keys[pos] == kk
            sl = pos[ok]
            i_ok, j_ok = ii[ok], (ii - d)[ok]
            np.add.at(Ncnt, sl, 1)
            for l in range(L):
                for h in range(H):
                    O = net.write(vs[l][b:b + 1].detach(), l, h)[0]                 # (T, D)
                    Oj = O[j_ok]
                    Osh = obar[l, h][perm[x[b, j_ok]]]
                    vals = {"total": (g_tot[l][b, i_ok] * Oj).sum(-1), "ltotal": (g_lt[l][b, i_ok] * Oj).sum(-1),
                            "dv": (g_dv[b, i_ok] * Oj).sum(-1), "echo": (g_ech[l][b, i_ok] * Oj).sum(-1),
                            "lecho": (g_le[l][b, i_ok] * Oj).sum(-1),
                            "shuf": (g_tot[l][b, i_ok] * Osh).sum(-1), "lshuf": (g_lt[l][b, i_ok] * Osh).sum(-1),
                            "oqc": scs[l][b, h, i_ok, j_ok] - scs[l][b, h, i_ok, i_ok]}
                    for q in names:
                        np.add.at(acc[q][l, h], sl, vals[q].detach().numpy())
                    if d in (1, 3) and b == 0:
                        for t_ in range(0, len(sl), 150):
                            inst.append((b, int(i_ok[t_]), int(j_ok[t_]), l, h, int(sl[t_]),
                                         {q: float(vals[q][t_].detach()) for q in names}))
    print("N: max |own - stored|", float(np.abs(Ncnt - z["N"]).max()), " total instances", int(Ncnt.sum()),
          "of", len(w) * sum(T - d for d in range(1, DMAX + 1)))
    for q in names:
        diff = np.abs(acc[q] - z[q])
        scl = np.abs(z[q]).max()
        print(f"{q:7s} max |own - stored| {diff.max():.3e}  (max |stored| {scl:.3e}, rel {diff.max() / scl:.2e})")
    # single-instance slots: the stored sum equals the instance value
    print("single-instance slot checks (b, i, j, layer, head): own vs stored")
    shown = 0
    for b, i, j, l, h, s, qv in inst:
        if z["N"][s] == 1 and shown < 12 and (shown < 6 or l == 0):
            print(f"  ({b},{i},{j},L{l}H{h}) N=1  ltotal {qv['ltotal']:+.6f} / {z['ltotal'][l, h, s]:+.6f}   "
                  f"total {qv['total']:+.6f} / {z['total'][l, h, s]:+.6f}   oqc {qv['oqc']:+.5f} / {z['oqc'][l, h, s]:+.5f}")
            shown += 1
    # finite differences: adding eps * o_h(j) to the residual after block l at position i changes F by eps * u
    print("finite-difference checks of u (TOTAL with exact readout, LTOTAL with the linearised readout):")
    sel = [r for r in inst if r[3] == 0][:5] + [r for r in inst if r[3] == L - 1][:3]
    for b, i, j, l, h, s, qv in sel:
        o = net.write(vs[l][b:b + 1].detach(), l, h)[0, j]
        res = []
        for read in ("total", "ltotal"):
            Fs = []
            for e in (+1e-4, -1e-4):
                with torch.no_grad():
                    lg, _, _, _, hh = net.run(x[b:b + 1], add=(l, 0, i, e * o))
                    lg_use = lg if read == "total" else net.lin_logits(hh, scale)
                    Fs.append(float((lg_use * R[b:b + 1]).sum()))
            res.append(((Fs[0] - Fs[1]) / 2e-4, qv[read]))
        print(f"  ({b},{i},{j},L{l}H{h}) total fd {res[0][0]:+.7f} autograd {res[0][1]:+.7f} | "
              f"ltotal fd {res[1][0]:+.7f} autograd {res[1][1]:+.7f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
