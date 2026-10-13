"""Where is the gap between a head's earlier-token effect and the program's account of it? (open-track insight;
no weight-derived numbers leave this script)

Target y: model log-prob change when head h's earlier-token part is replaced by its mean (harness 'earlier'
ablation). Single-direction pooled weighted R^2 (by position class) of y on:
  lin_exact     the exact linearisation: W_E (dLN_f(O_h (ctx_h - mean_ctx_h))) at the real final-LN state
  lin_const     the same with a constant final-LN scale and no projection term (pure additive votes)
  progattn_ov   program attention rule (head0/head2) x the head's real per-token value vectors, linearised as lin_const
  attn_corpus   the head's real attention x the program's corpus votes (PMI_{d+1} of the token read)
  program       the program part's own direction (its attention rule x corpus votes)
The steps from lin_exact to program show how much is lost to (LN nonlinearity), (attention rule), (vote table).

  python -m analysis.investigations.word.decomp [n_windows]
"""
from __future__ import annotations

import math
import sys

import numpy as np
import torch

from analysis.fitlib import train_windows
from analysis.investigations.word import wordstats as WS
from analysis.investigations.word.diag import classes
from analysis.mech_eval import EVAL_SEED, model_logprobs, program
from harness.common import head_means, load_model
from program.readers import skipgram_votes
from program.tests import measure as M

V = 2000


def parts(model, x):
    blk = model.blocks[0]
    B, T = x.shape
    H, Dh, D = 4, 32, 128
    x0 = model.wte(x) + model.wpe(torch.arange(T))[None]
    q, k, v = blk.attn.qkv(blk.ln1(x0)).split(D, dim=-1)
    q = q.view(B, T, H, Dh).transpose(1, 2)
    k = k.view(B, T, H, Dh).transpose(1, 2)
    v = v.view(B, T, H, Dh).transpose(1, 2)                         # (B, H, T, Dh)
    sc = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
    sc = sc.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
    a = sc.softmax(-1)
    z = (a @ v).transpose(1, 2)                                      # (B, T, H, Dh)
    hfull = x0 + blk.attn.out(z.reshape(B, T, -1))
    return a, v, z, hfull


def r2(ys, xs, ws):
    xtx = xty = sst = 0.0
    for y, X, w in zip(ys, xs, ws):
        y = y - (w * y).sum(1, keepdims=True)
        X = X - (w * X).sum(1, keepdims=True)
        xtx += (w * X * X).sum()
        xty += (w * X * y).sum()
        sst += (w * y * y).sum()
    b = xty / xtx
    return 1 - (sst - 2 * b * xty + b * b * xtx) / sst, b


def main(n=16):
    from program import head0, head2

    P = program()
    PIECE = WS.piece_flags()
    model = load_model("attn_s0")
    W = train_windows(n, seed=EVAL_SEED)
    means = head_means(model, W)
    lp = model_logprobs(W)
    E = model.wte.weight.detach().double()
    gf = model.ln_f.weight.detach().double()
    Wo = model.blocks[0].attn.out.weight.detach().double()
    toks = [np.asarray(w[:256], dtype=np.int64) for w in W]
    cls = []
    for t in toks:
        c, _ = classes(P, t, PIECE)
        cls.append(np.array(["inside" if x.startswith("inside") else x for x in c], dtype=object))
    rule = {0: head0, 2: head2}
    for h in [0, 2, 3]:
        lpa = model_logprobs(W, ablate_heads=[h], ablation="earlier", means=means)
        mc = torch.as_tensor(means["ctx"][h]).double()
        dirs = {k: [] for k in ["lin_exact", "lin_const", "progattn_ov", "attn_corpus", "program"]}
        with torch.no_grad():
            for b, t in enumerate(toks):
                x = torch.as_tensor(t[None])
                a, v, z, hfull = parts(model, x)
                a, v, hfull = a[0, h].double(), v[0, h].double(), hfull[0].double()
                T = len(t)
                eye = torch.eye(T, dtype=torch.float64)
                ctx = (a * (1 - eye)) @ v                                    # earlier-token part (T, Dh)
                delta = (ctx - mc) @ Wo[:, h * 32:(h + 1) * 32].T            # (T, D)
                hc = hfull - hfull.mean(1, keepdim=True)
                sig = hc.pow(2).mean(1, keepdim=True).sqrt()
                nh = hc / hc.norm(dim=1, keepdim=True)
                dc = delta - delta.mean(1, keepdim=True)
                dlin = (dc - nh * (nh * dc).sum(1, keepdim=True)) / sig
                dirs["lin_exact"].append(((dlin * gf) @ E.T).numpy())
                dirs["lin_const"].append(((dc * gf) @ E.T).numpy())
                if h in rule:
                    ap = torch.as_tensor(rule[h].attention(t, P.STATS)).double()
                    ctxp = (ap * (1 - eye)) @ v
                    dp = (ctxp - mc) @ Wo[:, h * 32:(h + 1) * 32].T
                    dirs["progattn_ov"].append(((dp - dp.mean(1, keepdim=True)) * gf @ E.T).numpy())
                    dirs["program"].append(M.earlier_votes(P, f"head{h}", t))
                dirs["attn_corpus"].append(skipgram_votes(a.numpy(), t, P.STATS, 0.0, 1.0, 1.0))
        print(f"head {h}")
        for k, xs in dirs.items():
            if not xs:
                continue
            row = []
            for c in ["inside", "onset", "after multi-token word", "other", "all"]:
                ys_, xs_, ws_ = [], [], []
                for p, pa, X, cc in zip(lp, lpa, xs, cls):
                    m = np.ones(len(cc), bool) if c == "all" else (cc == c)
                    ys_.append((p - pa)[m])
                    xs_.append(X[m])
                    ws_.append(np.exp(p[m]))
                val, beta = r2(ys_, xs_, ws_)
                row.append(f"{c[:8]} {val:.3f} (b {beta:.2f})")
            print(f"   {k:12s}", "  ".join(row), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
