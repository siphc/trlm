"""Where does each head look from inside a split word? Attention mass by (query class, key role).

Query classes: inside L=2 / L=3 / L>=4 (current token is a piece, word started L-1 tokens back), onset (current
token starts a word that usually continues), after a multi-token word, other. Key roles relative to the query:
self, word start (t_j0, only for inside queries), other earlier pieces of the current word, the token just before
the word, the previous word's pieces, everything else.

  python -m analysis.investigations.word.attn_dump
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

from analysis.fitlib import train_windows
from analysis.investigations.word import wordstats as WS
from analysis.investigations.word.diag import classes
from analysis.mech_eval import EVAL_SEED, program
from harness.common import load_model


def attention(model, x):
    blk = model.blocks[0]
    B, T = x.shape
    H, Dh, D = model.cfg.n_heads, model.cfg.d_head, model.cfg.d_model
    x0 = model.wte(x) + model.wpe(torch.arange(T))[None]
    q, k, v = blk.attn.qkv(blk.ln1(x0)).split(D, dim=-1)
    q = q.view(B, T, H, Dh).transpose(1, 2)
    k = k.view(B, T, H, Dh).transpose(1, 2)
    sc = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
    sc = sc.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
    return sc.softmax(-1)


def main(n=32):
    P = program()
    PIECE = WS.piece_flags()
    model = load_model("attn_s0")
    W = train_windows(n, seed=EVAL_SEED)
    roles = ["self", "word start", "earlier pieces of word", "token before word", "previous 1-3 tokens (other)", "rest"]
    acc = {}
    with torch.no_grad():
        for b in range(0, n, 8):
            x = torch.as_tensor(W[b : b + 8, :256])
            A = attention(model, x).numpy()                    # (B, H, T, T)
            for bi in range(x.shape[0]):
                t = x[bi].numpy()
                cl, Ls = classes(P, t, PIECE)
                for i in range(1, len(t)):
                    c = cl[i]
                    L = Ls[i]
                    j0 = i - L + 1
                    role = np.full(i + 1, 5)
                    role[max(i - 3, 0) : i] = 4
                    if L >= 2:
                        role[j0 + 1 : i] = 2
                        role[j0] = 1
                        if j0 >= 1:
                            role[j0 - 1] = 3
                    role[i] = 0
                    a = A[bi, :, i, : i + 1]                     # (H, i+1)
                    r = acc.setdefault(c, np.zeros((4, len(roles) + 1)))
                    for k in range(len(roles)):
                        r[:, k] += a[:, role == k].sum(1)
                    r[:, -1] += 1
    out = {}
    for c, r in sorted(acc.items()):
        m = r[:, -1:]
        out[c] = {f"head{h}": {roles[k]: round(float(r[h, k] / m[h, 0]), 3) for k in range(len(roles))}
                  for h in range(4)}
        out[c]["n"] = int(m[0, 0])
        print(c, int(m[0, 0]))
        for h in range(4):
            print(f"   head{h}", "  ".join(f"{roles[k]} {r[h, k] / m[h, 0]:.3f}" for k in range(len(roles))))
    with open(os.path.join(WS.OUT, "attn_dump.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
