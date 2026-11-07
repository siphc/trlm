"""The weights-reference forward pass for R1 (logs/ladder/design.md): the shortest plain NumPy code that runs a
one-block attention-only TinyLM from its weight arrays. Its compressed source plus the compressed quantised weights
is the description length an account must beat.

forward(X, W) -> (n, T, V) log-probs; W maps the state_dict names of src/train/model.py to arrays.
"""
import numpy as np


def ln(x, g):
    if g is None:
        return x
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * g


def forward(X, W, n_heads):
    E, P = W["wte.weight"], W["wpe.weight"]
    n, T = X.shape
    d = E.shape[1]
    dh = d // n_heads
    x = E[X] + P[:T]
    q, k, v = np.split(ln(x, W.get("blocks.0.ln1.weight")) @ W["blocks.0.attn.qkv.weight"].T, 3, -1)
    q, k, v = (a.reshape(n, T, n_heads, dh).transpose(0, 2, 1, 3) for a in (q, k, v))
    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh)
    s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
    a = np.exp(s - s.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    z = (a @ v).transpose(0, 2, 1, 3).reshape(n, T, d)
    h = ln(x + z @ W["blocks.0.attn.out.weight"].T, W.get("ln_f.weight")) @ E.T
    return h - np.logaddexp.reduce(h, -1, keepdims=True)
