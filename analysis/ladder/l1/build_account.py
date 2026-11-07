"""Build the L1 account for one seed: derive its named tables from the model's weights and write
ladder/L1_<arm>/account_s<seed>.py.

    .venv/bin/python -m analysis.ladder.l1.build_account --arm ln --seed 0

Every account file has the same structure (the template below); only the tables differ per seed. Each table is derived
by a stated decomposition of the weights:
- EMBED: the token embedding. POS_FIRST: position 0. POS_REST: the mean of positions 1-63.
- The head's score between query symbol a and key symbol b at key position j is
      LAM1[a] * KAP1[b] + LAM2[a] * KAP2[b] + ALPHA[a] * RHO[j] * BETA[b].
  The first two terms come from the head's score table at the mean position (q and k computed from
  LN(EMBED + POS_REST)). They are the two leading components of a frequency-weighted SVD after the per-query mean is
  removed; softmax ignores a per-query constant. The third term is a rank-1 fit (alternating least squares, frequency
  weighted) of how the true key positions change the score.
- VALUE[b]: the head's value vector for key symbol b at the mean position. HEAD_OUT: the head's output projection.
- OUT_GAIN: the final LayerNorm gain.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from analysis.ladder.common import REPO, load
from analysis.ladder.dump import symbol_names
from src.train import data_char

N_COMP = 3   # score components kept: with d_head 4, three is the whole row-centred score table


def ln(x, g):
    x = x - x.mean(-1, keepdims=True)
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-5) * g


def frequencies() -> np.ndarray:
    a = data_char.load_alphabet()
    # EOS count: number of stories in train (one EOS each), as used by compress_probe
    freq = np.array([2_099_719, a["unk_count"]] + a["counts"], np.float64)
    return freq / freq.sum()


def tables(model) -> dict:
    S = {k: v.detach().double().numpy() for k, v in model.state_dict().items()}
    E, P = S["wte.weight"], S["wpe.weight"]
    T, d = P.shape
    dh = model.cfg.d_head
    assert model.cfg.n_heads == 1
    g1 = S["blocks.0.ln1.weight"] if model.cfg.layernorm else np.ones(d)
    gf = S["ln_f.weight"] if model.cfg.layernorm else None
    Wq, Wk, Wv = np.split(S["blocks.0.attn.qkv.weight"], 3, 0)
    f = frequencies()
    p_rest = P[1:].mean(0)
    u = ln(E + p_rest, g1) if model.cfg.layernorm else E + p_rest
    Q, K, V = u @ Wq.T, u @ Wk.T, u @ Wv.T
    score = Q @ K.T / np.sqrt(dh)
    score = score - (score @ f)[:, None]
    sf = np.sqrt(f)
    U, s, Vt = np.linalg.svd(sf[:, None] * score * sf[None], full_matrices=False)
    lam = U[:, :N_COMP] * s[:N_COMP] / sf[:, None]
    kap = Vt[:N_COMP].T / sf[:, None]
    # scale each key scale to max |KAP| = 1, sign so that the space sits at the positive end of the first scale and
    # EOS at the positive end of the others
    space = 2 + data_char.load_alphabet()["chars"].index(" ")
    for r in range(N_COMP):
        ref = space if r == 0 else 0
        c = np.abs(kap[:, r]).max() * (1 if kap[ref, r] >= 0 else -1)
        kap[:, r] /= c
        lam[:, r] *= c
    # position effect on the score: D[a, j, b]
    ub = (ln(E[None] + P[:, None], g1) if model.cfg.layernorm else E[None] + P[:, None])
    D = np.einsum("ad,jbd->ajb", Q, ub @ Wk.T - K[None]) / np.sqrt(dh)
    alpha, rho, beta = np.ones(len(E)), D.mean((0, 2)), np.ones(len(E))
    for _ in range(100):
        alpha = np.einsum("ajb,j,b,b->a", D, rho, beta, f) / ((rho ** 2).sum() * (beta ** 2 * f).sum())
        rho = np.einsum("ajb,a,a,b,b->j", D, alpha, f, beta, f) / ((alpha ** 2 * f).sum() * (beta ** 2 * f).sum())
        beta = np.einsum("ajb,a,a,j->b", D, alpha, f, rho) / ((alpha ** 2 * f).sum() * (rho ** 2).sum())
    cb, cr = np.abs(beta).max() * np.sign(beta @ f), rho[0]
    beta, rho, alpha = beta / cb, rho / cr, alpha * cb * cr
    return {"EMBED": E, "POS_FIRST": P[0], "POS_REST": p_rest, "LAM": lam, "KAP": kap, "ALPHA": alpha, "BETA": beta, "RHO": rho, "VALUE": V,
            "HEAD_OUT": S["blocks.0.attn.out.weight"], "OUT_GAIN": gf, "D_HEAD": dh, "LAYERNORM": model.cfg.layernorm}


def num(x: float, p: int = 3) -> str:
    return f"{x:+.{p}f}"


def render(t: dict, arm: str, seed: int) -> str:
    names = symbol_names()
    rows = []
    for i, nm in enumerate(names):
        tup = lambda v: "(" + ", ".join(num(x) for x in v) + ")"
        cells = [tup(t["EMBED"][i]), "(" + ", ".join(num(x, 2) for x in t["LAM"][i]) + ")", tup(t["KAP"][i]),
                 num(t["ALPHA"][i], 2), num(t["BETA"][i]), tup(t["VALUE"][i])]
        label = repr(nm) + ","
        rows.append(f"    ({label:8s} " + ", ".join(cells) + "),")
    rho = ", ".join(num(v) for v in t["RHO"])
    vec = lambda v: "np.array([" + ", ".join(num(x) for x in v) + "])"
    mat = lambda M: "np.array([\n" + "".join("    [" + ", ".join(num(x) for x in r) + "],\n" for r in M) + "])"
    gain = vec(t["OUT_GAIN"]) if t["OUT_GAIN"] is not None else "None"
    return TEMPLATE.format(arm=arm, seed=seed, rows="\n".join(rows), rho=rho, pos_first=vec(t["POS_FIRST"]),
                           pos_rest=vec(t["POS_REST"]), head_out=mat(t["HEAD_OUT"]), out_gain=gain,
                           d_head=t["D_HEAD"], layernorm=t["LAYERNORM"])


TEMPLATE = '''"""Account of L1 ({arm} arm), seed {seed}: a one-block, one-head, attention-only character model with 4 dimensions.
Generated by analysis/ladder/l1/build_account.py; the structure is the same for every seed, the tables differ.

Top-down, for each position t of a window of symbols c_0 .. c_63:

1. Direct path. x0_t = EMBED[c_t] + (POS_FIRST if t == 0 else POS_REST).
   The position matters to the direct path only at the first position of the window.
2. Head: a one-dimensional search, plus a small second preference and a discount at the window start.
   Every earlier symbol b at position j <= t gets the score
       LAM1[a] * KAP1[b]  +  LAM2[a] * KAP2[b]  +  ALPHA[a] * RHO[j] * BETA[b],     a = c_t, the current symbol.
   - KAP1 is each symbol's place on one scale: spaces and the vowels i, u at +1; capital letters, sentence
     punctuation and EOS near -1 (mojibake characters beyond it).
   - LAM1 is the current symbol's signed sharpness on that scale. Letters have LAM1 > 0, so they look for spaces;
     a space has LAM1 < 0, so it looks for the most recent capital letter, sentence end or story start.
   - KAP2 / LAM2: a smaller second scale (story start EOS and capitals high; vowels low); punctuation and EOS
     queries prefer its high end.
   - RHO[j] discounts keys near the start of the window (RHO[0] = 1, falling to about 0 by position 32); ALPHA, BETA
     say how strongly each query and key symbol feel it.
   The weights are the softmax of the scores over j <= t; the head's output is the weighted mean of VALUE[c_j].
   Capitals share one value pattern, sentence punctuation another, EOS and the space their own; so the head reports
   which kind of marker the search found.
3. Output. The residual x0_t + HEAD_OUT @ head_t is layer-normalised, scaled by OUT_GAIN, and scored against every
   symbol's embedding (tied input and output): logit(b) = <scaled residual, EMBED[b]>. With only 4 tied dimensions
   this is a squeezed bigram; the best possible 4-dimension tied bigram of this shape reaches 2.638 nats on dev_full
   against 2.299 for the true bigram, which is why L1 is worse than the bigram.

Claims (P3): the direct path depends only on (symbol, first position or not) [key claim]; the head computes the
search above [value claim].
"""
import numpy as np

D_HEAD = {d_head}
LAYERNORM = {layernorm}

# One row per symbol, in symbol-id order:
#   (symbol, EMBED (4), LAM (3), KAP (3), ALPHA, BETA, VALUE (4))
# EMBED: the symbol's embedding (input and output); LAM/KAP: query and key loadings of the head's three score
# components; ALPHA/BETA: query and key sensitivity to the window-start discount; VALUE: what the head reads from
# the symbol. All from the model's weights (see the generator's docstring).
SYMBOL_ROWS = [
{rows}
]

# RHO[j]: the window-start discount profile over key positions j = 0 .. 63 (RHO[0] = 1).
RHO = np.array([{rho}])
POS_FIRST = {pos_first}   # position 0's embedding
POS_REST = {pos_rest}     # the mean embedding of positions 1-63
HEAD_OUT = {head_out}     # the head's output projection (residual x head)
OUT_GAIN = {out_gain}     # final LayerNorm gain


_cols = list(zip(*SYMBOL_ROWS))
EMBED, LAM, KAP, ALPHA, BETA, VALUE = (np.array(c) for c in _cols[1:])


def layer_norm(x, gain):
    if not LAYERNORM:
        return x
    x = x - x.mean(-1, keepdims=True)
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-5) * gain


def direct_path(X):
    """x0 (n, T, 4): the symbol's embedding plus the position vector (first position or not)."""
    pos = np.where(np.arange(X.shape[1])[:, None] == 0, POS_FIRST, POS_REST)
    return EMBED[X] + pos[None]


def head(X):
    """The head's output (n, T, D_HEAD): the softmax-weighted mean of VALUE over the search scores."""
    n, T = X.shape
    a, b = X[:, :, None], X[:, None, :]
    score = np.einsum("ntr,nsr->nts", LAM[X], KAP[X]) + ALPHA[a] * RHO[None, None, :T] * BETA[b]
    score = np.where(np.tril(np.ones((T, T), bool)), score, -np.inf)
    w = np.exp(score - score.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    return np.einsum("nts,nsd->ntd", w, VALUE[X])


def output(x0, z):
    logits = layer_norm(x0 + z @ HEAD_OUT.T, OUT_GAIN) @ EMBED.T
    logits = logits - logits.max(-1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(-1, keepdims=True))


def predict(X):
    X = np.asarray(X, np.int64)
    return output(direct_path(X), head(X))


def key_direct(X):
    """Direct-path claim: the embedding depends only on the symbol and on whether the position is the first."""
    t = np.broadcast_to(np.arange(X.shape[1]) == 0, X.shape)
    return np.stack([X, t], -1)


CLAIMS = [
    {{"name": "direct path: symbol and first-position flag", "site": "direct", "key": key_direct}},
    {{"name": "head: one-dimensional search for markers", "site": "head0", "value": head}},
]
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ln")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t = tables(load("L1", a.arm, a.seed).double())
    path = os.path.join(REPO, "ladder", f"L1_{a.arm}", f"account_s{a.seed}.py")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(render(t, a.arm, a.seed))
    print(path)


if __name__ == "__main__":
    main()
