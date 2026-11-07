"""Build the L1 account (v2) for one seed: ladder/L1_<arm>/account_s<seed>.py.

    .venv/bin/python -m analysis.ladder.l1.build_account --arm ln --seed 0

v2 is written for decision 19: it must be shorter than the quantised weights (P2), and every behavioural statement
in its description carries a test that holds on all five seeds and fails on the untrained model (P4).

The constants come from the model's weights:
- EMBED: the token embedding, rounded to 1 decimal.
- POS: the mean of the 64 position vectors, so position is dropped. Replacing every position vector by the mean
  costs 0.001-0.026 nats on the five ln seeds, below every seed floor.
- WQ, WK, WV: the head's query, key and value maps with the ln1 gain folded in; WQ also carries 1/sqrt(d_head).
- WO: the output map. OUT_GAIN: the final LayerNorm gain.
- WQ, WK, WV, WO and OUT_GAIN are rounded to 2 decimals.
BOUNDARY is part of the structure, the same on every seed: the ids of the space, EOS, newline, the punctuation and
quote marks, and the capital letters. The v1 account is in build_account_v1.py.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from analysis.ladder.common import REPO, load
from analysis.ladder.dump import symbol_names

BOUNDARY_NAMES = ["␣", "EOS", "\\n", ".", "!", "?", ",", '"', "'"]


def fmt(x: float, p: int) -> str:
    s = f"{x:.{p}f}"
    return "0." + "0" * p if s.startswith("-") and float(s) == 0 else s


def arr(M, p: int, indent: str = "    ") -> str:
    M = np.asarray(M)
    if M.ndim == 1:
        return "np.array([" + ", ".join(fmt(x, p) for x in M) + "])"
    return "np.array([\n" + "".join(f"{indent}[" + ", ".join(fmt(x, p) for x in r) + "],\n" for r in M) + "])"


def constants(model) -> dict:
    S = {k: v.detach().double().numpy() for k, v in model.state_dict().items()}
    d, dh = model.cfg.d_model, model.cfg.d_head
    assert model.cfg.n_heads == 1
    g1 = S.get("blocks.0.ln1.weight", np.ones(d))
    Wq, Wk, Wv = np.split(S["blocks.0.attn.qkv.weight"], 3, 0)
    return {"EMBED": S["wte.weight"], "POS": S["wpe.weight"].mean(0), "WQ": Wq * g1 / np.sqrt(dh), "WK": Wk * g1,
            "WV": Wv * g1, "WO": S["blocks.0.attn.out.weight"], "OUT_GAIN": S.get("ln_f.weight", np.ones(d)),
            "LAYERNORM": model.cfg.layernorm}


def render(c: dict, arm: str, seed: int) -> str:
    names = symbol_names()
    embed = "np.array([\n" + "".join(f"    [{', '.join(fmt(x, 1) for x in r)}],  # {nm}\n"
                                     for r, nm in zip(c["EMBED"], names)) + "])"
    boundary = [i for i, nm in enumerate(names) if nm in BOUNDARY_NAMES or (len(nm) == 1 and nm.isupper())]
    return TEMPLATE.format(arm=arm, seed=seed, embed=embed, pos=arr(c["POS"], 2), wq=arr(c["WQ"], 2),
                           wk=arr(c["WK"], 2), wv=arr(c["WV"], 2), wo=arr(c["WO"], 2), gain=arr(c["OUT_GAIN"], 2),
                           layernorm=c["LAYERNORM"], boundary=boundary,
                           boundary_names=" ".join(names[i] for i in boundary))


TEMPLATE = '''"""Account of L1 ({arm} arm), seed {seed} (v2). L1 is a one-block, one-head, attention-only character model with
4 dimensions and a 64-symbol window. Built by analysis/ladder/l1/build_account.py. The code and the description are
the same for every seed; only the numbers differ.

What the model computes, top-down. Each statement that says how the model behaves has a test in CLAIMS.

1. Each symbol is a point in 4 dimensions (EMBED). The same table reads the current symbol and scores every
   candidate next symbol (tied embedding).
2. The model ignores where a symbol is in the window: every position adds the same vector POS. Test: replacing the
   model's 64 position vectors by their mean costs at most 5% of what removing the direct path costs.
3. Direct path. The current symbol's vector goes straight to the output, and it carries most of the prediction.
   Test: removing it costs at least 1 nat per symbol.
4. The head mixes in the earlier symbols' value vectors (WV). Its attention weights come from a query map (WQ) and a
   key map (WK) applied to the symbols' vectors:
   - Attention is set by content. Test: the current and earlier symbols together explain at least 95% of the
     variance of the log attention weights.
   - The head favours boundary symbols: space, EOS, newline, punctuation, quote marks and capital letters (BOUNDARY).
     Test: their attention weight is at least 1.2 times their share of the keys.
   - The head matters, but mostly as a fixed correction for the current symbol, not as a use of context. Tests:
     removing it costs at least 0.05 nats; replacing its output by its mean for the current symbol costs at most a
     quarter of that.
5. Output. The direct path plus the head's output (mapped back by WO) is layer-normalised, scaled by OUT_GAIN and
   scored against every symbol's vector.

Claims for P3. The direct path depends only on the symbol (key claim). The head's output is computed by head(X)
below (value claim).
"""
import numpy as np

LAYERNORM = {layernorm}

# EMBED[s]: symbol s's 4 numbers (the model's embedding to 1 decimal), in symbol-id order.
EMBED = {embed}

POS = {pos}  # the mean position vector, added everywhere

# The head's maps (residual -> head space and back). The ln1 gain is folded into WQ, WK and WV, and 1/sqrt(4) into WQ.
WQ = {wq}
WK = {wk}
WV = {wv}
WO = {wo}
OUT_GAIN = {gain}  # final LayerNorm gain

BOUNDARY = {boundary}  # {boundary_names}


def norm(x):
    if not LAYERNORM:
        return x
    x = x - x.mean(-1, keepdims=True)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5)


def direct(X):
    return EMBED[X] + POS


def head(X):
    u = norm(direct(X))
    s = u @ WQ.T @ (u @ WK.T).transpose(0, 2, 1)
    s = np.where(np.tri(X.shape[1], dtype=bool), s, -np.inf)
    w = np.exp(s - s.max(-1, keepdims=True))
    return w / w.sum(-1, keepdims=True) @ (u @ WV.T)


def predict(X):
    X = np.asarray(X)
    logit = norm(direct(X) + head(X) @ WO.T) * OUT_GAIN @ EMBED.T
    logit = logit - logit.max(-1, keepdims=True)
    return logit - np.log(np.exp(logit).sum(-1, keepdims=True))


# Behaviour tests (P4): each measures the model through the probe (analysis/ladder/behaviour.py).

def ratio(a, b):
    return a / b if b > 0.01 else np.inf  # undefined when the reference part does not matter


def mean_for_current_symbol(X, z):
    flat = z.reshape(X.size, -1)
    total = np.zeros((len(EMBED), flat.shape[1]))
    np.add.at(total, X.ravel(), flat)
    return (total / np.maximum(np.bincount(X.ravel(), minlength=len(EMBED)), 1)[:, None])[X].reshape(z.shape)


def content_share(p):
    # queries t >= 32 (every 4th), keys 16 .. t; log weights centred per query
    q, k, logw = [], [], []
    for t in range(32, p.X.shape[1], 4):
        row = np.log(p.attn[:, 0, t, 16:t + 1] + 1e-30)
        logw.append((row - row.mean(1, keepdims=True)).ravel())
        q.append(np.repeat(p.X[:, t], t - 15))
        k.append(p.X[:, 16:t + 1].ravel())
    q, k, logw = (np.concatenate(a) for a in (q, k, logw))
    pair = q * len(EMBED) + k
    mean = np.bincount(pair, logw) / np.maximum(np.bincount(pair), 1)
    return 1 - ((logw - mean[pair]) ** 2).mean() / (logw ** 2).mean()


def boundary_ratio(p):
    b = np.isin(p.X, BOUNDARY)
    mass = (p.attn[:, 0, 16:] * b[:, None, :]).sum(-1).mean()
    share = np.mean([b[:, :t + 1].mean() for t in range(16, p.X.shape[1])])
    return mass / share


CLAIMS = [
    {{"name": "direct path", "site": "direct", "key": lambda X: X, "tests": [
        {{"statement": "removing the direct path costs at least 1 nat",
         "measure": lambda p: p.cost_resample("direct"), "op": ">=", "threshold": 1.0}},
        {{"statement": "position costs at most 5% of the direct path",
         "measure": lambda p: ratio(p.cost_positions(0 * p.positions + p.positions.mean(0)), p.cost_resample("direct")),
         "op": "<=", "threshold": 0.05}}]}},
    {{"name": "head", "site": "head0", "value": head, "tests": [
        {{"statement": "attention is set by content (at least 95% of the variance)",
         "measure": content_share, "op": ">=", "threshold": 0.95}},
        {{"statement": "the head favours boundary symbols (at least 1.2 times their share)",
         "measure": boundary_ratio, "op": ">=", "threshold": 1.2}},
        {{"statement": "removing the head costs at least 0.05 nats",
         "measure": lambda p: p.cost_resample("head0"), "op": ">=", "threshold": 0.05}},
        {{"statement": "context is at most a quarter of the head's effect",
         "measure": lambda p: ratio(p.cost_replaced(z=mean_for_current_symbol(p.X, p.z)), p.cost_resample("head0")),
         "op": "<=", "threshold": 0.25}}]}},
]
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ln")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    c = constants(load("L1", a.arm, a.seed).double())
    path = os.path.join(REPO, "ladder", f"L1_{a.arm}", f"account_s{a.seed}.py")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(render(c, a.arm, a.seed))
    print(path)


if __name__ == "__main__":
    main()
