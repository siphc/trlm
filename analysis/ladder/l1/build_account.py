"""Build the L1 account (v2) for one seed: ladder/L1_<arm>/account_s<seed>.py.

    .venv/bin/python -m analysis.ladder.l1.build_account --arm ln --seed 0

v2 is written for decision 19: it must be shorter than the quantised weights (P2), and every behavioural statement
in its description carries a test that holds on all five seeds and fails on the untrained model (P4).

The constants come from the model's weights:
- EMBED: the token embedding.
- POS: the mean of the 64 position vectors, so position is dropped. Replacing every position vector by the mean
  costs 0.001-0.026 nats on the five ln seeds, below every seed floor.
- WQ, WK, WV: the head's query, key and value maps with the ln1 gain folded in; WQ also carries 1/sqrt(d_head).
- WO: the output map. OUT_GAIN: the final LayerNorm gain.
- Each constant is rounded to the fewest decimals (2 down to 0) that keep KL at or below the P1 bar, the same greedy
  rule as the text weights reference (decision 21; choose_decimals).
BOUNDARY is part of the structure, the same on every seed: the ids of the space, EOS, newline, the punctuation and
quote marks, and the capital letters. The v1 account is in build_account_v1.py.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from analysis.ladder.common import REPO, load
from analysis.ladder.dump import symbol_names

MIN_HEAD_FAITH = 0.96
BOUNDARY_NAMES = ["␣", "EOS", "\\n", ".", "!", "?", ",", '"', "'"]


def fmt(x: float, p: int) -> str:
    s = f"{x:.{p}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


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


def render(c: dict, arm: str, seed: int, dec: dict) -> str:
    names = symbol_names()
    embed = "np.array([\n" + "".join(f"    [{', '.join(fmt(x, dec['EMBED']) for x in r)}],  # {nm}\n"
                                     for r, nm in zip(c["EMBED"], names)) + "])"
    marks = [i for i, nm in enumerate(names) if nm in BOUNDARY_NAMES]
    caps = [i for i, nm in enumerate(names) if len(nm) == 1 and nm.isupper()]
    assert caps == list(range(caps[0], caps[-1] + 1)), "capitals are contiguous in symbol-id order"
    boundary = f"{marks} + list(range({caps[0]}, {caps[-1] + 1}))"
    return TEMPLATE.format(arm=arm, seed=seed, embed=embed, pos=arr(c["POS"], dec["POS"]), wq=arr(c["WQ"], dec["WQ"]),
                           wk=arr(c["WK"], dec["WK"]), wv=arr(c["WV"], dec["WV"]), wo=arr(c["WO"], dec["WO"]),
                           gain=arr(c["OUT_GAIN"], dec["OUT_GAIN"]),
                           boundary=boundary,
                           boundary_names=" ".join(names[i] for i in marks) + f" and the capitals {names[caps[0]]}-{names[caps[-1]]}")


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

# EMBED[s]: symbol s's 4 numbers (the model's embedding), in symbol-id order.
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


# Behaviour tests (P4): named measurements of the model (analysis/ladder/behaviour.py; decision 20).
CLAIMS = [
    {{"name": "direct path", "site": "direct", "key": lambda X: X, "tests": [
        ("direct path matters", "resample direct", ">=", 1.0),
        ("position hardly matters", "positions_mean / resample direct", "<=", 0.05)]}},
    {{"name": "head", "site": "head0", "value": head, "tests": [
        ("attention is set by content", "content_share head0", ">=", 0.95),
        ("head favours boundary symbols", "attention_ratio head0 BOUNDARY", ">=", 1.2),
        ("head matters", "resample head0", ">=", 0.05),
        ("head hardly uses context", "context head0 / resample head0", "<=", 0.25)]}},
]
'''


def choose_decimals(model, c: dict, bar: float) -> dict:
    """Fewest decimals per constant (from 2 down to 0, largest constant first) with KL(model || account) on dev_full
    at most the P1 bar: the same greedy rule as the text weights reference (decision 21); in addition the head claim
    (replacement test) must keep faithfulness >= MIN_HEAD_FAITH, a margin over P3's 0.95."""
    import torch

    from analysis.ladder.common import kl_pos, model_logprobs, windows
    from analysis.ladder.l1.fit_account import Account

    from analysis.ladder.scrub import finish, parts, scrubbed

    W = windows("L1", "dev_full")
    X = W[:, :-1].astype(np.int64)
    lp_m = model_logprobs(model, W)
    x0, z = parts(model, X)
    lp64 = finish(model, x0, z)
    k_resample = kl_pos(lp64, scrubbed(model, x0, z, {"head0": None}, np.random.default_rng(0))).mean()

    def ok(dec):
        acc = Account({**c, **{k: np.round(c[k], d) for k, d in dec.items()}}, 0)
        with torch.no_grad():
            lp, zh = [], []
            for i in range(0, len(W), 200):
                x = torch.as_tensor(X[i:i + 200])
                lp.append(torch.log_softmax(acc(x), -1).numpy())
                zh.append(acc.head(x).numpy())
        kl = float(kl_pos(lp_m, np.concatenate(lp)).mean())
        zr = z.copy()
        zr[:, :, 0] = np.concatenate(zh)
        faith = 1 - kl_pos(lp64, finish(model, x0, zr)).mean() / k_resample
        return kl <= bar and faith >= MIN_HEAD_FAITH

    keys = sorted(["EMBED", "POS", "WQ", "WK", "WV", "WO", "OUT_GAIN"], key=lambda k: -np.size(c[k]))
    dec = {k: 2 for k in keys}
    assert ok(dec), "the account fails the bar at 2 decimals"
    for k in keys:
        while dec[k] > 0 and ok({**dec, k: dec[k] - 1}):
            dec[k] -= 1
    return dec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ln")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from analysis.ladder.length import p1_bar

    model = load("L1", a.arm, a.seed).double()
    c = constants(model)
    dec = choose_decimals(model, c, p1_bar("L1", a.arm, a.seed))
    path = os.path.join(REPO, "ladder", f"L1_{a.arm}", f"account_s{a.seed}.py")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(render(c, a.arm, a.seed, dec))
    print(path, dec)


if __name__ == "__main__":
    main()
