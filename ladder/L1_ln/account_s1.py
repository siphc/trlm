"""Account of L1 (ln arm), seed 1 (v2). L1 is a one-block, one-head, attention-only character model with
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
EMBED = np.array([
    [-1.7, -1.1, 0.0, -5.5],  # EOS
    [-0.8, -2.2, -0.2, 2.1],  # UNK
    [-0.3, -0.8, 0.7, -1.5],  # \n
    [-1.4, 2.0, -0.4, -14.5],  # ␣
    [-3.3, -1.4, -1.2, -1.6],  # !
    [-0.1, -0.6, 0.5, -0.9],  # "
    [-0.1, 1.0, -0.5, -7.6],  # '
    [-3.5, -1.2, -1.3, -1.6],  # ,
    [0.0, -0.4, -0.5, -0.6],  # -
    [-3.8, -1.2, -1.4, -1.9],  # .
    [-1.1, -2.5, -0.3, 3.2],  # 0
    [-0.7, -2.2, -0.2, 1.8],  # 1
    [-0.6, -2.0, -0.1, 1.4],  # 2
    [-1.4, -2.5, -0.7, 5.8],  # 3
    [-0.6, -2.2, -0.1, 1.7],  # 4
    [-0.8, -2.3, -0.2, 2.3],  # 5
    [-2.4, -1.5, -0.9, -1.2],  # :
    [-2.0, -2.2, -0.8, 3.2],  # ;
    [-2.7, -1.2, -1.1, -2.2],  # ?
    [0.8, -1.1, 0.8, -0.5],  # A
    [-0.2, -1.2, 0.2, 2.5],  # B
    [0.5, -1.3, 0.6, 0.3],  # C
    [0.6, -1.3, 0.6, 0.3],  # D
    [-0.2, -1.5, 0.2, 2.3],  # E
    [0.9, -1.1, 0.6, -0.2],  # F
    [-0.2, -1.1, 0.1, 0.5],  # G
    [-0.1, -1.0, 0.1, 3.2],  # H
    [0.4, -0.6, 0.4, 0.2],  # I
    [0.4, -0.9, 0.3, 1.1],  # J
    [-0.6, -1.6, -0.2, 2.6],  # K
    [0.0, -1.0, 0.3, 1.9],  # L
    [0.2, -0.9, 0.4, 1.0],  # M
    [-0.5, -1.3, 0.0, 0.8],  # N
    [0.8, -1.1, 0.9, -0.6],  # O
    [-0.1, -1.1, 0.1, 0.6],  # P
    [-0.4, -1.8, 0.0, 0.4],  # Q
    [-0.5, -1.4, -0.1, 2.0],  # R
    [1.6, -1.1, 1.0, 1.0],  # S
    [1.3, -1.2, 1.1, 1.1],  # T
    [-0.5, -2.0, -0.1, 1.6],  # U
    [-0.6, -1.9, -0.6, 4.4],  # V
    [-0.2, -1.2, 0.3, 1.7],  # W
    [-0.6, -2.0, -0.2, 1.5],  # X
    [-0.5, -1.0, 0.1, 1.1],  # Y
    [0.2, -1.6, 0.1, 0.0],  # Z
    [3.0, 1.9, -0.5, 0.0],  # a
    [1.6, 0.6, -0.5, 2.8],  # b
    [0.9, 0.5, -0.6, 2.7],  # c
    [-0.7, 0.2, -0.7, 1.8],  # d
    [-0.9, 0.1, -1.2, -1.1],  # e
    [1.1, 0.6, -0.5, 2.1],  # f
    [0.0, 0.3, -0.5, 1.8],  # g
    [1.8, 0.5, -0.9, 5.4],  # h
    [2.8, 1.9, -0.5, -1.2],  # i
    [1.0, -0.1, -0.4, 2.1],  # j
    [-0.2, -0.1, -0.8, 2.7],  # k
    [0.7, 0.7, -0.6, 1.8],  # l
    [0.3, 0.3, -0.7, 2.7],  # m
    [0.0, 0.8, -0.4, -0.4],  # n
    [2.0, 1.4, -0.6, 0.3],  # o
    [1.0, 0.6, -0.6, 1.8],  # p
    [0.6, -0.3, -0.5, 1.6],  # q
    [0.0, 0.5, -0.7, 1.7],  # r
    [0.4, 0.6, -0.5, 2.1],  # s
    [0.9, 0.8, -0.6, 2.9],  # t
    [2.1, 1.5, -0.5, -1.1],  # u
    [-0.1, -0.3, -0.8, 3.8],  # v
    [1.4, 0.7, -0.5, 2.9],  # w
    [-0.5, -0.1, -0.6, -0.6],  # x
    [-0.8, 0.0, -0.9, 1.8],  # y
    [-0.4, -0.6, -0.7, 1.9],  # z
    [-0.5, -2.0, -0.1, 1.2],  # ¦
    [-0.5, -1.6, 0.4, 0.1],  # â
    [-1.7, -2.5, -0.3, -1.1],  # œ
    [-0.7, -2.2, -0.2, 2.0],  # ˜
    [-3.2, -4.2, -0.9, 9.2],  # “
    [-1.4, -2.8, -0.4, 4.2],  # ”
    [-4.2, -6.3, -0.5, 13.4],  # €
    [-0.6, -0.8, -0.2, -6.1],  # ™
])

POS = np.array([0.0, 1.1, -0.6, -0.4])  # the mean position vector, added everywhere

# The head's maps (residual -> head space and back). The ln1 gain is folded into WQ, WK and WV, and 1/sqrt(4) into WQ.
WQ = np.array([
    [-0.9, 0.1, 1.5, -0.2],
    [2.0, -1.1, 0.4, -0.3],
    [1.4, 2.3, 0.5, -3.0],
    [-1.7, 2.0, -0.2, -0.2],
])
WK = np.array([
    [-3, 0, 3, 1],
    [3, -5, 3, -5],
    [0, 3, 1, -2],
    [-1, 6, 0, 1],
])
WV = np.array([
    [-2, -1, 6, 1],
    [-3, -7, 3, 5],
    [-1, 2, 1, -1],
    [-1, 3, 0, -2],
])
WO = np.array([
    [-0.02, 0.21, -0.27, -0.28],
    [-1.72, -0.45, 0.04, 0.13],
    [0.16, -0.21, 0.19, 0.20],
    [0.14, 0.12, 0.02, -0.02],
])
OUT_GAIN = np.array([2.7, 3.0, 9.4, -0.8])  # final LayerNorm gain

BOUNDARY = [0, 2, 3, 4, 5, 6, 7, 9, 18] + list(range(19, 45))  # EOS \n ␣ ! " ' , . ? and the capitals A-Z


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
    {"name": "direct path", "site": "direct", "key": lambda X: X, "tests": [
        ("direct path matters", "resample direct", ">=", 1.0),
        ("position hardly matters", "positions_mean / resample direct", "<=", 0.05)]},
    {"name": "head", "site": "head0", "value": head, "tests": [
        ("attention is set by content", "content_share head0", ">=", 0.95),
        ("head favours boundary symbols", "attention_ratio head0 BOUNDARY", ">=", 1.2),
        ("head matters", "resample head0", ">=", 0.05),
        ("head hardly uses context", "context head0 / resample head0", "<=", 0.25)]},
]
