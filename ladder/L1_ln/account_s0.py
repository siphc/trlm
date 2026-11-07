"""Account of L1 (ln arm), seed 0 (v2). L1 is a one-block, one-head, attention-only character model with
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
    [-1.5, 0.3, -2.7, -2.2],  # EOS
    [1.3, -0.6, -1.2, -0.7],  # UNK
    [-0.3, 1.2, -0.3, -0.4],  # \n
    [-0.8, -0.1, 5.9, -7.3],  # ␣
    [-2.3, -0.9, -2.6, -1.3],  # !
    [-0.7, 0.6, -0.1, -0.5],  # "
    [-1.3, -0.9, -1.1, -1.2],  # '
    [-2.4, -1.0, -1.9, -1.2],  # ,
    [-0.5, -0.8, -1.7, -0.2],  # -
    [-3.3, -1.0, -2.9, -1.3],  # .
    [0.6, -0.9, -3.6, 1.3],  # 0
    [0.7, -0.9, -3.3, 1.1],  # 1
    [0.5, -0.8, -3.6, 1.1],  # 2
    [0.0, -0.3, -2.2, 0.2],  # 3
    [0.9, -0.8, -2.7, 0.5],  # 4
    [0.5, -0.9, -3.6, 1.2],  # 5
    [-1.3, -0.8, -2.6, -0.6],  # :
    [-0.5, -0.7, -2.9, -0.2],  # ;
    [-1.8, -0.9, -2.4, -1.3],  # ?
    [0.9, 0.9, -0.4, 1.0],  # A
    [1.6, 0.7, 0.1, 2.2],  # B
    [1.1, 0.5, -0.4, 1.2],  # C
    [1.2, 0.5, -0.3, 1.5],  # D
    [1.0, 0.8, -0.5, 0.8],  # E
    [0.9, 0.7, -0.5, 1.0],  # F
    [0.7, 0.3, -0.7, 0.8],  # G
    [1.9, 0.7, 0.3, 2.8],  # H
    [1.2, 1.0, -0.1, 1.5],  # I
    [1.5, 0.7, -0.1, 1.8],  # J
    [0.3, 0.0, -1.0, 0.6],  # K
    [1.8, 0.8, 0.2, 2.3],  # L
    [1.6, 0.8, 0.0, 1.9],  # M
    [0.9, 0.4, -0.5, 1.2],  # N
    [1.1, 1.2, -0.3, 1.1],  # O
    [0.8, 0.2, -0.7, 1.1],  # P
    [0.7, -0.7, -2.6, 1.1],  # Q
    [0.6, 0.0, -0.8, 1.0],  # R
    [2.5, 1.0, 0.5, 2.8],  # S
    [2.7, 1.2, 0.7, 2.9],  # T
    [0.0, 0.0, -1.7, 0.0],  # U
    [-0.1, -0.9, -2.7, 0.5],  # V
    [1.7, 0.7, 0.0, 2.0],  # W
    [0.7, -0.8, -3.1, 0.8],  # X
    [1.3, 0.6, -0.2, 1.7],  # Y
    [1.9, -0.3, -0.6, 0.4],  # Z
    [0.8, -1.1, 2.5, 0.4],  # a
    [0.9, -1.0, 0.7, 2.2],  # b
    [0.4, -0.9, 1.1, 1.4],  # c
    [-1.2, -0.7, 0.7, 0.2],  # d
    [-2.1, -1.2, -0.9, -1.4],  # e
    [0.2, -0.8, 0.8, 1.3],  # f
    [-0.3, -0.7, 0.8, 0.8],  # g
    [1.0, -1.5, 0.3, 3.0],  # h
    [0.5, -1.0, 2.5, -0.6],  # i
    [0.9, -0.6, -0.3, 1.7],  # j
    [-0.3, -1.1, -0.1, 1.1],  # k
    [-0.2, -1.0, 1.0, 0.7],  # l
    [-0.2, -0.9, 0.7, 1.1],  # m
    [-0.5, -0.7, 1.7, 0.0],  # n
    [0.1, -1.2, 1.4, 0.3],  # o
    [-0.1, -1.0, 0.7, 0.9],  # p
    [0.2, -0.9, -1.1, 0.7],  # q
    [-0.7, -1.0, 0.7, 0.5],  # r
    [-0.2, -0.9, 1.3, 1.0],  # s
    [0.3, -1.0, 1.6, 1.7],  # t
    [0.4, -0.9, 2.1, -0.6],  # u
    [0.4, -1.1, -0.1, 2.0],  # v
    [0.7, -0.9, 1.2, 1.9],  # w
    [-1.0, -0.6, -0.7, -0.6],  # x
    [-1.2, -1.0, -0.1, 0.1],  # y
    [0.1, -1.0, -1.0, 0.9],  # z
    [0.8, -0.8, -3.0, 0.7],  # ¦
    [-2.9, -0.6, -6.3, 3.0],  # â
    [1.3, 1.0, 1.0, -3.6],  # œ
    [1.5, -0.5, -0.9, -0.8],  # ˜
    [0.2, -0.8, -3.5, 0.7],  # “
    [0.6, -0.8, -3.2, 0.9],  # ”
    [-2.4, -0.6, -4.8, -3.4],  # €
    [2.8, 0.5, -0.2, 0.3],  # ™
])

POS = np.array([-0.6, -1.3, 0.9, 0.6])  # the mean position vector, added everywhere

# The head's maps (residual -> head space and back). The ln1 gain is folded into WQ, WK and WV, and 1/sqrt(4) into WQ.
WQ = np.array([
    [-1.7, 1.5, 0.3, -1.8],
    [0.6, -0.1, 0.7, -1.5],
    [0.6, 0.0, 0.7, -1.7],
    [0.0, -1.6, 0.3, 0.5],
])
WK = np.array([
    [-0.1, 1.0, -4.0, 0.0],
    [-0.3, 0.3, -2.8, 4.0],
    [-0.2, 0.3, -3.0, 3.2],
    [-1.7, -0.5, 4.7, 0.3],
])
WV = np.array([
    [1.0, 0.8, -1.3, 1.2],
    [0.0, 0.0, 0.0, 0.0],
    [-1.2, 3.0, 0.0, -0.6],
    [-7.7, -3.4, 1.2, 1.1],
])
WO = np.array([
    [-0.1, 0.0, 0.2, -0.2],
    [-0.1, 0.0, 0.2, 0.3],
    [-0.1, 0.0, -0.7, 0.4],
    [0.2, 0.0, -0.1, -0.2],
])
OUT_GAIN = np.array([2.7, 8.6, 1.0, -1.4])  # final LayerNorm gain

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
