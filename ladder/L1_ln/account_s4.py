"""Account of L1 (ln arm), seed 4 (v2). L1 is a one-block, one-head, attention-only character model with
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
    [1.9, -1.8, -2.0, 2.5],  # EOS
    [-3.0, 0.2, -2.9, 3.3],  # UNK
    [-0.1, -1.8, -2.1, -0.3],  # \n
    [4.8, -1.0, 0.9, -6.5],  # ␣
    [1.4, -0.2, 1.1, 3.0],  # !
    [0.6, -1.5, -1.4, -0.7],  # "
    [2.9, -0.2, 0.4, -3.0],  # '
    [1.4, -0.3, 1.4, 2.5],  # ,
    [1.4, 0.0, 0.0, 0.0],  # -
    [1.2, -0.2, 1.7, 3.6],  # .
    [-3.7, 0.2, -3.0, 4.4],  # 0
    [-3.2, 0.0, -3.3, 3.2],  # 1
    [-3.2, 0.0, -3.2, 3.5],  # 2
    [-0.7, 0.2, -1.3, 1.5],  # 3
    [-2.9, -0.1, -3.3, 3.0],  # 4
    [-3.1, 0.0, -3.2, 3.4],  # 5
    [0.6, 0.2, 0.2, 2.4],  # :
    [-0.8, 0.4, -1.3, 1.7],  # ;
    [1.7, -0.2, 0.8, 2.5],  # ?
    [-1.4, -1.3, -2.6, -1.0],  # A
    [-4.3, -1.0, -3.3, 0.2],  # B
    [-2.5, -1.0, -3.0, -0.2],  # C
    [-2.6, -1.0, -2.9, -0.1],  # D
    [-4.5, -0.7, -3.6, 0.7],  # E
    [-2.0, -1.1, -2.9, -0.3],  # F
    [-2.1, -0.9, -2.8, 0.0],  # G
    [-4.0, -0.9, -2.7, 0.3],  # H
    [-1.3, -1.2, -2.1, -1.5],  # I
    [-2.9, -1.1, -3.0, -0.3],  # J
    [-3.0, -0.5, -2.8, 1.0],  # K
    [-4.3, -1.0, -3.3, 0.1],  # L
    [-2.9, -1.1, -2.9, -0.5],  # M
    [-2.6, -1.0, -3.1, -0.1],  # N
    [-2.0, -1.5, -3.2, -0.6],  # O
    [-2.3, -0.8, -2.8, 0.1],  # P
    [-3.0, 0.6, -2.3, 3.0],  # Q
    [-2.4, -0.7, -2.9, 0.4],  # R
    [-3.3, -1.0, -2.8, -1.0],  # S
    [-4.4, -1.1, -3.2, -0.8],  # T
    [-3.5, 0.5, -2.4, 3.2],  # U
    [-1.5, 1.0, -0.5, 3.6],  # V
    [-3.3, -1.1, -3.2, -0.3],  # W
    [-3.0, 0.0, -3.2, 3.1],  # X
    [-3.0, -1.1, -3.0, -0.2],  # Y
    [-2.0, 0.6, -1.8, 1.9],  # Z
    [2.5, -0.1, 1.2, -2.9],  # a
    [-1.5, 0.7, 0.4, -0.5],  # b
    [-0.7, 0.5, 0.5, -1.5],  # c
    [0.6, -0.3, 1.0, -0.4],  # d
    [3.1, -0.2, 1.5, -1.1],  # e
    [-0.5, 0.3, 0.4, -1.2],  # f
    [-0.1, 0.3, 0.6, -1.3],  # g
    [-3.6, 2.1, 1.2, 2.1],  # h
    [3.1, -0.1, 1.1, -3.3],  # i
    [-1.5, 0.9, -0.2, 0.5],  # j
    [-0.2, 0.6, 1.0, 0.2],  # k
    [0.3, 0.3, 0.9, -1.6],  # l
    [-0.3, 0.4, 0.7, -1.0],  # m
    [1.3, -0.4, 0.8, -2.4],  # n
    [2.1, 0.1, 1.2, -2.4],  # o
    [0.1, 0.4, 0.7, -1.5],  # p
    [-0.5, 0.9, 0.0, 0.4],  # q
    [0.7, -0.1, 0.9, -1.5],  # r
    [-0.1, 0.1, 0.7, -2.0],  # s
    [-0.8, 0.6, 0.7, -2.1],  # t
    [2.5, 0.0, 0.9, -2.9],  # u
    [-2.3, 1.1, 0.7, 2.0],  # v
    [-1.6, 0.6, 0.4, -0.9],  # w
    [1.9, -0.5, 0.0, -1.6],  # x
    [1.3, -0.1, 1.2, -0.1],  # y
    [-0.5, 0.9, 0.2, 0.4],  # z
    [-3.3, 0.0, -3.2, 3.9],  # ¦
    [-2.6, -1.2, -3.4, -0.3],  # â
    [2.1, -0.2, -0.6, 0.9],  # œ
    [-2.3, -0.3, -3.2, 2.5],  # ˜
    [-3.5, 1.0, -1.9, 4.7],  # “
    [-3.6, 0.4, -2.9, 3.8],  # ”
    [-9.2, 3.0, -2.3, 11.9],  # €
    [0.3, 0.6, -1.0, 1.5],  # ™
])

POS = np.array([0.14, -0.50, 1.39, -1.49])  # the mean position vector, added everywhere

# The head's maps (residual -> head space and back). The ln1 gain is folded into WQ, WK and WV, and 1/sqrt(4) into WQ.
WQ = np.array([
    [-0.2, 0.4, 0.5, -1.1],
    [0.2, 0.7, -0.4, -0.1],
    [0.3, 0.4, -0.3, -0.2],
    [0.3, 0.2, -0.2, -0.2],
])
WK = np.array([
    [0.4, -0.5, -1.5, 0.3],
    [-0.6, 0.1, -1.1, 2.4],
    [-0.3, -0.9, -0.6, 0.8],
    [-0.1, -1.1, -0.1, 0.5],
])
WV = np.array([
    [-1.2, -0.4, 0.2, 0.6],
    [-0.7, -0.9, 0.6, 0.2],
    [-1.2, -0.6, 0.4, 0.5],
    [-0.2, 0.5, -0.8, 0.4],
])
WO = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 1, 0, 0],
])
OUT_GAIN = np.array([-1.2, 8.0, 4.3, 0.5])  # final LayerNorm gain

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
