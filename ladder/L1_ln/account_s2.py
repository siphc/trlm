"""Account of L1 (ln arm), seed 2 (v2). L1 is a one-block, one-head, attention-only character model with
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
    [-1.8, 0.8, 0.3, -1.8],  # EOS
    [-7.2, 0.1, 1.6, 1.1],  # UNK
    [-0.4, 1.1, 1.0, -0.6],  # \n
    [14.3, 1.2, 0.1, -5.7],  # ␣
    [-2.0, 0.3, -0.6, -1.9],  # !
    [-0.4, 1.0, 0.6, -0.9],  # "
    [7.8, 0.5, 0.2, -3.0],  # '
    [-1.9, 0.4, -0.9, -2.0],  # ,
    [-1.2, 0.0, 0.0, -0.8],  # -
    [-2.3, 0.4, -1.1, -2.2],  # .
    [-8.0, 0.1, 1.5, 1.1],  # 0
    [-6.6, 0.1, 1.9, 1.2],  # 1
    [-6.5, 0.1, 1.8, 1.0],  # 2
    [-6.4, 0.0, 1.0, 1.1],  # 3
    [-5.1, 0.2, 1.9, 0.7],  # 4
    [-6.6, 0.2, 1.8, 1.0],  # 5
    [-1.4, 0.1, 0.0, -1.3],  # :
    [-7.5, 0.1, 0.4, 0.1],  # ;
    [-2.0, 0.3, -0.4, -1.8],  # ?
    [1.1, 1.0, 2.5, 1.5],  # A
    [-1.7, 0.9, 2.4, 2.7],  # B
    [0.0, 0.8, 1.9, 0.7],  # C
    [0.0, 0.8, 1.8, 0.8],  # D
    [0.9, 0.8, 2.4, 1.3],  # E
    [0.2, 0.9, 2.0, 0.8],  # F
    [-0.2, 0.8, 1.6, 0.2],  # G
    [-1.9, 0.8, 1.9, 2.8],  # H
    [0.8, 0.8, 1.4, 1.3],  # I
    [0.4, 0.9, 2.2, 1.4],  # J
    [-2.3, 0.6, 1.2, 0.6],  # K
    [0.2, 0.9, 1.8, 1.7],  # L
    [0.4, 0.9, 2.1, 1.6],  # M
    [-1.0, 0.8, 2.0, 1.1],  # N
    [1.1, 1.1, 3.3, 1.7],  # O
    [-1.6, 0.7, 1.8, 1.0],  # P
    [-3.1, 0.2, 2.3, 0.8],  # Q
    [-1.5, 0.7, 1.9, 1.0],  # R
    [1.7, 1.0, 3.4, 3.3],  # S
    [1.0, 1.0, 3.4, 3.7],  # T
    [-8.3, 0.3, 1.2, 1.1],  # U
    [-5.2, -0.2, 0.4, 0.2],  # V
    [-0.6, 0.9, 1.6, 1.0],  # W
    [-5.5, 0.2, 1.9, 0.9],  # X
    [-1.1, 0.8, 1.8, 1.4],  # Y
    [-2.2, 0.3, 1.9, 0.5],  # Z
    [2.1, -0.8, -1.4, 0.7],  # a
    [-0.3, -0.5, -0.8, 1.3],  # b
    [-0.4, -0.1, -0.8, 0.9],  # c
    [-1.0, 0.4, -1.0, -0.1],  # d
    [-1.2, -0.4, -1.5, -1.3],  # e
    [-0.2, -0.2, -0.7, 0.9],  # f
    [-0.6, 0.2, -0.7, 0.4],  # g
    [-2.1, -1.2, -1.7, 1.7],  # h
    [2.5, -0.7, -1.2, 0.1],  # i
    [-0.8, -0.3, -0.1, 0.6],  # j
    [-1.3, -0.1, -1.0, 0.0],  # k
    [-0.3, -0.2, -1.2, 0.6],  # l
    [-0.7, 0.0, -1.0, 0.6],  # m
    [0.2, 0.5, -0.7, 0.3],  # n
    [0.4, -0.7, -1.5, 0.4],  # o
    [-0.3, -0.2, -1.0, 0.6],  # p
    [-1.0, -0.2, -0.1, -0.2],  # q
    [-0.7, 0.0, -1.1, 0.3],  # r
    [0.0, 0.0, -1.0, 0.9],  # s
    [-0.1, -0.3, -1.2, 1.3],  # t
    [2.8, -0.5, -1.0, -0.2],  # u
    [-2.0, 0.3, -0.6, 0.3],  # v
    [-0.2, -0.4, -0.9, 1.3],  # w
    [-0.7, 0.3, -0.3, -0.7],  # x
    [-1.3, 0.1, -1.1, -0.5],  # y
    [-1.3, 0.1, -0.2, -0.5],  # z
    [-6.8, 0.1, 1.8, 1.1],  # ¦
    [-2.6, 0.6, 0.4, -1.3],  # â
    [3.7, -0.1, 4.3, 2.0],  # œ
    [-8.3, 0.1, 1.7, 1.5],  # ˜
    [-8.8, -0.1, 2.2, 2.3],  # “
    [-6.7, 0.1, 2.0, 1.4],  # ”
    [7.5, -0.1, 6.7, 1.3],  # €
    [2.8, -0.5, 3.9, 2.1],  # ™
])

POS = np.array([1.0, 0.2, -1.3, 0.4])  # the mean position vector, added everywhere

# The head's maps (residual -> head space and back). The ln1 gain is folded into WQ, WK and WV, and 1/sqrt(4) into WQ.
WQ = np.array([
    [0.5, 0.6, 0.2, -1.5],
    [0.8, -0.6, -0.1, -0.2],
    [0.8, -0.5, -0.1, -0.3],
    [0.1, -1.8, 0.8, -1.0],
])
WK = np.array([
    [0, 2, 2, -2],
    [3, -3, 1, -1],
    [3, -3, 1, -2],
    [0, -1, 3, 0],
])
WV = np.array([
    [2.2, -0.4, -1.2, 1.6],
    [-4.3, 1.9, 1.0, -0.4],
    [-1.1, 0.2, -0.3, 1.3],
    [-1.0, 1.0, -0.4, 0.9],
])
WO = np.array([
    [0.3, -0.3, 0.1, 0.1],
    [0.0, -0.1, -0.2, -0.3],
    [-0.2, 0.3, -0.1, 0.1],
    [-0.1, 0.3, 0.5, 0.3],
])
OUT_GAIN = np.array([0.3, 9.4, 3.3, -1.5])  # final LayerNorm gain

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
