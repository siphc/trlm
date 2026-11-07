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

LAYERNORM = True

# EMBED[s]: symbol s's 4 numbers (the model's embedding to 1 decimal), in symbol-id order.
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

POS = np.array([1.02, 0.20, -1.34, 0.36])  # the mean position vector, added everywhere

# The head's maps (residual -> head space and back). The ln1 gain is folded into WQ, WK and WV, and 1/sqrt(4) into WQ.
WQ = np.array([
    [0.46, 0.58, 0.16, -1.52],
    [0.82, -0.57, -0.07, -0.22],
    [0.82, -0.51, -0.07, -0.25],
    [0.07, -1.84, 0.82, -1.02],
])
WK = np.array([
    [0.15, 1.69, 2.31, -2.03],
    [2.82, -3.13, 0.93, -1.40],
    [2.88, -2.82, 0.94, -1.58],
    [-0.39, -1.09, 2.60, -0.11],
])
WV = np.array([
    [2.21, -0.39, -1.19, 1.59],
    [-4.30, 1.92, 0.96, -0.42],
    [-1.11, 0.23, -0.32, 1.34],
    [-0.95, 0.97, -0.40, 0.88],
])
WO = np.array([
    [0.26, -0.26, 0.08, 0.07],
    [-0.03, -0.09, -0.22, -0.32],
    [-0.21, 0.26, -0.10, 0.13],
    [-0.12, 0.27, 0.54, 0.29],
])
OUT_GAIN = np.array([0.31, 9.40, 3.34, -1.52])  # final LayerNorm gain

BOUNDARY = [0, 2, 3, 4, 5, 6, 7, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]  # EOS \n ␣ ! " ' , . ? A B C D E F G H I J K L M N O P Q R S T U V W X Y Z


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
    {"name": "direct path", "site": "direct", "key": lambda X: X, "tests": [
        {"statement": "removing the direct path costs at least 1 nat",
         "measure": lambda p: p.cost_resample("direct"), "op": ">=", "threshold": 1.0},
        {"statement": "position costs at most 5% of the direct path",
         "measure": lambda p: ratio(p.cost_positions(0 * p.positions + p.positions.mean(0)), p.cost_resample("direct")),
         "op": "<=", "threshold": 0.05}]},
    {"name": "head", "site": "head0", "value": head, "tests": [
        {"statement": "attention is set by content (at least 95% of the variance)",
         "measure": content_share, "op": ">=", "threshold": 0.95},
        {"statement": "the head favours boundary symbols (at least 1.2 times their share)",
         "measure": boundary_ratio, "op": ">=", "threshold": 1.2},
        {"statement": "removing the head costs at least 0.05 nats",
         "measure": lambda p: p.cost_resample("head0"), "op": ">=", "threshold": 0.05},
        {"statement": "context is at most a quarter of the head's effect",
         "measure": lambda p: ratio(p.cost_replaced(z=mean_for_current_symbol(p.X, p.z)), p.cost_resample("head0")),
         "op": "<=", "threshold": 0.25}]},
]
