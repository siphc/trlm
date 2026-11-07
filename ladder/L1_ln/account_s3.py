"""Account of L1 (ln arm), seed 3 (v2). L1 is a one-block, one-head, attention-only character model with
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
    [1.4, -0.2, -1.6, -0.4],  # EOS
    [-0.3, -6.8, 1.0, -3.0],  # UNK
    [0.7, 4.3, -2.7, -0.8],  # \n
    [-0.4, 6.5, -0.9, 0.7],  # ␣
    [0.1, -2.7, 0.6, 0.4],  # !
    [0.4, 4.6, -2.1, -0.4],  # "
    [-0.4, 5.1, -0.3, 0.1],  # '
    [0.0, -3.0, 0.5, 0.6],  # ,
    [0.5, 0.3, -0.6, -0.4],  # -
    [0.2, -2.8, 0.2, 0.7],  # .
    [-0.5, -7.1, 1.1, -3.3],  # 0
    [-0.6, -6.0, 1.0, -3.2],  # 1
    [-0.6, -6.1, 1.0, -3.3],  # 2
    [0.6, -8.2, 0.6, -1.2],  # 3
    [-0.7, -6.2, 1.1, -3.4],  # 4
    [-0.6, -6.5, 1.1, -3.3],  # 5
    [0.3, -4.5, 0.9, -0.1],  # :
    [0.5, -7.5, 1.1, -1.1],  # ;
    [0.0, -2.6, 0.8, 0.2],  # ?
    [-0.8, 1.5, -2.4, -2.6],  # A
    [-0.7, -0.2, -2.5, -2.6],  # B
    [0.0, 2.6, -1.7, -1.5],  # C
    [-0.1, 2.5, -1.8, -1.6],  # D
    [-0.6, -0.4, -2.0, -2.7],  # E
    [-0.3, 1.5, -1.9, -1.9],  # F
    [0.3, 0.2, -1.5, -1.4],  # G
    [0.3, -10.9, -1.2, -1.5],  # H
    [-0.7, 2.1, -2.2, -1.9],  # I
    [-0.5, 1.9, -2.2, -2.1],  # J
    [0.8, 0.3, -0.8, -0.7],  # K
    [-0.8, -0.1, -2.7, -2.8],  # L
    [-0.6, 1.9, -2.3, -2.1],  # M
    [0.2, 2.5, -1.8, -1.3],  # N
    [2.0, -40.3, 0.1, -2.6],  # O
    [0.2, 0.5, -1.7, -1.7],  # P
    [0.0, -0.2, -0.4, -2.1],  # Q
    [0.4, -0.2, -1.6, -1.5],  # R
    [-0.9, 2.1, -2.6, -2.2],  # S
    [-1.0, 1.8, -2.8, -2.4],  # T
    [0.5, -10.7, 0.8, -1.7],  # U
    [1.4, -10.6, -0.1, -0.9],  # V
    [-0.2, 2.5, -2.2, -1.6],  # W
    [-0.6, -6.8, 1.1, -3.4],  # X
    [-0.2, 0.8, -2.3, -2.2],  # Y
    [0.3, -0.3, -0.8, -1.5],  # Z
    [-0.3, 3.1, -0.5, 0.6],  # a
    [-0.2, -8.4, 1.3, 0.4],  # b
    [-0.8, -1.6, 1.1, 0.4],  # c
    [-0.5, -2.9, 1.6, 0.7],  # d
    [0.3, 2.4, -0.4, 0.7],  # e
    [-0.6, -5.6, 1.4, 0.4],  # f
    [-0.6, -1.8, 1.3, 0.5],  # g
    [0.9, -11.7, -0.4, 0.5],  # h
    [-0.2, 3.5, -0.4, 0.6],  # i
    [0.3, 0.2, -0.8, -0.3],  # j
    [0.0, -7.6, 1.3, 0.6],  # k
    [0.1, -9.4, 1.7, 0.8],  # l
    [-0.4, -6.4, 1.4, 0.6],  # m
    [-0.7, -4.8, 1.8, 0.6],  # n
    [-0.1, 4.0, -0.7, 0.6],  # o
    [0.0, -9.7, 1.7, 0.7],  # p
    [0.3, 0.4, -0.7, -0.4],  # q
    [-0.1, -7.9, 1.5, 0.8],  # r
    [-0.9, -1.7, 1.0, 0.4],  # s
    [-0.9, -1.3, 0.7, 0.5],  # t
    [-0.4, 0.8, 0.3, 0.5],  # u
    [-0.2, -5.4, 1.3, 0.5],  # v
    [-0.9, -2.0, 0.9, 0.3],  # w
    [-0.6, -2.1, 1.5, 0.2],  # x
    [-0.4, -2.1, 1.5, 0.6],  # y
    [0.2, -16.4, 3.5, 0.4],  # z
    [-0.7, -5.4, 1.1, -3.4],  # ¦
    [-0.2, 11.6, -2.0, -0.7],  # â
    [1.2, -4.0, 0.7, -1.3],  # œ
    [-0.4, -7.6, 1.0, -3.2],  # ˜
    [0.6, -11.9, 1.4, -2.1],  # “
    [-0.2, -10.0, 1.3, -3.1],  # ”
    [-1.0, -12.7, -1.2, -5.1],  # €
    [1.2, -14.7, 3.1, -1.2],  # ™
])

POS = np.array([-0.24, -0.16, -0.36, 0.83])  # the mean position vector, added everywhere

# The head's maps (residual -> head space and back). The ln1 gain is folded into WQ, WK and WV, and 1/sqrt(4) into WQ.
WQ = np.array([
    [-1.02, 0.41, -0.34, 0.50],
    [-0.43, 1.10, -0.14, -0.30],
    [-0.07, 1.02, -0.27, -0.63],
    [0.47, 0.65, -0.36, -0.88],
])
WK = np.array([
    [-2.07, 1.23, -0.36, 0.78],
    [-0.60, 2.07, -1.88, -0.51],
    [-0.06, 1.81, -1.43, -1.15],
    [0.51, 1.31, -0.78, -1.91],
])
WV = np.array([
    [-0.63, 0.73, -0.27, -1.90],
    [-0.20, 0.59, -0.95, -0.27],
    [0.34, -0.61, 0.83, 0.47],
    [0.29, -0.61, 0.88, 0.38],
])
WO = np.array([
    [0.05, -0.09, 0.07, 0.08],
    [-0.42, -0.26, 0.28, 0.27],
    [0.08, 0.21, -0.18, -0.20],
    [0.13, 0.02, -0.04, -0.03],
])
OUT_GAIN = np.array([2.98, -0.25, 3.12, 7.01])  # final LayerNorm gain

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
