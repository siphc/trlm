"""Fit a program head's attention-rule scalars to a model head's own attention pattern.

Objective (as in the circuit cards): pattern cross-entropy, the mean over query positions i >= 1 of
-sum_j A_model(i, j) log A_rule(i, j), on TRAIN windows. The rules are the program's own NumPy functions
(program/head*.py, attention(tokens, stats)); their named constants are set temporarily, and the gradient is taken
by central finite differences inside torch's L-BFGS (8-13 scalars per rule, so this is cheap and needs no torch copy
of each rule).

  from analysis.attention_fit import model_attention, fit_rule, pattern_ce
  A = model_attention("attn_s1", windows)          # (N, H, T, T)
  fit_rule(program.head0, A[:, 2], windows, stats)  # fit LOCAL-0's rule to head 2 of attn_s1
"""
from __future__ import annotations

import math

import numpy as np
import torch

# Named constants of each rule: (attribute, index or key or None, transform). "log" fits log(value).
RULE_PARAMS = {
    "head0": [("SLOPE", k, None) for k in range(4)] + [("SELF_BONUS", k, None) for k in range(4)]
             + [("USE_NEAR", None, None), ("USE_FAR", None, None)],
    "head2": [(a, None, None) for a in ("SELF_BASE", "SELF_INFO", "SELF_CONT", "SELF_PIECE", "DECAY", "DECAY_FAR",
                                         "PREVIOUS", "KEY_INFO", "KEY_CONT", "KEY_PIECE", "KEY_SPACE",
                                         "KEY_SENTENCE_END", "USE_NEAR", "USE_FAR")],
    "head1": [("MEMORY", None, "log")]
             + [("SALIENCE", g, None) for g in ("NAMES", "SENTENCE_ENDS", "WORD_ENDS_AND_LINKERS", "FUNCTION_WORDS",
                                                "CONTENT_WORDS")]
             + [("SELF_BONUS", None, None), ("SUCCESSOR", None, None)],
}


def model_attention(model_id, windows, bs=8):
    """Block-0 attention patterns of a model, (N, H, T, T) float32 (mirrors harness.common._attention_parts)."""
    from harness.common import load_model

    model = load_model(model_id) if isinstance(model_id, str) else model_id
    blk = model.blocks[0]
    H, Dh, D = model.cfg.n_heads, model.cfg.d_head, model.cfg.d_model
    out = []
    with torch.no_grad():
        for c in range(0, len(windows), bs):
            x = torch.as_tensor(np.asarray(windows[c : c + bs])[:, :256].astype(np.int64))
            B, T = x.shape
            x0 = model.wte(x) + model.wpe(torch.arange(T))[None]
            q, k, _ = blk.attn.qkv(blk.ln1(x0)).split(D, dim=-1)
            q = q.view(B, T, H, Dh).transpose(1, 2)
            k = k.view(B, T, H, Dh).transpose(1, 2)
            sc = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
            sc = sc.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
            out.append(sc.softmax(-1).float().numpy())
    return np.concatenate(out)


def _get(mod, spec):
    vals = []
    for attr, key, tr in spec:
        v = getattr(mod, attr) if key is None else getattr(mod, attr)[key]
        vals.append(math.log(v) if tr == "log" else float(v))
    return np.array(vals)


def _set(mod, spec, x):
    for (attr, key, tr), v in zip(spec, x):
        v = math.exp(v) if tr == "log" else float(v)
        if key is None:
            setattr(mod, attr, v)
        else:
            container = getattr(mod, attr)
            container = list(container) if isinstance(container, (list, tuple)) else dict(container)
            container[key] = v
            setattr(mod, attr, container)


def pattern_ce(mod, A_head, windows, stats):
    """Mean pattern cross-entropy and KL of the rule against one model head's patterns (N, T, T)."""
    ce, ent, n = 0.0, 0.0, 0
    for w, A in zip(windows, A_head):
        t = np.asarray(w[:256], dtype=np.int64)
        R = mod.attention(t, stats)
        m = np.tril(np.ones_like(A, dtype=bool))[1:]
        a, r = A[1:], R[1:]
        ce -= np.where(m, a * np.log(np.maximum(r, 1e-30)), 0.0).sum()
        ent -= np.where(m & (a > 0), a * np.log(np.maximum(a, 1e-30)), 0.0).sum()
        n += a.shape[0]
    return ce / n, (ce - ent) / n


def fit_rule(mod, A_head, windows, stats, iters=60, eps=1e-4, keep=False, spec=None):
    """Fit the rule's named constants (RULE_PARAMS, or `spec`: [(attribute, index or key or None, transform)]) to one
    head's patterns; returns {name: fitted value}. The module's constants are restored afterwards unless keep=True
    (use module_copy for a private instance)."""
    if spec is None:
        spec = RULE_PARAMS[mod.__name__.split(".")[-1].split("_")[0]]
    original = _get(mod, spec)
    x = torch.tensor(_get(mod, spec), dtype=torch.float64, requires_grad=True)

    def f(v):
        _set(mod, spec, v)
        return pattern_ce(mod, A_head, windows, stats)[0]

    opt = torch.optim.LBFGS([x], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-6,
                            tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        v = x.detach().numpy().copy()
        loss = f(v)
        g = np.zeros_like(v)
        for k in range(len(v)):
            up, dn = v.copy(), v.copy()
            up[k] += eps
            dn[k] -= eps
            g[k] = (f(up) - f(dn)) / (2 * eps)
        _set(mod, spec, v)
        x.grad = torch.tensor(g)
        return torch.tensor(loss)

    opt.step(closure)
    _set(mod, spec, x.detach().numpy() if keep else original)
    return {f"{a}" + ("" if k is None else f"[{k!r}]"): (math.exp(v) if tr == "log" else float(v))
            for (a, k, tr), v in zip(spec, x.detach().numpy())}


def module_copy(mod, tag="copy"):
    """A fresh instance of a program head module (its source re-executed under a new name), whose constants can be
    changed without touching the original (functions read their own module's globals)."""
    import importlib.util

    name = f"{mod.__name__}_{tag}"
    spec = importlib.util.spec_from_file_location(name, mod.__file__)
    new = importlib.util.module_from_spec(spec)
    new.__package__ = mod.__package__
    spec.loader.exec_module(new)
    return new
