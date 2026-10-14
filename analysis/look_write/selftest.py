"""Checks before any result: (1) the substitution forward pass reproduces the model's logits when given the model's
own pattern, for every head and every model; (2) the numpy score formula used for the pure-pair scores matches the
model's attention scores; (3) N_1 recount on the first 25M TRAIN tokens against analysis/out/candidates/qk/N_1.npy.

Run: OMP_NUM_THREADS=2 .venv/bin/python -m analysis.look_write.selftest
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from analysis.look_write import common as C


def check_substitution(model_id, wins):
    import torch

    m = C.load(model_id)
    x = torch.as_tensor(wins[:4].astype(np.int64))
    with torch.no_grad():
        ref = m(x).double()
        exp_ = C.forward_with_pattern(m, x).double()
        pats = C.attention_inputs(m, x)[4].softmax(-1)
        out = {"explicit_vs_model": float((ref - exp_).abs().max())}
        for h in range(C.H):
            sub = C.forward_with_pattern(m, x, h, pats[:, h].numpy()).double()
            out[f"own_pattern_head{h}"] = float((ref - sub).abs().max())
        # a different pattern must change the output (the substitution is live)
        uni = torch.tril(torch.ones(C.T, C.T)) / torch.arange(1, C.T + 1)[:, None]
        sub = C.forward_with_pattern(m, x, 0, uni.expand(4, C.T, C.T).numpy()).double()
        out["uniform_pattern_head0_changes_logits_by"] = float((ref - sub).abs().max())
        # numpy score formula vs model scores at (i, i-1) and (i, i)
        w = C.weights(m)
        sc = C.attention_inputs(m, x)[4].double().numpy()
        errs = []
        for b in range(4):
            for i in C.POS_SET:
                q_t, k_t = int(wins[b, i]), int(wins[b, i - 1])
                xq = C.layernorm(w["wte"][q_t] + w["wpe"][i], w["g1"])
                xk = C.layernorm(w["wte"][k_t] + w["wpe"][i - 1], w["g1"])
                for h in range(C.H):
                    s1 = (xq @ w["Wq"][h]) @ (xk @ w["Wk"][h]) / math.sqrt(C.DH)
                    s0 = (xq @ w["Wq"][h]) @ (xq @ w["Wk"][h]) / math.sqrt(C.DH)
                    errs += [abs(s1 - sc[b, h, i, i - 1]), abs(s0 - sc[b, h, i, i])]
        out["numpy_score_formula_max_err"] = float(max(errs))
    return out


def check_n1(n_tokens=25_000_000):
    from harness.common import load_corpus

    c = load_corpus()
    x = np.asarray(c[:n_tokens]).astype(np.int64)
    n1 = np.bincount(x[:-1] * C.V + x[1:], minlength=C.V * C.V).reshape(C.V, C.V).astype(np.float64)
    ref = np.load(os.path.join(C.REPO, "analysis", "out", "candidates", "qk", "N_1.npy")).astype(np.float64)
    frac = (n_tokens - 1) / len(c)
    mask = ref >= 200
    rel = n1[mask] / (ref[mask] * frac)
    return {
        "recount_total": float(n1.sum()), "file_total": float(ref.sum()), "corpus_len": int(len(c)),
        "cells_where_recount_exceeds_file": int((n1 > ref + 0.5).sum()),
        "cells_with_file_zero_but_recount_positive": int(((ref == 0) & (n1 > 0)).sum()),
        "corr_recount_vs_file_scaled": float(np.corrcoef(n1.ravel(), ref.ravel())[0, 1]),
        "median_ratio_recount_over_expected_cells_file_ge_200": float(np.median(rel)),
        "iqr_ratio": [float(np.percentile(rel, 25)), float(np.percentile(rel, 75))],
    }


def main():
    C.set_threads()
    wins = C.train_windows(C.EVAL_SEED)
    res = {mid: check_substitution(mid, wins) for mid in [f"attn_s{s}" for s in range(5)] + ["mlp_s0"]}
    res["N1_recount_first_25M"] = check_n1()
    print(json.dumps(res, indent=1))
    with open(os.path.join(C.OUT, "selftest.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
