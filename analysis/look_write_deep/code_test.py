"""Per-token code test (analysis only; logs/decisions.md item 8): is a local head's vote table the best low-rank
compression of a corpus statistic? Formula fixed in advance (logs/prereg_deep_look_write.md, section K).

Corpus statistics (whole TRAIN stream, pair counts analysis/out/candidates/qk/N_d.npy):
    n(k, v) = sum_{d=1..7} N_{d+1}(k, v)                       v sits d + 1 after k (what follows a key read d back)
    e(k, v) = sum_{d=1..7} sum_q N_d(k, q) P1(v | q)            the same, predicted by the bigram from the token at d
    C_res(k, v) = log[(n + 20 e_hat) / ((n(k) + 20) e_hat)]     e_hat = e / sum_v e   (bigram-residual; primary)
    C_pmi(k, v) = log[(n + 20 P(v)) / ((n(k) + 20) P(v))]       (secondary)
Weighting: rows by sqrt(P(k)), columns by sqrt(P(v)), after a weighted double centring. M_X = W X W'.
For a head h with vote table OV_h (direct path, token level, logit units): M_h = W OV_h W'.
    E32(h) = ||U32' M_h V32||^2 / ||M_h||^2   (U32, V32: the corpus statistic's top-32 singular subspaces)
Baselines: N1 random 32-dim subspaces inside the corpus top-128 (500 draws, 95th percentile); N2 the top-32
subspaces of the weighted embedding (input LN1(E + pbar) of the head's layer-0 input, output W_E); N3 the same
model's RECALL heads. Also R2_diag(R) = sum_{r <= R} <M_h, u_r v_r'>^2 / ||M_h||^2 for R = 32, 128.

    OMP_NUM_THREADS=2 .venv/bin/python -m analysis.look_write_deep.code_test attn_s1 attn_s2 ... [--dev attn_s0]
Writes analysis/out/look_write_deep/code_test.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.look_write import common as C1
from analysis.look_write_deep import common_deep as C

QK = os.path.join(C.REPO, "analysis", "out", "candidates", "qk")
PRIOR = 20.0


def corpus_stats():
    P1 = C.p1_table()
    N = {d: np.load(os.path.join(QK, f"N_{d}.npy")).astype(np.float64) for d in range(1, 9)}
    n = sum(N[d + 1] for d in range(1, 8))
    e = sum(N[d] @ P1 for d in range(1, 8))
    e_hat = e / np.maximum(e.sum(1, keepdims=True), 1e-300)
    uni = N[1].sum(1) + 1.0
    pu = uni / uni.sum()
    nk = n.sum(1, keepdims=True)
    e_hat = np.maximum(e_hat, 1e-12)
    c_res = np.log((n + PRIOR * e_hat) / ((nk + PRIOR) * e_hat))
    c_pmi = np.log((n + PRIOR * pu[None, :]) / ((nk + PRIOR) * pu[None, :]))
    return {"C_res": c_res, "C_pmi": c_pmi}, pu


def weighted(X, pu):
    """Weighted double centring, then W X W' with W = diag(sqrt(pu))."""
    w = pu / pu.sum()
    X = X - (X @ w)[:, None]
    X = X - (w @ X)[None, :]
    s = np.sqrt(w)
    return s[:, None] * X * s[None, :]


def head_tables(model_id):
    """(layer, head) -> token-level direct-path vote table (V, V) for every head of layer 0, plus classes."""
    m = C.load(model_id)
    L, H, DH, D = C.dims(m)
    blk = m.blocks[0]
    wte = m.wte.weight.detach().double().numpy()
    pbar = m.wpe.weight.detach().double().numpy()[1:C.T].mean(0)
    x = C1.layernorm(wte + pbar[None], blk.ln1.weight.detach().double().numpy())
    Wv = blk.attn.qkv.weight.detach().double().numpy()[2 * D:]
    Wo = blk.attn.out.weight.detach().double().numpy()
    gf = m.ln_f.weight.detach().double().numpy()
    scale = C.final_ln_scale(m, C.windows(C.FIT_SEED, C.N_FIT)[:, :C.T])
    out = {}
    for h in range(H):
        o = x @ Wv[h * DH:(h + 1) * DH].T @ Wo[:, h * DH:(h + 1) * DH].T
        o = o - o.mean(1, keepdims=True)
        out[h] = (o * gf) @ wte.T / scale
    classes = json.load(open(os.path.join(C.model_dir(model_id), "classes.json")))["heads"]
    cls = {c["head"]: c for c in classes if c["layer"] == 0}
    emb_in = x
    return out, cls, emb_in, wte


def energy(M, A, B):
    return float(np.linalg.norm(A.T @ M @ B) ** 2 / np.linalg.norm(M) ** 2)


def run(models, n_draws=500, seed=0):
    stats, pu = corpus_stats()
    sv = {}
    for name, X in stats.items():
        U, S, Vt = np.linalg.svd(weighted(X, pu))
        sv[name] = (U[:, :128], S[:128], Vt[:128].T)
    rng = np.random.default_rng(seed)
    res = {"models": {}}
    s = np.sqrt(pu / pu.sum())
    for mid in models:
        tabs, cls, emb_in, wte = head_tables(mid)
        A2 = np.linalg.svd(s[:, None] * (emb_in - (pu / pu.sum()) @ emb_in), full_matrices=False)[0][:, :32]
        B2 = np.linalg.svd(s[:, None] * (wte - (pu / pu.sum()) @ wte), full_matrices=False)[0][:, :32]
        rows = {}
        for h, OV in tabs.items():
            M = weighted(OV, pu)
            row = {"class": "LOCAL" if cls[h]["confirmatory"] else "RECALL"}
            for name, (U, S, Vv) in sv.items():
                e32 = energy(M, U[:, :32], Vv[:, :32])
                comps = np.einsum("kr,kv,vr->r", U, M, Vv) ** 2 / np.linalg.norm(M) ** 2
                rand = []
                for _ in range(n_draws):
                    Qa = np.linalg.qr(rng.standard_normal((128, 32)))[0]
                    Qb = np.linalg.qr(rng.standard_normal((128, 32)))[0]
                    rand.append(energy(M, U @ Qa, Vv @ Qb))
                row[name] = {"E32": e32, "N1_p95": float(np.percentile(rand, 95)), "N1_mean": float(np.mean(rand)),
                             "R2diag_32": float(comps[:32].sum()), "R2diag_128": float(comps.sum())}
            row["N2_embedding_E32"] = energy(M, A2, B2)
            rows[h] = row
            print(f"{mid} h{h} {row['class']}: E32 C_res {row['C_res']['E32']:.3f} (N1 p95 {row['C_res']['N1_p95']:.3f}) "
                  f"C_pmi {row['C_pmi']['E32']:.3f} | embedding {row['N2_embedding_E32']:.3f} | "
                  f"R2diag 32/128 {row['C_res']['R2diag_32']:.3f}/{row['C_res']['R2diag_128']:.3f}", flush=True)
        res["models"][mid] = rows
    with open(os.path.join(C.OUT, "code_test.json" if "attn_s0" not in models else "code_test_dev.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    run(a.models)
