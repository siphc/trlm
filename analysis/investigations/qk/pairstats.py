"""Pair statistics for pair-specific attention (prefix qk). Streams the TRAIN corpus in chunks.

For each distance d = 1..8 and each ordered pair (k, q) with k sitting d tokens before q (whole stream):
  N_d[k, q]  = number of occurrences
  S_d[k, q]  = sum over those occurrences of PMI_{min(d+1, 8)}(k, next token after q)
               ("how well k's skip-gram vote predicted the actual next token in that context")
For insight only (weights-derived, never used in a candidate): the same sums with the OV vote of heads 0 and 2
(OV_h[k, v]) instead of the corpus PMI, for d = 1..3.

Outputs float32 (V, V) tables under analysis/out/candidates/qk/.
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

REPO = "/Users/svi/Desktop/trlm"
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "analysis", "out", "candidates", "qk")
V = 2000
CHUNK = 20_000_000
DMAX = 8


def ov_table(h):
    from analysis.tools import Model
    m = Model("attn_s0")
    D, DH = 128, 32
    WV = m.Wqkv[2 * D + h * DH:2 * D + (h + 1) * DH]
    WO = m.WO[:, h * DH:(h + 1) * DH]
    pm = m.WP[1:].mean(0)
    hh, _ = m.ln(m.WE + pm, m.g1)
    out = (hh @ WV.T) @ WO.T
    out = out - out.mean(-1, keepdims=True)
    return ((out * m.gf) @ m.WE.T).astype(np.float32)


def main():
    from analysis.mech_eval import stats
    from harness.common import load_corpus

    t0 = time.time()
    s = stats()
    pmi = {d: np.ascontiguousarray(np.asarray(s.pmi[min(d + 1, 8)], dtype=np.float32)).reshape(-1) for d in range(1, DMAX + 1)}
    do_ov = "--ov" in sys.argv
    ov = {}
    if do_ov:
        for h in (0, 2):
            ov[h] = ov_table(h).reshape(-1)
            np.save(os.path.join(OUT, f"ov_h{h}.npy"), ov[h].reshape(V, V))
    corpus = load_corpus()
    n = len(corpus)
    N = {d: np.zeros(V * V, np.float64) for d in range(1, DMAX + 1)}
    S = {d: np.zeros(V * V, np.float64) for d in range(1, DMAX + 1)}
    SO = {(h, d): np.zeros(V * V, np.float64) for h in ov for d in (1, 2, 3)}
    for start in range(0, n - DMAX - 1, CHUNK):
        length = min(CHUNK, n - DMAX - 1 - start)
        x = np.asarray(corpus[start:start + length + DMAX + 1]).astype(np.int64)
        for d in range(1, DMAX + 1):
            k = x[:length]
            pair = k * V + x[d:d + length]
            kv = k * V + x[d + 1:d + 1 + length]
            N[d] += np.bincount(pair, minlength=V * V)
            S[d] += np.bincount(pair, weights=pmi[d][kv], minlength=V * V)
            if d <= 3:
                for h in ov:
                    SO[(h, d)] += np.bincount(pair, weights=ov[h][kv], minlength=V * V)
            del pair, kv
        print(f"{start + length:,} tokens, {time.time() - t0:.0f}s", flush=True)
    for d in range(1, DMAX + 1):
        np.save(os.path.join(OUT, f"N_{d}.npy"), N[d].astype(np.float32).reshape(V, V))
        np.save(os.path.join(OUT, f"S_{d}.npy"), S[d].astype(np.float32).reshape(V, V))
    for (h, d), arr in SO.items():
        np.save(os.path.join(OUT, f"SOV_h{h}_{d}.npy"), arr.astype(np.float32).reshape(V, V))
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
