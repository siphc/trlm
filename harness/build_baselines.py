"""Build n-gram baseline tables from the TRAIN split only. READ-ONLY for replacement agents.

Usage: python -m harness.build_baselines

Writes data/baselines/:
  unigram_logprobs.npy  (2000,)        add-one smoothed
  bigram_logprobs.npy   (2000, 2000)   P(b|a) = (c(a,b) + 1 * P_uni(b)) / (c(a) + 1)   (Dirichlet prior on unigram)
  trigram_{keys,counts}.npy, trigram_ctx_{keys,counts,types}.npy
      sparse trigram counts from the first TRIGRAM_TOKENS tokens of the train stream; the trigram
      program interpolates with the bigram by Witten-Bell.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
V = 2000
TRIGRAM_TOKENS = 100_000_000
CHUNK = 50_000_000


def main() -> None:
    os.chdir(REPO)
    out = "data/baselines"
    os.makedirs(out, exist_ok=True)
    s = np.memmap("data/tokens/train.bin", dtype=np.uint16, mode="r")
    N = len(s)
    uni = np.bincount(s, minlength=V).astype(np.float64)
    p_uni = (uni + 1) / (uni.sum() + V)
    np.save(f"{out}/unigram_logprobs.npy", np.log(p_uni))

    big = np.zeros(V * V, dtype=np.int64)
    for i in range(0, N - 1, CHUNK):
        a = np.asarray(s[i : min(i + CHUNK, N - 1)], dtype=np.int64)
        b = np.asarray(s[i + 1 : min(i + CHUNK, N - 1) + 1], dtype=np.int64)
        big += np.bincount(a * V + b, minlength=V * V)
    big = big.reshape(V, V).astype(np.float64)
    p_big = (big + p_uni[None, :]) / (big.sum(1, keepdims=True) + 1.0)
    np.save(f"{out}/bigram_logprobs.npy", np.log(p_big).astype(np.float32))
    print(f"bigram from {N:,} tokens; nonzero pairs {(big > 0).sum():,}")

    t = np.asarray(s[:TRIGRAM_TOKENS], dtype=np.int64)
    keys = (t[:-2] * V + t[1:-1]) * V + t[2:]
    k, c = np.unique(keys, return_counts=True)
    np.save(f"{out}/trigram_keys.npy", k)
    np.save(f"{out}/trigram_counts.npy", c.astype(np.int64))
    ctx = k // V
    ck, first = np.unique(ctx, return_index=True)
    ccount = np.add.reduceat(c, first)
    ctypes = np.diff(np.append(first, len(ctx)))
    np.save(f"{out}/trigram_ctx_keys.npy", ck)
    np.save(f"{out}/trigram_ctx_counts.npy", ccount.astype(np.int64))
    np.save(f"{out}/trigram_ctx_types.npy", ctypes.astype(np.int64))
    print(f"trigram from {len(t):,} tokens; {len(k):,} distinct trigrams, {len(ck):,} contexts")


if __name__ == "__main__":
    main()
