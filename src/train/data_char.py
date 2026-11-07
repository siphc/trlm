"""Character-level data for the ladder of tiny language models (logs/ladder_accounts/design.md, decision 16).

Run once:  .venv/bin/python -m src.train.data_char prepare        (about 10-20 min on the laptop CPU)

Same stories and splits as the main track (src/train/data.py): the dev slice is 20,000 stories sampled from the
official train split with seed 0; the official validation split (the test split) is not read at all.

Symbols:
  0 = end of story (appended after each story), 1 = UNK (any character seen fewer than RARE times in train-minus-dev),
  2.. = the kept characters in code-point order. The alphabet is counted on train-minus-dev only.

Outputs (under data/char/, gitignored):
  alphabet.json                 the kept characters, their ids and train counts, and the UNK characters' total
  {train,dev}.bin               uint8 symbol streams: shuffled stories, each followed by symbol 0
  eval/dev_{fast,full}_c{ctx}.npy   uint8 windows (n, ctx + 1) from the dev stream, for ctx in CONTEXTS
  baselines/{unigram,bigram,trigram}_logprobs.npy   n-gram tables from the train stream (smoothing below)
  meta.json                     counts, hashes, seeds

Windows: a stream of N symbols gives (N - 1) // ctx windows; window i is stream[ctx i : ctx i + ctx + 1].
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from src.train.data import (DEV_FAST, EVAL_SEED, STREAM_SHUFFLE_SEED, load_split, sha, split_official_train)

DIR = "data/char"
EOS_ID = 0
UNK_ID = 1
RARE = 1000                 # characters seen fewer times than this in train-minus-dev map to UNK (design, Data)
CONTEXTS = (64, 256)        # rung contexts (L1-L7: 64; L8 and up: 256)
DEV_FULL = 2000
MAX_CP = 0x110000
CHUNK = 20000               # stories per encoding chunk


# ----------------------------------------------------------------------------- alphabet and encoding

def count_chars(texts: list[str]) -> np.ndarray:
    counts = np.zeros(MAX_CP, np.int64)
    for i in range(0, len(texts), CHUNK):
        cp = np.frombuffer("".join(texts[i:i + CHUNK]).encode("utf-32-le"), np.uint32)
        counts += np.bincount(cp, minlength=MAX_CP)
    return counts


def build_alphabet(counts: np.ndarray) -> dict:
    assert counts[0] == 0, "NUL occurs in the text; it cannot double as the story separator"
    kept = [int(c) for c in np.flatnonzero(counts >= RARE)]
    return {
        "rare_threshold": RARE,
        "ids": {"eos": EOS_ID, "unk": UNK_ID},
        "chars": [chr(c) for c in kept],                 # symbol id = 2 + index
        "counts": [int(counts[c]) for c in kept],
        "unk_types": int(((counts > 0) & (counts < RARE)).sum()),
        "unk_count": int(counts[(counts > 0) & (counts < RARE)].sum()),
        "vocab_size": 2 + len(kept),
    }


def lookup_table(alpha: dict) -> np.ndarray:
    lut = np.full(MAX_CP, UNK_ID, np.uint8)
    lut[0] = EOS_ID                                      # the separator inserted between stories
    for i, ch in enumerate(alpha["chars"]):
        lut[ord(ch)] = 2 + i
    return lut


def encode_stream(texts: list[str], lut: np.ndarray, shuffle_seed: int) -> np.ndarray:
    """Stories in the order default_rng(shuffle_seed).permutation, each followed by symbol 0 (as data.py)."""
    order = np.random.default_rng(shuffle_seed).permutation(len(texts))
    parts = []
    for i in range(0, len(order), CHUNK):
        joined = "\x00".join(texts[j] for j in order[i:i + CHUNK]) + "\x00"
        parts.append(lut[np.frombuffer(joined.encode("utf-32-le"), np.uint32)])
    return np.concatenate(parts)


def decode(symbols, alpha: dict) -> str:
    chars = alpha["chars"]
    out = []
    for s in symbols:
        s = int(s)
        out.append("\n<EOS>\n" if s == EOS_ID else "�" if s == UNK_ID else chars[s - 2])
    return "".join(out)


# ----------------------------------------------------------------------------- windows

def n_windows(stream_len: int, ctx: int) -> int:
    return (stream_len - 1) // ctx


def windows(stream: np.ndarray, idx: np.ndarray, ctx: int) -> np.ndarray:
    return np.stack([np.asarray(stream[int(i) * ctx:int(i) * ctx + ctx + 1]) for i in idx])


# ----------------------------------------------------------------------------- baselines

def ngram_tables(s: np.ndarray, V: int, chunk: int = 100_000_000) -> dict:
    """Train-stream n-gram log-probs. Unigram add-one; bigram P(b|a) = (c(a,b) + P_uni(b)) / (c(a) + 1);
    trigram P(c|a,b) = (c(a,b,c) + P_big(c|b)) / (c(a,b) + 1): a Dirichlet prior of weight 1 on the next-lower order,
    as the main track's bigram (harness/build_baselines.py). Counted in chunks (the stream is a uint8 memmap)."""
    N = len(s)
    uni = np.zeros(V, np.float64)
    big = np.zeros(V * V, np.float64)
    tri = np.zeros(V ** 3, np.float64)
    for i in range(0, N, chunk):
        x = np.asarray(s[i:min(i + chunk + 2, N)], dtype=np.int64)
        n = min(chunk, N - i)                          # positions i .. i + n - 1 start an n-gram in this chunk
        uni += np.bincount(x[:n], minlength=V)
        m2 = min(n, len(x) - 1)
        big += np.bincount(x[:m2] * V + x[1:m2 + 1], minlength=V * V)
        m3 = min(n, len(x) - 2)
        tri += np.bincount((x[:m3] * V + x[1:m3 + 1]) * V + x[2:m3 + 2], minlength=V ** 3)
    p_uni = (uni + 1) / (uni.sum() + V)
    big = big.reshape(V, V)
    p_big = (big + p_uni[None]) / (big.sum(1, keepdims=True) + 1)
    tri = tri.reshape(V, V, V)
    p_tri = (tri + p_big[None]) / (tri.sum(2, keepdims=True) + 1)
    return {"unigram": np.log(p_uni), "bigram": np.log(p_big), "trigram": np.log(p_tri)}


def baseline_logprobs(windows_: np.ndarray, order: int, tables: dict | None = None) -> np.ndarray:
    """(n, ctx, V) log-probs of the n-gram baseline for the inputs windows_[:, :-1]. Row i predicts the symbol after
    inputs[: i + 1] and uses only context inside the window, as the model does: the trigram falls back to the bigram
    at row 0."""
    tables = tables or load_baselines()
    x = windows_[:, :-1].astype(np.int64)
    n, T = x.shape
    if order == 1:
        return np.broadcast_to(tables["unigram"], (n, T, len(tables["unigram"]))).copy()
    out = tables["bigram"][x]
    if order == 3 and T > 1:
        out[:, 1:] = tables["trigram"][x[:, :-1], x[:, 1:]]
    return out


# ----------------------------------------------------------------------------- prepare and load

def prepare() -> None:
    t0 = time.time()
    official_train = load_split("train")
    train, dev, idx = split_official_train(official_train)
    del official_train
    assert len(dev) == 20000
    print(f"[{time.time()-t0:.0f}s] stories: train-minus-dev {len(train):,}, dev {len(dev):,}", flush=True)

    alpha = build_alphabet(count_chars(train))
    os.makedirs(f"{DIR}/eval", exist_ok=True)
    os.makedirs(f"{DIR}/baselines", exist_ok=True)
    with open(f"{DIR}/alphabet.json", "w") as f:
        json.dump(alpha, f, indent=1, ensure_ascii=False)
    print(f"[{time.time()-t0:.0f}s] alphabet: {alpha['vocab_size']} symbols; UNK covers {alpha['unk_types']} "
          f"character types, {alpha['unk_count']:,} occurrences", flush=True)
    lut = lookup_table(alpha)

    meta = {"source": "roneneldan/TinyStories parquet, as src/train/data.py", "dev_indices_sha": sha(idx),
            "alphabet_fit_on": "train_minus_dev", "rare_threshold": RARE, "vocab_size": alpha["vocab_size"],
            "stream_shuffle_seed": {k: STREAM_SHUFFLE_SEED[k] for k in ("train", "dev")}, "eval_seed": EVAL_SEED,
            "n_symbols": {}, "eval_sha": {}}
    streams = {}
    for name, texts in (("train", train), ("dev", dev)):
        s = encode_stream(texts, lut, STREAM_SHUFFLE_SEED[name])
        s.tofile(f"{DIR}/{name}.bin")
        streams[name] = s
        meta["n_symbols"][name] = int(len(s))
        print(f"[{time.time()-t0:.0f}s] {name}: {len(s):,} symbols", flush=True)
    del train, dev

    for ctx in CONTEXTS:
        rng = np.random.default_rng([EVAL_SEED, ctx])
        pick = rng.choice(n_windows(len(streams["dev"]), ctx), size=DEV_FULL, replace=False)
        w = windows(streams["dev"], pick, ctx)
        np.save(f"{DIR}/eval/dev_full_c{ctx}.npy", w)
        np.save(f"{DIR}/eval/dev_full_c{ctx}_window_idx.npy", pick)
        np.save(f"{DIR}/eval/dev_fast_c{ctx}.npy", w[:DEV_FAST])
        meta["eval_sha"][f"dev_full_c{ctx}"] = sha(w)

    tables = ngram_tables(streams["train"], alpha["vocab_size"])
    for k, v in tables.items():
        np.save(f"{DIR}/baselines/{k}_logprobs.npy", v)
    for ctx in CONTEXTS:
        w = load_eval("dev_full", ctx)
        tgt = w[:, 1:].astype(np.int64)
        for order, name in ((1, "unigram"), (2, "bigram"), (3, "trigram")):
            lp = baseline_logprobs(w, order, tables)
            ce = -np.take_along_axis(lp, tgt[..., None], -1).mean()
            meta.setdefault("baseline_ce_dev_full", {})[f"{name}_c{ctx}"] = float(ce)
    with open(f"{DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta["baseline_ce_dev_full"], indent=1))
    print(f"[{time.time()-t0:.0f}s] done", flush=True)


def load_alphabet() -> dict:
    with open(f"{DIR}/alphabet.json") as f:
        return json.load(f)


def vocab_size() -> int:
    return load_alphabet()["vocab_size"]


def load_stream(split: str) -> np.ndarray:
    assert split in ("train", "dev"), "the ladder never reads the test split"
    return np.memmap(f"{DIR}/{split}.bin", dtype=np.uint8, mode="r")


def load_eval(name: str, ctx: int) -> np.ndarray:
    return np.load(f"{DIR}/eval/{name}_c{ctx}.npy")


def load_baselines() -> dict:
    return {k: np.load(f"{DIR}/baselines/{k}_logprobs.npy") for k in ("unigram", "bigram", "trigram")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare"])
    if ap.parse_args().cmd == "prepare":
        prepare()
