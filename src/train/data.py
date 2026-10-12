"""Data preparation: splits, BPE tokenizer, token binaries, fixed evaluation sets.

Run once:  python -m src.train.data prepare

Outputs (all under data/, gitignored):
  data/splits/dev_indices.npy      row indices into official train used as dev (seed 0, 20,000 rows)
  data/tokenizer.json              byte-level BPE, vocab 2000, id 0 = <|eos|>; fit on train-minus-dev
  data/tokens/{train,dev,test}.bin uint16 token streams: shuffled stories, each followed by EOS (id 0)
  data/eval/{dev_fast,dev_full,test_full}.npy   uint16 windows, shape (n, 257)
  data/prep_meta.json              counts, hashes, and every seed used

Windows: a stream of N tokens gives (N - 1) // 256 windows; window i is tokens[256 i : 256 i + 257],
so inputs are the first 256 tokens and targets the last 256. Every token is a target exactly once.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np

RAW_DIR = "data/raw/data"
DATA_DIR = "data"
CTX = 256
WIN = CTX + 1
EOS_ID = 0
EOS_TOKEN = "<|eos|>"
VOCAB = 2000
DEV_STORIES = 20000
DEV_SEED = 0
STREAM_SHUFFLE_SEED = {"train": 1, "dev": 2, "test": 3}  # story order inside each stream
EVAL_SEED = 0
EVAL_SIZES = {"dev_full": ("dev", 2000), "test_full": ("test", 2000)}
DEV_FAST = 200  # first 200 windows of dev_full


def load_split(name: str) -> list[str]:
    import pyarrow.parquet as pq

    pattern = "train-*.parquet" if name == "train" else "validation-*.parquet"
    files = sorted(glob.glob(os.path.join(RAW_DIR, pattern)))
    assert files, f"no parquet files for {name} in {RAW_DIR}"
    texts: list[str] = []
    for f in files:
        texts.extend(pq.read_table(f, columns=["text"]).column("text").to_pylist())
    return texts


def dev_indices(n_train_rows: int) -> np.ndarray:
    rng = np.random.default_rng(DEV_SEED)
    return np.sort(rng.choice(n_train_rows, size=DEV_STORIES, replace=False))


def split_official_train(texts: list[str]) -> tuple[list[str], list[str], np.ndarray]:
    """Return (train_minus_dev, dev, dev_idx). Dev is never used for tokenizer or model training."""
    idx = dev_indices(len(texts))
    mask = np.zeros(len(texts), dtype=bool)
    mask[idx] = True
    train = [t for t, m in zip(texts, mask) if not m]
    dev = [texts[i] for i in idx]
    return train, dev, idx


def fit_tokenizer(train_texts: list[str]):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB,
        special_tokens=[EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    def batches(bs: int = 10000):
        for i in range(0, len(train_texts), bs):
            yield train_texts[i : i + bs]

    tok.train_from_iterator(batches(), trainer=trainer, length=len(train_texts))
    assert tok.token_to_id(EOS_TOKEN) == EOS_ID, tok.token_to_id(EOS_TOKEN)
    assert tok.get_vocab_size() == VOCAB, tok.get_vocab_size()
    return tok


def encode_stream(tok, texts: list[str], shuffle_seed: int, chunk: int = 20000) -> np.ndarray:
    order = np.random.default_rng(shuffle_seed).permutation(len(texts))
    parts = []
    for i in range(0, len(order), chunk):
        batch = [texts[j] for j in order[i : i + chunk]]
        for enc in tok.encode_batch(batch, add_special_tokens=False):
            ids = enc.ids
            parts.append(np.asarray(ids + [EOS_ID], dtype=np.uint16))
    return np.concatenate(parts)


def n_windows(stream_len: int) -> int:
    return (stream_len - 1) // CTX


def window(stream: np.ndarray, i: int) -> np.ndarray:
    return stream[i * CTX : i * CTX + WIN]


def windows(stream: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.stack([window(stream, int(i)) for i in idx])


def sha(arr_or_bytes) -> str:
    b = arr_or_bytes.tobytes() if isinstance(arr_or_bytes, np.ndarray) else arr_or_bytes
    return hashlib.sha256(b).hexdigest()[:16]


def prepare() -> None:
    t0 = time.time()
    os.makedirs(f"{DATA_DIR}/splits", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/tokens", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/eval", exist_ok=True)

    official_train = load_split("train")
    test = load_split("test")
    train, dev, idx = split_official_train(official_train)
    del official_train
    np.save(f"{DATA_DIR}/splits/dev_indices.npy", idx)
    print(f"[{time.time()-t0:.0f}s] stories: train={len(train)} dev={len(dev)} test={len(test)}")

    tok = fit_tokenizer(train)
    tok.save(f"{DATA_DIR}/tokenizer.json")
    print(f"[{time.time()-t0:.0f}s] tokenizer fit on {len(train)} train-minus-dev stories")

    meta = {
        "source": "roneneldan/TinyStories parquet (original, not V2)",
        "n_stories": {"train": len(train), "dev": len(dev), "test": len(test)},
        "dev_seed": DEV_SEED,
        "dev_indices_sha": sha(idx),
        "tokenizer_fit_on": "train_minus_dev",
        "tokenizer_fit_n_stories": len(train),
        "stream_shuffle_seed": STREAM_SHUFFLE_SEED,
        "eval_seed": EVAL_SEED,
        "n_tokens": {},
        "n_windows": {},
    }
    streams = {}
    for name, texts in [("train", train), ("dev", dev), ("test", test)]:
        s = encode_stream(tok, texts, STREAM_SHUFFLE_SEED[name])
        s.tofile(f"{DATA_DIR}/tokens/{name}.bin")
        streams[name] = s
        meta["n_tokens"][name] = int(len(s))
        meta["n_windows"][name] = n_windows(len(s))
        print(f"[{time.time()-t0:.0f}s] {name}: {len(s):,} tokens, {n_windows(len(s)):,} windows")

    rng = np.random.default_rng(EVAL_SEED)
    for set_name, (split, n) in EVAL_SIZES.items():
        pick = rng.choice(n_windows(len(streams[split])), size=n, replace=False)
        w = windows(streams[split], pick)
        np.save(f"{DATA_DIR}/eval/{set_name}.npy", w)
        np.save(f"{DATA_DIR}/eval/{set_name}_window_idx.npy", pick)
        meta[f"{set_name}_sha"] = sha(w)
        if set_name == "dev_full":
            np.save(f"{DATA_DIR}/eval/dev_fast.npy", w[:DEV_FAST])
            meta["dev_fast_sha"] = sha(w[:DEV_FAST])

    with open(f"{DATA_DIR}/prep_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{time.time()-t0:.0f}s] done")


def load_stream(split: str) -> np.ndarray:
    return np.memmap(f"{DATA_DIR}/tokens/{split}.bin", dtype=np.uint16, mode="r")


def load_eval(name: str) -> np.ndarray:
    return np.load(f"{DATA_DIR}/eval/{name}.npy")


def load_tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(f"{DATA_DIR}/tokenizer.json")


class BatchSampler:
    """Deterministic batches: batch(step) depends only on (seed, step).

    Epoch e uses the window permutation default_rng([seed, e]).permutation(n_windows).
    Step s takes positions [s*B, (s+1)*B) of the concatenated per-epoch permutations.
    """

    def __init__(self, stream: np.ndarray, seed: int, batch_windows: int):
        self.stream = stream
        self.seed = seed
        self.B = batch_windows
        self.n = n_windows(len(stream))
        self._perm_cache: dict[int, np.ndarray] = {}

    def _perm(self, epoch: int) -> np.ndarray:
        if epoch not in self._perm_cache:
            self._perm_cache = {epoch: np.random.default_rng([self.seed, epoch]).permutation(self.n)}
        return self._perm_cache[epoch]

    def indices(self, step: int) -> np.ndarray:
        out = []
        pos = step * self.B
        while len(out) < self.B:
            e, off = divmod(pos, self.n)
            take = min(self.B - len(out), self.n - off)
            out.extend(self._perm(e)[off : off + take].tolist())
            pos += take
        return np.asarray(out)

    def batch(self, step: int) -> np.ndarray:
        idx = self.indices(step)
        return np.stack([np.asarray(window(self.stream, int(i)), dtype=np.int64) for i in idx])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare"])
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare()
