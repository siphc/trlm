"""Build the target pool for readable training (program v1.3's outputs on fixed TRAIN windows).

  .venv/bin/python -m analysis.readable_training.build_pool check_fresh   # cached statistics == fresh setup (3 windows)
  .venv/bin/python -m analysis.readable_training.build_pool coverage      # rule R1: choose k_out and k_h (256 windows)
  .venv/bin/python -m analysis.readable_training.build_pool build [--workers 8]   # resumable, in shards
  .venv/bin/python -m analysis.readable_training.build_pool verify        # shapes, finiteness, spot checks

Files in data/readable/pool/ (numpy .npy, memory-mapped):
  tokens.npy            (N_train + N_heldout, 257) uint16, train windows first
  out_ids.npy           (N_train, 256, k_out) uint16   the program's top-k next tokens per position
  out_lp.npy            (N_train, 256, k_out) float16  their log-probabilities
  out_tail.npy          (N_train, 256) float32         probability left outside the top k
  att{h}_keys.npy       (N_train, 256, k_h) uint8      rule h's top-k key positions per query row (h = 0, 1, 2 for
  att{h}_p.npy          (N_train, 256, k_h) float16    LOCAL-0, RECALL, LOCAL-2); padding entries point at the query
  att{h}_tail.npy       (N_train, 256) float32         itself with probability 0
  heldout_lp.npy        (N_heldout, 256, 2000) float16 full program log-probabilities
  heldout_att.npy       (N_heldout, 3, 256, 256) float16 full rule attention
  meta.json             sizes, k values, program sha, pool indices sha
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

from analysis.readable_training.common import (REPO, RULE_HEADS, T, config, load_rules, pool_dir, pool_indices,
                                               save_rules, windows_at)

V = 2000
SHARD = 256


def program():
    from analysis.mech_eval import program as P

    return P()


def targets_for(P, t):
    """Program log-probs (256, V) and the three rules' attention (3, 256, 256) for one window's first 256 tokens."""
    import importlib

    lq = P.predict(t)
    att = np.stack([importlib.import_module(f"program.{h}").attention(t, P.STATS) for h in RULE_HEADS])
    return lq, att


def topk_rows(x, k):
    """Indices and values of the k largest entries of each row (unsorted)."""
    idx = np.argpartition(-x, k - 1, axis=-1)[..., :k]
    return idx, np.take_along_axis(x, idx, -1)


def cmd_check_fresh():
    """Selftest 8: the analysis cache (analysis/out/program_stats) gives the same program outputs as a fresh harness
    setup(corpus), on 3 pool windows."""
    import importlib

    from harness.common import prepare_program

    tr, _ = pool_indices()
    W = windows_at(tr[:3])
    P = program()
    cached = [P.predict(w[:T]) for w in W]
    sys.modules.pop("program", None)
    fresh_mod = importlib.import_module("program")
    t0 = time.time()
    prepare_program(fresh_mod)
    print(f"fresh setup {time.time() - t0:.0f}s", flush=True)
    diffs = [float(np.abs(fresh_mod.predict(w[:T]) - c).max()) for w, c in zip(W, cached)]
    ok = max(diffs) < 1e-3
    print(json.dumps({"max_abs_logprob_diff": diffs, "ok": ok}))
    save_rules({"selftest8_fresh_vs_cache": {"max_abs_logprob_diff": max(diffs), "ok": ok}})
    if not ok:
        raise SystemExit("cached statistics differ from a fresh setup: rebuild analysis/out/program_stats first")


def cmd_coverage():
    """Rule R1 on 256 pool training windows."""
    cfg = config()["pool"]
    tr, _ = pool_indices()
    W = windows_at(tr[:256])
    P = program()
    out_cov = {k: [] for k in cfg["k_out"]}
    att_cov = {h: {k: [] for k in cfg["k_att"]} for h in range(3)}
    for w in W:
        lq, att = targets_for(P, w[:T])
        p = np.sort(np.exp(lq), -1)[:, ::-1]
        c = np.cumsum(p, -1)
        for k in out_cov:
            out_cov[k].append(c[:, k - 1].mean())
        for h in range(3):
            a = np.sort(att[h], -1)[:, ::-1]
            ca = np.cumsum(a, -1)
            for k in att_cov[h]:
                att_cov[h][k].append(ca[1:, min(k, T) - 1].mean())        # query rows i >= 1
    out_cov = {k: float(np.mean(v)) for k, v in out_cov.items()}
    att_cov = {h: {k: float(np.mean(v)) for k, v in d.items()} for h, d in att_cov.items()}
    k_out = next((k for k in sorted(out_cov) if out_cov[k] >= 0.95), max(out_cov))
    k_att = {h: next((k for k in sorted(att_cov[h]) if att_cov[h][k] >= 0.95), max(att_cov[h])) for h in range(3)}
    n = cfg["n_train"]
    gb = (n * T * k_out * 4 + n * T * 4 + sum(n * T * k * 3 + n * T * 4 for k in k_att.values())
          + cfg["n_heldout"] * T * (V + 3 * T) * 2) / 1e9
    res = {"R1": {"coverage_out": out_cov, "coverage_att": {RULE_HEADS[h]: v for h, v in att_cov.items()},
                  "k_out": k_out, "k_att": {RULE_HEADS[h]: k for h, k in k_att.items()}, "storage_gb": gb,
                  "storage_ok": gb <= 10.0, "n_windows": 256}}
    save_rules(res)
    print(json.dumps(res, indent=1))


def paths(d, rules):
    k_out = rules["R1"]["k_out"]
    k_att = [rules["R1"]["k_att"][h] for h in RULE_HEADS]
    return k_out, k_att


def create(d, n_tr, n_ho, k_out, k_att, tokens):
    from numpy.lib.format import open_memmap

    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "tokens.npy"), tokens.astype(np.uint16))
    specs = {"out_ids": ((n_tr, T, k_out), np.uint16), "out_lp": ((n_tr, T, k_out), np.float16),
             "out_tail": ((n_tr, T), np.float32), "heldout_lp": ((n_ho, T, V), np.float16),
             "heldout_att": ((n_ho, 3, T, T), np.float16)}
    for h, k in enumerate(k_att):
        specs[f"att{h}_keys"] = ((n_tr, T, k), np.uint8)
        specs[f"att{h}_p"] = ((n_tr, T, k), np.float16)
        specs[f"att{h}_tail"] = ((n_tr, T), np.float32)
    for name, (shape, dt) in specs.items():
        p = os.path.join(d, f"{name}.npy")
        if not os.path.exists(p):
            open_memmap(p, mode="w+", dtype=dt, shape=shape)
    os.makedirs(os.path.join(d, "done"), exist_ok=True)


def work(args):
    """Fill one shard (rows [lo, hi) of the pool, train or held-out)."""
    shard, lo, hi, kind = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    d = pool_dir()
    marker = os.path.join(d, "done", f"{kind}_{shard:04d}")
    if os.path.exists(marker):
        return shard
    rules = load_rules()
    k_out, k_att = paths(d, rules)
    tokens = np.load(os.path.join(d, "tokens.npy"), mmap_mode="r")
    P = program()
    n_tr = config()["pool"]["n_train"]
    if kind == "train":
        mm = {n: np.load(os.path.join(d, f"{n}.npy"), mmap_mode="r+") for n in
              ["out_ids", "out_lp", "out_tail"] + [f"att{h}_{x}" for h in range(3) for x in ("keys", "p", "tail")]}
        for r in range(lo, hi):
            lq, att = targets_for(P, np.asarray(tokens[r, :T], np.int64))
            ids, vals = topk_rows(lq, k_out)
            mm["out_ids"][r] = ids.astype(np.uint16)
            mm["out_lp"][r] = vals.astype(np.float16)
            mm["out_tail"][r] = np.clip(1.0 - np.exp(vals.astype(np.float64)).sum(-1), 0.0, 1.0)
            for h in range(3):
                k = k_att[h]
                a = att[h].astype(np.float64)
                keys, p = topk_rows(a, k)
                bad = keys > np.arange(T)[:, None]            # future positions (short rows): padding -> the query
                keys = np.where(bad, np.arange(T)[:, None], keys)
                p = np.where(bad, 0.0, p)
                mm[f"att{h}_keys"][r] = keys.astype(np.uint8)
                mm[f"att{h}_p"][r] = p.astype(np.float16)
                mm[f"att{h}_tail"][r] = np.clip(1.0 - p.sum(-1), 0.0, 1.0)
        for m in mm.values():
            m.flush()
    else:
        hl = np.load(os.path.join(d, "heldout_lp.npy"), mmap_mode="r+")
        ha = np.load(os.path.join(d, "heldout_att.npy"), mmap_mode="r+")
        for r in range(lo, hi):
            lq, att = targets_for(P, np.asarray(tokens[n_tr + r, :T], np.int64))
            hl[r] = lq.astype(np.float16)
            ha[r] = att.astype(np.float16)
        hl.flush(); ha.flush()
    open(marker, "w").close()
    return shard


def cmd_build(workers):
    import multiprocessing as mp

    from harness.common import program_sha

    rules = load_rules()
    assert "R1" in rules, "run coverage first (rule R1)"
    assert rules.get("selftest8_fresh_vs_cache", {}).get("ok"), "run check_fresh first (selftest 8)"
    cfg = config()["pool"]
    d = pool_dir()
    tr, ho = pool_indices()
    k_out, k_att = paths(d, rules)
    idx_sha = hashlib.sha256(np.concatenate([tr, ho]).tobytes()).hexdigest()[:16]
    meta_p = os.path.join(d, "meta.json")
    if os.path.exists(meta_p):
        meta = json.load(open(meta_p))
        assert meta["indices_sha"] == idx_sha and meta["k_out"] == k_out, "pool exists with other settings"
    else:
        create(d, len(tr), len(ho), k_out, k_att, windows_at(np.concatenate([tr, ho])))
        meta = {"n_train": len(tr), "n_heldout": len(ho), "k_out": k_out, "k_att": k_att, "indices_sha": idx_sha,
                "program_sha": program_sha(os.path.join(REPO, "program")),
                "rng": cfg["rng"], "created": time.strftime("%Y-%m-%d %H:%M")}
        np.save(os.path.join(d, "indices.npy"), np.concatenate([tr, ho]))
        with open(meta_p, "w") as f:
            json.dump(meta, f, indent=1)
    jobs = [(s, lo, min(lo + SHARD, len(tr)), "train") for s, lo in enumerate(range(0, len(tr), SHARD))]
    jobs += [(s, lo, min(lo + 64, len(ho)), "heldout") for s, lo in enumerate(range(0, len(ho), 64))]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for i, s in enumerate(pool.imap_unordered(work, jobs)):
            if i % 8 == 0 or i == len(jobs) - 1:
                print(f"{i + 1}/{len(jobs)} shards [{time.time() - t0:.0f}s]", flush=True)
    meta["complete"] = True
    meta["build_seconds"] = time.time() - t0
    with open(meta_p, "w") as f:
        json.dump(meta, f, indent=1)


def cmd_verify():
    d = pool_dir()
    meta = json.load(open(os.path.join(d, "meta.json")))
    assert meta.get("complete"), "pool build not complete"
    n_done = len(os.listdir(os.path.join(d, "done")))
    lp = np.load(os.path.join(d, "out_lp.npy"), mmap_mode="r")
    tail = np.load(os.path.join(d, "out_tail.npy"), mmap_mode="r")
    rng = np.random.default_rng(0)
    rows = rng.choice(lp.shape[0], 64, replace=False)
    mass = np.exp(lp[rows].astype(np.float64)).sum(-1) + tail[rows]
    res = {"shards_done": n_done, "finite": bool(np.isfinite(lp[rows]).all()),
           "mass_min": float(mass.min()), "mass_max": float(mass.max()),
           "mean_tail": float(np.asarray(tail[rows]).mean())}
    # recompute 2 rows exactly
    P = program()
    tok = np.load(os.path.join(d, "tokens.npy"), mmap_mode="r")
    ids = np.load(os.path.join(d, "out_ids.npy"), mmap_mode="r")
    for r in rows[:2]:
        lq, _ = targets_for(P, np.asarray(tok[r, :T], np.int64))
        res[f"row{r}_max_abs_diff"] = float(np.abs(lq[np.arange(T)[:, None], ids[r].astype(np.int64)]
                                                   - lp[r].astype(np.float64)).max())
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check_fresh", "coverage", "build", "verify"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    {"check_fresh": cmd_check_fresh, "coverage": cmd_coverage, "verify": cmd_verify}.get(
        a.cmd, lambda: cmd_build(a.workers))()


if __name__ == "__main__":
    main()
