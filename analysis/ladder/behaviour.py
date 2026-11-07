"""P4, behaviour tests (logs/ladder/design.md; decision 19).

Each claim in an account's CLAIMS may carry
    "tests": [{"statement": "...", "measure": f, "op": "<=" or ">=", "threshold": x}, ...]
where f(probe) returns one number measured on the MODEL (not the account). probe gives, on dev_full:
    probe.X       (n, T) input symbols
    probe.attn    (n, H, T, T) attention weights per head (query rows, key columns)
    probe.x0      (n, T, d) the direct-path embedding
    probe.z       (n, T, H, d_head) each head's output before the output projection
    probe.logp    (n, T, V) the model's log-probs
The test code is the same for every seed (one account structure); it may read the account's per-seed constants
(the account file of that seed is the one loaded).

A test passes if it holds on all five trained seeds and fails on the rung's untrained model (initialised with seed 0,
measured with seed 0's account file), so a test that every model passes cannot count. P4 passes if every claim has at
least one test and every test passes.

    .venv/bin/python -m analysis.ladder.behaviour --rung L1 --arm ln --account "ladder/L1_ln/account_s{seed}.py"
"""
from __future__ import annotations

import argparse
import json
import math
import operator

import numpy as np
import torch

from analysis.ladder.common import SEEDS, load, load_account, windows
from analysis.ladder.scrub import parts
from src.train.model import TinyLM

OPS = {"<=": operator.le, ">=": operator.ge}


class Probe:
    def __init__(self, model, W: np.ndarray):
        model = model.double().eval()
        self.X = W[:, :-1].astype(np.int64)
        self.x0, self.z = parts(model, self.X)
        self._model = model
        self._attn = None
        self._logp = None

    @property
    def attn(self) -> np.ndarray:
        if self._attn is None:
            self._attn = attention(self._model, self.X)
        return self._attn

    @property
    def logp(self) -> np.ndarray:
        if self._logp is None:
            with torch.no_grad():
                self._logp = torch.log_softmax(self._model(torch.as_tensor(self.X)), -1).numpy()
        return self._logp


@torch.no_grad()
def attention(model, X: np.ndarray) -> np.ndarray:
    x = torch.as_tensor(X, dtype=torch.int64)
    n, T = x.shape
    blk = model.blocks[0]
    d, H, dh = model.cfg.d_model, model.cfg.n_heads, model.cfg.d_head
    x0 = model.wte(x) + model.wpe(torch.arange(T))[None]
    q, k, _ = blk.attn.qkv(blk.ln1(x0)).split(d, dim=-1)
    q, k = (a.view(n, T, H, dh).transpose(1, 2) for a in (q, k))
    s = q @ k.transpose(-1, -2) / math.sqrt(dh)
    s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
    return s.softmax(-1).numpy()


def untrained(model):
    torch.manual_seed(0)
    return TinyLM(model.cfg)


def run(rung: str, arm: str, account_pattern: str) -> dict:
    W = windows(rung, "dev_full")
    res = {"rung": rung, "arm": arm, "account": account_pattern, "claims": {}}
    models = {f"s{s}": (load(rung, arm, s), account_pattern.format(seed=s)) for s in SEEDS}
    models["untrained"] = (untrained(models["s0"][0]), account_pattern.format(seed=0))
    measured = {}   # (claim name, test index) -> tag -> (statement, op, threshold, value)
    for tag, (model, path) in models.items():
        probe = Probe(model, W)
        for c in load_account(path).CLAIMS:
            for i, t in enumerate(c.get("tests", [])):
                measured.setdefault((c["name"], i), {})[tag] = (t["statement"], t["op"], float(t["threshold"]),
                                                                float(t["measure"](probe)))
    names = [c["name"] for c in load_account(account_pattern.format(seed=0)).CLAIMS]
    ok = True
    for name in names:
        rows = []
        for (cname, i), per in sorted(measured.items()):
            if cname != name:
                continue
            holds = {tag: bool(OPS[op](v, thr)) for tag, (_, op, thr, v) in per.items()}
            passed = all(holds.get(f"s{s}", False) for s in SEEDS) and not holds["untrained"]
            st, op, thr, _ = per["s0"]
            rows.append({"statement": st, "op": op, "threshold": thr,
                         "values": {tag: v[3] for tag, v in per.items()}, "holds": holds, "pass": passed})
            ok &= passed
        ok &= bool(rows)
        res["claims"][name] = {"tests": rows, "pass": bool(rows) and all(r["pass"] for r in rows)}
    res["P4"] = bool(ok)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True)
    ap.add_argument("--arm", default="ln")
    ap.add_argument("--account", required=True, help="path pattern with {seed}")
    a = ap.parse_args()
    print(json.dumps(run(a.rung, a.arm, a.account), indent=1))


if __name__ == "__main__":
    main()
