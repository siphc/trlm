"""Fit RECALL attention-rule variants to model head 1's OWN attention pattern (pattern cross-entropy), as v1 did.

score(i, j) = sum_f theta_f * feature_f(i, j) over keys j <= i; softmax over j. Every feature is a function of the
query token / position and key token / position only (what a one-layer head can compute). Fit on 32 TRAIN windows
(seed 2024), report held-out pattern KL(model || rule) on 16 TRAIN windows (seed 2000).

  python fit_attn.py base eos quad ...     (named variants below)
"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/recall")
import lib  # noqa: E402

from analysis.fitlib import train_windows  # noqa: E402
from analysis.mech_eval import program  # noqa: E402
from program import head1  # noqa: E402

torch.set_num_threads(2)
T = 256
_TABLES = {}


def tables(P):
    """Extra token-pair tables (corpus only)."""
    if not _TABLES:
        s = P.STATS
        simL = np.load(f"{lib.OUT}/simL.npy")
        simR = np.load(f"{lib.OUT}/simR.npy")
        P1 = np.exp(np.asarray(s.log_follow, np.float64))
        P1 /= P1.sum(1, keepdims=True)
        _TABLES["simR"] = simR
        _TABLES["succ_sim"] = (P1 @ simL.astype(np.float64)).astype(np.float32)   # E_{v~P1(.|q)} simL(v, k)
        _TABLES["logP1"] = np.maximum(np.asarray(s.log_follow, np.float32), -12.0)
        _TABLES["pmi1"] = np.asarray(s.pmi[1], np.float32)
        # INSIGHT ONLY (never for a program): exact token-token score term of head 1 from the weights
        _TABLES["TT_exact"] = np.load("/Users/svi/Desktop/trlm/analysis/scratch/head1_card/TT.npy")
    return _TABLES


def features(P, t, names):
    """(n, n, F) features for one window."""
    s = P.STATS
    n = len(t)
    d = (np.arange(n)[:, None] - np.arange(n)[None, :]).astype(np.float32)
    g = head1.GROUP[t]
    gq = g
    out = []
    for nm in names:
        if nm == "dist":
            out.append(-d / 32.0)
        elif nm == "dist2":
            out.append(-(d / 64.0) ** 2)
        elif nm.startswith("sal."):
            k = head1.GROUP_NAMES.index(nm[4:])
            out.append(np.broadcast_to((g == k)[None, :].astype(np.float32), (n, n)))
        elif nm == "self":
            out.append(np.eye(n, dtype=np.float32))
        elif nm == "succ":
            out.append(np.asarray(s.class_successor[t[:, None], t[None, :]], np.float32))
        elif nm == "eos":
            out.append(np.broadcast_to((t == 0)[None, :].astype(np.float32), (n, n)))
        elif nm.startswith("qdist."):                     # recency rate by query group
            k = head1.GROUP_NAMES.index(nm[6:])
            out.append((gq == k)[:, None].astype(np.float32) * (-d / 32.0))
        elif nm in ("simR", "succ_sim", "logP1", "pmi1", "TT_exact"):
            tab = tables(P)[nm]
            out.append(np.asarray(tab[t[:, None], t[None, :]], np.float32))
        elif nm in KEY_FLAGS:                             # per-key-token flag or corpus feature
            out.append(np.broadcast_to(key_flag(P, nm)[t][None, :].astype(np.float32), (n, n)))
        elif nm == "sentend_pair":                        # sentence end attending to earlier sentence ends
            se = np.asarray(s.sentence_end[t] > 0)
            out.append((se[:, None] & se[None, :]).astype(np.float32))
        else:
            raise KeyError(nm)
    return np.stack(out, -1)


KEY_FLAGS = ("cap", "cap_unspaced", "quote", "topical", "info8", "info1")
_FLAGS = {}


def key_flag(P, nm):
    """Spelling flags of the token text, or corpus features of the token (no model)."""
    if nm not in _FLAGS:
        from program.vocab import TOKENS
        s = P.STATS
        if nm == "cap":
            _FLAGS[nm] = np.array([x.lstrip(" ")[:1].isupper() for x in TOKENS], np.float32)
        elif nm == "cap_unspaced":
            _FLAGS[nm] = np.array([x[:1].isupper() for x in TOKENS], np.float32)
        elif nm == "quote":
            _FLAGS[nm] = np.array(['"' in x for x in TOKENS], np.float32)
        elif nm == "topical":                      # KL(P_topic(. | s) || P(.)): how much s tells about the next 9..24 tokens
            tp = np.asarray(s.topic, np.float64)
            pv = np.asarray(s.p_uni, np.float64)
            q = np.exp(tp) * pv[None, :]
            q /= q.sum(1, keepdims=True)
            _FLAGS[nm] = (q * (np.log(q) - np.log(pv)[None, :])).sum(1).astype(np.float32)
        elif nm == "info8":
            _FLAGS[nm] = np.asarray(s.info[7], np.float32)
        elif nm == "info1":
            _FLAGS[nm] = np.asarray(s.info[0], np.float32)
    return _FLAGS[nm]


V1 = {"dist": 32.0 / head1.MEMORY, **{f"sal.{k}": v for k, v in head1.SALIENCE.items()},
      "self": head1.SELF_BONUS, "succ": head1.SUCCESSOR}
BASE = ["dist"] + [f"sal.{k}" for k in head1.GROUP_NAMES] + ["self", "succ"]
VARIANTS = {
    "base": BASE,
    "eos": BASE + ["eos"],
    "quad": BASE + ["dist2"],
    "qdist": BASE + [f"qdist.{k}" for k in head1.GROUP_NAMES[:-1]],
    "simR": BASE + ["simR"],
    "succ_sim": BASE + ["succ_sim"],
    "logP1": BASE + ["logP1"],
    "pmi1": BASE + ["pmi1"],
    "sentend_pair": BASE + ["sentend_pair"],
    "TT_exact": BASE + ["TT_exact"],
    "TT_exact_only": ["dist", "self", "TT_exact"],
    "cap": BASE + ["cap"],
    "cap2": BASE + ["cap", "cap_unspaced"],
    "quote": BASE + ["quote"],
    "topical": BASE + ["topical"],
    "info8": BASE + ["info8"],
    "keyflags": BASE + ["cap", "cap_unspaced", "quote", "eos"],
    "keyflags_t": BASE + ["cap", "cap_unspaced", "quote", "eos", "topical"],
    "keyflags_all": BASE + ["cap", "cap_unspaced", "quote", "eos", "topical", "sentend_pair", "pmi1"],
}


def model_patterns(net, W):
    return [net.forward(np.asarray(w[:T], np.int64))["patterns"][1].astype(np.float32) for w in W]


def loss_fn(theta, X, A, mask):
    sc = (X * theta).sum(-1)
    sc = sc.masked_fill(~mask, -1e9)
    lsm = torch.log_softmax(sc, -1)
    return -(A * lsm).sum(-1).mean()


def fit(P, names, Wf, Af, init=None, iters=150):
    X = torch.from_numpy(np.stack([features(P, np.asarray(w[:T], np.int64), names) for w in Wf]))
    A = torch.from_numpy(np.stack(Af))
    mask = torch.ones(T, T, dtype=torch.bool).tril()[None]
    th0 = [(init or V1).get(nm, 0.0) for nm in names]
    theta = torch.tensor(th0, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-8, tolerance_change=1e-11, history_size=30)

    def closure():
        opt.zero_grad()
        loss = 0.0
        for c in range(0, len(X), 8):
            l = loss_fn(theta, X[c:c + 8], A[c:c + 8], mask) * len(X[c:c + 8]) / len(X)
            l.backward()
            loss += float(l)
        return torch.tensor(loss)

    opt.step(closure)
    return dict(zip(names, theta.detach().numpy().astype(float).tolist()))


def pattern_kl(P, names, theta, W, Am):
    th = torch.tensor([theta[n] for n in names], dtype=torch.float32)
    mask = torch.ones(T, T, dtype=torch.bool).tril()[None]
    tot = []
    for w, a in zip(W, Am):
        X = torch.from_numpy(features(P, np.asarray(w[:T], np.int64), names)[None])
        A = torch.from_numpy(a[None])
        ce = float(loss_fn(th, X, A, mask))
        ent = float(-(A * torch.log(A.clamp_min(1e-30))).sum(-1).mean())
        tot.append(ce - ent)
    return float(np.mean(tot))


def main():
    P = program()
    net = lib.net()
    Wf, We = train_windows(32, seed=2024), train_windows(16, seed=2000)
    Af, Ae = model_patterns(net, Wf), model_patterns(net, We)
    res = {}
    for vname in sys.argv[1:]:
        names = VARIANTS[vname]
        t0 = time.time()
        th = fit(P, names, Wf, Af)
        res[vname] = {"theta": {k: round(v, 4) for k, v in th.items()},
                      "pattern_kl_fit": pattern_kl(P, names, th, Wf, Af),
                      "pattern_kl_heldout": pattern_kl(P, names, th, We, Ae)}
        if vname == "base":
            res["v1 as is"] = {"pattern_kl_heldout": pattern_kl(P, names, V1, We, Ae)}
            print("v1 as is", res["v1 as is"], flush=True)
        print(vname, f"{time.time() - t0:.0f}s", json.dumps(res[vname]), flush=True)
    old = {}
    try:
        old = json.load(open(f"{lib.OUT}/fit_attn.json"))
    except Exception:
        pass
    old.update(res)
    json.dump(old, open(f"{lib.OUT}/fit_attn.json", "w"), indent=1)


if __name__ == "__main__":
    main()
