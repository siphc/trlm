"""D4: where does RECALL's attention rule differ from head 1's real pattern? (16 TRAIN windows, seed 2000)
Pattern KL(model || rule) per row, split by query group / query position; attention mass by key group and distance;
row sharpness (entropy) of model vs rule; tokens most under- and over-attended."""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/investigations/recall")
import lib  # noqa: E402
from analysis.fitlib import train_windows  # noqa: E402
from analysis.mech_eval import program  # noqa: E402
from program import head1  # noqa: E402
from program.vocab import TOKENS  # noqa: E402


def main():
    P = program()
    net = lib.net()
    W = train_windows(16, seed=2000)
    G = head1.GROUP_NAMES + ["EOS"]
    rows = defaultdict(list)
    mass_m, mass_r = defaultdict(float), defaultdict(float)
    dist_m, dist_r = defaultdict(float), defaultdict(float)
    tok_m, tok_r, tok_n = defaultdict(float), defaultdict(float), defaultdict(int)
    ent_m, ent_r = [], []
    qtok_kl = defaultdict(list)
    for w in W:
        t = np.asarray(w[:256], np.int64)
        A = net.forward(t)["patterns"][1]
        R = head1.attention(t, P.STATS)
        kl = (A * (np.log(A + 1e-30) - np.log(R + 1e-30))).sum(1)
        g = head1.GROUP[t].copy()
        g[t == 0] = 5
        cls = lib.position_classes(P, t)
        for i in range(len(t)):
            rows["qgroup " + G[g[i]]].append(kl[i])
            rows["qclass " + cls[i]].append(kl[i])
            rows[f"qpos {min(i // 32, 7) * 32}-"].append(kl[i])
            qtok_kl[TOKENS[t[i]]].append(kl[i])
        ent_m.append(-(A * np.log(A + 1e-30)).sum(1))
        ent_r.append(-(R * np.log(R + 1e-30)).sum(1))
        n = len(t)
        d = np.arange(n)[:, None] - np.arange(n)[None, :]
        for k in range(6):
            mass_m[G[k]] += A[:, g == k].sum()
            mass_r[G[k]] += R[:, g == k].sum()
        for lo, hi in [(0, 0), (1, 1), (2, 7), (8, 31), (32, 63), (64, 127), (128, 255)]:
            m = (d >= lo) & (d <= hi)
            dist_m[f"{lo}-{hi}"] += A[m].sum()
            dist_r[f"{lo}-{hi}"] += R[m].sum()
        am, ar = A.sum(0), R.sum(0)
        for j in range(n):
            tok_m[t[j]] += am[j]
            tok_r[t[j]] += ar[j]
            tok_n[t[j]] += 1
    tot = 16 * 256
    out = {"pattern_kl": float(np.mean(np.concatenate([np.asarray(v) for k, v in rows.items() if k.startswith("qpos")]))),
           "kl_by": {k: {"mean": round(float(np.mean(v)), 3), "n": len(v), "share": round(float(np.sum(v)) / 1, 1)}
                     for k, v in sorted(rows.items())},
           "mass_by_key_group": {k: [round(mass_m[k] / tot, 4), round(mass_r[k] / tot, 4)] for k in G},
           "mass_by_distance": {k: [round(dist_m[k] / tot, 4), round(dist_r[k] / tot, 4)] for k in dist_m},
           "entropy_model_rule": [float(np.mean(ent_m)), float(np.mean(ent_r))]}
    diff = {k: (tok_m[k] - tok_r[k]) / tot for k in tok_m}
    o = sorted(diff, key=lambda k: diff[k])
    out["over_attended_by_rule"] = [(TOKENS[k], round(diff[k], 4), tok_n[k]) for k in o[:25]]
    out["under_attended_by_rule"] = [(TOKENS[k], round(diff[k], 4), tok_n[k]) for k in o[::-1][:25]]
    qk = {k: (float(np.mean(v)), len(v), float(np.sum(v))) for k, v in qtok_kl.items()}
    out["worst_query_tokens_by_total_kl"] = sorted(((k, round(a, 2), n_, round(s, 1)) for k, (a, n_, s) in qk.items()),
                                                   key=lambda x: -x[3])[:30]
    print(json.dumps(out, indent=0, ensure_ascii=False))
    json.dump(out, open(f"{lib.OUT}/d4_pattern.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
