"""Where do the model's heads attend just after an end-of-story token? (measurement only)"""
import sys
import numpy as np
sys.path.insert(0, "/Users/svi/Desktop/trlm/analysis/scratch/spec_v1")
import v1probe as vp
from v1probe import T, train_windows
wins = train_windows(160, seed=2025)[:80]
x = wins[:, :T]
rows = {k: [] for k in ["eos itself", "1 after eos", "2-4 after eos", "5-16 after eos", "17-64 after eos"]}
for b in range(len(x)):
    xs = x[b]
    eos = np.flatnonzero(xs == 0)
    if len(eos) == 0:
        continue
    A = vp.model_patterns(wins[b:b + 1])[0].numpy()          # (H, T, T)
    for e in eos:
        for i in range(e, min(T, e + 65)):
            d = i - e
            k = ("eos itself" if d == 0 else "1 after eos" if d == 1 else "2-4 after eos" if d <= 4
                 else "5-16 after eos" if d <= 16 else "17-64 after eos")
            later_eos = eos[(eos > e) & (eos <= i)]
            if len(later_eos):
                continue
            # mass on: current token, the EOS token, tokens of the previous story (j < e), the current story (e < j < i)
            m = np.stack([A[:, i, i], A[:, i, e] if d > 0 else np.zeros(4), A[:, i, :e].sum(-1),
                          A[:, i, e + 1:i].sum(-1) if d > 1 else np.zeros(4)], 1)
            rows[k].append(m)
for k, v in rows.items():
    if not v:
        continue
    m = np.mean(v, 0)
    print(f"{k:16s} n={len(v):5d}  " + "  ".join(f"h{h}: self {m[h,0]:.2f} eos {m[h,1]:.2f} prev-story {m[h,2]:.2f} this-story {m[h,3]:.2f}" for h in range(4)))
