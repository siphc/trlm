"""Peek: for chosen query positions, the plausible tokens' head-3 per-source vote vs corpus statistics."""
import numpy as np
from h3lib import H3Run, T, V, tok_strs, wcorr
from analysis.fitlib import train_windows
from analysis.mech_eval import stats
s = stats(); S = tok_strs()
pmi = {k: np.asarray(s.pmi[k], np.float64) for k in range(1, 9)}
logf = np.asarray(s.log_follow, np.float64)
tk, tc = np.asarray(s.tri_keys), np.asarray(s.tri_counts)
W = train_windows(12, seed=11)[:, :T]
R = H3Run(); mc = R.mean_ctx(W)
shown = 0
for w in W[:6]:
    r = R.run(w, mc); t = r["t"]; a = r["a3"]; u = R.source_votes(r) / r["sdf"].mean()
    pw = np.exp(r["lp"])
    for i in range(1, T):
        j = int(np.argmax(a[i, :i])) if i > 0 else 0
        if a[i, j] < 0.6 or i - j != 1: continue
        src, q = t[j], t[i]
        key = src * V + q
        lo, hi = np.searchsorted(tk, key * V), np.searchsorted(tk, key * V + V)
        c = np.zeros(V); c[tk[lo:hi] - key * V] = tc[lo:hi]
        pc = (c + 20 * np.exp(logf[q])) / (c.sum() + 20)
        cond = np.log(pc) - logf[q]
        top = np.argsort(-pw[i])[:12]
        uu = u[j] - (pw[i] * u[j]).sum()
        print(repr("".join(S[x] for x in t[max(0,i-6):i+1])), "src", repr(S[src]), "a=%.2f" % a[i, j], "n(s,q)=%d" % c.sum(),
              "corr pmi2 %.2f cond %.2f" % (wcorr(u[j][None], pmi[2][src][None], pw[i][None])[0], wcorr(u[j][None], cond[None], pw[i][None])[0]))
        print("   ", " ".join(f"{S[v]!r}:p{pw[i,v]:.2f}/u{uu[v]:+.1f}/c{cond[v]:+.1f}/m{pmi[2][src,v]:+.1f}" for v in top))
        shown += 1
        if shown > 24: raise SystemExit
