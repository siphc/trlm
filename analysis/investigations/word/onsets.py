"""What happens at word onsets (current token starts a word that usually continues)? Examples, and how often the
model's / program's top piece was seen earlier in the window right after the same onset, or anywhere earlier."""
import numpy as np

from analysis.fitlib import train_windows
from analysis.investigations.word import wordstats as WS
from analysis.investigations.word.diag import classes
from analysis.mech_eval import EVAL_SEED, model_logprobs, program
from program.tests import measure as M
from program.vocab import TOKENS

P = program()
PIECE = WS.piece_flags()
W = train_windows(64, seed=EVAL_SEED)
lp = model_logprobs(W)
top = lambda l, k=5: "  ".join(f"{TOKENS[j]!r}:{np.exp(l[j]):.2f}" for j in np.argsort(-l)[:k])
stats = {"n": 0, "kl": 0.0, "seen_after_same": 0, "kl_seen": 0.0, "model_top_is_seen_follower": 0,
         "prog_top_is_seen_follower": 0, "model_p_seen_follower": 0.0, "prog_p_seen_follower": 0.0}
shown = 0
for w, p in zip(W, lp):
    t = np.asarray(w[:256])
    q = P.predict(t)
    cl, _ = classes(P, t, PIECE)
    kl = M.kl(p, q)
    for i in range(1, len(t) - 1):
        if cl[i] != "onset":
            continue
        stats["n"] += 1
        stats["kl"] += kl[i]
        prev = [j for j in range(i) if t[j] == t[i] and j + 1 < i]
        foll = set(int(t[j + 1]) for j in prev)
        if foll:
            stats["seen_after_same"] += 1
            stats["kl_seen"] += kl[i]
            fl = list(foll)
            stats["model_top_is_seen_follower"] += int(np.argmax(p[i])) in foll
            stats["prog_top_is_seen_follower"] += int(np.argmax(q[i])) in foll
            stats["model_p_seen_follower"] += np.exp(p[i][fl]).sum()
            stats["prog_p_seen_follower"] += np.exp(q[i][fl]).sum()
        if shown < 25 and kl[i] > 1.0:
            shown += 1
            ctx = "".join(TOKENS[x] for x in t[max(0, i - 10): i + 1]).replace("\n", "\\n")
            print(f"--- ...{ctx!r} next={TOKENS[t[i + 1]]!r} KL {kl[i]:.2f} seen-before followers {[TOKENS[x] for x in foll]}")
            print("   model  ", top(p[i]))
            print("   program", top(q[i]))
n, m = stats["n"], stats["seen_after_same"]
print(f"onsets {n}, mean KL {stats['kl'] / n:.3f}; onset seen earlier in window (followed by something) {m} ({m / n:.2f}), "
      f"their KL {stats['kl_seen'] / max(m, 1):.3f}")
print(f"  where seen: model top = an earlier follower {stats['model_top_is_seen_follower'] / m:.2f}, program top {stats['prog_top_is_seen_follower'] / m:.2f}; "
      f"mass on earlier followers model {stats['model_p_seen_follower'] / m:.2f}, program {stats['prog_p_seen_follower'] / m:.2f}")
