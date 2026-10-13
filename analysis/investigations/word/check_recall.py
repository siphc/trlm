import numpy as np
from analysis.fitlib import train_windows
from analysis.mech_eval import program
from analysis.vote_fit import base_features, load_candidate
P = program()
c = load_candidate("analysis/candidates/word_recall.py")
t = np.asarray(train_windows(1, seed=7)[0][:256])
F, _ = base_features(P, t)
c.B_WORD = 0.0
import functools
c_attention = c.attention
c.attention = functools.partial(c_attention, b_word=0.0)
f = c.features(P, t)
print(max(float(np.abs(F["head1." + k] - v).max()) for k, v in f.items()))
