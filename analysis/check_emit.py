"""Check that program/ reproduces a student checkpoint up to rounding.
Usage: python -m analysis.check_emit analysis/out/<model>/student_<tag>.pt"""
import sys
import time

import numpy as np
import torch

from analysis.student import Student

t0 = time.time()
import program  # noqa: E402
from program.tables_lib import TOKEN_CLASS  # noqa: E402

print(f"program import {time.time() - t0:.1f}s")
sd = torch.load(sys.argv[1])
st = Student(sd["salience"].shape[0], TOKEN_CLASS, int(TOKEN_CLASS.max()) + 1)
st.load_state_dict(sd)
w = np.load("data/eval/claims.npy")[:3, :256].astype(int)
with torch.no_grad():
    s = st(torch.as_tensor(w)).double().numpy()
t0 = time.time()
diffs = []
for k in range(len(w)):
    p = program.predict(w[k].tolist())
    ps = np.exp(s[k])
    diffs.append(float((ps * (s[k] - p)).sum(-1).mean()))
print(f"predict {(time.time() - t0) / len(w):.2f}s/window; KL(student || program) per window: "
      f"{[round(d, 6) for d in diffs]}")
