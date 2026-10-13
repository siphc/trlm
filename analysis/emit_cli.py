"""Emit a saved Student checkpoint into program/tables/ and check program == student.

Usage: python -m analysis.emit_cli analysis/out/attn_s0/student_<tag>.pt "<note>" [model] [--raw]
Default: the readable (phase-6) format adopted 2026-10-12 (one decimal, importance order, annotations;
importance statistics on 640 TRAIN windows, seed 3000). --raw: the old two-decimal format."""
import subprocess
import sys

import torch

from analysis.emit_program import emit
from analysis.fitlib import train_windows
from analysis.student import load_student
from program.tables_lib import TOKEN_CLASS

args = [a for a in sys.argv[1:] if not a.startswith("--")]
raw = "--raw" in sys.argv
ck, note = args[0], args[1]
model = args[2] if len(args) > 2 else "attn_s0"
torch.set_num_threads(4)
st = load_student(ck, TOKEN_CLASS)
if raw:
    print(emit(st, model, note))
else:
    print(emit(st, model, note, decimals=1, readable=True, windows=torch.as_tensor(train_windows(640, seed=3000))))
r = subprocess.run([sys.executable, "-m", "analysis.check_emit", ck], capture_output=True, text=True)
print(r.stdout, r.stderr[-1500:])
