"""Emit a saved Student checkpoint into program/tables/ and check program == student.
Usage: python -m analysis.emit_cli analysis/out/attn_s0/student_<tag>.pt "<note>" [model]"""
import subprocess
import sys

import torch

from analysis.emit_program import emit
from analysis.student import Student
from program.tables_lib import TOKEN_CLASS

ck, note = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "attn_s0"
sd = torch.load(ck, map_location="cpu")
st = Student(sd["salience"].shape[0], TOKEN_CLASS, int(TOKEN_CLASS.max()) + 1)
st.load_state_dict(sd)
print(emit(st, model, note))
r = subprocess.run([sys.executable, "-m", "analysis.check_emit", ck], capture_output=True, text=True)
print(r.stdout, r.stderr[-1500:])
