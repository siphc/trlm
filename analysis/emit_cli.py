"""Emit a saved Student checkpoint into program/tables/ and check program == student.
Usage: python -m analysis.emit_cli analysis/out/attn_s0/student_<tag>.pt "<note>" [model]"""
import subprocess
import sys

import torch

from analysis.emit_program import emit
from analysis.student import Student, load_student

ck, note = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "attn_s0"
st = load_student(ck)
print(emit(st, model, note))
r = subprocess.run([sys.executable, "-m", "analysis.check_emit", ck], capture_output=True, text=True)
print(r.stdout, r.stderr[-1500:])
