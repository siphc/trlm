# Component consistency check (AGENTS.md section 9)

Tool: `python -m analysis.consistency --model <id> --n 50` (dev_fast windows; MEAN ablation on both sides:
model head output before W_O replaced by its mean over the 200 `claims` windows; program head votes replaced
by their mean vote vector over the same windows). Columns: model self-effect KL(model || model with head h
mean-ablated); program self-effect KL(program || program with head h mean-ablated); fidelity loss = increase
of KL(model || program) when the program's head h is mean-ablated / removed.

## 2026-10-12 — attn_s0, program v4 (program_sha c26aa523b7142d15), KL(model || program) = 1.1658 on 50 windows

| head | model self-effect | program self-effect | fidelity loss (mean-abl.) | fidelity loss (removed) |
|---|---|---|---|---|
| 0 | 3.0701 | 0.3216 | 0.3285 | 0.3111 |
| 1 | 0.4161 | 0.1965 | 0.1987 | 0.1975 |
| 2 | 1.5200 | 0.6853 | 0.6883 | 0.6627 |
| 3 | 0.5402 | 0.3250 | 0.3251 | 0.2955 |

Reading: the model's head 0 (and to a lesser degree head 2) is essential because it carries the current
token's identity (it attends ~49% to self); in v4 that information lives in FOLLOW instead, so program head 0
is ~10x less important than model head 0. The program's head 0 does not mean what the model's head 0 means.
