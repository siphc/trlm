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

## 2026-10-12 — attn_s0, program v6 (program_sha 42b1a03981dc1a8e), KL(model || program) = 0.8420 on 50 windows

| head | model self-effect | program self-effect | fidelity loss (mean-abl.) | fidelity loss (removed) |
|---|---|---|---|---|
| 0 | 3.0701 | 0.2904 | 0.3008 | 0.2728 |
| 1 | 0.4161 | 0.1961 | 0.2057 | 0.2338 |
| 2 | 1.5200 | 0.3552 | 0.3620 | 0.3397 |
| 3 | 0.5402 | 0.2958 | 0.3038 | 0.2472 |

Same mismatch as v4: the current token's own vote lives in FOLLOW, which in the model is carried by heads 0
and 2 attending to self. Either route that vote through heads 0/2 or map FOLLOW to "heads 0+2 self-attention
+ direct path" when comparing.

## 2026-10-12 — attn_s0, program v7 (program_sha 30394393a97d7448), KL(model || program) = 0.5154 on 50 windows

v7's heads are silent on the current token, so the matched model-side ablation is the earlier-token part only
(keep a_ii v_i, mean-ablate sum_{j<i} a_ij v_j; exact numpy decomposition, means over 50 claims windows).

| head | model: whole head | model: earlier-token part | program self-effect | fidelity loss (mean-abl.) | fidelity loss (removed) |
|---|---|---|---|---|---|
| 0 | 3.0701 | 0.4692 | 0.1566 | 0.1649 | 0.1747 |
| 1 | 0.4161 | 0.3830 | 0.1615 | 0.1749 | 0.1880 |
| 2 | 1.5200 | 0.4923 | 0.2824 | 0.2889 | 0.2896 |
| 3 | 0.5402 | 0.3365 | 0.1618 | 0.1695 | 0.2033 |

Reading: with matched definitions, every program head carries one third to one half of the model head's
context effect; head 2 is the largest in both. The remaining KL is mostly missing context, worst for head 0.

## 2026-10-12 — attn_s0, program v10 (tables of commit 3f06996), KL(model || program) = 0.4655 on 50 windows

| head | model: whole head | model: earlier-token part | program self-effect | fidelity loss (mean-abl.) | fidelity loss (removed) |
|---|---|---|---|---|---|
| 0 | 3.0701 | 0.4692 | 0.1460 | 0.1466 | 0.1518 |
| 1 | 0.4161 | 0.3830 | 0.1606 | 0.1655 | 0.1741 |
| 2 | 1.5200 | 0.4923 | 0.2660 | 0.2705 | 0.2708 |
| 3 | 0.5402 | 0.3365 | 0.1556 | 0.1605 | 0.1797 |

Same pattern as v7: head 2 largest on both sides; program heads carry 31% (head 0), 42% (head 1), 54% (head 2) and
46% (head 3) of the model heads' earlier-token effect.
