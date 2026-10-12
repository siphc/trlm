# Dead-end ledger

Every hypothesis that failed, with the evidence. Negative results are data.
Format: date, model, hypothesis, what was tried, numbers, why it failed.

## 2026-10-12 attn_s0 — prune exceptions by |value| * sqrt(token frequency)

- Hypothesis: the largest residuals between the model's mean bigram and the class-level default are the
  exceptions worth keeping.
- Tried: bigram-only program, exceptions initialised to log(mean bigram) minus class-level part, keep top N
  by |value| * sqrt(freq(t)), refit (analysis/run_fit.py, tags v3_after20000 / v3_after60000 first run).
- Numbers (KL on held-out train windows): dense 1.116; kept 20k -> 2.222, kept 60k -> 2.084 (class-level
  alone is 2.434).
- Why it failed: the largest |residuals| are huge negative values on tokens the model already gives ~0
  probability; dropping them costs nothing, while the dropped small positive residuals carry the mass.
  Replaced by a KL-cost ranking (freq(t) * |w| * max(q_with, q_without)): 20k -> 1.356, 60k -> 1.250.

## 2026-10-12 attn_s0 — one global exception budget by data importance (v5)

- Hypothesis: fitting all exceptions densely (FOLLOW and every head's RECALL), then keeping the 40k with the
  largest data-measured first-order cost, beats spending the budget on FOLLOW only.
- Numbers: dense student 0.247 (held-out train KL); after keeping 40k: 1.715; after refit: 1.176
  (dev_fast harness: 1.1898). v4 (20k FOLLOW exceptions, class-level heads, no RECALL exceptions): 1.160
  (dev_fast 1.1696). Commit: "v5" row in logs/iterations.csv.
- Why it failed: the dense solution spreads its information over millions of small entries; the
  first-order importance of any one entry is a poor guide when most of them are removed at once
  (removing 99.8% is far outside the linear regime). The kept set was 2.6k FOLLOW + 37k RECALL exceptions.
  Next: sparsity has to be built in during fitting (gradual pruning / group structure), not cut afterwards.

## 2026-10-12 attn_s0 — final-LayerNorm scale from attention concentration

- Hypothesis: sigma^2 of the final LayerNorm ~ c + sum_h w_h * focus_h, with focus_h = sum over distinct
  tokens of (head h's attention mass on that token)^2 (value vectors of different tokens ~orthogonal).
- Tried: least squares on 24 dev_fast windows (6,144 positions), open-track decomposition.
- Numbers: R^2 = 0.179; median relative error in sigma 12.9%; coefficients c 0.585, w = 0.779, 0.178, 0.601,
  0.608 (heads 0-3).
- Why it failed (so far): value vectors of different tokens are not near-orthogonal, and cross-head terms
  and the token/position embeddings contribute. A confidence term may still help but needs token-dependent
  inputs (e.g. per-token value "loudness"), not just attention shape.
