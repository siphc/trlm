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

## 2026-10-12 attn_s0 — design-panel and post-v9 probes (numbers on held-out TRAIN windows)

Workflow `compress-program-design` (5 proposers with CPU experiments, 3 judges, synthesis). Scripts were in
analysis/scratch/ (deleted for disk space after logging). Paired SEs on holdout(200) unless noted.

- **Add-back swap at a fixed budget** (3k entries in by behavioural residual gain, 3k out by lowest Fisher, refit):
  helped v8 (-0.0068, holdout(48)) but hurts v9 (+0.0017 +- 0.0007 after an identical 150-step refit). Re-centring
  already captured what it fixed (add side -0.0111 on v8 vs -0.0049 on v9; drop side unchanged at +0.008).
- **CONFIDENCE (per-current-token temperature)** fitted on frozen v9 tables: global +0.00006; 64 per-class
  +0.0005 +- 0.0001; 2000 per-token +0.0014 +- 0.0002 (worse). v9 is already calibrated per current token.
  Removed from program/ as a no-op; frozen by default in fitting.
- **Longer constant-lr refits**: 150 extra steps at lr 0.006 make v8 and v9 worse by 0.0008. A low-lr polish gives
  -0.0009 to -0.0017 (now a standard final stage, applied to every arm).
- **One-shot pruning pipeline** (re-centred, Fisher, 600-step refit): 60k 0.5684 vs gradual 0.4885 (paired -0.080
  +- 0.002); its 200k student 0.4043 vs the gradual run's 158k 0.3679.
- **Exact removal cost / self-KL ranking** instead of Fisher: exact-cost drop rule +0.0166 vs Fisher +0.0077 for 3k
  drops at v8's support; self-KL's 0.01-0.03 gain only without refit.
- **Proximal-Adam L1**: 272k nonzeros at 0.464 after debias vs prune+refit 0.444 at 200k.
- **Owner-deduplicated feature induction** (one table per (t, v) fact): superseded by heads-silent-on-self; at equal
  150-step refits induction lost to Fisher+NR pruning at 40k (0.9221 vs 0.8776).
- **Concepts outside head 1 / factorising dense residual tables**: strict extraction found 0-1 concepts; forcing 459
  capped concepts 1.0296 vs 1.0021 plain; factorisation gave artifact blocks with votes up to +-58 nats. Head-1 concepts
  gain ~0.009 (seed-stable) — kept analysis-only by human decision.
- **Per-query-token SELF/PREV/REC_Q attention tables**: better attention-pattern fit (0.769 -> 0.432 output KL in the
  exact model) but no KL gain with class-level votes (1.214 vs 1.170 at 20k) for +16-23k numbers.
- **Heads reusing the FOLLOW exception table**: at most 0.02.
- **FOLLOW frozen to the model's mean next-token percent lists**: +0.169 +- 0.002 at v8 (worse on 200/200 windows).
  Survives only as a display idea for the emitter.
- **Moving RECALL column effects into WORD** with heads silent on self: +0.0442 (not exact). Use the nr-safe moves.
- **Coarse value formats** on v8 without refit: SALIENCE 8 levels +0.0192 (5 levels +0.0454); sparse class rules
  |x| >= 0.5 +0.0171 (>= 1.0 +0.104); ln2 grid +0.0124; combined +0.0387. Each breaks the 2% refactor gate alone.
  One decimal: +0.0004 to +0.0013 (fine).
- **More classes / overlapping class systems instead of exceptions**: K=128 1.221 at 84k numbers, K=256 0.936 at 331k,
  two systems 1.254 at 43k, vs 64 classes + 20k exceptions 0.968 at 43k.
- **Absolute-position terms**: at most 0.044 in the exact model.
