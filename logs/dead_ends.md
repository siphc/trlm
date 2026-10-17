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

## 2026-10-13 attn_s0 — mechanism program v1 ("explain, don't fit"; corpus statistics + named scalars)

Numbers: 160 held-out TRAIN windows, paired against the reference variant V1F (KL 0.5804; analysis/scratch/spec_v1/
v1ablate.py and v1ablate_results.json; the spec_v1 scripts, logs and results are force-added to git). Adoption bar: gain >= 0.005 and >= 3 SE (>= 0.01 for a
new component).

- **Head-3 reader** (skip-gram votes, distance-matched like heads 0 and 2, attention rule fitted to head 3's own
  pattern, pattern KL 0.86): removing it changes KL by -0.0005 +- 0.0003 (it slightly hurts). Head 3's votes are not
  skip-grams of what it reads; head 3 stays unexplained (model cost: mean ablation 0.517, earlier part 0.320).
- **Echo** (a separate vote for the current token): removal +0.0002 +- 0.0000. RECALL already repeats the current token
  at window position 0, the only place the model does (top-1 = current token: 51% at position 0, 1.6% elsewhere).
- **In-context successor term in RECALL attention** (prefer earlier tokens that followed an earlier copy of the
  current token, an induction-like rule): +0.0046 +- 0.0003 when removed, below the bar; dropped. The class successor
  term stays (both removed: +0.0194 +- 0.0013; head 1 pattern KL 0.589 -> 0.649).
- **RECALL votes tied across word groups** (3 instead of 15 vote scalars): +0.0083 +- 0.0008; **topic votes only**
  (no copy, no next): +0.0097 +- 0.0004. Both rejected; per-group votes kept.
- **End-of-story sink for RECALL** (extra salience for token 0): -0.0013 +- 0.0005, below the bar; not adopted, although
  it fits head 1's attention much better (pattern KL 0.594 -> 0.543) and the first token after a story boundary
  (1.355 -> 0.985). Candidate if a later fidelity gate is relaxed for pattern-faithful terms.
- **Attention fitted freely to the output** instead of to each head's own pattern: about 0.03 better (dev_fast 0.549)
  but repurposes the heads (head 3's decay turns negative, pattern KL 0.86 -> 4.93; head 1 0.59 -> 0.92). Rejected as
  unfaithful. The penalised joint fit (KL + lam * pattern cross-entropy) gains 0.025 / 0.022 / 0.015 at lam 0.01 / 0.03 /
  0.1, with head 1 pattern KL 0.784 / 0.692 / 0.622 (frozen: 0.589). Only lam >= 0.1 keeps patterns within +0.03.
- **Cancellation as the reason the program's head parts are weak** (consistency check: earlier-token removal moves
  the program 0.04-0.08 nats, the model 0.41-0.49): adding a "copy what the head reads" direction to each head's
  votes raises the pooled weighted R^2 of the model head's earlier-token effect only for head 0 (0.035 -> 0.060,
  coefficient -9.7) and not at all for heads 1 (0.441) and 2 (0.104) (analysis/cancellation.py, 32 TRAIN windows).
  The heads are not large opposing copy votes that net out; most of what heads 0 and 2 do from earlier tokens is
  simply not skip-gram voting.

## 2026-10-13 attn_s0 — discovery round 2 (workflow mech-v2-investigations; details in logs/mech_v2_investigations.md)

Paired KL differences from analysis/vote_fit.py (all strengths refitted; 32 fit / 64 eval TRAIN windows unless noted).
Adoption bar: gain >= 0.005 and >= 3 SE.

- **Conditional skip-gram** (what an earlier token adds beyond the current token, from trigram counts) as votes for
  heads 0/2: -0.0004 +- 0.0002; it matches the heads' value/output rows worse (0.30-0.32) than the plain bigram does.
- **Word-level skip-grams alone** -0.0008 +- 0.0004; **earlier tokens' bigram alone** -0.0038 +- 0.0004; **slot veto +
  copy alone** -0.0034. Only the combination (ov02_earlier_votes4) passes.
- **Low-rank skip-gram votes** (SVD rank 8-128 of PMI): explain the heads' votes no better than the full tables, usually
  worse (head 2 PMI_2 R^2: full 0.212, rank 32 0.063-0.084).
- **Whole-word continuation votes** (log P(next | word so far) - log P1): -0.00007 +- 0.00002, strength about 0; forcing a
  positive strength hurts. The model does not look up whole words. **Word-end votes** -0.0011 +- 0.0002 (negative
  strength). **cont-gated successor recall** -0.0058 +- 0.0016 but lowers head 1's R^2 (0.426 -> 0.416);
  **onset-gated** -0.0050 +- 0.0015 (seed 2000), -0.0040 +- 0.0008 (seed 3001): near-miss, not adopted.
- **Head-3 votes**: substitutes -0.00012, substitutes + PMI -0.00021, linker-specific -0.00014 (R^2 0.039 -> 0.059),
  word-onset completion -0.0015 +- 0.0003. Next-token-class votes: per-source R^2 0.034-0.036. PHRASE already covers
  head 3's phrase knowledge.
- **RECALL key-salience flags** (EOS, topicality, capitals, quotes) fitted to head 1's pattern: pattern KL 0.606 ->
  0.477 but output +0.0146 +- 0.0012 (worse). **Recency shape** variants: no change. **Referent-family votes**
  -0.00045 +- 0.00039. **Class-level vote strengths** (74-197 directions): R^2 0.594-0.630 vs 0.582 under exact
  attention, too little per scalar.
- **RECALL token-level successor attention + band 25..64 votes**: -0.0069 +- 0.0006 for the proposer, refuted on fresh
  windows (-0.0040 +- 0.0007). Successor term alone -0.0046 +- 0.0004; band votes alone -0.0026 +- 0.0004.
- **Pair attention from conditional MI / trigram information gain**: correlation with the exact pair interaction
  0.06-0.07; pattern KL gains under 0.015. **Corpus vote-usefulness attention** for head 2: -0.0045 +- 0.0003 (64
  windows), -0.0038 +- 0.0003 (128 windows); head 0 variants +0.0003 to +0.0028 (worse). **Key-shape salience for head
  0**: better pattern, worse output (+0.0028 +- 0.0003).
- **The model's exact attention for heads 0/2 with skip-gram votes** (ceiling): only -0.0063 +- 0.0013. The votes, not
  the attention, are the bottleneck.

## 2026-10-17 deep study: "where a head looks comes from what it writes" beyond the output layer (logs/deep_look_write_results.md)

- **The law for heads whose writes are read by later layers.** Pre-registered on 2- and 4-block attention-only models
  (seeds 1-4): layer-0 heads (F1, 11 heads) REFUTED (D1 5/11; median own gain 0.007, 15% of the oracle; controls 45% of
  the own gain); middle-layer heads of 4-block models (F3, 22 heads) REFUTED (D1 8/22). The last layer (F2) is MIXED
  (9/16 meet D1-D3; D1 15/16). The law is a property of heads that write to the output.
- **Propagating usefulness through later layers (LTOTAL vs DV, D4).** Fails: for early and middle heads the usefulness
  of the write as read by later blocks predicts attention no better than its direct effect, mostly slightly worse.
- **The exact-gradient readout (TOTAL).** On the one-layer calibration it scores half as well as the readout with the
  final LayerNorm linearised (median gain 0.049 vs 0.104).
- **A universal coupling constant (D5).** Fails: about 0.25 in one-layer models, 0.65-0.8 in the last layer of deeper
  models, 0.07-0.33 elsewhere.
- **Vote codes as the best low-rank compression of corpus statistics (code test K1b, K3, K4).** Fail as expected: the
  model's embedding subspace holds 0.57 of a head's vote energy, the corpus top-32 subspace 0.20; PMI beats the
  bigram-residual statistic; local heads do not beat far-looking heads.
- **Recursive generalisation of a head's key set (growth G3).** Not robust: the effect disappears when the current
  attention level is matched (2 of 8 heads).

## 2026-10-17 goal 2, program v1.2 (analysis/goal2/, logs/goal2_plan.md)

Paired KL differences from analysis/vote_fit.py (all strengths refitted; 32 fit / 64 eval TRAIN windows). Bar: >= 0.005
and >= 3 SE.

- **Softmax bottleneck** (M-projection of the program's output onto a 128-dimensional span): with the model's own
  embedding span (oracle) 0.519 -> 0.477 on 8 windows; spans from corpus statistics (PCA of the bigram, PMI, skip-grams)
  make it worse (0.54-1.28).
- **One normalisation of the whole score** (rescale to a constant size, or to the model's own logit size as an oracle):
  worse (oracle 0.87-1.25). The program's scores are too wide over the whole vocabulary and too narrow among frequent
  tokens; no single scale fixes both.
- **FOLLOW variants:** piecewise-linear response to log P1 -0.0045 +- 0.0003 (near miss; on top of RECALL saturation
  only -0.0034); strengths by current-token shape -0.0018; Kneser-Ney continuation preference -0.0005; FOLLOW as an
  interpolated Kneser-Ney bigram +0.053 (worse); similarity-based smoothing (Dagan, Lee & Pereira) -0.0005.
- **Calibrating to corpus moments:** the model's mean prediction after each token is further from the corpus bigram
  (0.102) than the program's (0.078), so making the program match corpus moments would move it away from the model.
- **The model's token code from the corpus:** spectral spans of the bigram, PMI and the symmetrised co-occurrence
  matrix (what small-initialisation training theory predicts) share only the leading directions with the embedding
  (mean squared canonical correlation 0.14-0.41 vs random 0.07-0.23).
- **Frequency-dependent trust in the bigram** (the model shrinking rare tokens' bigram rows): the bigram slope is the
  same in every frequency bin of the candidate (0.69-0.70); R^2 unchanged.
- **Current token supplies what the context does not** (current-token table = bigram - b x mean context vote): holds
  with the model's own context (R^2 0.65 -> 0.73, b = -0.44) but not with the program's context votes (0.656).
- **Saturation of LOCAL-0, LOCAL-2, PHRASE** (as adopted for RECALL): -0.0020, -0.0023, 0.0000.
