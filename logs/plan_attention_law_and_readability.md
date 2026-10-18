# Plan: three studies on attention and readability (approved 2026-10-18; logs/decisions.md item 10)

Written for a coding agent that continues this work without the conversation that produced it. Read the whole file
before acting. Ask the human when a decision is not covered here; do not guess silently (AGENTS.md section 13).

| study | short name | question | status |
|---|---|---|---|
| A | `fixed_votes` | How does attention form when what a head writes is fixed and known? Does the look-follows-write law predict it, causally? | not started |
| B | `law_pressure` | Can the look-follows-write law be Goodharted: what happens when training penalises it? | not started |
| C | `canonical` | Does pulling models toward one readable program remove the arbitrariness between training runs? | part A ready to run (gate G2); part B not started |

Suggested order: C part A runs first (it is ready and only needs GPU time), A and B are coded while it runs, then A,
then B, then C part B. One Apple-silicon GPU (MPS): training runs one at a time; CPU analysis runs alongside.

## 1. Where things stand (read these first)

- `README.md` (repository map), `AGENTS.md` (the brief), `logs/decisions.md` (human decisions 1-10; 9 and 10 matter
  most here), `logs/README.md` (index of reports).
- **Goal reframed (decision 9).** A readable replacement with KL near zero is not reachable for a model trained without
  regard to readability: another seed of the same network is 0.38-0.42 nats from attn_s0 (logs/mech_v3_report.md),
  and the remaining gap sits in per-head tables locked to the model's own attention (logs/token_geometry_results.md).
  The readable program v1.3 (`program/`, dev_fast KL 0.5497 to attn_s0) is the target the side studies use.
- **The look-follows-write law** (study complete and independently verified; `writeups/look_follows_write_predraft.md`,
  `logs/look_follows_write_results.md`, `logs/look_follows_write_addendum_results.md`, `logs/deep_look_write_results.md`):
  in one-layer attention-only models, the pair-specific part of a local head's attention follows the usefulness of the
  head's OWN vote for the next token beyond a corpus bigram (pure-pair correlation 0.34-0.44, other heads' votes <= 0.08,
  coupling about 0.25 attention-score units per logit, about a third of the exact token-token attention term). Attention
  and vote co-adapt during training (no clear order; usefulness predicts which pairs become strongly attended). In
  deeper models it holds for heads that write to the output (last layer) and fails for heads read by later layers.
- **The readable-training side study** (`logs/readable_training.md` running log, `logs/prereg_readable_training.md`
  FINAL pre-registration committed f323a8e, `configs/side_readable.yaml`, code `src/train/readable.py`,
  `analysis/readable_training/`): new 1-block models trained with an extra loss toward program v1.3, arm O (output
  level) and arm I (internal level: attention toward the program's rules, votes toward corpus tables), lambda 0.3 each.
  Development result on seed 0: each arm changes the level it is trained on (also out of distribution) and barely
  moves the other; strong output pressure moves attention away from the rules. Decoupling of this kind is already
  reported elsewhere (a replication of Gao et al. 2025, "weight-sparse circuits may be interpretable yet unfaithful";
  work on training against probes), which is why the studies below were chosen.
- **Infrastructure to reuse** (all tested):
  - training hooks: `src/train/train.py` (TrainConfig fields `readable_*`, `lambda_*`; default off and bitwise
    unchanged), `src/train/readable.py` (Pool, PoolSampler, losses, differentiable vote tables `ov_tables_torch`,
    `ReadableAux`), `src/train/runs.py` (`readable_dev`, `readable`), tests `src/train/test_sanity_readable.py`;
  - target pool `data/readable/pool/` (32,768 + 512 TRAIN windows, program v1.3 top-k outputs and rule attention, 13 GB,
    `analysis/readable_training/build_pool.py`), vote targets `data/readable/vote_targets.npz`;
  - measurement suite `analysis/readable_training/suite.py` (role assignment and head-aligned checkpoints, fidelity on
    dev_fast and out-of-distribution sets, attention-pattern KL, vote correlations, ablation shifts, the "lock",
    harness with claim tests, refitted program), evaluation sets `analysis/readable_training/evalsets.py`
    (`data/readable/cache/`), seed distances `analysis/readable_training/seed_distances.py`;
  - law measurements: one layer `analysis/look_write/` (uov_tables.py, fit_arms.py), any depth
    `analysis/look_write_deep/` (pass_bgu.py, fit_arms_deep.py, classify.py, echo_traj.py, growth.py). They take model
    ids; `common_deep.model_path` also accepts paths, but output folders are named after the id
    (`common_deep.model_dir`), so add an id form for new models (for example `readable/O_l0.3_s1` ->
    `models/readable/O_l0.3_s1/final.pt`, folder `readable_O_l0.3_s1`) before using them on new models;
  - factorial and head swaps: `analysis/token_geometry/tables_vs_attention.py --model <checkpoint>`.

## 2. Rules that apply to every study

- **Pre-registration.** For each study: a DRAFT pre-registration (`logs/prereg_<short name>.md`) committed before any
  output it uses; decision rules for development choices fixed in the DRAFT; seed 0 is development only; a FINAL
  version with the rule values committed before any confirmatory run (seeds 1-4); the verdict script refuses to run
  before the FINAL commit (pattern: `scripts/deep_study/confirmatory_fits.sh`, `scripts/readable/confirm_queue.sh`).
  Report every pre-registered test, including failures; label exploratory analyses as such.
- **Human gates.** G0 (design and DRAFT), G1 (training code before any long run; AGENTS.md section 4), G2 (development
  results and chosen settings before confirmatory runs), G3 (results). Stop and report at each gate.
- **Twins.** Seed s uses the initialisation and data order of attn_s{s} (`BatchSampler`), so every new model has an
  untreated twin among attn_s0..s4 and comparisons are paired.
- **Records.** A running log per study (`logs/<short name>.md`), a results file, entries in `logs/dead_ends.md` and
  `logs/surprises.md`, a line in `logs/decisions.md` for every human decision, a row in the README status.
- **Machine.** Laptop: training on MPS only with the charger connected and the lid open; launch queues with
  `caffeinate -dimsu` and a battery watchdog (`scripts/readable/watchdog.sh <queue pattern>`); CPU analysis at
  `nice -n 20` with 2 threads; check `df -h` before writing large files (about 34 GB free on 2026-10-18); the harness
  sandbox cannot read `analysis/`, so program copies the harness must import go under `programs/` (gitignored).
- **Never** edit `harness/` or `configs/frozen.yaml`, never use the test split, never commit `logs/.blackbox_key`.

## 3. Study C: readability as a way to remove arbitrariness between training runs

**Question.** Five seeds of the same network reach the same loss, yet each is 0.38-0.42 nats from attn_s0
(underspecification, the "Rashomon set"). Does pulling models toward one readable program make different training runs
agree with each other, in behaviour and in mechanism, and how much of the variation between runs is free (not needed
for prediction)? Why it matters: if a small loss cost selects one canonical member of the set, explanations become
reproducible across training runs; and "the program describes this model as well as one training run describes
another" is a natural yardstick for fidelity. Seed 0 (development): arm O at lambda 0.3 reached KL 0.396 to the program
at +0.8% cross-entropy.

**Part A (ready; pre-registered in logs/prereg_readable_training.md).** Needs the human's G2 approval.
1. `nohup caffeinate -dimsu scripts/readable/confirm_queue.sh > logs/runs/readable_confirm.out 2>&1 &` and
   `nohup scripts/readable/watchdog.sh scripts/readable/confirm_queue.sh > /dev/null 2>&1 &`: arms O and I at lambda 0.3,
   seeds 1-4, one epoch each (about 45 and 65 min per model), with the suite (`--lock --harness --refit`) on each model
   in the background as it finishes. Resumable: re-run the same commands after an interruption.
2. Seed distances: `.venv/bin/python -m analysis.readable_training.seed_distances O_s1=models/readable/O_l0.3_s1/final.pt
   ... I_s4=models/readable/I_l0.3_s4/final.pt` (all eight new models).
3. Verdicts: write `analysis/readable_training/verdicts.py` that computes, from the suite.json files and rules.json, the
   closures, transfer coefficients, out-of-distribution transfer, Goodhart flags and per-arm verdicts exactly as defined
   in the FINAL pre-registration, plus the separately reported quantities (cost, seed convergence, attn_s0 specificity,
   lock). Helper: `analysis/readable_training/apply_rules.tool_values`. It must refuse to run before the FINAL commit.
4. Results: `logs/readable_training_results.md`; gate G3.

**Part B (new; write an addendum pre-registration `logs/prereg_canonical_addendum.md` before any part-B output).**
1. A pressure curve: arm O at lambda 1 for seeds 1-4 (seed 0's one-epoch cost at lambda 1 is unknown; at 100M tokens it
   cost +3.1%).
2. **A non-readable canonicaliser at matched cost** (the key control). Pulling every seed toward any common fixed
   target shrinks their mutual distance; the question is whether a readable target removes more arbitrariness per unit
   of cost than an unreadable one. Target: attn_s0's own predictions (a single trained model, not readable). Build a
   second pool with the same windows and attn_s0's top-k log-probs (extend `build_pool.py` with a target source
   `model:<id>`; model forwards are fast). Choose the control's lambda on seed 0 at 100M tokens so its cross-entropy cost
   matches arm O's at 0.3; train seeds 1-4.
3. Measurements: (a) behavioural convergence: mean pairwise KL among the four treated seeds vs among attn_s1..s4;
   (b) the pre-registered convergence ratio (relative drop of the pairwise distance / relative drop of the distance to
   the program; near 1 = all seeds move to one point, near 0 = each approaches from its own direction); (c) mechanistic
   convergence: do head roles and per-head value codes become alike across seeds (subspace overlaps with
   `analysis/token_geometry/named_features.py` `basis` and the overlap code in `named_features_more.py`; baseline:
   best-matching local head across seeds about 0.22, uniform weighting); (d) cost per unit of between-seed distance
   removed, readable target vs attn_s0 target; (e) how well program v1.3 describes the control models (does an
   unreadable canonicaliser also make models more describable?).
4. Possible readings: strong convergence at low cost (arbitrariness is largely free and a human description removes
   it); behaviour converges but mechanisms do not (decoupling measured across runs: different runs reach the program's
   behaviour through different internals); no convergence (the program constrains only what seeds already share).

Compute: part A about 8 h MPS plus about 10 CPU-hours; part B about 8 h MPS.

## 4. Study A: how attention forms when what a head writes is fixed and known

**Question.** In normal models a head's attention and its vote form together, so the look-follows-write law could only
be shown as mutual consistency, not as cause. If a head's vote table is fixed to a known corpus table and only its
attention is learned, the direction is forced. (1) What attention does training produce? (2) Does the law (attention
follows the usefulness of the head's own, now known, vote) predict it, and how much of it? (3) What does fixing the votes
cost in prediction? The human expects that this will not yield a readable language model; the study is about how
attention works, which is little understood in the field. A readable model is a possible by-product, not the goal.

**Model variant** (new module, for example `src/train/fixed_votes.py`; TinyLM is unchanged):
- logits_i = beta_F * F(t_i, .) + sum_h beta_h * sum_{j <= i} a^h_ij * T_h(t_j, .) (+ a learned unigram bias if needed).
  F: the corpus bigram log-probability table (program/corpus.py `log_follow`); T_h: one fixed V x V table per head,
  distance-independent like a real head's vote; beta: learned scalars (one per head and one for F).
- Attention a^h: standard causal softmax of q.k over LN(token embedding + position embedding) with learned W_Q, W_K per
  head; the embeddings are used only for attention (the only learned per-token code).
- Head tables (each nameable): head 0 PMI_1 (what usually follows the token), head 1 copy (one-hot: repeat it), head 2
  the topic table (words 9-24 tokens later), head 3 PMI_2 (what comes two tokens later). Rows standardised or clipped
  so their scales are comparable; record the choice in the DRAFT.
- Memory: the vote term is (B, T, T) @ (B, T, V) per head; at B = 64, T = 256, V = 2000 one (B, T, V) float tensor is
  131 MB. Use chunking or a smaller batch with gradient accumulation if MPS memory is short; tables as fp16 buffers.
- Same data, schedule and budget conventions as the main track (frozen v3); budget to be set in the DRAFT (100-500M
  tokens, about 1 h per model expected; measure throughput first).
- Sanity tests (`src/train/test_sanity_fixed_votes.py`): causality; tables are buffers that receive no gradient; with
  attention forced to the identity the model equals F plus the self votes; resume reproduces bitwise on CPU; a tiny
  overfit test.

**Measurements** (seed 0 development, seeds 1-4 confirmatory):
1. Cost: dev cross-entropy against attn_s0..s4 (2.49-2.54 dev_full) and against program v1.3 (dev cross-entropy about
   2.51; the program uses the same kinds of tables with hand-made attention rules, so the fixed-vote model should do at
   least as well).
2. **The law, now model-free.** For a fixed table, a head's own-vote usefulness is a corpus statistic:
   U_h(k, q, d) = E[T_h(k, next) | k sits d tokens before q] - E_{v ~ P1(.|q)} T_h(k, v), counted on the TRAIN stream
   (same shrinkage as `analysis/look_write/uov_tables.py`). Fit goal 1's generic base attention rule (distance buckets,
   shapes) plus one own-usefulness feature to each head's attention pattern, and measure, as in
   `analysis/look_write/fit_arms.py`: output gain from substituting the arm's pattern, pure-pair correlation with the
   exact token-token term, other heads' usefulness and a shuffled table as controls, and the ceiling (the model's exact
   token-token attention term). Pre-register thresholds from goal 1's tests (own usefulness >= half of the
   pre-registered ceiling's output gain, controls <= a quarter of own, pure-pair correlation >= 0.25; against the
   stricter `tt_self` ceiling of goal 1's addendum own usefulness recovered 26-46%). Done: logs/prereg_fixed_votes.md.
3. **Intervention (the causal test).** Train variants that change one head's table (swapping tables between heads
   changes nothing, since heads differ only by their table; a shuffled table would likely be switched off; the DRAFT
   uses head 3 voting PMI_3 instead of PMI_2, which moves the predicted distance from 1 to 2) and test that the attention
   of the changed head reorganises as its new table's usefulness predicts, and the others do not.
4. Coupling constant: compare the fitted coefficient with one-layer models (about 0.25) and the last layer of deeper
   models (0.65-0.8).
5. Growth over training (checkpoints are saved log-spaced): does attention converge toward the law's prediction, and
   in what order (`analysis/look_write_deep/growth.py` and `echo_traj.py` adapted)?
6. By-product: a readable description of the trained model (fixed tables plus attention rules plus the law) and its KL
   to the model; with the program's harness.

Possible readings: the law predicts most of the learned attention (a causal confirmation, and a nearly readable model);
it predicts a part as in normal models (the rest of attention has other drivers, now isolable because the votes are
known exactly); it fails (the law in normal models reflects co-adaptation of learned votes, not a cause).

## 5. Study B: can the look-follows-write law be Goodharted?

**Question.** Goodhart results so far concern probes, which models evade cheaply. The look-follows-write law follows
from first-order training dynamics (a key's attention score grows where the head's value write points along the
prediction error). If training penalises the law, does the model (1) pay in prediction quality, (2) break the law for
free, or (3) game the penalty while the law survives in measurements it was not trained on? A law that is costly to
break would be an interpretability handle that resists training pressure (Hubinger's "robust-to-training" level, for a
law instead of a metric). No prior work training against a coupling of attention and value writes was found (search on
2026-10-18; not exhaustive).

**Penalty** (new arm in `src/train/readable.py` style, for example `readable_arm = "L"`), on the auxiliary windows:
- for every head h and pair (query i, key j) with 1 <= i - j <= 7: the raw attention score s_ij (q_i . k_j / sqrt(d_head),
  computed explicitly from LN1, W_Q, W_K) and the instance usefulness u_ij = OV_h(t_j, y_i) - sum_v P1(v | t_i) OV_h(t_j, v)
  (y_i the next token; OV_h from `ov_tables_torch`, differentiable; P1 the corpus bigram as a fixed device table);
- centre s and u within each query row and within each distance; penalty = sum over heads of the squared correlation of
  s and u, pooled over the batch's pairs;
- arms: L-both (gradients reach attention and votes), L-attn (u detached: only attention may move away), L-vote
  (s detached: only the votes may change so that usefulness decorrelates).

**Measurements:**
1. Cost: dev cross-entropy against the lambda grid and the untreated twin.
2. The trained metric: in-batch correlation (falls by construction).
3. **The held-out law**: goal 1's pre-registered tools on corpus-aggregated tables, which the penalty never sees: UOV
   tables (`analysis/look_write/uov_tables.py`), the arms fit (`fit_arms.py`: own-usefulness output gain, pure-pair
   correlation, other heads and shuffled controls, coefficients).
4. Where the work goes: head-specificity (does another head's usefulness now predict this head's attention), positional
   share of attention, ablation shifts per head (`analysis/readable_training/internals.ablation_shifts`).
5. **Restoring force**: from a penalised model's final checkpoint, continue training with lambda 0 for a short schedule
   (for example 3,000 steps at low learning rate) and measure how fast the held-out law returns. A fast return means
   the law is an attractor of training.
6. Seeds: development grid on seed 0 at 100M tokens (lambda in a geometric grid; choose by a pre-registered rule, for
   example the smallest lambda that brings the in-batch correlation below 20% of the control's), then seeds 1-4 for
   L-both (and the other two arms if development shows they differ).

Possible readings: costly and the law survives (a near-necessary consequence of optimisation; tools built on it resist
Goodharting); cheap and truly broken (attention can do the same job without following its own vote's usefulness; the
gradient story is not forced); in-batch metric down but aggregated law intact (instance-level measurement is gameable;
tells which way of measuring a law is robust). Main risk: instance usefulness is noisy, and a correlation penalty can
be met by inflating score noise; the aggregated law and the restoring-force test guard against that.

## 6. Compute and schedule (estimates)

| step | MPS | CPU | gate |
|---|---|---|---|
| C part A: confirmatory runs and suites | about 8 h | about 10 h, overlapping | G2 before, G3 after |
| A: code and tests, DRAFT | | | G0, G1 |
| A: development (seed 0, plus interventions) | about 3 h | law tables about 1 h | G2 |
| A: confirmatory (seeds 1-4, base and intervention variants) | about 6-8 h | about 4 h | G3 |
| B: code and tests, DRAFT | | | G0, G1 |
| B: development grid and restoring-force runs | about 3 h | law passes about 3 h | G2 |
| B: confirmatory (seeds 1-4) | about 4-8 h | about 4 h | G3 |
| C part B: addendum, control pool, runs | about 8 h | about 6 h | G2, G3 |

## 7. Literature found so far (2026-10-18; mostly abstracts; verify before citing)

- Training toward an interpretable surrogate: tree regularization (Wu et al., AAAI 2018; JAIR 2021).
- Training a network to implement a given program causally: interchange intervention training (Geiger et al., ICML
  2022); causal distillation (Wu et al., NAACL 2022).
- Interpretable architectures: Transformer Programs (Friedman et al., NeurIPS 2023), Backpack language models (Hewitt et
  al., ACL 2023), codebook features (Tamkin et al., ICML 2024), softmax linear units (Anthropic 2022), weight-sparse
  transformers (Gao et al., 2025) and a replication arguing their circuits may be unfaithful, Guide Labs' concept-based
  language models (2026), "Legible-by-Construction" attention (2026: value channels as sigmoid detectors).
- Training against interpretability tools: obfuscated activations (Bailey et al., 2024); probe-based training and
  Goodhart (2025-2026).
- Underspecification and Rashomon sets: D'Amour et al.; Rashomon-set work (2026 survey of explanation sets).
- Attention dynamics: Tian et al. 2023 (Scan and Snap), Tarzanagh et al. 2023, Bietti et al. 2023, "Faster query-key
  learning sharpens attention" (2026: the ratio of query-key to value learning rates sets attention sharpening).
- Attention as corpus statistics: a 2026 JADT paper (attention scores capture co-occurrence frequencies).
