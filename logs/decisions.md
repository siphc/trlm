# Human decisions

## 2026-10-12 (evening)

1. **Table size:** optimise and present the main line at ~60k exceptions (v9: ~106k numeric literals).
   Larger and smaller budgets stay as curve points only.
2. **Harness:** approved to extend harness/claims.py (and its helper in harness/common.py) with MEAN ablation
   and earlier-token-only ablation of a model head, for claim tests. The scorer (eval_fidelity) is unchanged.
3. **Protocol:** fidelity changes are adopted on paired held-out TRAIN comparisons at a fixed budget
   (analysis/paired.py; gain >= 0.005 and >= 3 SE, >= 0.01 for a new class system or component). Only those
   fixed-budget fidelity iterations count toward the stopping rule; curve points, readability and explore commits
   do not. dev_fast is still run and logged for every commit.
4. **Concepts (section 8):** analysis only. Head-1 concepts may be used to write claims and to group head-1
   exceptions under named comments, not as a runtime mechanism.

Earlier: frozen.yaml v2 (training on the laptop's MPS GPU), approved 2026-10-12.

## 2026-10-13

5. **Verdict on the distillation program (v11):** the model-fitted tables read like reverse-engineered weights and
   give little research insight; only the scaffolding is readable. The fitted-table approach ends here; the state is
   tagged `distillation-v11` (commit 96c60a9) as a return point.
6. **New direction, "explain, don't fit"** (agent's plan, accepted): token-level knowledge only from transparent
   sources (statistics counted from the training corpus by stated rules, human-named word classes); numbers fitted to
   the model only as a few named scalars per mechanism (working cap: a few hundred in total); every mechanism starts
   as a hypothesis from circuit analysis and is kept only if it reduces KL and matches the model's internals; the
   headline result is a mechanism ledger (what each mechanism explains) plus a characterised residual.
7. **One-time exception to AGENTS.md section 8 "no file I/O at runtime":** the program may use the TRAIN-split corpus
   at startup to compute its statistics. Implementation: the harness reads `data/tokens/train.bin` and passes it to
   `program.setup(corpus)` once before scoring (the program itself still does no file I/O and the lint is unchanged);
   the sandbox allows reading that one file only. Dev/test text and models stay unreadable. configs/frozen.yaml v3.
8. **Side study: does "where a head looks comes from what it writes" hold in deeper models?** (2026-10-14, human
   decisions on the agent's proposal; plan in logs/look_write/deep_study.md). Train own multi-layer attention-only models (2 and 4
   blocks, otherwise the frozen v3 settings) on the laptop only (MPS); no public models and no causal frozen-vote runs
   for now; the study comes before further work on the readable copy; a per-token corpus code (e.g. a low-rank
   factorisation of corpus PMI) is allowed as an analysis test only, not as a program component. configs/frozen.yaml
   is unchanged: the main track stays at 1 block; the side study is recorded in configs/side_deep_attn.yaml.
9. **Goal reframed; side study "can a language model be trained to be readable?"** (2026-10-17, human decisions on the
   agent's proposal after reading Hubinger's "A transparency and interpretability tech tree"; plan in
   logs/study_C_readable/readable_training.md). (a) Main track: a readable replacement with KL near zero is not reachable for a model
   trained without regard to readability (another seed of the same network is 0.38-0.42 nats from attn_s0; the
   remaining gap is in per-head tables locked to the model's own attention, logs/main_track/token_geometry_results.md). The main
   track's goal becomes best-case understanding (most important pieces, why they formed) with an honest coverage
   measure of what is not explained. (b) Side study: train new 1-block attention-only models with an extra loss pulling
   them toward the readable program, and check with tools that were not trained on whether the agreement is genuine or
   Goodharted. Choices: both arms (O: the program's next-token predictions; I: heads' attention follows the program's
   rules and their votes follow its corpus tables), laptop MPS only, target = program v1.3 frozen (scalars as fitted
   to attn_s0). Pre-registered (logs/study_C_readable/prereg_readable_training.md), with human gates G0-G3. configs/frozen.yaml is
   unchanged; the side study is recorded in configs/side_readable.yaml.
10. **Three next studies** (2026-10-18, after the readable-training development results and a literature check showed
    that output/mechanism decoupling alone is not new): (A) `fixed_votes`: models whose head votes are fixed corpus
    tables and only attention is learned, to see how attention forms and whether the look-follows-write law predicts it
    causally; the human expects no readable language model from it and values it as a study of how attention works;
    (B) `law_pressure`: training against the look-follows-write law (can it be Goodharted?); (C) `canonical`: does
    readability pressure remove the arbitrariness between training runs (part A = the readable-training confirmatory
    runs as pre-registered; part B = a pressure curve and a non-readable canonicaliser control). Plan:
    logs/plans/plan_attention_law_and_readability.md.
11. **Readable training, test M amended** (2026-10-18, gate G2 approved the same day): the manipulation check's
    threshold is lowered from 0.3 to 0.2 trained closure, before any confirmatory output is read, because both seed-0
    development runs sit at 0.27-0.28 while their trained tools moved far beyond noise. The as-written verdict is
    reported as well. Amendment section in logs/study_C_readable/prereg_readable_training.md.
12. **Studies A and B approved; their runs move to a cloud GPU** (2026-10-19): the human approved gates G0 and G1 for
    study A (fixed votes) and study B (training against the law): DRAFT pre-registrations, training code and tests. The
    human has USD 200 of AWS credits and 2-3 days; the plan is one g6.8xlarge instance (1 NVIDIA L4, 32 vCPUs,
    128 GiB) once AWS grants the GPU quota. Each study runs on one kind of hardware: studies A and B entirely on the
    instance (development, confirmatory runs and their controls), falling back to the laptop GPU only if the instance
    is not available; the readable-training study stays on the laptop (its baselines, development runs and harness
    sandbox are there). Until then, work that carries over (setup kit, rule scripts) is prepared on the laptop.
13. **Gate G2 for studies A and B** (2026-11-06, after the human reviewed the two FINAL pre-registrations and the
    seed-0 development results): **study B approved in full** as pre-registered (lambda* = 1, law_windows 16, arm both
    only; twins, penalised models, continuations and law tools for seeds 1-4). It runs on the **laptop GPU** (MPS), the
    pre-registered fallback, since the instance's GPU quota has not arrived; every run of study B's confirmatory phase
    (twins, penalised models, continuations) stays on that hardware. **Study A's confirmatory runs are not run**:
    seed 0 already makes H3 near-certain and near-built-in and H1 a clear failure, and 8 one-epoch runs (about 32 h of
    laptop GPU) would add little. Its pre-registered tests are therefore never scored; the seed-0 results remain
    development results with no verdict weight. Instead the pre-registered exploratory analysis "usefulness relative
    to the model's own prediction" (explaining away) runs on the seed-0 models (CPU). Study A's confirmatory runs may
    be reopened only if that analysis shows something that needs seeds to confirm, and then as a new decision.
14. **Readable training closed; priorities; the coverage measure** (2026-11-06): the human read
    logs/study_C_readable/readable_training_results.md and passed gate G3 (both arms GOODHARTED as written). The to-do list was compiled
    into one order (plan approved by the human): close out study B and study A's exploratory analysis; main track
    step 1, the honest coverage measure of item 9 = (KL_bigram - KL_program) / (KL_bigram - KL_floor) with the floor
    the probability average of the other four seeds (logs/main_track/coverage.md; v1.3: 0.728); step 2, a diagnostic of the
    copying positions where the seeds agree but the program does not, then candidates by the usual adoption rule; a
    running write-up draft (writeups/draft.md); then study C part B with a pressure curve and two non-readable
    controls at matched cost, one for arm O (attn_s0's predictions as target) and one for arm I (attn_s0's attention
    patterns and vote tables as target), both chosen by the human, under an addendum pre-registration with gates
    G0-G3. Deferred: the mlp variant, the black-box track; AWS unused unless part B needs it.
15. **Side studies closed; new main line; repository clean-up** (2026-11-06, after the human reviewed study B's
    results, a literature check and the agent's directions; plan approved by the human):
    - **Study B: gate G3 passed** on logs/study_B_law_pressure/law_pressure_results.md as written (verdict MIXED,
      restoring force SLOW). No further study B work.
    - **Study C part B shelved:** the addendum stays a DRAFT, code and tests are kept, nothing is run.
    - **Main deliverable changed.** The goal is to truly understand a language model "by any means", including insight
      from the weights. The deliverable becomes a measured, complete account of each model on a ladder of tiny
      language models (character-level TinyStories, from a few hundred parameters upward, one change per rung), not a
      weight-free program. The weight-free program track ends at v1.3 as the top reference point; AGENTS.md section 8
      binds only that track.
    - **New side models** outside configs/frozen.yaml are allowed for the ladder (configs/side_ladder.yaml);
      frozen.yaml is unchanged.
    - **The ladder's design** (data, rungs, completeness criteria fixed before any account is judged) gets its own
      human gate before any rung is trained.
    - **Also approved, in parallel on the CPU:** a map of what the seeds know that the program does not, by kind of
      language.
    - **Repository clean-up:** finished or abandoned code deleted (recoverable from git tag `pre-cleanup`);
      logs/ organised into one folder per study.
16. **Ladder design gate passed** (2026-11-06, after the human reviewed logs/ladder/design.md and the agent's
    proposal for criteria the wider field can use): the design is FINAL as written.
    - **Rung list** approved as drafted (L1-L10, branch M).
    - **LayerNorm:** rungs L1-L3 trained in two arms, `ln` and `noln`, five seeds each, with the decision rule in the
      design (the human's proposal).
    - **Pass conditions:** P1 KL ≤ 0.01 nats per symbol; P2 coverage ≥ 0.95; P3 causal scrubbing of every claim,
      faithfulness ≥ 0.95 jointly and per claim. All five seeds, one account structure.
    - **Reported, not passed or failed:** R1 description length against the quantised-weights reference (the headline
      curve); R2 the ARC-style mechanistic estimate against sampling at equal compute; R3 the human's readability
      scores (AGENTS.md section 12).
17. **P3 gets a replacement test** (2026-11-06, before any account was scored, after the agent showed that swap-test
    donors run out: on dev_full, keys of the last three characters leave 0.95% of positions without a donor, and a
    larger pool gains only about one character per 5x): a claim is either a key claim ("depends only on"), tested by
    the swap test with donors from dev_full and the 1% rule as before, or a value claim ("computes this"), tested by
    replacing the site's activation with the account's computed value. Same faithfulness score and thresholds. The
    donor pool stays dev_full. logs/ladder/design.md updated.
