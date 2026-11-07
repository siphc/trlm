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
16. **Ladder design gate passed** (2026-11-06, after the human reviewed logs/ladder_accounts/design.md and the agent's
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
    donor pool stays dev_full. logs/ladder_accounts/design.md updated.
18. **P1 against the seed floor** (2026-11-06, before any account was scored, after the first L1 models showed that
    0.01 nats is 7-30 times below their seed floor (0.066-0.32), so meeting it needs the embedding copied to about two
    decimals; the human's proposal): P1 passes if KL(model || account) <= max(KL_floor, 0.01), KL_floor = KL(model ||
    probability average of the other four seeds). Coverage (the former P2) is reported, not passed. A rung is
    understood if one account structure meets P1 and P3 for all five seeds.
19. **A size pass and behaviour tests** (2026-11-06, after the first L1 account was scored; the human's choice
    "both"). The trigger: the account (logs/ladder_accounts/L1.md) passed P1 and P3 on all five L1 ln seeds while being 3 times
    the size of the weights. Its stated mechanism also held for seed 0 only. Value claims (decision 17) let a
    restatement of the computation pass P3. Two pass conditions are added; P1 and P3 are unchanged.
    - **P2, size:** for each seed, R1 of the account file must be at most R1 of the weights reference. The reference
      is now quantised to the P1 bar max(KL_floor, 0.01), the same fidelity the account must reach; it was 0.01.
    - **P4, behaviour:**
      - Each claim carries at least one behaviour test: code in the account that measures a stated behaviour of the
        model and compares it with a stated threshold. Example: the mean attention entropy of space queries is at
        most 1 nat.
      - A test runs on the model, not the account: its attention, its site activations and its outputs on dev_full.
      - The code is the same for every seed. It may read the account's named per-seed constants.
      - It passes if it holds on all five seeds and fails on the same rung's untrained model (init seed 0), so a
        vacuous test cannot pass.
      - Every behavioural statement in the description has a test. The human checks this mapping at R3.
    - **Understood** now means one account structure meets P1, P2, P3 and P4 on all five seeds. The first L1 account
      fails P2 (33-36k bits against about 9-12k).
20. **Behaviour tests: the claim counts, the measuring tool does not** (2026-11-06, after L1 account v2 passed P1, P3
    and P4 but failed P2 at 15.4k bits against 8.2-9.1k, 7.6k of it test code; the human's choice).
    - Generic measurements live in analysis/ladder/behaviour.py, shared by every account, like scrub.py. Examples:
      the cost of resample-ablating a site, of replacing the positions by their mean, or of replacing a head by its
      mean for the current symbol; a head's content share; attention on a named symbol set against that set's share
      of the keys.
    - An account states each test in one line: the statement, the measurement by name (or a ratio of two), the
      comparison and the threshold. That line counts toward R1.
    - A test that needs a measurement the tool lacks adds it to the tool. The tool stays generic: it never names an
      account's constants, except a symbol set the account defines and the test names.
21. **The P2 reference is written as text, like an account** (2026-11-06, after L1 v2; the human's choice).
    - The trigger: L1's embedding needs about 0.1 precision. In text that costs about 13 bits per number, against
      about 7 for the binary-packed reference, so a readable account paid about twice per number before any
      structure counted.
    - The reference now writes each weight tensor as a NumPy array literal in the account format: no plus signs,
      at the fewest decimals per tensor that keep KL(model || rounded model) at or below the P1 bar.
    - Decimals are lowered greedily from 4, largest tensor first. The reference counts the stripped, compressed
      source of the literals plus reference_forward.py.
    - The binary reference stays as a reported number.
22. **No copied maps; a context test for heads** (2026-11-07, after the human read L1 account v2 and found that it
    copies the head's matrices, so it does not explain attention; the human's choice).
    - **Tables (lint, analysis/ladder/length.py).** Every module-level NumPy array of an account is declared in
      AXES = {name: (axis, ...)}. An axis is one of:
      - "symbol" (size V);
      - "position" (size up to the context);
      - "dim" (a hidden dimension, d_model or d_head);
      - a category list the account defines in CATEGORIES, {name: [labels]};
      - a component list it defines in COMPONENTS, {name: one-line meaning}.

      At most one axis per array may be "dim". So per-symbol vectors are allowed (EMBED, what a symbol writes, a
      gain), and maps between hidden spaces are not (the query, key, value and output matrices).
    - **Component axes.** These carry factorised scores, such as an attention score built from per-symbol query and
      key loadings. Each component needs a stated meaning. Whether the meaning is real is judged at R3.
    - **Head value claims** may give the head's write to the residual stream (shape (n, T, d_model); the claim states
      "space": "residual"). So the output map folds into per-symbol values.
    - **Context faithfulness (added to P3 for every head claim).**
      1 − KL(model || head patched by the claim) / KL(model || head replaced by its mean for the current symbol)
      must be at least 0.95. The account must explain what the head does with context, not only its per-symbol
      average, even where that is below the seed floor.
    - **P2 is removed** (the human's amendment). R1 and the text and binary references are still reported.
      Per-symbol score tables cost more bits than four 4×4 maps, so with the table rule P2 would reward copying
      again or block the attempt.
    - **Understood** now means: one account structure meets P1, P3 (with context faithfulness) and P4 on all five
      seeds, and passes the table rule.
    - L1 account v2 fails the table rule (WQ, WK, WV, WO).
    - **Against running in circles** (the human's concern: decisions 18-22 were all made at L1, each after an account
      passed in a way that did not read as understanding):
      - The criteria are frozen for the L1 attention account. No further changes until the stopping point below.
      - Stopping point: 10 committed account versions for L1 ln, or earlier if a version passes everything.
      - Outcomes, fixed now:
        1. **Pass on all seeds:** L1 ln is understood. The human reads it (R3). Then L1 noln and L2 follow under the
           same criteria.
        2. **No pass by the stopping point:** L1 is recorded as the first rung where the table form fails. The
           record says which condition fails, on which seeds, and why. The next decision is about the method (what
           kind of account), not the criteria.
