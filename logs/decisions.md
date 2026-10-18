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
   decisions on the agent's proposal; plan in logs/deep_study.md). Train own multi-layer attention-only models (2 and 4
   blocks, otherwise the frozen v3 settings) on the laptop only (MPS); no public models and no causal frozen-vote runs
   for now; the study comes before further work on the readable copy; a per-token corpus code (e.g. a low-rank
   factorisation of corpus PMI) is allowed as an analysis test only, not as a program component. configs/frozen.yaml
   is unchanged: the main track stays at 1 block; the side study is recorded in configs/side_deep_attn.yaml.
9. **Goal reframed; side study "can a language model be trained to be readable?"** (2026-10-17, human decisions on the
   agent's proposal after reading Hubinger's "A transparency and interpretability tech tree"; plan in
   logs/readable_training.md). (a) Main track: a readable replacement with KL near zero is not reachable for a model
   trained without regard to readability (another seed of the same network is 0.38-0.42 nats from attn_s0; the
   remaining gap is in per-head tables locked to the model's own attention, logs/token_geometry_results.md). The main
   track's goal becomes best-case understanding (most important pieces, why they formed) with an honest coverage
   measure of what is not explained. (b) Side study: train new 1-block attention-only models with an extra loss pulling
   them toward the readable program, and check with tools that were not trained on whether the agreement is genuine or
   Goodharted. Choices: both arms (O: the program's next-token predictions; I: heads' attention follows the program's
   rules and their votes follow its corpus tables), laptop MPS only, target = program v1.3 frozen (scalars as fitted
   to attn_s0). Pre-registered (logs/prereg_readable_training.md), with human gates G0-G3. configs/frozen.yaml is
   unchanged; the side study is recorded in configs/side_readable.yaml.
10. **Three next studies** (2026-10-18, after the readable-training development results and a literature check showed
    that output/mechanism decoupling alone is not new): (A) `fixed_votes`: models whose head votes are fixed corpus
    tables and only attention is learned, to see how attention forms and whether the look-follows-write law predicts it
    causally; the human expects no readable language model from it and values it as a study of how attention works;
    (B) `law_pressure`: training against the look-follows-write law (can it be Goodharted?); (C) `canonical`: does
    readability pressure remove the arbitrariness between training runs (part A = the readable-training confirmatory
    runs as pre-registered; part B = a pressure curve and a non-readable canonicaliser control). Plan:
    logs/plan_attention_law_and_readability.md.
