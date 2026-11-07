# Logs index

Reorganised 2026-11-06: one folder per study; the moves are commit 46d6101. Raw run output is in `runs/`.
`configs/frozen.yaml` still names `logs/pilot.md`, which is now `setup/pilot.md`; frozen.yaml is read-only.

## Top level: running records

| file | what |
|---|---|
| `decisions.md` | every human decision, numbered; read first |
| `dead_ends.md` | every failed hypothesis with its evidence |
| `surprises.md` | unexpected findings, as they appeared |
| `iterations.csv` | one row per harness run; the harness writes it, and `test_looks.csv` for the test split |
| `blackbox_queries.jsonl` | the black-box server's query log |
| `param_counts.json`, `param_counts_deep.json` | parameter counts counted in code |

## Folders

**`setup/`: phases 0-4.**
- `phase0-2_report.md`, `phase3-5_status.md`, `phase4_report.md`;
- `pilot.md`: the token budget;
- `throughput.md`.

**`main_track/`: the readable program for attn_s0.**
- **Distillation era** (tables fitted to the model; ended by decision 5; git tag `distillation-v11`):
  `compression_experiments.md`, which holds the fidelity-complexity curve, and `consistency.md`.
- **Mechanism program** ("explain, don't fit"):
  - `mechanism_cards.md`: one card per head;
  - `mech_v1_report.md` -> `mech_v2_investigations.md` -> `mech_v3_report.md`: v1 to v1.3, and how far apart the
    seeds are;
  - `token_geometry_results.md`: named properties against the token geometry the seeds share; tables against
    attention, the lock;
  - `coverage.md`: the coverage measure (decision 14) and where the remaining gap is;
  - `human_judging.md`: the judging sheet (AGENTS.md section 12); not yet filled in.

**`look_write/`: "where a head looks follows from what it writes".**
- One layer: `prereg_look_follows_write.md` -> `look_follows_write_results.md`; addenda
  `prereg_look_follows_write_addendum.md` and `..._addendum2.md` -> `look_follows_write_addendum_results.md`.
- Deeper models: `deep_study.md` (running log and rules), `prereg_deep_look_write.md` -> `deep_look_write_results.md`
  (independently verified).
- Draft for outside readers: `../writeups/look_follows_write_predraft.md`.

**`study_A_fixed_votes/`: votes fixed to corpus tables, attention learned.**
- `prereg_fixed_votes.md`, `fixed_votes.md` (running log), `fixed_votes_results.md`.
- Development only; confirmatory runs dropped (decision 13).

**`study_B_law_pressure/`: training against the look-follows-write law.**
- `prereg_law_pressure.md`, `law_pressure.md` (running log), `law_pressure_results.md`.
- Verdict MIXED, restoring force SLOW.

**`study_C_readable/`: training toward the program.**
- Part A: `prereg_readable_training.md`, `readable_training.md` (running log), `readable_training_results.md`. Both
  arms GOODHARTED; G3 passed.
- Part B: `prereg_canonical_addendum.md`, a DRAFT, shelved.

**`proofs/`: the main line from decision 23.** Worst-case properties with proofs on the ladder's models:
`design.md` (the properties and soundness rules; human gate before any bound is computed), then results per rung.

**`ladder/`: the ladder's models.** `pilot.md` (learning rates and budgets), `trained.md` (dev losses of every
trained rung), `param_counts.json`, `dumps/` (readable weight dumps).

**`ladder_accounts/`: the account study (decisions 16-22; closed by decision 23).** `README.md` (question, findings,
why it closed), `design.md` (criteria with every amendment), `L1.md` (accounts v1 and v2, the attention study),
`compress_probe.md`, `iterations.csv`.

**`plans/`.**
- `plan_attention_law_and_readability.md`: the plan behind studies A, B and C.
- `plan_cleanup_and_ladder.md`: the clean-up and the ladder of tiny models (decision 15).
- `plan_proofs.md`: the second clean-up and the proofs line (decision 23).
- `original_brief.md`: the original AGENTS.md, verbatim (until 2026-11-07).

**`literature/`.** `literature_features_and_tiny_models.md`: feature decompositions of small transformers and the
tiny-model challenges (AlgZoo).

**`blackbox/`.** `blackbox_track.md`: started, not continued.

## Before the clean-ups

Code removed on 2026-11-06 is at git tag `pre-cleanup`. Reports may still cite such files, for example
`analysis/candidates/...` in `dead_ends.md`. Recover them with `git checkout pre-cleanup -- <path>`. Deleted caches
are listed in `runs/cleanup_deleted_caches.txt`. Code removed on 2026-11-07 (the ladder account tools, and the
closed main-track and side studies' analysis and training code) is at git tag `pre-proofs`.
