# Plan: clean up, close the side studies, start a ladder of tiny language models (2026-11-06)

## Context

The main track has reached a ceiling it can measure. Program v1.3 covers 0.728 of the way from the bigram to the
seed floor. Both remaining routes hit the same wall:
- the fitted-table curve needed about 1.78M numbers to reach the floor;
- Anthropic's 2026 one-layer study found that the helpful virtual weights exceed the model's own parameter count
  (logs/literature_features_and_tiny_models.md).

ARC's AlgZoo shows where the field stands. The largest model it considers fully understood has 32 parameters; it has
not cracked a 432-parameter one. Ours is about 0.35M.

The human's goal is to truly understand a language model, by any means. The new main line is therefore a ladder of
tiny language models:
- start small enough to understand completely;
- grow one dimension at a time;
- find where understanding breaks, and why.

The repository is hard to follow after many studies, so step 0 cleans it up first. The human chose:
- delete finished or abandoned code, recoverable from a git tag;
- give logs/ one folder per study.

## 0. Clean up the repository

**0a. Freeze the current state.**
- Commit the working tree: logs/iterations.csv, the watchdog log, and the untracked run logs in logs/runs/.
- Tag it `pre-cleanup`. Every deleted file stays recoverable with `git checkout pre-cleanup -- <path>`.
- Check `git worktree list` and status first.

**0b. Code: keep what runs or reproduces a cited result; delete the rest.**
- **Keep untouched:**
  - program/;
  - harness/ (read-only rule);
  - configs/frozen.yaml;
  - src/train/ (model, training, all sanity tests);
  - baselines/, blackbox/, notebooks/.
- **Keep the study packages whose results the write-up cites:**
  - analysis/look_write/, look_write_deep/, token_geometry/, program_gap/;
  - analysis/readable_training/, law_pressure/, fixed_votes/;
  - analysis/canonical/ (shelved, but ready to run).
  - Also the shared libraries they import: analysis/fitlib.py (113 importers), mech_eval.py (111), tools.py,
    attention_fit.py, vote_fit.py, rules_common.py, paired.py, and whatever else the import check below finds.
- **Delete:**
  - analysis/candidates/ and analysis/investigations/ (discovery rounds; outcomes are in logs/dead_ends.md);
  - distillation-era modules (emit_program, emit_cli, check_emit, student, prune, refit, compress_bigram, cocluster,
    consistency, run_fit, taxonomy, scratch_eval), unless a kept module imports them;
  - finished-run orchestration: scripts/deep_study, readable, next_studies, training, law_pressure, aws;
  - scripts/mech_seed_transfer.py if seed_transfer.py supersedes it.
- **Keep scripts/canonical/** (shelved study).
- **Reusable helpers** (battery watchdog, start-on-AC) move to scripts/common/.
- **Method:** delete in one commit. Then run an import check: import every kept .py, and repeat until nothing fails.
  Anything a kept module needs comes back from the tag.

**0c. logs/: one folder per study.**
- **Stay at top level:**
  - decisions.md, dead_ends.md, surprises.md;
  - iterations.csv and blackbox_queries.jsonl (the harness and the server write these; paths fixed);
  - param_counts*.json, README.md.
  - Before moving anything, check harness/ for every logs/ path it reads or writes. Those stay where they are.
- **New folders:**

  | folder | contents |
  |---|---|
  | `setup/` | phase reports, pilot, throughput |
  | `main_track/` | mechanism cards, mech v1-v3 reports, coverage, token geometry, compression experiments, consistency, human judging |
  | `look_write/` | the 1-block and deep preregs, addenda, results, deep_study.md |
  | `study_A_fixed_votes/` | |
  | `study_B_law_pressure/` | |
  | `study_C_readable/` | readable_training*, prereg, results, canonical addendum |
  | `plans/` | |
  | `literature/` | |
  | `blackbox/` | |
  | `runs/` | unchanged |

- **Moves:** one commit of `git mv` only, with no content changes, so the pre-registration history stays auditable.
- **References:** a second commit rewrites every reference to a moved path in tracked .md, .py, .sh and .yaml files.
  Not in harness/ or configs/frozen.yaml.
- **Pre-registration gate checks** switch to `git log --follow -- <new path>`. They are in:
  - analysis/{readable_training,law_pressure,fixed_votes}/verdicts.py;
  - scripts/aws/make_jobs.py, deleted anyway;
  - the canonical and readable queue scripts.

**0d. Disk: delete regenerable caches of finished work** (untracked, so not in the tag).
- **Delete:** large .npy/.pt caches under analysis/out/:
  - candidates (2.3 GB);
  - look_write uov tables of attn_s* (4.3 GB folder);
  - look_write_deep (3.2 GB);
  - fixed_votes (1.2 GB).
- **Keep:** every .json result, analysis/out/program_stats (mech_eval reads it), corpus_stats_d8.npz, and every
  model in models/final.
- models/pilot and models/sanity: keep result.json, delete checkpoints, unless a sanity test loads them (check
  first).
- Before deleting, print the exact list and sizes into the cleanup commit message.

**0e. Documentation.**
- README.md rewritten top-down:
  - what the project is, and the one-paragraph result so far;
  - the folder map;
  - how to run the program, harness, training and tests;
  - the status of every line of work: done, shelved or active.
- A three-line README in each kept analysis/ subfolder: question, entry point, results file.
- logs/README.md becomes the index by study.
- writeups/draft.md paths updated.

**0f. Verification of step 0.**
- All sanity tests pass: `.venv/bin/python -m pytest src/train` and the program's claim tests through the harness.
- A harness run of program v1.3 reproduces dev_fast KL 0.5497.
- Every kept .py imports.
- Each verdicts.py `final_committed()` still finds its FINAL.
- A grep for each old logs/ path returns nothing outside harness/.
- `df` shows the freed space.

## 1. Close the side studies and record the decisions

logs/decisions.md item 15 (approving this plan approves these):
- study B closed: gate G3 passed on logs/law_pressure_results.md, verdict MIXED, restoring force SLOW;
- study C part B shelved: code and tests kept, DRAFT not run;
- the main deliverable changes, from a weight-free program to a measured, complete account of each model on a ladder.
  The weight-free program track ends at v1.3 as the top reference point;
- new side models outside configs/frozen.yaml, recorded in configs/side_ladder.yaml; frozen.yaml unchanged.

Also update writeups/draft.md: status table, section 6, and a new part 7, the ladder. Update the memory resume point.

## 2. The ladder of tiny language models (main line)

**2a. Design doc first** (logs/ladder_accounts/design.md), human gate before any rung is trained.
- **Data:** character-level TinyStories. The alphabet is counted on the TRAIN split, with rare characters mapped to
  one UNK and an end-of-story symbol. Same dev slice as the main track, excluded from fitting.
- **Models:** reuse src/train/model.py and train.py (TinyLM). Check that small d_model, few heads and a small vocab
  work.
  - Context about 128 characters.
  - Five seeds per rung, so each rung has its own seed floor, computed with coverage.py's method.
- **Rungs,** smallest first, one change per step:
  - attention-only, 1 layer: d 8 with 1 head, then 2 heads, then d 16 / 2 heads, d 32 / 4 heads;
  - then add an MLP;
  - then 2 layers;
  - then a BPE vocabulary of 256, then 2,000. That rung meets the current models.
  - Parameter counts are computed in code.
- **Completeness criteria, fixed in the design doc before any rung's account is judged:**
  - KL from the model to the account, relative to the rung's seed floor;
  - coverage (as in logs/coverage.md);
  - ARC's mechanistic test: estimate the model's dev loss from the account alone, without sampling. Compare its error
    with a sampling estimate at equal compute.
  - A rung is "understood" if it meets all three thresholds.
- **What an account may use:** anything, including the weights ("by any means"). It is written as readable code plus
  a description, and judged by the human with the AGENTS.md section 12 questions.

**2b. Tooling** (analysis/ladder/).
- A ladder harness that reuses harness/ functions by import; harness/ is not edited.
- Per-rung table dumps of QK, OV, positions and the direct path, small enough to read whole at the bottom rungs.
- Reuse at higher rungs: the law tools (analysis/look_write), the tables-versus-attention factorial
  (analysis/token_geometry/tables_vs_attention.py), and the seed-floor code (analysis/program_gap/coverage.py).

**2c. Climb.**
- Train a rung on MPS: minutes for the small rungs, under an hour each.
- Write its account, measure, then go one rung up.
- At the first rung that fails, record what breaks: which criterion, which part, and why.
- Result file: logs/ladder/results.md. Dead ends and surprises as usual.

## 3. In parallel (CPU): what the seeds know that the program does not

Direction 2, on the existing attn_s0..s4 and v1.3. Reuse analysis/program_gap/consensus.py (positions where the seeds
agree and the program misses) and worst-KL clustering.

New analysis/program_gap/phenomena.py:
- groups those positions by kind of language: names and pronouns, dialogue and quotes, agreement, story openings and
  endings, word completion, and the rest;
- runs contrast tests (one word changed);
- reports the share of the shared gap per kind.

Results go to logs/main_track/phenomena.md. These are the targets the ladder should explain once it reaches the
2,000-token rung.

## Order and gates

1. Step 0 (CPU; about half a day).
2. Step 1 (records).
3. Step 2a design doc. **Human gate.**
4. Steps 2b-2c on MPS and step 3 on the CPU, in parallel.

The human also reads each rung's account (section 12 questions). Check `df` and the battery before runs.

## Verification

- Step 0: section 0f.
- Ladder:
  - a sanity test per rung: causality, a beaten bigram, param counts written to logs/ladder/param_counts.json;
  - the ladder harness reproduces the real harness's KL on attn_s0 against v1.3 (0.5497);
  - the mechanistic loss estimate is checked against sampled loss on the smallest rung.
- Step 3: phenomena.py with no grouping reproduces consensus.py's totals.
