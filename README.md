# trlm: understanding tiny TinyStories language models

We train tiny language models on TinyStories and try to understand them completely. The original brief (`AGENTS.md`)
asks for a human-readable NumPy program that reproduces a model's next-token distribution without its weights. The
human's decisions since then are in `logs/decisions.md`, numbered; read them before acting.

## Result so far

- **The models.** One block, attention only, 4 heads, d_model 128, a 2,000-token vocabulary: about 0.35M parameters.
- **The readable program** (`program/`, v1.3): built from corpus counts and a few hundred named scalars, it is
  0.55 nats from its target model attn_s0. That covers 0.728 of the way from the bigram to the seed floor, the
  average of four other training runs (`logs/main_track/coverage.md`).
- **The rest is locked.** No single part of the program carries the remaining gap. The model's attention and its vote
  tables help only together (`logs/main_track/token_geometry_results.md`).
- **Fitting tables to the model** (the earlier distillation track) needed about 1.78M numbers to reach the floor
  (`logs/main_track/compression_experiments.md`).
- **Side findings:**
  - where a head looks follows from what it writes (`logs/look_write/`);
  - training toward the program separates output agreement from mechanism agreement (`logs/study_C_readable/`);
  - penalising the look-follows-write law breaks its pair-specific form at no cost (`logs/study_B_law_pressure/`).
- **Next line:** a ladder of tiny character-level models, from a few hundred parameters upward, to find the largest
  model that can be understood completely and what breaks first (plan in `logs/plans/`).

Running write-up for outside readers (not published): `writeups/draft.md`.

## Status of each line of work

| line | state | where |
|---|---|---|
| readable program for attn_s0 (main track) | ended at v1.3, kept as the reference point | `program/`, `logs/main_track/` |
| where a head looks follows from what it writes | done, verified | `logs/look_write/`, `writeups/look_follows_write_predraft.md` |
| study A, fixed-vote models | development only, no verdicts | `logs/study_A_fixed_votes/` |
| study B, training against the law | done: MIXED, restoring force SLOW | `logs/study_B_law_pressure/` |
| study C part A, readable training | done: both arms GOODHARTED | `logs/study_C_readable/` |
| study C part B, seed convergence | shelved (DRAFT, code and tests ready) | `logs/study_C_readable/prereg_canonical_addendum.md` |
| black-box track | started, not continued | `logs/blackbox/`, `blackbox/` |
| ladder of tiny models | next | `logs/plans/` |

## Folders

| path | what |
|---|---|
| `AGENTS.md` | the original brief: mission, frozen decisions, rules for the program, phases |
| `configs/` | `frozen.yaml` (main track, read-only) and the side studies' configs |
| `program/` | the readable program for attn_s0 (start with `program/README.md`); claim tests in `program/tests/` |
| `harness/` | the scorer (read-only): fidelity, lint, sandbox, claim tests |
| `src/train/` | tokenizer, model, training loop, the side studies' training losses, all sanity tests |
| `analysis/` | analysis code, one package per study (each has a short README); shared libraries at top level |
| `scripts/` | `iterate.py` (harness run plus commit), `fit_scalars.py`, `seed_transfer.py`, `score_on_seeds.sh`, `setup_blackbox.py`; `common/watchdog.sh` (battery watchdog); `canonical/` (study C part B queues) |
| `logs/` | decisions, dead ends, surprises at top level; one folder per study (index: `logs/README.md`); raw run output in `logs/runs/` |
| `writeups/` | drafts for readers outside the project |
| `baselines/`, `blackbox/`, `notebooks/` | n-gram baseline programs, the black-box model server, the Colab training notebook |
| `data/`, `models/` | token streams, evaluation sets, corpus pair counts, checkpoints (not in git) |

**Shared analysis libraries** (`analysis/*.py`):
- `fitlib.py`, `mech_eval.py`: windows, models, corpus statistics;
- `tools.py`, `attention_fit.py`, `vote_fit.py`: model internals, attention and vote fits;
- `corpus_pairs.py`: whole-stream pair counts;
- `rules_common.py`: the side studies' decision-rule helpers;
- `worst.py`, `head_effects.py`, `consistency.py`, `claims_measure.py`: the brief's discovery and consistency tools;
- `taxonomy.py`, `cocluster.py`, `gen_categories.py`, `gen_vocab.py`: generate `program/categories.py` and
  `program/vocab.py`.

`analysis/scratch/` is untracked scratch from the distillation era and is not maintained.

**Removed code.** Code from finished or abandoned work (the discovery rounds' candidates, the distillation-era
modules, the finished run queues, the cloud kit) was removed on 2026-11-06. It is at git tag `pre-cleanup`:
`git checkout pre-cleanup -- <path>`. The regenerable caches deleted at the same time are listed in
`logs/runs/cleanup_deleted_caches.txt`.

## How to run

```bash
# score the program on dev (appends a row to logs/iterations.csv)
.venv/bin/python -m harness.eval_fidelity --program program --model attn_s0 --split dev_fast
# sanity tests (model, training, side-study losses)
.venv/bin/python -m pytest src/train
# train one model (frozen settings)
.venv/bin/python -m src.train.runs --help
# corpus pair counts used by the law tools (about 15 s)
.venv/bin/python -m analysis.corpus_pairs
```

Long runs go on the laptop GPU (MPS), on the charger with the lid open. Use
`nohup caffeinate -dimsu <queue> &` and `scripts/common/watchdog.sh <queue pattern>`. Check free disk first.
