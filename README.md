# trlm: worst-case guarantees for tiny TinyStories language models

We train tiny language models on TinyStories and try to prove things about them that hold on **every** input:
- an upper bound from a proof;
- a lower bound from an input found by search;
- exact when the two meet.

The question is how far such proofs scale with model size, and whether mechanistic insight makes them shorter and
tighter than generic verification. Agents start with `AGENTS.md`. The human's decisions are in `logs/decisions.md`,
numbered.

## Where things stand (2026-11-07)

- **The models.** A ladder of character-level models, one change per rung, five seeds each:
  - L1 (644 parameters) to L7 (84k) are trained, attention only, one block;
  - L1-L3 also without LayerNorm;
  - dev losses in `logs/ladder/trained.md`.
- **What L1 does on typical text** (the account study, `logs/ladder_accounts/`):
  - it ignores position;
  - its prediction is a bigram squeezed into 4 tied dimensions;
  - its head acts mostly as a fixed correction for the current character;
  - context changes its output by 0.003-0.07 nats on dev stories.
- **Why proofs.** Pass conditions for "the model is understood" were met every time by restating the computation in
  a new shape, and residuals measured on data are not guarantees (decision 23).
- **Next.** The proofs design doc (`logs/proofs/design.md`), a human gate, then L1. The properties:
  - **locality:** how much characters k or more positions back can change the output;
  - **forbidden output:** the largest probability of a symbol pair never seen in training.
- **Earlier work, kept as reference points:**
  - **the main track:** a weight-free readable program (`program/`, v1.3) reaches 0.728 of the way from the bigram
    to the seed floor on a 355k-parameter model (`logs/main_track/`);
  - **side studies:** the look-follows-write law (`logs/look_write/`) and three training studies
    (`logs/study_*`).

Running write-up for outside readers (not published): `writeups/draft.md`.

## Status of each line of work

| line | state | where |
|---|---|---|
| worst-case properties with proofs | active: design next | `logs/proofs/`, `analysis/proofs/` |
| ladder models | L1-L7 trained; L8 and the MLP branch not yet | `logs/ladder/`, `src/train/ladder_runs.py` |
| ladder accounts | closed (decision 23) | `logs/ladder_accounts/` |
| readable program for attn_s0 (main track) | ended at v1.3, reference point | `program/`, `logs/main_track/` |
| look-follows-write law | done, verified | `logs/look_write/` |
| studies A, B, C | closed or shelved | `logs/study_A_fixed_votes/`, `logs/study_B_law_pressure/`, `logs/study_C_readable/` |
| black-box track | started, not continued | `logs/blackbox/`, `blackbox/` |

## Folders

| path | what |
|---|---|
| `AGENTS.md` | mission and standing rules for agents (the original brief: `logs/plans/original_brief.md`) |
| `analysis/proofs/` | proof tools |
| `analysis/ladder/` | ladder model access: load, eval windows, KL, seed floor, weight dumps, NumPy forward pass |
| `src/train/` | model, training loop, BPE and character data, ladder rungs, main-track runs, sanity tests |
| `scripts/` | `ladder/` (training queues), `common/watchdog.sh` (battery watchdog) |
| `harness/`, `program/` | the main track's scorer and program v1.3 (read-only) |
| `configs/` | `frozen.yaml` (main track, read-only), `side_ladder.yaml` (the ladder), the closed studies' configs |
| `logs/` | decisions, dead ends, surprises; one folder per study (index `logs/README.md`); raw run output in `runs/` |
| `writeups/` | drafts for outside readers |
| `baselines/`, `blackbox/`, `notebooks/` | n-gram baseline programs, the black-box model server, the Colab notebook |
| `data/`, `models/` | data and checkpoints (not in git) |

**Removed code** is at git tags:
- `pre-cleanup` (2026-11-06): discovery rounds, distillation era, run queues, cloud kit;
- `pre-proofs` (2026-11-07): ladder accounts, main-track analysis libraries, side-study code.

Recover with `git checkout <tag> -- <path>`. Some stored results cite removed code, e.g.
`analysis/out/look_write_deep/verify/`, which needs `analysis/look_write/` from `pre-proofs`.

## How to run

```bash
# sanity tests (model, training, ladder data) and the ladder tool tests
.venv/bin/python -m pytest src/train analysis/ladder
# dev loss, baselines and seed floor of one ladder model
.venv/bin/python -m analysis.ladder.eval --rung L1 --arm ln --seed 0
# train ladder rungs (see the scripts for devices)
scripts/ladder/rungs_L1_L3.sh
# the main track's program, scored by the harness
.venv/bin/python -m harness.eval_fidelity --program program --model attn_s0 --split dev_fast
```

Long runs go on the charger with the lid open. Use `nohup caffeinate -dimsu <queue> &` and
`scripts/common/watchdog.sh <queue pattern>`. Check free disk first.
