# trlm: readable replacement programs for tiny language models

Train tiny TinyStories language models, then write a human-readable NumPy program that reproduces each model's
next-token distribution without using its weights. The rules, frozen decisions and phases are in `AGENTS.md`; the
human's decisions since then are in `logs/decisions.md`.

## Where things are

| path | what |
|---|---|
| `AGENTS.md` | the brief: mission, frozen decisions, rules for the replacement program, phases |
| `configs/` | frozen settings (`frozen.yaml`) and the side study's (`side_deep_attn.yaml`) |
| `program/` | the current replacement program for `attn_s0` (start with `program/README.md`); claim tests in `program/tests/` |
| `harness/` | the scorer (read-only for agents): fidelity, lint, sandbox, claim tests |
| `src/train/` | tokenizer, model, training loop, sanity tests |
| `analysis/` | open-track analysis tools; one folder per study (below) |
| `scripts/` | `iterate.py` (harness run + commit), `fit_scalars.py`, `seed_transfer.py`, `score_on_seeds.sh`; finished run orchestration in `scripts/deep_study/` and `scripts/training/` |
| `logs/` | reports, decisions, pre-registrations, dead ends, surprises (index: `logs/README.md`); raw run outputs in `logs/runs/` |
| `writeups/` | drafts for readers outside the project |
| `baselines/`, `blackbox/`, `notebooks/` | n-gram baseline programs, the black-box track's model server, the Colab training notebook |
| `data/`, `models/` | token streams, evaluation sets, checkpoints (not in git) |

Analysis folders: `look_write/` and `look_write_deep/` (the study "where a head looks follows from what it writes"),
`program_gap/` (where the mechanism program differs from the model), `token_geometry/` (named properties versus the
models' shared token geometry; tables versus attention), `readable_training/` (the side study "can a language model be
trained to be readable?"; training code in `src/train/readable.py`), `investigations/` and `candidates/` (discovery rounds and the
candidate mechanisms they tested), `out/` (outputs; large caches are not in git), `scratch/` (not in git).

## Status (2026-10-17)

- Models: `attn_s0..s4` and `mlp_s0..s4` (one block), side-study models with 2 and 4 blocks.
- Program: mechanism program v1.3, dev_fast KL 0.5497 on `attn_s0` (corpus bigram alone: 1.20; another seed of the
  same network: 0.38-0.42 on TRAIN windows). Reports: `logs/mech_v1_report.md`, `logs/mech_v2_investigations.md`,
  `logs/mech_v3_report.md`.
- Study "where a head looks follows from what it writes": complete and independently verified; pre-draft in
  `writeups/look_follows_write_predraft.md`.
- Token geometry: `logs/token_geometry_results.md`.
- Goal reframed (logs/decisions.md item 9): the main track aims at best-case understanding with an honest coverage
  measure. New side study, "can a language model be trained to be readable?": `logs/readable_training.md`,
  pre-registration `logs/prereg_readable_training.md`, config `configs/side_readable.yaml`.
- Next (decision 10): three studies on attention and readability, planned in
  `logs/plan_attention_law_and_readability.md` (fixed-vote models, training against the look-follows-write law,
  readability and the arbitrariness between training runs).
