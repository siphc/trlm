# Ladder of tiny language models: design (DRAFT, awaiting the human's gate)

DRAFT, 2026-11-06. Decision 15; plan logs/plans/plan_cleanup_and_ladder.md, step 2. Nothing has been trained. No
rung's account exists. Everything below can change at the gate. After the gate, the completeness criteria are fixed
before any account is judged.

## Question

What is the largest language model of this family that can be understood completely? And when it grows by one step
past that point, what is the first thing that stops being understood?

For scale:
- ARC's AlgZoo has not fully understood a 432-parameter RNN on an algorithmic task.
- Our attn_s0 (355k parameters) is 0.728 of the way to its seed floor under a readable account
  (logs/literature/literature_features_and_tiny_models.md).

## Data

- **Source.** TinyStories, the same split as the main track: 20,000 stories sampled from the official train split
  with seed 0 form the dev slice; the official validation split is the test split, not used. Reuse
  `src/train/data.py` (`load_split`, `split_official_train`, `dev_indices`).
- **Symbols are characters.**
  - The alphabet is counted on the train split minus the dev slice.
  - Characters seen fewer than 1,000 times map to one UNK symbol.
  - End-of-story is symbol 0, appended after each story, as in the main track.
  - Expected alphabet: about 70-80 symbols (measured in step 2b and recorded).
- **Streams.** Story order is shuffled and the stories concatenated into one stream, cut into windows that cross
  story boundaries (as in the main track). Pre-encoded once to uint8 (about 1.9 GB for train).
- **Fixed evaluation sets:** `dev_fast` (200 windows) and `dev_full` (2,000 windows), from the dev slice.
- **Baselines** from the train split: character unigram, bigram and trigram.

## Models

- **Architecture: the main track's TinyLM** (src/train/model.py), changing only the sizes. Pre-LayerNorm, final
  LayerNorm, tied embeddings, learned positions, no biases. So the top rung is exactly attn_s0's architecture.
- **Seeds.** Five per rung (0-4). Each rung gets its own seed floor, the same definition as logs/main_track/coverage.md.
- **Training.**
  - AdamW with the frozen settings: betas (0.9, 0.95), cosine to 10%, weight decay 0.1, clip 1.0.
  - Batch 64 windows.
  - Budget and peak LR from a mini pilot on seed 0 at rungs L1 and L5:
    - LR in {1e-3, 3e-3, 1e-2};
    - budgets 25M, 100M, 400M symbols;
    - choose the smallest budget whose dev loss is within 1% of the best.
  - That budget applies to its rung and the rungs next to it; it is re-checked when the context or vocabulary changes.

**Rungs** (one change per step; parameter counts from `count_params` with an alphabet of 72; recounted with the real
one):

| rung | change | d_model | heads | context | vocab | MLP | params |
|---|---|---|---|---|---|---|---|
| L1 | smallest | 4 | 1 | 64 | chars | no | 616 |
| L2 | width | 8 | 1 | 64 | chars | no | 1,360 |
| L3 | heads | 8 | 2 | 64 | chars | no | 1,360 |
| L4 | width | 16 | 2 | 64 | chars | no | 3,232 |
| L5 | width, heads | 32 | 4 | 64 | chars | no | 8,512 |
| L6 | width | 64 | 4 | 64 | chars | no | 25,216 |
| L7 | width | 128 | 4 | 64 | chars | no | 83,200 |
| L8 | context | 128 | 4 | 256 | chars | no | 107,776 |
| L9 | vocabulary | 128 | 4 | 256 | BPE 256 | no | 131,328 |
| L10 | vocabulary = attn_s0 | 128 | 4 | 256 | BPE 2,000 | no | 354,560 (the existing attn_s0..s4) |
| branch M | MLP, from the last understood rung | | | | | yes | e.g. 238,976 at L8 |

The ladder is climbed from L1 and stops one rung above the first failure. L10 needs no training.

## What an account is

An account may use anything, including the weights ("by any means"). It has two parts:
- **Code:** a readable NumPy program, `ladder/<rung>/account.py`. It exposes `predict(symbols) -> log-probs`; its
  constants are named and say where they come from.
- **Description:** what each part computes, in words.

At the bottom rungs, the account is expected to come from reading every weight table whole:
- QK and OV per head;
- positions;
- the direct path.

An account that merely re-expresses the weights (for example a multiplied-out table with no stated meaning) counts
for fidelity but fails criterion C3 below.

## Completeness criteria (proposed; fixed at the gate)

A rung is **understood** if one account meets all of C1-C3 for seed 0, and the same account structure, with its
named constants refitted, meets C1 and C2 for seeds 1-4.

- **C1, fidelity:** KL(model || account) on dev_full is at most 0.01 nats per symbol.
- **C2, coverage:** (KL_bigram − KL_account) / (KL_bigram − KL_floor) is at least 0.95, against the rung's character
  bigram and its seed floor. If the floor is below 0.005 nats per symbol, the seeds are nearly identical and only C1
  applies.
- **C3, readability:** the human reads the account blind to the metrics. Every component scores at least 1 on each of
  the four AGENTS.md section 12 questions, and at least 2 on "Can I predict what happens if it is removed?". Each
  removal prediction is checked against the measured removal.
- **C4, ARC-style, proposed as optional:** estimate the model's dev loss from the account and corpus n-gram statistics
  alone, with no forward pass of the model or the account on dev windows. The estimate must be within 1% of the
  measured loss. This adapts ARC's "mechanistic estimate without sampling" to text.

## Tools (step 2b, analysis/ladder/)

- `src/train/data_char.py`: the alphabet, the streams and the eval sets, reusing src/train/data.py.
- `analysis/ladder/eval.py`: KL, top-1, coverage, seed floor and baselines for any rung.
  - It reuses the harness's functions by import, but not its fixed vocabulary of 2,000.
  - Check: on L10 it must reproduce the harness's KL of program v1.3 against attn_s0 (0.5497).
- `analysis/ladder/dump.py`: per-rung tables (QK, OV, positions, the direct path) as readable arrays and plots.
- Higher rungs reuse:
  - the law tools (analysis/look_write);
  - the tables-versus-attention factorial (analysis/token_geometry/tables_vs_attention.py);
  - the floor code (analysis/program_gap/coverage.py).
- Sanity tests per rung (src/train/test_sanity_ladder.py): causality, the bigram beaten, parameter counts written to
  logs/ladder/param_counts.json.

## Questions for the gate

1. **LayerNorm.** Keep it at every rung, so the top rung is exactly attn_s0's architecture (recommended)? Or drop it at
   the bottom rungs, which makes exact reading easier, and add it back as its own rung? The final LayerNorm's
   per-position scale was most of the residual when everything else was exact (logs/main_track/token_geometry_results.md).
   It is a likely early breaking point.
2. **C4.** Include the ARC-style loss estimate as a criterion, or report it only?
3. **The thresholds** of C1 (0.01 nats per symbol) and C2 (0.95).
4. **The rung list,** especially the context change (L8) and the vocabulary steps (L9, L10).

## Compute

- Training: L1-L7 take minutes to tens of minutes per seed on MPS; L8-L9 under an hour per seed.
- Data preparation: about 30 min on the CPU.
- Disk: about 2 GB for the streams; checkpoints are small, keeping final ones only.
