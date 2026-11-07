# Ladder of tiny language models: design (FINAL, gate passed 2026-11-06)

FINAL, 2026-11-06. Decisions 15-18; plan logs/plans/plan_cleanup_and_ladder.md, step 2. The human passed the gate:
rung list as drafted; LayerNorm tested in two arms at the start; the completeness criteria below. Nothing has been
trained and no account exists at the time of fixing. Changes to the criteria after this point need a new decision
in logs/decisions.md, made before the affected rung's account is scored.
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
  - Measured alphabet: 79 symbols (77 characters, EOS, UNK); UNK covers 97 rare character types, 8,465 occurrences.
- **Streams.** Story order is shuffled and the stories concatenated into one stream, cut into windows that cross
  story boundaries (as in the main track). Pre-encoded once to uint8 (about 1.9 GB for train).
- **Fixed evaluation sets:** `dev_fast` (200 windows) and `dev_full` (2,000 windows), from the dev slice.
- **Baselines** from the train split: character unigram, bigram and trigram.

## Models

- **Architecture: the main track's TinyLM** (src/train/model.py), changing only the sizes. Pre-LayerNorm, final
  LayerNorm, tied embeddings, learned positions, no biases. So the top rung is exactly attn_s0's architecture.
- **LayerNorm in two arms at the start.** Rungs L1, L2 and L3 are trained twice, five seeds each: arm `ln` (as above)
  and arm `noln` (both LayerNorms replaced by the identity; nothing else changes). Each arm gets its own account.
  - **Decision rule, fixed now:** LayerNorm is the breaking point if, at some rung among L1-L3, the `noln` arm is
    understood (the pass conditions below) and the `ln` arm is not. Then the ladder continues from L4 without
    LayerNorm, and LayerNorm returns later as its own rung (L_LN: the last understood rung plus LayerNorm). Otherwise
    the ladder continues with LayerNorm only.
  - Reported either way: the dev-loss cost of removing LayerNorm per rung, and which parts of the `ln` account exist
    only because of LayerNorm.
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

**Tables** (decision 22, checked by analysis/ladder/length.py):
- Every module-level array is declared in AXES. Its axes are symbol, position, dim, a CATEGORIES list or a
  COMPONENTS list (each component with a stated meaning).
- At most one axis per array is "dim": per-symbol vectors are allowed, maps between hidden spaces are not.

**Rules for the code** (clarified at tooling, 2026-11-06; checked by analysis/ladder/length.py): imports from the
harness's stdlib allowlist and NumPy only, and no file reads, so every number the account uses is written in its source
and R1 counts it. It may define `setup(train_stream)` to compute corpus statistics from the train symbol stream (free,
as decision 7 for the main track). Interface: `predict(X)` with X (n, T) returns (n, T, V) log-probs; `CLAIMS` lists
the claims for P3.

At the bottom rungs, the account is expected to come from reading every weight table whole:
- QK and OV per head;
- positions;
- the direct path.

An account that merely re-expresses the weights (for example a multiplied-out table with no stated meaning) can pass
P1, but it does not compress (R1 below), and its claims cannot name keys coarser than the input itself, so P3's donor
rule fails it.

## Completeness criteria (fixed at the gate)

A rung (and arm) is **understood** if one account structure meets the pass conditions P1, P3 and P4 and the table rule
(decisions 19, 22; P2 was removed by decision 22 and its size comparison is reported under R1) for all five
seeds, with only its named constants refitted per seed. The pass conditions can all be measured by anyone, without
our human, so other groups can apply them to their own models. The remaining measures are reported, not passed or
failed.

All measures use `dev_full` unless stated. Symbols are characters; "per symbol" means the mean over every position of
every window.

### Pass conditions

- **P1, fidelity against the seed floor** (decision 18; it replaced the fixed 0.01 nats and the coverage condition).
  KL(model || account) is at most **max(KL_floor, 0.01) nats per symbol**, where KL_floor = KL(model || the
  probability average of the other four seeds of the same rung and arm), as in logs/main_track/coverage.md. The
  account must predict the model at least as well as its sibling seeds do together; detail beyond that belongs to one
  training run, not to how the model works. The 0.01 minimum keeps P1 achievable when the seeds are nearly identical.
- **P2, size** (decisions 19 and 21; REMOVED as a pass condition by decision 22, reported only). For each seed, R1 of the account is at most R1 of the text weights reference:
  each weight tensor written as an array literal in the account format, at the fewest decimals per tensor that keep
  KL at or below the P1 bar max(KL_floor, 0.01), plus the reference forward pass (analysis/ladder/length.py).
- *(The original P2, coverage, is now reported as R0.)* Coverage (KL_bigram − KL_account) / (KL_bigram − KL_floor), against the rung's
  character bigram (train split); P1 implies coverage at least 1 when KL_floor ≥ 0.01.
- **P3, mechanistic faithfulness** (decision 17 added the replacement test). Each claim is tested in one of two ways,
  chosen by the account, and passes by that test.
  - **What a claim is.** The account is written as a list of claims. Each claim names one *site* in the model (for
    a one-block attention-only model: the embedding as it enters the final sum, or head h's output) and either
    - a *key*: a function of the input window and the position, stated in the account's code, that says which inputs
      the site treats as the same (for example "the previous two characters"): a "depends only on" claim; or
    - a *value*: code in the account that computes the site's activation itself at every position: a "computes this"
      claim.
    Every component of the account carries at least one claim; the claims together cover every site that writes to
    the residual stream, one claim per site.
  - **Swap test, for key claims: causal scrubbing** (Chan et al., Redwood Research, 2022). For each position of each
    dev window, the site's activation in the model is replaced by its activation at a donor position: a randomly
    chosen (window, position) from `dev_full` with the same key value (excluding the position itself). The model is
    then run on as usual (residual sum, final LayerNorm, unembedding). If no donor shares the key value, the position
    keeps its own activation and is counted; if more than 1% of positions lack a donor, the claim fails. (The 1% rule
    also marks claims that need exact long contexts, which are close to lookup tables.)
  - **Replacement test, for value claims.** The site's activation is replaced by the account's computed value at every
    position, and the model is run on as usual. It needs no donors, so it scales with rung size.
  - **The score**, for both: faithfulness = 1 − KL(model || patched model) / KL(model || resample-ablated model), where
    the resample ablation replaces the same site with its activation at a donor drawn at random with no constraint.
  - **Context faithfulness** (decision 22), for every head claim:
    1 − KL(model || head patched) / KL(model || head replaced by its mean for the current symbol) ≥ 0.95.
  - **Pass:** faithfulness is at least **0.95** with all claims applied together, and at least 0.95 for each claim
    applied alone. Claims are scored at the granularity the account uses; a coarse summary claim does not replace the
    detailed claims under it. A value claim that reproduces the site by copying weights passes P3 but fails P2.
- **P4, behaviour** (decision 19).
  - Each claim carries at least one behaviour test: code in the account that measures a stated behaviour of the model
    (its attention, site activations or outputs on dev_full) and compares it with a stated threshold.
  - The code is the same for every seed and may read the account's named per-seed constants.
  - A test passes if it holds on all five trained seeds and fails on the rung's untrained model (init seed 0).
  - Every behavioural statement in the description has a test; the human checks the mapping at R3.
  - Decision 20: generic measurements live in analysis/ladder/behaviour.py; the account states each test in one line
    (statement, named measurement or ratio of two, comparison, threshold), and only that line counts toward R1.

### Reported measures

- **R1, description length (headline result).** The account's size in bits: its source code with comments and
  docstrings removed (Python `tokenize`), compressed with `lzma` at preset 9, times 8. Numeric constants count at the
  precision they are written. Reported as a curve of KL against bits for every committed version of the account.
  - **Reference point:** the weights themselves, quantised uniformly per tensor to the fewest bits per weight that
    keep KL(model || quantised model) at most the P1 bar max(KL_floor, 0.01) (decision 19; it was 0.01), then
    compressed the same way, plus the compressed reference forward pass. An account that is
    not shorter than this reference has not compressed the model, whatever else it scores.
  - This is the trade-off of Gross et al.'s compact proofs and ARC's surprise accounting.
- **R2, ARC-style mechanistic estimate.** Estimate the model's mean dev loss from the account and corpus n-gram
  statistics alone, with no forward pass of the model or the account on dev windows. Compare its squared error with
  that of a sampling estimate (the mean loss of the model on randomly drawn dev positions) at the same compute
  (counted in floating-point operations), over several compute levels. Reported: the compute at which sampling first
  becomes better, if any.
- **R3, human readability.** The human reads the account blind to the metrics and scores each component on the four
  AGENTS.md section 12 questions (0, 1 or 2). Each "what happens if it is removed" prediction is checked against the
  measured removal.
- Also reported per rung: dev loss, top-1 agreement with the model, the bigram and trigram baselines, and the seed
  floor.

## Tools (step 2b, analysis/ladder/)

- `src/train/data_char.py`: the alphabet, the streams and the eval sets, reusing src/train/data.py.
- `analysis/ladder/eval.py`: KL, top-1, coverage, seed floor and baselines for any rung (P1, P2).
- `analysis/ladder/scrub.py`: causal scrubbing of an account's claims (P3).
- `analysis/ladder/length.py`: description length of an account and of the quantised weights (R1).
  - It reuses the harness's functions by import, but not its fixed vocabulary of 2,000.
  - Check: on L10 it must reproduce the harness's KL of program v1.3 against attn_s0 (0.5497).
- `analysis/ladder/dump.py`: per-rung tables (QK, OV, positions, the direct path) as readable arrays and plots.
- Higher rungs reuse:
  - the law tools (analysis/look_write);
  - the tables-versus-attention factorial (analysis/token_geometry/tables_vs_attention.py);
  - the floor code (analysis/program_gap/coverage.py).
- Sanity tests per rung (src/train/test_sanity_ladder.py): causality, the bigram beaten, the `noln` arm's identity, parameter counts written to
  logs/ladder/param_counts.json.

## Gate decisions (2026-11-06)

1. **LayerNorm:** both arms at L1-L3 with the decision rule above (the human's proposal).
2. **ARC-style estimate:** reported only (R2), in ARC's relative form (against sampling at equal compute), not the
   draft's "within 1% of the measured loss", which had no reference point.
3. **Criteria:** the draft's human-readability pass condition (C3) is replaced by causal scrubbing (P3), so every pass
   condition is measurable by others; human readability is reported (R3); description length is added as the headline
   measure (R1). Thresholds 0.01 nats per symbol and 0.95 kept. (Decision 18 later replaced P1's fixed 0.01 by the
   seed floor and made coverage a reported measure.)
4. **Rung list:** approved as drafted.

## Compute

- Training: the LayerNorm arms double L1-L3 (30 extra runs, minutes each). L1-L7 take minutes to tens of minutes per seed on MPS; L8-L9 under an hour per seed.
- Data preparation: about 30 min on the CPU.
- Disk: about 2 GB for the streams; checkpoints are small, keeping final ones only.
