# Phases 0–2 report: freeze, harness and baselines, training code

Date: 2026-10-12. Machine: Apple M4 (10 cores, 16 GB), macOS, Python 3.12, torch 2.14.1, numpy 2.5.4,
tokenizers 0.23.3.

## Assumptions (read first)

1. **Data source = parquet.** `roneneldan/TinyStories` `data/*.parquet` (train 2,119,719 rows,
   validation 21,990 rows). The `.txt` files differ in whitespace (they collapse `\n\n` to `\n`); the
   parquet keeps paragraph breaks. Recorded in `configs/frozen.yaml`.
2. **Tokenizer = byte-level BPE** (HF `tokenizers`, GPT-2 regex pre-tokenizer, no prefix space), 2000
   ids: id 0 `<|eos|>`, 256 byte symbols, 1,743 merges. Byte-level means no unknown tokens.
3. **Raw text kept**, including cp1252 mojibake (`â€™` etc., ~7k occurrences per 22k stories) and
   truncated stories. See `logs/surprises.md`.
4. **Windows** are 257 tokens with stride 256 over each split's stream, so every token is a target
   exactly once. Story order in each stream is shuffled once (fixed seeds 1/2/3 for train/dev/test);
   each model's data order is a per-epoch window permutation from `default_rng([seed, epoch])`.
5. **dev_fast = first 200 windows of dev_full** (dev_full is a seed-0 random draw of 2000 dev windows).
   An extra held-out set `claims` (200 dev windows outside dev_full) feeds the claim tests.
6. **Head ablation = zero ablation** (head output zeroed before the output projection). The section 9
   consistency check uses this; mean ablation would be a reasonable alternative.
7. **Weight decay** applies to every `nn.Linear` weight (qkv, attention out, MLP in/out), not to
   embeddings or LayerNorm gains.
8. **Checkpoints** at 20 log-spaced steps from step 16 to the end, plus `final.pt`; `latest.pt` holds full
   resume state and is refreshed every 5 minutes.

## Phase 0 — freeze

`configs/frozen.yaml` mirrors section 2 plus the assumptions above. Training budget is `null` until the
pilot sets it.

## Phase 1 — harness and baselines

- `harness/eval_fidelity.py`: lint → sandboxed subprocess → causality/validity checks → KL(model ‖ program),
  top-1, cross-entropies, loss recovered → claim tests → `logs/iterations.csv` (+ `test_looks.csv`, budget 10
  per model, `--note` required). Rows carry the git commit, a dirty flag and a hash of the program files.
- Sandbox: macOS `sandbox-exec` denies reads of `models/`, `data/`, `logs/`, `analysis/` and all network
  access for the program process. Verified by `harness/test_harness.py` (reads fail with "Operation not
  permitted"). On other OSes the harness prints that it cannot enforce this.
- `harness/lint_program.py`: stdlib allowlist + numpy; bans `open/eval/exec/compile/__import__/getattr/...`,
  `np.load` and friends, dunder attribute access, literals with > 64 numeric elements (nested count),
  long strings and long unbroken runs (base64), long bytes.
- Baselines (train split only): unigram (add-one), bigram (Dirichlet prior on unigram, α = 1), trigram
  (Witten-Bell over bigram, counts from the first 100M train tokens). They load tables from
  `data/baselines/`, so they run with lint and sandbox off. They are references, not replacement programs.

Self-test against a 10M-token `mlp` sanity model (`models/sanity/short_run_mlp`), dev_fast, unlogged:

| program | KL(model ‖ program) | top-1 | CE vs data |
|---|---|---|---|
| uniform | 3.854 | 0.005 | 7.601 |
| unigram | 2.267 | 0.159 | 5.990 |
| bigram | 1.189 | 0.581 | 3.528 |
| trigram | 2.252 | 0.482 | 2.741 |
| model wrapper (self-test, lint/sandbox off) | 0.000 | 1.000 | 3.436 |

Uniform is large, the wrapper is zero: the harness passes its self-test. The trigram beats the model in
data CE (the sanity model is undertrained, and dev has exact-duplicate stories in train) but is far from it
in KL.

## Phase 2 — training code and sanity tests

`src/train/{model,data,train,runs,throughput}.py`. `python -m pytest -q src/train/test_sanity.py`: **15 passed**.

1. Causality: perturbing tokens after t leaves logits ≤ t unchanged (both variants, SDPA and explicit paths).
2. Single-batch overfit (4 windows, 600 steps, LR 3e-3): loss 0.0008 (`attn`), 0.0006 (`mlp`).
3. Short run (`mlp`, 10M tokens, MPS, 40 s): dev_full loss **3.401** vs bigram **3.516** vs unigram **5.984**.
4. Tokenizer fit on 2,099,719 train-minus-dev stories; dev indices reproduce from seed 0; split disjoint.
5. Parameters: `attn` **354,560**, `mlp` **485,760**, equal to hand arithmetic (`logs/param_counts.json`).
   Tokens: train **527,489,811** (2,060,507 windows), dev 5,031,401, test 5,356,523. EOS count equals story
   count in every stream; tokenizer round-trips text exactly.
6. Extra: batches depend only on (seed, step); a run interrupted at step 20 of 40 and resumed gives
   bit-identical final weights to an uninterrupted run (CPU).

## Open items measured (section 14)

- One epoch = **527.5M tokens**, so the pilot ladder's 800M rung is capped at one epoch (527.5M).
- Throughput: see `logs/throughput.md`.
