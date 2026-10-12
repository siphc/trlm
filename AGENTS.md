# AGENTS.md

Context file for coding agents working on this repository. Read all of it before writing code. Ask the human when blocked. Do not guess silently.

## 1. Mission

Train tiny language models, then write a human-readable Python program that reproduces each model's next-token distribution without using the model's weights.

Fidelity is measured against the model, never against the data. A human reads the program and judges readability. The program may be large. There is no size cap on code, but there is a cap on opaque numbers (section 8).

Two outputs matter besides the program itself:

1. A curve of fidelity against program complexity, built from the commit history.
2. A dead-end ledger (`logs/dead_ends.md`): every hypothesis that failed, with the evidence. Negative results are data here.

## 2. Frozen decisions

Do not change these without human approval. A change means a new version of `configs/frozen.yaml` and a full rerun.

| Item | Value |
|---|---|
| Dataset | Original TinyStories (not the V2-GPT4 variant) |
| Tokenizer | BPE, vocab 2000, trained on the training split only (dev slice excluded) |
| Context length | 256 |
| Blocks | 1 |
| d_model | 128 |
| Heads | 4 (d_head 32) |
| Variants | `attn` (attention only, no MLP) and `mlp` (MLP hidden 512, ReLU) |
| Positions | Learned absolute |
| Normalization | Pre-LayerNorm, final LayerNorm, no bias terms where the framework allows |
| Embeddings | Tied input and output |
| Seeds | 0 to 4 for each variant (10 models). Seed changes init and data order |
| Training budget | Set by the pilot in section 5, then identical for all 10 models |
| Hardware | Laptop CPU for development, Google Colab GPU for training runs |
| Library for training | PyTorch |
| Library for replacement program | Python stdlib and NumPy only |
| Stopping rule | Plateau: dev KL improves by less than 0.5% (relative) over 5 consecutive logged iterations |
| Opaque-numerics lint | Flag any array or list literal with more than 64 numeric elements |
| Tracks | Open and black-box, in separate agent sessions with no shared context |
| Black-box access | Logits only |

Parameter counts, from hand arithmetic (verify by counting in code and write the result to `logs/param_counts.json`): about 0.49M for `mlp`, about 0.35M for `attn`. Both sit below the original 1M target. Token embeddings (256k) are over half of the `mlp` model. Expect the program to spend much of its effort on embedding structure.

## 3. Data

Source: `roneneldan/TinyStories` on Hugging Face. The hub listing shows `TinyStories-train.txt` (about 1.9 GB) and `TinyStories-valid.txt` (about 19 MB), and a parquet version with train (about 2.12M rows) and validation (about 22k rows) splits. Check the actual format (story separators, encoding) before writing the loader.

Splits:

- **Train:** official train minus the dev slice.
- **Dev:** 20,000 stories sampled from official train with seed 0. Never used for tokenizer or model training. This is the only split used inside the discovery loop.
- **Test:** the official validation split. Touched rarely (section 9).

Preprocessing:

- Keep raw casing and punctuation.
- Append an end-of-story token (id 0) after each story.
- Concatenate shuffled stories into a stream and cut it into windows of 256 tokens. Attention crosses story boundaries. This is the simplest option, and the human may change it later.
- Pre-tokenize once into a `uint16` binary file per split. Vocab 2000 fits.
- Data order is a deterministic function of (seed, step). A resumed run must reproduce the same batches.
- Fixed evaluation sets, built once and stored: `dev_fast` (200 windows), `dev_full` (2000 windows), `test_full` (2000 windows), and `probe_samples` (2000 windows sampled from each model at temperature 1 with a fixed seed, stored per model).

## 4. Training code requirements

- Plain PyTorch. One config dataclass or YAML. All randomness seeded and logged.
- Initialization: normal with std 0.02, residual output projections scaled by 1/sqrt(2 * n_blocks).
- Defaults, tunable only through the pilot: AdamW, betas (0.9, 0.95), peak LR 1e-3, 1% linear warmup, cosine decay to 10% of peak, weight decay 0.1 (not on embeddings or LayerNorm), gradient clip 1.0, batch 64 windows (16,384 tokens), no dropout. These are starting guesses, not verified optima.
- Checkpoints: about 20 log-spaced in tokens seen, plus the final one. Training must be resumable from any checkpoint.
- Mixed precision (fp16 autocast) is allowed on GPU. Every fidelity measurement uses an fp32 copy of the final model. The fp32 model is the target.
- Colab: install dependencies at the top of the notebook, mount Google Drive, write checkpoints and logs to Drive. Assume sessions can disconnect.

Sanity tests that must pass before any long run (the human reviews this code):

1. Causality: changing a future token must not change earlier logits.
2. The model can overfit a single batch to near-zero loss.
3. Dev loss at the end of a short run beats unigram and bigram baselines computed from the training split.
4. The tokenizer was fit without dev or test text.
5. Token counts and parameter counts match expectations.

## 5. Pilot to set the token budget

Intermediate checkpoints of a long cosine run sit at a high learning rate, so they cannot stand in for a shorter run's endpoint. The pilot therefore uses several complete short runs.

1. LR check (3 runs at the smallest budget below): peak LR in {5e-4, 1e-3, 2e-3}. Pick the best dev loss.
2. Budget runs: `mlp` seed 0 at token budgets in a geometric ladder, for example 12.5M, 50M, 200M, 800M tokens, each with its own full cosine schedule. Cap at one epoch over the training split. Measure the real token count of one epoch first and adjust the ladder.
3. Choose the smallest budget whose final dev loss is within 1% of the best run. Write the choice and the full table to `logs/pilot.md`.
4. Apply that budget to all 10 models, including `attn`. If `attn` is still improving noticeably at that budget, record it but keep the budget equal for comparability.

Before the pilot, run a two-minute throughput test on CPU and on Colab. Report tokens per second and extrapolate the cost of each pilot run and of the 10 final runs.

## 6. Repository layout

```
AGENTS.md
configs/frozen.yaml        # mirrors section 2. Read-only after commit.
data/                      # gitignored. Token binaries, eval sets.
src/train/                 # tokenizer, model, training loop, sanity tests
models/                    # checkpoints. Not readable in the black-box track.
harness/                   # READ-ONLY for replacement agents
  eval_fidelity.py
  lint_program.py
  probes/
analysis/                  # open-track tools (ablations, logit decomposition, attention dumps)
blackbox/query_api.py      # black-box track: the only door to the model
program/                   # the replacement program. The agent writes here.
  README.md                # top-down description of the algorithm
  tests/                   # one claim test per component
logs/
  iterations.csv
  test_looks.csv
  dead_ends.md
  surprises.md
```

## 7. Program interface

`program/__init__.py` exposes:

```python
def predict(tokens: Sequence[int]) -> np.ndarray:
    """Return log-probabilities, shape (len(tokens), 2000).
    Row i is the next-token distribution after tokens[: i + 1]."""
```

The harness checks causality by perturbing future tokens. It also checks that every row is a valid log-distribution.

Structure the program hierarchically so a human can read it top-down:

- One top-level function that states the whole algorithm in order.
- One module per component: bigram path, each attention head, MLP rules if any.
- Each module opens with a `CLAIM` docstring that states a falsifiable prediction about its behavior.
- Each claim has a test in `program/tests/` that the harness runs on held-out data.
- Every named constant says what it is and how it was found.

## 8. Rules for the replacement program

The program must not use the model's weights.

- No file I/O at runtime. No `open`, `pathlib`, `os`, `np.load`, `pickle`, `json` file loads, `base64`, `zlib`, `eval`, `exec`.
- Imports limited to the stdlib and NumPy. No PyTorch.
- No large string literals that could carry encoded data.
- No embedded copies of weight matrices, and no rotated, projected, quantized or rescaled versions of them.
- No small neural network fitted to the model's outputs and pasted in as float literals. Learning a table by regression is allowed if the table is indexed by named categories and is under the lint limit.
- Any array or list literal with more than 64 numeric elements fails the lint. If a legitimate table needs more, split it by named category or ask the human to raise the limit.

Insight may come from the weights in the open track. Numbers copied from the weights may not.

The harness runs the program in a subprocess with `models/` unreadable where the OS allows it. If it cannot enforce this, it says so in its output.

## 9. Harness, metrics, logging

The harness is the only scorer. Replacement agents call it and never edit it.

Primary metric, on dev: mean per-token KL(model || program) in nats.

Secondary metrics:

- Top-1 agreement with the model.
- Loss recovered: (L_bigram - L_program) / (L_bigram - L_model), with cross-entropy against the data.
- KL on `probe_samples`.
- Lint status and counts: lines of code, numeric literals, largest literal.
- Claim-test pass rate.

Logging:

- Every iteration is a git commit. The commit message contains the metrics.
- `logs/iterations.csv` gets one row per harness run: commit hash, model id, split, all metrics.
- Test split: `--split test` requires `--note`, and appends to `logs/test_looks.csv`. Budget: 10 looks per model, total. Never show test data, or summaries of it, to the discovery loop.

Component consistency check (log it at each milestone): for each component, compare the KL change from removing it from the program with the KL change from ablating the matching head in the model. Disagreement points at a component that does not mean what its claim says.

## 10. Tracks

**Open track.** Full repository access, including `analysis/` tools and checkpoints. Insight from the weights is allowed. Section 8 still applies to the artifact.

**Black-box track.** A separate session in a separate repo copy. It sees only `blackbox/query_api.py` (`query_logits(batch_of_token_lists)`), the data, the harness and this file. It has no memory of the open track. It may query the model on any input, including synthetic ones.

After both tracks finish a model, report the gap in final fidelity and in program size. That gap measures how much the weights help beyond behavior.

## 11. Phases

Each phase ends with a short written report in `logs/` and a human go-ahead.

0. **Freeze.** `configs/frozen.yaml` written and reviewed.
1. **Harness and baselines.** Build `harness/`, then unigram, bigram and n-gram baselines as programs. Self-test: a uniform program gets a large KL, and a program that wraps the model directly (self-test only, lint off, never committed under `program/`) gets KL near zero.
2. **Training code and sanity tests** (section 4).
3. **Pilot** (section 5).
4. **Train 10 models.**
5. **Discovery loop**, starting with `attn` seed 0 as the dev model. Cycle: analysis, hypothesis, code, harness run, commit. Open-track analysis tools: per-head ablation loss deltas, direct-path logit contributions, attention pattern dumps, worst-KL tokens grouped into clusters. Run until the stopping rule fires or the human stops it.
6. **Refactor.** Improve structure and readability only. Gate: dev KL may not worsen by more than 2% relative to the pre-refactor commit. Fidelity work and readability work stay in separate commits.
7. **Human judging** at milestones (section 12).
8. **Seed transfer.** Freeze the program's structure from seed 0. For seeds 1 to 4, record exactly what had to change: constants, tables, rules, structure. Then repeat the workflow for `mlp`.
9. **Test evaluation**, logged.

## 12. Human judging

The human reads the program, blind to the metrics. Write for this reader. For each component the human answers:

1. Can I state in one sentence what it does?
2. Can I predict what happens to the output if it is removed?
3. Do names and comments match the behavior?
4. Is any table or constant opaque to me?

Each answer is scored 0, 1 or 2. Low scores go back to the agent as concrete rewrite requests.

## 13. Working rules for agents

- Verify library and API behavior in the documentation. Do not assume a function exists or that a default behaves a certain way. If unsure, say so in the code review notes.
- Report only numbers the harness produced. Include failed runs.
- Record anything unexpected in `logs/surprises.md` the moment it appears.
- Exploration budget: up to 10% of discovery iterations may chase a hypothesis outside the current plan. Label those commits `explore:` and log the outcome.
- Do not edit `harness/`, `configs/frozen.yaml`, or any evaluation set.
- State assumptions at the top of each report. If an assumption changes a frozen decision, stop and ask.

## 14. Open items to measure early

- Real token count of one epoch under the 2000-token BPE.
- Training throughput on CPU and Colab.
- Whether a 0.49M one-block model produces coherent stories. The TinyStories paper shows fluent text from models under 10M parameters, and from one-block models. It does not cover this size, so check by sampling.
- Whether the lint threshold of 64 is workable. It is arbitrary.
- Whether packed windows that cross story boundaries make the program harder to write.

## 15. References

- TinyStories paper: https://arxiv.org/abs/2305.07759
- TinyStories dataset: https://huggingface.co/datasets/roneneldan/TinyStories
- Compact proofs via mechanistic interpretability (Gross et al.): https://arxiv.org/abs/2406.11779
- A Mathematical Framework for Transformer Circuits (Elhage et al.): https://transformer-circuits.pub/2021/framework
- Training Compute-Optimal Large Language Models (Hoffmann et al.): https://arxiv.org/abs/2203.15556