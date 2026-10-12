# Surprises

Record anything unexpected the moment it appears. Newest at the bottom.

## 2026-10-12 — Data format (phase 0/2)

- The parquet and `.txt` versions of the original TinyStories differ in whitespace. The `.txt`
  (stories separated by `<|endoftext|>`) collapses paragraph breaks `\n\n` to `\n` and carries a
  leading space/newline per story. The parquet keeps `\n\n`. We use the parquet as the source.
- The corpus contains cp1252 mojibake of UTF-8 punctuation: in the validation split, `â€` occurs
  7,144 times (`â€œ`, `â€™`, ...), against 108 correct `“`. AGENTS.md says keep raw casing and
  punctuation, so we do not repair it. The byte-level BPE will spend some vocab on these bytes.
- The first validation story starts mid-sentence ("Spot. Spot saw the shiny car..."). Some stories
  are truncated. Not repaired.
- `huggingface_hub.snapshot_download` stalled on the train parquet files; plain `curl` on the
  `resolve/main` URLs worked.

## 2026-10-12 — Duplicate stories (phase 2)

- 6,213 rows of train-minus-dev are byte-identical to some dev story, and 6,553 are identical to some
  test (official validation) story (`logs/duplicates.json`). Dev rows are excluded by index, so the
  tokenizer and model never see the dev *rows*, but they do see identical text from other rows.
  Fidelity is measured against the model, so this mostly affects data-loss numbers (model and
  n-gram cross-entropy on dev/test are slightly optimistic), not KL.
- `\n\n` paragraph breaks tokenize as two separate `\n` tokens (the GPT-2 byte-level regex splits
  a whitespace run before a letter), so the BPE never merges them.

## 2026-10-12 — KL direction vs cross-entropy (phase 1 self-test)

- On a 10M-token sanity model, the bigram baseline is only 0.09 nats worse than the model in data
  cross-entropy, yet KL(model || bigram) = 1.19 nats. The bigram's Dirichlet smoothing gives
  unseen pairs ~1e-10, and the (undertrained, diffuse) model puts mass there. KL(model || program)
  punishes any token the program rules out that the model does not. Programs need a floor.

## 2026-10-12 — Throughput

- On the M4, `torch.set_num_threads(10)` cuts CPU training throughput from 137k to 37k tokens/s
  (efficiency cores). The torch default (4 threads) is best. MPS gives ~250k tokens/s, which makes the
  whole pilot plus 10 final runs feasible on the laptop in a few hours.

## 2026-10-12 — Pilot LR check

- Dev loss falls steeply with LR at 12.5M tokens: 3.670 (5e-4), 3.233 (1e-3), 2.917 (2e-3). The best LR is
  at the top edge of the AGENTS.md grid, so the true optimum is probably higher. Protocol followed
  (2e-3 chosen); one extra explore point at 4e-3 is run after the ladder for information only.
- The first two LR runs took 352 s and 702 s, the third 74 s (same config, same device). The slow ones
  ran while the machine was otherwise idle waiting on a human answer; likely macOS throttling/App Nap
  of a background process. Wall time is not a result here, but throughput numbers can be unreliable.

## 2026-10-12 — Vocabulary structure (pre-phase 5, data only)

- 106 of the 2000 tokens never occur in the train stream (byte tokens fully absorbed by BPE merges).
  The model can only learn to push their logits down; the program can treat them as one category.
- Ward clustering of train-bigram contexts gives clean word classes (prepositions, modals, past-tense
  verbs, names, determiners, emotions, colours, word-onset fragments, byte fragments).
- Class structure alone explains under half of the bigram: KL(data bigram || class->class) = 1.38 nats
  vs 2.47 for unigram; adding the top-32 exact continuations per token brings it to 0.22-0.26.

## 2026-10-12 — attn_s0 first look (phase 5)

- Zero-ablating one head of attn_s0 costs up to 3.9 nats (head 0) and all four heads 35.6 nats of KL.
  Head outputs have norms 3.5-7.6 against 2.2 for the token embedding, so the final LayerNorm sees a
  completely different vector. Zero ablation is meaningless here; the consistency check needs mean
  ablation. The harness (built in phase 1) only does zero ablation. Raised with the human.
- Heads 0 and 2 put ~50% / ~34% of attention on the current token itself: most of the model's "bigram"
  goes through these heads, not through the direct embedding path.
- Final LayerNorm scale varies 0.57-2.13 across positions (mean 1.13, sd 0.24): a ~4x range in effective
  temperature that an additive program does not have.
- Harness self-test with the model wrapper failed the causality check on attn_s0: float32 log-probs for a
  5-token prefix differ from the full-window run by 2.9e-5 (> 1e-5 tolerance) because of different kernel
  shapes. In float64 the difference is 2.5e-14. Fixed the self-test fixture to run in float64; scorer
  unchanged.
