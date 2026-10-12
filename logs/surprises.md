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
