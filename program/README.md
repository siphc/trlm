# Replacement program for `attn_s0`

A NumPy program that predicts the next-token distribution of a one-block, attention-only TinyStories
language model without using the model's weights. Entry point: `predict(tokens)` in `__init__.py`.

## The algorithm, top-down

For a text of tokens `t_0 … t_n`, the program computes, at every position `i`, a score for each of the
2000 possible next tokens `v`, and turns the scores into probabilities with a softmax:

```
score_i[v] = FOLLOW(t_i, v)                                  follow.py
           + Σ_h Σ_{j ≤ i} a^h_ij · RECALL^h(t_j, v)           head0.py … head3.py (attention.py)
```

1. **FOLLOW** (`follow.py`): what usually comes right after the current token, ignoring everything earlier.
   `FOLLOW(t, v) = WORD[v] + CLASS_FOLLOW[class(t)][class(v)] + AFTER exceptions(t, v)`.
2. **Four attention heads** (`head0.py` … `head3.py`, machinery in `attention.py`). Head `h` spreads its
   attention over the positions `j ≤ i`:
   `a^h_ij ∝ exp(SALIENCE^h[t_j] + PREFER^h[class(t_i)][class(t_j)] + RECENCY^h[bucket(i − j)])`,
   and every attended token `s` votes for next tokens with
   `RECALL^h(s, v) = CLASS_RECALL^h[class(s)][class(v)] + COPY^h[class(s)]·[v = s] + RECALL exceptions(s, v)`.
3. The scores are added and normalised (`log_softmax`).

## Vocabulary and word classes

- `vocab.py`: the 2000 token strings of the tokenizer (byte-level BPE trained on the TinyStories train split).
- `categories.py`: 64 named word classes (≤ 64 tokens each), found by clustering each token's neighbours in
  the training text (no model involved), then named by hand: e.g. `NAMES`, `DETERMINERS`, `PAST_VERBS_MOTION`.

## Tables

All numbers live in `tables/` and are written per class or per token so that no literal holds more than 64
numbers. Values are in nats and are added to scores. They were fitted by regression to the model's
next-token log-probabilities on training-split windows (`analysis/run_fit.py`), never copied from weights.

## Status

The current version and its measured fidelity are in the latest commit message and `logs/iterations.csv`.
The CLAIM docstrings in each module state what each component does, with numbers; `tests/` checks them.
