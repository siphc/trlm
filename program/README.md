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
   `FOLLOW(t, v) = WORD[v] + CLASS_FOLLOW[context(t)][target(v)] + AFTER exceptions(t, v)`.
2. **Four attention heads** (`head0.py` … `head3.py`, machinery in `attention.py`). Head `h` spreads its
   attention over the positions `j ≤ i`:
   `a^h_ij ∝ exp(SALIENCE^h[t_j] + PREFER^h[attention(t_i)][attention(t_j)] + RECENCY^h[bucket(i − j)])`,
   and every attended token `s` votes for next tokens with
   `RECALL^h(s, v) = CLASS_RECALL^h[context(s)][target(v)] + COPY^h[context(s)]·[v = s] + RECALL exceptions(s, v)`.
3. The scores are added and normalised (`log_softmax`).

`context(t)`, `target(v)` and `attention(t)` are a token's class in the three class systems below; each table
file in `tables/` names the systems it uses.

## Vocabulary and word classes

- `vocab.py`: the 2000 token strings of the tokenizer (byte-level BPE trained on the TinyStories train split).
- `categories.py`: three systems of named word classes (≤ 64 tokens each), all found from the training text
  alone (no model involved) and named by hand after reading the members:
  - `CONTEXT_CLASSES` (71) group tokens by what tends to come *after* them, e.g. `TO_AND_MODALS`
    (` to`, ` can`, ` could`: a bare verb follows). A token's CONTEXT class picks the row of `CLASS_FOLLOW`
    and `CLASS_RECALL`, and its `COPY` entry.
  - `TARGET_CLASSES` (64) group tokens by what tends to come *before* them, e.g. `BARE_VERBS_1`
    (` play`, ` be`, ` go`). The next token's TARGET class picks the column.
  - CONTEXT and TARGET classes were found together by information-theoretic co-clustering of the training
    bigram counts (`analysis/cocluster.py`).
  - `ATTENTION_CLASSES` (64), from clustering each token's neighbours on both sides (`analysis/taxonomy.py`),
    are used by the heads' `PREFER` tables.

## Tables

All numbers live in `tables/` and are written per class or per token so that no literal holds more than 64
numbers. Values are in nats and are added to scores. They were fitted by regression to the model's
next-token log-probabilities on training-split windows (`analysis/run_fit.py`), never copied from weights.

## Status

The current version and its measured fidelity are in the latest commit message and `logs/iterations.csv`.
The CLAIM docstrings in each module state what each component does, with numbers; `tests/` checks them.
