# Replacement program for `attn_s0`

A NumPy program that predicts the next-token distribution of `attn_s0`, a one-block, attention-only
TinyStories language model (4 heads, vocabulary of 2000 tokens), without using the model's weights.
Entry point: `predict(tokens)` in `__init__.py`. Row `i` of its output is the distribution after
`tokens[: i + 1]`.

Fidelity (harness, KL(model ‖ program) in nats per token): **0.452** on dev_fast (200 windows); the
program's top token is the model's top token 70% of the time. This version (v11) has not yet been measured
on dev_full or on text sampled from the model (v10: 0.463 on dev_full, 0.488 on model samples).

## The algorithm, top-down

At every position `i` the program gives each of the 2000 possible next tokens `v` a score, adds the
votes of five parts, and turns the scores into probabilities with a softmax:

```
score_i[v] = FOLLOW(t_i, v)                                   follow.py
           + Σ_h Σ_{j < i} a^h_ij · RECALL^h(t_j, v)            head0.py … head3.py (machinery: attention.py)
```

1. **FOLLOW** — what usually comes right after the current token, ignoring everything earlier:
   `WORD[v] + CLASS_FOLLOW[context(t)][target(v)] + AFTER exceptions(t, v)`.
2. **Four heads.** Head `h` spreads its attention over the positions `j ≤ i` with
   `a^h_ij ∝ exp(SALIENCE^h[t_j] + PREFER^h[attention(t_i)][attention(t_j)] + RECENCY^h[bucket(i − j)])`.
   Attention on the current token (`j = i`) is an **abstention**: it takes weight away from earlier tokens
   but casts no vote (everything the current token says is already in FOLLOW). Each attended earlier token
   `s` votes with `RECALL^h(s, v) = CLASS_RECALL^h[context(s)][target(v)] + COPY^h[context(s)]·[v = s] +
   RECALL exceptions(s, v)`.
3. Add and normalise (`log_softmax`). Nothing else happens.

`context(t)`, `target(v)` and `attention(t)` are a token's class in the three class systems below; each table
file in `tables/` names the systems it uses.

## What each part does (numbers measured on the program, 200 held-out windows; see each module's CLAIM)

| part | one-sentence role | removal cost* | top-1 after removal | model's matching head** |
|---|---|---|---|---|
| FOLLOW (`follow.py`) | the guess from the current token alone; alone it is 1.39 nats from the model | exceptions: +0.89 | — | heads 0 and 2 attending to the current token, plus the direct path |
| head 0 | passes on the last few tokens; matters most inside words split into pieces | +0.15 | 69% → 63% | carries 33% of model head 0's context effect |
| head 1 | recalls words from far back (mean distance 34) and votes to repeat them | +0.18 | at recall positions 60% → 38% | 44% of model head 1's |
| head 2 | completes phrases from the last few words; opens a quote after "said," | +0.19 | 69% → 60% | 39% of model head 2's |
| head 3 | carries a phrase across linking words such as " and", " of" and " to" | +0.21 | 69% → 58% | 64% of model head 3's |

\* Increase of KL(model ‖ program) when the part's votes are replaced by their average (for FOLLOW: when its
token-pair exceptions are; for its class table CLASS_FOLLOW the increase is 0.75). Heads 2 and 3 overlap, so
single-head costs do not add up. All four heads together remove 0.94 of FOLLOW alone's 1.39 nats.
\*\* Share of the KL change from ablating only the earlier-token part of the model head (harness
`claims.model_logprobs(..., ablation="earlier")`); details in `logs/consistency.md`.

## Vocabulary and word classes

- `vocab.py`: the 2000 token strings of the tokenizer (byte-level BPE trained on the TinyStories train
  split). `'<0xE2>'`-style names are fragments of multi-byte characters.
- `categories.py`: three systems of named word classes (at most 64 tokens each), all found from the
  training text alone (no model involved) and named by hand after reading the members:
  - `CONTEXT_CLASSES` (71) group tokens by what tends to come *after* them, e.g. `TO_AND_MODALS`
    (` to`, ` can`, ` could`: a bare verb follows). A token's CONTEXT class picks the row of `CLASS_FOLLOW`
    and `CLASS_RECALL`, and its `COPY` entry.
  - `TARGET_CLASSES` (64) group tokens by what tends to come *before* them, e.g. `BARE_VERBS_1`
    (` play`, ` be`, ` go`). The next token's TARGET class picks the column.
  - CONTEXT and TARGET classes were found together by information-theoretic co-clustering of the training
    bigram counts (`analysis/cocluster.py`); splitting the two roles was worth 0.013 nats of KL.
  - `ATTENTION_CLASSES` (64), from clustering each token's neighbours on both sides (`analysis/taxonomy.py`),
    are used by the heads' `PREFER` tables. Some are behavioural groups of word pieces (`FRAGMENTS_*`).

## How to read the tables (`tables/`)

- Values are in nats (natural-log units) and are added to scores before the softmax: +0.7 doubles a
  token's odds, +2.3 multiplies them by 10. A missing entry means 0.
- Per-token tables are grouped by word class, class tables have one row per class, and token-pair
  exceptions are written `pair(TABLE, first token, CLASS_OF_SECOND={second token: value})`, so no literal
  holds more than 64 numbers.
- Exception rows are in order of importance (estimated KL cost of removing the row), boosts before
  suppressions. Each FOLLOW row carries a comment with the top probabilities FOLLOW alone gives after that
  token, e.g. `# after '<|eos|>' (FOLLOW alone): 'Once' 35%, 'One' 13%, 'Lily' 11%`.
- Head votes act through attention weights. Each head file states its abstention share and the typical
  weight of an attended earlier token; for head 1 that weight is 0.04, so its large-looking COPY and
  RECALL values (10-20 nats) add about 0.4-0.8 nats where they apply.

## How the tables were found

All numbers were fitted by regression to the model's next-token log-probabilities on training-split text
(never copied from weights; insight from the weights only guided the structure): a dense fit of the
structure above (analysis/student.py), heads silenced on the current token, a lossless re-centring of the
tables, 16 rounds of gradual pruning to 60,000 token-pair exceptions ranked by their second-order KL cost,
a final refit, and rounding to one decimal (analysis/prune.py, analysis/emit_program.py). The fitting
history, dead ends and the fidelity-vs-size curve are in `logs/compression_experiments.md` and
`logs/dead_ends.md`.

## Fidelity against size

| token-pair exceptions | 10k | 17k | 30k | 51k | **60k (this program)** | 158k | 256k |
|---|---|---|---|---|---|---|---|
| dev_fast KL | 0.812 | 0.703 | 0.593 | 0.506 | **0.452** | 0.371 | 0.328 |

Only the 60k point is this version (v11). The other points come from earlier runs (v9 recipe, one class
system for rows and columns); v10, this program with that single class system, had 0.465 at 60k.

The class tables and attention tables cost about 47k numbers on top of the exceptions (107k numbers in
total at 60k exceptions).

## Known gaps

- The program heads carry 33-64% of the model heads' context effect; most of the remaining KL is missing
  context, worst for head 0.
- Some work sits in a different head than in the model: inside split words the model leans most on its
  head 2, the program on head 0; ' eyes' after 'He closed his' comes from model head 2 but program head 3.
- Class-level votes cannot tell `' She'` from `' He'` or one name from another unless an exception says so.
- Model heads 1 and 3 rarely look at the current token (4%, 19%), but the program's heads 1 and 3 abstain
  on 28% and 30% of their attention.
