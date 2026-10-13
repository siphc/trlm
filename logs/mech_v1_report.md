# Mechanism program v1 for attn_s0 (2026-10-13)

## Assumptions

- Corpus at startup: the harness passes the TRAIN token stream to `program.setup(corpus)` (one-time exception,
  `logs/decisions.md` item 7). Every token-specific number in the program is counted from it.
- Numbers fitted to the model are named scalars only: 28 attention-rule parameters fitted to each head's own
  attention rows, 28 vote strengths and 3 smoothing numbers fitted to the model's log-probs, 1 back-off weight
  chosen from a grid. No table is fitted to the model.
- Analysis numbers below come from TRAIN windows (`analysis.fitlib.train_windows(n, seed=2000)`); harness numbers
  come from dev_fast. The claim tests use the harness's held-out claim windows (first 32).

## Result

| | dev_fast KL | top-1 | loss recovered | numbers fitted to the model | lint |
|---|---|---|---|---|---|
| mechanism v1 (commit 2705fd1, tag `mechanism-v1`) | **0.5887** +- 0.0039 | 0.623 | 0.987 | 60 scalars | ok, largest literal 5 |
| distillation v11 (tag `distillation-v11`) | 0.4519 | 0.697 | 0.711* | ~107k table entries | ok |
| corpus bigram alone (FOLLOW refitted alone) | 1.196 | 0.372 | | 2 | |

\* loss recovered is not comparable across the two rows: v11's "LR 0.711" used the same definition, but v11's
cross-entropy against the data was 2.82 vs the model's 2.53; v1's is 2.547.

The NumPy program reproduces the reference implementation (analysis/scratch/spec_v1/v1probe.py, 0.5886) to the
fourth decimal. All corpus tables match the references (logs/verify_corpus.out). Corpus build: 104-112 s, peak RSS
4.6 GB.

## Ledger (64 TRAIN windows; KL with everything on 0.582)

| switched off | KL rise | SE |
|---|---|---|
| FOLLOW | +1.295 | 0.008 |
| PHRASE | +0.170 | 0.004 |
| LOCAL-0 (head 0) | +0.094 | 0.002 |
| RECALL (head 1) | +0.086 | 0.005 |
| LOCAL-2 (head 2) | +0.076 | 0.002 |

FOLLOW alone, with the strengths as fitted: 1.701.

## Component consistency check (AGENTS.md section 9)

For each head: replace its earlier-token part by its mean, in the model (harness `earlier` ablation) and in the
program (the part's earlier-token votes replaced by their mean over positions). Agreement is the median over
positions of the probability-weighted correlation of the two log-prob changes over the vocabulary
(`program/tests/measure.py`). 64 TRAIN windows:

| head | model shift (nats) | program shift | agreement | agreement with uniform attention |
|---|---|---|---|---|
| 0 | 0.464 | 0.043 | 0.27 | 0.03 |
| 1 | 0.409 | 0.079 | 0.64 | 0.55 |
| 2 | 0.493 | 0.056 | 0.36 | 0.08 |

The program's parts point the right way, and for heads 0 and 2 the attention rules are what make them point the right
way. But they move the output 5-10 times less than the heads do.

**How much of each head is explained** (analysis/head_effects.py, 32 TRAIN windows): the pooled weighted R^2 of the
model head's earlier-token effect, regressed on the program parts' votes:

| model head | shift | LOCAL-0 | LOCAL-2 | RECALL | PHRASE | all program parts |
|---|---|---|---|---|---|---|
| 0 | 0.461 | 0.035 | 0.045 | 0.004 | 0.040 | 0.071 |
| 2 | 0.471 | 0.074 | 0.104 | 0.000 | 0.062 | 0.116 |
| 0 and 2 together | 0.973 | 0.122 | 0.167 | 0.001 | 0.116 | 0.208 |
| 1 | 0.435 | 0.070 | 0.032 | 0.441 | 0.009 | 0.441 |
| 3 | 0.329 | 0.028 | 0.029 | 0.003 | 0.032 | 0.040 |

- RECALL is a fair account of head 1 (44% of its effect; 53% after sentence ends). The model does the same thing
  about 1.6 times more strongly (regression coefficient 1.62).
- The local readers explain only about a fifth of what heads 0 and 2 do from earlier tokens. Heads 0 and 2 are
  better explained together than apart, so the program's split of the work between LOCAL-0 and LOCAL-2 is not the
  model's. PHRASE adds 0.02.
- Head 3 is unexplained (4%).
- Where the unexplained shift sits (heads 0 and 2 together): ordinary positions 64% of the shift (71% of positions),
  word pieces 23% (13%), after sentence ends 11% (8%), word onsets 3% (7%).
- Not cancellation (analysis/cancellation.py): adding a "copy what the head reads" direction helps only head 0, and
  only a little (R^2 0.035 -> 0.060). See `logs/dead_ends.md`.

## Claim tests (held-out claim windows, 32; all 6 pass)

| claim | threshold | TRAIN | claim windows |
|---|---|---|---|
| program within 0.62 nats; each mechanism needed | KL <= 0.62, rises >= 0.05 (FOLLOW >= 1.0) | 0.582 | 0.588; rises 1.31 / 0.087 / 0.071 / 0.088 / 0.162 |
| FOLLOW is the context-blind model | KL to context-ablated model <= 1.35 and <= 0.8 x KL to the model | 1.15 vs 1.70 | 1.14 vs 1.69 |
| LOCAL-0 agrees with head 0; attention matters; captures a small part | agreement >= 0.18; uniform < half; shift < 1/4 | 0.27 / 0.03 | 0.27 / 0.04; 0.044 vs 0.467 |
| LOCAL-2 agrees with head 2 (same form) | agreement >= 0.25 | 0.36 / 0.08 | 0.35 / 0.08; 0.057 vs 0.483 |
| RECALL agrees with head 1 | agreement >= 0.5, uniform lower | 0.64 / 0.55 | 0.65 / 0.53 |
| PHRASE stands in for heads 0 and 2 | agreement with h0+h2 >= 0.3 and >= 2 x with h1 | 0.40 / 0.08 | 0.39 / 0.09 |

Attention-pattern claims (for example "at an end-of-story token head 0 looks at itself with weight >= 0.9") need the
model's attention patterns in the claim tests, which `harness/claims.py` does not expose. That is a harness change,
so it needs the human.

## What this says about how the model works (so far)

1. The model's guess from the current token alone is the corpus bigram, slightly smoothed toward the word's class.
   In the model it is the sum of the direct path and of heads 0 and 2 looking at the current token.
2. Heads 0 and 2 read the last few tokens. Their attention follows simple rules: head 0 uses recency by token
   spelling; head 2 uses informativeness, sentence ends, and skipping word pieces. What they write is related to
   "what follows that token at that distance" (skip-gram PMI), but this explains only about a fifth of it.
3. Head 1 recalls earlier names and content words with a slow decay, and favours words of the kind that comes
   next. It votes to repeat them and for their topic. This is the best-understood head.
4. Head 3 (clause context) is not understood.

## Next

The largest unexplained mass is what heads 0 and 2 write from earlier tokens (0.97 nats of shift, 79%
unexplained), then head 3 (0.33, 96%), then the rest of head 1 (0.44, 56%). Hypotheses to test:
conditional skip-grams (what an earlier token predicts beyond what the current token already predicts),
word assembly inside split words, query-conditional votes for head 3 after linkers, and head-1 strength and
end-of-story terms.
