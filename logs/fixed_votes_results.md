# Results: fixed-vote models (study A; development and exploratory only)

Written 2026-11-06. **No verdicts.** At gate G2 the human decided not to run the confirmatory runs (seeds 1-4;
logs/decisions.md item 13), so H1-H3 of logs/prereg_fixed_votes.md are never scored. Everything below is seed 0
(development) or exploratory analysis written after the FINAL. Details and every number: logs/fixed_votes.md.

**Assumptions.** One seed per arm. The `base` model votes with four fixed corpus tables (next, copy, topic, skip2);
`shift` replaces skip2 by skip3. Only attention, the bigram strength and four vote strengths are learned. Peak LR
4e-3 (rule R1), one epoch, laptop GPU. All measurements on TRAIN windows or dev; no test-split looks.

## What was found

1. **Cost.** Fixing the votes costs 1.3% CE (base, dev_full 2.543 vs attn_s0 2.510); shift costs 3.1% more.
2. **Attention follows the table it serves.** Each head's attention-by-distance matches its own table's usefulness
   profile. The causal test from the design worked at seed 0: changing head 3's table from skip2 to skip3 moved its
   attention peak from d = 1 (mass 0.53) to d = 2 (0.39), and its own-usefulness arm went from the skip2 table's to the
   skip3 table's (0.0269 vs 0.0010).
3. **The law explains a minority of that attention.** Own usefulness recovers 10-45% of the output gain of the
   token-token ceiling (normal models: at least 50% in 7 of 8 local heads). The pure-pair correlation is as strong as
   in normal models for the local heads (0.36-0.48).
4. **Explaining away does not change that much** (exploratory). Measuring usefulness against what the other heads and
   the bigram already predict helps only the d = 1 head (next: 0.24 -> 0.50 of the ceiling in base, 0.45 -> 0.58 in
   shift). For the other heads it matches the bigram baseline within 0.001. Measured against the full model's
   prediction, the law nearly vanishes, as expected at a fixed point.
5. **No readable model by this route** (exploratory). Replacing all four heads' attention by readable rules at once
   gives a program 0.27 nats from the fixed-vote model and 0.77-0.82 from attn_s0 (program v1.3: 0.55), at CE +11% and +13% against attn_s0 on the same windows.
   Recorded in logs/dead_ends.md.
6. **The fixed-vote model is a different model, not an approximation of attn_s0**: KL(attn_s0 || base_s0) 0.64,
   against 0.42 for another ordinary seed.

## What this means

Fixed votes make the write side of every head readable, and attention then provably adapts to the table (finding 2).
But even with the write side known, a usefulness rule computed from corpus counts describes less than half of what
the learned attention adds, whichever baseline it uses. The gap between the law and the token-token ceiling is not
an artefact of the bigram baseline. One seed only; none of this is confirmed.

## Files

analysis/out/fixed_votes/{base_s0,shift_s0}/{law.json, readable_all.json, explain_away.json}; models/fixed_votes/
(gitignored); code analysis/fixed_votes/.
