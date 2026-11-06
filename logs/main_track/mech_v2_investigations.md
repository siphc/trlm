# Discovery round 2 on attn_s0: what the heads do that mechanism v1 misses (2026-10-13)

## Assumptions

- Starting point: mechanism program v1 (tag `mechanism-v1`, dev_fast 0.5887). Corpus statistics come from the
  training stream at startup; only named scalars are fitted to the model.
- Five investigators ran in parallel (workflow `mech-v2-investigations`); every candidate recommended for adoption was
  re-run by an independent verifier on fresh windows (fit seed 2031, eval seed 2032, 96 windows).
- Evaluation tool: `analysis/vote_fit.py`. All vote strengths are refitted with and without the candidate; the
  comparison is paired on held-out TRAIN windows; the R^2 of the claimed heads' earlier-token effect is also reported.
- Numbers below come from the investigators' notes (`analysis/investigations/<prefix>/NOTES.md`) and the vote_fit
  JSON files (`analysis/out/candidates/*.json`). Some analyses used model weights for insight (allowed in the open
  track); none of those numbers enter any candidate. Weight-derived caches live only in the gitignored
  `analysis/out/candidates/<prefix>/`.

## Outcome

| candidate | heads | paired KL gain (proposer) | fresh windows (verifier) | head R^2 | verdict |
|---|---|---|---|---|---|
| ov02_earlier_votes4 | 0, 2 | -0.0102 +- 0.0006 | **-0.0094 +- 0.0005** | 0.219 -> 0.256 | **confirmed**; to integrate |
| recall_attn_pmi1_votes_b25 | 1 | -0.0069 +- 0.0006 | -0.0040 +- 0.0007 | 0.422 -> 0.391 | refuted (below 0.005) |
| word, head3, qk candidates (19) | | best -0.0058 +- 0.0016 | not sent | | none reached the bar |

**ov02_earlier_votes4** (4 strengths; the WPMI_3 strength fits near zero and can go, leaving 3). Under the program's
own attention, earlier tokens only:
- LOCAL-0: an earlier token votes for its own usual successors (PMI_1), and against words that fill the same slot
  (KPRED(s, v) = sum_a sqrt(P(prev = a | s) P(prev = a | v)), diagonal zero).
- LOCAL-2: an earlier token votes against being repeated (one-hot).
- LOCAL-2: word-level skip-gram WPMI_3 (strength about 0.035; to drop).

The verifier found no rule violations, confirmed causality, and timed the features at 0.007 s per window. One
documentation fix: the unit boundary for WPMI is "shape != continuation", not "non-piece token" (88 capitalised
no-space tokens join the previous unit).

## What we learned about how the model works

1. **Heads 0, 2 and 3 are each about 90% "read a token, write a fixed vector"** (additive; the final LayerNorm adds
   about 5%). Each head writes the same vector for a token whether it is the current token or an earlier one (the
   value/output map cannot see distance). Head 0 spends half its attention on the current token, where its job is
   the bigram, so from an earlier token it mostly writes that token's usual successors, plus a veto on the token and
   on words like it. That is the structure ov02_earlier_votes4 encodes.
2. **What the local heads write is a learned 32-dimensional code, not a corpus statistic.** With the model's real
   attention, the best mix of corpus tables explains 31% of heads 0+2's effect, against 84% for the exact
   value/output map. The map is high-rank: rank 8 gives 37%, rank 32 gives 84%. Only 43-51% of a PMI row lies in a
   head's 32-dimensional output span; projecting PMI onto that span raises its correlation with the map from 0.41 to
   0.56. Distance-matched, conditional, word-level, class-level and low-rank skip-grams were all tried.
3. **Heads 0, 2 and 3 are one local-reading team.** Their explained parts are almost collinear (correlation 0.92 for
   heads 0 and 2), and their unexplained parts partly cancel (-0.26; head 3 vs heads 0+2: -0.19, -0.36 at noun
   queries). The effects add almost exactly (non-additivity under 2%). The program explains the joint effect of heads
   {0, 2, 3} at R^2 0.289, against 0.211 for {0, 2} and 0.046 for head 3 alone. Head 3 is not a separate mechanism:
   it reads a word's first piece from inside the word (' fr' at 0.93 from 'frustr', then votes 'ated' +10), the word
   before a linker ("Mom and" -> ' Dad' +8.2, ' he' -7.6), or the word before a linking verb. What it adds is phrase
   knowledge that PHRASE's trigram and 4-gram counts already give the program.
4. **Where a local head looks follows from what it writes.** Beyond position and shape, heads 0, 2 and 3 read an
   earlier token when their own vote from that token fits what follows better than the current token's prediction
   does ("own-vote usefulness"). In the exact model, a 2-scalar usefulness term beats the exact token-token
   query-key term in output KL (head 0: 0.266 vs 0.318; head 2: 0.222 vs 0.264; head 3: 0.229 vs 0.226), with about
   0.3 per nat for every head and distance. Another head's vote predicts nothing (-0.11). This is the shape gradient
   training gives a query-key circuit. Head 2's existing rule (SELF_INFO, KEY_INFO) is already this rule with the
   unigram as baseline. So the local heads' attention is not an independent mystery: fix the votes and the
   attention comes with them.
5. **The bottleneck is the votes, not the attention.** With skip-gram votes, even the model's exact attention
   patterns gain only 0.0063 nats for heads 0+2, although they raise head 2's explained share from 0.12 to 0.21.
6. **Head 1 (RECALL) is selective by token.** Its gap splits evenly between attention and votes (each alone: R^2
   about 0.59 with the other exact; both from RECALL: 0.44-0.46). The "model is 1.6x stronger" puzzle is attention
   diffuseness: with head 1's real attention, RECALL's own votes fit at coefficient 0.92. The rule's rows are too flat
   (entropy 3.96 vs 3.36 nats), although its mass by key group and by distance matches. The missing part is a rank-32
   query-key preference for the kind of word that comes next. Bigram statistics describe 13% of it, every pair
   statistic tried together 20%. Its exact form would gain 0.032 nats (ceiling, insight only). Fitting head 1's
   pattern better with key-salience flags (pattern KL 0.606 -> 0.477) made the output worse (+0.0146): the pattern
   weights every key alike, but the output depends only on keys whose votes matter.
7. **Head 1's votes are referent families.** A recalled ' she' writes herself +11.4, she +10.3, her +9.5, She +8.9 and
   female-name pieces. Quotes write a dialogue on/off signal (closing quotes -27 to -31 on dialogue words). Head 1 also
   completes words in context by bigram affinity, not induction: after 'trophies' it completes 'S' with 'op' (0.91).
8. **The model is less certain inside words than the corpus.** At the second piece of a word, the corpus
   distribution given the whole word-so-far is farther from the model (KL 1.26-2.28) than the bigram is (1.83). The
   model acts like a sum of single-token votes, not a word lookup.

## Consequence for the program

Within "corpus statistics + named scalars", the local heads look close to their ceiling: corpus tables top out near
31% of heads 0+2's effect even with perfect attention. To explain these heads further, the program would need to
describe a 32-dimensional learned code. One option is a per-token code built from a closed-form corpus construction
(for example, a 32-dimensional spectral embedding of the bigram matrix, counted at startup) instead of full PMI
rows. Whether that still counts as "explain, don't fit" is a question for the human. The confirmed candidate and the
"local team" framing (compare LOCAL-0 + LOCAL-2 + PHRASE with heads {0, 2, 3} jointly) can be adopted within the
current rules.

## Next steps

1. Integrate ov02_earlier_votes4 (dropping WPMI_3) into head0.py and head2.py, with KPRED counted in corpus.py.
   Then harness run, claim updates, and commit.
2. Re-state the consistency check with heads {0, 2, 3} as one local team.
3. Seed transfer (scripts/mech_seed_transfer.py, ready). Head roles move to different indices across seeds (attn_s1:
   the LOCAL-0 rule fits head 3, the RECALL rule fits head 0).
4. Ask the human: (a) may a low-dimensional corpus embedding (closed-form, counted at startup) stand in for the
   heads' learned codes? (b) a read-only model_attention(windows) helper in harness/claims.py for attention-pattern
   claims.
