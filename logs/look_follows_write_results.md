# Results: "where a head looks comes from what it writes" (2026-10-14)

Pre-registration: `logs/prereg_look_follows_write.md` (committed before seeds 1-4 were analysed). Independent
re-implementation and full numbers: `analysis/look_write/` (RESULTS.md, results.json). Seed 0 serves as the
cross-check against discovery round 2.

## Verdict

**SUPPORTED on the literal pre-registered reading (7 of 8 LOCAL heads in seeds 1-4 meet P1-P3; the bar is 80%).
On the strict reading it is MIXED (6 of 8).** The difference is one head, attn_s1/h3. It passes P1 only because the
pre-registered ceiling (the exact token-token query-key term) makes its output worse (gain -0.052 +- 0.011). A
negative ceiling gives no meaningful ratio. The ceiling stays negative in every form tried, so the ceiling is broken
for that head, not the usefulness term (own-vote usefulness gains +0.092 +- 0.004 there).

| prediction | holds for (seeds 1-4, 8 LOCAL heads) |
|---|---|
| P1 own-vote usefulness recovers >= half of the exact pair term's output gain | 7/8 literal, 6/8 strict (s3/h1: 0.38) |
| P2 another head's vote, or a token-shuffled vote, gives <= 1/4 of that | **8/8** (at most 0.10) |
| P3 pure pair correlation >= 0.25 and above every other head's | **8/8** (own 0.34-0.44; others <= 0.08) |
| P4 coefficient within 1.5x of the median (reported separately) | 7/8 (median 0.23 / 0.25 logit per logit) |

## What is established

1. **Head-specificity is robust.** In every local head of every seed, the pair-specific part of where the head
   looks lines up with the usefulness of its OWN vote (correlation 0.34-0.44), and not with another head's vote
   (at most 0.08) or a shuffled vote. Adding own-vote usefulness to a generic position and spelling rule improves
   the exact model's output by 0.06-0.18 nats per head. The control arms give at most a tenth of that.
2. **The constant is roughly shared.** About 0.24 logit of attention score per logit of usefulness, within 1.5x
   for 7 of 8 heads, up to 2x off for a few.
3. **It is not always the whole pair term.** For one head (s3/h1) usefulness recovers only 38% of the exact
   term's gain, and for two more 76-84%. Something besides own-vote usefulness also shapes those heads' attention.
4. **Scope:** local heads of one-layer attention-only models. Near-threshold heads that look 12-13 tokens back
   behave the same. For far-looking heads, usefulness at distances 1-7 recovers only 6-38% (not tested at long
   range). In the model with an MLP (mlp_s0), measuring the vote through the direct path alone shows nothing. That
   test is not meaningful, because those heads act through the MLP.

## In the program (Track B)

The law, computed from corpus counts and the program's own votes, became part of the readable copy (v1.2, commit
1ef9b7a). Each local reader's attention adds 0.35-0.55 per nat of usefulness of its own vote. That is worth
0.0070 +- 0.0005 nats on fresh windows, and the dev_fast KL drops from 0.578 to 0.571. Head 2's "key informativeness"
weight halved once the usefulness term was available: the law absorbed an earlier heuristic. The gain is much smaller
than in the exact model (0.06-0.18 per head). As the law predicts, the program's attention is limited by how well
its votes match what the heads write.

## Not run (exploratory)

The ceiling variant with the token-token term at the self position (the likely fix for s1/h3); the long-range
version for far-looking heads; paired SEs for seeds 0, 3 and 4.
