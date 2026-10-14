# Pre-registration: "where a head looks comes from what it writes" (2026-10-14)

Written before any analysis of seeds 1-4. Seed-0 numbers come from discovery round 2 (analysis/investigations/qk/
NOTES.md); they motivated the hypothesis and are not evidence for it.

## Hypothesis

In a trained one-layer transformer, the pair-specific part of a head's attention score (query token q, key token k at
distance d) is set by how useful the head's OWN vote from k is for the next token, given q:

    UOV_h,d(k, q) = E[ OV_h(k, next) | k sits d tokens before q ]  -  E_{v ~ P1(.|q)}[ OV_h(k, v) ]

- OV_h(k, v): the logit head h adds to token v when it attends fully to token k (value/output path through the
  embedding, first LayerNorm at a typical position, W_V, W_O, final LayerNorm gain and centring, and the tied
  unembedding), divided by the model's typical final-LayerNorm scale so that it is in logit units.
- E[. | k d before q]: average over all corpus positions where k sits d tokens before q (TRAIN stream), with the
  next token being the token after q. Shrinkage: (S - N B) / (N + 20), as in round 2.
- P1(. | q): the program's bigram rule (corpus.py), standing for "what q alone predicts".

The training-dynamics reading: a key's attention score grows where its value vector points along the prediction
error. The law says the QK circuit's pair term is (approximately) a function of the OV circuit and corpus counts.

## Models and heads

attn_s1, attn_s2, attn_s3, attn_s4 (confirmatory); attn_s0 (cross-check of the re-implementation against round 2).
A head is LOCAL if its mean attention distance to earlier tokens (excluding itself) is at most 10 tokens on 32 TRAIN
windows; all other heads are RECALL heads. Confirmatory predictions are for LOCAL heads. RECALL heads, and the mlp_s*
models, are exploratory (no prediction).

## Protocol (identical for every head)

1. Base attention rule: a generic log-linear rule (distance buckets, query shape x self, key shape), fitted to the
   head's own attention pattern (pattern cross-entropy) on train_windows(32, seed=2024).
2. Arms, each adding ONE pair feature (separate coefficients for d = 1 and d = 2..7) to the base rule, refitted the
   same way: (a) own UOV; (b) each other head's UOV; (c) own UOV with the rows of OV_h randomly permuted over k
   (token-shuffled control); (d) the exact token-token QK term from the weights (ceiling).
3. Output test: substitute each arm's attention pattern into the exact model (all else exact, head's own OV kept) and
   measure output KL against the model on train_windows(32, seed=2025). Gain = KL(base) - KL(arm).
4. Pattern test: double-centred (query and key main effects removed) correlation at d = 1 between each feature and
   the exact QK pair interaction, pairs with count >= 20, weighted by count.

## Predictions (for each LOCAL head of seeds 1-4)

- P1. Own-UOV gain >= 0.5 x exact-token-token gain. (Seed 0: 1.71, 1.63, 0.96.)
- P2. Every other-head UOV gain, and the token-shuffled gain, <= 0.25 x own-UOV gain. (Seed 0: <= 0.13.)
- P3. Pure-pair correlation at d = 1: own UOV >= 0.25 and above every other head's UOV. (Seed 0: 0.39 / 0.45.)
- P4. The fitted own-UOV coefficients (logit units) lie within a factor 1.5 of their median over all LOCAL heads of
  seeds 0-4, separately for d = 1 and d = 2..7.

Verdict: SUPPORTED if P1-P3 hold for at least 80% of LOCAL heads in seeds 1-4; REFUTED if P1 fails for at least half
of them; otherwise MIXED. P4 is reported separately (law with a universal constant vs per-head constant).

## Why this matters for the readable copy

If the law holds, a head's attention needs only position/shape rules plus one universal usefulness term computed from
its vote, so the program must explain the votes and gets the pair-specific attention for free. It also gives a test
for any proposed vote rule: a correct vote rule predicts where its head looks.
