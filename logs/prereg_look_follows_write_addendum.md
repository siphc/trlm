# Addendum to the pre-registration (2026-10-14): one-layer loose ends and the echo test

Written after the one-layer replication (logs/look_follows_write_results.md) and before running any of the analyses
below. The original verdict (SUPPORTED literally, 7/8; MIXED strictly, 6/8) stands as recorded. Everything here is
an addendum, labelled as such in the results.

## A. Ceiling with the self position (`tt_self`, analysis/look_write/extra_arms.py)

The exact token-token term is also applied at the self position (j = i), with its own coefficient.
- Prediction A1: with this ceiling, the gain is positive for attn_s1/h3 and for the near-threshold head attn_s2/h0
  (both negative under the pre-registered ceiling).
- Prediction A2: with `tt_self` as the ceiling, P1 holds for at least 7 of the 8 confirmatory LOCAL heads (own gain
  >= 0.5 x ceiling gain, ceiling gain > 0).

## B. Paired standard errors for attn_s0, s3, s4

Re-run fit_arms.py with per-window KLs. No prediction; this completes the reporting.

## C. Far-looking heads: usefulness over distances 8..63

UOV aggregated over the distance band 8..63 (one coefficient), for the six RECALL heads of attn_s1..s4 plus attn_s0.
- Prediction C1: for at least 4 of the 6 seeds-1-4 RECALL heads, the own band-UOV gain exceeds the own d = 1..7 gain.
- Prediction C2: other heads' band-UOV gains and the shuffled control stay <= 0.25 x the own band gain for at least 4
  of the 6.

## D. The echo test (E1/E2): does the relation come before calibration?

Concern: in a calibrated model, E[next | k, q] matches the model's own prediction, which contains the head's own
attention times its vote. So usefulness can mirror attention without causing it. Test on the saved checkpoints of
attn_s1..s4 (steps 229, 718, 2246, 7032, 22009 and final). At each checkpoint, using that checkpoint's own weights:
- r(c): the pure-pair correlation at d = 1 between the head's QK pair interaction and its own UOV;
- m(c): the correlation between UOV and the model-echo part MGU (the model's own prediction minus P1, instead of the
  true next token), as a calibration index.

The training-dynamics reading ("comes from") is supported if all three hold:
- D1: r(c) >= 0.2 for at least 6 of the 8 confirmatory heads at a checkpoint where the learning rate is still at
  least 90% of peak and the dev loss is at least 0.2 nats above its final value.
- D2: cross-lagged, pooled over heads and consecutive checkpoint pairs: corr(score change c -> c+1, UOV(c)) exceeds
  corr(UOV change c -> c+1, score(c)).
- D3: r(c) at that early checkpoint is at least half of r(final).

If D1-D3 fail, the result is reported as "mutual consistency" (attention is predictable from the vote and the data),
not "comes from".
