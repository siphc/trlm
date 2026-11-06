# Addendum 2 to the pre-registration (2026-10-16): the growth test

Written while the echo test (addendum D) was running and before any of its outputs or any growth number had been
seen. Motivation: Cammarata's essay "Neural Networks Are Already Clean" (quotes in logs/deep_study.md) proposes that
"established routes have a head start in capturing residual error". For attention heads this predicts that a head's
vote (what it writes) forms first, and its attention grows onto the keys where that vote already captures error.

## Data

- Heads: the 8 confirmatory LOCAL heads of attn_s1..s4 (analysis/out/look_write_deep/attn_sK/classes.json).
- Pair set P: the d = 1 pairs with sample N >= 20 in the echo-test passes (analysis/look_write_deep/echo_traj.py; the
  pair set is identical across a model's checkpoints because the sample is), weights N.
- At checkpoint c (models/final/attn_sK/ckpt_step*.pt, and final.pt):
  - OV_c: the head's direct-path vote table (analysis/look_write/common.ov_tables on the checkpoint's weights);
  - S_c: the exact pure-pair attention score at d = 1 (common.pure_pair_scores), double-centred over P;
  - U_c: the head's usefulness table at d = 1 (DV readout, = UOV), double-centred over P (echo-test passes; only at
    the six echo-test checkpoints 229, 718, 2246, 7032, 22009 and final).

## G1. The vote forms before the attention (confirmatory)

For every one of the 20 checkpoints and final:
- vote convergence: corr_w(OV_c, OV_final), over the rows of the 500 most frequent key tokens (all V columns), each
  row weighted by the key's unigram frequency;
- attention convergence: corr_N(S_c, S_final) over P.

conv(X) = the first checkpoint from which the correlation stays >= 0.8 at every later checkpoint. G1 holds for a head
if conv(OV) is a strictly earlier checkpoint than conv(S). Prediction: G1 holds for at least 6 of the 8 heads.
Thresholds 0.5 and 0.9 are reported as sensitivity analyses, not as tests.

## G2. Attention is acquired where the vote is useful (confirmatory)

For each consecutive pair of echo-test checkpoints (c, c+1), 5 transitions:
- T_c = the top 10% of P by S_c (unweighted count of pairs);
- entering pairs E = in T_{c+1} and not in T_c;
- control pairs = in neither T_c nor T_{c+1}.
Each entering pair is compared with control pairs from the same decile of N. Statistic: mean U_c over entering pairs
minus the mean over their decile-matched controls, pooled over the 5 transitions (each pair weighted equally).
Significance: permutation of entering/control labels within N deciles and transitions, 2000 draws, one-sided.
G2 holds for a head if the statistic is > 0 with p < 0.05. Prediction: at least 6 of the 8 heads.

G2 differs from D2 of addendum 1: D2 is a linear correlation of continuous changes over all pairs; G2 asks which pairs
become strongly attended, with frequency matched.

## G3. Recursive generalisation (exploratory)

For each transition and each query token q with at least 3 pairs in T_c (q's core keys): for each key k entering T_{c+1}
with that q, s(k) = cos_w(OV_c[k], mean of OV_c over q's core keys), with columns weighted by the unigram frequency of
v. Controls: keys of the same q not in T_c or T_{c+1}, in the same N decile. Statistic: mean s(entering) - mean
s(controls), pooled; permutation test as in G2. Exploratory prediction: entering keys are more similar for at least 5
of the 8 heads.

## Interpretation

- G1 and G2 both hold: the vote forms first, and attention is acquired where the vote captures residual error. This is
  the developmental direction of the law, consistent with the essay's reuse picture and with Bietti et al. (2023).
- G1 fails (attention converges first or together) while the final-model relation holds: the law is better read as
  mutual consistency (or the reverse direction) than as "comes from".
- G2 fails: newly acquired attention is not where the vote is useful at the time, against the developmental reading.
