# Worst-case properties with proofs: design (DRAFT, awaiting the human's gate)

Decision 23. Nothing here is computed until the human approves this document. A change after approval needs a new
decision in logs/decisions.md first.

## Models

- **Rungs:** the ladder's trained rungs (logs/ladder/trained.md), attention only, one block, context T = 64,
  alphabet V = 79 symbols:
  - L1-L7 with LayerNorm, seeds 0-4;
  - L1-L3 without LayerNorm, seeds 0-4.
- **Later, each behind its own gate:** L8 (context 256) and the MLP branch, which need training and their own proof
  techniques.
- **The model function** is the exact real-number function of the stored float32 weights. Proofs evaluate it in
  float64 interval arithmetic. The rounding of a float32 forward pass on a device is not covered; this is stated as
  an assumption.

## Inputs

- **An input** is any prefix x = (x_0, ..., x_{n-1}) of symbols in {0, ..., 78}, for n = 1, ..., 64. Every symbol may
  appear anywhere, including EOS and UNK.
- **The model's output** p(x) is its next-symbol distribution after x: row n − 1 of the forward pass on x.
- **The input space** has 79^n prefixes of length n, so no property below is checked by enumeration at full length.

## Properties

**Distance.** d(p, q) = max over symbols b of |log p_b − log q_b|, in nats. It bounds the others:
- KL(p || q) ≤ d;
- total variation ≤ e^d − 1.

It is also what interval arithmetic on log-probabilities gives directly.

**Locality δ(k)**, for k in {1, 2, 4, 8, 16, 32}:

  δ(k) = max over n ≥ k + 1, and over prefixes x, x' of length n with x_i = x'_i for i ≥ n − k,
         of d(p(x), p(x'))

The two inputs agree on their last k symbols and have the same length, so positions stay aligned. Anything before
the last k symbols may differ. δ(1) is **context sensitivity**: the same current symbol, any context. δ(k) does not
increase with k. Also reported: δ(k) per current symbol, and per position n.

**Forbidden output φ.**
- **F** is the set of pairs (a, b) whose bigram count in the train stream is 0. The stream is the stories joined by
  EOS, as in src/train/data_char.py.
- F is computed once and stored in data/char/forbidden_pairs.npy, with its size and a sha in
  logs/proofs/forbidden_pairs.md. Dev and test are not read.
- For (a, b) in F: φ(a, b) = max over prefixes x whose last symbol is a of p_b(x).
- Reported: max over F of φ, the distribution of the per-pair maxima (quantiles), and the ten largest pairs with
  their inputs.

## What a result is

For each property, rung, arm and seed, three numbers:
- **L, the lower bound:** the value at a stored input found by search (two inputs for δ).
- **U, the upper bound:** the output of a proof.
- **The status**, from the ratio U / L:

| status | U / L |
|---|---|
| exact | ≤ 1.001 |
| tight | ≤ 1.1 |
| informative | ≤ 10 |
| loose | otherwise |

Also recorded:
- **the method;**
- **the proof cost:** compressed source bits of the proof's code (analysis/proofs/cost.py: every module the proof
  imports from analysis/proofs/), plus wall-clock seconds;
- **the typical case on dev**, for comparison only. For δ(k): over dev_full positions, the mean of d(p(x), p(x'))
  where x' replaces the symbols before the last k with those of a random other dev window (8 draws). For φ: the
  largest p_b over dev positions ending in a.

## Methods

- **Generic (G).** Interval bound propagation through the whole forward pass.
  - For δ(k), the last symbol and its position are fixed exactly; branch over all 79 symbols and all n.
  - Every other position's embedding is the box over all 79 rows plus its position vector.
  - So for k > 1 it bounds δ(k) by its δ(1) bound.
  - No fact about the model is used beyond its weights.
- **Mechanistic (M).** A proof that uses a stated fact about the model's computation. Each fact is written as an
  assumption with its own check. The first fact (one-block attention-only models): each head's output is a convex
  combination of the per-(symbol, position) value vectors in view.
  - So for a fixed query symbol and position, it lies in the convex hull of at most 79 · n points.
  - The output is then bounded over that region by branch and bound with interval arithmetic.
  - For δ(k), the shared last k symbols form a block whose contribution is the same in both inputs. The softmax
    weight between the shared block and the rest is bounded from the ranges of the scores.
  - Later proof ideas are added as new mechanistic methods (M2, M3, ...), each with its stated fact.
- **Search (lower bounds).**
  - Greedy coordinate ascent over the free symbols, with random restarts.
  - Constructions: one symbol repeated; one strong key placed where its score is highest.
  - The best inputs from dev_full, as starting points.

## Soundness rules

A proof's U is accepted only if all of the following hold:
1. **Interval arithmetic** is float64. Every interval is widened outward with np.nextafter after each operation.
   - Elementary functions (exp, log, sqrt) are monotone and bounded at the endpoints.
   - LayerNorm is bounded from intervals of its mean and variance, never by sampling.
2. **The proof states its assumptions.** Each mechanistic fact has a test.
3. **No observed value exceeds U:** on dev_full, and on every input the search tried.
4. **Brute force:** on the same model restricted to short prefixes (n ≤ 3: 79³ = 493,039 prefixes; pairs of prefixes
   for δ), L ≤ exact maximum ≤ U, for δ(1), δ(2) and φ, on every seed of L1. The truncated domain uses the same
   proof code with n capped.

The interval primitives are tested on 10,000 random boxes per operation: sampled points stay inside the computed
intervals.

## The scaling report

For each rung, a table of property × method: U / L (median and worst over the seeds), proof bits and seconds. Each
row also notes which step of the proof loosens first:
- the hull being too generous;
- the LayerNorm intervals;
- several heads interacting;
- the shared-block weight bound for δ(k).

The headline curves are U / L against parameter count, for G and for the best M, and the proof bits for the best M.

## Order

1. Forbidden pairs.
2. The tools (analysis/proofs/) and their tests.
3. L1 (both arms, all seeds), with the report logs/proofs/L1.md. Human review.
4. L2-L7, one rung at a time, appending to logs/proofs/results.csv and the scaling report.
