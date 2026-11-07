# Plan: clean up, then worst-case properties with proofs on the ladder (2026-11-07)

## Context

The ladder's "account" line has ended. Over decisions 18-22, every pass condition we set for "understood" was met by
restating the model's computation in a new shape: per-symbol score tables (v1), copied 4×4 maps (v2), and next,
tables indexed by symbol and position. The human's reading (R3) caught the gap each time, but no measured test did.

The discussion of 2026-11-07 changed the target:
- alignment needs guarantees that hold on every input, not averages over dev stories;
- a residual measured on data is not robust;
- a result is therefore a **property of a model, proven or disproven for all inputs**.

Each property gets an upper bound from a proof and a lower bound from an input found by search. When the two meet,
the worst case is known exactly. The gap between them shows where understanding is missing. The scaling question:
for a fixed set of properties, do short proofs with tight bounds stay possible as the rungs grow? And does a proof
that uses mechanistic insight beat a generic verifier?

What L1 already shows, which the proofs start from:
- position is ignored;
- the model is a squeezed tied bigram;
- the head acts mostly as a per-symbol correction (logs/ladder/L1.md).

The human asked for a clean-up first (step 0), as before the ladder.

## 0. Clean up the repository

The human's choice: remove the ladder account machinery and the code of the closed main-track and side studies.
Keep harness/, program/ v1.3 (the reference point), the shared training code, and all logs and results.
Checkpoints are not deleted.

**0a. Freeze.**
- `git status` and `git worktree list` first.
- Commit the working tree (logs/runs/battery_watchdog.log).
- Tag `pre-proofs`. Every deleted file is recoverable with `git checkout pre-proofs -- <path>`.

**0b. Delete the ladder account machinery**, in one commit:
- analysis/ladder/: behaviour.py, compress_probe.py, length.py, scrub.py, the whole l1/ package;
- ladder/ (the account files);
- scripts/ladder/: compress.sh, l1_accounts.sh.

What survives from them:
- `parts()`, `finish()` and an unconstrained resample helper (`resampled()`) move into analysis/ladder/common.py.
  dump.py needs them for site importance.
- `compressed_bits()` and `strip_source()` from length.py move to analysis/proofs/cost.py. They measure proof length.
- eval.py: drop the account scoring and `load_account`/`account_logprobs`. Keep the model and baseline scores, the
  seed floor and `--check_l10`.
- test_tools.py: drop the account tests; keep the tests of what remains.

**0c. Delete the closed studies' code.**
- **Main-track analysis:**
  - the shared libraries at analysis/ top level (fitlib, mech_eval, tools, attention_fit, vote_fit, corpus_pairs,
    rules_common, worst, head_effects, consistency, claims_measure, taxonomy, cocluster, gen_categories, gen_vocab);
  - the packages canonical/, fixed_votes/, law_pressure/, look_write/, look_write_deep/, program_gap/,
    readable_training/, token_geometry/;
  - analysis/scratch/ (untracked: deleted outright).
- **Training code:**
  - src/train/: fixed_votes.py, law_pressure.py, readable.py and their sanity tests (test_sanity_fixed_votes,
    test_sanity_law_pressure, test_sanity_readable, test_sanity_canonical);
  - the side-study hooks in train.py (about 23 lines: readable_*, law_*, fixed-vote settings);
  - the side-study commands in runs.py.
  - Kept: the frozen main-track training (`runs.py` main commands), test_sanity.py, test_sanity_deep.py (checked: it
    imports only data, model, train and test_sanity), data.py, data_char.py, ladder_runs.py and test_sanity_ladder.py.
- **Scripts:** scripts/iterate.py, fit_scalars.py, seed_transfer.py, score_on_seeds.sh, setup_blackbox.py,
  scripts/canonical/.
  - Kept: scripts/common/watchdog.sh, scripts/ladder/{pilot_queue,rungs_L1_L3,rungs_L4_L7}.sh.
- **Kept untouched:** harness/, program/, baselines/, blackbox/, notebooks/, configs/ (all yamls: they record what
  was trained), analysis/out/ (results), models/.
- **Import check after the deletions:**
  - import every remaining .py;
  - run `python -m harness.eval_fidelity --program program --model attn_s0 --split dev_fast`, which must still give
    0.5497;
  - anything a kept file needs comes back from the tag.

**0d. Logs.**
- **Account study to its own folder:** move logs/ladder/{design.md, L1.md, compress_probe.md, iterations.csv} to
  `logs/ladder_accounts/` (git mv only, no content change).
  - Add a README there: the question, what was found, why it closed (decisions 16-23), and where the code is (tag
    pre-proofs).
- **logs/ladder/ keeps the models:** pilot.md, trained.md, param_counts.json, dumps/.
- **New `logs/proofs/`:** design doc, results, and a dead-ends section for proof attempts.
- **References:** a second commit rewrites every reference to a moved path in tracked .md/.py/.sh (not harness/ or
  configs/frozen.yaml).
- **Decision 23 in logs/decisions.md:**
  - the account line is closed; decision 22's frozen attempt ended at the pause with no pass (its outcome 2: the
    next decision is about method);
  - the main line becomes properties with proofs;
  - code removed at tag pre-proofs.

**0e. Documents.**
- **AGENTS.md** is rewritten for the new mission (the human's choice).
  - Contents: the mission (worst-case properties with proofs on the ladder; what a result is); the repository map;
    the standing rules.
  - Standing rules carried over: the decisions log; dead ends and surprises recorded at once; report only measured
    numbers; never look at the test split; harness/ and configs/frozen.yaml read-only; human gates before
    pre-registered work; commit before removing worktrees; check df and battery before long runs.
  - The original brief is kept verbatim as `logs/plans/original_brief.md`, and AGENTS.md links it.
- **README.md:** the new top-down summary, folder map, status table (accounts closed, proofs active), how to run.
- **Also:** logs/README.md index; analysis/ladder/README.md; writeups/draft.md section 7 (what the account line found
  and why it closed) and a new section 8 (proofs, planned); the memory resume point.

**0f. Verification of step 0.**
- `pytest src/train analysis/ladder`: all remaining tests pass.
- **Training reproduces:** a 200-step L1 run (seed 0, CPU) gives the same losses before and after the train.py edit,
  compared to the last digit.
- **Harness:** the harness run of v1.3 gives dev_fast KL 0.5497, and `eval.py --check_l10` agrees.
- **Every kept .py imports.**
- **References:** a grep for each removed or moved path finds nothing outside the tag and the deleted-files list.

## 1. Design doc, then a human gate (logs/proofs/design.md)

Fixed before any bound is computed.

**Properties.** All three of the human's list; there is no argument against all three. Context sensitivity is
locality at k = 1, so there are two families.
- **Locality δ(k)**, for k in {1, 2, 4, 8, 16, 32}: the largest change in the next-symbol log-probabilities between
  two windows that agree on their last k symbols, over all windows and positions.
  - Distance: d(p, q) = max over symbols b of |log p_b − log q_b| (nats). It bounds KL and total variation, and it
    is what interval arithmetic gives directly.
  - δ(1) is context sensitivity: same current symbol, any context.
- **Forbidden output φ:** for every pair (a, b) whose train-split bigram count is 0, the largest probability of b
  over all windows that end in a. Reported: the maximum over the forbidden pairs, and the distribution of the
  per-pair maxima.
  - The forbidden set is computed once from the train stream and stored (data/char/forbidden_pairs.npy). Dev and
    test are not used.

**What a result is**, per property × rung × arm × seed:
- the lower bound L (an input found by search, stored);
- the upper bound U (the output of a proof);
- the status: exact when U ≤ L·(1 + 1e-3), otherwise the gap U / L;
- the proof cost: the compressed source bits of the proof code (cost.py) and the compute time;
- the method: **generic** (interval bound propagation with each position's input as the box over all symbols' rows;
  no insight) or **mechanistic** (uses a stated property of the model, e.g. "the head's output is an average of
  per-symbol values").

**Soundness rules for proofs:**
- float64 interval arithmetic, with every interval widened outward by np.nextafter after each operation;
- every proof states its assumptions;
- a proof is accepted only if it never falls below a value observed on dev_full and on the search inputs, and if
  it passes the brute-force check below.

**Seeds and rungs:**
- all five seeds, both arms at L1-L3, ln at L4-L7;
- a method works at a rung if its bounds are non-vacuous on all five seeds: U finite and below the trivial bound,
  e.g. δ below the largest log-probability range;
- L8 (context 256) and branch M (MLP) need training and a separate gate.

## 2. Tools (new package analysis/proofs/)

Reused: analysis/ladder/common.py (`load`, `windows`, `kl_pos`, `model_logprobs`, `parts`, `finish`),
analysis/ladder/reference_forward.py (the NumPy forward pass, the base of the interval version), and
src/train/data_char.py (streams; chunked counting as in `ngram_tables`).

| file | what |
|---|---|
| `interval.py` | sound float64 interval arithmetic: add, matmul with interval inputs, LayerNorm (bounds via mean and variance intervals), exp, softmax and log-softmax |
| `reach.py` | the reachable set of a head's output for a fixed query symbol and position: the convex hull of the per-(symbol, position) value vectors; and the tighter softmax-aware set (weights tied to the fixed scores) |
| `bounds.py` | upper bounds for δ(k) and φ, by branch and bound over the reachable set (split boxes, interval bounds, keep the boxes that can still hold the maximum) |
| `generic.py` | the generic baseline: interval bound propagation through the whole model |
| `attack.py` | lower bounds by search: greedy coordinate ascent over the window's symbols with random restarts, plus constructions (one symbol repeated, one strong key placed where its score is highest) |
| `brute.py` | exact maxima by enumeration for a truncated context (T ≤ 3: 79³ = 493k windows per query), to check L ≤ exact ≤ U |
| `cost.py` | proof length (compressed_bits, strip_source from the old length.py) |
| `run.py` | one command per property × rung × arm × seed; appends to logs/proofs/results.csv |
| `test_proofs.py` | the tests in Verification |

**Locality with k > 1** fixes the last k symbols in both windows. The proof treats them as a shared block:
- its contribution enters through the same reachable set in both windows;
- the softmax weight between the shared block and the variable far block is bounded from the score ranges;
- the shared block is covered by branching on its hull, not by enumeration.

## 3. L1 first

1. Forbidden pairs from the train stream; δ(k) and φ for all five seeds, both arms, generic and mechanistic.
2. Report logs/proofs/L1.md: for each property, L, U, the gap and the cost, and which method closed it.
   - Also: the worst-case inputs and what they look like.
   - Also: the worst case against the dev average (δ(1) against the 0.003-0.07 nats measured on stories).
3. Human review before climbing.

## 4. Climb

- **L2-L7** (trained), the same properties and methods.
- For each rung, record where the bounds stop meeting and which step of the proof loosens:
  - the hull too generous;
  - the LayerNorm intervals;
  - several heads together.
- Each new proof idea is logged as a mechanistic method with its assumption.
- L8 and branch M: a gate for training, then their own proof techniques (several heads at context 256; ReLU
  relaxations for the MLP).

## Order and gates

1. Step 0 (CPU, about half a day), then verification 0f.
2. Step 1: the design doc. **Human gate.**
3. Step 2: tools and tests.
4. Step 3: L1 report. **Human review.**
5. Step 4: the climb, rung by rung.

Check df and the battery before long runs.

## Verification

- **Step 0:** 0f.
- **Interval arithmetic:** for 10,000 random boxes per operation, the true outputs of sampled points lie inside the
  computed intervals.
- **Brute force, L1 at T ≤ 3** (each seed): L ≤ exact ≤ U for δ(1), δ(2) and φ. The generic and the mechanistic
  proof are both checked.
- **Soundness on data:** no U below any value observed on dev_full windows or on the attack inputs.
- **Attack sanity:** on dev_full, the attack's L is at least the largest value observed among dev windows.
