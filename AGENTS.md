# AGENTS.md

Context file for coding agents working on this repository. Read all of it before writing code. Ask the human when
blocked. Do not guess silently.

The original brief (a weight-free readable program for a 0.35M-parameter TinyStories model) is kept verbatim in
`logs/plans/original_brief.md`. References to "AGENTS.md section N" in `harness/`, `configs/frozen.yaml` and logs
written before 2026-11-07 mean that original brief.

## 1. Mission

Prove things about language models that hold on **every** input, and find out how far such proofs scale.

The models are a ladder of tiny character-level TinyStories models, from 644 parameters upward:
- one change per rung;
- five seeds per rung;
- details in `logs/ladder/` and `configs/side_ladder.yaml`.

A **result** is a property of one model, stated for all inputs. Each property gets two bounds:
- **an upper bound U** from a proof: code that computes a certified bound, sound under float64 interval arithmetic
  with outward rounding;
- **a lower bound L** from an input found by search, stored.

The result is **exact** when the two meet. Otherwise the gap U / L shows where understanding is missing. Each result
also records the proof's cost (compressed source bits, compute) and its method:
- **generic:** no insight into the model, e.g. interval bound propagation;
- **mechanistic:** uses a stated fact about the model, e.g. "the head's output is an average of per-symbol values".

The scaling question: for a fixed set of properties, do short proofs with tight bounds stay possible as the rungs
grow, and does mechanistic insight beat generic verification? The properties, the soundness rules and what counts as
non-vacuous are fixed in `logs/proofs/design.md`. That document passes a human gate before any bound is computed.

Why this mission (decision 23): an earlier line tried to certify that a model is "understood" with pass conditions on
written accounts. Every version was met by restating the computation in a new shape (`logs/ladder_accounts/`).
A residual measured on data is an average-case claim; alignment needs guarantees.

## 2. Repository map

| path | what |
|---|---|
| `analysis/proofs/` | the proof tools: interval arithmetic, reachable sets, bounds, search, brute-force checks, proof cost |
| `analysis/ladder/` | model access for the ladder: `common.py` (load models, eval windows, KL, the split into direct path and heads), `eval.py` (dev loss, baselines, seed floor), `dump.py` (readable weight dumps), `reference_forward.py` (NumPy forward pass) |
| `src/train/` | training: `model.py`, `train.py`, `data.py` (BPE, main track), `data_char.py` (characters, the ladder), `ladder_runs.py` (rungs, pilot, chosen configs), `runs.py` (main-track runs), sanity tests |
| `scripts/` | `ladder/` (training queues), `common/watchdog.sh` (battery watchdog) |
| `logs/` | `decisions.md`, `dead_ends.md`, `surprises.md` at top level; one folder per study (index `logs/README.md`); `proofs/` is the current one |
| `harness/`, `program/` | the main track's scorer and its readable program v1.3, kept as the reference point (read-only) |
| `models/`, `data/` | checkpoints and data (not in git) |
| `writeups/` | drafts for readers outside the project; never publish without asking |

Code of closed studies is at git tags (`pre-cleanup`, `pre-proofs`): `git checkout <tag> -- <path>`.

## 3. Standing rules

- **Decisions.** Every human decision goes into `logs/decisions.md`, numbered, with its reason, before work that
  depends on it. Read the decisions before acting.
- **Gates.** Pre-registered work, meaning a design doc, a property list or a criterion, passes a human gate before
  any result is computed. A change after the gate needs a new decision first.
- **Record at once:**
  - every failed hypothesis or proof attempt in `logs/dead_ends.md` (or the study's dead-ends section), with
    evidence;
  - anything unexpected in `logs/surprises.md` the moment it appears.
- **Report only measured numbers.** Include failed runs. State assumptions at the top of each report.
- **Soundness first.** A proof is accepted only if:
  - it states its assumptions;
  - it never falls below an observed value on dev data or on search inputs;
  - it passes the brute-force check on a truncated context where enumeration is possible.
- **Data:**
  - never look at the test split and never copy it anywhere;
  - dev is the discovery split;
  - the train split may be used for corpus statistics, such as the forbidden pairs.
- **Read-only:** `harness/`, `configs/frozen.yaml`, every evaluation set.
- **Verify library behaviour** in the documentation. Do not assume a function exists or a default behaves a certain
  way. If unsure, say so.
- **Machine:**
  - check `df` and the battery before long runs; train on the charger with the lid open;
  - commit before removing a git worktree.
- **Ask first:**
  - before anything billable or outward-facing (cloud, publishing);
  - before changing a frozen file;
  - before changing gated criteria.

  Otherwise work through the agreed plan without pausing.
- **Commits.** Every iteration is a commit, and the message carries the numbers.

## 4. References

- Compact proofs via mechanistic interpretability (Gross et al.): https://arxiv.org/abs/2406.11779
- Formalizing the presumption of independence / heuristic arguments (ARC): https://arxiv.org/abs/2211.06738
- Causal scrubbing (Chan et al., Redwood Research, 2022; Alignment Forum post), used by the closed account study
- Interval bound propagation (Gowal et al.): https://arxiv.org/abs/1810.12715; CROWN (Zhang et al.):
  https://arxiv.org/abs/1811.00866
- A Mathematical Framework for Transformer Circuits (Elhage et al.): https://transformer-circuits.pub/2021/framework
- TinyStories (Eldan and Li): https://arxiv.org/abs/2305.07759
