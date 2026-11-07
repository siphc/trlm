# The ladder's account study (closed, 2026-11-07)

**Question.** Can a tiny language model on the ladder be "understood" by a written account? An account is a
readable program plus claims. It was scored by pass conditions fixed in advance:
- P1: fidelity against the seed floor;
- P2: size against the weights;
- P3: causal faithfulness;
- P4: behaviour tests.

**What was found** (L1, 644 parameters, all five seeds; details in L1.md):
- **Position:** L1 ignores where a symbol is in the window.
- **Direct path:** it is a bigram squeezed into 4 tied dimensions, which is why L1 is worse than the true bigram.
- **The head:** it attends by content, favours boundary symbols, and acts mostly as a fixed per-symbol correction.
- **Context:** what the head does with context is small (0.003-0.07 nats on dev) and differs by seed. It does not
  reduce to symbol classes.
- **Compression probe** (compress_probe.md): generic simplifications compress L1-L7 by only 1.1-2.7 times at the seed
  floor.

**Why it closed** (decisions 16-23 in logs/decisions.md):
- Each version of the pass conditions was met by restating the model's computation in a new shape:
  - per-symbol tables (v1);
  - copied 4×4 maps (v2);
  - next, tables indexed by symbol and position (decision 22).
- Only the human's reading caught it.
- The main line became worst-case properties with proofs (logs/proofs/).

**Files:**
- design.md: the design and criteria, with every amendment;
- L1.md: the accounts v1 and v2 and their scores, and the decision-22 attention study;
- compress_probe.md: the compression probe;
- iterations.csv: account scores.

The code (analysis/ladder/{behaviour,compress_probe,length,scrub}.py, analysis/ladder/l1/, ladder/) is at git tag
pre-proofs: `git checkout pre-proofs -- <path>`.
