# Ladder: trained rungs (dev_full loss, nats per symbol)

Configs as in logs/ladder/pilot.md. L1-L3 use LR 3e-2 and 25M symbols; L4-L7 use LR 1e-2 and 400M symbols. Each
run's numbers are in models/ladder/<rung>_<arm>_s<seed>/result.json (gitignored).

Baselines on dev_full c64: unigram 3.076, bigram 2.299, trigram 1.704.

| rung | params (ln) | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|
| L4 | 3344 | 1.9057 | 1.8892 | 1.8894 | 1.9242 | 1.9438 |
| L5 | 8736 | 1.6304 | 1.6129 | 1.6194 | 1.6227 | 1.6286 |
| L6 | 25664 | 1.5235 | 1.5209 | 1.5157 | 1.5122 | 1.5189 |
| L7 | 84096 | 1.5177 | 1.5124 | 1.5041 | 1.5082 | 1.5110 |

L1 ln: 2.5497, 2.5491, 2.5531, 2.6025, 2.5934 (below the bigram). L1 noln s0: 2.5074.

Reading:
- L4 is between the bigram and the trigram. L5 and above beat the trigram.
- L6 to L7 gains about 0.008 for 3.3 times the parameters. At context 64, one attention-only block saturates near
  1.51.
