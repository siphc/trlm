# Token geometry and the program's gap (2026-10-17; analysis only, model weights used as data)

Two measurements requested after mechanism program v1.3 (logs/mech_v3_report.md): (1) how much of the models' shared
token geometry named token properties explain; (2) how much of the program's gap is in what the heads write (per-token
tables) versus where they look (attention). Code: analysis/token_geometry/. Outputs: analysis/out/token_geometry/.
TRAIN windows only.

## M2. Tables versus attention (tables_vs_attention.py; attn_s0; 16 fit / 32 eval windows; every arm refitted)

The model's logits (final LayerNorm at its typical scale) = direct-path table + sum over heads of attention-weighted
vote tables. Readable versions: corpus tables (FOLLOW, skip-gram, copy/topic votes; head 3 read with skip-gram votes)
and readable attention rules (the program's; head 3: LOCAL-0's rule fitted to head 3's pattern, pattern KL 1.03).
PHRASE is a direction in every arm.

| arm | attention | tables | KL |
|---|---|---|---|
| A | readable | corpus | 0.551 |
| B | readable | exact | 0.921 |
| C | exact | corpus | 0.523 |
| D | exact | exact | 0.115 |
| B_cur | readable | exact for the current token only | 0.870 |
| C_cur | exact | exact for the current token only | 0.754 |

Paired differences (SE over 32 windows): B - A +0.371 (0.013); C - A -0.028 (0.003); D - C -0.408 (0.011); D - B -0.806
(0.016). D's 0.115 is mostly the per-position LayerNorm scale, which no arm models.

**Tables help only with the model's attention, and attention only with the model's tables.** The program and the
model are two internally consistent solutions; exchanging either half alone makes things worse or barely better.

Per-unit swaps, starting from D (everything exact) and replacing one unit:

| unit | corpus table, exact attention | readable attention, exact table | both readable |
|---|---|---|---|
| current token (direct path) | 0.239 | | |
| head 0 (LOCAL-0) | 0.843 | 0.348 | 0.826 |
| head 1 (RECALL) | 0.284 | 0.286 | 0.334 |
| head 2 (LOCAL-2) | 0.680 | 0.346 | 0.689 |
| head 3 | 0.417 | 0.361 | 0.434 |

- Within a head, what it writes dominates: readable attention rules cost 0.17-0.25 nats per head given the right table;
  corpus tables cost 0.17 (RECALL) to 0.73 (LOCAL-0); once the table is a corpus table, the attention rule barely matters.
- The local heads' tables are the furthest from corpus statistics; RECALL's and the direct path's are closest.
- The full program (A, 0.551) is better than any single local-head swap (0.68-0.84) because all its parts adapt to
  each other; a single corpus table among exact parts cannot be compensated.

## M1. Named properties versus shared token geometry (named_features.py, named_features_more.py; attn_s0..s4)

Each seed's token embedding spans a 128-dimensional subspace of token space (2,000 tokens). Directions held by every
seed are found as eigenvectors of the mean of the five projectors. Named properties (no model): 192 word classes
(categories.py, three systems), 15 spelling features, 12 corpus statistics, 6 corpus association axes (she vs he,
they vs it, dialogue; after and before). Coverage of a direction = its squared length inside the span of the named
properties. Two weightings: uniform (every token alike) and frequency.

| | uniform | frequency |
|---|---|---|
| directions held by all seeds (mean-projector eigenvalue > 0.9 / > 0.8) | 19 / 36 | 36 / 57 |
| seed-to-seed subspace overlap (mean) | 0.40 | 0.53 |
| shared directions covered: word classes | 0.45 (shuffled classes 0.10) | 0.62 (shuffled 0.40) |
| all named properties | 0.53 | 0.69 |
| + 8 / 16 / 32 / 64 story-topic axes + valence | 0.60 / 0.65 / 0.70 / 0.72 | 0.73 / 0.76 / 0.79 / 0.81 |
| story-topic axes (32) + valence alone | 0.36 | 0.27 |

- The frequency weighting inflates coverage (shuffled classes already reach 0.40); the uniform numbers are the
  meaningful ones.
- **The shared directions that the named properties miss are semantic** (uniform weighting; named_features_directions.txt):
  valence (' broken', ' upset', ' wrong', ' bad', ' cry' vs ' enjoy', ' delicious', ' yummy', ' joy', ' wonderful'), food vs
  spectacle (' eat', ' dinner', ' soup', ' hungry' vs ' amazed', ' flying', ' castle', ' balloon'), bedtime and safety vs
  work (' dream', ' safe', ' sleep', ' bed' vs ' worked', ' idea', ' try', ' job'), outdoors vs treats (' boat', ' rocks',
  ' climbed' vs ' shop', ' candy', ' party'). Each is held by about four of the five seeds (eigenvalue 0.8).
- Story-topic axes (top eigenvectors of the corpus topic table: which words occur 9-24 tokens apart) close a large part
  of that gap (uniform 0.53 -> 0.70 with 32 axes). A single valence axis adds little on its own (0.02).
- **Head value codes** (the 32-dimensional code each head reads from a token): named properties cover 0.26-0.39 of a
  head's code (uniform). Single local heads match the best local head of another seed at 0.22 on average and other
  seeds' local heads together at 0.33; the local heads' union matches the other seeds' union at 0.30 (random 0.035).
  Far-looking (RECALL) heads match the best RECALL head of another seed at 0.28. Head codes are much less shared
  across seeds than the embedding as a whole. Local-head counts differ by seed (3, 1, 1, 3, 3 heads within 10 tokens).

## Reading

1. The model's token geometry is largely nameable: syntax (word classes), spelling and story topic together cover
   about 70% (uniform) of the directions all seeds share. The remaining shared part is not seed noise; its leading
   directions are recognisable semantic contrasts.
2. But naming the geometry is not enough for the program: what the heads write is a map from that geometry to the
   vocabulary, specific to each head, and the per-head swaps show these maps (tables) carry most of the gap, worst for
   the local heads. A program built on named token properties would still need each head's map; with ~260 named
   dimensions, a full map has tens of thousands of numbers (beyond "a few named scalars").
3. The program cannot be improved by importing model-faithful parts one at a time (M2): its parts co-adapt, as the
   model's do (the co-adaptation of attention and vote found in the look-follows-write study, here at the scale of the
   whole model).
