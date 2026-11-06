# Mechanism cards for attn_s0 (2026-10-13)

Workflow `mechanism-cards`: one agent per head plus a knowledge agent, CPU analysis of the exact model and the
TRAIN corpus (logs/decisions.md items 5-7). Each card: a human-readable description, a rule built from corpus
statistics + a few named scalars, tested by substituting it into the exact model (output KL vs the model).

## Circuit card: attn_s0 head 0, a recency-weighted reader of the current token and the last few tokens that votes bigram and skip-gram and suppresses repeats

Head 0 puts about half its attention on the current token and spreads the rest over the last few tokens, with attention falling exponentially with distance. In the weights, the position term is linear in distance, and the query token only changes the slope (an exactly rank-1 term). A 2-scalar rule (score = -0.21*d, +1.37 on the current token) substituted into the exact model gives dev_fast output KL 0.458, against 2.763 for uniform attention and 0.441 for the measured distance profile; 8 scalars (slope and self bonus by token shape) give 0.393, and 28 give 0.339. The votes are only partly corpus statistics: bigram for the current token and skip-gram for earlier tokens, plus a strong anti-copy that cancels the direct path's tendency (tied embeddings) to predict the current token again. Replacing the whole head with 8 scalars (2 attention, 6 vote) gives dev_fast KL 0.927, against 2.594 for removing it, so about 64% of the head's effect is recovered. The rest is query-key pair interactions in where it looks (mostly with the previous token) and OV structure that no corpus statistic reproduces.

### Rule

```
CARD (2-4 sentences): Head 0 reads the current token and the last few tokens. Attention odds drop by a factor of about 0.81 per token back, and the current token gets an extra 3.9x. What it reads from the current token is 'what usually comes next' (corpus bigram) plus a strong veto on repeating that token; this veto cancels the direct path, which with tied embeddings votes for the current token again. What it reads from an earlier token at distance d is 'what follows that token d+1 later' (corpus skip-gram) plus a weaker veto on repeating it.

# Token shape (spelling only): WORD_START = leading space + letter; CONTINUATION = starts with a letter (word piece);
# PUNCT_OTHER = everything else; EOS = token 0.
# Corpus tables (TRAIN stream, stories joined by token 0): L_k[s][v] = log(P(token k later = v | s) + 1e-4), centred
# over v; L_1 is the bigram. LU[v] = log(P(v) + 1e-4), centred.

def head0(tokens, i):
    # WHERE IT LOOKS (rule A, 2 scalars)
    for j in 0..i:
        d = i - j
        score[j] = -SLOPE * d + (SELF_BONUS if d == 0 else 0)
    a = softmax(score)
    #  rule B (8 scalars): SLOPE and SELF_BONUS depend on the current token's shape:
    #     SLOPE    WORD_START 0.275 | CONTINUATION 0.365 | PUNCT_OTHER 0.105 | EOS 0.242
    #     SELF     WORD_START 1.27  | CONTINUATION 0.46  | PUNCT_OTHER 1.52  | EOS 3.50
    #     (after punctuation it looks about 2.6x further back; at end of story it looks at itself)
    #  rule C (28): add key-shape penalty for d>=1 (WORD_START -1.27, CONTINUATION -1.57, PUNCT -0.38, EOS -0.07)
    #     and, when the current token is a word piece, +1.33 for its word's first piece and +1.07 for earlier pieces of the same word.
    # WHAT IT VOTES FOR (logits; times sdf0/sdf_i, the typical over the actual final-LN scale)
    vote  = a[i] * ( B_SELF * L_1[t_i] - K_SELF * onehot(t_i) + U_SELF * LU )
    for j < i:  d = i - j
        vote += a[j] * ( B_SKIP * L_{min(d+1,8)}[t_j] - K_EARLIER * onehot(t_j) + U_EARLIER * LU )
    return vote

Scalars (rule A + votes, fitted on TRAIN windows seed 12): SLOPE 0.21, SELF_BONUS 1.367; B_SELF 1.36, K_SELF 16.86, U_SELF -0.89; B_SKIP 1.73, K_EARLIER 9.04, U_EARLIER -0.85. Fixed constants: log floor 1e-4, sdf0 1.127.
```

### Scalars

- SLOPE = 0.21 per token (rule A; attention odds x0.81 per step back); exact position term about 0.33 per token at typical LN1 scale
- SELF_BONUS = 1.367 (rule A; current token x3.9)
- rule B SLOPE by current-token shape: WORD_START 0.275, CONTINUATION 0.365, PUNCT_OTHER 0.105, EOS 0.242
- rule B SELF_BONUS by shape: WORD_START 1.272, CONTINUATION 0.455, PUNCT_OTHER 1.524, EOS 3.495
- rule C (28): slope [-0.280,-0.274,-0.105,-0.217], self [0.162,0.094,0.452,2.581], key-shape penalty d>=1 [-1.266,-1.574,-0.380,-0.069], same-word bonus for word-piece queries: key WORD_START +1.333, key CONTINUATION +1.068
- votes (fitted with rule A attention): B_SELF 1.36, K_SELF 16.86 (anti-copy), U_SELF -0.89, B_SKIP 1.73, K_EARLIER 9.04 (anti-copy), U_EARLIER -0.85
- votes (fitted with real attention): self bigram 1.25, copy -23.5, unigram -0.61; earlier skip 1.16, copy -4.00, unigram -0.46
- fixed: log floor eps 1e-4; typical final-LN scale sdf0 1.127; skip-gram distance capped at 8

### Measurements

- ATTENTION STATS (64 TRAIN windows, seed 11): mass by distance d=0 0.489, d=1 0.162, d=2 0.086, d=3 0.059, d=4 0.041, d=5 0.031, beyond 40 0.004; mean distance 2.56; mean entropy 1.47 nats
- self weight by current-token shape: WORD_START 0.536 (n=10906), CONTINUATION 0.408 (2703), PUNCT_OTHER 0.366 (2706), EOS 0.919 (69); self weight quantiles 5/25/50/75/95%: 0.145/0.329/0.488/0.646/0.821
- weight on previous token (query shape <- key shape): CONTINUATION<-WORD_START 0.338, PUNCT<-PUNCT 0.290, WORD_START<-PUNCT 0.229, WORD_START<-WORD_START 0.152, CONTINUATION<-CONTINUATION 0.168, WORD_START<-CONTINUATION 0.098, PUNCT<-WORD_START 0.048
- attention on the current word's pieces: 0.690 / 0.732 / 0.789 when the word has 1 / 2 / 3 earlier pieces
- QK covariance shares of the row-centred score, uniform over keys / attention-weighted: position 0.849/0.768, key salience 0.009/-0.020, query-token x key-position 0.135/-0.214, token x token 0.007/0.466 (Ebar from the window tokens)
- QK term removal, output KL (TRAIN 64): no position 1.453, no key salience 0.213, no query x position 0.057, no token x token 1.177, position only 0.883, LN1 scales set constant 0.025
- position term (exact, relative to self): linear in distance, about -0.0185 per token raw, about -0.33 per token at the typical LN1 scale 0.235 (std over query positions grows from 0.05 at d=1 to 0.5 at d=20, in s03 units); key position 0 penalised by -15 to -25 at early queries (|WP[0]| = 7.45 vs median 1.33)
- query-token x key-position term: first SVD component carries 1.00 of the energy and is linear in d, so the query token only changes the decay slope; PUNCT/EOS queries flatten it, CONTINUATION steepens it
- error split (TRAIN, rule B): real self weight + rule's earlier keys 0.321; rule's self weight + real earlier-key distribution 0.140; rule alone 0.401
- pair interactions (exact token-level tables, interaction replaced by its per-query mean): none 0.236, exact for d=1 only 0.097, d<=2 0.062, d<=4 0.039, d<=8 0.024, same-word keys only 0.155
- corpus statistics vs exact interaction with the previous token (frequency-weighted correlation over 132,880 bigrams): PMI 0.382, trigram information gain 0.192, log P(cur|prev) 0.108, cur-is-word-piece 0.334
- attention rules with corpus features (TRAIN): 28-param base 0.341 (pattern KL 0.393); +PMI 0.337; +key informativeness 0.331; +log freq 0.339; +all three 0.325; +PMI by query shape 0.313 (pattern KL 0.350); 64-class tables: 133 params 0.322, 243 params 0.277
- PATTERN SUBSTITUTION, dev_fast 200 windows (exact OV kept), output KL: uniform 2.763; measured distance profile 0.441; rule A (2) 0.458 (pattern KL 0.477); rule B (8) 0.393 (0.413); rule C (28) 0.339 (0.370). TRAIN 64: 2.674 / 0.449 / 0.470 / 0.401 / 0.341; free 50-bin distance profile 0.448
- OV vs corpus (frequency-weighted row correlation): bigram 0.380, skip-2 0.334, skip-3 0.305, skip-4 0.286, unigram 0.326; regression on bigram+skip2+skip3+unigram R2 0.205 (adding the direct-path table: 0.243)
- copying: the attended token is never its own top vote (frequency-weighted share 0.0); OV[s,s] is 9.05 below the row mean (row std 4.76); the direct path's DP[s,s] is 8.78 above its row mean
- vote rules with REAL attention, whole head contribution replaced in logit space (TRAIN): head removed 2.552; unigram 2.420; bigram of attended token 1.326; bigram(self)+skip(earlier) 1.118; +bigram(earlier) 1.101; +copy 0.919 (copy weight -11.6); head's own position-averaged OV table x real pattern 0.041
- split by part (TRAIN, real attention): EARLIER part zero 0.475, mean 0.441 (harness 'earlier' ablation on same windows 0.480), rule 0.296 (skip 1.10, copy -3.39, unigram -0.57); SELF part zero 1.893, rule 0.569 (bigram 1.24, copy -21.4, unigram -0.42); both parts by rule 0.896
- full replacement with rule attention (TRAIN): rule A 0.935, rule C 0.947, real attention 0.896; earlier part with rule A attention 0.354
- FULL REPLACEMENT, dev_fast 200 windows: head removed 2.594; earlier part mean-ablated 0.423; real pattern + vote rule 0.892; rule A pattern + vote rule 0.927; top-1 agreement with model 0.544 (head removed 0.200)
- full replacement KL by current-token shape (TRAIN): WORD_START 0.921 (66% of total), CONTINUATION 0.997 (18%), PUNCT_OTHER 0.948 (17%), EOS 0.189
- window position 0: full-replacement KL 3.81 vs 0.924 at positions 1-255 (adds 0.015 to the mean)
- self part + direct path replaced together by 0.73*bigram + 1.29*copy - 0.24*unigram: KL 0.489 (self part alone by rule 0.569); removing self+direct 1.189, self part alone 1.893, direct path alone 0.446

### Failures and dead ends

- Where it looks: pair interactions between the current token and specific earlier tokens, mostly the previous one, are not in the rule. In the exact decomposition they are worth 0.236 -> 0.097 nats (d=1) and 0.024 (d<=8). Corpus PMI and trigram information gain barely predict them (correlation 0.38 and 0.19), and adding them to the rule gains only 0.341 -> 0.313.
- Where it looks: token-level key salience, self bonus and slope vary within shape and class groups. Exact token tables without pair terms reach 0.236; shape or class rules reach 0.34-0.40, and 64-class tables with 243 parameters reach 0.277.
- Votes: the head's OV table is not a corpus statistic (R2 <= 0.24). Corpus votes leave 0.85 nats against the head's own OV table with the real pattern (0.892 vs 0.041). Part of the OV cancels the direct path (tied embeddings: the direct path votes +8.8 for the current token, head 0 -9.05), and part presumably complements heads 2/3.
- Inside split words the model completes words the rule misses: 'He so' -> 'ared' (KL 10.4), 'the ge' -> 'e' (10.2), 'The whe' -> 'el' (9.6), 'De' -> 'e' (8.8).
- Window position 0 artifact: the model predicts the current token again (' sand' -> ' sand' p=0.75, ' The' -> ' The', KL up to 15.2). Position 0 has a huge position embedding (norm 7.45), which presumably weakens head 0's anti-copy there. KL is 3.81 at position 0 vs 0.92 elsewhere, adding 0.015 to the mean.
- Multi-byte characters: 'â' -> '€' (model p=0.97, KL 9.0). Speech tags: '"I hurt my paw," Benny said' -> ',' (rule says ' replied', ' explained').
- The 28-parameter attention rule (C) improves pattern substitution (0.339 vs 0.458) but not full replacement (0.947 vs 0.935), because the vote rule's error dominates.
- s02 covariance shares used Ebar from the 64 windows' tokens, not the corpus unigram (it ran before counts.npz existed); the rule fits and KLs do not depend on Ebar.

## Circuit card, attn_s0 head 1: recall an earlier word that could come next, and vote for it and its topic

Head 1 is the model's long-range memory. It looks back with an exponential recency decay (tau 33.6 tokens, so the weight per token halves every ~23 tokens; mean distance 26). It prefers names and content words over word endings, linkers and sentence ends. Above all it prefers earlier tokens of the kind that can follow the current token: names and sentence starters after '.', '\n' or ',', nouns after 'the' or 'his', speech verbs after a name, word continuations after an onset. Each attended token votes for itself (copy) and for words that co-occur with it in stories (dog -> owner/tail/Spot, Lucy -> she/her, '.' -> She/They). On dev_fast, a 9-scalar attention rule built from corpus statistics removes 57% of the head's mean-ablation KL (0.444 -> 0.193) and a 15-scalar vote rule removes 50% (-> 0.221). Substituting both removes only 30% (-> 0.312), mainly because the query-key preference is token-level (the exact token-token term reaches 0.066 on TRAIN windows) and because of a word-completion mode inside multi-token words.

### Rule

```
CORPUS TABLES (TRAIN stream; no model):
  P_bigram(k|q)   bigram counts (data/baselines/bigram_counts.npy), Dirichlet pseudo-count 10 toward the unigram
  classPPMI(q,k)  = max(0, log P(TARGETclass(k) | CONTEXTclass(q)) - log P(TARGETclass(k)))   (bigram counts pooled by class)
  P_ic(k|q)       = Nhit/Nctx over positions whose current token is q and where k occurred among the last 64 tokens:
                    the fraction where the next token is k (pseudo-count 5 toward P_bigram)            [first 50M tokens]
  PMI_far(s,v)    = log P(v sits 9..24 tokens after the most recent s) - log P(v)  (Dirichlet 100)   [first 50M tokens]
  PMI_next(s,v)   = the same at distance 1 ('what follows s')
GROUPS (unions of ATTENTION classes): NAMES {NAMES, NAMES_UNSPACED}; SENTENCE_ENDS; WORD_ENDS_AND_LINKERS {FRAGMENTS_H/G/D/ENDINGS/VC/SHORT,
  SUFFIXES_PLURAL, SUFFIXES_ING_ED, LETTERS_LOWER, LINKERS, PREPOSITIONS}; FUNCTION_WORDS {SUBJECT_PRONOUNS, OBJECT_PRONOUNS, LINKING_VERBS,
  PARTICLES, DETERMINERS, MODALS, TIME_WORDS, SENTENCE_STARTS, ADVERBS}; CONTENT_WORDS = all other classes.

def head1_logits(tokens, sdf):            # sdf[i] = final-LayerNorm scale (taken from the model in every test below)
    for i, q in enumerate(tokens):
        for j in range(i + 1):            # the current token is included
            k, d = tokens[j], i - j
            score[j] = (-d / TAU                                   # recency
                        + SAL[group(k)]                            # names up; word endings, linkers, sentence ends down
                        + B_CLASS * classPPMI(q, k)                # k is the KIND of word that follows q
                        + B_IC * max(log P_ic(k | q), -6))         # k really did follow q in stories where k had just appeared
        w = softmax(score)
        out[i] = sum_j w[j] / sdf[i] * ( COPY[g_j] * onehot(k_j)            # repeat the attended word
                                       + TOPIC[g_j] * PMI_far(k_j, .)       # and words of its topic
                                       + NEXT[g_j]  * PMI_next(k_j, .) )    # sentence ends: vote for what follows them
    return out          # added to the logits in place of head 1's exact contribution
```

### Scalars

- TAU = 33.6 tokens (recency: score -d/TAU; weight per token halves every ~23 tokens). Fitted by pattern KL; exp-distance-only fit gives 33.9.
- KAPPA_SELF = -0.01 (self bonus at d=0, effectively 0 in the 5-group rule).
- SAL relative to CONTENT_WORDS: NAMES +0.72, FUNCTION_WORDS -0.30, WORD_ENDS_AND_LINKERS -0.93, SENTENCE_ENDS -0.94.
- B_CLASS = 0.34 (weight of class PPMI successor term).
- B_IC = 0.099 (weight of max(log P_ic(k|q), -6)).
- Votes (COPY, TOPIC on PMI d9-24, NEXT on PMI d1), logits per unit attention before /sdf: NAMES (4.77, 2.27, -0.27); SENTENCE_ENDS (1.93, -0.48, 0.63); WORD_ENDS_AND_LINKERS (1.61, 2.86, -0.12); FUNCTION_WORDS (5.57, 1.77, -0.05); CONTENT_WORDS (3.21, 2.83, 0.08).
- Total: 9 attention scalars + 15 vote scalars = 24. Optional 6-group variant adds WORD_ONSETS (salience -0.25; votes 8.65 / 2.57 / 0.70) and per-query-group self/previous bonuses (12 numbers).

### Measurements

- Numpy float64 forward pass vs torch fp32 model: max |logp diff| 8.7e-5, KL 5.8e-12 (window 0).
- Head-1 attention mass by distance (100 TRAIN windows, seed 11): d0 4.4%, d1 3.7%, d2-7 18.6%, d8-31 43.0%, d32-127 29.6%, d>=128 0.8%; mean distance 26.2; entropy 3.37 nats; mean max weight 0.17; mass on end-of-story tokens 3.1%.
- Mean weight per available key by distance: d0 4.37%, d1 3.71%, d40 0.95%, d48 0.75%, d64 0.46%, d96 0.17%, d128 0.063%, d192 0.009%. A free distance profile in score units, relative to d0: -0.36 at d1, -0.6 at d12, -2.26 at d64, -4.84 at d128, -11.1 at d255.
- Attention share relative to uniform over available keys, by key class: NAMES_UNSPACED 3.18x, NAMES 2.47x, OPENERS_UNSPACED 2.08x, CREATURES 2.06x, SENTENCE_STARTS 1.48x; PREPOSITIONS 0.42x, LINKERS 0.53x, fragments and suffixes ~0.55x, SENTENCE_ENDS 0.59x, DETERMINERS 0.59x.
- Exact score split score = [(E_q+P_i) B (E_k+P_j)] / (sd_i sd_j). Within-row covariance shares: token-token 19%, query-token x key-position 36%, query-position x key-token 5%, position-position 39%. Removing each term (replaced by its row mean): pattern KL 0.56 / 0.12 / 0.12 / 0.15; output KL 0.236 / 0.024 / 0.041 / 0.031.
- Additive least squares on real scores (3.29M pairs; within-row R2): distance profile 0.679; + per-token key salience 0.777 (with 64-class salience instead: 0.735); + exact token-token interaction 0.895; + distance profile per query class 0.926. A free per-row scale times (distance + salience) gives 0.855; scale quantiles 10/50/90% = 0.40/0.97/1.36. Flattest queries are single-letter onsets (' p', ' c', ' g'); sharpest are ' named', ' came', ' It', ' in'.
- Key salience vs corpus: unrelated to burstiness (re-use ratio, Pearson -0.03); word-start tokens +0.87 vs fragments; with log-frequency, R2 0.35.
- Least-squares R2 gain on top of distance + key salience (0.777), by interaction feature: exact token-token term +0.140; bigram log P(k|q) +0.016; log P(TARGETclass(k)|q) +0.018; in-context successor log P_ic +0.014; model's own direct-path prediction +0.010; expected exact-OV vote under the bigram +0.014; reverse bigram and co-occurrence PMIs <= +0.001.
- Token-token interaction favours each query's top-10 bigram successors: z = +1.18 (positive for 94% of 400 frequent queries); predecessors z = +0.07; fragment successors +1.62, word-start successors +0.87.
- Corpus fact (first 50M TRAIN tokens): 44.9% of next tokens (22,453,822 of 49,999,999) already occurred among the previous 64 tokens. Re-use ratio P(next=v | last v was d back)/P(v), geometric mean: d=1 e^-5.64, d=2 e^-3.01, d3-4 e^-0.28, d5-8 e^0.70, d9-16 e^1.13, d17-32 e^1.27, d33-64 e^1.15, d65-128 e^0.75, d129-256 e^0.32; names reach e^2.7 at d17-64.
- Attention substitution in the exact model, output KL vs model (fit on 60 TRAIN windows seed 11, test on 50 TRAIN windows seed 12). References: mean ablation 0.466, earlier-token ablation 0.426, zero 0.466. Patterns: uniform over j<=i 0.322; uniform over j<i 0.329; free distance profile 0.264; exp distance 0.265 (tau 33.9); + 64-class salience 0.232; + per-token salience 0.226; + bigram log(P+1e-4) 0.214; + log P(TARGETclass|q) 0.2035; + max(log P_ic,-6) 0.193; + class PPMI and P_ic 0.191; + model self-prediction 0.218.
- Ceilings for the same test: free class-pair bonus tables 0.159 (ATTENTION x ATTENTION, 4096 numbers) and 0.133 (CONTEXT query x TARGET key, 4544 numbers); exact token-token interaction 0.066 (with exp distance + class salience) and 0.034 (with free distance + per-token salience).
- Strongest fitted CONTEXT->TARGET pair bonuses (weighted by attention mass): SENTENCE_ENDS->SENTENCE_ENDS +2.12, SENTENCE_ENDS->NAMES +1.56, THE_AND_LITTLE->ONSETS_1 +2.14, LINE_STARTS->NAMES +1.39, SENTENCE_ENDS->NAMES_UNSPACED +2.21, THE_AND_LITTLE->ANIMALS_AND_OBJECTS +2.53, COMMAS->NAMES +1.26, POSSESSIVES->FAMILY_AND_BODY +2.26, NAMES_AND_FAMILY->SPEECH_AND_REACTION_VERBS +1.66, ONSETS->FRAGMENTS +1.8..+2.7, HE_AND_SHE->PAST_VERBS +0.96, TO_AND_MODALS->BARE_VERBS_1 +1.09. Weighted correlation with the corpus class bigram: 0.386.
- OV from the weights (vote per unit attention, before /sdf): variation across key positions is 7% of token variance. A token is its own top vote for 79.9% of tokens (freq > 2e-6; 46.9% frequency-weighted); self vote is +16.4 above the row mean (z 4.7). Copying is weak for LINKERS (self 6.2, top-1 27%), OPENERS_UNSPACED (6.3, 21%), PREPOSITIONS (7.2, 33%), NAMES_UNSPACED (8.7, 17%). Same-word variants: median rank 361 (+3.05 vs self +15.85).
- OV associations: female names vote she/her/She/Her +2..+5 and he/his -3..-5 (e.g. ' Lucy': she +4.39, his -5.41); male names the reverse (' Bob': he +4.17, her -3.86). ' dog' -> owner 11.9, Spot 11.1, tail 10.6; ' tree' -> ' bran' 18.3 (above self 16.6), leaves; ' cake' -> ' bir', ' fro', party; ' red' -> blue, yellow; '.' -> They 15.7, She 14.9, We, You, The.
- OV rows vs corpus (1342 source x 1744 target tokens, mean row R2): copy one-hot 0.012; PMI d25-64 0.119; PMI d9-24 0.102; bigram 0.010; copy + all distance bands 0.179.
- Vote substitution with exact attention (logit space; fit on 20-30 TRAIN windows, test on 50): no head-1 logits 0.4435; constant mean vote 0.4498; copy only 0.364 (9.87/sdf); copy + PMI d9-24 0.259-0.266; 5 named groups x (copy, topic, next) 0.227; 64 classes x 3 (192 numbers) 0.2105; 64-class copy + PMI d25-64 0.2914 (vs 0.2953 with one copy scalar).
- FINAL dev_fast (200 windows), output KL vs model, with the fraction of mean-ablation KL removed. References: mean ablation 0.444; earlier-token ablation 0.410 (8%); zero ablation 0.449; logit-space zero 0.427. Uniform attention 0.308 (31%).
- FINAL dev_fast, attention rules: QK-dist (2 scalars) 0.258 (42%, pattern KL 0.734); QK-sal5 (7) 0.235 (47%, 0.652); QK-full (9) 0.193 (57%, 0.592); 64-class salience version (68 numbers) 0.185 (58%, 0.557); 6 groups + local self/previous bonuses 0.182 (59%, 0.578).
- FINAL dev_fast, votes and combined: vote rule with exact attention, 5 groups 0.221 (50%), 6 groups 0.2145. Both substituted: QK-full + votes5 0.312 (30%); 6-group version 0.303; QK-dist + votes5 0.354 (20%); QK-full + votes5 with 1/sdf replaced by its per-window mean 0.319. Logit-space check: QK-full attention + exact OV gives 0.194, vs 0.193 in the exact-model substitution.
- Local bonuses (6-group fit, total self bonus = global +0.37 + group term): SENTENCE_ENDS queries +1.74 and WORD_ONSETS queries +1.55 attend to themselves; NAMES -1.62 and FUNCTION_WORDS -0.62 avoid themselves. Onset votes: copy 8.65, topic 2.57, next 0.70.

### Failures and dead ends

- Worst dev_fast positions with both rules substituted, all with KL 5-9 nats, are word completion inside multi-token words: ' prom' -> 'ise' (the real head puts 0.71-0.79 on the current ' prom'; the rule spreads attention), 'stra' -> 'ight' (head 0.63 on the previous ' st'), 'pract' -> 'iced', ' inst' -> 'ead', 'perfu' -> 'med', 'siz' -> 'ss' (head attends to four earlier 'ss' from 'chess'). Adding a WORD_ONSETS group with self/previous-token bonuses helped little (attention-only 0.193 -> 0.182, both 0.312 -> 0.303).
- 'a ... named Sue': at the name, the real head puts 0.94 on ' named' (model predicts '.'). The rule attends to ' Sue' itself and copies it (rule P(' Sue') = 1.00), KL 5.75.
- Multi-byte characters and rare fragments ('€' -> 'œ'; a story-initial 'X' attended with 0.77 at distance 32) are not handled.
- Corpus successor statistics capture only ~25% of the token-token interaction's value: (0.232 - 0.191) / (0.232 - 0.066). The class-pair table ceiling captures 60%. The rest is token-level, and part of it is not a successor relation (sentence ends attend to earlier sentence ends, +2.1, the strongest class-pair entry).
- Attention and vote errors compound: each rule alone removes 50-57% of the mean-ablation KL, but together they remove 30%.
- Negative results: (1) key salience is not corpus burstiness (r = -0.03); (2) the model's own direct-path next-token prediction is a worse interaction feature (0.218) than the corpus in-context statistic (0.193); (3) the expected-vote feature sum_v P_bigram(v|q) OV[k,v] explains R2 +0.014 only; (4) per-class copy strengths (64) add nothing over one copy scalar (0.2914 vs 0.2953); (5) a constant mean vote does not help in logit space (0.4498 vs 0.4435).
- Bug: the class x class table printout in s9_inter.py is wrong (huge negative R2 values from a weighting error). It is superseded by the KL-fitted tables in s19_pairs.py/s20_pairtable.py. Its SVD part is valid.

## Circuit card: attn_s0 head 2 reads the local context and votes with corpus skip-grams

I wrote a 15-scalar rule for head 2 built only from training-corpus counts and the token text: 12 scalars for where it looks, 3 for what it votes. The attention half works well: putting the rule's pattern into the exact model gives output KL 0.330 on 64 train windows (set B). For comparison the plain distance profile gives 0.533, uniform gives 1.385 and mean ablation 1.340. The vote half is weaker: corpus skip-grams with the real attention give 0.689, and the full rule (rule attention plus rule votes) gives 0.765, about 43% of the head's mean-ablation cost recovered. The tests point to two causes: head 2 shares the bigram work with head 0, and it makes sharp word-pair-specific attention choices that neither word classes nor trigram statistics capture.

### Rule

```
CARD: head 2 = local-context reader (scripts in analysis/scratch/h2card/; fits on train set A, all reported numbers on train set B, 64 windows)

In plain words: Head 2 reads the current token when that token is an unfinished word or strongly predicts what comes next (onsets like ' cr', ' gra', ' sh', ' e'; ' lots'; modals). Otherwise it looks back over the last few tokens. Weights fall by about 25% per token. It prefers sentence ends and informative words, and it skips word onsets and word pieces, so inside a split word it reads the words before that word. Each token it reads votes for what the training stories put at the next position, given that token and its distance: the bigram for the current token, skip-grams for earlier tokens.

Corpus features (counted from data/tokens/train.bin, stories separated by token 0, pairs may cross story boundaries):
  info0(s)  = KL( P(next | s) || P(next) )                       how strongly s predicts its successor (nats)
  info_d(s) = KL( P(token d+1 after s) || P(token) ), d = 1..7    (distance-8 value used for d >= 8)
  cont(s)   = P(next token is a word piece | s)                  "s is an unfinished word"
  piece(s)  = text of s starts with a lowercase letter (no space)
  space(s)  = text of s starts with a space
  sentend(s)= s in { . ! ? ." !" ?" ". }

WHERE IT LOOKS (query position i, key position j <= i, d = i - j):
  if d == 0:  score = -0.34 + 0.45*info0(t_i) + 0.67*cont(t_i) - 1.19*piece(t_i)
  else:       score = -0.29*min(d-1, 11) - 0.11*max(d-12, 0) - 0.06*[d == 1]
                      + 0.50*info_d(t_j) - 1.28*cont(t_j) - 1.22*piece(t_j) - 0.34*space(t_j) + 0.89*sentend(t_j)
  a_ij = softmax over j <= i of score

WHAT IT VOTES FOR (added to the logits of position i):
  vote_i[v] = (1/sdf_i) * sum_j a_ij * ( beta_d * log P(x_{j+d+1} = v | x_j = t_j) + gamma * log P(v) )
  beta_d = 1.0 for d < 8; beta_far = 0.4 for d >= 8 (uses distance-8 counts); gamma = -0.5 (half-PMI discount on frequent tokens)
  P = TRAIN pair counts, Dirichlet-smoothed towards the unigram (kappa = 50); sdf_i = the model's final-LayerNorm scale (kept exact in the tests)

Scalars (15): the 12 attention scalars were fitted to the real head-2 patterns by attention cross-entropy (32 windows of set A). The 3 vote scalars came from a grid search on output KL with the real attention (16 windows of set A).
```

### Scalars

- self_base = -0.34 (score of the current token before features)
- self_info = 0.45 (self score per nat of info0, how strongly the current token predicts its successor)
- self_cont = 0.67 (self score x P(next is a word piece | current token))
- self_piece = -1.19 (a word piece does not read itself)
- decay = 0.29 per token for d = 1..12 (weight ratio about 0.75 per token)
- decay_far = 0.11 per token beyond d = 12
- prev_bonus = -0.06 (previous token; about 0)
- key_info = 0.50 (key score per nat of info_d, the key's skip-gram informativeness about the next token)
- key_cont = -1.28 (unfinished-word keys, i.e. onsets, are skipped)
- key_piece = -1.22 (word-piece keys are skipped)
- key_space = -0.34 (keys starting with a space)
- key_end = +0.89 (sentence ends . ! ? ." !" ?" ". are salient keys)
- beta = 1.0 (vote strength for log P(token at distance d+1 | key), d < 8)
- beta_far = 0.4 (vote strength for d >= 8, distance-8 counts)
- gamma = -0.5 (vote weight on log unigram, a half-PMI discount)
- Fixed choice, not fitted: smoothing kappa = 50 (Dirichlet towards the unigram)

### Measurements

- Real head-2 attention (64 train windows, set A): share of attention mass by distance d = 0, 1, 2, 3, 4, 5 is 0.340, 0.205, 0.125, 0.080, 0.057, 0.043; 0.039 lies beyond d = 12. Mean distance 2.88, entropy 1.51 nats per row.
- QK exact split (set A, 64 windows), output KL / pattern KL when one term is dropped. Token-token (TT): 0.759 / 0.478. Position-position (PP): 0.423 / 4.78. Query-position x key-token (PT): 0.107 / 0.127. Query-token x key-position (TP): 0.038 / 0.164. Keeping only PP: 0.707; only TT: 0.801.
- PT is pure key salience: replacing it by its average over query positions costs output KL 0.0036 (set A).
- Token-token term (set A): with key salience kept but all interaction removed, output KL is 0.920. Adding back only the same-token (diagonal) interaction gives 0.222; this is mostly the self-attention bonus. Set B: exact scores minus the off-diagonal interaction give 0.220, and adding a 64x64 class-pair interaction table only reaches 0.188.
- LN1 scale: sd(e+p) is about 0.25 at positions 1 and later (std over tokens about 0.03) and 0.67 at position 0. The factor 1/(sd_i*sd_j), about 16, turns a raw positional score slope of about -0.02 per token into about -0.3 per token.
- Pattern substitution into the exact model, set B (64 windows), output KL (pattern KL): uniform over earlier tokens 1.385 (3.05); empirical distance profile 0.533 (0.621); fitted free distance profile 0.539. Adding self features gives 0.408, adding key features 0.354, adding key informativeness 0.330. The compact 12-scalar rule gives 0.330 (0.457), top-1 agreement 0.707. Adding 64 per-class self bonuses and 64 per-class key saliences gives 0.247; those class tables without the corpus features give 0.278.
- The compact rule reproduces summary statistics: mean self-attention 0.345 (real 0.343), mean distance 2.881 (real 2.883).
- Self-attention (365 query tokens with at least 8 occurrences in set A): corr(logit a_ii, info0) = 0.644; with cont 0.430; with H(next|t) -0.134. Mean a_ii by class: ONSETS_SINGLE 0.87, ONSETS_A 0.72, DETERMINERS_RARE 0.70, MODALS 0.65, LINKERS 0.41, DETERMINERS 0.18, NAMES 0.16, PLACES 0.10, OBJECTS 0.09, CREATURES 0.06.
- Highest-salience keys (score units): ' From' 2.2, 'When' 1.9, ' As' 1.8, ' After' 1.8, '!' 1.75, '.' 1.7, ' moral' 1.7. Lowest: word onsets and pieces ('sel' -10.4, 'ri' -3.8, 'l' -3.7, ' pe' -3.0).
- OV is position-independent: OV rows at positions 1/5/50/200/255 correlate 0.974 to 0.997 with the mean-position OV. Replacing the exact votes with the mean-position token table under the real attention costs 0.037 (set A).
- OV vs corpus (frequency-weighted mean row correlation over frequent tokens): bigram log P 0.436; skip-gram at distance 2: 0.351, at 3: 0.318; unigram 0.311. Head 2 does not copy: the attended token is the top vote for 0.1% of frequent tokens (median rank 1454 of 2000). The OV diagonal sits 2.72 logits below the row mean, and 70% of diagonal entries are negative.
- Vote substitution with the real attention, set B (64 windows), output KL: zero 1.444; mean ablation 1.340; distance-matched skip-gram rule 0.689 (top-1 0.583); 9 per-distance betas fitted (0.88 to 1.34, tail 0.41, gamma -0.54) also 0.688; bigram for every attended token 1.077; token-specific mixture of distances 0.806.
- Vote rule with substituted attention, set B: with the uniform pattern 1.400; with the distance profile 0.837; with the compact attention rule (full head-2 replacement) 0.765 (SE 0.009, top-1 0.557).
- Head-2 parts, set B: earlier-token part from the rule (self part exact) 0.354, vs 0.559 with the earlier part zeroed. Self part from the rule (earlier part exact) 0.308, vs 0.735 zeroed.
- Division of labour, set B (32 windows), self votes replaced by beta*a_ii*log P(v|t_i): head 2 alone 0.329 vs 0.750 zeroed (56% explained); head 0 alone 0.785 vs 1.920 (59%); heads 0 and 2 together 0.898 vs 3.157 (72%).
- Smoothing scan for the vote rule (set B, 32 windows): kappa 5: 0.720; kappa 500: 0.709; kappa 5000: 0.705.
- Anti-copy term in the vote rule (set B, 32 windows): best 0.693 vs 0.705 without it. The token-embedding direct path pushes the current token 8.35 logits above its row mean; head 2's net push on it is +0.37.
- Trigram conditional gain KL(P(next | t_i, t_{i-d}) || P(next | t_i)), from the first 80M train tokens, count-weighted mean: d=1 0.834, d=2 0.749, d=3 0.613 nats. As an extra attention feature it moves set-B output KL only from 0.330 to 0.321; distance profile plus gain alone gives 0.512.
- Where the attention rule's KL falls (set B): LINKERS carry 17% of the total KL (0.47 per token); FRAGMENTS_VC 0.80, LETTERS_LOWER 0.82, FRAGMENTS_SHORT 0.67 per token. Pattern KL is worst for FRAGMENTS_MIXED 0.81, QUOTE_OPENERS 0.68 and SENTENCE_STARTS 0.66, and best for ONSETS_C 0.16, ONSETS_SINGLE 0.21 and MODALS 0.26.
- Numpy forward check against the torch model: max |log-prob difference| 1.3e-4.

### Failures and dead ends

- The attention rule is too soft where the head makes sharp, word-pair-specific choices. Examples: 'take a' reads ' take' at 0.82 (rule 0.28); ' to' reads itself at 0.95 (rule 0.34); the word piece 'ful' in ' was che-er-ful' reads ' was' at 0.55 (rule 0.15); '\n' after '!"' reads '!"' at 0.36 (rule 0.61).
- The off-diagonal token-token interaction is worth up to 0.22 nats of output KL, and neither tested statistic captures it. A 64x64 class-pair table recovers only 0.220 -> 0.188. The trigram conditional-information gain recovers only 0.330 -> 0.321 (dead end).
- The skip-gram votes explain about half of head 2's votes: 58% of the self part and 37% of the earlier-token part relative to zeroing that part. The full rule recovers 43% of the mean-ablation cost (0.765 vs 1.340).
- Head 2's votes come from one token table shared by all distances, which reproduces the head to 0.037, and that table is split with head 0. The corpus bigram explains the combined self votes of heads 0 and 2 (72%) better than either head alone (56% / 59%). A rule written for one head cannot reproduce how the model splits the bigram between heads 0 and 2.
- Worst vote-rule positions, all inside split words: 'Emm' -> model 'ily' 0.90, rule ' was' 0.25 (KL 10.8); 'wind started to pick up. Emm' (KL 9.5); 'very strong' -> model ' and' 0.68, rule 'er' 0.99; '<|eos|>There' -> model ' was' 0.65, rule 'ad' 0.95; ' exc' -> model 'iting' 0.70, rule 'l' 1.00; ' sh' -> model 'in' 0.87, rule ' sh' 0.99.
- Dead ends: (a) voting with a token-specific mixture of distances, i.e. what follows s at the distances from which head 2 reads s: 0.806, worse than distance-matched 0.689. (b) Bigram votes for every attended token: 1.077. (c) Smoothing strength kappa barely matters (0.705 to 0.720). (d) An anti-copy scalar helps little (0.705 -> 0.693).
- The vote tests multiply by the model's own final-LayerNorm scale 1/sdf_i. A program would need its own rule for that scale.
- The attention scalars were fitted to the patterns, not to output KL, so fitting them on output KL might trade pattern accuracy for fidelity differently.

## Circuit card: attn_s0 head 3, a short-range "clause context" head

Head 3 is a short-range head whose attention depends on the kind of word at the current position. Nouns and most content words look back to the latest clause boundary ('.', ',', ' and', a sentence-opening word), whose votes are the weakest, so these tokens largely abstain. Linking words look at the word just before them, often a verb, and word pieces, function words and pronouns attend to themselves. The head then votes for what usually follows the attended token at that distance. A readable rule (27 fitted scalars over 7 hand-named word groups for attention, plus 3 scalars on TRAIN-corpus skip-gram PMI for the votes) reaches output KL 0.436 against 0.529 for mean ablation on 80 TRAIN windows, so it recovers only about 18% of the head's effect: the attention rule alone recovers 46% (0.287) and the vote rule alone 25% (0.395). What the rule misses is token-specific query-key matching spread over 8 to 16 of the head's 32 dimensions, a learned rank-32 vote map that no corpus statistic I tried matches well, and sharp attention inside multi-token words, where all the worst failures are.

### Rule

```
CARD (head 3, attn_s0). Every token looks back over the last few tokens with a recency decay. What it looks for depends on its own word group: nouns and names look for the latest clause boundary (and mostly abstain there), linking words look at the word just before them, and word pieces, function words, pronouns and verbs mostly look at themselves. It adds a vote for what typically follows the attended token at that distance in the corpus.

GROUPS (unions of program/categories.py ATTENTION classes)
  BOUNDARY = SENTENCE_ENDS + SENTENCE_STARTS + OPENERS_UNSPACED + QUOTE_OPENERS + INTERJECTIONS
  LINKER   = LINKERS
  VERB     = VERBS_BEFORE_TO + PAST_VERBS_TRANSITIVE + PAST_VERBS_MOTION + SPEECH_VERBS + NEGATED_AUXILIARIES + MODALS
             + BARE_VERBS_TRANSITIVE + BARE_VERBS_ACTIVITY + MENTAL_VERBS + LINKING_VERBS + ING_FORMS
  NOUN     = NAMES + CREATURES + OBJECTS + PLACES + FAMILY_AND_BODY + MASS_AND_PLURAL_NOUNS + THINGS_OUTDOORS + THINGS_MISC
             + ABSTRACT_NOUNS + AFFECTION_NOUNS + OBJECT_PRONOUNS + NAMES_UNSPACED + TIME_WORDS
  PIECE    = all FRAGMENTS_* + SUFFIXES_ING_ED + SUFFIXES_PLURAL + NAME_ENDINGS + LETTERS_LOWER + all ONSETS_* + VERB_ONSETS
  FUNCTION = DETERMINERS + DETERMINERS_RARE + PREPOSITIONS + SUBJECT_PRONOUNS + ADVERBS + PARTICLES
  OTHER    = every other class

WHERE IT LOOKS (query position i, keys j <= i, d = i - j, g = group)
  DIST(d) = [+0.07, -0.03, -0.37, -0.98][d]           for d <= 3
          = -0.98 - 0.129*(d - 3)                     for 3 < d <= 32
          = DIST(32) - 0.024*(d - 32)                 for d > 32
  score(j) = DIST(d) + [d == 0]*SELF[g(t_i)] + [d == 1]*PREV[g(t_i)] + KEY[g(t_j)]
  SELF = PIECE +1.46, FUNCTION +0.85, VERB +0.51, OTHER +0.10, LINKER -0.25, NOUN -0.84, BOUNDARY -1.47
  PREV = LINKER +1.08, PIECE +0.92, OTHER +0.45, VERB +0.30, FUNCTION +0.13, NOUN -0.56, BOUNDARY -1.38
  KEY  = BOUNDARY +0.80, LINKER +0.01, VERB 0.00, FUNCTION -0.48, OTHER -0.48, NOUN -0.50, PIECE -1.03
  a(i, j) = softmax over j <= i of score(j)

WHAT IT VOTES FOR (logit added for each candidate next token v)
  n_k[s, v] = number of positions p in the TRAIN stream with stream[p] = s and stream[p + k] = v   (k = 1..16)
  P_k(v | s) = (n_k[s, v] + 20 * P(v)) / (n_k[s] + 20)            (Dirichlet smoothing toward the unigram)
  PMI_k(s, v) = clip(log P_k(v | s) - log P(v), -5, +5)
  vote_i(v) = (1 / sdf_i) * ( 0.80 * a(i, i) * PMI_1(t_i, v)
                            + 0.78 * sum_{j < i} a(i, j) * PMI_{min(d + 1, 16)}(t_j, v)
                            + 0.137 * log P(v) )
  sdf_i = final-LayerNorm scale at position i. Every test took it from the exact model; a standalone program needs its own estimate.

OPTIONAL REFINEMENTS (measured; see measurements)
  per-group recency slope, plus a 7x7 query-group x key-group table: attention-only KL 0.287 -> 0.269.
    NOUN row: BOUNDARY +1.68, LINKER +0.79, VERB -0.52, NOUN -1.03, PIECE -1.38.
    LINKER row: VERB +0.66, OTHER +0.47, BOUNDARY +0.45, LINKER -0.58, PIECE -0.90, with recency slope -1.30 per 10 tokens.
    BOUNDARY row: BOUNDARY +0.99, with slope +0.30 per 10 tokens (looks far back).
  per-query-group vote strengths (15 scalars): combined KL 0.436 -> 0.4275.
```

### Scalars

- DIST w0..w3 (score for d = 0, 1, 2, 3) = +0.07, -0.03, -0.37, -0.98 (fitted by pattern KL, 50 FIT windows)
- r1 (recency per token, 3 < d <= 32) = 0.129
- r2 (recency per token, d > 32) = 0.024
- SELF (current-token bonus by query group): PIECE +1.46, FUNCTION +0.85, VERB +0.51, OTHER +0.10, LINKER -0.25, NOUN -0.84, BOUNDARY -1.47
- PREV (previous-token bonus by query group): LINKER +1.08, PIECE +0.92, OTHER +0.45, VERB +0.30, FUNCTION +0.13, NOUN -0.56, BOUNDARY -1.38
- KEY (key salience by key group): BOUNDARY +0.80, LINKER +0.01, VERB 0.00, FUNCTION -0.48, OTHER -0.48, NOUN -0.50, PIECE -1.03
- b_self (current-token bigram-PMI vote) = 0.80 with rule attention (0.70 with exact attention); fitted by output KL on 25 FIT windows
- b_earlier (distance-matched PMI vote) = 0.78 with rule attention (0.65 with exact attention)
- b_uni (log unigram bias) = 0.137 (0.142 with exact attention)
- Corpus constants (not fitted): 20 Dirichlet pseudo-counts, PMI clipped to [-5, 5], offsets capped at 16
- Optional per-query-group vote strengths with rule attention. b_self: BOUNDARY .80, LINKER .85, VERB .50, NOUN .52, PIECE 1.20, FUNCTION .63, OTHER .70. b_earlier: BOUNDARY .59, LINKER 1.09, VERB .76, NOUN .40, PIECE 1.00, FUNCTION .50, OTHER .35. b_uni .136.

### Measurements

- Setup: FIT = analysis.fitlib.train_windows(n, seed=11), TEST = train_windows(n, seed=12), TRAIN split only, dev never used. Output KL = mean per-token KL(model || variant) with head 3 replaced inside an exact float64 numpy forward pass, everything else exact. Numpy vs torch max |logprob diff| = 1.3e-4.
- Head-3 basics (100 FIT windows): 19.5% of attention on the current token; mean distance 7.74; entropy 1.90 nats. Attention by distance (queries i>=64): d0 .190, d1 .167, d2 .106, d3 .075, d4 .061, d5 .050, d6 .041, d7 .033, d8 .028. Cumulative mass: d<=4 .60, d<=8 .75, d<=16 .87, d<=32 .94, d<=64 .98.
- Reference ablations: mean-ablating head 3 costs KL 0.525 on 100 FIT windows and 0.529 +- 0.010 on 80 TEST windows. Mean-ablating only the earlier-token part: 0.322 FIT / 0.328 TEST. Zero ablation: 0.548 FIT. Removing head 3's logits entirely: 0.527 TEST.
- Fixed substitute patterns, exact values (100 FIT windows): uniform over j<=i 0.4725 (pattern KL 2.66); uniform over earlier tokens 0.4836; mean distance profile 0.307 (pattern KL 1.00); current token only 1.282; previous token only 1.276. Distance profile on 80 TEST windows: 0.316 +- 0.006.
- Exact score split s = (TT+TP+PT+PP)/(sigma_i sigma_j sqrt32), 60 FIT windows, output KL when one term is dropped: TT (query token x key token) 0.321 (pattern KL 0.72); TP (query token x key position) 0.053; PT (query position x key token) 0.057; PP (positions) 0.167 (pattern KL 3.87). Keeping only PP: 0.333. Only PP+TT: 0.097. Key LN scale set to its mean: 0.015. Query LN scale set to its mean: 0.0001.
- Attention-weighted shares of within-query score variance: TT 0.53, PP 0.64, TP -0.21, PT 0.05. With all keys weighted equally: TT 0.01, TP 0.44, PT 0.01, PP 0.55 (dominated by far keys).
- PP is nearly a function of distance: within-query variance falls from 84.7 to 18.3 once f(d) is removed. f(d)-f(0) is about -0.19 per token near the query (d8 -1.51, d16 -3.09, d64 -9.97). Key position 0 gets about -40 (window-start artefact).
- Mean-query split (60 FIT windows): dropping the query-specific x key-token content term costs 0.332 (pattern KL 0.67), the same as positions only (0.330). Dropping mean-query key salience: 0.105. Dropping query-specific position preference: 0.053. Dropping the positional ramp: 0.221 (pattern KL 5.85). Ramp + salience with no query-specific terms: 0.385.
- Low-rank content term (whitened SVD, 60 TEST windows, other terms exact), by rank: 0 0.314, 1 0.295, 2 0.225, 3 0.178, 4 0.155, 6 0.120, 8 0.096, 12 0.068, 16 0.045, 32 0. Variance shares 0.76, 0.10, 0.02, ... Component 1: noun-like queries -> boundary keys ('.', ',', '!', '?', ' and') vs function-word queries -> past-verb keys. Component 2: word-piece queries -> word-onset keys.
- Attention by class (100 FIT windows). Share of all attention received by key class: SENTENCE_ENDS 17.5%, LINKERS 10.6%, SENTENCE_STARTS 9.8%. Mass on SENTENCE_ENDS keys: from NAMES 0.66, CREATURES 0.53, FAMILY_AND_BODY 0.47, OBJECT_PRONOUNS 0.37. Self-attention: PREPOSITIONS 0.55, FRAGMENTS_SHORT 0.48, LETTERS_LOWER 0.47, SUBJECT_PRONOUNS 0.33, NAMES 0.02, SENTENCE_ENDS 0.01. Previous token: LINKERS 0.31, LINKING_VERBS 0.29. Mean distance: SENTENCE_ENDS 26.2, DETERMINERS 15.2, LINKERS 3.0, PREPOSITIONS 1.7.
- Class-level attention rules fitted by pattern KL (100 TEST windows, exact values): free f(d) 0.308 (pattern KL 1.01, 65 params); + self[64 query classes] 0.270; + prev[query class] 0.252; key[64 key classes] alone 0.303 (pattern KL 0.90); self,prev[query class] + key[key class] 0.232 (pattern KL 0.75, 257 params); + per-class slope 0.2235 (321); full 64x64 class table 0.181 (pattern KL 0.55, 4353 params).
- Hand rule (7 groups, 27 scalars) on 80 TEST windows, exact values: 0.287 +- 0.006 (pattern KL 0.855, top-1 0.725), i.e. 46% of the 0.529 mean-ablation cost; distance profile alone recovers 40%. With per-group recency: 0.282 (pattern KL 0.794). With per-group recency + 7x7 group table + free distance profile (135 params): 0.269 (pattern KL 0.755).
- Corpus-statistic attention features (100 TEST windows). INFO (naive-Bayes information gain of key about next token) has within-row correlation 0.40 with log attention but improves f(d)+self only from 0.309 to 0.297; with 64 class self bonuses 0.270 -> 0.2525; with 7 groups 0.2775 -> 0.2695. PRED (key predicted the query; row corr 0.30): 0.307. Self bonus from H(next|query): 0.3075. Word-onset / word-distance features (80 TEST windows): 0.311-0.315 alone, 0.288-0.290 with groups.
- OV vs corpus: weight-derived per-source votes for the 300 most-attended sources, attention-mass-weighted row correlations. Copy -0.001. PMI_k (k=1..16) between -0.11 and 0.085. After removing the shared mean vote (12.7% of vote variance): left+right context similarity 0.092; coordination ('s and/,/or v') 0.007-0.037; same TARGET/CONTEXT/ATTENTION class 0.026-0.045. Top-20 overlap with any candidate <= 0.057.
- Top weight-derived votes (insight only): ' run' -> ' jump', ' join', ' slide', ' hide', ' play', ' swing'; ' Lily' -> ' Ben', ' Tom', ' Max', ' Sam'; ' red' -> ' yellow', ' blue', ' orange', ' green'; ' looked' -> ' saw', ' noticed', ' realized'; ' count' -> ' two', ' three'. Vote std: '.' 1.10, ',' 1.27, '!' 1.36, ' and' 1.42, against an attention-weighted mean of 2.11, so boundary tokens are the quietest sources.
- Vote rules with exact attention (100 TEST windows): no head-3 votes 0.514; constant mean vote 0.497; log P(v) only 0.502; + self PMI_1 0.440. Adding earlier-token PMI at a fixed offset: k=1 0.437, k=2 0.421, k=3 0.430, k=5 0.430. Distance-matched PMI_{d+1}: 0.387. Coordination PMI: 0.439. Context similarity: 0.4375. All six features: 0.381.
- Vote rules on 80 TEST windows: PMI_{d+1} with 3 scalars 0.3946 +- 0.006 (25% of the 0.527 removal cost); 5 scalars (distance buckets) 0.3945; per-query-group strengths (15 scalars) 0.3889; offsets capped at 12: 0.397; TARGET-class PMI 0.4485; rank-32 SVD of PMI 0.490.
- Both rules together (hand attention + PMI_{d+1} votes), 80 TEST windows: 0.4356 +- 0.007 (top-1 0.660 vs 0.628 under mean ablation), recovering 18% of the 0.529 cost. With per-group vote strengths: 0.4275 +- 0.007 (19%).
- Combined rule by query class (80 TEST windows), mean KL per position, rule vs mean ablation: LINKERS 0.714 vs 0.934 (19% of the rule's total KL); SENTENCE_ENDS 0.375 vs 0.386; LETTERS_LOWER 1.93 vs 2.66; FRAGMENTS_SHORT 1.10 vs 1.58; FRAGMENTS_VC 0.83 vs 1.35; NEGATED_AUXILIARIES 3.18 vs 5.22; DETERMINERS 0.317 vs 0.339; SENTENCE_STARTS 0.245 vs 0.322. Per position, head 3 matters most for word pieces and after linkers.
- Worst positions for the combined rule (head's attention vs rule's): 'Lily excl' -> 'a', KL 9.4 (0.61 on ' exc' vs 0.14). 'his happin' -> 'ess', 9.0 (0.80 on ' happ' vs 0.24). Opening quote + 'O' + 'f' -> ' course', 8.5 and 8.2 (0.85 on 'O' vs 0.16). 'butterfl' -> 'ies', 8.4 (0.79 on ' butterf' vs 0.13). ' wasn' -> apostrophe-t, 8.3 (0.93 on itself vs 0.25). 'neckl' -> 'ace', four cases at 7.5-8.3 (0.89-0.94 on 'l' itself vs 0.23-0.47). 'gott' -> 'en', 7.5 (0.94 on 't' itself vs 0.22).

### Failures and dead ends

- The readable rule explains little: both rules together recover about 18% of head 3's mean-ablation cost (0.436 vs 0.529); the attention rule alone recovers 46% and the vote rule alone 25%.
- Corpus-derived attention features add at most 0.012 output KL over a distance profile. These were information gain about the next token, 'key predicted the query', a self bonus from next-token entropy, and word-onset / word-distance features.
- Class tables hit a ceiling: even the full 64x64 query-class x key-class table (4353 params) leaves attention-only KL at 0.181. The query-key content matching is token-specific and spread over 8 to 16 dimensions (rank 8 still 0.096).
- No corpus statistic matches the OV votes well. Coordination counts, context similarity, TARGET-class PMI and rank-32 SVD PMI all do worse than plain token PMI, and the best vote rule still leaves 0.39 nats even with the exact attention pattern.
- Mid-word completion is the main failure: the model head puts 0.6-0.94 on a single token (the word onset or the current piece) and completes the word; the rule spreads 0.13-0.47 there and loses 7.5-9.4 nats per position.
- After sentence ends the rule recovers almost nothing (0.375 vs 0.386 under mean ablation).
- Every vote substitution divides by the exact model's final-LayerNorm scale, so the vote rule is not yet standalone.

## Knowledge baseline for attn_s0: plain training-corpus statistics with 23 model-fitted scalars reach dev_fast KL 0.615, closing 82% of the gap from the unigram (3.47) to 0

Corpus counts plus 23 named scalars fitted to the model reach dev_fast KL 0.615 (top-1 agreement 0.609) and close 82.3% of the gap from the unigram (3.47) to 0. The best context-free rule is the corpus bigram mixed with a constant 2-4% unigram plus a class blur; it reaches 1.196, against the model's own context-free optimum of 1.097, and temperature plays no role (tau ~ 0.99). Short-range n-grams and skip-grams are complementary (each about 0.79 alone, 0.683 together). Long-range rules (co-occurrence bands plus a copy/name bonus) add only 0.068 nats, far below head 1's 0.38-nat earlier-token ablation cost. 142 per-class gates and temperatures add only 0.006. The remaining 0.61 nats is spread out, largest after ' the'/' little' (which noun comes next), sentence ends, linkers and word fragments, plus a model-specific echo of the current token at window position 0.

### Rule

```
setup(corpus):                                   # all counts from data/tokens/train.bin, no model
  P_uni(v) = (count(v)+1)/(N+V)
  C_d[a,v] = #pairs (t_j=a, t_{j+d}=v) with no EOS strictly between, d=1..8
  Pdir(v|t) = (C_1[t,v] + 44*P_uni(v)) / (C_1[t] + 44)
  P1(v|t)  = 0.940*Pdir(v|t) + 0.036*mean_{t' in CONTEXT_class(t)} Pdir(v|t') + 0.024*P_uni(v)      # alpha 44.0, beta 0.0363, lambda 0.0238
  PMI_d(a,v) = log[(1-0.036)*(C_d[a,v]+47*P_uni)/(C_d[a]+47) + 0.036*P_uni] - log P_uni(v), d=2..8 (JM alpha 46.9, lambda 0.0359)
  PMI_b(a,v): same, counts pooled within story over bands b = [9,16],[17,32],[33,64],[65,128],[129,255] (8 sampled distances each)
  c3[(a,b),v], c4[(a,b,c),v] = trigram / 4-gram continuation counts
predict(tokens), row i (only tokens inside the window; skip-gram/band tokens must be in the current story):
  P3 = (c3[t_{i-1},t_i] + 300*P1(.|t_i)) / (n3 + 300);  P4 = (c4[t_{i-2..i}] + 300*P3) / (n4 + 300)   # back off if the context crosses an EOS
  s(v) = 0.039*log P_uni(v) + 0.777*log P1(v|t_i)
       + sum_{d=2..8} w_d * PMI_d(t_{i+1-d}, v)                      # w = .178 .148 .156 .165 .146 .132 .142
       + 0.308*(log P3(v) - log P1(v|t_i)) + 0.195*(log P4(v) - log P3(v))
       + sum_b u_b * mean_{j in band b} PMI_b(t_j, v)                # u = .836 .994 .616 -.236 .150
       + 0.591*[v occurred earlier in this story] + 0.655*[same, and v is a NAME (TARGET NAMES/NAMES_UNSPACED)]
  return log_softmax(s)
(The optional L5c adds gate[c]*context_part and temp[c]*s, where c = CONTEXT class of t_i (2 x 71 scalars): dev 0.6146 -> 0.6082.)
```

### Scalars

- L1 bigram rule: Dirichlet alpha = 44.0, unigram mix lambda = 0.0238, CONTEXT-class blur beta = 0.0363 (fitted via the exact identity); log-linear weights uni 0.039 / big 0.777 inside L4 (alone: -0.002 / 0.986)
- Skip-gram tables: JM alpha = 46.9, lambda = 0.0359 (shared by all distances and bands)
- L4 skip-gram weights d=2..8: 0.178 0.148 0.156 0.165 0.146 0.132 0.142 (without n-grams, L2c: 0.407 0.258 0.214 0.226 0.220 0.242 0.302)
- L4 n-gram: mu3 = 300, weight 0.308; mu4 = 300, weight 0.195 (grid {3, 30, 300}; extended-grid optimum mu3 about 1000-3000, mu4 about 1000)
- L4 band weights [9,16] 0.836, [17,32] 0.994, [33,64] 0.616, [65,128] -0.236, [129,255] 0.150
- L4 copy bonus 0.591 nats for any token already in the story; extra 0.655 for names (total 1.246, i.e. x3.5 vs x1.8)
- Context-free per-class JM lambda range: 0.004 (SENTENCE_ENDS) to 0.69 (WORD_ENDINGS_RARE)

### Measurements

- Corpus: 527,489,811 train tokens, 2,099,719 stories (EOS), 106 never-seen tokens; my bigram counts equal data/baselines/bigram_counts.npy exactly
- Context-free rules, exact identity KL = 1.0974 + sum_t w_t KL(big_t||q_t) on the 20k-window mean_dists table (in-sample): unigram add-one 3.4713; Dirichlet alpha=1 (harness baseline) 1.3363; Witten-Bell (0 scalars) 1.2471; add-k (k=4.0) 1.2130; Dirichlet with best alpha=11,530 1.2067; JM (alpha 46.9, lambda 0.036) 1.2021; JM + CONTEXT-class blur (3 scalars) 1.1938; + temperature 1.1938 (tau 0.9986); Dirichlet^tau*unigram^gamma 1.2063 (tau 0.975); blur alone (CONTEXT, beta 0.037) 1.1973, (ATTENTION, beta 0.027) 1.1984; 71 per-CONTEXT-class JM lambdas 1.1953; model optimum 1.0974
- Per-class JM lambda (model smoothing toward unigram): 0.004 after SENTENCE_ENDS, 0.017 after THE_AND_LITTLE / ARTICLE_A, 0.12-0.14 after word fragments / WORD_ENDINGS, 0.69 after WORD_ENDINGS_RARE
- Low-rank SVD of centred log Dirichlet bigram (unweighted), KL with temperature: rank 32 2.352, 64 2.093, 128 1.818, 256 1.547, 512 1.322 (all worse than full rank 1.207)
- Context-free gap to model optimum (JM+blur): 0.0964 nats; largest single token '\n' contributes 0.0015; by current-token corpus count: [1e4,1e5) 9.6% of positions / 21.1% of gap (KL/occ 0.211); [1e5,1e6) 37.5% / 55.7% (0.143); [1e6,1e7) 29.0% / 16.3% (0.054); >=1e7 23.8% / 6.5% (0.026); highest KL/occ classes ADJECTIVES_AND_QUANTIFIERS 0.224, WORD_ENDINGS 0.229, FRAGMENTS_2 0.230
- Slope of log model-mean vs log corpus P(v|t) over pairs with C>=50, by count of t: [1e3,1e4) 1.384, [1e4,1e5) 1.136, [1e5,1e6) 1.052, [1e6,1e7) 1.007, >=1e7 0.945
- Model mean vs corpus (frequency weighted): entropy 3.623 vs 3.516 nats; mass on corpus top continuation 0.238 vs 0.241; model mass on continuations never seen after t in 527M tokens 0.0147; on continuations seen <10 times 0.0101 (corpus 0.0024)
- Ladder, KL(model||rule), held-out TRAIN (160 windows, seed 2025) / dev_fast (200 windows) / % of unigram gap closed on dev / top-1 dev: L0 unigram 3.4705 / 3.4719 / 0% / 0.107; L1 bigram rule (5 scalars) 1.1925 / 1.1959 / 65.6% / 0.372; L2a +trigram (7) 0.8325 / 0.8375 / 75.9% / 0.530; L2b +4-gram (9) 0.7898 / 0.7937 / 77.1% / 0.556; L2c bigram+skip-grams d=2..8 (12) 0.7768 / 0.7863 / 77.4% / 0.534; L2d skip+tri+4-gram (16) 0.6745 / 0.6826 / 80.3% / 0.588; L3a L2c+copy+name (14) 0.7439 / 0.7507 / 78.4% / 0.544; L3b L2c+bands (17) 0.7274 / 0.7388 / 78.7% / 0.552; L3c L2c+bands+copy+name (19) 0.7048 / 0.7126 / 79.5% / 0.558; L4 all (23) 0.6073+-0.0039 / 0.6146+-0.0040 / 82.3% / 0.609; L5a +71 gates 0.6044 / 0.6115; L5b +71 temperatures 0.6037 / 0.6107; L5c +142 0.6019 / 0.6082 / 82.5% / 0.611
- Skip-gram ladder (first run, held-out TRAIN): +pmi2 0.9163, +pmi2..3 0.8556, +pmi2..4 0.8273, +pmi2..8 0.7768; whole-stream counts without story mask 0.7776; +copy 0.7462; +copy+name 0.7439; recency copy (tau 3/12/48)+name instead of binary copy 0.7598; binary copy+name+recency 0.7425; bigram+copy+name without skip-grams 1.1439
- Copy across stories (whole window) vs within story: 0.7453 vs 0.7439 (TRAIN), 0.7516 vs 0.7507 (dev)
- Long-range rules (bands+copy+name) on top of skip-grams and n-grams: dev 0.6826 -> 0.6146 (0.068 nats); copy+name alone on L2c 0.036; bands alone on L2c 0.048
- Positions where the model's top token equals the corpus top continuation of the last 2 (3) tokens (context count >=20, share >=0.5): 22.2% (25.9%) of dev positions; KL there: unigram 4.361 (4.200), bigram 1.145 (1.227), trigram rule 0.432 (0.558), 4-gram rule 0.416 (0.479), L4 0.354 (0.396); L4 on the remaining positions 0.701
- N-gram backoff strength on FIT (32 windows): trigram rule KL mu3=3 0.9115, 30 0.8792, 300 0.8519, 1000 0.8437, 3000 0.8431, 10000 0.8532; 4-gram rule mu4=3 0.8226, 30 0.8165, 300 0.8109 (mu3=300); mu3=1000 & mu4=1000 0.7997, mu4=3000 0.8009
- L4 residual by CONTEXT class of current token (TRAIN): THE_AND_LITTLE 7.0% of KL at 4.2% freq (KL/occ 1.014, bigram rule 1.795); SENTENCE_ENDS 6.0% at 7.0% (0.521); AND_AND_ADVERBS 4.8% at 3.8% (0.765); COMMAS 3.7% (0.697); POSSESSIVES 3.3% (0.771); FRAGMENTS_1 KL/occ 0.879; WORD_ENDINGS 0.978
- L4 residual by model confidence p_top: [0,0.2) 22.8% of positions, KL 0.725; [0.2,0.5) 42.0%, 0.646; [0.5,0.8) 20.4%, 0.585; [0.8,1] 14.8%, 0.347. Model top token already in window: 59.3% of positions, KL 0.581 vs 0.646 otherwise; model top token is a name: 3.1% of positions, KL 0.726
- L4 by tokens since story start: current token EOS (0.4% of positions) KL 0.341 vs bigram rule 0.085 (global PoE weights under-weight the base when context features are absent); 1-4 tokens 0.277; 5-16 0.552; 17-64 0.593; 65-255 0.617
- Echo at window position 0: model top-1 = current token 50.6% (positions >=1: 1.6%; corpus P(next = current) 1.1%); mean model probability of repeating it 0.145; L4 KL at position 0 1.467 vs 0.604 elsewhere. Model top-1 = suffix piece of the current token: 0.02% of positions (0.0% of KL)
- Model attention mass by distance delta=i-j (12 TRAIN windows): head 0 self .492, 1 .158, 2 .088, 3 .058, 4 .041, 5-7 .071, 8-15 .060, 16-31 .025; head 1 self .042, 8-15 .192, 16-31 .240, 32-63 .205, 64-127 .087; head 2 self .342, 1 .209, 2 .123, 3 .079, 5-7 .099, 8-15 .068; head 3 self .194, 1 .162, 2 .111, 5-7 .129, 8-15 .140, 16-31 .071, 32-63 .038
- Attention on tokens from an earlier story (1,415 positions whose story began inside the window): head 0 0.011, head 1 0.147, head 2 0.017, head 3 0.037
- Compute: about 55 min wall time at no more than 2 threads (counts 3 min, bands 9 min, trigram/4-gram 3 min, ctx2 fit+eval 16 min)

### Failures and dead ends

- Absolute-discounting rules (unigram and Kneser-Ney continuation base) returned NaN: my script divides by zero for the 106 never-seen tokens (C_t = 0). I did not re-run it, so no number exists for that rule.
- Unweighted SVD low-rank smoothing of the log bigram is a dead end: rank 128 gives KL 1.818 vs 1.207 at full rank. Unweighted SVD is dominated by impossible pairs. A KL-weighted (softmax) factorisation was not tested.
- The hierarchical Dirichlet backoff (token -> class -> unigram) scored 1.242, worse than a plain Dirichlet at 1.207. Its alpha_class search hit the upper edge of its range (545.9 in both class systems), so 1.242 is only an upper bound for that rule.
- Temperature is a no-op. Global tau = 0.99-1.0 changes KL by less than 0.0004 context-free. 71 per-class temperatures on L4 give dev 0.6146 -> 0.6107.
- Per-class gates plus temperatures (142 scalars) are worth only 0.006 on dev (0.6146 -> 0.6082).
- Story masking of skip-grams has no measurable effect (0.7768 vs 0.7776). Copying within the story vs the whole window is also indistinguishable (0.7439 vs 0.7453), even though head 1 puts 14.7% of its attention across story boundaries.
- Recency-weighted copy does little. As an addition to binary copy it gains 0.0014; as a replacement it is worse (0.7598 vs 0.7439).
- The individual band weights are not interpretable: band 65-128 gets weight -0.236. The bands are collinear; a single pooled long-range band would be more readable.
- The main run's n-gram backoff strengths (mu3 = mu4 = 300) sat at the grid edge. The extended grid's optimum is about mu3 1000-3000 and mu4 1000, worth about 0.01 on the n-gram-only rules. This was not propagated into L4's dev number.
- The fragment-suffix echo hypothesis from the worst-position list fails: it covers 0.02% of positions and 0.0% of KL.
- Run mishaps: scipy is not installed, so I wrote golden-section and coordinate-descent optimisers (multi-scalar fits may be slightly off-optimum). The first ctx.py run crashed (backward under no_grad); after the fix the rerun completed.
