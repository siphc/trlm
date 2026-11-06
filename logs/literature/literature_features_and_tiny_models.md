# Literature check (2026-11-06): feature decompositions of small transformers, and tiny-model challenges

Prompted by the human after the proposal of new directions: (1) explain features rather than tokens (sparse
dictionaries on a one-layer model); (3) shrink the model until it is fully understood. Web search only; numbers are as
quoted by the sources; full papers not read end to end.

## Direction 1: features in small transformers (done before, in close variants)

- **Bricken et al. 2023, "Towards Monosemanticity"** (Anthropic): sparse autoencoder on the MLP of a one-layer
  transformer. The precedent for one-layer feature work; MLP, not attention.
- **Kissane, Krzyzanowski, Bloom, Conmy, Nanda 2024, "Interpreting Attention Layer Outputs with Sparse Autoencoders"**
  (ICML workshop; https://arxiv.org/abs/2406.17759): SAEs on the concatenated head outputs (z) of attention layers,
  GPT-2 Small and others up to 2B. Features: long-range context, short-range context, induction. At least 90% of
  GPT-2 Small's heads polysemantic. With the attention pattern frozen, an attention layer is linear in earlier
  residual streams, so features trace back to source tokens (weight-based head attribution). Earlier AF post on a
  two-layer model: https://www.alignmentforum.org/posts/DtdzGwFh9dCfsekZZ
- **Turner, Wu, Batson, Aug 2026, "Characterizing interference weights in a tiny language model"** (Anthropic;
  https://transformer-circuits.pub/2026/interference_effectiveness_helpfulness/index.html). Closest to us.
  - Model: one-layer transformer, attention plus MLP (MLP replaced by a 4,096-feature transcoder), no normalisation,
    d_model 256, vocabulary 4,096, context 1,024, about 2.9M parameters.
  - The model is rewritten as virtual weights between tokens, positions, features and logits (six families; QK the
    one bilinear family). This reproduces the forward pass up to the transcoder error, but materialising the virtual
    weights grows 2.9M parameters to about 331M (100x).
  - Pruning the least effective 70% costs 0.01 nats, 85% costs 0.1. Helpful weights are "tens of percent" of the
    virtual weights, and the helpful set still exceeds the original parameter count.
  - QK and token-to-feature weights are less effective and less helpful than weights into logits.
  - **Relation to our results:** the same wall from another side. Our distillation curve needed about 1.78M fitted
    entries to reach about the seed floor (logs/main_track/compression_experiments.md); their helpful virtual weights exceed the
    model's own parameter count.
- **Bushnaq, Braun, Clive-Griffin, Bussmann, Hu, Ivanitskiy, Linsefors, Sharkey, Oct 2026, "Interpreting Language Model
  Parameters"** (adVersarial Parameter Decomposition, VPD; https://arxiv.org/abs/2610.20748).
  - 4-layer 67M model on a Pile subset; 38,912 rank-1 subcomponents, about 9,972 alive, about 205 active per position.
  - Decomposes W_Q, W_K, W_V, W_O.
  - Fidelity: unmasked 82.4% of pretraining compute recovered; stochastic masks 26.9%; causal-importance masks 9.4%.
    Not robust to adversarial ablation at 160 PGD steps or more.
- **Seed dependence of SAE features.**
  - Paulo and Belrose 2025 (https://arxiv.org/abs/2501.16615): SAEs differing only in seed share about 30% of features
    (Llama 3 8B, 131k latents).
  - arXiv 2410.06981: two SAEs on TinyStories-1L-21M, mean cross-seed similarity 0.376.
  - arXiv 2606.12138: stable features carry most of the signal; subspaces are more reproducible than single latents.
  - None of these compares features across independently trained *models* with a measured seed floor, as our five
    seeds would allow.

**Reading for us.** The method of direction 1 is established. What is not covered, as far as this search shows:
- an attention-only model, where every virtual weight is a QK or OV token table;
- an exact KL harness;
- universality across five training seeds, tied to a measured seed floor.

The Anthropic result predicts that a feature-level account of our model will not be smaller than the model.

## Direction 3: tiny models that resist full understanding

- **AlgZoo** (Jacob Hilton for ARC, 26 Jan 2026; work by Zihao Chen, George Robinson, David Matolcsi, Jacob
  Stavrianos, Jiawei Li, Michael Sklar; https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/,
  code https://github.com/alignment-research-center/alg-zoo).
  - RNNs and transformers with 8 to 1,408 parameters, trained on algorithmic tasks (four families).
  - Understanding means a mechanistic (deductive, not sampled) estimate of the model's accuracy, judged by mean squared
    error versus compute against random sampling, and by "surprise accounting" (an explanation's surprise in bits,
    compared with the bits used to optimise the model).
  - The largest model they consider more or less understood: M_{4,3}, 32 parameters (98.5% accuracy).
  - **M_{16,10}:** 432 parameters, a one-layer ReLU RNN, hidden size 16, sequence length 10, on the second-argmax task
    (95.3% accuracy). Not understood. Partly isolated subcircuits compute leave-one-out maxima. A handcrafted
    22-neuron model with those features reaches 99%, but it does not map onto the trained neurons.
  - Challenge: a mechanistic accuracy estimate for M_{16,10} that matches random sampling in MSE versus compute. No
    prize mentioned.
  - Thesis: fully understanding few-hundred-parameter models must become easy before multi-billion-parameter models
    can be understood.
  - Probably the "~400-parameter model" the human remembered; the post's tweet was not found.
- **Weight-sparse transformers** (Gao et al., OpenAI, Nov 2025; https://arxiv.org/abs/2511.13653): training with most
  weights exactly zero gives circuits about 16x smaller at matched loss. Interpretable by construction, at a
  capability cost.
- **A 777-parameter transformer for 10-digit addition** (blog, Feb 2026: https://themenonlab.blog/blog/tiny-transformers-challenge).

**Reading for us.**
- Calibration: our model (about 0.35M parameters) is about 800 times the size of the smallest model that ARC's
  strictest standard has not cracked.
- No language-model counterpart of AlgZoo turned up: a ladder of tiny language models, from hundreds of parameters
  upward, with a strict completeness criterion. That gap fits direction 3.
