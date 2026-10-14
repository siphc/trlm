"""Measurements shared by the claim tests (held-out claim windows) and analysis/mech_eval.py (TRAIN windows).

Comparing a program part with a model head. The model side is the harness's earlier-token ablation: the head's
output from earlier tokens is replaced by its mean, and the model's log-probs move by d_model = lp - lp_ablated.
The program side does the same to the part: its votes from earlier tokens are replaced by their mean over
positions, and the program's log-probs move by d_program. Two numbers summarise the comparison:

  agreement   per position, the correlation between d_model and d_program over the vocabulary, weighted by the
              model's probabilities (so it is about the tokens the model considers); median over positions
  shift       mean KL between the distributions before and after removal (how far the part moves the output)
"""
import numpy as np

V = 2000


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def kl(p_log, q_log):
    p = np.exp(p_log)
    return np.where(p > 0, p * (p_log - q_log), 0.0).sum(-1)


def weighted_corr(a, b, w):
    """Per-row correlation of a and b over the last axis, weighted by w (rows of w sum to 1)."""
    a = a - (w * a).sum(-1, keepdims=True)
    b = b - (w * b).sum(-1, keepdims=True)
    cov = (w * a * b).sum(-1)
    return cov / np.sqrt((w * a * a).sum(-1) * (w * b * b).sum(-1) + 1e-30)


def earlier_votes(P, name, tokens, attention=None):
    """The votes of program part `name` (head0, head1, head2) that come from tokens before the current one.
    `attention` replaces the part's attention rule (an (n, n) matrix) if given."""
    from program import head0, head1, head2

    s = P.STATS
    mod = {"head0": head0, "head1": head1, "head2": head2}[name]
    a = mod.attention(tokens, s) if attention is None else attention
    if name == "head1":
        a = a * (1.0 - np.eye(len(tokens)))
        return a @ mod.recalled_votes(tokens, s)
    return mod.earlier_votes(a, tokens, s)


def uniform_attention(n):
    return np.tril(np.ones((n, n))) / np.arange(1, n + 1)[:, None]


def compare_with_head(P, windows, lp_model, lp_ablated, part_votes):
    """part_votes(tokens) -> (n, V) votes of the program part from earlier tokens. Returns agreement and shifts."""
    toks = [np.asarray(w[:256], dtype=np.int64) for w in windows]
    votes = [part_votes(t) for t in toks]
    mean_votes = np.mean([v.mean(0) for v in votes], axis=0)
    agree, shift_model, shift_program = [], [], []
    for t, v, p, pa in zip(toks, votes, lp_model, lp_ablated):
        q = P.predict(t)
        qa = log_softmax(q - v + mean_votes)
        agree.append(weighted_corr(p - pa, q - qa, np.exp(p)))
        shift_model.append(kl(p, pa))
        shift_program.append(kl(q, qa))
    return {"agreement": float(np.median(np.concatenate(agree))),
            "shift_model": float(np.mean(shift_model)), "shift_program": float(np.mean(shift_program))}


def vote_agreement(windows, lp_model, lp_ablated, votes_of):
    """Median weighted correlation between the model's change when heads are ablated and some program votes
    (centred by their mean over positions), without removing anything from the program."""
    toks = [np.asarray(w[:256], dtype=np.int64) for w in windows]
    votes = [votes_of(t) for t in toks]
    mean_votes = np.mean([v.mean(0) for v in votes], axis=0)
    agree = [weighted_corr(p - pa, v - mean_votes, np.exp(p)) for v, p, pa in zip(votes, lp_model, lp_ablated)]
    return float(np.median(np.concatenate(agree)))


def removal_rise(P, windows, lp_model, names):
    """Mean rise of KL(model || program) when each listed mechanism is switched off."""
    toks = [np.asarray(w[:256], dtype=np.int64) for w in windows]
    base = np.mean([kl(p, P.predict(t)).mean() for t, p in zip(toks, lp_model)])
    out = {"all": float(base)}
    for m in names:
        out[m] = float(np.mean([kl(p, P.predict(t, off=(m,))).mean() for t, p in zip(toks, lp_model)]) - base)
    return out
