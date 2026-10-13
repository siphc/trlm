"""Measurements behind the CLAIM docstrings in program/*.py.

Shared by the claim tests (program/tests/test_*.py) and analysis/claims_measure.py, so that a claim, its
test and the printed number use one definition. Uses only harness.claims: the program through
claims.program(), the model (the reference) through claims.model_logprobs(). Reads no files.

Definitions. KL and top-1 numbers average over all 256 positions of every window, as the harness does.
Attention numbers average over positions 1..255 (position 0 has no earlier token).

  KL                 per-position KL(model || program), in nats
  top-1              share of positions where the program's most likely next token is the model's
  abstain            attention a head puts on the current token; it casts no vote with it (attention.py)
  previous / near / far
                     attention on the previous token / on 2 to 8 tokens back / on more than 8 back
  mean distance      sum_{j<i} a_ij (i - j) / sum_{j<i} a_ij, pooled over positions: how far back the
                     attention that does vote comes from
  beyond D           attention on tokens more than D back, at positions with more than D earlier tokens
  cost               KL(model || program with the component's votes replaced by their average) minus
                     KL(model || program). Averages are over all positions of the 200 claim windows
                     ("mean-ablation"; it removes what the component adds beyond a fixed vote).
  model cost         KL(model || model with head h's earlier-token part mean-ablated), from
                     claims.model_logprobs(..., ablation="earlier"). This is the matching model-side
                     number: program heads are silent on the current token, so only the model head's
                     earlier-token part is comparable.
  gain               log q(model's top token) minus the same with the head mean-ablated (model gain: the
                     same on the model's own log-probabilities with ablation="earlier")
  recall positions   positions i >= 1 where the model's top token occurs among the earlier tokens
                     t_0 .. t_{i-1} and FOLLOW alone gives it less than 5%
  inside a word      the current token starts with a letter and the previous token ends with one
                     (' pro' then 'ble', as in 'problem')
  after speech verb  the previous token is in SPEECH_VERBS (' said', ' says', ' replied', ...)
  'said,' positions  after a speech verb and the current token is ','
  after a linker     the current token is in LINKERS (' and', ',', ' to', ' of', ...)
  self vote          median, over the tokens s seen in the windows, of RECALL(s, s) minus the median of
                     RECALL(s, v) over the other tokens v of s's class: how much more a token the head
                     attends to votes for itself than for similar words (a property of the tables)
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import numpy as np

from harness import claims

N_HEADS = 4
CTX = 256
CHUNK = 10                 # windows per model call (keeps memory near 1 GB)
HORIZONS = (32, 40, 48, 64, 96, 128)
RECALL_FOLLOW_MAX = 0.05   # "FOLLOW alone gives it less than 5%"


def log_softmax(score: np.ndarray) -> np.ndarray:
    score = score - score.max(-1, keepdims=True)
    return score - np.log(np.exp(score).sum(-1, keepdims=True))


@lru_cache(maxsize=1)
def token_facts() -> dict:
    """Token ids and per-token flags for the position splits (from the program's vocab and classes)."""
    P = claims.program()
    vocab, lib = P.vocab, P.tables_lib
    tokens = vocab.TOKENS
    cls = lib.TOKEN_CLASS
    idx = lib.CLASS_INDEX
    return {
        "starts_letter": np.array([len(s) > 0 and s[0].isalpha() for s in tokens]),
        "ends_letter": np.array([len(s) > 0 and s[-1].isalpha() for s in tokens]),
        "speech_verb": cls == idx["SPEECH_VERBS"],
        "linker": cls == idx["LINKERS"],
        "name": cls == idx["NAMES"],
        "upon": vocab.ID[" upon"], "a": vocab.ID[" a"], "comma": vocab.ID[","], "quote": vocab.ID[' "'],
    }


@lru_cache(maxsize=1)
def follow_exceptions() -> np.ndarray:
    """FOLLOW's token-pair exceptions (the AFTER table) as a (2000, 2000) matrix."""
    P = claims.program()
    return P.tables_lib.pairs_matrix(P.follow.T.AFTER)


@lru_cache(maxsize=1)
def average_votes() -> tuple[np.ndarray, np.ndarray]:
    """Average vote vector of each head, and of FOLLOW's exceptions, over all positions of the 200 claim
    windows. The same windows define the model-side means in harness.claims."""
    P = claims.program()
    heads = np.zeros((N_HEADS, P.vocab.VOCAB_SIZE))
    counts = np.zeros(P.vocab.VOCAB_SIZE)
    for w in claims.claim_windows():
        t = w[:CTX].astype(int)
        for h, module in enumerate(P.HEADS):
            heads[h] += module.votes(t).sum(0)
        counts += np.bincount(t, minlength=P.vocab.VOCAB_SIZE)
    n = counts.sum()
    return heads / n, counts @ follow_exceptions() / n


def _earlier_seen(top: np.ndarray, t: np.ndarray) -> np.ndarray:
    """seen[i] = top[i] occurs in t[0 .. i-1]."""
    seen, out = set(), np.zeros(len(t), dtype=bool)
    for i in range(len(t)):
        out[i] = top[i] in seen
        seen.add(int(t[i]))
    return out


def self_vote_strength(windows: np.ndarray) -> list[float]:
    """Per head: the 'self vote' defined above, over the tokens that occur in `windows`."""
    P = claims.program()
    cls = P.tables_lib.TOKEN_CLASS
    seen = np.unique(np.asarray(windows)[:, :CTX])
    out = []
    for module in P.HEADS:
        recall = module.HEAD.recall
        diffs = []
        for s in seen:
            mates = np.flatnonzero(cls == cls[s])
            mates = mates[mates != s]
            if len(mates):
                diffs.append(recall[s, s] - np.median(recall[s, mates]))
        out.append(float(np.median(diffs)))
    return out


def measure(windows: np.ndarray | None = None, chunk: int = CHUNK, hook=None) -> dict:
    """All claim numbers on `windows` (default: the 200 claim windows). Returns a flat dict of floats.
    hook(arrays), if given, is called once per window with its per-position arrays (analysis only)."""
    P = claims.program()
    W = claims.claim_windows() if windows is None else windows
    avg_heads, avg_exceptions = average_votes()
    exceptions = follow_exceptions()
    f = token_facts()
    S = defaultdict(float)
    r = np.arange(CTX)
    dist = r[:, None] - r[None, :]
    earlier = dist > 0
    rows1 = r[1:]
    for c in range(0, len(W), chunk):
        block = W[c:c + chunk]
        lp_model = claims.model_logprobs(block)
        lp_model_abl = [claims.model_logprobs(block, ablate_heads=(h,), ablation="earlier")
                        for h in range(N_HEADS)]
        for b in range(len(block)):
            t = block[b, :CTX].astype(int)
            p = lp_model[b]
            top = p.argmax(1)
            q = P.predict(t)                                   # the program itself
            follow = P.follow.votes(t)
            votes = [m.votes(t) for m in P.HEADS]
            weights = [m.HEAD.weights(t) for m in P.HEADS]
            score = follow + sum(votes)
            err = float(np.abs(log_softmax(score) - q).max())
            S["decomposition_error"] = max(S["decomposition_error"], err)
            kl = claims.kl(p, q)
            q_follow = log_softmax(follow)
            S["positions"] += CTX
            S["kl"] += kl.sum()
            S["top1"] += (q.argmax(1) == top).sum()
            S["kl_follow"] += claims.kl(p, q_follow).sum()
            S["top1_follow"] += (q_follow.argmax(1) == top).sum()
            err = float(np.abs(q[0] - q_follow[0]).max())
            S["first_position_error"] = max(S["first_position_error"], err)
            S["kl_no_exceptions"] += claims.kl(p, log_softmax(score - exceptions[t] + avg_exceptions)).sum()
            self_votes = sum(weights[h] @ m.HEAD.recall[t] for h, m in enumerate(P.HEADS))
            S["kl_self_votes"] += claims.kl(p, log_softmax(follow + self_votes)).sum()

            # FOLLOW alone at ' upon'
            m = t == f["upon"]
            S["upon_n"] += m.sum()
            S["upon_follow_matches_model"] += (q_follow[m].argmax(1) == top[m]).sum()
            S["upon_model_top_is_a"] += (top[m] == f["a"]).sum()
            S["upon_kl_follow"] += claims.kl(p[m], q_follow[m]).sum()
            S["upon_follow_p_a"] += np.exp(q_follow[m, f["a"]]).sum()

            # position splits
            recall = _earlier_seen(top, t) & (np.exp(q_follow[r, top]) < RECALL_FOLLOW_MAX)
            recall[0] = False
            other = ~recall
            other[0] = False
            inside = np.r_[False, f["starts_letter"][t[1:]] & f["ends_letter"][t[:-1]]]
            after_sv = np.r_[False, f["speech_verb"][t[:-1]]]
            said_comma = after_sv & (t == f["comma"])
            linker = f["linker"][t]
            splits = {"inside_word": inside, "after_speech_verb": after_sv, "after_linker": linker}
            S["recall_n"] += recall.sum()
            S["other_n"] += other.sum()
            S["recall_top1"] += (q.argmax(1) == top)[recall].sum()
            S["recall_top1_follow"] += (q_follow.argmax(1) == top)[recall].sum()
            S["said_comma_n"] += said_comma.sum()
            S["said_comma_model_quote"] += (top[said_comma] == f["quote"]).sum()
            S["said_comma_p_quote"] += np.exp(q[said_comma, f["quote"]]).sum()
            for name, mask in splits.items():
                S[f"{name}_n"] += mask.sum()
            S["name_tokens"] += f["name"][t].sum()

            costs, model_costs = [], []
            for h, module in enumerate(P.HEADS):
                key = f"head{h}."
                q_h = log_softmax(score - votes[h] + avg_heads[h])
                cost = claims.kl(p, q_h) - kl
                model_cost = claims.kl(p, lp_model_abl[h][b])
                costs.append(cost)
                model_costs.append(model_cost)
                S[key + "cost"] += cost.sum()
                S[key + "model_cost"] += model_cost.sum()
                S[key + "top1_without"] += (q_h.argmax(1) == top).sum()
                for name, mask in splits.items():
                    S[key + "cost_" + name] += cost[mask].sum()
                    S[key + "cost_not_" + name] += cost[~mask].sum()
                    S[key + "model_cost_" + name] += model_cost[mask].sum()
                    S[key + "model_cost_not_" + name] += model_cost[~mask].sum()
                gain = q[r, top] - q_h[r, top]
                model_gain = p[r, top] - lp_model_abl[h][b][r, top]
                S[key + "gain_recall"] += gain[recall].sum()
                S[key + "gain_other"] += gain[other].sum()
                S[key + "model_gain_recall"] += model_gain[recall].sum()
                S[key + "model_gain_other"] += model_gain[other].sum()
                S[key + "recall_top1_without"] += (q_h.argmax(1) == top)[recall].sum()
                S[key + "said_comma_p_quote_without"] += np.exp(q_h[said_comma, f["quote"]]).sum()

                a = weights[h]
                a1, d1 = a[1:], dist[1:]
                S[key + "abstain"] += a[rows1, rows1].sum()
                S[key + "previous"] += a[rows1, rows1 - 1].sum()
                S[key + "near"] += a1[(d1 >= 2) & (d1 <= 8)].sum()
                S[key + "far"] += a1[d1 > 8].sum()
                S[key + "distance_sum"] += (a1 * d1 * (d1 > 0)).sum()
                S[key + "earlier_mass"] += (a1 * (d1 > 0)).sum()
                for D in HORIZONS:
                    S[key + f"beyond_{D}"] += a[D + 1:][dist[D + 1:] > D].sum()
                S[key + "name_mass"] += (a * earlier)[:, f["name"][t]].sum()
            if hook is not None:
                hook({"tokens": t, "model": p, "program": q, "follow": q_follow, "costs": costs,
                      "model_costs": model_costs})
            for D in HORIZONS:
                S[f"beyond_{D}_positions"] += CTX - D - 1
    M = _finish(S, len(W))
    for h, v in enumerate(self_vote_strength(W)):
        M[f"head{h}.self_vote"] = v
    return M


def _finish(S: dict, n_windows: int) -> dict:
    n, n1 = S["positions"], n_windows * (CTX - 1)
    M = {
        "windows": n_windows,
        "kl": S["kl"] / n,
        "top1": S["top1"] / n,
        "kl_follow": S["kl_follow"] / n,
        "top1_follow": S["top1_follow"] / n,
        "heads_gain": (S["kl_follow"] - S["kl"]) / n,
        "exceptions_cost": (S["kl_no_exceptions"] - S["kl"]) / n,
        "kl_self_votes": S["kl_self_votes"] / n,
        "self_votes_cost": (S["kl_self_votes"] - S["kl"]) / n,
        "decomposition_error": S["decomposition_error"],
        "first_position_error": S["first_position_error"],
        "upon_n": S["upon_n"],
        "upon_follow_matches_model": S["upon_follow_matches_model"] / max(S["upon_n"], 1),
        "upon_model_top_is_a": S["upon_model_top_is_a"] / max(S["upon_n"], 1),
        "upon_kl_follow": S["upon_kl_follow"] / max(S["upon_n"], 1),
        "upon_follow_p_a": S["upon_follow_p_a"] / max(S["upon_n"], 1),
        "recall_share": S["recall_n"] / n1,
        "recall_top1": S["recall_top1"] / S["recall_n"],
        "recall_top1_follow": S["recall_top1_follow"] / S["recall_n"],
        "said_comma_n": S["said_comma_n"],
        "said_comma_model_quote": S["said_comma_model_quote"] / max(S["said_comma_n"], 1),
        "said_comma_p_quote": S["said_comma_p_quote"] / max(S["said_comma_n"], 1),
        "name_token_share": S["name_tokens"] / n,
    }
    for name in ("inside_word", "after_speech_verb", "after_linker"):
        M[name + "_share"] = S[name + "_n"] / n
    for h in range(N_HEADS):
        k = f"head{h}."
        M[k + "abstain"] = S[k + "abstain"] / n1
        M[k + "previous"] = S[k + "previous"] / n1
        M[k + "near"] = S[k + "near"] / n1
        M[k + "far"] = S[k + "far"] / n1
        M[k + "mean_distance"] = S[k + "distance_sum"] / S[k + "earlier_mass"]
        for D in HORIZONS:
            M[k + f"beyond_{D}"] = S[k + f"beyond_{D}"] / S[f"beyond_{D}_positions"]
        M[k + "name_enrichment"] = (S[k + "name_mass"] / S[k + "earlier_mass"]) / M["name_token_share"]
        M[k + "cost"] = S[k + "cost"] / n
        M[k + "model_cost"] = S[k + "model_cost"] / n
        M[k + "share_of_model"] = M[k + "cost"] / M[k + "model_cost"]
        M[k + "top1_without"] = S[k + "top1_without"] / n
        for name in ("inside_word", "after_speech_verb", "after_linker"):
            m_in, m_out = S[name + "_n"], n - S[name + "_n"]
            for side in ("", "model_"):
                inside = S[k + side + "cost_" + name] / m_in
                outside = S[k + side + "cost_not_" + name] / m_out
                M[k + side + "cost_" + name] = inside
                M[k + side + "cost_not_" + name] = outside
                M[k + side + name + "_ratio"] = inside / outside
        M[k + "gain_recall"] = S[k + "gain_recall"] / S["recall_n"]
        M[k + "gain_other"] = S[k + "gain_other"] / S["other_n"]
        M[k + "recall_ratio"] = M[k + "gain_recall"] / M[k + "gain_other"]
        M[k + "model_gain_recall"] = S[k + "model_gain_recall"] / S["recall_n"]
        M[k + "model_gain_other"] = S[k + "model_gain_other"] / S["other_n"]
        M[k + "model_recall_ratio"] = M[k + "model_gain_recall"] / M[k + "model_gain_other"]
        M[k + "recall_top1_without"] = S[k + "recall_top1_without"] / S["recall_n"]
        M[k + "said_comma_p_quote_without"] = (S[k + "said_comma_p_quote_without"]
                                               / max(S["said_comma_n"], 1))
    return {key: float(v) for key, v in M.items()}


@lru_cache(maxsize=1)
def claim_numbers() -> dict:
    """measure() on the 200 claim windows, computed once per test session."""
    return measure()


def within(value: float, measured: float, rel: float = 0.30) -> None:
    """The claim's number, +- rel (default 30%)."""
    low, high = measured * (1 - rel), measured * (1 + rel)
    assert low <= value <= high, f"{value:.4g} outside [{low:.4g}, {high:.4g}] (claim states {measured:.4g})"


def at_least(value: float, measured: float, slack: float = 0.25) -> None:
    """One-sided: value >= measured * (1 - slack)."""
    low = measured * (1 - slack)
    assert value >= low, f"{value:.4g} below {low:.4g} (claim states {measured:.4g})"


def at_most(value: float, bound: float) -> None:
    """One-sided: value <= bound."""
    assert value <= bound, f"{value:.4g} above the claimed bound {bound:.4g}"
