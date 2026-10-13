"""Measure every number cited in the CLAIM docstrings of program/*.py, on the program's own behaviour.

  python -m analysis.claims_measure [--model attn_s0] [--no_train]

Prints each claim number on the 200 claim windows (held-out dev windows; program/tests/ checks these
values) and on held-out TRAIN windows (analysis.fitlib.train_windows(200, seed=2000)) as a cross-check.
The definitions live in program/tests/measurements.py, which the claim tests share, so a printed
number and its test cannot drift apart. Also prints analysis-only numbers that no test checks: the
model heads' own attention profile (same definitions), FOLLOW-alone continuations of a few tokens,
per-token head costs after common linkers, and the phrase examples quoted in the head docstrings.

Re-run after any table change, then update the CLAIM docstrings and the MEASURED values in program/tests/.
Writes nothing. Peak memory about 1 GB; about 3 minutes on the laptop CPU.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "program", "tests"))

FOLLOW_EXAMPLES = [" upon", "Once", " wanted", " couldn", " named", " time", " Lily"]
LINKER_EXAMPLES = [" and", ",", " to", " of", " with"]
# The phrase examples quoted in the head docstrings: (text, next token, head).
PHRASE_EXAMPLES = [("She had a proble", "m", 0),
                   ("Once upon a time, there was a little girl. She was very tired. She closed her", " eyes", 2),
                   ("He closed his", " eyes", 2),
                   ("Tim said,", ' "', 2),
                   ("He likes to run and", " jump", 3)]


class PerToken:
    """Analysis hook: per-token head costs and FOLLOW-alone agreement at chosen current tokens."""

    def __init__(self, ids: dict):
        self.ids = ids
        self.n = defaultdict(float)
        self.cost = defaultdict(float)
        self.follow_match = defaultdict(float)

    def __call__(self, a: dict) -> None:
        t, top = a["tokens"], a["model"].argmax(1)
        follow_top = a["follow"].argmax(1)
        for name, tid in self.ids.items():
            m = t == tid
            if not m.any():
                continue
            self.n[name] += m.sum()
            self.follow_match[name] += (follow_top[m] == top[m]).sum()
            for h, (c, mc) in enumerate(zip(a["costs"], a["model_costs"])):
                self.cost[name, h, "program"] += c[m].sum()
                self.cost[name, h, "model"] += mc[m].sum()


def model_attention_profile(model_id: str, windows: np.ndarray) -> dict:
    """The MODEL heads' attention, with the program's definitions (positions 1..255)."""
    from analysis.tools import Model

    M = Model(model_id)
    r = np.arange(256)
    dist = r[:, None] - r[None, :]
    d1 = dist[1:]
    acc = defaultdict(float)
    for w in windows:
        pats = M.run(w[:256])["patterns"]
        for h, a in enumerate(pats):
            a1 = a[1:]
            acc[h, "self"] += a[r[1:], r[1:]].sum()
            acc[h, "previous"] += a[r[1:], r[1:] - 1].sum()
            acc[h, "near"] += a1[(d1 >= 2) & (d1 <= 8)].sum()
            acc[h, "far"] += a1[d1 > 8].sum()
            acc[h, "dist"] += (a1 * d1 * (d1 > 0)).sum()
            acc[h, "earlier"] += (a1 * (d1 > 0)).sum()
    n1 = len(windows) * 255
    out = {}
    for h in range(len(pats)):
        out[h] = {k: acc[h, k] / n1 for k in ("self", "previous", "near", "far")}
        out[h]["mean_distance"] = acc[h, "dist"] / acc[h, "earlier"]
    return out


def phrase_examples(P, measurements) -> None:
    """Program probability of the quoted next token with the head, and with its votes replaced by their
    average (mean-ablation, as in the claims); the model's top token for comparison."""
    from harness import claims
    from src.train.data import load_tokenizer

    tok = load_tokenizer()
    avg_heads, _ = measurements.average_votes()
    for text, nxt, h in PHRASE_EXAMPLES:
        t = np.array(tok.encode(text).ids)
        follow = P.follow.votes(t)
        votes = [m.votes(t) for m in P.HEADS]
        score = follow + sum(votes)
        q = measurements.log_softmax(score)[-1]
        q_h = measurements.log_softmax(score - votes[h] + avg_heads[h])[-1]
        p = claims.model_logprobs(t[None, :])[0, -1]
        v = P.vocab.ID[nxt]
        print(f"    {text[-30:]!r:>34} -> {nxt!r:<7} head {h}: p {np.exp(q[v]):.2f} with, "
              f"{np.exp(q_h[v]):.2f} without (top without: {P.vocab.TOKENS[q_h.argmax()]!r}) | "
              f"model top {P.vocab.TOKENS[p.argmax()]!r} {np.exp(p.max()):.2f}")


def _fmt(v: float) -> str:
    return f"{v:12.2e}" if v != 0 and abs(v) < 0.01 else f"{v:12.4f}"


def print_numbers(Mc: dict, Mt: dict | None) -> None:
    groups = [("program/__init__.py", ["kl", "top1", "kl_follow", "top1_follow", "heads_gain",
                                       "decomposition_error"]),
              ("attention.py (abstention rule)", ["first_position_error", "kl_self_votes", "self_votes_cost"]
               + [f"head{h}.abstain" for h in range(4)]),
              ("follow.py", ["kl_follow", "top1_follow", "exceptions_cost", "heads_gain", "upon_n",
                             "upon_follow_matches_model", "upon_model_top_is_a", "upon_kl_follow",
                             "upon_follow_p_a"]),
              ("position splits", ["recall_share", "recall_top1", "recall_top1_follow", "inside_word_share",
                                   "after_speech_verb_share", "after_linker_share", "said_comma_n",
                                   "said_comma_model_quote", "said_comma_p_quote", "name_token_share"])]
    for h in range(4):
        groups.append((f"head{h}.py", sorted(k for k in Mc if k.startswith(f"head{h}."))))
    for title, keys in groups:
        print(f"\n== {title}")
        print(f"  {'number':<44}{'claims':>12}{'train':>12}")
        for k in keys:
            t = _fmt(Mt[k]) if Mt is not None else ""
            print(f"  {k:<44}{_fmt(Mc[k])}{t}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="attn_s0")
    ap.add_argument("--no_train", action="store_true", help="skip the held-out TRAIN cross-check")
    args = ap.parse_args()
    os.environ["TRLM_MODEL"] = args.model
    os.environ.setdefault("TRLM_PROGRAM", os.path.join(REPO, "program"))

    from harness import claims
    from harness.common import program_sha

    import measurements

    P = claims.program()
    ids = {name: P.vocab.ID[name] for name in FOLLOW_EXAMPLES + LINKER_EXAMPLES if name in P.vocab.ID}
    hook_c = PerToken(ids)
    print(f"program_sha {program_sha(os.environ['TRLM_PROGRAM'])}  model {args.model}")
    Mc = measurements.measure(hook=hook_c)
    Mt = None
    if not args.no_train:
        from analysis.fitlib import train_windows

        Mt = measurements.measure(train_windows(200, seed=2000))
    print_numbers(Mc, Mt)

    print("\n== analysis only (not tested)")
    W = claims.claim_windows()
    prof = model_attention_profile(args.model, W)
    print("  MODEL heads' attention on the claim windows (program's definitions; 'self' = abstain):")
    for h, d in prof.items():
        print(f"    head {h}: " + "  ".join(f"{k} {v:.3f}" for k, v in d.items()))
    print("  FOLLOW alone, top continuations (static) and agreement with the model's top token at held-out "
          "occurrences (claims windows):")
    for name in FOLLOW_EXAMPLES:
        if name not in ids:
            continue
        pf = np.exp(measurements.log_softmax(P.follow.FOLLOW[ids[name]]))
        top3 = np.argsort(-pf)[:3]
        n = hook_c.n.get(name, 0)
        agree = hook_c.follow_match[name] / n if n else float("nan")
        print(f"    after {name!r:>10}: " + ", ".join(f"{P.vocab.TOKENS[v]!r} {pf[v]:.3f}" for v in top3)
              + f"   | held-out occurrences {n:.0f}, FOLLOW top = model top {agree:.3f}")
    print("  Head costs (program / model earlier-token ablation) per current token, claims windows:")
    for name in LINKER_EXAMPLES:
        n = hook_c.n.get(name, 0)
        if not n:
            continue
        cells = "  ".join(f"h{h} {hook_c.cost[name, h, 'program'] / n:.3f}/"
                          f"{hook_c.cost[name, h, 'model'] / n:.3f}" for h in range(4))
        print(f"    {name!r:>7} (n {n:.0f}): {cells}")
    print("  Phrase examples quoted in the head docstrings (short synthetic texts):")
    phrase_examples(P, measurements)


if __name__ == "__main__":
    main()
