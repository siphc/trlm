"""Spec probe for mechanism program v1: do the circuit-card rules compose into a STANDALONE program?

Every circuit card measured one head at a time, with the rest of the model exact and the model's own
final-LayerNorm scale. Here every part is a rule; there is no exact model part and no model scale.
All token-level knowledge comes from TRAIN-corpus statistics cached by the earlier cards (knowledge/,
head1_card/, h2card/); only named scalars are fitted, to the MODEL's log-probs (KL) on TRAIN windows.

score_i[v] = U*logP_uni(v) + B*logP_bigram(v | t_i)                        current token (direct path)
           + sum_r LOCAL_r(i, v)    r = head 0, 2, 3 readers                 local context
           + TRI*tri_i(v) + QUAD*quad_i(v)                                   phrase continuation
           + RECALL(i, v)                                                    head 1 recall
           + ECHO*[i == 0]*[v == t_0]                                        window-start echo
LOCAL_r(i, v) = S_SELF*a_r(i,i)*PMI_1(t_i, v) + S_CTX*sum_{1<=d<=7} a_r(i,i-d)*PMI_{d+1}(t_{i-d}, v)
              + S_FAR*sum_{d>=8} a_r(i,i-d)*PMI_8(t_{i-d}, v),     a_r = softmax_j(card attention rule r)
RECALL(i, v)  = sum_j b(i,j) * (COPY[g]*[v == t_j] + TOPIC[g]*PMI_far(t_j, v) + NEXT[g]*PMI_1(t_j, v)), g = group of t_j

Run: OMP_NUM_THREADS=2 .venv/bin/python analysis/scratch/spec_v1/v1probe.py [n_eval]
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import torch

torch.set_num_threads(2)
REPO = "/Users/svi/Desktop/trlm"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/analysis/scratch/knowledge")
import ctx  # noqa: E402  (LBIG, LU, PMI d=2..8: knowledge-baseline corpus tables)
import ctx2  # noqa: E402  (trigram / 4-gram features, band PMIs)
from kb import V, class_index, model_logprobs, train_windows  # noqa: E402
from program.vocab import TOKENS  # noqa: E402

HERE = f"{REPO}/analysis/scratch/spec_v1"
T = 256
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


# ---------------------------------------------------------------- corpus tables (no model involved)
LU = ctx.LU.astype(np.float32)
LBIG = ctx.LBIG.astype(np.float32)
PMI = {1: (LBIG - LU[None, :]).astype(np.float32)}
PMI.update({d: ctx.PMI[d] for d in range(2, 9)})
H1 = f"{REPO}/analysis/scratch/head1_card"
PMIFAR = np.load(f"{H1}/PMI_9_24.npy").astype(np.float32)          # head-1 topic table (first 50M tokens)
cidx, _ = class_index("CONTEXT")
tidx, _ = class_index("TARGET")
CPPMI = np.maximum(np.load(f"{H1}/PMIcc.npy"), 0)[cidx][:, tidx].astype(np.float32)   # classPPMI[q, k]
LOGPIC = np.maximum(np.load(f"{H1}/logPic.npy"), -6.0).astype(np.float32)             # in-context successor
F2 = np.load(f"{REPO}/analysis/scratch/h2card/features.npz")
INFO = F2["info"].astype(np.float32)        # INFO[d-1, s] = KL(P(token d later | s) || P)
CONT = F2["cont"].astype(np.float32)        # P(next is a word piece | s)
PIECE = F2["is_cont"].astype(np.float32)    # s is a word piece
SPACE = F2["lead_space"].astype(np.float32)
SENTEND = F2["sentend"].astype(np.float32)


def _shape(tok):
    if tok == "<|eos|>":
        return 3
    if tok.startswith(" ") and len(tok) > 1 and tok[1].isalpha():
        return 0
    if tok[:1].isalpha():
        return 1
    return 2


SHAPE = np.array([_shape(t) for t in TOKENS])            # WORD_START, CONTINUATION, PUNCT_OTHER, EOS
aidx, anames = class_index("ATTENTION")
G3 = {"BOUNDARY": ["SENTENCE_ENDS", "SENTENCE_STARTS", "OPENERS_UNSPACED", "QUOTE_OPENERS", "INTERJECTIONS"],
      "LINKER": ["LINKERS"],
      "VERB": ["VERBS_BEFORE_TO", "PAST_VERBS_TRANSITIVE", "PAST_VERBS_MOTION", "SPEECH_VERBS", "NEGATED_AUXILIARIES",
               "MODALS", "BARE_VERBS_TRANSITIVE", "BARE_VERBS_ACTIVITY", "MENTAL_VERBS", "LINKING_VERBS", "ING_FORMS"],
      "NOUN": ["NAMES", "CREATURES", "OBJECTS", "PLACES", "FAMILY_AND_BODY", "MASS_AND_PLURAL_NOUNS", "THINGS_OUTDOORS",
               "THINGS_MISC", "ABSTRACT_NOUNS", "AFFECTION_NOUNS", "OBJECT_PRONOUNS", "NAMES_UNSPACED", "TIME_WORDS"],
      "PIECE": ["FRAGMENTS_SHORT", "FRAGMENTS_VC", "FRAGMENTS_ENDINGS", "FRAGMENTS_MIXED", "FRAGMENTS_D", "FRAGMENTS_G",
                "FRAGMENTS_H", "FRAGMENTS_L", "SUFFIXES_ING_ED", "SUFFIXES_PLURAL", "NAME_ENDINGS", "LETTERS_LOWER",
                "ONSETS_SINGLE", "ONSETS_A", "ONSETS_B", "ONSETS_C", "ONSETS_D", "VERB_ONSETS"],
      "FUNCTION": ["DETERMINERS", "DETERMINERS_RARE", "PREPOSITIONS", "SUBJECT_PRONOUNS", "ADVERBS", "PARTICLES"]}
G1 = {"NAMES": ["NAMES", "NAMES_UNSPACED"], "SENTENCE_ENDS": ["SENTENCE_ENDS"],
      "WORD_ENDS_AND_LINKERS": ["FRAGMENTS_H", "FRAGMENTS_G", "FRAGMENTS_D", "FRAGMENTS_ENDINGS", "FRAGMENTS_VC",
                                "FRAGMENTS_SHORT", "SUFFIXES_PLURAL", "SUFFIXES_ING_ED", "LETTERS_LOWER", "LINKERS",
                                "PREPOSITIONS"],
      "FUNCTION_WORDS": ["SUBJECT_PRONOUNS", "OBJECT_PRONOUNS", "LINKING_VERBS", "PARTICLES", "DETERMINERS", "MODALS",
                         "TIME_WORDS", "SENTENCE_STARTS", "ADVERBS"]}


def group_index(groups):
    g_of_class = np.full(len(anames), len(groups))
    for gi, members in enumerate(groups.values()):
        for c in members:
            g_of_class[anames.index(c)] = gi
    return g_of_class[aidx]


GRP3 = group_index(G3)     # 7 groups (last = OTHER)
GRP1 = group_index(G1)     # 5 groups (last = CONTENT_WORDS)
log("tables ready")

# ---------------------------------------------------------------- per-window inputs
DD = np.arange(T)[:, None] - np.arange(T)[None, :]
MASK = torch.as_tensor(DD >= 0)
DDc = np.clip(DD, 0, None)
DDT = torch.as_tensor(DDc, dtype=torch.float32)


def window_inputs(wins, loo, extras=False):
    """Feature tensors for windows (n, 257): everything rows-by-source-token, no model."""
    x = np.asarray(wins)[:, :T].astype(np.int64)
    n = len(x)
    G = np.zeros((n, 8, T, V), np.float32)
    G[:, 0] = PMI[1][x]
    for d in range(1, 8):
        G[:, d, d:] = PMI[d + 1][x[:, :-d]]
    inp = {"x": torch.as_tensor(x), "G": torch.as_tensor(G),
           "F8": torch.as_tensor(PMI[8][x]), "FAR": torch.as_tensor(PMIFAR[x]),
           "BIG": torch.as_tensor(LBIG[x])}
    ng, _ = ctx2.ngram_feats(wins, 300.0, 300.0, loo)
    inp["TRI"] = torch.as_tensor(ng["tri"].reshape(n, T, V))
    inp["QUAD"] = torch.as_tensor(ng["quad"].reshape(n, T, V))
    inp["infod"] = torch.as_tensor(np.stack([INFO[np.clip(DDc, 0, 7), xi[None, :]] for xi in x]))
    inp["cpp"] = torch.as_tensor(np.stack([CPPMI[xi[:, None], xi[None, :]] for xi in x]))
    inp["lpic"] = torch.as_tensor(np.stack([LOGPIC[xi[:, None], xi[None, :]] for xi in x]))
    if extras:   # knowledge-baseline long-range features (bands + copy + name), for the L4 comparison
        F = ctx.features(wins, ["copy", "name"])
        F.update(ctx2.band_feats(wins))
        for k in ["copy", "name"] + [f"band{b}" for b in range(5)]:
            inp[k] = torch.as_tensor(F[k].reshape(n, T, V))
    return inp


def tok(a, x):
    return torch.as_tensor(a)[x]


# ---------------------------------------------------------------- the program as a torch module
P0 = {  # card values (pattern-fitted attention; votes rescaled where the card divided by the model's LN scale)
    "U": 0.1, "B": 0.8,
    "h0_slope": [0.275, 0.365, 0.105, 0.242], "h0_self": [1.272, 0.455, 1.524, 3.495],
    "h2_self_base": -0.34, "h2_self_info": 0.45, "h2_self_cont": 0.67, "h2_self_piece": -1.19, "h2_prev": -0.06,
    "h2_decay": 0.29, "h2_decay_far": 0.11, "h2_key_info": 0.50, "h2_key_cont": -1.28, "h2_key_piece": -1.22,
    "h2_key_space": -0.34, "h2_key_end": 0.89,
    "h3_w": [0.07, -0.03, -0.37, -0.98], "h3_r1": 0.129, "h3_r2": 0.024,
    "h3_self": [-1.47, -0.25, 0.51, -0.84, 1.46, 0.85, 0.10], "h3_prev": [-1.38, 1.08, 0.30, -0.56, 0.92, 0.13, 0.45],
    "h3_key": [0.80, 0.01, 0.00, -0.50, -1.03, -0.48, -0.48],
    "h1_logtau": float(np.log(33.6)), "h1_sal": [0.72, -0.94, -0.93, -0.30, 0.0], "h1_kappa": -0.01,
    "h1_bclass": 0.34, "h1_bic": 0.099,
    "h1_copy": [4.23, 1.71, 1.43, 4.94, 2.85], "h1_topic": [2.01, -0.43, 2.54, 1.57, 2.51],
    "h1_next": [-0.24, 0.56, -0.11, -0.04, 0.07],
    "S0": [0.3, 0.6, 0.3], "S2": [0.3, 0.6, 0.3], "S3": [0.2, 0.5, 0.3],
    "TRI": 0.3, "QUAD": 0.2, "ECHO": 0.0,
    "W_skip": [0.4, 0.26, 0.21, 0.23, 0.22, 0.24, 0.3],       # fixed-distance skip-gram weights (L2c-style)
    "W_l4": [0.84, 0.99, 0.62, -0.24, 0.15, 0.59, 0.66],      # bands 0-4, copy, name (L4-style)
}
ATTN_KEYS = {"h0": ["h0_slope", "h0_self"],
             "h2": [k for k in P0 if k.startswith("h2_")],
             "h3": [k for k in P0 if k.startswith("h3_")],
             "h1": ["h1_logtau", "h1_sal", "h1_kappa", "h1_bclass", "h1_bic"]}


class Prog(torch.nn.Module):
    def __init__(self, p=None):
        super().__init__()
        p = p or P0
        self.p = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32)) for k, v in p.items()})
        self.parts = set()

    # --- attention rules (scores before softmax), each (n, T, T)
    def scores(self, r, inp):
        p, x = self.p, inp["x"]
        eye = (DDT == 0).float()
        if r == "h0":
            sh = tok(SHAPE, x)
            s = -p["h0_slope"][sh][:, :, None] * DDT + eye * p["h0_self"][sh][:, :, None]
        elif r == "h2":
            selfs = (p["h2_self_base"] + p["h2_self_info"] * tok(INFO[0], x) + p["h2_self_cont"] * tok(CONT, x)
                     + p["h2_self_piece"] * tok(PIECE, x))
            dist = (-p["h2_decay"] * torch.clamp(DDT - 1, 0, 11) - p["h2_decay_far"] * torch.clamp(DDT - 12, min=0)
                    + p["h2_prev"] * (DDT == 1).float())
            keys = (p["h2_key_cont"] * tok(CONT, x) + p["h2_key_piece"] * tok(PIECE, x) + p["h2_key_space"] * tok(SPACE, x)
                    + p["h2_key_end"] * tok(SENTEND, x))
            s = eye * selfs[:, :, None] + (1 - eye) * (dist + keys[:, None, :] + p["h2_key_info"] * inp["infod"])
        elif r == "h3":
            w = p["h3_w"]
            near = torch.stack([w[0], w[1], w[2], w[3]])[torch.clamp(DDT, max=3).long()]
            mid = w[3] - p["h3_r1"] * torch.clamp(DDT - 3, 0, 29)
            far = -p["h3_r2"] * torch.clamp(DDT - 32, min=0)
            dist = torch.where(DDT <= 3, near, mid + far)
            g = tok(GRP3, x)
            s = (dist + eye * p["h3_self"][g][:, :, None] + (DDT == 1).float() * p["h3_prev"][g][:, :, None]
                 + p["h3_key"][g][:, None, :])
        elif r == "h1":
            g = tok(GRP1, x)
            s = (-DDT / torch.exp(p["h1_logtau"]) + p["h1_sal"][g][:, None, :] + eye * p["h1_kappa"]
                 + p["h1_bclass"] * inp["cpp"] + p["h1_bic"] * inp["lpic"])
        return s.masked_fill(~MASK, float("-inf"))

    def attn(self, r, inp):
        return torch.softmax(self.scores(r, inp), -1)

    def local_votes(self, r, inp):
        a = self.attn(r, inp)
        S = self.p["S" + r[1]]
        G = inp["G"]
        n = a.shape[0]
        diag = [torch.diagonal(a, offset=-d, dim1=1, dim2=2) for d in range(8)]
        selfv = diag[0][:, :, None] * G[:, 0]
        near = 0.0
        for d in range(1, 8):
            w = torch.cat([torch.zeros(n, d), diag[d]], 1)
            near = near + w[:, :, None] * G[:, d]
        far = torch.matmul(a * (DDT >= 8).float(), inp["F8"])
        return S[0] * selfv + S[1] * near + S[2] * far

    def recall_votes(self, inp):
        p, x = self.p, inp["x"]
        b = self.attn("h1", inp)
        g = tok(GRP1, x)
        onehot = torch.nn.functional.one_hot(x, V).float()
        src = (p["h1_copy"][g][:, :, None] * onehot + p["h1_topic"][g][:, :, None] * inp["FAR"]
               + p["h1_next"][g][:, :, None] * inp["G"][:, 0])
        return torch.matmul(b, src)

    def forward(self, inp, off=()):
        p = self.p
        parts = self.parts - set(off)
        s = p["U"] * torch.as_tensor(LU) + p["B"] * inp["BIG"]
        if "skip" in parts:
            for d in range(1, 8):
                s = s + p["W_skip"][d - 1] * inp["G"][:, d]
        for r in ("h0", "h2", "h3"):
            if r in parts:
                s = s + self.local_votes(r, inp)
        if "phrase" in parts:
            s = s + p["TRI"] * inp["TRI"] + p["QUAD"] * inp["QUAD"]
        if "recall" in parts:
            s = s + self.recall_votes(inp)
        if "echo" in parts:
            e = torch.zeros_like(s)
            e[:, 0, :] = torch.nn.functional.one_hot(inp["x"][:, 0], V).float()
            s = s + p["ECHO"] * e
        if "l4long" in parts:
            for k, nm in enumerate([f"band{b}" for b in range(5)] + ["copy", "name"]):
                s = s + p["W_l4"][k] * inp[nm]
        return torch.log_softmax(s, -1)


def sub(inp, sl):
    return {k: v[sl] for k, v in inp.items()}


def kl_t(plog, qlog):
    return (plog.exp() * (plog - qlog)).sum(-1)


def fit(prog, inp, plog, free, iters=80, chunk=8):
    for k, prm in prog.p.items():
        prm.requires_grad_(k in free)
    params = [prog.p[k] for k in free]
    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-10, history_size=20)
    n = len(plog)

    def closure():
        opt.zero_grad()
        tot = 0.0
        for c in range(0, n, chunk):
            sl = slice(c, c + chunk)
            loss = kl_t(plog[sl], prog(sub(inp, sl))).sum() / (n * T)
            loss.backward()
            tot += loss.item()
        return torch.tensor(tot)

    loss = opt.step(closure)
    with torch.no_grad():
        tot = sum(float(kl_t(plog[c:c + chunk], prog(sub(inp, slice(c, c + chunk)))).sum()) for c in range(0, n, chunk))
    return tot / (n * T)


# ---------------------------------------------------------------- model attention patterns (measurement only)
_EX = None


def model_patterns(wins):
    global _EX
    if _EX is None:
        from analysis.tools import Model
        _EX = Model("attn_s0")
    out = []
    for w in np.asarray(wins)[:, :T]:
        out.append(_EX.run(w)["patterns"].astype(np.float32))
    return torch.as_tensor(np.stack(out))     # (n, H, T, T)


def pattern_kl(Am, a):
    """mean over query rows of KL(model row || rule row)."""
    Am = Am.double()
    a = a.double()
    m = Am > 0
    return float(torch.where(m, Am * (torch.log(Am.clamp_min(1e-30)) - torch.log(a.clamp_min(1e-30))), 0.0).sum(-1).mean())


LOC = ["U", "B", "S0", "S2", "S3"]
LOC_ATT = ATTN_KEYS["h0"] + ATTN_KEYS["h2"] + ATTN_KEYS["h3"]
REC = ["h1_copy", "h1_topic", "h1_next"]
ALL_ATT = sum(ATTN_KEYS.values(), [])
FULL = {"h0", "h2", "h3", "phrase", "recall", "echo"}
VARIANTS = [  # name, parts, free scalars, warm start (None: previous variant of the main chain; else a variant name)
    ("A base: bigram", set(), ["U", "B"], None),
    ("B base + fixed skip-grams (L2c)", {"skip"}, ["U", "B", "W_skip"], None),
    ("C base + 3 local readers, card attention frozen", {"h0", "h2", "h3"}, LOC, None),
    ("D base + 3 local readers, attention fitted", {"h0", "h2", "h3"}, LOC + LOC_ATT, None),
    ("E D + phrase (tri, 4-gram)", {"h0", "h2", "h3", "phrase"}, LOC + ["TRI", "QUAD"] + LOC_ATT, None),
    ("F E + recall (head 1), card attention frozen", {"h0", "h2", "h3", "phrase", "recall"},
     LOC + ["TRI", "QUAD"] + REC, None),
    ("G E + recall, all fitted", {"h0", "h2", "h3", "phrase", "recall"}, LOC + ["TRI", "QUAD"] + REC + ALL_ATT, None),
    ("H G + window-start echo", FULL, LOC + ["TRI", "QUAD", "ECHO"] + REC + ALL_ATT, None),
    ("I fixed-weight L4 (skip + phrase + bands + copy + name)", {"skip", "phrase", "l4long"},
     ["U", "B", "W_skip", "TRI", "QUAD", "W_l4"], "B base + fixed skip-grams (L2c)"),
    ("J H + bands + copy + name", FULL | {"l4long"}, LOC + ["TRI", "QUAD", "ECHO", "W_l4"] + REC + ALL_ATT,
     "H G + window-start echo"),
    ("D1 base + 1 local reader (head-2 rule), attention fitted", {"h2"}, ["U", "B", "S2"] + ATTN_KEYS["h2"],
     "A base: bigram"),
    ("H1 H with one local reader (head-2 rule)", {"h2", "phrase", "recall", "echo"},
     ["U", "B", "S2", "TRI", "QUAD", "ECHO"] + REC + ATTN_KEYS["h2"] + ATTN_KEYS["h1"], "H G + window-start echo"),
    ("K H without phrase (no n-gram conjunctions)", {"h0", "h2", "h3", "recall", "echo"},
     LOC + ["ECHO"] + REC + ALL_ATT, "H G + window-start echo"),
]
DEV_VARIANTS = ["A base: bigram", "B base + fixed skip-grams (L2c)", "E D + phrase (tri, 4-gram)", "H G + window-start echo",
                "I fixed-weight L4 (skip + phrase + bands + copy + name)", "J H + bands + copy + name",
                "H1 H with one local reader (head-2 rule)", "K H without phrase (no n-gram conjunctions)"]
VNAMES = [v[0] for v in VARIANTS]
MAIN = "H G + window-start echo"


def evaluate(fitted, wins, loo, names, patterns, removal, label):
    perpos = {nm: [] for nm in names}
    top1 = {nm: [] for nm in names}
    offk = ["h0", "h2", "h3", "phrase", "recall", "echo", "recall:copy only", "recall:no copy"]
    without = {k: [] for k in offk}
    pkl = {nm: {r: [] for r in ["h0", "h1", "h2", "h3"]} for nm in names}
    parts_of = {v[0]: v[1] for v in VARIANTS}
    n = len(wins)
    for c in range(0, n, 16):
        w = wins[c:c + 16]
        pe = torch.as_tensor(model_logprobs(w))
        ie = window_inputs(w, loo=loo, extras=True)
        Am = model_patterns(w[:4]) if (patterns and c < 32) else None
        for nm in names:
            prog = Prog()
            prog.load_state_dict(fitted[nm])
            prog.parts = set(parts_of[nm])
            with torch.no_grad():
                q = prog(ie)
                perpos[nm].append(kl_t(pe, q).reshape(-1).numpy())
                top1[nm].append((pe.argmax(-1) == q.argmax(-1)).reshape(-1).numpy())
                if removal and nm == MAIN:
                    for k in offk:
                        if k.startswith("recall:"):
                            saved = {kk: prog.p[kk].data.clone() for kk in REC}
                            if k == "recall:copy only":
                                prog.p["h1_topic"].data.zero_(); prog.p["h1_next"].data.zero_()
                            else:
                                prog.p["h1_copy"].data.zero_()
                            without[k].append(kl_t(pe, prog(ie)).reshape(-1).numpy())
                            for kk, vv in saved.items():
                                prog.p[kk].data.copy_(vv)
                        else:
                            without[k].append(kl_t(pe, prog(ie, off=(k,))).reshape(-1).numpy())
                if Am is not None:
                    for r, h in [("h0", 0), ("h1", 1), ("h2", 2), ("h3", 3)]:
                        if r in prog.parts or (r == "h1" and "recall" in prog.parts):
                            pkl[nm][r].append(pattern_kl(Am[:, h], prog.attn(r, sub(ie, slice(0, 4)))))
        log(f"{label} {c + len(w)}/{n}")
    out = {}
    for nm in names:
        k = np.concatenate(perpos[nm])
        pw = k.reshape(-1, T).mean(1)
        out[nm] = {"kl": float(k.mean()), "se": float(pw.std(ddof=1) / np.sqrt(len(pw))),
                   "top1": float(np.concatenate(top1[nm]).mean()),
                   "pattern_kl": {r: float(np.mean(v)) for r, v in pkl[nm].items() if v}}
        log(f"{label} {nm:58s} KL {out[nm]['kl']:.4f} +- {out[nm]['se']:.4f} top1 {out[nm]['top1']:.3f} "
            f"pattern KL {out[nm]['pattern_kl']}")
    if removal:
        base = np.concatenate(perpos[MAIN])
        out["removal deltas (main, no refit)"] = {k: float(np.concatenate(v).mean() - base.mean()) for k, v in without.items()}
        log(f"{label} removal deltas:", out["removal deltas (main, no refit)"])
    return out, {k: np.concatenate(v) for k, v in perpos.items()}


def main():
    from harness.common import eval_windows
    n_eval = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    smoke = os.environ.get("SMOKE") == "1"
    fitw = train_windows(32, seed=2024)[: 4 if smoke else 32]
    plog = torch.as_tensor(model_logprobs(fitw))
    inp = window_inputs(fitw, loo=True, extras=True)
    log("fit inputs ready", {k: tuple(v.shape) for k, v in inp.items()})
    Am = model_patterns(fitw[:8])
    prog = Prog()
    with torch.no_grad():
        card_pkl = {r: pattern_kl(Am[:, h], prog.attn(r, sub(inp, slice(0, min(8, len(fitw)))))) for r, h in
                    [("h0", 0), ("h1", 1), ("h2", 2), ("h3", 3)]}
    log("card attention pattern KL on 8 FIT windows:", card_pkl)
    results = {"card_pattern_kl_fit8": card_pkl, "variants": {}}
    fitted = {}
    state = {k: v.detach().clone() for k, v in prog.p.items()}
    for name, parts, free, src in VARIANTS:
        prog = Prog()
        if src is None:
            for k, v in state.items():
                prog.p[k].data.copy_(v)
        else:
            prog.load_state_dict(fitted[src])
        prog.parts = set(parts)
        tl = fit(prog, inp, plog, free, iters=2 if smoke else 80)
        n_free = sum(prog.p[k].numel() for k in free)
        log(f"fit {name}: train KL {tl:.4f} ({n_free} free scalars)")
        fitted[name] = {k: v.detach().clone() for k, v in prog.state_dict().items()}
        if src is None:
            state = {k: v.detach().clone() for k, v in prog.p.items()}
        results["variants"][name] = {"train_kl": tl, "n_free": int(n_free), "parts": sorted(parts),
                                     "params": {k: np.round(prog.p[k].detach().numpy(), 4).tolist() for k in free}}
        json.dump(results, open(f"{HERE}/v1probe_results.json", "w"), indent=1)
    torch.save(fitted, f"{HERE}/v1probe_fitted.pt")
    del inp, plog
    evw = train_windows(160, seed=2025)[: (4 if smoke else n_eval)]
    out, pp = evaluate(fitted, evw, True, VNAMES, True, True, "train-eval")
    results["eval_train_seed2025"] = {"n_windows": len(evw), **out}
    np.savez_compressed(f"{HERE}/v1probe_perpos.npz", **{"train|" + k: v for k, v in pp.items()})
    json.dump(results, open(f"{HERE}/v1probe_results.json", "w"), indent=1)
    dw = eval_windows("dev_fast")[: (4 if smoke else 200)]
    out, ppd = evaluate(fitted, dw, False, DEV_VARIANTS, False, True, "dev_fast")
    results["eval_dev_fast"] = {"n_windows": len(dw), **out}
    np.savez_compressed(f"{HERE}/v1probe_perpos.npz", **{"train|" + k: v for k, v in pp.items()},
                        **{"dev|" + k: v for k, v in ppd.items()})
    json.dump(results, open(f"{HERE}/v1probe_results.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
