"""Growth test (logs/prereg_look_follows_write_addendum2.md): does the vote form before the attention, and is attention
acquired where the vote is useful? One-layer models attn_s1..s4, their confirmatory LOCAL heads.

Needs the echo-test outputs (analysis/out/look_write_deep/attn_sK@<ckpt>/traj_tables.npz from echo_traj.py).

    OMP_NUM_THREADS=2 .venv/bin/python -m analysis.look_write_deep.growth [--models attn_s1 ...] [--draws 2000]
Writes analysis/out/look_write_deep/growth.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.look_write import common as C1
from analysis.look_write.fit_arms import double_centre
from analysis.look_write_deep import common_deep as C
from analysis.look_write_deep.echo_traj import CKPTS

ALL_CKPTS = ["ckpt_step16", "ckpt_step23", "ckpt_step34", "ckpt_step50", "ckpt_step73", "ckpt_step107", "ckpt_step157",
             "ckpt_step229", "ckpt_step335", "ckpt_step491", "ckpt_step718", "ckpt_step1050", "ckpt_step1536",
             "ckpt_step2246", "ckpt_step3286", "ckpt_step4807", "ckpt_step7032", "ckpt_step10286", "ckpt_step15046",
             "ckpt_step22009", "final"]
N_TOP_KEYS = 500
TOP_FRAC = 0.10


def wcorr_rows(A, B, wr):
    """Weighted correlation of two (rows x V) tables, rows weighted by wr, columns equally."""
    w = np.repeat(wr[:, None], A.shape[1], 1).ravel()
    a, b = A.ravel(), B.ravel()
    a = a - (w * a).sum() / w.sum()
    b = b - (w * b).sum() / w.sum()
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum()))


def wcorr(a, b, w):
    a = a - (w * a).sum() / w.sum()
    b = b - (w * b).sum() / w.sum()
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum()))


def conv_point(corrs, thr):
    """Index of the first checkpoint from which corr >= thr at every later checkpoint (len if never)."""
    for i in range(len(corrs)):
        if all(c >= thr for c in corrs[i:]):
            return i
    return len(corrs)


def deciles(w):
    edges = np.quantile(w, np.linspace(0, 1, 11)[1:-1])
    return np.searchsorted(edges, w, side="right")


def stratified_perm(values, is_entering, strata, rng, draws):
    """Observed statistic: mean over entering items of (value - mean value of the control items in the same stratum);
    one-sided permutation p-value, entering labels shuffled within strata (vectorised). Strata without both kinds of
    items are dropped."""
    values = np.asarray(values, np.float64)
    is_entering = np.asarray(is_entering, bool)
    _, g = np.unique(np.asarray(strata), return_inverse=True)
    n_g = np.bincount(g)
    n_e = np.bincount(g, weights=is_entering).astype(int)
    keep_g = (n_e > 0) & (n_e < n_g)
    keep = keep_g[g]
    if not keep.any():
        return None, None, 0
    values, is_entering, g = values[keep], is_entering[keep], g[keep]
    G = g.max() + 1
    n_g = np.bincount(g, minlength=G)
    n_e = np.bincount(g, weights=is_entering, minlength=G).astype(int)
    tot_g = np.bincount(g, weights=values, minlength=G)

    def stat(lab):
        se = np.bincount(g, weights=values * lab, minlength=G)
        ok = n_e > 0
        sc = tot_g - se
        contrib = se[ok] - n_e[ok] * sc[ok] / (n_g[ok] - n_e[ok])
        return contrib.sum() / n_e[ok].sum()

    obs = stat(is_entering)
    order0 = np.argsort(g, kind="stable")
    starts = np.concatenate([[0], np.cumsum(n_g)[:-1]])
    count = 0
    for _ in range(draws):
        order = np.lexsort((rng.random(len(g)), g))           # items grouped by stratum, random order inside
        rank_in_group = np.empty(len(g), int)
        rank_in_group[order] = np.arange(len(g)) - starts[g[order]]
        lab = rank_in_group < n_e[g]
        if stat(lab) >= obs:
            count += 1
    del order0
    return float(obs), (count + 1) / (draws + 1), int(n_e.sum())


def run(models, draws, seed=0):
    rng = np.random.default_rng(seed)
    from analysis.mech_eval import stats

    uni = np.asarray(stats().unigram, np.float64)
    pu = uni / uni.sum()
    top_keys = np.argsort(-uni)[:N_TOP_KEYS]
    out = {"models": {}, "draws": draws}
    for model in models:
        classes = json.load(open(os.path.join(C.model_dir(model), "classes.json")))["heads"]
        heads = [c["head"] for c in classes if c["confirmatory"]]
        traj = {ck: np.load(os.path.join(C.model_dir(f"{model}@{ck}"), "traj_tables.npz")) for ck in CKPTS}
        keys, w = traj["final"]["keys"], traj["final"]["w"]
        kk, qq = (keys // C.V) % C.V, keys % C.V
        # G1: vote and attention tables at all checkpoints
        ov, S = {}, {}
        for ck in ALL_CKPTS:
            m = C1.load(C.model_path(f"{model}@{ck}"))
            wts = C1.weights(m)
            ovt = C1.ov_tables(wts, 1.0)                       # correlation is scale-free
            pps = C1.pure_pair_scores(wts)
            ov[ck] = {h: ovt[h][top_keys].astype(np.float64) for h in heads}
            S[ck] = {}
            for h in heads:
                Y = np.zeros((C.V, C.V))
                Y[kk, qq] = pps[h][kk, qq]
                S[ck][h] = double_centre(Y, kk, qq, w)[0]
        ov_full = {ck: C1.ov_tables(C1.weights(C1.load(C.model_path(f"{model}@{ck}"))), 1.0) for ck in CKPTS[:-1]}
        res = {}
        dec = deciles(w)
        for h in heads:
            ov_c = [wcorr_rows(ov[ck][h], ov["final"][h], pu[top_keys]) for ck in ALL_CKPTS]
            s_c = [wcorr(S[ck][h], S["final"][h], w) for ck in ALL_CKPTS]
            g1 = {f"thr{t}": {"conv_vote": ALL_CKPTS[min(conv_point(ov_c, t), len(ALL_CKPTS) - 1)],
                              "conv_attention": ALL_CKPTS[min(conv_point(s_c, t), len(ALL_CKPTS) - 1)],
                              "holds": conv_point(ov_c, t) < conv_point(s_c, t)} for t in (0.5, 0.8, 0.9)}
            # G2 and G3 over the echo-test transitions
            vals2, ent2, str2 = [], [], []
            vals3, ent3, str3 = [], [], []
            for ti, (a, b) in enumerate(zip(CKPTS[:-1], CKPTS[1:])):
                Sa, Sb, Ua = traj[a]["S"][h], traj[b]["S"][h], traj[a]["U"][h]
                M = len(Sa)
                ntop = int(TOP_FRAC * M)
                Ta = np.zeros(M, bool)
                Ta[np.argsort(-Sa)[:ntop]] = True
                Tb = np.zeros(M, bool)
                Tb[np.argsort(-Sb)[:ntop]] = True
                entering = Tb & ~Ta
                control = ~Ta & ~Tb
                sel = entering | control
                vals2.append(Ua[sel]); ent2.append(entering[sel]); str2.append([f"{ti}:{d}" for d in dec[sel]])
                # G3: per query token with >= 3 core keys
                ova = ov_full[a][h].astype(np.float64)
                core = {}
                for i in np.nonzero(Ta)[0]:
                    core.setdefault(qq[i], []).append(kk[i])
                for i in np.nonzero(sel)[0]:
                    q = qq[i]
                    if len(core.get(q, [])) < 3:
                        continue
                    centroid = ova[core[q]].mean(0)
                    v = ova[kk[i]]
                    a_ = v - (pu * v).sum()
                    c_ = centroid - (pu * centroid).sum()
                    sim = (pu * a_ * c_).sum() / np.sqrt((pu * a_ * a_).sum() * (pu * c_ * c_).sum())
                    vals3.append(sim); ent3.append(entering[i]); str3.append(f"{ti}:{q}:{dec[i]}")
            g2_obs, g2_p, g2_n = stratified_perm(np.concatenate(vals2), np.concatenate(ent2),
                                                 sum(str2, []), rng, draws)
            g3_obs, g3_p, g3_n = stratified_perm(np.asarray(vals3), np.asarray(ent3), str3, rng, draws) \
                if vals3 else (None, None, 0)
            res[h] = {"vote_corr_to_final": ov_c, "attention_corr_to_final": s_c, "G1": g1,
                      "G2": {"stat": g2_obs, "p": g2_p, "n_entering": g2_n,
                             "holds": g2_obs is not None and g2_obs > 0 and g2_p < 0.05},
                      "G3": {"stat": g3_obs, "p": g3_p, "n_entering": g3_n,
                             "holds": g3_obs is not None and g3_obs > 0 and g3_p < 0.05}}
            print(f"{model} h{h}: G1(0.8) vote {g1['thr0.8']['conv_vote']} vs attention {g1['thr0.8']['conv_attention']} "
                  f"-> {g1['thr0.8']['holds']} | G2 {g2_obs:+.3f} p={g2_p:.4f} (n={g2_n}) | "
                  f"G3 {g3_obs if g3_obs is None else round(g3_obs, 4)} p={g3_p} (n={g3_n})", flush=True)
        out["models"][model] = res
    flat = [r for m in out["models"].values() for r in m.values()]
    out["summary"] = {"n_heads": len(flat),
                      "G1_holds_0.8": sum(r["G1"]["thr0.8"]["holds"] for r in flat),
                      "G1_holds_0.5": sum(r["G1"]["thr0.5"]["holds"] for r in flat),
                      "G1_holds_0.9": sum(r["G1"]["thr0.9"]["holds"] for r in flat),
                      "G2_holds": sum(r["G2"]["holds"] for r in flat),
                      "G3_holds": sum(r["G3"]["holds"] for r in flat)}
    with open(os.path.join(C.OUT, "growth.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["summary"]), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["attn_s1", "attn_s2", "attn_s3", "attn_s4"])
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    run(a.models, a.draws)
