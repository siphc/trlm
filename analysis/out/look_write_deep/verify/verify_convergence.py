"""Convergence check of the arm fits (L-BFGS in analysis.look_write.fit_arms.fit_rule): refit the own (primary) and
oracle arms for chosen heads, report the max |gradient| at the solution, then restart L-BFGS from the solution and
report the change in pattern CE and output gain.

    .venv/bin/python analysis/out/look_write_deep/verify/verify_convergence.py attn2L_s4:L0H3 attn4L_s4:L0H0 ...
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/Users/svi/Desktop/trlm")
from analysis.look_write.fit_arms import Rule  # noqa: E402
from analysis.look_write_deep import common_deep as C  # noqa: E402
from analysis.look_write_deep.fit_arms_deep import slot_features, tables  # noqa: E402

torch.set_num_threads(2)


def refit(A, rule, feats, theta0, max_iter=2000):
    theta = {k: v.clone().requires_grad_(True) for k, v in theta0.items()}
    nq = A.shape[0] * (A.shape[1] - 1)
    opt = torch.optim.LBFGS(list(theta.values()), lr=1.0, max_iter=max_iter, tolerance_grad=1e-12,
                            tolerance_change=1e-15, history_size=50, line_search_fn="strong_wolfe")
    low = rule.mask[None]

    def closure():
        opt.zero_grad()
        lp = rule.logpattern(theta, feats)
        loss = -(torch.where(low, A * lp, torch.zeros_like(lp))).sum() / nq
        loss.backward()
        return loss

    opt.step(closure)
    loss = float(closure())
    g = max(float(p.grad.abs().max()) for p in theta.values())
    return {k: v.detach().clone() for k, v in theta.items()}, loss, g


def main(specs):
    from analysis.look_write.fit_arms import fit_rule

    for spec in specs:
        mid, hk = spec.split(":")
        l, h = int(hk[1]), int(hk[3])
        m = C.load(mid)
        d = C.model_dir(mid)
        z = np.load(f"{d}/bgu.npz")
        keys = z["keys"]
        tabs = tables(z, "ltotal")
        ref = json.load(open(f"{d}/arms_ltotal.json"))["heads"][hk]["arms"]
        shapes = C.token_shapes()
        fit_w = C.windows(C.FIT_SEED, C.N_FIT)[:, :C.T]
        ev_w = C.windows(C.EVAL_SEED, C.N_EVAL)[:, :C.T]
        A_f = torch.as_tensor(C.patterns(m, fit_w)[l][:, h].astype(np.float64))
        lp_ref = C.logprobs(m, ev_w)
        rf, re_ = Rule(fit_w, shapes), Rule(ev_w, shapes)
        base_kl = ref["base"]["kl"]
        for arm, tab in (("base", None), ("own_ltotal", tabs["own_ltotal"][l, h]), ("oqc", tabs["oqc"][l, h])):
            if tab is None:
                ff, fe = {}, {}
            else:
                F1f, F2f = slot_features(fit_w, keys, tab)
                F1e, F2e = slot_features(ev_w, keys, tab)
                ff = {"c1": torch.as_tensor(F1f), "c2": torch.as_tensor(F2f)}
                fe = {"c1": torch.as_tensor(F1e), "c2": torch.as_tensor(F2e)}
            th, loss, g = fit_rule(A_f, rf, ff)
            th2, loss2, g2 = refit(A_f, rf, ff, th)
            out = []
            for t in (th, th2):
                with torch.no_grad():
                    pat = torch.exp(re_.logpattern(t, fe)).float().numpy()
                out.append(C.kl(lp_ref, C.logprobs(m, ev_w, sub=(l, h, pat))))
            print(f"{mid} {hk} {arm:10s}: CE {loss:.8f} -> {loss2:.8f} (max|grad| {g:.1e} -> {g2:.1e}); "
                  f"KL {out[0]:.6f} (committed {ref[arm]['kl']:.6f}) -> restarted {out[1]:.6f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
