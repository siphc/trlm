"""Compare a re-run arms file with the committed one: every gain, KL, per-window KL, coefficient, pattern CE, pure-pair
correlation, readout-table correlation and near-duplicate list.

    .venv/bin/python analysis/out/look_write_deep/verify/compare_rerun.py attn2L_s1 verify2L_s1
"""
import json
import sys

import numpy as np

OUT = "/Users/svi/Desktop/trlm/analysis/out/look_write_deep"


def main(ref, new):
    A = json.load(open(f"{OUT}/{ref}/arms_ltotal.json"))
    B = json.load(open(f"{OUT}/{new}/arms_ltotal.json"))
    assert set(A["heads"]) == set(B["heads"])
    worst = {}
    n_vals = 0

    def upd(name, a, b):
        nonlocal n_vals
        n_vals += 1
        d = abs(a - b)
        if d > worst.get(name, (0, None))[0]:
            worst[name] = (d, where)

    for hk in sorted(A["heads"]):
        ha, hb = A["heads"][hk], B["heads"][hk]
        assert ha["class"] == hb["class"]
        assert set(ha["arms"]) == set(hb["arms"]), hk
        for arm in ha["arms"]:
            a, b = ha["arms"][arm], hb["arms"][arm]
            where = f"{hk}/{arm}"
            for k in ("kl", "gain", "pattern_ce_fit", "pattern_ce_eval"):
                upd(k, a[k], b[k])
            for x, y in zip(a["kl_win"], b["kl_win"]):
                upd("kl_win", x, y)
            assert set(a["coef"]) == set(b["coef"])
            for k in a["coef"]:
                upd("coef_" + k, a["coef"][k], b["coef"][k])
        for k in ha["pure_pair_corr"]:
            where = f"{hk}/{k}"
            upd("pure_pair_corr", ha["pure_pair_corr"][k], hb["pure_pair_corr"][k])
        for k in ha["readout_table_corr_d1"]:
            where = f"{hk}/{k}"
            upd("readout_table_corr_d1", ha["readout_table_corr_d1"][k], hb["readout_table_corr_d1"][k])
        if ha["near_duplicates"] != hb["near_duplicates"]:
            print("near-duplicates differ", hk, ha["near_duplicates"], hb["near_duplicates"])
    print(f"{n_vals} values compared")
    for k, (d, w) in worst.items():
        print(f"  {k:24s} max |diff| {d:.3e}  at {w}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
