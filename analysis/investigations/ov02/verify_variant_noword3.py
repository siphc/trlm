"""verify_variant_noword3: ov02_earlier_votes4 with some directions dropped (default: h2.word3, whose fitted strength
was ~0.03), as a vote_fit candidate. Verifier-side leave-one-out check of which directions carry the gain.

OV02_DROP = comma-separated direction names to drop (default "h2.word3").
Run: OV02_DROP=h2.word3 .venv/bin/python analysis/investigations/ov02/verify_run.py \
         analysis/investigations/ov02/verify_variant_noword3.py --fit_seed 2031 --eval_seed 2032 --eval 96 \
         --no_heads --tag _verify_drop_h2.word3
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "ov02_cand4", "/Users/svi/Desktop/trlm/analysis/candidates/ov02_earlier_votes4.py")
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)

NAME = "ov02_verify_drop"
HEADS = [0, 2]
DROP = [d for d in os.environ.get("OV02_DROP", "h2.word3").split(",") if d]


def features(P, tokens):
    f = _c.features(P, tokens)
    for d in DROP:
        f.pop(d)
    return f
