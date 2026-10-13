"""Black-box query server: holds the model and answers logit queries over a local socket. Logits only.

Started from the OPEN repo (which has models/); the black-box repo's blackbox/query_api.py and its harness
talk to it, so the black-box session never needs (or gets) the checkpoint.

  python -m blackbox.server --model attn_s0 --address /tmp/trlm_blackbox.sock

Protocol (multiprocessing.connection with an auth key from the environment variable TRLM_BB_KEY):
  request:  {"tokens": [[int, ...], ...]}      sequences of 1..256 token ids in [0, 2000)
  response: {"logits": [np.ndarray float32 (len_i, 2000), ...]}  or {"error": str}
Every query is appended to a log (count of sequences and tokens) so the query budget can be reported.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing.connection import Listener

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from harness.common import load_model  # noqa: E402

MAX_BATCH = 512


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="attn_s0")
    ap.add_argument("--address", default="/tmp/trlm_blackbox.sock")
    ap.add_argument("--log", default=os.path.join(REPO, "logs", "blackbox_queries.jsonl"))
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    key = os.environ.get("TRLM_BB_KEY", "trlm-blackbox").encode()
    torch.set_num_threads(args.threads)
    model = load_model(args.model)
    if os.path.exists(args.address):
        os.remove(args.address)
    listener = Listener(args.address, family="AF_UNIX", authkey=key)
    os.chmod(args.address, 0o600)
    print(f"serving {args.model} logits on {args.address}", flush=True)
    n_seq = n_tok = 0
    while True:
        conn = listener.accept()
        try:
            while True:
                try:
                    req = conn.recv()
                except EOFError:
                    break
                seqs = req.get("tokens") if isinstance(req, dict) else None
                try:
                    if not isinstance(seqs, list) or not 1 <= len(seqs) <= MAX_BATCH:
                        raise ValueError(f"send 1..{MAX_BATCH} sequences under the key 'tokens'")
                    out = []
                    with torch.no_grad():
                        for s in seqs:
                            s = [int(t) for t in s]
                            if not 1 <= len(s) <= 256 or min(s) < 0 or max(s) >= 2000:
                                raise ValueError("each sequence needs 1..256 token ids in [0, 2000)")
                            out.append(model(torch.tensor([s]))[0].float().numpy())
                    conn.send({"logits": out})
                    n_seq += len(seqs)
                    n_tok += sum(len(s) for s in seqs)
                    with open(args.log, "a") as f:
                        f.write(json.dumps({"t": time.time(), "model": args.model, "seqs": len(seqs),
                                            "tokens": sum(len(s) for s in seqs), "total_seqs": n_seq,
                                            "total_tokens": n_tok}) + "\n")
                except Exception as e:  # report and keep serving
                    conn.send({"error": str(e)})
        finally:
            conn.close()


if __name__ == "__main__":
    main()
