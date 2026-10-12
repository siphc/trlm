"""Build probe_samples for a model: 2000 windows sampled from the model at temperature 1.
READ-ONLY for replacement agents. Usage: python -m harness.probes.make_probe_samples attn_s0 [...]

Each window starts with the end-of-story token (id 0) and continues with 256 sampled tokens,
giving shape (2000, 257) like the other evaluation sets. Sampling uses the fp32 model on CPU with
torch.Generator seeded by PROBE_SEED (the same seed for every model). Stored once; never rebuilt.
"""
import os
import sys

import numpy as np
import torch

from harness.common import REPO, load_model

N, CTX, PROBE_SEED, BS = 2000, 256, 12345, 100


@torch.no_grad()
def sample(model, n: int, seed: int) -> np.ndarray:
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(0, n, BS):
        x = torch.zeros((min(BS, n - i), 1), dtype=torch.long)
        for _ in range(CTX):
            logits = model(x)[:, -1].float()
            nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=g)
            x = torch.cat([x, nxt], dim=1)
        out.append(x.numpy().astype(np.uint16))
    return np.concatenate(out)


def main() -> None:
    torch.set_num_threads(8)
    os.makedirs(os.path.join(REPO, "data", "eval", "probe"), exist_ok=True)
    for model_id in sys.argv[1:]:
        path = os.path.join(REPO, "data", "eval", "probe", f"{model_id}.npy")
        if os.path.exists(path):
            print(f"{model_id}: exists, skipping (evaluation sets are immutable)")
            continue
        w = sample(load_model(model_id).float().eval(), N, PROBE_SEED)
        np.save(path, w)
        print(f"{model_id}: {w.shape} -> {path}")


if __name__ == "__main__":
    main()
