"""Self-test ONLY (lint off, sandbox off, never under program/): wraps the trained model directly.
Expect KL near zero. The model is chosen by the TRLM_MODEL environment variable."""
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)
import torch  # noqa: E402

from harness.common import load_model  # noqa: E402

torch.set_num_threads(1)
_M = load_model(os.environ.get("TRLM_MODEL_PATH") or os.path.join(_REPO, "models", "final", "attn_s0", "final.pt"))


def predict(tokens):
    with torch.no_grad():
        x = torch.as_tensor([list(tokens)], dtype=torch.long)
        return torch.log_softmax(_M(x)[0].float(), -1).double().numpy()
