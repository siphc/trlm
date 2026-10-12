"""Baseline (not a replacement program): bigram from the train split. Reads data/baselines at import."""
import os

import numpy as np

_LP = np.load(os.path.join(os.path.dirname(__file__), "..", "..", "data", "baselines", "bigram_logprobs.npy"))


def predict(tokens):
    return _LP[np.asarray(tokens, dtype=int)].astype(np.float64)
