"""Baseline (not a replacement program): unigram from the train split. Reads data/baselines at import."""
import os

import numpy as np

_LP = np.load(os.path.join(os.path.dirname(__file__), "..", "..", "data", "baselines", "unigram_logprobs.npy"))


def predict(tokens):
    return np.tile(_LP, (len(tokens), 1))
