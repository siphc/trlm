"""Self-test program: uniform distribution over the vocabulary. Expect a large KL."""
import numpy as np


def predict(tokens):
    return np.full((len(tokens), 2000), -np.log(2000.0))
