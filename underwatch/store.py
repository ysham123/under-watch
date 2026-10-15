"""Save and load parameters (any nested dict/list of arrays) with pickle."""

import os
import pickle

import jax


def save(path, tree):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(jax.device_get(tree), f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)
