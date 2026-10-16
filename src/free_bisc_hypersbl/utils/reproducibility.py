"""Deterministic RNG setup for NumPy, Python, and PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np


def set_reproducible_seed(seed: int) -> np.random.Generator:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
    return np.random.default_rng(seed)

