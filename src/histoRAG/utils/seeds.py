"""Utilities for setting all random seeds for full reproducibility."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Set seeds for Python random, NumPy, PyTorch CPU and CUDA.

    Also sets cuDNN deterministic mode to ensure bit-exact results
    across runs with the same seed on the same hardware.

    Call this once at the start of any script that requires reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
