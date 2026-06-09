"""
Global seed management for full reproducibility.

Sets seeds for random, numpy, and torch to ensure deterministic
simulation and training across runs.
"""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set all random seeds for deterministic behaviour.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not installed — skip


def get_rng(seed: int) -> np.random.Generator:
    """Create a seeded NumPy random generator.

    Use this for per-component randomness that shouldn't
    affect the global RNG state.

    Args:
        seed: Integer seed value.

    Returns:
        A seeded numpy Generator.
    """
    return np.random.default_rng(seed)
