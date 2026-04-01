"""
Reproducibility utilities.

Provides:
- `set_seed(...)` to seed Python/NumPy/PyTorch (CPU/CUDA/MPS) and enable deterministic ops.
- `seed_worker(...)` / `make_worker_seed_fn()` for PyTorch DataLoader worker seeding.
"""

from __future__ import annotations

import os
import random
import numpy as np
from typing import Callable

import torch


# ----------------------------------------------------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True, warn_only: bool = True) -> None:
    """
    Method to allow global reproducibility across Python, NumPy, and PyTorch (CPU/CUDA/MPS).
    Call once at startup, before building datasets/loaders/models.

    Parameters
    ----------
    seed : int
        User-defined seed value to initialize the internal state of Python, NumPy, and PyTorch (CPU/CUDA/MPS).
    deterministic : bool
        Boolean flag to be passed as the `mode` parameter for `torch.use_deterministic_algorithms`.
        If True, makes nondeterministic operations switch to deterministic. If False, nondeterministic operations are
        used.
        Optional. Default: True.
    warn_only : bool
        Boolean flag to be passed as the `warn_only` parameter for `torch.use_deterministic_algorithms`.
        If True, throw warning instead of error for operations without deterministic implementation.
        Optional. Default: True.

    Returns
    -------
    None

    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault(key="CUBLAS_WORKSPACE_CONFIG", value=":4096:8")

    random.seed(a=seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed=seed)

    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic)
    except Exception as ee:
        print(f"WARNING - torch exception triggered: {ee}")

    torch.use_deterministic_algorithms(mode=bool(deterministic), warn_only=bool(warn_only))


# ----------------------------------------------------------------------------------------------------------------------
def seed_worker(worker_id: int) -> None:
    """
    Worker init function for PyTorch DataLoader.

    Derives a per-worker RNG seed from PyTorch's internal worker seed and seeds NumPy and Python's `random` module
    accordingly.

    """

    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed=worker_seed)
    random.seed(a=worker_seed)


# ----------------------------------------------------------------------------------------------------------------------
def make_worker_seed_fn() -> Callable[[int], None]:
    """
    Backwards-compatible helper returning the top-level `seed_worker` function.

    This indirection keeps the public API flexible while ensuring the returned callable is picklable (required for
    DataLoader workers with 'spawn').

    """

    return seed_worker
