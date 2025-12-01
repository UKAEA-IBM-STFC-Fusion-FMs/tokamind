from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True, warn_only: bool = True):
    """
    Global reproducibility across Python, NumPy, and PyTorch (CPU/CUDA/MPS).
    Call once at startup, before building datasets/loaders/models.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Needed for strict cuBLAS determinism (matmul). Safe if CUDA isn't present.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN + global deterministic guard
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic)
    except Exception as ee:
        print(f"WARNING - torch exception triggered: {ee}")
        pass

    torch.use_deterministic_algorithms(bool(deterministic), warn_only=bool(warn_only))
