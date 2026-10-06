"""Runtime helpers shared across integration layers."""

from __future__ import annotations

import torch


def setup_device() -> torch.device:
    """
    Auto-detect the best available torch device.

    Returns
    -------
    torch.device
        ``cuda`` if available, else ``mps`` if available, else ``cpu``.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
