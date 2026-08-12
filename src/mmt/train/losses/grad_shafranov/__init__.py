"""Grad-Shafranov physics-informed loss implementations and supporting utilities."""

from .strong import GradShafranovResidualLoss
from .weak import WeakFormGradShafranovLoss

__all__ = [
    "GradShafranovResidualLoss",
    "WeakFormGradShafranovLoss",
]
