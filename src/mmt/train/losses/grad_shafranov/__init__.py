"""Grad-Shafranov physics-informed loss implementations and supporting utilities."""

from .strong import StrongFormGradShafranovLoss
from .weak import WeakFormGradShafranovLoss

__all__ = [
    "StrongFormGradShafranovLoss",
    "WeakFormGradShafranovLoss",
]
