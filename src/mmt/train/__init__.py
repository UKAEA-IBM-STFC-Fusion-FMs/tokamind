"""
Training utilities for MMT.

This package provides training loops, loss functions, and optimization
utilities for multi-modal transformer models.

Key modules
-----------
- loop.py       : main training loop implementation
- loop_utils.py : training loop helper functions
- losses.py     : loss function implementations
- scheduler.py  : learning rate scheduling utilities
"""

from .loop import train_finetune

__all__ = [
    "train_finetune",
]
