"""
Training utilities for MMT.

This package provides training loops, loss functions, and optimization utilities for multi-modal transformer models.

Key modules
-----------
- loop.py       : main training loop implementation
- loop_utils.py : training loop helper functions
- losses.py     : loss function implementations
- scheduler.py  : learning rate scheduling utilities
"""

__all__ = ["train_finetune"]


# ----------------------------------------------------------------------------------------------------------------------
def __getattr__(name: str):
    """Lazily expose the training entrypoint without creating import cycles."""

    if name == "train_finetune":
        from .loop import train_finetune

        return train_finetune
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
