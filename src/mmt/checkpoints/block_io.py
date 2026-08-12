"""
Save and load model parameter blocks.

Models declare checkpointable blocks through ``get_named_blocks()``. Each block is saved as ``<block_name>.pt`` in the
checkpoint directory.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
import torch.nn as nn

from mmt.models.blocks import get_named_model_blocks

from .io import atomic_save, torch_load


# ----------------------------------------------------------------------------------------------------------------------
def save_model_blocks(model: nn.Module, subdir: str) -> None:
    """
    Save all model-declared blocks.

    Parameters
    ----------
    model : nn.Module
        Model whose declared blocks should be saved.
    subdir : str
        Target checkpoint subdirectory.

    Returns
    -------
    None

    """

    for block_name, block in get_named_model_blocks(model=model).items():
        atomic_save(obj=block.state_dict(), path=os.path.join(subdir, f"{block_name}.pt"))


# ----------------------------------------------------------------------------------------------------------------------
def load_model_blocks(
    model: nn.Module,
    subdir: str,
    *,
    map_location: Callable | torch.device | str | dict[str, str] | None = "cpu",
    strict: bool = True,
) -> None:
    """
    Load all model-declared blocks.

    Parameters
    ----------
    model : nn.Module
        Model whose declared blocks should be loaded.
    subdir : str
        Target checkpoint subdirectory.
    map_location : Callable | torch.device | str | dict[str, str] | None
        Same as ``map_location`` parameter of ``torch.load()``.
        Optional. Default: "cpu".
    strict : bool
        Whether to activate strict mode on each block's ``load_state_dict``.
        Optional. Default: True.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If checkpoint directory ``subdir`` is missing a required block file.

    """

    for block_name, block in get_named_model_blocks(model=model).items():
        block_path = os.path.join(subdir, f"{block_name}.pt")
        if not os.path.exists(block_path):
            raise FileNotFoundError(
                f"Checkpoint directory '{subdir}' missing required file: {os.path.basename(block_path)}."
            )

        block.load_state_dict(
            state_dict=torch_load(path=block_path, map_location=map_location),
            strict=strict,
        )
