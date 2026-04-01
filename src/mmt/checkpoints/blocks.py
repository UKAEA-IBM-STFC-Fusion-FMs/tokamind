"""
Save and load model parameter blocks.

This module handles checkpointing of the four learnable MMT model blocks: token encoder, backbone, modality heads, and
output adapters. It provides both atomic saving and strict loading utilities used by resume and evaluation workflows.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
import torch.nn as nn

from .io import atomic_save, torch_load


# ======================================================================================================================
# Save/load model blocks (FOUR BLOCKS)
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def save_model_quadruplet(model: nn.Module, subdir: str) -> None:
    """
    Save the four learnable model blocks:
      • token_encoder
      • backbone
      • modality_heads
      • output_adapters

    Parameters
    ----------
    model : nn.Module
        Input model for which all four model blocks will be loaded in strict mode by default.
    subdir : str
        Target checkpoint subdirectory.

    Returns
    -------
    None

    """

    atomic_save(
        obj=model.get_token_encoder_state_dict(),
        path=os.path.join(subdir, "token_encoder.pt"),
    )
    atomic_save(obj=model.get_backbone_state_dict(), path=os.path.join(subdir, "backbone.pt"))
    atomic_save(
        obj=model.get_modality_heads_state_dict(),
        path=os.path.join(subdir, "modality_heads.pt"),
    )
    atomic_save(
        obj=model.get_output_adapters_state_dict(),
        path=os.path.join(subdir, "output_adapters.pt"),
    )


# ----------------------------------------------------------------------------------------------------------------------
def load_model_quadruplet(
    model: nn.Module,
    subdir: str,
    *,
    map_location: Callable | torch.device | str | dict[str, str] | None = "cpu",
    strict_token: bool = True,
    strict_backbone: bool = True,
    strict_heads: bool = True,
    strict_adapters: bool = True,
) -> None:
    """
    Load all four model blocks in strict mode by default (used for strict resume/eval).

    Parameters
    ----------
    model : nn.Module
        Input model for which all four model blocks will be loaded in strict mode by default.
    subdir : str
        Target checkpoint subdirectory.
    map_location : Callable | torch.device | str | dict[str, str] | None
        Same as `map_location` parameter of `torch.load()`.
        Optional. Default: "cpu".
    strict_token : bool
        Whether to activate strict mode in `model.load_token_encoder_state_dict()`.
        Optional. Default: True.
    strict_backbone : bool
        Whether to activate strict mode in `model.load_backbone_state_dict()`.
        Optional. Default: True.
    strict_heads : bool
        Whether to activate strict mode in `model.load_modality_heads_state_dict()`.
        Optional. Default: True.
    strict_adapters : bool
        Whether to activate strict mode in `model.load_output_adapters_state_dict)`.
        Optional. Default: True.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If checkpoint directory 'subdir' missing a required file in ["token_encoder.pt", "backbone.pt",
        "modality_heads.pt", "output_adapters.pt"].

    """

    fn_token = os.path.join(subdir, "token_encoder.pt")
    fn_backb = os.path.join(subdir, "backbone.pt")
    fn_heads = os.path.join(subdir, "modality_heads.pt")
    fn_adapt = os.path.join(subdir, "output_adapters.pt")

    for fn in (fn_token, fn_backb, fn_heads, fn_adapt):
        if not os.path.exists(fn):
            raise FileNotFoundError(f"Checkpoint directory '{subdir}' missing required file: {os.path.basename(fn)}.")

    model.load_token_encoder_state_dict(
        state=torch_load(path=fn_token, map_location=map_location),
        strict=strict_token,
    )
    model.load_backbone_state_dict(
        state=torch_load(path=fn_backb, map_location=map_location),
        strict=strict_backbone,
    )
    model.load_modality_heads_state_dict(
        state=torch_load(path=fn_heads, map_location=map_location),
        strict=strict_heads,
    )
    model.load_output_adapters_state_dict(
        state=torch_load(path=fn_adapt, map_location=map_location),
        strict=strict_adapters,
    )
