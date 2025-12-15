"""
Save and load model parameter blocks.

This module handles checkpointing of the four learnable MMT model blocks:
token encoder, backbone, modality heads, and output adapters. It provides
both atomic saving and strict loading utilities used by resume and
evaluation workflows.
"""

from __future__ import annotations
import os

import torch.nn as nn

from .io import atomic_save, tload

# ======================================================================
# Save/load model blocks (FOUR BLOCKS)
# ======================================================================


def save_model_quadruplet(model: nn.Module, subdir: str) -> None:
    """
    Save the four learnable model blocks:
      • token_encoder
      • backbone
      • modality_heads
      • output_adapters
    """
    atomic_save(
        model.get_token_encoder_state_dict(),
        os.path.join(subdir, "token_encoder.pt"),
    )
    atomic_save(
        model.get_backbone_state_dict(),
        os.path.join(subdir, "backbone.pt"),
    )
    atomic_save(
        model.get_modality_heads_state_dict(),
        os.path.join(subdir, "modality_heads.pt"),
    )
    atomic_save(
        model.get_output_adapters_state_dict(),
        os.path.join(subdir, "output_adapters.pt"),
    )


def load_model_quadruplet(
    model: nn.Module,
    subdir: str,
    *,
    map_location="cpu",
    strict_token=True,
    strict_backbone=True,
    strict_heads=True,
    strict_adapters=True,
) -> None:
    """
    Strictly load all four model blocks (used for strict resume/eval).
    """
    fn_token = os.path.join(subdir, "token_encoder.pt")
    fn_backb = os.path.join(subdir, "backbone.pt")
    fn_heads = os.path.join(subdir, "modality_heads.pt")
    fn_adapt = os.path.join(subdir, "output_adapters.pt")

    for fn in (fn_token, fn_backb, fn_heads, fn_adapt):
        if not os.path.exists(fn):
            raise FileNotFoundError(
                f"Checkpoint directory '{subdir}' missing required file: {os.path.basename(fn)}"
            )

    model.load_token_encoder_state_dict(
        tload(fn_token, map_location), strict=strict_token
    )
    model.load_backbone_state_dict(
        tload(fn_backb, map_location), strict=strict_backbone
    )
    model.load_modality_heads_state_dict(
        tload(fn_heads, map_location), strict=strict_heads
    )
    model.load_output_adapters_state_dict(
        tload(fn_adapt, map_location), strict=strict_adapters
    )
