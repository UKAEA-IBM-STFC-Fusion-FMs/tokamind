"""
Model block contract: stable block names and the runtime accessor.

A "block" is a coarse, independently transferable unit of a model (token encoder, backbone, output adapters, ...).
Block names are part of the config and checkpoint contract:

  • config validation (``mmt.utils.config.validator``) uses ``MODEL_BLOCKS`` *before* model instantiation to check
    that each stage's ``optimizer.lr`` / ``optimizer.wd`` / ``freeze`` mappings name exactly the selected model's
    blocks;
  • each model exposes the *same* names at runtime through ``get_named_blocks()``, deriving membership and ordering
    from the constants below so the two layers can never drift;
  • checkpointing (``mmt.checkpoints.block_io``) saves one ``<block_name>.pt`` per block.

Keeping the names and the accessor together makes this the single place that defines "what blocks a model has".
"""

from __future__ import annotations

import torch.nn as nn


# ----------------------------------------------------------------------------------------------------------------------
# Stable block-name contract (shared by models and config validation)
# ----------------------------------------------------------------------------------------------------------------------

MMT_BLOCK_NAMES = ("token_encoder", "backbone", "modality_heads", "output_adapters")
MODEL_BLOCKS: dict[str, tuple[str, ...]] = {
    "mmt": MMT_BLOCK_NAMES,
}


# ----------------------------------------------------------------------------------------------------------------------
def get_named_model_blocks(model: nn.Module) -> dict[str, nn.Module]:
    """
    Return model-declared learnable blocks.

    Parameters
    ----------
    model : nn.Module
        Model exposing ``get_named_blocks()``.

    Returns
    -------
    dict[str, nn.Module]
        Mapping from stable block name to module.

    Raises
    ------
    AttributeError
        If ``model`` does not expose ``get_named_blocks``.
    TypeError
        If ``get_named_blocks`` returns an invalid mapping.

    """

    get_named_blocks = getattr(model, "get_named_blocks", None)
    if get_named_blocks is None:
        raise AttributeError("Model must expose get_named_blocks().")

    blocks = get_named_blocks()
    if not isinstance(blocks, dict) or not all(
        isinstance(k, str) and isinstance(v, nn.Module) for k, v in blocks.items()
    ):
        raise TypeError("model.get_named_blocks() must return dict[str, nn.Module].")

    return blocks
