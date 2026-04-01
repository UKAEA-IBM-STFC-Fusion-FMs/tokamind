"""
Warm-start (overlap) loading utilities for MMT checkpoints.

This module implements *partial checkpoint loading* used to initialize a model from a previous run when the
architecture is compatible but not identical (e.g., new or removed signals).

Key behavior
------------
- Loads parameters only when *both key and tensor shape match*.
- Never overwrites mismatched or missing parameters (they remain randomly initialized).
- Does NOT restore optimizer, scheduler, RNG, or training state.
- Prefers checkpoints/best over checkpoints/latest when both exist.

In addition to loading, the module provides detailed component-level logging (reused / initialized / incompatible /
removed) for token projections and output adapters, making warm-start behavior explicit and auditable.

This functionality is intended for pretraining → finetuning or cross-task initialization, not for strict training
resume.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Callable, Sequence
from typing import Any, Optional

import torch
import torch.nn as nn

from .io import torch_load, best_or_latest_dir
import logging


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.WarmStart")


# ======================================================================================================================
# Warm-start overlap loading
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def _filter_overlap_state(loaded: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    """
    Filter items in input loaded state.

    Parameters
    ----------
    loaded : Mapping[str, Any]
        Input loaded state mapping to be filtered.
    current : current: Mapping[str, Any]
        Reference mapping for filtering. Items in `loaded` but not in `current`, items in `loaded` with different shape
        in `current`, and items in `loaded` with different type in `current`, will be filtered out.

    Returns
    -------
    dict[str, Any]
        Dictionary with filtered loaded items.

    """

    out = {}
    for k, v in loaded.items():
        if k not in current:
            continue
        if not (isinstance(v, torch.Tensor) and isinstance(current[k], torch.Tensor)):
            continue
        if v.shape != current[k].shape:
            continue
        out[k] = v

    return out


# ----------------------------------------------------------------------------------------------------------------------
def _format_name_list(names: Sequence[str] | set[str] | list[str], *, max_items: int = 20) -> str:
    """
    Format a possibly-long list for logs.

    Parameters
    ----------
    names : Sequence[str] | set[str] | list[str]
        A sequence of names to be formated as string.
    max_items : int
        Maximum number of items for formatting.

    Returns
    -------
    str
        Formatted list in str format for logs.

    """

    names = sorted(set(names))
    if len(names) <= max_items:
        return "[" + ", ".join(names) + "]"
    head = names[:max_items]

    return "[" + ", ".join(head) + f", ... (+{len(names) - max_items} more)]"


# ----------------------------------------------------------------------------------------------------------------------
def _extract_token_proj_component(key: str) -> Optional[str]:
    """
    Extract TokenEncoder per-signal projection component from a state_dict key.

    Expected patterns include:
      - "proj_layers.<something>.<param>"
    Where <something> is often like:
      - "input:pf_active-coil_current"
      - "output:pf_active-solenoid_current"

    Parameters
    ----------
    key : str
        Input state_dict key used for extraction of token projection component.

    Returns
    -------
    Optional[str]
        Extracted token projection component in str format, or None if not possible.

    """

    if not key.startswith("proj_layers."):
        return None

    rest = key[len("proj_layers.") :]  # E.g., "proj_layers.output:pf_active-coil_current.weight"
    comp = rest.split(sep=".", maxsplit=1)[0]  # "output:pf_active-coil_current"

    return comp or None


# ----------------------------------------------------------------------------------------------------------------------
def _extract_output_adapter_component(key: str) -> Optional[str]:
    """
    Extract output adapter name from a key in the *output_adapters* state_dict.

    Note: model.get_output_adapters_state_dict() is typically scoped, so keys look like:
      - "<adapter_key>.weight"
      - "<adapter_key>.bias"

    Parameters
    ----------
    key : str
        Input state_dict key used for extraction of output adapter name.

    Returns
    -------
    Optional[str]
        Extracted output adapter name in str format, or None if not possible.

    """

    if not key:
        return None

    return key.split(sep=".", maxsplit=1)[0]  # "<adapter_key>"


# ----------------------------------------------------------------------------------------------------------------------
def _component_sets(
    loaded_sd: Mapping[str, Any], current_sd: Mapping[str, Any], *, extractor: Callable
) -> dict[str, set[str]]:
    """
    Compute component-level categories (reused / initialized / incompatible / removed) based on state_dict keys and
    tensor shapes.

    Definitions
    -----------
    reused:
        Component exists in both checkpoint and current model AND all common tensor parameters match in shape (i.e., no
        shape mismatches for that component).
        (Example: Linear weight+bias both match.)

    incompatible:
        Component exists in both, but at least one common tensor parameter has a shape mismatch (even if others match).
        This avoids reporting "reused" when only bias matches but weight does not.

    initialized:
        Component exists only in current model (not in checkpoint).

    removed:
        Component exists only in checkpoint (not in current model).

    Parameters
    ----------
    loaded_sd : Mapping[str, Any]
        Loaded state_dict mapping.
    current_sd : Mapping[str, Any]
        Current state_dict mapping.
    extractor : Callable
        Target function for component extraction.

    Returns
    -------
    dict[str, set[str]]
        Dictionary with resulting components in ["reused", "initialized", "incompatible", "removed"] categories and
        "shape_mismatch" information.

    """

    loaded_keys = [k for k in loaded_sd.keys() if extractor(k) is not None]
    current_keys = [k for k in current_sd.keys() if extractor(k) is not None]

    loaded_comps = {extractor(k) for k in loaded_keys}
    current_comps = {extractor(k) for k in current_keys}

    present_in_both = loaded_comps & current_comps
    removed_comps = loaded_comps - current_comps
    initialized_comps = current_comps - loaded_comps

    common_keys = set(loaded_keys) & set(current_keys)

    # Track, per component, whether we saw any matching/mismatching tensor params.
    comp_has_match: dict[str, bool] = dict.fromkeys(present_in_both, False)
    comp_has_mismatch: dict[str, bool] = dict.fromkeys(present_in_both, False)

    for k in common_keys:
        comp = extractor(k)
        if (comp is None) or (comp not in present_in_both):
            continue

        v_old = loaded_sd.get(k)
        v_new = current_sd.get(k)
        if not (isinstance(v_old, torch.Tensor) and isinstance(v_new, torch.Tensor)):
            continue

        if v_old.shape == v_new.shape:
            comp_has_match[comp] = True
        else:
            comp_has_mismatch[comp] = True

    # Components with any mismatch are incompatible, even if some params match.
    incompatible_comps = {c for c in present_in_both if comp_has_mismatch.get(c, False)}

    # Reused means: at least one param matched AND no mismatches.
    reused_comps = {
        c for c in present_in_both if comp_has_match.get(c, False) and (not comp_has_mismatch.get(c, False))
    }

    # For debugging / optional logs
    shape_mismatch = incompatible_comps.copy()

    return {
        "reused": reused_comps,
        "initialized": initialized_comps,
        "incompatible": incompatible_comps,
        "removed": removed_comps,
        "shape_mismatch": shape_mismatch,
    }


# ----------------------------------------------------------------------------------------------------------------------
def load_parts_from_run_dir(  # NOSONAR - Ignore cognitive complexity
    model: nn.Module,
    run_dir: str,
    *,
    load_parts: Optional[Mapping[str, bool]] = None,
    map_location: Callable | torch.device | str | dict[str, str] | None = "cpu",
) -> None:
    """
    Overlap-load selected parts of `model` from a previous run_dir.

    This function is meant for *initialising a new run from pretraining*, not for strict resume.
    Optimizer/scheduler/scaler/RNG are NOT touched.

    It looks for either:
        run_dir/checkpoints/best/
    or
        run_dir/checkpoints/latest/
    (prefers best if it exists).

    Parameters
    ----------
    model : nn.Module
        MultiModalTransformer instance exposing:
          - get_backbone_state_dict()
          - get_modality_heads_state_dict()
          - get_output_adapters_state_dict()
          - load_backbone_state_dict(sd, strict=False)
          - load_modality_heads_state_dict(sd, strict=False)
          - load_output_adapters_state_dict(sd, strict=False)

    run_dir : str
        Path to a *previous* run directory.

    load_parts : Optional[Mapping[str, bool]]
        Mapping (dict) with optional boolean flags:
            {
              "backbone": True/False,
              "modality_heads": True/False,
              "output_adapters": True/False
            }
        Optional. Default: None.

        If None, defaults to loading all three with overlap:
            {"backbone": True, "modality_heads": True, "output_adapters": True}

        For each part with True:
          - loads its state_dict from checkpoint
          - intersects with current state_dict (key+shape overlap)
          - loads that filtered dict with strict=False

    map_location : Callable | torch.device | str | dict[str, str] | None
        Device to map tensors to when loading. Same as `map_location` parameter of `torch.load()`.

    Raises
    ------
    FileNotFoundError
        If no 'checkpoints/best' or 'checkpoints/latest' directory is found under run_dir, or if required .pt files are
        missing.

    """

    # ..................................................................................................................
    def _count(sd: Mapping[str, Any] | torch.Tensor) -> int:
        """Count number of elements in values of input state_dict `sd`."""
        return sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))

    # ..................................................................................................................
    def _load(blk: str, get_fn: Callable, load_fn: Callable, filename: str) -> None:
        """
        Supporting function to load model blocks.

        Parameters
        ----------
        blk : str
            Block name to be loaded.
        get_fn : Callable
            Get method to be used during block loading.
        load_fn : Callable
            Load method to be used during block loading.
        filename : str
            Target checkpoint filename for the block to be loaded.

        Returns
        -------
        None

        """

        if not load_parts.get(blk, False):
            return

        path = os.path.join(ckpt, filename)  # type: ignore[arg-type]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing '{filename}' in checkpoint {ckpt}.")

        loaded_sd_ = torch_load(path=path, map_location=map_location)  # type: ignore[arg-type]
        current_sd_ = get_fn()

        overlap_sd_ = _filter_overlap_state(loaded=loaded_sd_, current=current_sd_)
        if overlap_sd_:
            load_fn(overlap_sd_, strict=False)

        stats[blk] = (_count(sd=overlap_sd_), _count(sd=current_sd_))
        _debug_sds[blk] = (loaded_sd_, current_sd_)

    # ..................................................................................................................

    stats: dict[str, tuple[int, int]] = {}

    # Store state_dicts for component-level reporting (only for blocks we load)
    _debug_sds: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    ckpt = best_or_latest_dir(run_dir=run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoints/best or checkpoints/latest found under '{run_dir}'.")

    if load_parts is None:
        load_parts = {
            "token_encoder": True,
            "backbone": True,
            "modality_heads": True,
            "output_adapters": True,
        }

    # ..................................................................................................................
    # Load four blocks
    # ..................................................................................................................

    _load(
        blk="token_encoder",
        get_fn=model.get_token_encoder_state_dict,  # type: ignore[arg-type]
        load_fn=model.load_token_encoder_state_dict,  # type: ignore[arg-type]
        filename="token_encoder.pt",
    )
    _load(
        blk="backbone",
        get_fn=model.get_backbone_state_dict,  # type: ignore[arg-type]
        load_fn=model.load_backbone_state_dict,  # type: ignore[arg-type]
        filename="backbone.pt",
    )
    _load(
        blk="modality_heads",
        get_fn=model.get_modality_heads_state_dict,  # type: ignore[arg-type]
        load_fn=model.load_modality_heads_state_dict,  # type: ignore[arg-type]
        filename="modality_heads.pt",
    )
    _load(
        blk="output_adapters",
        get_fn=model.get_output_adapters_state_dict,  # type: ignore[arg-type]
        load_fn=model.load_output_adapters_state_dict,  # type: ignore[arg-type]
        filename="output_adapters.pt",
    )

    # ..................................................................................................................
    # Summary: block-level param overlap (keep your existing style)
    # ..................................................................................................................

    summary = []
    for block in ("token_encoder", "backbone", "modality_heads", "output_adapters"):
        if load_parts.get(block, False):
            if block in stats:
                L, T = stats[block]  # noqa - Ignore lowercase warning
                summary.append(f"{block}: {L}/{T} params matched")
            else:
                summary.append(f"{block}: loaded (no overlapping params found)")
        else:
            summary.append(f"{block}: skipped (load_parts=False)")

    logger.info("")
    logger.info(f"Loaded from {ckpt}: " + " | ".join(summary))

    # ..................................................................................................................
    # Detailed warm-start report: token encoder projections
    # ..................................................................................................................

    if load_parts.get("token_encoder", False) and ("token_encoder" in _debug_sds):
        loaded_sd, current_sd = _debug_sds["token_encoder"]
        rep = _component_sets(
            loaded_sd=loaded_sd,
            current_sd=current_sd,
            extractor=_extract_token_proj_component,
        )

        logger.info("")
        logger.info("Warm-start detail [token_encoder.proj_layers]")
        logger.info(
            "  reused=%d | initialized=%d | incompatible=%d | removed=%d",
            len(rep["reused"]),
            len(rep["initialized"]),
            len(rep["incompatible"]),
            len(rep["removed"]),
        )

        if rep["reused"]:
            logger.info("  reused: %s", _format_name_list(names=rep["reused"]))
        if rep["initialized"]:
            logger.info(
                "  initialized (new in current): %s",
                _format_name_list(names=rep["initialized"]),
            )
        if rep["incompatible"]:
            logger.info(
                "  incompatible (present but not reusable): %s",
                _format_name_list(names=rep["incompatible"]),
            )
        if rep["removed"]:
            logger.info(
                "  removed (present in checkpoint only): %s",
                _format_name_list(names=rep["removed"]),
            )

    # ..................................................................................................................
    # Detailed warm-start report: output adapters
    # ..................................................................................................................

    if load_parts.get("output_adapters", False) and ("output_adapters" in _debug_sds):
        loaded_sd, current_sd = _debug_sds["output_adapters"]
        rep = _component_sets(
            loaded_sd=loaded_sd,
            current_sd=current_sd,
            extractor=_extract_output_adapter_component,
        )

        logger.info("")
        logger.info("Warm-start detail [output_adapters]")
        logger.info(
            "  reused=%d | initialized=%d | incompatible=%d | removed=%d",
            len(rep["reused"]),
            len(rep["initialized"]),
            len(rep["incompatible"]),
            len(rep["removed"]),
        )

        if rep["reused"]:
            logger.info("  reused: %s", _format_name_list(names=rep["reused"]))
        if rep["initialized"]:
            logger.info(
                "  initialized (new in current): %s",
                _format_name_list(names=rep["initialized"]),
            )
        if rep["incompatible"]:
            logger.info(
                "  incompatible (present but not reusable): %s",
                _format_name_list(names=rep["incompatible"]),
            )
        if rep["removed"]:
            logger.info(
                "  removed (present in checkpoint only): %s",
                _format_name_list(names=rep["removed"]),
            )

        logger.info("")
