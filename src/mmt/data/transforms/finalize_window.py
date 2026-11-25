"""
FinalizeWindowTransform
=======================

Lightweight *end-of-pipeline* transform used to prune window dictionaries before they are cached to RAM or passed to
collation.

Rationale
---------
Most upstream transforms operate on a rich, nested window representation (raw groups, chunk lists, per-chunk
embeddings, etc.). After `BuildTokensTransform`, the model/collate only needs the token fields and output embeddings.

This transform centralizes the *policy* of what to keep:

- Always remove intermediate / heavy fields that are never used by the model.
- Optionally keep or drop native outputs for evaluation/native-space losses.

Expected position in the chain (v0)
----------------------------------

    ChunkWindowsTransform
      → SelectValidWindowsTransform
        → TrimChunksTransform
          → EmbedChunksTransform
            → BuildTokensTransform
              → FinalizeWindowTransform   <-- HERE

Fields kept (train/eval)
------------------------
This transform does **not** touch the token contract produced by `BuildTokensTransform` (e.g.,
emb_chunks/ID/pos/mod/role/output_emb/...) so it is safe to insert without changing collate/model.
Output timestamps are retained separately from native targets because coordinate-aware models use them to place
future query cells even when training only in embedding space.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any


# ======================================================================================================================
class FinalizeWindowTransform:
    """Prune window dict to the minimal payload needed downstream.

    Attributes
    ----------
    keep_output_native : bool
        Whether to keep output native output payload.

    Methods
    -------
    __call__(window)
        Call method for the class instances to behave like a function.

    Notes
    -----
    This is a *contract enforcer*: by default it keeps only the fields that are required by collation/model (plus a
    tiny amount of debug metadata). This makes it robust to datasets that carry additional per-window keys.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        *,
        keep_output_native: bool,
        native_output_names: Collection[str] | None = None,
    ) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        keep_output_native : bool
            If True, keep the native output payload under `window["output"]`. This is needed for evaluation
            metrics/traces that operate in native space. If False, drop native values while retaining the small
            ``window["output_time"]`` mapping for coordinate-aware models.
        native_output_names : Collection[str] | None
            Optional output signal names to keep under `window["output"]` when `keep_output_native=True`. If None, keep
            all native outputs.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors, as this is a callable class.

        """

        self.keep_output_native = bool(keep_output_native)
        self.native_output_names = set(native_output_names) if native_output_names is not None else None

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, window: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Call method for the class instances to behave like a function.

        Parameters
        ----------
        window : dict[str, Any] | None
            Window on which the transform is applied.

        Returns
        -------
        dict[str, Any] | None
            Transformed (mutated in-place) window for a valid window, otherwise None.

        """

        if window is None:
            return None

        # ..............................................................................................................
        # Keep-only policy
        # ..............................................................................................................

        # Required by MMTCollate / model:
        keep_keys = {
            "emb_chunks",
            "pos",
            "id",
            "mod",
            "role",
            "token_time",
            "output_emb",
        }

        # Optional persistence baseline used by output-residual models. The source-ID mapping lets collation disable
        # the bypass when the corresponding newest input token is dropped.
        if "output_baseline_emb" in window:
            keep_keys.update({"output_baseline_emb", "output_baseline_source_id"})

        # Small debug metadata (cheap, useful in logs/metrics).
        keep_keys.update({"shot_id", "window_index", "t_cut"})

        # Output query coordinates are needed independently of native-space loss/evaluation. Preserve only the
        # timestamp vectors; native values remain subject to the keep_output_native policy below.
        output = window.get("output")
        if isinstance(output, dict):
            output_time = {
                str(name): payload["time"]
                for name, payload in output.items()
                if isinstance(payload, dict) and payload.get("time") is not None
            }
            if output_time:
                window["output_time"] = output_time
                keep_keys.add("output_time")

        # Optional physical coordinates used by grid-native models.  The key is
        # retained only when the source dataset supplied it.
        if window.get("space_grid") is not None:
            keep_keys.add("space_grid")

        # Native outputs are only needed for eval/traces or selected native-space loss targets.
        if self.keep_output_native:
            if self.native_output_names is not None:
                output = window.get("output")
                if isinstance(output, dict):
                    window["output"] = {
                        name: value for name, value in output.items() if str(name) in self.native_output_names
                    }
            keep_keys.add("output")

        # Mutate in-place to avoid extra dict allocations per window.
        for k in list(window.keys()):  # NOSONAR - Ignore weak warning
            if k not in keep_keys:
                window.pop(k, None)

        return window
