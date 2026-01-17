"""
FinalizeWindowTransform
=======================

Lightweight *end-of-pipeline* transform used to prune window dictionaries
before they are cached to RAM or passed to collation.

Rationale
---------
Most upstream transforms operate on a rich, nested window representation
(raw groups, chunk lists, per-chunk embeddings, etc.). After
`BuildTokensTransform`, the model/collate only needs the token fields and
output embeddings.

This transform centralizes the *policy* of what to keep:

- Always remove intermediate / heavy fields that are never used by the model.
- Optionally keep or drop native outputs for evaluation.

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
This transform does **not** touch the token contract produced by
`BuildTokensTransform` (e.g. emb_chunks/id/pos/mod/role/output_emb/...)
so it is safe to insert without changing collate/model.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class FinalizeWindowTransform:
    """Prune window dict to the minimal payload needed downstream.

    Parameters
    ----------
    keep_output_native:
        If True, keep the native output payload under ``window["output"]``.
        This is needed for evaluation metrics/traces that operate in native
        space.

        If False, drop ``window["output"]`` entirely to reduce memory.

    Notes
    -----
    This is a *contract enforcer*: by default it keeps only the fields that
    are required by collation/model (plus a tiny amount of debug metadata).
    This makes it robust to datasets that carry additional per-window keys.
    """

    def __init__(self, *, keep_output_native: bool) -> None:
        self.keep_output_native = bool(keep_output_native)

    def __call__(self, window: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if window is None:
            return None

        # ------------------------------------------------------------------
        # Keep-only policy
        # ------------------------------------------------------------------
        # Required by MMTCollate / model:
        keep_keys = {
            "emb_chunks",
            "pos",
            "id",
            "mod",
            "role",
            "output_emb",
        }

        # Small debug metadata (cheap, useful in logs/metrics).
        keep_keys.update({"shot_id", "window_index", "t_cut"})

        # Native outputs are only needed for eval/traces.
        if self.keep_output_native:
            keep_keys.add("output")

        # Mutate in-place to avoid extra dict allocations per window.
        for k in list(window.keys()):
            if k not in keep_keys:
                window.pop(k, None)

        return window
