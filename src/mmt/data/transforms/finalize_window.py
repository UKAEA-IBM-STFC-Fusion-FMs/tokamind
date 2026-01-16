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
    This transform intentionally removes fields that are *never* consumed by
    `MMTCollate` / the model once tokenization has completed:

    - raw groups: "input", "actuator" (large, redundant after embedding)
    - chunk structures: "chunks" (large, redundant after emb_chunks)
    - intermediate output embedding buffers: "embedded_output", "embedded_output_shapes"
    """

    def __init__(self, *, keep_output_native: bool) -> None:
        self.keep_output_native = bool(keep_output_native)

    def __call__(self, window: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if window is None:
            return None

        # Mutate in-place: the pipeline already mutates windows (e.g. BuildTokens).
        # This avoids extra dict allocations per window.
        window.pop("chunks", None)
        window.pop("embedded_output", None)
        window.pop("embedded_output_shapes", None)

        # Raw groups are never used after tokenization.
        window.pop("input", None)
        window.pop("actuator", None)

        # Native outputs are only needed for eval/traces.
        if not self.keep_output_native:
            window.pop("output", None)

        return window
