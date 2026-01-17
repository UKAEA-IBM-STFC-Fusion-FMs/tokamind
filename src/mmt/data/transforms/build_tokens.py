"""
BuildTokensTransform
====================

Convert embedded chunks into the final per-token representation consumed by
MMT collation and the model.

Contract (must match downstream collate/train)
----------------------------------------------
This transform writes the following fields to ``window``:

    window["emb_chunks"]   : List[np.ndarray]
        Ragged list of token embeddings (one array per token).

    window["pos"]          : np.ndarray[int32] shape (L,)
        Relative token positions (1 = closest-to-output).

    window["id"]           : np.ndarray[int32] shape (L,)
        Token signal IDs (SignalSpec.signal_id).

    window["mod"]          : np.ndarray[int16] shape (L,)
        Modality IDs (small ints, stable within a registry).

    window["role"]         : np.ndarray[int8] shape (L,)
        Token roles (ROLE_CONTEXT / ROLE_ACTUATOR).

    window["output_emb"]   : Dict[int, np.ndarray]
        Window-level output embeddings keyed by output signal_id.

Notes (v0)
----------
- We **do not store per-token signal names** in the window anymore.
  Name-based dropout overrides are converted to id-based once at startup.
- We **do not store per-window output_names/output_shapes** in the window.
  The output signal_id is sufficient for training; eval code can map id→name
  using the SignalSpecRegistry.

Simplified logic (v0)
--------------------
- Positions are NOT computed from time here.
- Each chunk is expected to carry an integer ``pos`` computed upstream by
  ``TrimChunksTransform``.
- Deterministic ordering is done by (pos, chunk_index_in_window), then signal_id.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

import numpy as np

from mmt.data.signal_spec import SignalSpecRegistry
from mmt.constants import ROLE_CONTEXT, ROLE_ACTUATOR

logger = logging.getLogger("mmt.BuildTokens")


class BuildTokensTransform:
    """Construct a deterministic, aligned token sequence for a window."""

    def __init__(self, signal_specs: SignalSpecRegistry) -> None:
        self.signal_specs = signal_specs

        # Stable modality_id table (sorted for determinism).
        modalities = signal_specs.modalities
        self._modality_to_id = {m: i for i, m in enumerate(modalities)}

        # signal_id -> modality_id
        self._signal_id_to_mod_id: Dict[int, int] = {}
        for spec in signal_specs.specs:
            if spec.modality not in self._modality_to_id:
                raise KeyError(f"Unknown modality {spec.modality!r}")
            self._signal_id_to_mod_id[int(spec.signal_id)] = int(
                self._modality_to_id[spec.modality]
            )

    def __call__(self, window: Dict[str, Any] | None) -> Optional[Dict[str, Any]]:
        if window is None:
            return None

        chunks_dict = window.get("chunks") or {}
        input_chunks = chunks_dict.get("input", []) or []
        act_chunks = chunks_dict.get("actuator", []) or []

        def _chunk_sort_key(ch: Dict[str, Any]) -> tuple[int, int]:
            if "pos" not in ch:
                raise KeyError(
                    "BuildTokensTransform requires chunk['pos'] (computed upstream)"
                )
            if "chunk_index_in_window" not in ch:
                raise KeyError(
                    "BuildTokensTransform requires chunk['chunk_index_in_window']"
                )
            return (int(ch["pos"]), int(ch["chunk_index_in_window"]))

        # Deterministic ordering:
        # - chunks sorted by (pos, chunk_index_in_window) -> closest-to-output first
        # - signals within chunk sorted by signal_id
        input_chunks_sorted = sorted(input_chunks, key=_chunk_sort_key)
        act_chunks_sorted = sorted(act_chunks, key=_chunk_sort_key)

        emb_list: List[np.ndarray] = []
        sig_list: List[int] = []
        role_list: List[int] = []
        mod_list: List[int] = []
        pos_list: List[int] = []

        # -------------------------
        # 1) CONTEXT TOKENS (INPUT)
        # -------------------------
        for ch in input_chunks_sorted:
            pos = int(ch["pos"])
            emb_map = ch.get("embeddings") or {}

            for sig_id in sorted(emb_map.keys()):
                sid = int(sig_id)
                if sid not in self._signal_id_to_mod_id:
                    raise KeyError(f"Unknown signal_id={sid} (missing from registry)")

                emb_list.append(emb_map[sig_id])
                sig_list.append(sid)
                role_list.append(ROLE_CONTEXT)
                mod_list.append(self._signal_id_to_mod_id[sid])
                pos_list.append(pos)

        # -------------------------
        # 2) ACTUATOR TOKENS
        # -------------------------
        for ch in act_chunks_sorted:
            pos = int(ch["pos"])
            emb_map = ch.get("embeddings") or {}

            for sig_id in sorted(emb_map.keys()):
                sid = int(sig_id)
                if sid not in self._signal_id_to_mod_id:
                    raise KeyError(f"Unknown signal_id={sid} (missing from registry)")

                emb_list.append(emb_map[sig_id])
                sig_list.append(sid)
                role_list.append(ROLE_ACTUATOR)
                mod_list.append(self._signal_id_to_mod_id[sid])
                pos_list.append(pos)

        # -------------------------
        # 3) OUTPUTS (window-level)
        # -------------------------
        embedded_output = window.get("embedded_output") or {}
        output_emb: Dict[int, np.ndarray] = {
            int(k): v for k, v in embedded_output.items()
        }

        # -------------------------
        # 4) WRITE BACK (contract)
        # -------------------------
        window["emb_chunks"] = emb_list
        window["pos"] = np.asarray(pos_list, dtype=np.int32)
        window["id"] = np.asarray(sig_list, dtype=np.int32)
        window["mod"] = np.asarray(mod_list, dtype=np.int16)
        window["role"] = np.asarray(role_list, dtype=np.int8)
        window["output_emb"] = output_emb

        logger.debug(
            "win %s (shot %s) | tokens=%d (context=%d, act=%d) | pos=[%s..%s]",
            window.get("window_index"),
            window.get("shot_id"),
            len(emb_list),
            sum(r == ROLE_CONTEXT for r in role_list),
            sum(r == ROLE_ACTUATOR for r in role_list),
            int(window["pos"].min()) if window["pos"].size else None,
            int(window["pos"].max()) if window["pos"].size else None,
        )

        return window
