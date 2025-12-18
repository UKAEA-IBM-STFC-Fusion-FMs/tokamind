"""
BuildTokensTransform
====================

Convert embedded chunks into the final per-token representation consumed by
the MMT model and collate.

Contract (must match downstream collate/train)
----------------------------------------------
This transform writes the following fields to `window`:

    window["emb_chunks"]   = emb_list                       (List[np.ndarray])
    window["pos"]          = pos                            (np.ndarray int32)
    window["id"]           = np.asarray(sig_list, int32)    (np.ndarray)
    window["signal_name"]  = np.asarray(name_list, object)  (np.ndarray)
    window["mod"]          = np.asarray(mod_list, int16)    (np.ndarray)
    window["role"]         = np.asarray(role_list, int8)    (np.ndarray)

    window["output_emb"]   = output_emb                     (Dict[int, np.ndarray])
    window["output_shapes"]= output_shapes                  (Dict[int, tuple])
    window["output_names"] = output_names                   (Dict[int, str])

Simplified logic (v0)
--------------------
- Positions are NOT computed from time here.
- Each chunk is expected to carry an integer `pos` computed upstream by
  `TrimChunksTransform`.
- Deterministic ordering is done by (pos, chunk_index_in_window), then signal_id.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import logging

from mmt.data.signal_spec import SignalSpecRegistry
from mmt.constants import (
    ROLE_CONTEXT,
    ROLE_ACTUATOR,
)

logger = logging.getLogger("mmt.BuildTokens")


class BuildTokensTransform:
    """
    Construct a deterministic, aligned token sequence for a window.

    Input requirements
    ------------------
    Each chunk in window["chunks"][role] must contain:
      - "pos"                 : int (computed upstream)
      - "chunk_index_in_window": int (deterministic tie-breaker)
      - "embeddings"          : Dict[int, np.ndarray] mapping signal_id → embedding

    window must contain:
      - "embedded_output"        : Dict[int, np.ndarray]
      - "embedded_output_shapes" : Dict[int, tuple]

    Output
    ------
    Writes the exact arrays/dicts expected by collate/train (see module header).
    """

    def __init__(self, signal_specs: SignalSpecRegistry) -> None:
        self.signal_specs = signal_specs

        # Stable modality_id table
        modalities = signal_specs.modalities  # sorted by registry
        self._modality_to_id = {m: i for i, m in enumerate(modalities)}

        # signal_id -> modality_id
        self._signal_id_to_mod_id: Dict[int, int] = {}
        for spec in signal_specs.specs:
            if spec.modality not in self._modality_to_id:
                raise KeyError(f"Unknown modality {spec.modality!r}")
            self._signal_id_to_mod_id[spec.signal_id] = self._modality_to_id[
                spec.modality
            ]

    # ------------------------------------------------------------------ main API

    def __call__(self, window: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
        name_list: List[str] = []
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
                emb = emb_map[sig_id]
                spec = self.signal_specs.get_by_id(sig_id)

                emb_list.append(emb)
                sig_list.append(int(sig_id))
                name_list.append(spec.name)
                role_list.append(ROLE_CONTEXT)
                mod_list.append(int(self._signal_id_to_mod_id[sig_id]))
                pos_list.append(pos)

        # -------------------------
        # 2) ACTUATOR TOKENS
        # -------------------------
        for ch in act_chunks_sorted:
            pos = int(ch["pos"])
            emb_map = ch.get("embeddings") or {}

            for sig_id in sorted(emb_map.keys()):
                emb = emb_map[sig_id]
                spec = self.signal_specs.get_by_id(sig_id)

                emb_list.append(emb)
                sig_list.append(int(sig_id))
                name_list.append(spec.name)
                role_list.append(ROLE_ACTUATOR)
                mod_list.append(int(self._signal_id_to_mod_id[sig_id]))
                pos_list.append(pos)

        pos = np.asarray(pos_list, dtype=np.int32)

        # -------------------------
        # 3) OUTPUTS (window-level)
        # -------------------------
        output_emb: Dict[int, np.ndarray] = {}
        output_shapes: Dict[int, Any] = {}
        output_names: Dict[int, str] = {}

        embedded_output = window.get("embedded_output") or {}
        output_shapes_all = window.get("embedded_output_shapes") or {}

        for sig_id, emb in embedded_output.items():
            spec = self.signal_specs.get_by_id(int(sig_id))
            output_emb[int(sig_id)] = emb
            output_names[int(sig_id)] = spec.name

            if int(sig_id) not in output_shapes_all:
                raise KeyError(f"Missing output shape for signal_id={sig_id}")
            output_shapes[int(sig_id)] = output_shapes_all[int(sig_id)]

        # -------------------------
        # 4) WRITE BACK (contract)
        # -------------------------
        window["emb_chunks"] = emb_list
        window["pos"] = pos
        window["id"] = np.asarray(sig_list, dtype=np.int32)
        window["signal_name"] = np.asarray(name_list, dtype=object)
        window["mod"] = np.asarray(mod_list, dtype=np.int16)
        window["role"] = np.asarray(role_list, dtype=np.int8)

        window["output_emb"] = output_emb
        window["output_shapes"] = output_shapes
        window["output_names"] = output_names

        logger.debug(
            "win %s (shot %s) | tokens=%d (context=%d, act=%d) | pos=[%s..%s]",
            window.get("window_index"),
            window.get("shot_id"),
            len(emb_list),
            sum(r == ROLE_CONTEXT for r in role_list),
            sum(r == ROLE_ACTUATOR for r in role_list),
            int(pos.min()) if pos.size else None,
            int(pos.max()) if pos.size else None,
        )

        return window
