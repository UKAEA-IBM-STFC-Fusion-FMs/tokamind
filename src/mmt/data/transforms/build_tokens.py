"""
BuildTokensTransform
====================

This module converts the output of EmbedChunksTransform into the
final per-token representation consumed by the MMT model and collate.

This updated implementation is **deterministic**: it enforces a stable
ordering of tokens across all windows, shots, and workers, eliminating
the misalignment between token embeddings and token metadata that caused
dimension mismatches inside TokenEncoder.

Main features:
--------------
• Deterministic ordering of chunks and signals within chunks
• Embeddings (`emb_chunks`) and metadata (`id`, `pos`, `mod`, `role`, `name`)
  are always aligned index-by-index
• No assumptions about upstream dict ordering
• No accidental reordering due to signal masking / cache behaviour
• Context (input) tokens come first, then actuator tokens
• Output embeddings are untouched (still handled as window-level vectors)

The transform **does not truncate** history or modify chunk structure:
it only builds a flattened token sequence and computes their temporal
positions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging
import numpy as np

from mmt.data.signal_spec import SignalSpecRegistry

logger = logging.getLogger("mmt.BuildTokens")

# Token roles (shared across transforms and model)
ROLE_CONTEXT = 0
ROLE_ACTUATOR = 1
ROLE_OUTPUT = 2  # for the learned Y-token (not created here)


class BuildTokensTransform:
    """
    Construct a deterministic, aligned token sequence for a window.

    Purpose
    -------
    Convert the embedded chunks from `EmbedChunksTransform` into a list of
    token embeddings (`emb_chunks`) and aligned metadata arrays:
        id, pos, mod, role, signal_name.

    This transform ensures **stable ordering**, which is essential for
    consistency across windows. All nondeterministic elements from Python
    dictionaries (signal masking order, iteration order, cache effect, etc.)
    are removed.

    Input (per window)
    ------------------
    A window dict containing:
        "t_cut": float
        "chunks": {
            "input":    [chunk_dict, ...],
            "actuator": [chunk_dict, ...],
        }
        Each chunk_dict contains:
            "chunk_start_time" : float
            "embeddings"       : { signal_id: np.ndarray(D,) }
            "orig_shapes"      : { signal_id: tuple(...) }

        window["embedded_output"]        : { signal_id: np.ndarray(D_out,) }
        window["embedded_output_shapes"] : { signal_id: orig_shape }

    Output fields added to the window
    ---------------------------------
        window["emb_chunks"]   : List[np.ndarray]
        window["pos"]          : np.ndarray(L,)
        window["id"]           : np.ndarray(L,)
        window["mod"]          : np.ndarray(L,)
        window["role"]         : np.ndarray(L,)
        window["signal_name"]  : np.ndarray(L,)

        window["outputs_emb"]
        window["outputs_shapes"]
        window["outputs_names"]

    Deterministic ordering rules
    ----------------------------
    1. Context (input) tokens first, then actuator tokens
    2. Within each role:
        • chunks sorted by chunk_start_time
        • signals within each chunk sorted by signal_id
    3. No dependence on Python dict insertion order
    """

    def __init__(
        self,
        chunk_length_sec: float,
        delta: float,
        output_length: float,
        signal_specs: SignalSpecRegistry,
    ) -> None:
        self.chunk_length_sec = float(chunk_length_sec)
        self.delta = float(delta)
        self.output_length = float(output_length)
        self.signal_specs = signal_specs

        # Build modality_id table
        modalities = signal_specs.modalities  # sorted by registry
        self._modality_to_id = {m: i for i, m in enumerate(modalities)}

        # Map physical signal_id → modality_id
        self._signal_id_to_mod_id: Dict[int, int] = {}
        for spec in signal_specs.specs:
            if spec.modality not in self._modality_to_id:
                raise KeyError(f"Unknown modality {spec.modality!r}")
            self._signal_id_to_mod_id[spec.signal_id] = self._modality_to_id[
                spec.modality
            ]

    # ------------------------------------------------------------------ helpers

    def _compute_pos_ids(self, t_cut: float, chunk_times: List[float]) -> np.ndarray:
        """
        Compute temporal positions for tokens.

        pos = number of chunk-length steps before the output window end,
              starting from 1 (pos=1 means the chunk ends exactly at output end).
        """
        t_out_end = t_cut + self.delta + self.output_length
        L = self.chunk_length_sec

        pos_ids = []
        for t_start in chunk_times:
            chunk_end = t_start + L
            steps = round((t_out_end - chunk_end) / L)
            pos_ids.append(steps + 1)

        return np.asarray(pos_ids, dtype=np.int32)

    # ------------------------------------------------------------------ main API

    def __call__(self, window: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if window is None:
            return None
        if "t_cut" not in window or window["t_cut"] is None:
            raise ValueError("BuildTokensTransform: missing t_cut")

        t_cut = float(window["t_cut"])

        input_chunks = (window.get("chunks", {}) or {}).get("input", []) or []
        act_chunks = (window.get("chunks", {}) or {}).get("actuator", []) or []

        # Sort chunks deterministically
        input_chunks_sorted = sorted(
            input_chunks, key=lambda in_ch: float(in_ch["chunk_start_time"])
        )
        act_chunks_sorted = sorted(
            act_chunks, key=lambda act_ch: float(act_ch["chunk_start_time"])
        )

        emb_list: List[np.ndarray] = []
        sig_list: List[int] = []
        name_list: List[str] = []
        role_list: List[int] = []
        mod_list: List[int] = []
        t_list: List[float] = []

        # ------------------------------------------------------------------
        # 1) CONTEXT TOKENS (INPUT)
        # ------------------------------------------------------------------
        for ch in input_chunks_sorted:
            t_start = float(ch["chunk_start_time"])
            emb_map = ch.get("embeddings") or {}

            # Deterministic ordering: sorted by signal_id
            for sig_id in sorted(emb_map.keys()):
                emb = emb_map[sig_id]
                spec = self.signal_specs.get_by_id(sig_id)

                emb_list.append(emb)
                sig_list.append(sig_id)
                name_list.append(spec.name)
                role_list.append(ROLE_CONTEXT)
                mod_list.append(self._signal_id_to_mod_id[sig_id])
                t_list.append(t_start)

        # ------------------------------------------------------------------
        # 2) ACTUATOR TOKENS
        # ------------------------------------------------------------------
        for ch in act_chunks_sorted:
            t_start = float(ch["chunk_start_time"])
            emb_map = ch.get("embeddings") or {}

            for sig_id in sorted(emb_map.keys()):
                emb = emb_map[sig_id]
                spec = self.signal_specs.get_by_id(sig_id)

                emb_list.append(emb)
                sig_list.append(sig_id)
                name_list.append(spec.name)
                role_list.append(ROLE_ACTUATOR)
                mod_list.append(self._signal_id_to_mod_id[sig_id])
                t_list.append(t_start)

        # ------------------------------------------------------------------
        # 3) POSITIONS (aligned with embedding order)
        # ------------------------------------------------------------------
        pos = self._compute_pos_ids(t_cut, t_list)

        # ------------------------------------------------------------------
        # 4) OUTPUTS (unchanged)
        # ------------------------------------------------------------------
        outputs_emb = {}
        outputs_shapes = {}
        outputs_names = {}

        embedded_output = window.get("embedded_output") or {}
        output_shapes_all = window.get("embedded_output_shapes") or {}

        for sig_id, emb in embedded_output.items():
            spec = self.signal_specs.get_by_id(sig_id)
            outputs_emb[sig_id] = emb
            outputs_names[sig_id] = spec.name

            if sig_id in output_shapes_all:
                outputs_shapes[sig_id] = output_shapes_all[sig_id]
            else:
                raise KeyError(f"Missing output shape for signal_id={sig_id}")

        # ------------------------------------------------------------------
        # 5) Write back into window
        # ------------------------------------------------------------------
        window["emb_chunks"] = emb_list
        window["pos"] = pos
        window["id"] = np.asarray(sig_list, dtype=np.int32)
        window["signal_name"] = np.asarray(name_list, dtype=object)
        window["mod"] = np.asarray(mod_list, dtype=np.int16)
        window["role"] = np.asarray(role_list, dtype=np.int8)

        window["outputs_emb"] = outputs_emb
        window["outputs_shapes"] = outputs_shapes
        window["outputs_names"] = outputs_names

        logger.debug(
            "win %s (shot %s) | tokens=%d (context=%d, act=%d) | pos=[%s..%s]",
            window.get("window_index"),
            window.get("shot_id"),
            len(emb_list),
            sum(r == ROLE_CONTEXT for r in role_list),
            sum(r == ROLE_ACTUATOR for r in role_list),
            pos.min() if len(pos) else None,
            pos.max() if len(pos) else None,
        )

        return window
