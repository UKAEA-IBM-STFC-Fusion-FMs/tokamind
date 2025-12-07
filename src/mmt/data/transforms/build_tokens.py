from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

import numpy as np

from mmt.data.signal_spec import SignalSpecRegistry

logger = logging.getLogger("mmt.BuildTokens")

# Token roles (shared with collate + model)
ROLE_CONTEXT = 0
ROLE_ACTUATOR = 1
ROLE_OUTPUT = 2  # reserved for learned Y token inside the model


class BuildTokensTransform:
    """
    Convert the output of EmbedChunksTransform into token-level metadata
    that the collate will batch and the model will consume.

    This transform is intentionally **simple**:
      - it does NOT perform any truncation (max history, masking, etc.),
      - it just flattens per-chunk embeddings into a per-token sequence and
        assigns temporal positions and IDs.

    History truncation (max_context, max_chunks, …) is handled upstream
    by e.g. TrimChunksTransform. Here we simply assume that the chunks we
    receive are already limited to the desired history.

    Input (per window)
    ------------------
    After EmbedChunksTransform, a window dict contains at least:

        window["t_cut"]                  : float
        window["chunks"]["input"]        : [chunk_dict, ...]
        window["chunks"]["actuator"]     : [chunk_dict, ...]
        window["embedded_output"]        : { signal_id: np.ndarray(D_out,) }
        window["embedded_output_shapes"] : { signal_id: orig_shape }

    Each chunk_dict (for input/actuator) has:

        {
          "role": "input" | "actuator",
          "chunk_index_in_window": int,
          "chunk_index_global": int,
          "chunk_start_time": float,             # absolute time (sec) for chunk start
          "chunk_start_sample": int,             # global sample index on shot timeline
          "chunk_size_samples": int,
          "embeddings": { signal_id: np.ndarray(D_enc,) },
          "orig_shapes": { signal_id: tuple(...) },
          # "signals" has already been dropped by EmbedChunksTransform
        }

    Output (per window)
    -------------------
    This transform **adds** the following fields:

        # Token sequence for this window (length L)
        window["emb_chunks"]  : List[np.ndarray(D_enc,)]
        window["pos"]         : np.ndarray(L,)   # int32, temporal positions (see below)
        window["id"]          : np.ndarray(L,)   # int32, physical signal IDs
        window["mod"]         : np.ndarray(L,)   # int16, modality IDs
        window["role"]        : np.ndarray(L,)   # int8, ROLE_CONTEXT / ROLE_ACTUATOR
        window["signal_name"] : np.ndarray(L,)   # dtype=object, human-readable names

        # Outputs (still keyed by physical signal_id)
        window["outputs_emb"]   : { signal_id: np.ndarray(D_out,) }
        window["outputs_shapes"]: { signal_id: orig_shape }
        window["outputs_names"] : { signal_id: str }

    Positional convention
    ---------------------
    Let:

        t_out_end = t_cut + delta + output_length

    and consider a chunk that covers [t_start, t_start + chunk_length_sec].

    We define:

        chunk_end = t_start + chunk_length_sec
        steps     = round((t_out_end - chunk_end) / chunk_length_sec)
        pos       = steps + 1

    So:

        • chunk that ends exactly at t_out_end       → steps = 0  → pos = 1
        • one chunk earlier                          → steps = 1  → pos = 2
        • etc.

    In other words, `pos` is a discrete “how many chunks before the output”
    counter, as a **positive integer starting from 1**.

    The (learned) Y token lives **inside the model** with:

        pos = 0, role = ROLE_OUTPUT, id = special_Y_id.

    Notes
    -----
    • This transform does *not* apply any history truncation. History is
      limited upstream (e.g. by TrimChunksTransform), so BuildTokensTransform
      can remain model-agnostic.
    • The sequence is grouped by role: all context tokens first, then all
      actuator tokens. Temporal structure is encoded in `pos`.
    • Embeddings are kept ragged: `emb_chunks` is a Python list of
      per-token vectors; projection to `d_model` happens inside the model.
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

        # Build modality_id mapping from registry
        modalities = signal_specs.modalities  # sorted unique modality names
        self._modality_to_id: Dict[str, int] = {m: i for i, m in enumerate(modalities)}

        # Map physical signal_id → modality_id
        self._signal_id_to_mod_id: Dict[int, int] = {}
        for spec in signal_specs.specs:
            m = spec.modality
            if m not in self._modality_to_id:
                raise KeyError(f"Unknown modality {m!r} for signal_id={spec.signal_id}")
            self._signal_id_to_mod_id[spec.signal_id] = self._modality_to_id[m]

        logger.debug(
            "[BuildTokens] init: chunk_length=%.6fs, delta=%.6fs, "
            "output_length=%.6fs, modalities=%s",
            self.chunk_length_sec,
            self.delta,
            self.output_length,
            self._modality_to_id,
        )

    # ------------------------------------------------------------------ helpers

    def _compute_pos_ids(self, t_cut: float, chunk_times: List[float]) -> np.ndarray:
        """
        Compute temporal positions `pos` for each chunk start time.

        The returned array has dtype int32 and follows the convention:

            pos = 1, 2, 3, ...

        where pos = 1 is the chunk that ends exactly at the end of the
        output window, and larger values are further in the past.
        """
        if t_cut is None:
            raise ValueError("BuildTokensTransform: window['t_cut'] must not be None")

        t_out_end = float(t_cut) + self.delta + self.output_length
        L = self.chunk_length_sec

        pos_ids: List[int] = []
        for t_start in chunk_times:
            if t_start is None:
                raise ValueError(
                    "BuildTokensTransform: chunk_start_time is None; "
                    "ChunkingTransform must set it."
                )

            chunk_end = float(t_start) + L
            steps = round((t_out_end - chunk_end) / L)
            pos_ids.append(steps + 1)

        return np.asarray(pos_ids, dtype=np.int32)

    # ------------------------------------------------------------------ main API

    def __call__(self, window: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transform a single window after EmbedChunksTransform.
        Produce final per-token metadata for the collate and model.

        The input window is modified in-place and returned.
        """
        if window is None:
            return None

        if "t_cut" not in window or window["t_cut"] is None:
            raise ValueError("BuildTokensTransform: window missing 't_cut' field")

        t_cut = float(window["t_cut"])

        chunks = window.get("chunks") or {}
        input_chunks = chunks.get("input", []) or []
        actuator_chunks = chunks.get("actuator", []) or []

        emb_list: List[np.ndarray] = []
        sig_list: List[int] = []
        name_list: List[str] = []
        role_list: List[int] = []
        mod_list: List[int] = []
        t_list: List[float] = []

        # --------------------------------------------------------------- #
        # 1. CONTEXT / INPUT TOKENS
        # --------------------------------------------------------------- #
        for ch in input_chunks:
            t_start = ch.get("chunk_start_time", None)
            if t_start is None:
                raise ValueError(
                    "BuildTokensTransform: input chunk missing 'chunk_start_time'"
                )

            emb_map = ch.get("embeddings") or {}
            if not emb_map:
                # All signals in this chunk were dropped / masked
                continue

            for sig_id, emb in emb_map.items():
                sig_id_int = int(sig_id)
                if sig_id_int not in self._signal_id_to_mod_id:
                    raise KeyError(
                        f"BuildTokensTransform: signal_id={sig_id_int} "
                        f"not found in SignalSpecRegistry"
                    )

                # --- Add token ---
                emb_list.append(emb)
                sig_list.append(sig_id_int)

                spec = self.signal_specs.get_by_id(sig_id_int)
                if spec is None:
                    raise KeyError(
                        f"SignalSpecRegistry: unknown signal_id={sig_id_int}"
                    )

                name_list.append(spec.name)
                role_list.append(ROLE_CONTEXT)
                mod_list.append(self._signal_id_to_mod_id[sig_id_int])
                t_list.append(float(t_start))

        # --------------------------------------------------------------- #
        # 2. ACTUATOR TOKENS
        # --------------------------------------------------------------- #
        for ch in actuator_chunks:
            t_start = ch.get("chunk_start_time", None)
            if t_start is None:
                raise ValueError(
                    "BuildTokensTransform: actuator chunk missing 'chunk_start_time'"
                )

            emb_map = ch.get("embeddings") or {}
            if not emb_map:
                continue

            for sig_id, emb in emb_map.items():
                sig_id_int = int(sig_id)
                if sig_id_int not in self._signal_id_to_mod_id:
                    raise KeyError(
                        f"BuildTokensTransform: signal_id={sig_id_int} "
                        f"not found in SignalSpecRegistry"
                    )

                emb_list.append(emb)
                sig_list.append(sig_id_int)

                spec = self.signal_specs.get_by_id(sig_id_int)
                if spec is None:
                    raise KeyError(
                        f"SignalSpecRegistry: unknown signal_id={sig_id_int}"
                    )

                name_list.append(spec.name)
                role_list.append(ROLE_ACTUATOR)
                mod_list.append(self._signal_id_to_mod_id[sig_id_int])
                t_list.append(float(t_start))

        # --------------------------------------------------------------- #
        # 3. POSITIONS
        # --------------------------------------------------------------- #
        emb_chunks = emb_list
        pos = self._compute_pos_ids(t_cut, t_list)

        # --------------------------------------------------------------- #
        # 4. OUTPUTS
        # --------------------------------------------------------------- #
        outputs = window.get("embedded_output") or {}
        output_shapes_all = window.get("embedded_output_shapes") or {}

        outputs_emb: Dict[int, np.ndarray] = {}
        outputs_shapes: Dict[int, Any] = {}
        outputs_names: Dict[int, str] = {}

        for sig_id, emb in outputs.items():
            sig_id_int = int(sig_id)
            outputs_emb[sig_id_int] = emb

            spec = self.signal_specs.get_by_id(sig_id_int)
            if spec is None:
                raise KeyError(f"SignalSpecRegistry: unknown signal_id={sig_id_int}")

            outputs_names[sig_id_int] = spec.name

            if sig_id in output_shapes_all:
                outputs_shapes[sig_id_int] = output_shapes_all[sig_id]
            elif sig_id_int in output_shapes_all:
                outputs_shapes[sig_id_int] = output_shapes_all[sig_id_int]
            else:
                raise KeyError(
                    f"BuildTokensTransform: missing shape for output signal_id={sig_id_int}"
                )

        # --------------------------------------------------------------- #
        # 5. Attach everything to the window
        # --------------------------------------------------------------- #
        window["emb_chunks"] = emb_chunks
        window["pos"] = pos
        window["id"] = np.asarray(sig_list, dtype=np.int32)
        window["signal_name"] = np.asarray(name_list, dtype=object)
        window["mod"] = np.asarray(mod_list, dtype=np.int16)
        window["role"] = np.asarray(role_list, dtype=np.int8)

        window["outputs_emb"] = outputs_emb
        window["outputs_shapes"] = outputs_shapes
        window["outputs_names"] = outputs_names

        logger.debug(
            "[BuildTokens] shot=%s win=%s | tokens=%d "
            "(context=%d, actuator=%d) | pos=[%s, %s] | outputs=%d",
            window.get("shot_id"),
            window.get("window_index"),
            len(emb_list),
            sum(1 for r in role_list if r == ROLE_CONTEXT),
            sum(1 for r in role_list if r == ROLE_ACTUATOR),
            int(pos.min()) if len(pos) else None,
            int(pos.max()) if len(pos) else None,
            len(outputs_emb),
        )

        return window
