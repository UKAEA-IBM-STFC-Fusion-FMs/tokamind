"""
EmbedChunksTransform
====================

This transform embeds chunk-level and window-level output signals using
the codecs and SignalSpec definitions provided at configuration time.

Input windows contain:

    window = {
        "chunks": {
            "input":    [chunk_dict, ...],
            "actuator": [chunk_dict, ...],
        },
        "output": {
            <signal_name>: {"values": ndarray or list}
        },
        "shot_id": <identifier>,
        "window_index": <int>,
        ...
    }

Each chunk_dict has:

    {
        "signals": { <signal_name>: ndarray or list },
        "chunk_start_sample": <int>,
        "chunk_start_time": <float>,
        ...
    }

This transform:

1. Encodes every signal in each chunk using the appropriate codec.
2. Stores results as:
       chunk["embeddings"][signal_id]   = embedded_vector
       chunk["orig_shapes"][signal_id]  = original_array_shape
3. Encodes window-level outputs (if present) into:
       window["embedded_output"]
       window["embedded_output_shapes"]
4. Optionally removes raw "values" arrays if keep_output_native=False.
5. Leaves the window schema intact while adding the new embedding fields.
6. Uses deterministic caching for repeated chunk encodings.

A compact debug summary is emitted at DEBUG level through the logger.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple
import logging
import numpy as np

from mmt.data.signal_spec import SignalSpecRegistry

logger = logging.getLogger("mmt.EmbedChunks")


class EmbedChunksTransform:
    """
    Embed all chunk-level and output-level signals of a window according to the
    provided SignalSpecRegistry and codec mapping.

    Parameters
    ----------
    signal_specs : SignalSpecRegistry
        Registry mapping (role, name) → SignalSpec containing signal_id,
        modality, and embedding dimension metadata.

    codecs : Mapping[int, Any]
        Mapping from signal_id → codec. Each codec must implement:
            encoded = codec.encode(np.ndarray)

    keep_output_native : bool, default=False
        If True, preserve the raw output arrays in window["output"][name]["values"].
        Otherwise, strip these values after embedding.

    Output
    ------
    window : Dict[str, Any]
        The same input window, augmented with:
            chunk["embeddings"]
            chunk["orig_shapes"]
            window["embedded_output"]
            window["embedded_output_shapes"]
    """

    def __init__(
        self,
        signal_specs: SignalSpecRegistry,
        codecs: Mapping[int, Any],
        keep_output_native: bool = False,
    ) -> None:
        self.signal_specs = signal_specs
        self.codecs = dict(codecs)
        self.keep_output_native = bool(keep_output_native)

        # Deterministic cache: (shot_id, role, signal_id, chunk_start_sample)
        # NOTE: This cache is internal state, but it does not mutate input windows/chunks.
        self._cache: Dict[Tuple[Any, str, int, int], np.ndarray] = {}
        self._orig_shapes: Dict[Tuple[Any, str, int, int], Tuple[int, ...]] = {}

    # ------------------------------------------------------------------

    def _get_spec(self, role: str, name: str):
        spec = self.signal_specs.get(role, name)
        if spec is None:
            raise KeyError(f"No SignalSpec found for role={role!r}, name={name!r}")
        return spec

    def _get_codec(self, sid: int):
        if sid not in self.codecs:
            raise KeyError(f"No codec registered for signal_id={sid}")
        return self.codecs[sid]

    # ------------------------------------------------------------------

    def __call__(self, window: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if window is None:
            return None

        shot_id = window.get("shot_id")
        w_idx = window.get("window_index")
        chunks_dict = window.get("chunks") or {}

        # Stats for logging
        n_chunks_total = 0
        n_signal_emb_new = 0
        n_signal_cache_hits = 0
        n_out_signals = 0
        n_out_emb_new = 0

        # We'll build a new window dict to avoid mutating shared structures.
        out_window = dict(window)

        # ------------------------------------------------------------------
        # Embed chunk-level signals (pure: copy chunks before adding fields)
        # ------------------------------------------------------------------
        new_chunks: Dict[str, Any] = {}
        for role in ("input", "actuator"):
            role_chunks = chunks_dict.get(role) or []
            n_chunks_total += len(role_chunks)

            new_role_chunks = []
            for ch in role_chunks:
                # Copy the chunk dict (do not mutate `ch` in place).
                ch2 = dict(ch)

                signals = ch.get("signals") or {}
                # We will not mutate the original signals dict; produce a new one.
                # After embedding, we set signals to None in the copied chunk to reduce memory.
                # (This matches historical behavior without touching the original chunk.)
                emb_map: Dict[int, np.ndarray] = {}
                shape_map: Dict[int, Tuple[int, ...]] = {}

                # Determine chunk start
                chunk_start = ch.get("chunk_start_sample")
                if chunk_start is None:
                    chunk_start = ch.get("chunk_index_global")
                if chunk_start is None:
                    raise ValueError(
                        f"Chunk missing 'chunk_start_sample' "
                        f"(role={role}, window={w_idx}, shot={shot_id})"
                    )
                chunk_start = int(chunk_start)

                for name, values in signals.items():
                    if values is None:
                        continue

                    spec = self._get_spec(role, name)
                    codec = self._get_codec(spec.signal_id)
                    arr = np.asarray(values)

                    key = (shot_id, role, spec.signal_id, chunk_start)

                    if key in self._cache:
                        emb = self._cache[key]
                        n_signal_cache_hits += 1
                    else:
                        emb = codec.encode(arr)
                        self._cache[key] = emb
                        self._orig_shapes[key] = arr.shape
                        n_signal_emb_new += 1

                    emb_map[int(spec.signal_id)] = emb
                    shape_map[int(spec.signal_id)] = tuple(arr.shape)

                # Attach embedding fields to copied chunk
                ch2["embeddings"] = emb_map
                ch2["orig_shapes"] = shape_map

                # Remove raw values to reduce memory (on the copy only)
                ch2["signals"] = None

                new_role_chunks.append(ch2)

            new_chunks[role] = new_role_chunks

        # Preserve any other roles/keys under "chunks" if present
        for k, v in (chunks_dict or {}).items():
            if k not in new_chunks:
                new_chunks[k] = v

        out_window["chunks"] = new_chunks

        # ------------------------------------------------------------------
        # Embed output-level signals (pure: copy output dict/entries before editing)
        # ------------------------------------------------------------------
        outputs = window.get("output") or {}
        new_outputs: Dict[str, Any] = (
            dict(outputs) if isinstance(outputs, dict) else outputs
        )

        emb_out: Dict[int, np.ndarray] = {}
        shape_out: Dict[int, Any] = {}

        if isinstance(outputs, dict):
            for name, info in outputs.items():
                if not isinstance(info, dict):
                    # Keep as-is (or normalize) without mutating original.
                    continue

                values = info.get("values")
                if values is None:
                    continue

                spec = self._get_spec("output", name)
                codec = self._get_codec(spec.signal_id)

                arr = np.asarray(values)
                emb = codec.encode(arr)

                emb_out[int(spec.signal_id)] = emb
                shape_out[int(spec.signal_id)] = tuple(arr.shape)

                n_out_signals += 1
                n_out_emb_new += 1  # outputs are not cached

                if not self.keep_output_native:
                    # Copy entry dict before stripping values
                    info2 = dict(info)
                    info2["values"] = None
                    new_outputs[name] = info2

        out_window["output"] = new_outputs

        if emb_out:
            out_window["embedded_output"] = emb_out
            out_window["embedded_output_shapes"] = shape_out

        # ------------------------------------------------------------------
        # Debug summary
        # ------------------------------------------------------------------
        logger.debug(
            "win %s (shot %s): "
            "chunks=%d, signal_new_emb=%d, signal_cache_hits=%d, "
            "out_signals=%d, out_new_emb=%d",
            w_idx,
            shot_id,
            n_chunks_total,
            n_signal_emb_new,
            n_signal_cache_hits,
            n_out_signals,
            n_out_emb_new,
        )

        return out_window
