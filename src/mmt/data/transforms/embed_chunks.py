"""
EmbedChunksTransform
====================

Embed chunk-level and window-level output signals using the codecs and
SignalSpec definitions provided at configuration time.

Expected input window
---------------------
window = {
    "chunks": {
        "input":    [chunk_dict, ...],
        "actuator": [chunk_dict, ...],
    },
    "output": { <signal_name>: {"values": ndarray or list, ...}, ... },
    "shot_id": <identifier>,
    "window_index": <int>,
    ...
}

Each chunk_dict must contain (from ChunkWindowsTransform / TrimChunksTransform):

    {
        "signals": { <signal_name>: ndarray or list (or None), ... },
        "chunk_index_in_window": <int>,   # 0,1,2,... within the role span
        "chunk_index_global": <int>,      # stable slot id on the stride grid
        "pos": <int>,                     # added upstream by TrimChunksTransform
        ...
    }

This transform:
1) Encodes every signal in each chunk using the appropriate codec.
2) Stores results as:
       chunk["embeddings"][signal_id]   = embedding_vector
3) Encodes window-level outputs (if present) into:
       window["embedded_output"]
       window["embedded_output_shapes"]
4) Drops chunk raw "signals" on the returned copy to reduce memory.


Caching (v0)
------------
Cache key (robust, no fallbacks):

    (shot_id, role, signal_id, chunk_index_global)

This should produce cache hits for overlapping windows within a shot when
window_stride_sec == chunk_stride_sec.
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
    """

    def __init__(
        self,
        signal_specs: SignalSpecRegistry,
        codecs: Mapping[int, Any],
    ) -> None:
        self.signal_specs = signal_specs
        self.codecs = dict(codecs)

        # Deterministic cache:
        # (shot_id, role, signal_id, chunk_index_global) -> embedding
        self._cache: Dict[Tuple[Any, str, int, int], np.ndarray] = {}
        # We only need within-shot reuse; clear caches when shot_id changes.
        self._last_shot_id: Any = None

    def _get_spec(self, role: str, name: str):
        spec = self.signal_specs.get(role, name)
        if spec is None:
            raise KeyError(f"No SignalSpec found for role={role!r}, name={name!r}")
        return spec

    def _get_codec(self, sid: int):
        if sid not in self.codecs:
            raise KeyError(f"No codec registered for signal_id={sid}")
        return self.codecs[sid]

    def __call__(self, window: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if window is None:
            return None

        shot_id = window.get("shot_id")
        w_idx = window.get("window_index")
        if w_idx is None:
            raise ValueError("EmbedChunksTransform requires window['window_index']")

        # ------------------------------------------------------------------
        # Prevent unbounded cache growth across shots.
        # The cache key includes shot_id, so cross-shot reuse is impossible;
        # keeping old entries only wastes RAM during long streaming/caching runs.
        # ------------------------------------------------------------------
        if shot_id != self._last_shot_id:
            self._cache.clear()
            self._last_shot_id = shot_id

        chunks_dict = window.get("chunks") or {}

        # Stats for logging
        n_chunks_total = 0
        n_signal_emb_new = 0
        n_signal_cache_hits = 0
        n_out_signals = 0
        n_out_emb_new = 0

        out_window = dict(window)

        # ------------------------------------------------------------------
        # Embed chunk-level signals
        # ------------------------------------------------------------------
        new_chunks: Dict[str, Any] = {}
        for role in ("input", "actuator"):
            role_chunks = chunks_dict.get(role) or []
            n_chunks_total += len(role_chunks)

            new_role_chunks = []
            for ch in role_chunks:
                ch2: Dict[str, Any] = dict(ch)

                if "chunk_index_global" not in ch:
                    raise KeyError(
                        f"Chunk missing 'chunk_index_global' (role={role}, win={w_idx}, shot={shot_id})"
                    )
                chunk_g = int(ch["chunk_index_global"])

                signals = ch.get("signals") or {}

                emb_map: Dict[int, np.ndarray] = {}

                for name, values in signals.items():
                    if values is None:
                        continue

                    spec = self._get_spec(role, name)
                    sid = int(spec.signal_id)
                    codec = self._get_codec(sid)

                    arr = np.asarray(values)
                    key = (shot_id, role, sid, chunk_g)

                    if key in self._cache:
                        emb = self._cache[key]
                        n_signal_cache_hits += 1
                    else:
                        emb = codec.encode(arr)
                        self._cache[key] = emb
                        n_signal_emb_new += 1

                    emb_map[sid] = emb

                ch2["embeddings"] = emb_map

                # Drop raw values on the returned copy (reduces memory)
                ch2["signals"] = None

                new_role_chunks.append(ch2)

            new_chunks[role] = new_role_chunks

        # Preserve any other keys under "chunks" if present
        for k, v in (chunks_dict or {}).items():
            if k not in new_chunks:
                new_chunks[k] = v

        out_window["chunks"] = new_chunks

        # ------------------------------------------------------------------
        # Embed output-level signals (not cached in v0)
        # ------------------------------------------------------------------
        outputs = window.get("output") or {}

        emb_out: Dict[int, np.ndarray] = {}
        shape_out: Dict[int, Any] = {}

        if isinstance(outputs, dict):
            for name, info in outputs.items():
                if not isinstance(info, dict):
                    continue

                values = info.get("values")
                if values is None:
                    continue

                spec = self._get_spec("output", name)
                sid = int(spec.signal_id)
                codec = self._get_codec(sid)

                arr = np.asarray(values)
                emb = codec.encode(arr)

                emb_out[sid] = emb
                shape_out[sid] = tuple(arr.shape)

                n_out_signals += 1
                n_out_emb_new += 1

        if emb_out:
            out_window["embedded_output"] = emb_out
            out_window["embedded_output_shapes"] = shape_out

        # ------------------------------------------------------------------
        # Debug summary
        # ------------------------------------------------------------------
        logger.debug(
            "win %s (shot %s) | chunks=%d, signal_new_emb=%d, signal_cache_hits=%d, "
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
