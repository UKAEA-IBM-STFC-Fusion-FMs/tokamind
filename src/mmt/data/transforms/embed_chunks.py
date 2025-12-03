from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple
import logging

import numpy as np

from mmt.data.embeddings.signal_spec import SignalSpecRegistry

logger = logging.getLogger("mmt.Embedding")


class EmbedChunksTransform:
    """
    Attach per-chunk and per-window embeddings to a window, with caching.

    This transform runs **after** ChunkingTransform and DropNaChunksTransform,
    as part of the TaskModelTransformWrapper model_transform chain.

    It:
      - iterates over chunks in window["chunks"]["input"] and ["actuator"],
      - for each non-masked signal in each chunk:
          * looks up its SignalSpec (role + name → signal_id, encoder),
          * looks up a pre-built codec for that signal_id,
          * uses a **dedup-safe cache key**:

                (shot_id, role, signal_id, chunk_start_sample, chunk_size_samples)

            where:
                - chunk_start_sample: absolute sample index on the shot timeline,
                - chunk_size_samples: number of time samples in the chunk;
          * computes or reuses the embedding,
          * stores embeddings and original shapes in the chunk dict,
          * clears chunk["signals"] to save memory;
      - iterates over window["output"] (window-level outputs),
          * embeds each non-masked output signal using the same
            SignalSpec + codec machinery,
          * stores embeddings and original shapes in the window dict,
          * clears output["values"] to save memory.

    It does **not**:
      - flatten chunks into a model-ready sequence,
      - apply max_history, masks, or tokenization.
    These are handled later by the MMT-specific transform or collate.

    Expected input (per window)
    ---------------------------
    Same window dict as returned by DropNaChunksTransform, i.e. with:

        window["chunks"]["input"]    = [chunk_dict, ...]
        window["chunks"]["actuator"] = [chunk_dict, ...]
        window["output"]             = {
            "<signal_name>": {
                "values": np.ndarray | None,
                ...
            },
        }

    Each chunk_dict has:

        {
          "role": "input" | "actuator",
          "chunk_index_in_window": int,
          "chunk_start_sample": int,      # absolute sample index
          "chunk_size_samples": int,
          "chunk_start_time": float,
          "signals": {
              var_name: np.ndarray | None,  # chunk time axis is the last axis
          },
        }

    Output
    ------
    The same window dict, enriched with:

        # per-chunk embeddings
        chunk["embeddings"]   = { signal_id: np.ndarray(D,) }
        chunk["orig_shapes"]  = { signal_id: tuple(...) }
        chunk["signals"]      = None   # raw signals dropped

        # per-window output embeddings
        window["embedded_output"]        = { signal_id: np.ndarray(D,) }
        window["embedded_output_shapes"] = { signal_id: tuple(...) }

        # raw output values cleared:
        window["output"][name]["values"] = None

    Parameters
    ----------
    signal_specs : SignalSpecRegistry
        Registry describing each signal (role, name → signal_id, encoder info).

    codecs : Mapping[int, Any]
        Mapping from signal_id to an encoder object that exposes:
            encode(x: np.ndarray) -> np.ndarray(D,)
        For example: DCT3DCodec, FPCAEncoder, IdentityCodec.
    """

    def __init__(
        self,
        signal_specs: SignalSpecRegistry,
        codecs: Mapping[int, Any],
    ) -> None:
        self.signal_specs = signal_specs
        self.codecs = dict(codecs)

        # Cache:
        #   (shot_id, role, signal_id, chunk_start_sample, chunk_size_samples) -> embedding (D,)
        self._cache: Dict[Tuple[Any, str, int, int, int], np.ndarray] = {}
        # Original shapes of embedded arrays (useful for decode)
        self._orig_shapes: Dict[Tuple[Any, str, int, int, int], Tuple[int, ...]] = {}

    # ------------------------------------------------------------------ helpers

    def _get_signal_spec(self, role: str, name: str):
        spec = self.signal_specs.get(role, name)
        if spec is None:
            raise KeyError(f"No SignalSpec found for role={role!r}, name={name!r}")
        return spec

    def _get_codec(self, signal_id: int):
        try:
            return self.codecs[signal_id]
        except KeyError:
            raise KeyError(f"No codec registered for signal_id={signal_id}")

    # ------------------------------------------------------------------ main API

    def __call__(self, window: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Embed all chunks and outputs in a single window, with caching across
        windows for input/actuator chunks (but **not** for outputs).
        """
        if window is None:
            return None

        chunks_dict = window.get("chunks")
        if not isinstance(chunks_dict, dict):
            chunks_dict = {}

        shot_id = window.get("shot_id", None)
        w_idx = window.get("window_index", None)

        # Counters for logging
        n_chunks_total = 0
        n_signal_emb_new = 0
        n_signal_cache_hits = 0
        cache_hit_keys: list[Tuple[Any, str, int, int, int]] = []

        n_out_signals = 0
        n_out_emb_new = 0

        # -------------------- Embed input & actuator chunks --------------------

        for role in ("input", "actuator"):
            role_chunks = chunks_dict.get(role) or []
            for ch in role_chunks:
                n_chunks_total += 1

                signals = ch.get("signals") or {}
                emb_map = ch.setdefault("embeddings", {})
                shape_map = ch.setdefault("orig_shapes", {})

                chunk_start_sample = ch.get("chunk_start_sample")
                if chunk_start_sample is None:
                    # Back-compat: fall back to chunk_index_global if present
                    chunk_start_sample = ch.get("chunk_index_global")
                if chunk_start_sample is None:
                    raise ValueError(
                        f"Chunk is missing 'chunk_start_sample' / 'chunk_index_global' "
                        f"(role={role}, window_index={w_idx}, shot_id={shot_id})"
                    )
                chunk_start_sample = int(chunk_start_sample)

                chunk_size_samples = ch.get("chunk_size_samples")

                for name, values in signals.items():
                    if values is None:
                        continue

                    spec = self._get_signal_spec(role, name)
                    codec = self._get_codec(spec.signal_id)

                    arr = np.asarray(values)
                    # If chunk_size_samples was not stored, infer from time axis
                    this_chunk_size = (
                        int(chunk_size_samples)
                        if chunk_size_samples is not None
                        else int(arr.shape[-1])
                    )

                    key = (
                        shot_id,
                        role,
                        spec.signal_id,
                        chunk_start_sample,
                        this_chunk_size,
                    )

                    if key in self._cache:
                        emb = self._cache[key]
                        n_signal_cache_hits += 1
                        cache_hit_keys.append(key)
                    else:
                        emb = codec.encode(arr)  # (D,)
                        self._cache[key] = emb
                        self._orig_shapes[key] = arr.shape
                        n_signal_emb_new += 1

                    emb_map[spec.signal_id] = emb
                    shape_map[spec.signal_id] = arr.shape

                # Always drop raw signals to save memory
                ch["signals"] = None

        # -------------------------- Embed outputs ------------------------------

        outputs = window.get("output") or {}
        emb_out: Dict[int, np.ndarray] = {}
        shape_out: Dict[int, Tuple[int, ...]] = {}

        for name, info in outputs.items():
            if not isinstance(info, dict):
                continue

            values = info.get("values")
            if values is None:
                continue

            spec = self._get_signal_spec("output", name)
            codec = self._get_codec(spec.signal_id)

            arr = np.asarray(values)
            emb = codec.encode(arr)  # (D,)

            emb_out[spec.signal_id] = emb
            shape_out[spec.signal_id] = arr.shape
            n_out_signals += 1
            n_out_emb_new += 1

            # Drop raw output values to save memory
            info["values"] = None

        if emb_out:
            window["embedded_output"] = emb_out
            window["embedded_output_shapes"] = shape_out

        # ----------------------------- Logging ---------------------------------

        logger.debug(
            "[EmbeddingTransform] window %s (shot %s): "
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

        return window
