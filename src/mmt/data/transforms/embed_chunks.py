"""
EmbedChunksTransform
====================

Embed chunk-level and window-level output signals using the codecs and SignalSpec definitions provided at configuration
time.

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
        "chunk_index_global": <int>,      # stable slot ID on the stride grid
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

NaN imputation
--------------
Controlled by ``nan_imputation`` (from ``preprocess.embed_chunks.nan_imputation`` in config, default ``"zero"``).

``"zero"``: any NaN/inf values that survive SelectValidWindowsTransform are zero-filled on a **local copy**
immediately before encoding. If the data are standardized, zero corresponds to the signal mean; otherwise
this is a literal zero-fill.

``"interpolate"``: Non-finite values are filled by local interpolation before encoding:

  1. **Temporal interpolation** along the T axis (per spatial position): fills missing timesteps from valid
     neighbours. ``np.interp`` clamps at boundaries, so trailing non-finite values are held constant at the
     last valid value — avoiding a jump to the global standardized mean.
  2. **Spatial interpolation** along the H axis (per timestep): fills remaining non-finite positions from
     neighbouring positions along H. Applied after step 1 so that entirely-missing slices can still be
     filled from spatial neighbours if any valid neighbour exists.
  3. **Zero fallback**: any position still non-finite after both passes is zero-filled as a last resort.

  Both interpolation passes use ``~np.isfinite`` so that ±inf values are treated identically to NaN and
  do not survive into the codec. The zero fallback explicitly clears both NaN and ±inf.
  Interpolation never produces hard step discontinuities, so encoder coefficients are not contaminated
  by artificial edges.

``None``: no imputation is performed. The array is passed directly to the codec. An error is raised at
construction time if any registered codec has ``requires_finite_input=True``, since those codecs cannot
handle non-finite arrays. Use ``None`` only when all codecs can handle non-finite inputs natively.

For output signals, imputation (regardless of strategy) is applied only to the local copy used for encoding.
The original values in ``window["output"][name]["values"]`` are **never modified**, preserving NaN locations
for benchmark-comparable evaluation metrics (e.g. nanmean in the tokamark evaluator).

Caching (v0)
------------
Cache key (robust, no fallbacks):

    (shot_id, role, signal_id, chunk_index_global)

This should produce cache hits for overlapping windows within a shot when window_stride_sec == chunk_stride_sec.
"""

from __future__ import annotations

from typing import Any, Literal
from collections.abc import Mapping
import logging
import numpy as np

from mmt.data.signal_spec import SignalSpecRegistry


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.EmbedChunks")


# ======================================================================================================================
class EmbedChunksTransform:
    """
    Embed all chunk-level and output-level signals of a window according to the provided `SignalSpecRegistry` and codec
    mapping.

    Attributes
    ----------
    signal_specs : SignalSpecRegistry
        Registry of signal specifications.
    codecs : Mapping[int, Any]
        Codec mapping (signal_id -> codec).
    nan_imputation : str | None
        NaN imputation strategy before encoding. ``"zero"`` zero-fills (equal to signal mean only in standardized space),
        ``"interpolate"`` uses temporal then spatial interpolation with zero fallback, ``None`` passes the
        array to the codec unchanged.
    _cache : dict[tuple[Any, str, int, int], np.ndarray]
        Supporting variable to cache (shot_id, role, signal_id, chunk_index_global).
    _last_shot_id : Any
        Supporting variable to hold the last shot ID.

    Methods
    -------
    _interpolate_nans(arr)
        Fill NaN values via temporal then spatial interpolation, with zero fallback.
    __call__(window)
        Call method for the class instances to behave like a function.
    _get_spec(role, name)
        Get `SignalSpec` instance associated with a given role/name.
    _get_codec(sid)
        Get codec from signal ID.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        signal_specs: SignalSpecRegistry,
        codecs: Mapping[int, Any],
        nan_imputation: Literal["zero", "interpolate"] | None = "zero",
    ) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        signal_specs : SignalSpecRegistry
            Registry of signal specifications.
        codecs : Mapping[int, Any]
            Codec mapping (signal_id -> codec).
        nan_imputation : "zero" | "interpolate" | None
            Non-finite imputation strategy applied before ``codec.encode()``.
            ``"zero"`` (default): zero-fill on a local copy; zero equals signal mean only in standardized space.
            ``"interpolate"``: temporal then spatial linear interpolation with zero fallback. Both passes
            use ``~np.isfinite`` so ±inf is treated identically to NaN.
            ``None``: no imputation; the array is passed to the codec as-is. Raises ``ValueError`` at
            construction time if any registered codec has ``requires_finite_input=True``.
            Optional. Default: ``"zero"``.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors, as this is a callable class.

        Raises
        ------
        ValueError
            If ``nan_imputation`` is not one of ``"zero"``, ``"interpolate"``, or ``None``.
        ValueError
            If ``nan_imputation=None`` and any registered codec has ``requires_finite_input=True``.

        """

        _VALID = {"zero", "interpolate", None}
        if nan_imputation not in _VALID:
            raise ValueError(
                f"[EmbedChunksTransform] Invalid nan_imputation={nan_imputation!r}. "
                "Must be one of: 'zero', 'interpolate', None."
            )

        if nan_imputation is None:
            bad = [sid for sid, c in codecs.items() if getattr(c, "requires_finite_input", True)]
            if bad:
                raise ValueError(
                    f"[EmbedChunksTransform] nan_imputation=None but codec(s) for signal_id={bad} "
                    "have requires_finite_input=True. Set nan_imputation='zero' or 'interpolate', "
                    "or use codecs that can handle non-finite inputs natively."
                )

        self.signal_specs = signal_specs
        self.codecs = dict(codecs)
        self.nan_imputation = nan_imputation

        # Deterministic cache:
        # (shot_id, role, signal_id, chunk_index_global) -> embedding
        self._cache: dict[tuple[Any, str, int, int], np.ndarray] = {}

        # We only need within-shot reuse; clear caches when shot_id changes.
        self._last_shot_id: Any = None

    # ------------------------------------------------------------------------------------------------------------------
    def _get_spec(self, role: str, name: str) -> Any:
        """
        Get `SignalSpec` instance associated with a given role/name.

        Parameters
        ----------
        role : str
            Target role.
        name : str
            Target name.

        Returns
        -------
        Any
            `SignalSpec` instance associated with the passed `role` and `name`.

        Raises
        ------
        KeyError
            If no `SignalSpec` instance found within `self.signal_specs` for passed `role`.

        """

        spec = self.signal_specs.get(role, name)
        if spec is None:
            raise KeyError(f"[EmbedChunksTransform] No SignalSpec found for `role={role!r}`, name={name!r}.")

        return spec

    # ------------------------------------------------------------------------------------------------------------------
    def _get_codec(self, sid: int) -> Any:
        """
        Get codec from signal ID.

        Parameters
        ----------
        sid : int
            Target signal ID.

        Returns
        -------
        Any
            Codec associated with target signal ID.

        Raises
        ------
        KeyError
            If no codec is registered for the target signal ID.

        """

        if sid not in self.codecs:
            raise KeyError(f"[EmbedChunksTransform] No codec registered for `signal_id={sid}`.")

        return self.codecs[sid]

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
        """
        Fill NaN values via temporal then spatial linear interpolation, with zero fallback.

        Uses the same native-to-DCT view convention as ``DCT3DCodec``:
        ``(T,) -> (1, 1, T)``, ``(C, T) -> (C, 1, T)``, ``(H, W, T) -> (H, W, T)``.
        The returned array has the same shape as the input.

        Steps
        -----
        1. **Temporal interpolation** (along T, per H×W position): fills non-finite timesteps from valid
           neighbours. ``np.interp`` clamps to boundary values, so trailing non-finite positions are held
           at the last valid value — avoiding a jump to the global standardized mean.
        2. **Spatial interpolation** (along H, per W×T position): fills remaining non-finite positions from
           neighbouring positions along H. Applied after step 1 so that entirely-missing slices can still
           be filled from spatial neighbours if any valid neighbour exists.
        3. **Zero fallback**: any position still non-finite after both passes is replaced with 0.0 in-place.
           ±inf is explicitly cleared here via ``posinf=0.0, neginf=0.0``.

        Both interpolation passes check ``~np.isfinite`` so ±inf is treated identically to NaN and
        cannot survive into the encoder.

        Parameters
        ----------
        arr : np.ndarray
            Input array of shape ``(T,)``, ``(C, T)``, or ``(H, W, T)``. Must be a writable copy — modified
            in-place.

        Returns
        -------
        np.ndarray
            Array with all non-finite values filled. Guaranteed finite on return.

        Raises
        ------
        ValueError
            If ``arr`` is not 1D, 2D, or 3D.

        """

        original_shape = arr.shape
        if arr.ndim == 1:
            arr_view = arr.reshape(1, 1, arr.shape[0])
        elif arr.ndim == 2:
            arr_view = arr.reshape(arr.shape[0], 1, arr.shape[1])
        elif arr.ndim == 3:
            arr_view = arr
        else:
            raise ValueError(f"[EmbedChunksTransform] Interpolation supports 1D/2D/3D arrays, got shape={arr.shape}.")

        H, W, T = arr_view.shape

        # Step 1: temporal interpolation per (h, w)
        t_idx = np.arange(T)
        for h in range(H):
            for w in range(W):
                row = arr_view[h, w]
                non_finite_mask = ~np.isfinite(row)
                if non_finite_mask.any() and not non_finite_mask.all():
                    valid = np.where(~non_finite_mask)[0]
                    arr_view[h, w] = np.interp(t_idx, valid, row[valid])

        # Step 2: spatial interpolation along H per (w, t)
        if H > 1:
            h_idx = np.arange(H)
            for w in range(W):
                for t in range(T):
                    col = arr_view[:, w, t]
                    non_finite_mask = ~np.isfinite(col)
                    if non_finite_mask.any() and not non_finite_mask.all():
                        valid = np.where(~non_finite_mask)[0]
                        arr_view[:, w, t] = np.interp(h_idx, valid, col[valid])

        # Step 3: zero fallback for any remaining non-finite values (NaN and ±inf)
        np.nan_to_num(arr_view, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        return arr_view.reshape(original_shape)

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(  # NOSONAR - Ignore cognitive complexity
        self, window: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """
        Call method for the class instances to behave like a function.

        Parameters
        ----------
        window : dict[str, Any] | None
            Window on which the transform is applied.

        Returns
        -------
        dict[str, Any] | None
            Extended window with updated/new values for "chunks", "embedded_output", and "embedded_output_shapes" keys.

        Raises
        ------
        ValueError
            If `window` does not define the required key "window_index".
        KeyError
            If `window["chunks"]` does not have the required "chunk_index_global" for "input" or "actuator" role.

        """

        if window is None:
            return None

        shot_id = window.get("shot_id")
        w_idx = window.get("window_index")
        if w_idx is None:
            raise ValueError("[EmbedChunksTransform] `window['window_index']` is required.")

        # ..............................................................................................................
        # Prevent unbounded cache growth across shots.
        # The cache key includes shot_id, so cross-shot reuse is impossible; keeping old entries only wastes RAM during
        # long streaming/caching runs.
        # ..............................................................................................................

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

        # ..............................................................................................................
        # Embed chunk-level signals
        # ..............................................................................................................

        new_chunks: dict[str, Any] = {}
        for role in ("input", "actuator"):
            role_chunks = chunks_dict.get(role) or []
            n_chunks_total += len(role_chunks)

            new_role_chunks = []
            for ch in role_chunks:
                ch2: dict[str, Any] = dict(ch)  # type: ignore[arg-type]

                if "chunk_index_global" not in ch:
                    raise KeyError(
                        f"[EmbedChunksTransform] Chunk missing 'chunk_index_global' (role={role}, win={w_idx}, "
                        f"shot={shot_id})."
                    )

                chunk_g = int(ch["chunk_index_global"])
                signals = ch.get("signals") or {}
                emb_map: dict[int, np.ndarray] = {}

                for name, values in signals.items():
                    if values is None:
                        continue

                    spec = self._get_spec(role=role, name=name)
                    sid = int(spec.signal_id)
                    codec = self._get_codec(sid=sid)

                    arr = np.asarray(values)
                    key = (shot_id, role, sid, chunk_g)

                    if key in self._cache:
                        emb = self._cache[key]
                        n_signal_cache_hits += 1
                    else:
                        if not np.isfinite(arr).all():
                            if self.nan_imputation == "zero":
                                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                            elif self.nan_imputation == "interpolate":
                                arr = self._interpolate_nans(arr.copy())
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

        # ..............................................................................................................
        # Embed output-level signals (not cached in v0)
        # ..............................................................................................................

        outputs = window.get("output") or {}

        emb_out: dict[int, np.ndarray] = {}
        shape_out: dict[int, Any] = {}

        if isinstance(outputs, dict):
            for name, info in outputs.items():
                if not isinstance(info, dict):
                    continue

                values = info.get("values")
                if values is None:
                    continue

                spec = self._get_spec(role="output", name=name)
                sid = int(spec.signal_id)
                codec = self._get_codec(sid=sid)

                # Identity-encoded outputs: skip embedding entirely.
                # The native values kept by FinalizeWindowTransform are sufficient for
                # native_sparse_mse. Storing output_emb separately would duplicate the
                # data in memory for every cached window.
                if getattr(codec, "is_identity", False):
                    continue

                arr = np.asarray(values)
                # Impute on a local copy only — native values in window["output"] are preserved for eval metrics.
                if not np.isfinite(arr).all():
                    if self.nan_imputation == "zero":
                        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    elif self.nan_imputation == "interpolate":
                        arr = self._interpolate_nans(arr.copy())
                emb = codec.encode(arr)

                emb_out[sid] = emb
                shape_out[sid] = tuple(arr.shape)

                n_out_signals += 1
                n_out_emb_new += 1

        if emb_out:
            out_window["embedded_output"] = emb_out
            out_window["embedded_output_shapes"] = shape_out

        # ..............................................................................................................
        # Debug summary
        # ..............................................................................................................

        logger.debug(
            "win %s (shot %s) | chunks=%d, signal_new_emb=%d, signal_cache_hits=%d, out_signals=%d, out_new_emb=%d",
            w_idx,
            shot_id,
            n_chunks_total,
            n_signal_emb_new,
            n_signal_cache_hits,
            n_out_signals,
            n_out_emb_new,
        )

        return out_window
