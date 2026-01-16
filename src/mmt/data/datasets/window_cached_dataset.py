"""mmt.data.window_cached_dataset

RAM-backed dataset and helpers for cached *window-level* MMT data.

This module provides:

- WindowCachedDataset:
    A simple torch.utils.data.Dataset storing pre-tokenised **windows**
    in memory, ready to be fed directly to MMTCollate.

- materialize_tokenized_split_to_ram():
    A helper that runs the full model-specific transform chain once per
    window on top of a *shot-level* dataset adapter and returns a
    WindowCachedDataset.

Internally it also defines FlattenedStreamingDataset and a trivial collate
function used only during the parallel caching step.

The goal is to keep a clean separation between:

- shot-level datasets (shot -> iterable of windows), and
- cached, window-level datasets (index -> single window dict),

while keeping a symmetric, user-friendly API in the train scripts.

Notes on shuffling
------------------
- WindowCachedDataset is a map-style Dataset. When used with a standard
  DataLoader, shuffling is applied at the *window* level.
- During caching/materialisation itself, we can optionally shuffle *shots*
  (the order of indices into the shot-level dataset adapter) to avoid bias
  when using max_windows caps.

Dtype / memory
--------------
We accept only:
  - None
  - "float16"
  - "float32"

The dtype cast is applied only to cached *embedding* arrays (token embeddings
and output embeddings), never to native outputs.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence
import logging
import os
import random

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("mmt.Cache")


# -----------------------------------------------------------------------------
# Small RAM helper
# -----------------------------------------------------------------------------


def get_ram_gb() -> float:
    """Return the current RAM usage of this process in GiB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


# -----------------------------------------------------------------------------
# Dtype helpers (cache only)
# -----------------------------------------------------------------------------


def _normalize_cache_dtype(dtype: Optional[str]) -> Optional[np.dtype]:
    """Normalize a user-provided dtype spec.

    Accepted values
    ---------------
    - None
    - "float16"
    - "float32"

    Returns
    -------
    np.dtype | None
        The normalized numpy dtype, or None meaning "do not cast".
    """

    if dtype is None:
        return None

    s = dtype.strip().lower()
    if s == "float16":
        return np.dtype(np.float16)
    if s == "float32":
        return np.dtype(np.float32)

    raise ValueError(
        f"Unsupported cache dtype={dtype!r}. Expected 'float16' or 'float32' (or null)."
    )


def _cast_window_embeddings_inplace(window: Dict[str, Any], *, dtype: np.dtype) -> None:
    """Cast cached *embedding* arrays in-place.

    This intentionally only touches *embeddings*:
    - token embeddings (window['emb_chunks'])
    - output embeddings (window['output_emb'])

    Native outputs (window['output'] values) are not modified.
    """

    # Token embeddings (ragged list)
    emb_chunks = window.get("emb_chunks")
    if isinstance(emb_chunks, list):
        for i, arr in enumerate(emb_chunks):
            if arr is None:
                continue
            emb_chunks[i] = np.asarray(arr, dtype=dtype)

    # Output embeddings (dict sid -> embedding)
    out_emb = window.get("output_emb")
    if isinstance(out_emb, dict):
        for sid, arr in list(out_emb.items()):
            if arr is None:
                continue
            out_emb[sid] = np.asarray(arr, dtype=dtype)

    # (Defensive) legacy fields if present
    emb_out = window.get("embedded_output")
    if isinstance(emb_out, dict):
        for sid, arr in list(emb_out.items()):
            if arr is None:
                continue
            emb_out[sid] = np.asarray(arr, dtype=dtype)


# -----------------------------------------------------------------------------
# In-RAM dataset of tokenised windows
# -----------------------------------------------------------------------------


class WindowCachedDataset(Dataset):
    """Simple in-RAM dataset of *tokenised windows*."""

    def __init__(self, windows: Iterable[Dict[str, Any]]) -> None:
        self._windows: List[Dict[str, Any]] = list(windows)
        if len(self._windows) == 0:
            logger.warning(
                "[WindowCachedDataset] Created with 0 windows. "
                "Downstream collate / loaders may raise on empty datasets."
            )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._windows[idx]

    @classmethod
    def from_streaming(
        cls,
        streaming_dataset: Sequence[Any],
        max_windows: int | None = None,
        num_workers_cache: int = 0,
        *,
        shuffle_shots: bool = False,
        seed: int = 0,
        dtype: Optional[str] = None,
    ) -> "WindowCachedDataset":
        return materialize_tokenized_split_to_ram(
            streaming_dataset=streaming_dataset,
            max_windows=max_windows,
            num_workers_cache=num_workers_cache,
            shuffle_shots=shuffle_shots,
            seed=seed,
            dtype=dtype,
        )


# -----------------------------------------------------------------------------
# Collate function used *only* for parallel caching
# -----------------------------------------------------------------------------


def _cache_collate_identity(batch):
    """Collate used during caching only (batch_size=1)."""
    return batch[0]


# -----------------------------------------------------------------------------
# Wrapper for multiprocessing caching
# -----------------------------------------------------------------------------


class FlattenedStreamingDataset(Dataset):
    """Wrap a shot-level dataset so __getitem__ returns a *list* of window dicts."""

    def __init__(self, ds: Sequence[Any]) -> None:
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        item = self.ds[idx]
        if item is None:
            return []

        if isinstance(item, dict):
            return [item]

        # Materialise iterables (e.g. generators) so the result is picklable
        return [w for w in item if w is not None]


# -----------------------------------------------------------------------------
# Main caching helper
# -----------------------------------------------------------------------------


def materialize_tokenized_split_to_ram(
    *,
    streaming_dataset: Sequence[Any],
    max_windows: Optional[int] = None,
    num_workers_cache: int = 0,
    shuffle_shots: bool = False,
    seed: int = 0,
    dtype: Optional[str] = None,
) -> WindowCachedDataset:
    """Materialise a *shot-level* dataset adapter into a RAM-backed window dataset."""

    dtype_np = _normalize_cache_dtype(dtype)

    def _iter_windows() -> Iterable[Dict[str, Any]]:
        ds_flat = FlattenedStreamingDataset(streaming_dataset)

        # Multi-worker path: parallelise ds_flat.__getitem__()
        if num_workers_cache > 0:
            gen = None
            if shuffle_shots:
                gen = torch.Generator()
                gen.manual_seed(int(seed))

            # NOTE: prefetch_factor is only valid when num_workers > 0.
            loader = DataLoader(
                ds_flat,
                batch_size=1,
                shuffle=bool(shuffle_shots),
                num_workers=int(num_workers_cache),
                collate_fn=_cache_collate_identity,
                prefetch_factor=1,
                generator=gen,
            )

            for window_list in loader:  # each is List[window_dict]
                for w in window_list:
                    yield w
            return

        # Single-process path: direct indexing (optionally shuffled)
        indices = list(range(len(ds_flat)))
        if shuffle_shots and indices:
            random.Random(int(seed)).shuffle(indices)

        for idx in indices:
            for w in ds_flat[idx]:
                yield w

    flat_windows: List[Dict[str, Any]] = []
    log_interval = 1000

    for n, w in enumerate(_iter_windows(), start=1):
        if dtype_np is not None:
            _cast_window_embeddings_inplace(w, dtype=dtype_np)

        flat_windows.append(w)

        if max_windows is not None and n >= int(max_windows):
            break

        if n % log_interval == 0:
            logger.info(
                "Cached %d windows so far (RAM: %.3f GB)",
                len(flat_windows),
                get_ram_gb(),
            )

    logger.info(
        "Materialised %d tokenised windows (num_workers_cache=%d, shuffle_shots=%s, dtype=%s, Final RAM: %.3f GB)",
        len(flat_windows),
        int(num_workers_cache),
        bool(shuffle_shots),
        str(dtype) if dtype is not None else "none",
        get_ram_gb(),
    )

    return WindowCachedDataset(flat_windows)
