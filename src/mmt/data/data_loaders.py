"""
MMT DataLoader initialization utilities.

This module contains MMT-side helpers to build PyTorch DataLoaders for window-level datasets produced by the MMT
pipeline.

Scope
-----
- This module does NOT build datasets or perform benchmark integration. Dataset construction (shot → windows, metadata,
etc.) is handled upstream by the entrypoint scripts (run_*.py) and benchmark utilities.
- This module focuses on DataLoader concerns:
  - cached (map-style) vs streamed (IterableDataset) handling,
  - deterministic shuffling/seeding,
  - worker RNG initialization,
  - consistent defaults (e.g., pin_memory),
  - tagging loaders with `loader.is_streaming`.

Key behaviour
-------------
- Map-style datasets (cached windows):
    * DataLoader-level `shuffle=True` is allowed.
    * A torch.Generator is used for deterministic shuffling when `seed` is provided.

- IterableDataset (streamed windows):
    * DataLoader-level `shuffle` is forced to False (PyTorch restriction).
    * Any shuffling must be implemented inside the dataset itself.
    * The returned loader is tagged with `loader.is_streaming = True`.

Exports
-------
- initialize_mmt_dataloader

Performance defaults
--------------------
When num_workers > 0, we force:

  * prefetch_factor=1
  * persistent_workers=True

This reduces the number of in-flight batches (and therefore the number of shared storages / file descriptors) and
avoids worker respawns every epoch. These defaults are especially helpful on cluster environments where the default
ulimit for open file descriptors can be low.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset

from mmt.utils.seed import make_worker_seed_fn


# ======================================================================================================================
# DataLoader helpers (window-level, cached/streamed aware)
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def _make_data_generator(seed: int) -> torch.Generator:
    """
    Create a torch.Generator to control DataLoader shuffling deterministically.

    This generator is only used for *map-style* datasets where the DataLoader's `shuffle=True` behaviour is active.
    For IterableDataset (e.g., WindowStreamedDataset), shuffling is handled by the dataset itself and this generator is
    effectively unused.

    Parameters
    ----------
    seed : int

    Returns
    -------
    torch.Generator
        Torch generator instance.

    """
    g = torch.Generator()
    g.manual_seed(int(seed))

    return g


# ======================================================================================================================
# DataLoader init
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def initialize_mmt_dataloader(
    dataset: Dataset | IterableDataset | None,
    collate_fn: Any,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    drop_last: bool = False,
    verbose: bool = False,
    seed: Optional[int] = None,
    pin_memory: Optional[bool] = None,
) -> Optional[DataLoader]:
    """
    Build PyTorch DataLoaders for the MMT pipeline (window-level).

    This function handle both:

      - map-style datasets (e.g., WindowCachedDataset), and
      - IterableDataset (e.g., WindowStreamedDataset).

    Behaviour
    ---------
    - For **map-style datasets**:
        * Concretely, DataLoader-level shuffling is enabled only when `shuffle=True`.
        * A torch.Generator (if `seed` is provided) controls the shuffle order deterministically.

    - For **IterableDataset**:
        * PyTorch forbids `shuffle=True` at DataLoader level, so we always force `shuffle=False` there.
        * Any shuffling is expected to be implemented inside the dataset itself (e.g.,
          `WindowStreamedDataset(shuffle_shots=True, ...)`).

    Parameters
    ----------
    dataset : Dataset | None
        Dataset or IterableDataset instance, or None.

    collate_fn : Any
        Collate function (e.g., MMTCollate) that merges a list of window dicts into a batch suitable for the model.
        Same as the `collate_fn` parameter of `torch.utils.data.dataloader.DataLoader`.

    batch_size : int
        Number of windows per batch.

    num_workers : int
        Number of DataLoader workers.

    shuffle : bool
        shuffle flag:
        - IterableDataset → ignored at DataLoader level (always False);dataset is responsible for shuffling.
        Optional. Default: True.

    drop_last : bool
        Whether to drop the last incomplete batch.
        Optional. Default: False.

    verbose : bool
        If True, prints a short summary.
        Optional. Default: False.

    seed : int
        If provided, used to:
        - create a torch.Generator (for deterministic shuffle on map-style datasets),
        - create a worker_init_fn via `make_worker_seed_fn()` to seed NumPy / Python RNG per worker.
        Optional. Default: None.

    pin_memory : bool
        If None, defaults to `torch.cuda.is_available()`. Otherwise, forwarded to the DataLoader.
        Optional. Default: None.

    Returns
    -------
    Optional[DataLoader]
        None if invalid `dataset`, otherwise Initialized DataLoader instance.

    Notes
    -----
    Each DataLoader now has an attribute:

        loader.is_streaming : bool

    which train/evaluation loops can use to determine dataset type:

      - Cached mode (is_streaming=False)  → full epoch, len(dataloader) is meaningful
      - Streaming mode (is_streaming=True) → IterableDataset, len(dataloader) may not be available

    """

    if verbose:
        print("\n\n---------- MMT DATASET & DATALOADER INITIALIZATION ----------\n")

    # ..................................................................................................................
    # Optional deterministic seeding
    # ..................................................................................................................

    worker_fn = None
    generator = None
    if seed is not None:
        worker_fn = make_worker_seed_fn()
        generator = _make_data_generator(seed=seed)

    # ..................................................................................................................
    # Pin memory default
    # ..................................................................................................................

    pin_memory_bool: bool = pin_memory if pin_memory is not None else torch.cuda.is_available()

    # ..................................................................................................................
    # Build loaders
    # ..................................................................................................................

    if dataset is None:
        return None

    # Detect streaming mode
    is_streaming = isinstance(dataset, IterableDataset)

    # IterableDataset → DataLoader shuffle MUST be False.
    effective_shuffle = (not is_streaming) and bool(shuffle)

    if verbose:
        ds_type = "IterableDataset (STREAMING)" if is_streaming else "Map-style Dataset (CACHED)"
        print(
            f"[MMT] Building DataLoader' "
            f"→ {ds_type}, batch_size={batch_size}, "
            f"shuffle={effective_shuffle}, num_workers={num_workers}"
        )

    # Performance / robustness defaults for multiprocessing.
    # NOTE: PyTorch only accepts these arguments when num_workers > 0.
    dl_kwargs: dict[str, Any] = {}
    if int(num_workers) > 0:
        dl_kwargs["prefetch_factor"] = 1

    # Actual DataLoader creation
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=effective_shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
        worker_init_fn=worker_fn,
        generator=generator,
        pin_memory=pin_memory_bool,
        **dl_kwargs,  # type: ignore[arg-type]
    )

    # Tag loader with streaming info (Option A core)
    loader.is_streaming = is_streaming

    return loader
