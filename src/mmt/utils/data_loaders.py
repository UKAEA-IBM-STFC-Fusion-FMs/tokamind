"""
MMT DataLoader initialization utilities.

This module contains MMT-side helpers to build PyTorch DataLoaders for
window-level datasets produced by the MMT pipeline.

Scope
-----
- This module does NOT build datasets or perform benchmark integration.
  Dataset construction (shot → windows, metadata, etc.) is handled upstream
  by the entrypoint scripts (run_*.py) and benchmark utilities.
- This module focuses on DataLoader concerns:
  - cached (map-style) vs streamed (IterableDataset) handling,
  - deterministic shuffling/seeding,
  - worker RNG initialization,
  - consistent defaults (e.g. pin_memory),
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
- initialize_mmt_dataloaders
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset

from mmt.utils.seed import make_worker_seed_fn


__all__ = ["initialize_mmt_dataloaders"]


# ----------------------------------------------------------------------------- #
# DataLoader helpers (window-level, cached/streamed aware)
# ----------------------------------------------------------------------------- #
def _make_data_generator(seed: int) -> torch.Generator:
    """
    Create a torch.Generator to control DataLoader shuffling deterministically.

    This generator is only used for *map-style* datasets where the
    DataLoader's `shuffle=True` behaviour is active. For IterableDataset
    (e.g. WindowStreamedDataset), shuffling is handled by the dataset
    itself and this generator is effectively unused.
    """
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


# ----------------------------------------------------------------------------- #
# DataLoader init
# ----------------------------------------------------------------------------- #
def initialize_mmt_dataloaders(
    datasets: Mapping[str, Optional[Dataset]],
    collate_fn,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    drop_last: bool = False,
    verbose: bool = False,
    seed: Optional[int] = None,
    pin_memory: Optional[bool] = None,
) -> Dict[str, Optional[DataLoader]]:
    """
    Build PyTorch DataLoaders for the MMT pipeline (window-level).

    This function handle both:

      - map-style datasets (e.g. WindowCachedDataset), and
      - IterableDataset (e.g. WindowStreamedDataset).

    Behaviour
    ---------
    - For **map-style datasets**:
        * `shuffle` is forwarded to the DataLoader.
        * A torch.Generator (if `seed` is provided) controls the shuffle
          order deterministically.

    - For **IterableDataset**:
        * PyTorch forbids `shuffle=True` at DataLoader level, so we always
          force `shuffle=False` there.
        * Any shuffling is expected to be implemented inside the dataset
          itself (e.g. `WindowStreamedDataset(shuffle_shots=True, ...)`).

    Parameters
    ----------
    datasets :
        Mapping from split name (e.g. "train", "val", "test") to a Dataset
        or IterableDataset instance, or None.

    collate_fn :
        Collate function (e.g. MMTCollate) that merges a list of window
        dicts into a batch suitable for the model.

    batch_size : int
        Number of windows per batch.

    num_workers : int
        Number of DataLoader workers.

    shuffle : bool, default True
        Global shuffle flag:
        - Map-style datasets → passed through to DataLoader.
        - IterableDataset → ignored at DataLoader level (always False);
          dataset is responsible for shuffling.

    drop_last : bool, default False
        Whether to drop the last incomplete batch.

    verbose : bool, default False
        If True, prints a short summary for each split.

    seed : int, optional
        If provided, used to:
        - create a torch.Generator (for deterministic shuffle on map-style datasets),
        - create a worker_init_fn via `make_worker_seed_fn()` to seed NumPy /
          Python RNG per worker.

    pin_memory : bool, optional
        If None, defaults to `torch.cuda.is_available()`. Otherwise forwarded
        to the DataLoader.

    Returns
    -------
    dict
        Mapping from split name to DataLoader or None.

    NOTES:
    --------------
    Each DataLoader now has an attribute:

        loader.is_streaming : bool

    which train/evaluation loops can use to apply streaming logic:

      - Cached mode  → full epoch (len(dataloader) meaningful)
      - Streaming    → epoch length controlled by
                       cfg.loader.streaming.batches_per_epoch

    """

    if verbose:
        print("\n\n---------- MMT DATASET & DATALOADER INITIALIZATION ----------\n")

    loaders: Dict[str, Optional[DataLoader]] = {k: None for k in datasets.keys()}

    # ----------------------------------------------------------------------
    # Optional deterministic seeding
    # ----------------------------------------------------------------------
    worker_fn = None
    generator = None
    if seed is not None:
        worker_fn = make_worker_seed_fn()
        generator = _make_data_generator(seed)

    # ----------------------------------------------------------------------
    # Pin memory default
    # ----------------------------------------------------------------------
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # ----------------------------------------------------------------------
    # Build loaders
    # ----------------------------------------------------------------------
    for split, ds in datasets.items():
        if ds is None:
            loaders[split] = None
            if verbose:
                print(f"[MMT] Split '{split}': dataset is None → no DataLoader.")
            continue

        # Detect streaming mode
        is_streaming = isinstance(ds, IterableDataset)

        # IterableDataset → DataLoader shuffle MUST be False
        effective_shuffle = False if is_streaming else bool(shuffle)

        if verbose:
            ds_type = (
                "IterableDataset (STREAMING)"
                if is_streaming
                else "Map-style Dataset (CACHED)"
            )
            print(
                f"[MMT] Building DataLoader for split='{split}' "
                f"→ {ds_type}, batch_size={batch_size}, "
                f"shuffle={effective_shuffle}, num_workers={num_workers}"
            )

        # Actual DataLoader creation
        loader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=effective_shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            worker_init_fn=worker_fn,
            generator=generator,
            pin_memory=pin_memory,
        )

        # Tag loader with streaming info (Option A core)
        loader.is_streaming = is_streaming

        loaders[split] = loader

    return loaders
