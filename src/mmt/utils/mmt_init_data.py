from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset

from mmt.utils.seed import make_worker_seed_fn

# We still rely on the baseline TaskModelTransformWrapper to define
# how a "shot" is turned into an iterable of windows.
from scripts.pipelines.utils.utils import TaskModelTransformWrapper

__all__ = ["initialize_mmt_datasets", "initialize_mmt_dataloaders"]


# ----------------------------------------------------------------------------- #
# 1. MMT model-level datasets (shot -> windows via TaskModelTransformWrapper)
# ----------------------------------------------------------------------------- #
def initialize_mmt_datasets(
    datasets_train_val_test: Mapping[str, Optional[Dataset]],
    dict_metadata: Mapping[str, Any],
    config_task: Mapping[str, Any],
    model_specific_transform=None,
    verbose: bool = False,
) -> Dict[str, Optional[TaskModelTransformWrapper]]:
    """
    Wrap baseline datasets with TaskModelTransformWrapper for the MMT pipeline.

    This function is the MMT-side counterpart of the baseline
    `initialize_model_datasets`, but lives in the MMT repo so that:

      - The baseline repo remains the source of truth for task/data semantics
        (MastDataset, TaskModelTransformWrapper, metadata).
      - The MMT repo owns how model-specific transforms are composed and used.

    Parameters
    ----------
    datasets_train_val_test :
        Mapping with keys typically ``"train"``, ``"val"``, ``"test"``,
        whose values are baseline shot-level datasets (e.g. MastDataset)
        or ``None``.

    dict_metadata :
        Metadata dictionary produced by the baseline pipeline, containing
        per-signal information (dt, shapes, etc.) used by the wrapper.

    config_task :
        Baseline task configuration dict (usually `cfg_task`) containing
        the `task_window_segmenter` section (input/actuator/output keys,
        window lengths, delta, etc.).

    model_specific_transform :
        Optional callable implementing the MMT model-specific pipeline,
        typically a `ComposeTransforms([...])` of:

            ChunkWindowsTransform
            → SelectValidWindowsTransform
            → TrimChunksTransform
            → EmbedChunksTransform
            → BuildTokensTransform

        If None, the raw windows from TaskModelTransformWrapper are returned.

    verbose : bool, default False
        If True, prints basic information about the wrapped datasets.

    Returns
    -------
    dict
        A dict with the same keys as `datasets_train_val_test`, where each
        value is a `TaskModelTransformWrapper` (shot-level dataset) or None.

    Notes
    -----
    - The resulting datasets are *shot-based*: `__getitem__(idx)` yields a
      generator or iterable of windows for that shot.
    - Window-level behaviour (cached vs streamed) is handled downstream
      by WindowCachedDataset / WindowStreamedDataset.
    """
    datasets_mmt: Dict[str, Optional[TaskModelTransformWrapper]] = {
        "train": None,
        "val": None,
        "test": None,
    }

    # Train
    base_train = datasets_train_val_test.get("train")
    if base_train is not None:
        datasets_mmt["train"] = TaskModelTransformWrapper(
            base_train,
            dict_metadata,
            config_task,
            model_specific_transform,
            verbose=verbose,
        )
        if verbose:
            print(f"[MMT] len(mast_train_dataset): {len(datasets_mmt['train'])}")

    # Val
    base_val = datasets_train_val_test.get("val")
    if base_val is not None:
        datasets_mmt["val"] = TaskModelTransformWrapper(
            base_val,
            dict_metadata,
            config_task,
            model_specific_transform,
            verbose=False,
        )
        if verbose:
            print(f"[MMT] len(mast_val_dataset): {len(datasets_mmt['val'])}")

    # Test
    base_test = datasets_train_val_test.get("test")
    if base_test is not None:
        datasets_mmt["test"] = TaskModelTransformWrapper(
            base_test,
            dict_metadata,
            config_task,
            model_specific_transform,
            verbose=False,
        )
        if verbose:
            print(f"[MMT] len(mast_test_dataset): {len(datasets_mmt['test'])}")

    return datasets_mmt


# ----------------------------------------------------------------------------- #
# 2. DataLoader helpers (window-level, cached/streamed aware)
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

    This function is the MMT-specific analogue of the baseline
    `initialize_dataloaders`, extended to correctly handle both:

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
    """
    if verbose:
        print("\n\n---------- MMT DATASET & DATALOADER INITIALIZATION ----------\n")

    loaders: Dict[str, Optional[DataLoader]] = {k: None for k in datasets.keys()}

    # Seeding helpers (optional)
    worker_fn = None
    generator = None
    if seed is not None:
        worker_fn = make_worker_seed_fn()
        generator = _make_data_generator(seed)

    # Default pin_memory
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    for split, ds in datasets.items():
        if ds is None:
            loaders[split] = None
            if verbose:
                print(f"[MMT] Split '{split}': dataset is None → no DataLoader.")
            continue

        is_iterable = isinstance(ds, IterableDataset)
        effective_shuffle = False if is_iterable else bool(shuffle)

        if verbose:
            ds_type = "IterableDataset" if is_iterable else "Map-style Dataset"
            print(
                f"[MMT] Building DataLoader for split='{split}' "
                f"({ds_type}), shuffle={effective_shuffle}, "
                f"batch_size={batch_size}, num_workers={num_workers}"
            )

        loaders[split] = DataLoader(
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

    return loaders
