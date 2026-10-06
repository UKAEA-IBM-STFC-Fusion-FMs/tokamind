"""
scripts_mast.mast_utils.gpo

GPO (Generalized Preference Optimization) dataset collection utilities.

This sub-package builds the offline preference dataset needed for GPO/DPO-style
fine-tuning of TokaMind.  The preference learning framing in the continuous
regression setting is:

    (x, y_w_emb, y_l_emb)

where
    x        — the model *prompt*: input-signal tokens + actuator tokens for one window.
                Not stored directly; reconstructed on demand from MAST data using the
                saved collection config (shot_id + window_index are sufficient keys).
    y_w_emb  — the *preferred* response: ground-truth embedding in coefficient space (B, D)
    y_l_emb  — the *dispreferred* response: model prediction in coefficient space (B, D)

Pairs are stored in **embedding (coefficient) space** (schema v3) so the GPO
training loop can operate in the same space as the model's training loss. Schema
v3 also stores compact decoded native-space error diagnostics for pair selection.

The dataset is stored as a collection of NumPy `.npz` shards under the training
run directory, so each model owns its own preference data:

    runs/<run_id>/gpo_pairs/
        metadata.json            — human-readable provenance (task, split, val_fraction, …)
        collection_config.json   — machine-readable reconstruction recipe (schema v3)
        <signal>__shard_000000.npz  — embeddings plus native error diagnostics
        ...

Split convention
----------------
Collection runs over the **train split** by default.  A configurable
``val_fraction`` (default 10%) of windows is held out for GPO validation.
The test split is reserved exclusively for evaluation and must not be used
for pair collection.

Input reconstruction
--------------------
The input context x for each pair can be recovered deterministically by calling
:func:`reconstruct_dataloader`, which reads ``collection_config.json`` and
replays the exact same dataloader (MAST dataset + transform pipeline + embeddings).
Matching pairs to windows is then a lookup by ``(shot_id, window_index)``.

Schema compatibility
--------------------
Schema v1 shards (native-space y_w / y_l arrays, collected with older code)
are supported by the stats visualizer for backwards compatibility but cannot
be used for GPO training. Re-run collection to generate v3 shards when native
pair selection is required.

Public API
----------
GpoPairWriter           — streams (y_w_emb, y_l_emb) pairs to sharded .npz files
collect_gpo_pairs       — drives the forward loop over a dataloader, writing pairs
load_collection_config  — parse and validate collection_config.json
reconstruct_dataloader  — rebuild the collection DataLoader from collection_config.json
GpoPairDataset          — map-style PyTorch Dataset over schema-v2/v3 .npz shards
build_gpo_dataloaders   — convenience factory: returns (train_loader, val_loader)
"""

from .writer import GpoPairWriter
from .collect import collect_gpo_pairs
from .reconstruct import load_collection_config, reconstruct_dataloader
from .dataset import GpoPairDataset, build_gpo_dataloaders

__all__ = [
    "GpoPairWriter",
    "collect_gpo_pairs",
    "load_collection_config",
    "reconstruct_dataloader",
    "GpoPairDataset",
    "build_gpo_dataloaders",
]
