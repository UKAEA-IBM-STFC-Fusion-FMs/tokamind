"""
Embedding resolution for pretrain, finetune, and eval phases (MAST integration).

Package layout:
- signals.py   : signal selection / filter / set helpers
- policy.py    : profile detection and encoder tune/source policy
- artifacts.py : load / stage / merge artifacts + inherited-artifact validation
- resolve.py   : pretrain / finetune / eval orchestration

Public API is re-exported here so ``from mast_utils.embedding_resolution import ...`` keeps working.
"""

from .artifacts import stage_task_used_dct3d_artifacts_from_source
from .resolve import (
    save_config_snapshot,
    resolve_pretrain_embeddings,
    resolve_finetune_embeddings,
    resolve_eval_embeddings,
)

__all__ = [
    "resolve_pretrain_embeddings",
    "resolve_finetune_embeddings",
    "resolve_eval_embeddings",
    "save_config_snapshot",
    "stage_task_used_dct3d_artifacts_from_source",
]
