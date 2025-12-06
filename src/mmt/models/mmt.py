from __future__ import annotations

from typing import Any, Dict, List, Mapping

import torch
import torch.nn as nn

from mmt.data.signal_spec import SignalSpecRegistry

from .token_encoder import TokenEncoder
from .trunk import Trunk
from .modality_heads import ModalityHead
from .output_adapters import OutputAdapter

import logging

logger = logging.getLogger("mmt.Model")


class MultiModalTransformer(nn.Module):
    """
    MultiModalTransformer: foundation + task model for the open-source MMT pipeline.

    This module implements the final stage of the MMT architecture: a lightweight,
    fully modular transformer that consumes *tokenized embeddings* produced by the
    MMT preprocessing pipeline (Chunk → DropNa → Trim → EmbedChunks → BuildTokens →
    MMTCollate) and generates predictions for all output signals of the task.

    The model architecture has four conceptual blocks:

        TokenEncoder  →  Transformer trunk  →  Modality heads  →  Output adapters

    Each block is intentionally simple and explicitly separated so that:
        • pretraining and finetuning can freeze/unfreeze components independently,
        • researchers can swap individual modules (token logic, trunk type,
          modality heads, adapter shapes, etc.) without touching the others,
        • the model stays fully transparent and easy to understand.

    -------------------------------------------------------------------------------
    1. TokenEncoder
    -------------------------------------------------------------------------------
    The TokenEncoder receives batched, padded per-chunk embeddings produced by
    `MMTCollate`. For each token it:
        • selects the correct projection layer for its signal_id and maps the
          embedding from D_enc(signal) → d_model,
        • adds positional, signal-ID and role embeddings,
        • prepends a learned CLS token (pos=0, role=OUTPUT).

    The resulting tensor has shape:
        tokens : (B, L+1, d_model)

    An attention “keep mask” of shape (B, L+1) marks which tokens are real
    (including CLS). Padding is managed by MMTCollate upstream; all model-level
    token construction is deterministic and free of task-specific logic.

    -------------------------------------------------------------------------------
    2. Transformer trunk
    -------------------------------------------------------------------------------
    A standard PyTorch `nn.TransformerEncoder` (batch_first=True). It processes the
    token sequence and returns a contextualised representation for each token.

    We use `~attn_keep` as the Transformer `src_key_padding_mask` on CPU/CUDA.
    On MPS devices a special workaround is applied: padding-only columns are pruned,
    and the padding mask is dropped to avoid PyTorch nested-tensor limitations.

    The trunk output has shape:
        h : (B, L+1, d_model)

    -------------------------------------------------------------------------------
    3. Modality heads
    -------------------------------------------------------------------------------
    Each modality (e.g. "timeseries", "profile", "video") receives its own small MLP
    mapping the CLS embedding (h_cls) to a modality-specific latent dimension G_mod.

        group_latent[g] = head_g(h_cls)    # shape (B, G_mod)

    These heads represent “shared modality subspaces”, analogous to group-level
    heads in the original MMT implementation.

    -------------------------------------------------------------------------------
    4. Output adapters
    -------------------------------------------------------------------------------
    Every output signal (role="output" in SignalSpec) receives:
        • an output_dim K_t = SignalSpec.embedding_dim,
        • an OutputAdapter: a simple linear or small MLP mapping G_mod → K_t.

        pred[sid] = adapter_sid(group_latent[modality_of_sid])

    This cleanly separates:
        - modality-level representation learning (shared across many signals),
        - per-signal task heads that project onto the correct output space.

    -------------------------------------------------------------------------------
    Input format (from MMTCollate)
    -------------------------------------------------------------------------------
    The forward pass expects a dict containing at least:
        batch["emb"]           : ragged list of per-token embeddings
        batch["pos"]           : LongTensor (B, L)
        batch["id"]            : LongTensor (B, L)  -- physical signal IDs
        batch["role"]          : LongTensor (B, L)
        batch["padding_mask"]  : BoolTensor (B, L)

    All fields come directly from BuildTokensTransform + MMTCollate.
    No raw data, no chunks, and no signal names are handled inside the model.

    -------------------------------------------------------------------------------
    Forward() return structure
    -------------------------------------------------------------------------------
    Returns a dict:

        {
            "h_cls"       : Tensor (B, d_model),
            "group_latent": Dict[str, Tensor(B, G_mod)],
            "pred"        : Dict[int, Tensor(B, K_t)],
        }

    where:
        - h_cls is the pooled representation of the CLS token,
        - group_latent contains one latent per modality,
        - pred maps each output signal_id → its prediction vector.

    Downstream loss functions (MSE, masked MSE, etc.) operate directly on `pred`.

    """

    def __init__(
        self,
        signal_specs: SignalSpecRegistry,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dim_ff: int,
        dropout: float,
        max_positions: int,
        modality_heads_cfg: Mapping[str, Mapping[str, Any]],
        output_adapters_cfg: Mapping[str, Any],
        backbone_activation: str = "relu",
        debug_tokens: bool = False,
        # verbose: bool = False,
    ):
        super().__init__()
        self.signal_specs = signal_specs
        # self.verbose = verbose

        # ------------------------------------------------------------------
        # 1) Token encoder + trunk
        # ------------------------------------------------------------------
        self.tokens = TokenEncoder(
            d_model=d_model,
            signal_specs=signal_specs,
            max_positions=max_positions,
            debug_checks=debug_tokens,
        )
        self.trunk = Trunk(
            d_model=d_model,
            n_heads=n_heads,
            dim_ff=dim_ff,
            n_layers=n_layers,
            dropout=dropout,
            activation=backbone_activation,
        )
        self.trunk_out_dim = d_model

        # ------------------------------------------------------------------
        # 2) Output specs and modality groups
        # ------------------------------------------------------------------
        output_specs = [s for s in signal_specs.specs if s.role == "output"]
        if not output_specs:
            raise ValueError("SignalSpecRegistry contains no outputs (role='output').")
        self.output_specs = output_specs

        self.output2group: Dict[int, str] = {}
        groups = sorted(set(s.modality for s in output_specs))
        for spec in output_specs:
            self.output2group[spec.signal_id] = spec.modality

        # Per-modality head configuration (from model.modality_heads)
        per_mod_hidden: Dict[str, int] = {}
        per_mod_dim: Dict[str, int] = {}
        for g in groups:
            cfg = modality_heads_cfg.get(g)
            if cfg is None:
                raise KeyError(
                    f"No modality head configuration provided for modality={g!r}"
                )
            per_mod_hidden[g] = int(cfg.get("hidden", 0) or 0)
            per_mod_dim[g] = int(cfg.get("out_dim", d_model) or d_model)

        self.group_heads = nn.ModuleDict(
            {
                g: ModalityHead(
                    in_dim=int(self.trunk_out_dim),
                    out_dim=per_mod_dim[g],
                    hidden=per_mod_hidden[g],
                    layers=2,
                )
                for g in groups
            }
        )

        # ------------------------------------------------------------------
        # 3) Per-output adapters (from model.adapters)
        # ------------------------------------------------------------------
        hidden_default = int(output_adapters_cfg.get("hidden_default", 0) or 0)
        hidden_overrides = {
            str(k): int(v)
            for k, v in output_adapters_cfg.get("hidden_overrides", {}).items()
        }

        self.target_adapters = nn.ModuleDict()
        self.output_dims: Dict[int, int] = {}

        for spec in output_specs:
            g = spec.modality
            G_mod = int(self.group_heads[g].out_dim)

            # Target dimension: stored in SignalSpec.embedding_dim
            K_t = getattr(spec, "embedding_dim", None)
            if K_t is None:
                raise AttributeError(
                    "SignalSpec is expected to have an 'embedding_dim' field for outputs. "
                    "Please update SignalSpecRegistry to attach embedding_dim."
                )
            K_t = int(K_t)

            hidden = hidden_overrides.get(spec.name, hidden_default)

            self.target_adapters[str(spec.signal_id)] = OutputAdapter(
                in_dim=G_mod,
                out_dim=K_t,
                hidden=hidden,
            )
            self.output_dims[spec.signal_id] = K_t

        self._print_init_summary(groups, per_mod_hidden, per_mod_dim)

    # ------------------------------------------------------------------
    def _print_init_summary(
        self,
        groups: List[str],
        per_mod_hidden: Dict[str, int],
        per_mod_dim: Dict[str, int],
    ) -> None:
        logger.info("MultiModalTransformer initialised:")
        logger.info(f"  Trunk: d_model={self.trunk_out_dim}")

        logger.info("  Group heads:")
        for g in groups:
            logger.info(
                f"    - {g}: hidden={per_mod_hidden[g]}, out_dim={per_mod_dim[g]}"
            )

        logger.info("  Output adapters:")
        for spec in self.output_specs:
            sid = spec.signal_id
            g = self.output2group[sid]
            logger.info(
                f"    - {spec.name} (id={sid}, group={g}): dim={self.output_dims[sid]}"
            )

    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        batch: output of MMTCollate, with at least:
          * "emb"           : ragged embeddings (List[List[Tensor]])
          * "pos", "id", "role" : LongTensor (B, L)
          * "padding_mask"  : BoolTensor (B, L)
        """
        # 1) Tokens + attention mask (True = keep)
        tokens, attn_keep = self.tokens(batch)  # (B, L+1, d_model)
        src_key_padding_mask = ~attn_keep  # True = PAD (for Transformer)

        # --- MPS workaround (drop mask; keep any-present token columns) ---
        if tokens.device.type == "mps":
            keep_cols = attn_keep.any(dim=0)  # (L+1,)
            if bool((~keep_cols).any()):
                idx = keep_cols.nonzero(as_tuple=False).squeeze(1)
                tokens = tokens.index_select(dim=1, index=idx)
                attn_keep = attn_keep.index_select(dim=1, index=idx)
            src_key_padding_mask = None  # avoid problematic MPS code path

        # 2) Transformer trunk
        h = self.trunk(tokens, src_key_padding_mask=src_key_padding_mask)

        # 3) CLS
        h_cls = h[:, 0, :]  # (B, d_model)

        # 4) Per-modality group heads
        group_latent: Dict[str, torch.Tensor] = {
            g: self.group_heads[g](h_cls) for g in self.group_heads.keys()
        }

        # 5) Per-output adapters
        preds: Dict[int, torch.Tensor] = {}
        for spec in self.output_specs:
            sid = spec.signal_id
            g = self.output2group[sid]
            adapter = self.target_adapters[str(sid)]
            preds[sid] = adapter(group_latent[g])

        return {
            "h_cls": h_cls,
            "group_latent": group_latent,
            "pred": preds,
        }

    # ------------------------------------------------------------------
    # Checkpoint I/O — three explicit parts: trunk, group_heads, target_adapters
    # ------------------------------------------------------------------
    def get_trunk_state_dict(self) -> dict:
        return self.trunk.state_dict()

    def load_trunk_state_dict(self, state: dict, strict: bool = True):
        missing, unexpected = self.trunk.load_state_dict(state, strict=strict)
        return missing, unexpected

    def get_group_heads_state_dict(self) -> dict:
        return self.group_heads.state_dict()

    def load_group_heads_state_dict(self, state: dict, strict: bool = True):
        missing, unexpected = self.group_heads.load_state_dict(state, strict=strict)
        return missing, unexpected

    def get_target_adapters_state_dict(self) -> dict:
        return self.target_adapters.state_dict()

    def load_target_adapters_state_dict(self, state: dict, strict: bool = True):
        missing, unexpected = self.target_adapters.load_state_dict(state, strict=strict)
        return missing, unexpected

    # ------------------------------------------------------------------
    # Freezing helpers — explicit and independent
    # ------------------------------------------------------------------
    def freeze_trunk(self) -> None:
        for p in self.trunk.parameters():
            p.requires_grad = False

    def unfreeze_trunk(self) -> None:
        for p in self.trunk.parameters():
            p.requires_grad = True

    def freeze_group_heads(self) -> None:
        for p in self.group_heads.parameters():
            p.requires_grad = False

    def unfreeze_group_heads(self) -> None:
        for p in self.group_heads.parameters():
            p.requires_grad = True

    def freeze_target_adapters(self) -> None:
        for p in self.target_adapters.parameters():
            p.requires_grad = False

    def unfreeze_target_adapters(self) -> None:
        for p in self.target_adapters.parameters():
            p.requires_grad = True
