from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from mmt.data.signal_spec import SignalSpecRegistry
from mmt.data.collate import ROLE_OUTPUT


class TokenEncoder(nn.Module):
    """
    Build transformer tokens from precomputed per-signal embeddings.

    Responsibilities
    ----------------
    * Project each per-signal embedding emb[b][t] (D_i,) to d_model using a
      per-signal Linear.
    * Add learned positional, signal-id and role embeddings.
    * Prepend a CLS token at position 0.

    Assumptions (provided by MMTCollate)
    ------------------------------------
    batch must contain:
      * "emb"  : List[List[Tensor]]; emb[b][t] is (D_i,)
      * "pos"  : LongTensor (B, L), pos >= 1 for real tokens
      * "id"   : LongTensor (B, L) in [0, num_signals-1]
      * "role" : LongTensor (B, L)
      * "padding_mask": BoolTensor (B, L), True where there is a real token

    Returns
    -------
    tokens    : Tensor (B, L+1, d_model) with CLS at position 0
    attn_keep : BoolTensor (B, L+1), True where token is real (incl. CLS).
                To build src_key_padding_mask for TransformerEncoder, use ~attn_keep.
    """

    def __init__(
        self,
        d_model: int,
        signal_specs: SignalSpecRegistry,
        max_positions: int,
        debug_checks: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.max_positions = int(max_positions)
        self.debug_checks = bool(debug_checks)

        self.num_signals = signal_specs.num_signals

        # One Linear per signal id; lazily created on first use
        self._proj_by_sid = nn.ModuleDict()
        self._in_dim_by_sid: Dict[str, int] = {}

        # CLS content vector
        self.cls_content = nn.Parameter(torch.randn(self.d_model))

        # Embeddings:
        # - pos: 0..max_positions (0 reserved for CLS)
        # - id : 0..num_signals (num_signals reserved for CLS id)
        # - role: {0,1,2} = {context, actuator, output}
        self.pos_embed = nn.Embedding(self.max_positions + 1, self.d_model)
        self.id_embed = nn.Embedding(self.num_signals + 1, self.d_model)
        self.role_embed = nn.Embedding(3, self.d_model)

        # CLS id index
        self.cls_id = self.num_signals

    # ------------------------------------------------------------------
    def _get_proj(self, sid: int, in_dim: int, device: torch.device) -> nn.Linear:
        key = str(int(sid))
        if key not in self._proj_by_sid:
            layer = nn.Linear(in_dim, self.d_model)
            self._proj_by_sid[key] = layer.to(device)
            self._in_dim_by_sid[key] = in_dim
        else:
            if self._in_dim_by_sid[key] != in_dim:
                raise ValueError(
                    f"Signal id {sid} was first seen with in_dim={self._in_dim_by_sid[key]}, "
                    f"but now has in_dim={in_dim}."
                )
        return self._proj_by_sid[key]

    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> (torch.Tensor, torch.Tensor):
        emb = batch["emb"]
        pos = batch["pos"]
        sid = batch["id"]
        role = batch["role"]
        padding_mask = batch["padding_mask"]

        if not isinstance(emb, list):
            raise TypeError("batch['emb'] must be a list of lists of tensors.")

        B, L = pos.shape
        device = pos.device

        if self.debug_checks:
            if (
                sid.shape != (B, L)
                or role.shape != (B, L)
                or padding_mask.shape != (B, L)
            ):
                raise ValueError("id, role, padding_mask must all have shape (B, L).")
            if pos.min().item() < 1:
                raise ValueError("pos must be >= 1 for real tokens; 0 is CLS.")
            if pos.max().item() > self.max_positions:
                raise ValueError(
                    f"pos contains values > max_positions ({self.max_positions}). "
                    "Increase trim_chunks.max_chunks or adjust preprocessing."
                )

        # Content projection
        tokens = torch.zeros(B, L, self.d_model, dtype=torch.float32, device=device)

        for b in range(B):
            row = emb[b]
            if len(row) < L and self.debug_checks:
                raise ValueError(
                    f"batch['emb'][{b}] has length {len(row)}, expected at least L={L}."
                )
            for t in range(L):
                if not padding_mask[b, t]:
                    continue  # padding position
                vec = row[t]
                if not isinstance(vec, torch.Tensor):
                    raise TypeError("emb[b][t] must be a torch.Tensor after collate.")
                if vec.ndim != 1:
                    raise ValueError(
                        f"emb[{b}][{t}] must be 1D (D_i,), got shape {tuple(vec.shape)}."
                    )

                in_dim = int(vec.shape[0])
                proj = self._get_proj(int(sid[b, t].item()), in_dim, device=device)
                tokens[b, t] = proj(vec.to(device=device, dtype=torch.float32))

        # CLS token content
        cls = self.cls_content.view(1, 1, -1).expand(B, 1, -1)

        # Concatenate CLS + tokens
        tokens = torch.cat([cls, tokens], dim=1)  # (B, L+1, d_model)

        # Metadata embeddings: CLS + originals
        pos_cls = torch.zeros(B, 1, dtype=torch.long, device=device)
        sid_cls = torch.full((B, 1), self.cls_id, dtype=torch.long, device=device)
        role_cls = torch.full((B, 1), ROLE_OUTPUT, dtype=torch.long, device=device)

        pos_full = torch.cat([pos_cls, pos], dim=1)  # (B, L+1)
        sid_full = torch.cat([sid_cls, sid], dim=1)
        role_full = torch.cat([role_cls, role], dim=1)

        tokens = (
            tokens
            + self.pos_embed(pos_full)
            + self.id_embed(sid_full)
            + self.role_embed(role_full)
        )

        # Attention "keep" mask: CLS always kept
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        attn_keep = torch.cat([cls_mask, padding_mask], dim=1)  # (B, L+1)

        return tokens, attn_keep
