"""
Token encoder for MMT.

The TokenEncoder converts ragged per-token embeddings produced by the data
pipeline (BuildTokensTransform + MMTCollate) into a dense token tensor of shape
(B, L+1, d_model) suitable for the transformer backbone.

It:
- projects each token embedding into d_model using per-signal projection layers,
- adds positional, signal-id, modality, and role embeddings,
- prepends a learned CLS token,
- returns an attention-keep mask aligned with the padded token sequence.

Projection layers are keyed by stable canonical keys ("role:name") to keep
checkpoint loading and warm-start robust to signal ordering.
"""

from __future__ import annotations

from typing import Any, Dict, cast

import torch
import torch.nn as nn

from mmt.data.signal_spec import SignalSpecRegistry


class TokenEncoder(nn.Module):
    """
    TokenEncoder
    =============

    Projects per-signal embeddings (one vector per token) into a shared
    `d_model`, adds metadata embeddings (pos/id/mod/role), and prepends
    a learned CLS token.

    -------------------------------------------------------------------
    Projection layers and canonical keys
    -------------------------------------------------------------------
    Per-signal projection layers are **not keyed by signal_id**. Instead
    each role-specific signal uses a **canonical name**:

        canonical_key = f"{role}:{name}"

    Examples:
        "input:pf_active-coil_current"
        "actuator:summary-power_nbi"
        "output:pf_active-coil_current"

    This ensures:
      • different roles of the same physical signal get different
        projection layers,
      • projection dimensions remain consistent,
      • checkpoint loading (resume/warm-start) is stable and predictable,
      • ordering of signals in the registry does not affect model state.

    All projection layers are created **eagerly in __init__** using each
    SignalSpec's encoder output dimension. This guarantees that strict
    checkpoint loading works (no “unexpected key” errors).

    -------------------------------------------------------------------
    Batch structure expected from MMTCollate
    -------------------------------------------------------------------
    batch = {
        "emb":  List[List[Tensor]]        # emb[b][t] has shape (D_i,)
        "pos":  LongTensor(B, L)
        "id":   LongTensor(B, L)          # physical signal id
        "mod":  LongTensor(B, L)          # modality id
        "role": LongTensor(B, L)          # 0=input, 1=actuator, 2=output
        "padding_mask": BoolTensor(B, L)
    }

    -------------------------------------------------------------------
    Outputs
    -------------------------------------------------------------------
    tokens:     Tensor(B, L+1, d_model)
        CLS token at position 0, followed by projected & embedded tokens.

    attn_keep:  BoolTensor(B, L+1)
        True where tokens are real (including CLS), used as attention mask.

    -------------------------------------------------------------------
    Additional notes
    -------------------------------------------------------------------
    • TokenEncoder is a first-class model block: it is saved/loaded
      separately in checkpoints and supports warm-start.

    • Device semantics follow the model: projection layers are moved to
      the model device via `model.to(device)`, and input vectors are
      cast to the correct device automatically in forward().
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
        self.num_modalities = len(signal_specs.modalities)

        # ------------------------------------------------------------------
        # Stable mapping from signal_id → canonical key "role:name"
        # ------------------------------------------------------------------
        self.sid_to_key: Dict[int, str] = {}
        for spec in signal_specs.specs:
            self.sid_to_key[spec.signal_id] = spec.canonical_key

        # ------------------------------------------------------------------
        # Per-signal projection layers (keyed by canonical name)
        # Pre-create all of them so that strict checkpoint loading works.
        # ------------------------------------------------------------------
        self.proj_layers = nn.ModuleDict()
        self.proj_in_dim: Dict[str, int] = {}

        # Only signals that actually become tokens need projection layers.
        # In this architecture, outputs are NOT tokenized
        # (they are predicted via output_adapters),
        # so we intentionally skip role="output" here.
        TOKEN_ROLES = {"input", "actuator"}

        for spec in signal_specs.specs:
            if spec.role not in TOKEN_ROLES:
                continue  # skip outputs (and any future non-token roles)

            key = spec.canonical_key
            in_dim = int(spec.embedding_dim)
            if in_dim <= 0:
                continue

            if key in self.proj_layers:
                # Same canonical key across roles is not expected; if you ever
                # share projections manually, you can handle it here.
                continue

            self.proj_layers[key] = nn.Linear(in_dim, self.d_model)
            self.proj_in_dim[key] = in_dim

        # ------------------------------------------------------------------
        # CLS token parameters
        # ------------------------------------------------------------------
        self.cls_content = nn.Parameter(torch.randn(self.d_model))
        self.cls_id = self.num_signals  # CLS occupies id=num_signals

        # ------------------------------------------------------------------
        # Metadata embeddings
        # ------------------------------------------------------------------
        self.pos_embed = nn.Embedding(self.max_positions + 1, self.d_model)
        self.id_embed = nn.Embedding(self.num_signals + 1, self.d_model)
        self.mod_embed = nn.Embedding(self.num_modalities, self.d_model)
        self.role_embed = nn.Embedding(3, self.d_model)

        # ------------------------------------------------------------------

    def _get_proj(
        self,
        canonical_key: str,
        in_dim: int,
    ) -> nn.Linear:
        """
        Return the projection layer associated with a role-specific signal.

        All projection layers are created in __init__ using the encoder
        output dimension (embedding_dim) for each signal. Here we only
        check that the incoming vectors have a consistent size.
        """
        if in_dim <= 0:
            raise RuntimeError(
                f"TokenEncoder._get_proj called with invalid in_dim={in_dim}."
            )

        if canonical_key not in self.proj_layers:
            raise KeyError(
                f"No projection layer registered for canonical key '{canonical_key}'."
            )

        if self.proj_in_dim[canonical_key] != in_dim:
            raise ValueError(
                f"Projection for '{canonical_key}' was first created with "
                f"in_dim={self.proj_in_dim[canonical_key]}, now sees in_dim={in_dim}."
            )

        return cast(nn.Linear, self.proj_layers[canonical_key])

    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        emb = batch["emb"]
        pos = batch["pos"]
        sid = batch["id"]
        mod = batch["mod"]
        role = batch["role"]
        padding_mask = batch["padding_mask"]

        if not isinstance(emb, list):
            raise TypeError("batch['emb'] must be a list-of-lists of tensors.")

        B, L = pos.shape
        device = pos.device

        # ---------------------------------------------------------------
        # Project token content
        # ---------------------------------------------------------------
        tokens = torch.zeros(B, L, self.d_model, dtype=torch.float32, device=device)

        for b in range(B):
            for t in range(L):
                if not padding_mask[b, t]:
                    continue

                vec = emb[b][t]
                if not isinstance(vec, torch.Tensor):
                    raise TypeError("emb[b][t] must be a torch.Tensor.")

                if vec.ndim != 1:
                    raise ValueError(
                        f"emb[{b}][{t}] must be a 1D vector, got shape={tuple(vec.shape)}."
                    )

                in_dim = vec.shape[0]
                sid_bt = int(sid[b, t].item())

                if sid_bt >= 0 and in_dim > 0:
                    canonical = self.sid_to_key.get(sid_bt)
                    if canonical is None:
                        raise KeyError(
                            f"No canonical key found for signal_id={sid_bt}."
                        )

                    proj = self._get_proj(canonical, in_dim)
                    tokens[b, t] = proj(
                        vec.to(device=tokens.device, dtype=torch.float32)
                    )

        # ---------------------------------------------------------------
        # CLS token
        # ---------------------------------------------------------------
        cls_tok = self.cls_content.view(1, 1, -1).expand(B, 1, -1)
        tokens = torch.cat([cls_tok, tokens], dim=1)

        # ---------------------------------------------------------------
        # Metadata embeddings
        # ---------------------------------------------------------------
        pos_cls = torch.zeros(B, 1, dtype=torch.long, device=device)
        sid_cls = torch.full((B, 1), self.cls_id, dtype=torch.long, device=device)

        pos_full = torch.cat([pos_cls, pos], dim=1)  # (B, L+1)
        sid_full = torch.cat([sid_cls, sid], dim=1)  # (B, L+1)

        tokens = tokens + self.pos_embed(pos_full) + self.id_embed(sid_full)

        # Non-CLS tokens also get modality and role embeddings
        tokens[:, 1:, :] += self.mod_embed(mod) + self.role_embed(role)

        # ---------------------------------------------------------------
        # Attention mask
        # ---------------------------------------------------------------
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        attn_keep = torch.cat([cls_mask, padding_mask], dim=1)

        return tokens, attn_keep
