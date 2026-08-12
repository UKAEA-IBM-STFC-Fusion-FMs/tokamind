"""
Token encoder for MMT.

The TokenEncoder converts packed token embeddings produced by the data pipeline (BuildTokensTransform + MMTCollate)
into a dense token tensor of shape (B, L+1, d_model) suitable for the transformer backbone.

It:
- projects each token embedding into d_model using per-signal projection layers,
- adds positional, signal-id, modality, and role embeddings,
- prepends a learned CLS token,
- returns an attention-keep mask aligned with the padded token sequence.

Projection layers are keyed by stable canonical keys ("role:name") to keep checkpoint loading and warm-start robust to
signal ordering.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

from mmt.data.signal_spec import SignalSpecRegistry


# ======================================================================================================================
class TokenEncoder(nn.Module):
    """
    TokenEncoder
    =============

    Projects per-signal embeddings (one vector per token) into a shared `d_model`, adds metadata embeddings
    (pos/ID/mod/role), and prepends a learned CLS token.

    --------------------------------------------------------------------------------------------------------------------
    Projection layers and canonical keys
    --------------------------------------------------------------------------------------------------------------------

    Per-signal projection layers are **not keyed by signal_id**. Instead, each role-specific signal uses a **canonical
    name**:

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

    All projection layers are created **eagerly in __init__** using each SignalSpec's encoder output dimension. This
    guarantees that strict checkpoint loading works (no “unexpected key” errors).

    --------------------------------------------------------------------------------------------------------------------
    Batch structure expected from MMTCollate
    --------------------------------------------------------------------------------------------------------------------

    batch = {
        "emb":       dict[int, Tensor]         # emb[sid] has shape (N_sid, D_sid)
        "emb_index": dict[int, LongTensor]     # flat indices (b * L + t), shape (N_sid,)
        "pos":       LongTensor(B, L)
        "id":        LongTensor(B, L)          # signal ID (PAD_ID = -1)
        "mod":       LongTensor(B, L)          # modality ID (PAD_MOD = -1)
        "role":      LongTensor(B, L)          # 0=input, 1=actuator, 2=output (PAD_ROLE = -1)
        "padding_mask": BoolTensor(B, L)
    }

    --------------------------------------------------------------------------------------------------------------------
    Outputs
    --------------------------------------------------------------------------------------------------------------------

    tokens:     Tensor(B, L+1, d_model)
        CLS token at position 0, followed by projected & embedded tokens.

    attn_keep:  BoolTensor(B, L+1)
        True where tokens are real (including CLS), used as attention mask.

    --------------------------------------------------------------------------------------------------------------------
    Additional notes
    --------------------------------------------------------------------------------------------------------------------

    • TokenEncoder is a first-class model block: it is saved/loaded separately in checkpoints and supports warm-start.

    • Device semantics follow the model: projection layers are moved to the model device via `model.to(device)`, and
      input vectors are cast to the correct device automatically in forward().

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        d_model: int,
        signal_specs: SignalSpecRegistry,
        max_positions: int,
        debug_checks: bool = False,
    ):
        """
        Initialize class attributes.

        Parameters
        ----------
        d_model : int
            The number of expected features in the input.
        signal_specs : SignalSpecRegistry
            Registry with one spec per signal (name, role, modality, encoder, embedding_dim).
        max_positions : int
            Maximum number of temporal positions for positional embeddings.
        debug_checks : bool
            Whether to perform debugging checks.
            Optional. Default: False.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking mistakes.

        """

        super().__init__()
        self.d_model = int(d_model)
        self.max_positions = int(max_positions)
        self.debug_checks = bool(debug_checks)

        self.num_signals = signal_specs.num_signals
        self.num_modalities = len(signal_specs.modalities)

        # ..............................................................................................................
        # Stable mapping from signal_id → canonical key "role:name"
        # ..............................................................................................................

        self.sid_to_key: dict[int, str] = {}
        for spec in signal_specs.specs:
            self.sid_to_key[spec.signal_id] = spec.canonical_key

        # ..............................................................................................................
        # Per-signal projection layers (keyed by canonical name).
        # Pre-create all of them so that strict checkpoint loading works.
        # ..............................................................................................................

        self.proj_layers = nn.ModuleDict()
        self.proj_in_dim: dict[str, int] = {}

        # Only signals that actually become tokens need projection layers.
        # In this architecture, outputs are NOT tokenized (they are predicted via output_adapters), so we intentionally
        # skip role="output" here.
        TOKEN_ROLES = ["input", "actuator"]  # noqa - Ignore lowercase warning

        for spec in signal_specs.specs:
            if spec.role not in TOKEN_ROLES:
                continue  # Skip outputs (and any future non-token roles)

            key = spec.canonical_key
            in_dim = int(spec.embedding_dim)
            if in_dim <= 0:
                continue

            if key in self.proj_layers:
                # Same canonical key across roles is not expected; if you ever share projections manually, you can
                # handle it here.
                continue

            self.proj_layers[key] = nn.Linear(in_features=in_dim, out_features=self.d_model)
            self.proj_in_dim[key] = in_dim

        # ..............................................................................................................
        # CLS token parameters
        # ..............................................................................................................

        self.cls_content = nn.Parameter(torch.randn(self.d_model))
        self.cls_id = self.num_signals  # CLS occupies id=num_signals

        # ..............................................................................................................
        # Metadata embeddings
        # ..............................................................................................................

        self.pos_embed = nn.Embedding(num_embeddings=self.max_positions + 1, embedding_dim=self.d_model)
        self.id_embed = nn.Embedding(num_embeddings=self.num_signals + 1, embedding_dim=self.d_model)
        self.mod_embed = nn.Embedding(num_embeddings=self.num_modalities, embedding_dim=self.d_model)
        self.role_embed = nn.Embedding(num_embeddings=3, embedding_dim=self.d_model)

    # ------------------------------------------------------------------------------------------------------------------
    def _get_proj(self, canonical_key: str, in_dim: int) -> nn.Linear:
        """
        Return the projection layer associated with a role-specific signal.

        All projection layers are created in __init__ using the encoder output dimension (embedding_dim) for each
        signal. Here we only check that the incoming vectors have a consistent size.

        Parameters
        ----------
        canonical_key : str
            Canonical key of target projection layer.
        in_dim : int
            Input dimension.

        Returns
        -------
        nn.Linear
            Projection layer corresponding to the provided `canonical_key`.

        Raises
        ------
        RuntimeError
            If invalid `in_dim` is provided.
        KeyError
            If no projection layer is registered for 'canonical_key'.
        ValueError
            If `self.proj_in_dim[canonical_key]` does not match `in_dim`.

        """

        if in_dim <= 0:
            raise RuntimeError(f"TokenEncoder._get_proj called with invalid `in_dim={in_dim}`.")

        if canonical_key not in self.proj_layers:
            raise KeyError(f"No projection layer registered for canonical key '{canonical_key}'.")

        if self.proj_in_dim[canonical_key] != in_dim:
            raise ValueError(
                f"Projection for '{canonical_key}' was first created with "
                f"in_dim={self.proj_in_dim[canonical_key]}, now sees in_dim={in_dim}."
            )

        return cast(nn.Linear, self.proj_layers[canonical_key])

    # ------------------------------------------------------------------------------------------------------------------
    def forward(  # NOSONAR - Ignore cognitive complexity
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        TokenEncoder's forward function.

        Parameters
        ----------
        batch : dict[str, Any]
            Input batch used for token generation.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Resulting tokens along with attention keep mask.

        Raises
        ------
        TypeError
            If `batch["emb"]` is not of type dict[int, Tensor].
            If `batch["emb_index"]` is not of type dict[int, LongTensor].
            If an item of `batch["emb"]` is not of type a torch.Tensor.
        ValueError
            If an item of `batch["emb"]` is not 2D, i.e., (N_sid, D_sid).
            If an item of `batch["emb_index"]` is not 1D, i.e., (N_sid,).
            If row mismatch for an item of `batch["emb_index"]`.
        KeyError
            If `batch["emb_index"]` misses required key.
            If no canonical key found for an item of `batch["emb_index"]`.
            If no projection layer found for an item of `batch["emb_index"]`.

        """

        emb = batch["emb"]
        emb_index = batch["emb_index"]
        pos = batch["pos"]
        sid = batch["id"]
        mod = batch["mod"]
        role = batch["role"]
        padding_mask = batch["padding_mask"]

        if not isinstance(emb, dict):
            raise TypeError("`batch['emb']` must be a dict[int, Tensor] (packed by sid).")

        if not isinstance(emb_index, dict):
            raise TypeError("`batch['emb_index']` must be a dict[int, LongTensor].")

        B, L = pos.shape
        device = pos.device

        # ..............................................................................................................
        # Project token content (scatter by sid → dense (B, L, d_model))
        # ..............................................................................................................

        tokens_flat = torch.zeros(B * L, self.d_model, dtype=torch.float32, device=device)

        for sid_i, vecs in emb.items():
            if not isinstance(vecs, torch.Tensor):
                raise TypeError(f"`batch['emb'][{sid_i!r}]` must be a torch.Tensor, got {type(vecs)}.")

            if vecs.ndim != 2:
                raise ValueError(f"`batch['emb'][{sid_i!r}]` must be 2D (N_sid, D_sid), got shape={tuple(vecs.shape)}.")

            idx = emb_index.get(sid_i)
            if idx is None:
                raise KeyError(f"Missing `batch['emb_index'][sid]` for sid={sid_i}.")

            if not isinstance(idx, torch.Tensor):
                idx = torch.as_tensor(idx, dtype=torch.long, device=device)
            else:
                idx = idx.to(device=device, dtype=torch.long)

            if idx.ndim != 1:
                raise ValueError(f"`batch['emb_index'][{sid_i!r}]` must be 1D (N_sid,), got shape={tuple(idx.shape)}.")

            if vecs.shape[0] != idx.shape[0]:
                raise ValueError(
                    f"Row mismatch for sid={sid_i}: emb has N={vecs.shape[0]} rows, "
                    f"emb_index has N={idx.shape[0]} indices."
                )

            canonical = self.sid_to_key.get(int(sid_i))
            if canonical is None:
                raise KeyError(f"No canonical key found for signal_id={sid_i}.")

            if canonical not in self.proj_layers:
                raise KeyError(
                    f"No projection layer found for canonical_key={canonical!r}. "
                    "Only token roles (input/actuator) should appear in batch['emb']."
                )

            proj = self._get_proj(canonical, int(vecs.shape[1]))
            # NOTE: under AMP autocast on CUDA, `proj(...)` may return bf16 even if we pass float32 inputs.
            # `index_copy_` requires the same dtype on both sides, so we cast explicitly to the token buffer dtype.
            y = proj(vecs.to(device=device, dtype=torch.float32))
            if y.dtype != tokens_flat.dtype:
                y = y.to(dtype=tokens_flat.dtype)
            tokens_flat.index_copy_(0, idx, y)

        tokens = tokens_flat.view(B, L, self.d_model)

        # ..............................................................................................................
        # CLS token
        # ..............................................................................................................

        cls_tok = self.cls_content.view(1, 1, -1).expand(B, 1, -1)
        tokens = torch.cat([cls_tok, tokens], dim=1)  # (B, L+1, d_model)

        # ..............................................................................................................
        # Attention keep mask (exclude PAD/dropped tokens)
        # ..............................................................................................................

        keep = padding_mask.to(dtype=torch.bool) & (sid >= 0)
        cls_keep = torch.ones(B, 1, dtype=torch.bool, device=device)
        attn_keep = torch.cat([cls_keep, keep], dim=1)  # (B, L+1)

        # ..............................................................................................................
        # Metadata embeddings (clamped indices, then masked by `keep`)
        # ..............................................................................................................

        pos_cls = torch.zeros(B, 1, dtype=torch.long, device=device)
        sid_cls = torch.full((B, 1), self.cls_id, dtype=torch.long, device=device)

        pos_safe = pos.clamp(min=0, max=self.max_positions)
        sid_safe = sid.clamp(min=0, max=self.num_signals)

        # For PAD slots we clamp to 0, then mask out below.
        max_mod = max(self.num_modalities - 1, 0)
        mod_safe = mod.clamp(min=0, max=max_mod)
        role_safe = role.clamp(min=0, max=2)

        pos_full = torch.cat([pos_cls, pos_safe], dim=1)
        sid_full = torch.cat([sid_cls, sid_safe], dim=1)

        tokens = tokens + self.pos_embed(pos_full) + self.id_embed(sid_full)
        tokens[:, 1:, :] += self.mod_embed(mod_safe) + self.role_embed(role_safe)

        # Mask out PAD/dropped token vectors (non-CLS positions only).
        tokens[:, 1:, :] = tokens[:, 1:, :] * keep.unsqueeze(-1).to(tokens.dtype)

        # ..............................................................................................................
        # Return
        # ..............................................................................................................

        return tokens, attn_keep

        # ..............................................................................................................
