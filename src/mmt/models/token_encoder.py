from __future__ import annotations

from typing import Any, Dict, cast

import torch
import torch.nn as nn

from mmt.data.signal_spec import SignalSpecRegistry


class TokenEncoder(nn.Module):
    """
    TokenEncoder
    =============

    This module projects per-signal embeddings (one vector per token)
    into a common d_model, adds metadata embeddings (pos/id/mod/role),
    and prepends a CLS token.

    -------------------------------------------------------------------
    IMPORTANT ARCHITECTURAL DECISION
    -------------------------------------------------------------------
    Projection layers are **not** keyed by `signal_id` alone, but by the
    pair `(signal_id, role_id)`.

    Why?
    ----
    The same physical signal (same signal_id) may appear in multiple roles:

        • as input   → small DCT embedding (e.g., dim=10)
        • as actuator → larger profile embedding (e.g., dim=40)
        • as output   → scalar embedding (e.g., dim=1)

    These different roles naturally have **different embedding dimensions**.
    Keying projections only by signal_id forces incompatible vectors to
    share the same Linear, causing the classic runtime error:

        “Signal id X was first seen with in_dim=A but now has in_dim=B.”

    With the correct design:
        projection_key = (signal_id, role_id)

    we preserve *physical identity* (signal id remains meaningful) but
    avoid dimension collisions between roles.

    -------------------------------------------------------------------
    Expected batch structure (from MMTCollate)
    -------------------------------------------------------------------
    batch = {
        "emb":  List[List[Tensor]]          # emb[b][t] is (D_i,)
        "pos":  LongTensor(B, L)
        "id":   LongTensor(B, L)            # physical signal id
        "mod":  LongTensor(B, L)            # modality id
        "role": LongTensor(B, L)            # 0=input, 1=actuator, 2=output
        "padding_mask": BoolTensor(B, L)
    }

    Output:
        tokens:    Tensor(B, L+1, d_model)
        attn_keep: BoolTensor(B, L+1)
    """

    ROLE_CONTEXT = 0
    ROLE_ACTUATOR = 1
    ROLE_OUTPUT = 2

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

        # ⚠️ projection modules keyed by (signal_id, role_id)
        self._proj = nn.ModuleDict()
        self._proj_in_dim: Dict[str, int] = {}

        # ------------------------------------------------------------------
        # CLS token parameters
        self.cls_content = nn.Parameter(torch.randn(self.d_model))
        self.cls_id = self.num_signals  # CLS occupies id=num_signals
        # ------------------------------------------------------------------

        # Metadata embeddings
        self.pos_embed = nn.Embedding(self.max_positions + 1, self.d_model)
        self.id_embed = nn.Embedding(self.num_signals + 1, self.d_model)
        self.mod_embed = nn.Embedding(self.num_modalities, self.d_model)
        self.role_embed = nn.Embedding(3, self.d_model)

    # ------------------------------------------------------------------
    def _get_proj(
        self,
        sig_id: int,
        role_id: int,
        in_dim: int,
        device: torch.device,
    ) -> nn.Linear:
        """
        Return (or lazily create) the projection layer for a given (signal_id, role_id).

        Projection key format:
            key = f"{sig_id}:{role_id}"

        This ensures:
            • same physical signal in different roles → different Linear layers
            • dims never collide
            • TokenEncoder never mismatches in_dim again
        """
        key = f"{int(sig_id)}:{int(role_id)}"

        # Skip PAD tokens (sid < 0 or in_dim == 0)
        if sig_id < 0 or in_dim == 0:
            raise RuntimeError(
                f"TokenEncoder._get_proj called on PAD token sid={sig_id}, in_dim={in_dim}"
            )

        if key not in self._proj:
            layer = nn.Linear(in_dim, self.d_model).to(device)
            self._proj[key] = layer
            self._proj_in_dim[key] = in_dim
        else:
            if self._proj_in_dim[key] != in_dim:
                raise ValueError(
                    f"(signal_id={sig_id}, role_id={role_id}) was first seen with "
                    f"in_dim={self._proj_in_dim[key]}, now has in_dim={in_dim}."
                )

        return cast(nn.Linear, self._proj[key])

    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        emb = batch["emb"]
        pos = batch["pos"]
        sid = batch["id"]
        mod = batch["mod"]
        role = batch["role"]
        padding_mask = batch["padding_mask"]

        if not isinstance(emb, list):
            raise TypeError("batch['emb'] must be a list-of-lists of tensors")

        B, L = pos.shape
        device = pos.device

        # ------------------------------------------------------------------
        # Project token content
        # ------------------------------------------------------------------
        tokens = torch.zeros(B, L, self.d_model, dtype=torch.float32, device=device)

        for b in range(B):
            for t in range(L):
                if not padding_mask[b, t]:
                    continue

                vec = emb[b][t]
                if not isinstance(vec, torch.Tensor):
                    raise TypeError("emb[b][t] must be a torch.Tensor")

                if vec.ndim != 1:
                    raise ValueError(
                        f"emb[{b}][{t}] must be 1D vector, got shape={tuple(vec.shape)}"
                    )

                in_dim = vec.shape[0]
                sid_bt = int(sid[b, t].item())
                rid_bt = int(role[b, t].item())

                if sid_bt >= 0 and in_dim > 0:
                    proj = self._get_proj(sid_bt, rid_bt, in_dim, device)
                    tokens[b, t] = proj(vec.to(device=device, dtype=torch.float32))

        # ------------------------------------------------------------------
        # CLS token
        cls_tok = self.cls_content.view(1, 1, -1).expand(B, 1, -1)
        tokens = torch.cat([cls_tok, tokens], dim=1)

        # ------------------------------------------------------------------
        # Metadata embeddings
        pos_cls = torch.zeros(B, 1, dtype=torch.long, device=device)
        sid_cls = torch.full((B, 1), self.cls_id, dtype=torch.long, device=device)

        pos_full = torch.cat([pos_cls, pos], dim=1)  # (B, L+1)
        sid_full = torch.cat([sid_cls, sid], dim=1)  # (B, L+1)

        tokens = tokens + self.pos_embed(pos_full) + self.id_embed(sid_full)

        # Non-CLS modality + role embeddings
        tokens[:, 1:, :] += self.mod_embed(mod) + self.role_embed(role)

        # ------------------------------------------------------------------
        # Attention mask
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        attn_keep = torch.cat([cls_mask, padding_mask], dim=1)

        return tokens, attn_keep
