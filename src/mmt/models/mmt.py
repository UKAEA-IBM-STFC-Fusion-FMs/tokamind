"""
Multi-Modal Transformer (MMT) model.

This module defines the main model used by the MMT pipeline. It consumes token
batches produced by:

  Chunk → SelectValidWindows → TrimChunks → EmbedChunks → BuildTokens → MMTCollate

and produces predictions for all output signals of the task.

The architecture is intentionally modular:
- TokenEncoder: projects per-token embeddings into d_model and adds metadata
- Backbone: transformer encoder over the token sequence
- Modality heads: map CLS to modality-specific latent vectors
- Output adapters: per-signal heads mapping modality latent → output embedding
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import torch
import torch.nn as nn

from mmt.data.signal_spec import SignalSpecRegistry

from .token_encoder import TokenEncoder
from .modality_heads import ModalityHead
from .output_adapters import OutputAdapter, resolve_output_adapter_hiddens
from .backbone import Backbone

import logging

logger = logging.getLogger("mmt.Model")


class MultiModalTransformer(nn.Module):
    """
     MultiModalTransformer: foundation + task model for the open-source MMT pipeline.

     This module implements the final stage of the MMT architecture: a lightweight,
     fully modular transformer that consumes *tokenized embeddings* produced by the
     MMT preprocessing pipeline:

         Chunk → SelectValidWindows → TrimChunks → EmbedChunks → BuildTokens → MMTCollate

     and generates predictions for all output signals of a task.

     The model is intentionally simple, explicit, and easy to reason about. It is
     composed of four cleanly separated blocks:

         TokenEncoder  →  Transformer backbone  →  Modality heads  →  Output adapters


     -------------------------------------------------------------------------------
     1. TokenEncoder
     -------------------------------------------------------------------------------
     The TokenEncoder receives packed token embeddings (by signal_id) and metadata produced by
     `MMTCollate`. For each token it:

       • selects the correct projection layer for its signal_id and maps the
         chunk-level embedding from `D_enc(signal)` → `d_model`,
       • adds positional, signal-ID and role embeddings,
       • prepends a learned CLS token (`pos = 0`, `role = OUTPUT`).

     Internally, per-signal projection layers are keyed by a stable canonical
     string derived from the SignalSpec, so that module names remain independent
     of the numeric `signal_id` used in the preprocessing pipeline.

     Output:
         tokens : (B, L+1, d_model)
         attn_keep : (B, L+1)  — True where the token is real (including CLS)

     No raw chunks or raw signals enter the model; all preprocessing happens outside.


     -------------------------------------------------------------------------------
     2. Transformer backbone
     -------------------------------------------------------------------------------
     A standard PyTorch `nn.TransformerEncoder` (`batch_first=True`) processes the
     sequence of tokens and produces a contextualised representation for each token.

     Masking:
         • On CPU/CUDA, we use `src_key_padding_mask = ~attn_keep`.
         • On MPS (Apple Silicon), padding-only columns are pruned and the mask is
           dropped to avoid unsupported nested-tensor code paths.

     Output:
         h : (B, L+1, d_model)


     -------------------------------------------------------------------------------
     3. Modality heads
     -------------------------------------------------------------------------------
     Each modality (e.g. `"timeseries"`, `"profile"`, `"video"`) receives its own
     small MLP that maps the CLS token to a modality-specific latent vector
     `G_mod`:

         modality_latent[mod] = head_mod(h_cls)     # (B, G_mod)

     This corresponds to the shared “modality subspace” in the original MMT:
     inputs of the same modality share statistical structure.


     -------------------------------------------------------------------------------
     4. Output adapters
     -------------------------------------------------------------------------------
     Every output signal (role="output") receives:

       • an output dimension `K_t = SignalSpec.embedding_dim`,
       • an OutputAdapter: a small linear or MLP mapping `G_mod → K_t`.

         pred[sid] = adapter_sid(modality_latent[modality_of_sid])

     Internally, output adapters are stored in a ModuleDict keyed by a canonical
     string derived from the SignalSpec (e.g. `"output:pf_active-coil_current"`),
     while the public `pred` dictionary is still keyed by numeric `signal_id`.
     This ensures that warm-starting and checkpoint loading are driven by
     stable, human-readable keys rather than by internal ID ordering.

     This cleanly separates:
         • modality-level representation learning (shared),
         • per-signal heads (task-specific).


     -------------------------------------------------------------------------------
     Input format (from MMTCollate)
    -------------------------------------------------------------------------------
     The forward pass expects a batch dictionary containing at least:

         batch["emb"]           : Dict[int, Tensor] packed by signal_id (sid)
         batch["emb_index"]     : Dict[int, LongTensor] flat indices (b*L+t) aligned with emb[sid]
         batch["pos"]           : LongTensor (B, L)
         batch["id"]            : LongTensor (B, L)  — physical signal IDs
         batch["role"]          : LongTensor (B, L)
         batch["padding_mask"]  : BoolTensor (B, L)

     All fields are produced by BuildTokensTransform + MMTCollate.
     No raw arrays, no dicts of chunks, and no signal names are used here.


     -------------------------------------------------------------------------------
     Model initialization parameters
     -------------------------------------------------------------------------------

     Parameters
     ----------
     signal_specs : SignalSpecRegistry
         Registry with one spec per signal (name, role, modality, encoder, embedding_dim).
         Determines which signals are inputs/outputs and the required output dimensions.

     d_model : int
         Transformer model dimension (size of token embeddings after projection).

     n_layers : int
         Number of TransformerEncoder layers in the backbone.

     n_heads : int
         Number of attention heads per layer.

     dim_ff : int
         Feed-forward dimension inside Transformer layers.

     dropout : float
         Dropout probability inside the backbone.

     max_positions : int
         Maximum number of temporal positions for positional embeddings.
         Usually equal to preprocessing.trim_chunks.max_chunks.

     modality_heads_cfg : dict
         Configuration of modality heads. Example:
             {
               "timeseries": {"hidden": 128, "out_dim": 128},
               "profile":    {"hidden": 128, "out_dim": 128},
               "video":      {"hidden": 192, "out_dim": 128},
             }

     output_adapters_cfg : dict
         Configuration of output adapters. Example:
             {
               "hidden_dim": {
                 "default": 0,
                 "bucketed": {
                   "enable": True,
                   "rules": [
                     {"max_out_dim": 64, "hidden": 0},
                     {"max_out_dim": None, "hidden": "d_model"},
                   ],
                 },
                 "manual": {"equilibrium-psi": 32},
               }
             }

     backbone_activation : str
         Activation function for the Transformer backbone ("relu", "gelu", …).

     debug_tokens : bool
         Enable extra consistency checks in the TokenEncoder.


     -------------------------------------------------------------------------------
     Forward() return structure
     -------------------------------------------------------------------------------
     The forward method returns:

         {
             "h_cls"          : Tensor (B, d_model),
             "modality_latent": Dict[str, Tensor(B, G_mod)],
             "pred"           : Dict[signal_id, Tensor(B, K_t)],
         }

     • `h_cls` is the pooled representation (CLS token).
     • `modality_latent` contains one latent vector per modality.
     • `pred` maps each output signal_id → its prediction vector (dimension K_t).

     Downstream train code computes losses directly from `pred`
     (MSE, masked MSE, task-specific losses, etc.).

    Why preds --> signal_id?
    ----------------
    Internally, the model uses integer signal IDs for fast routing, indexing,
    masking, and adapter selection.  These IDs are stable within a given task
    and ensure that the transformer does not depend on user-facing string names.

    Why not return names here?
    ---------------------------
    Higher-level components (training loop, evaluation, trace saving, etc.)
    convert the model outputs from:

        signal_id → canonical output name

    using the SignalSpecRegistry.  This separation ensures:

        • the model stays efficient and ID-keyed internally,
        • user-facing APIs (metrics, CSVs, adapters, configs) remain name-keyed.

    Downstream training code computes all losses directly from the returned
    dict[int → Tensor], and evaluation performs ID→name conversion before
    decoding and destandardizing outputs.
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
    ):
        super().__init__()
        self.signal_specs = signal_specs

        # ------------------------------------------------------------------
        # 1) Token encoder + backbone
        # ------------------------------------------------------------------
        self.tokens = TokenEncoder(
            d_model=d_model,
            signal_specs=signal_specs,
            max_positions=max_positions,
            debug_checks=debug_tokens,
        )
        self.backbone = Backbone(
            d_model=d_model,
            n_heads=n_heads,
            dim_ff=dim_ff,
            n_layers=n_layers,
            dropout=dropout,
            activation=backbone_activation,
        )
        self.backbone_out_dim = d_model

        # ------------------------------------------------------------------
        # 2) Output specs and modalities
        # ------------------------------------------------------------------
        output_specs = [s for s in signal_specs.specs if s.role == "output"]
        if not output_specs:
            raise ValueError("SignalSpecRegistry contains no outputs (role='output').")
        self.output_specs = output_specs

        # Map signal_id → modality name
        self.output2modality: Dict[int, str] = {}
        modalities = sorted(set(s.modality for s in output_specs))
        for spec in output_specs:
            self.output2modality[spec.signal_id] = spec.modality

        # Per-modality head configuration (from model.modality_heads)
        per_mod_hidden: Dict[str, int] = {}
        per_mod_dim: Dict[str, int] = {}
        for mod in modalities:
            cfg = modality_heads_cfg.get(mod)
            if cfg is None:
                raise KeyError(
                    f"No modality head configuration provided for modality={mod!r}"
                )
            per_mod_hidden[mod] = int(cfg.get("hidden", 0) or 0)
            per_mod_dim[mod] = int(cfg.get("out_dim", d_model) or d_model)

        self.modality_heads = nn.ModuleDict(
            {
                mod: ModalityHead(
                    in_dim=int(self.backbone_out_dim),
                    out_dim=per_mod_dim[mod],
                    hidden=per_mod_hidden[mod],
                    layers=2,
                )
                for mod in modalities
            }
        )

        # ------------------------------------------------------------------
        # 3) Per-output adapters (from model.output_adapters)
        # ------------------------------------------------------------------
        hidden_dim_cfg = output_adapters_cfg.get("hidden_dim", None)
        hidden_by_name = resolve_output_adapter_hiddens(
            output_specs=output_specs,
            d_model=d_model,
            hidden_dim_cfg=hidden_dim_cfg,
        )

        self.output_adapters = nn.ModuleDict()
        self.output_dims: Dict[int, int] = {}
        self.output_hidden: Dict[int, int] = {}
        # Mapping from signal_id → canonical adapter key "role:name"
        self.output_sid_to_key: Dict[int, str] = {}

        for spec in output_specs:
            g = spec.modality
            G_mod = int(self.modality_heads[g].out_dim)

            # Target dimension: stored in SignalSpec.embedding_dim
            K_t = getattr(spec, "embedding_dim", None)
            if K_t is None:
                raise AttributeError(
                    "SignalSpec is expected to have an 'embedding_dim' field for outputs. "
                    "Please update SignalSpecRegistry to attach embedding_dim."
                )
            K_t = int(K_t)

            hidden = hidden_by_name.get(spec.name, 0)

            self.output_hidden[spec.signal_id] = int(hidden)

            # Use a stable canonical key for the adapter name (role:name)
            adapter_key = spec.canonical_key
            self.output_adapters[adapter_key] = OutputAdapter(
                in_dim=G_mod,
                out_dim=K_t,
                hidden=hidden,
            )

            self.output_dims[spec.signal_id] = K_t
            self.output_sid_to_key[spec.signal_id] = adapter_key

        self._print_init_summary(modalities, per_mod_hidden, per_mod_dim)

    # ------------------------------------------------------------------
    def _print_init_summary(
        self,
        modalities: List[str],
        per_mod_hidden: Dict[str, int],
        per_mod_dim: Dict[str, int],
    ) -> None:
        logger.info("MultiModalTransformer initialised:")
        logger.info(f"  backbone: d_model={self.backbone_out_dim}")

        logger.info("  modality heads:")
        for mod in modalities:
            logger.info(
                f"    - {mod}: hidden={per_mod_hidden[mod]}, out_dim={per_mod_dim[mod]}"
            )

        logger.info("  Output adapters:")
        for spec in self.output_specs:
            sid = spec.signal_id
            mod = self.output2modality[sid]
            hidden = int(self.output_hidden.get(sid, 0))
            logger.info(
                f"    - {spec.name} (id={sid}, modality={mod}): dim={self.output_dims[sid]}, hidden={hidden}"
            )

    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        batch: output of MMTCollate, with at least:
          * "emb"           : packed embeddings (Dict[int, Tensor])
          * "emb_index"     : packed indices (Dict[int, LongTensor])
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
                _attn_keep = attn_keep.index_select(dim=1, index=idx)
            src_key_padding_mask = None  # avoid problematic MPS code path

        # 2) Transformer backbone
        h = self.backbone(tokens, src_key_padding_mask=src_key_padding_mask)

        # 3) CLS
        h_cls = h[:, 0, :]  # (B, d_model)

        # 4) Per-modality heads
        modality_latent: Dict[str, torch.Tensor] = {
            mod: self.modality_heads[mod](h_cls) for mod in self.modality_heads.keys()
        }

        # 5) Per-output adapters
        preds: Dict[int, torch.Tensor] = {}
        for spec in self.output_specs:
            sid = spec.signal_id
            g = self.output2modality[sid]
            adapter_key = self.output_sid_to_key[sid]
            adapter = self.output_adapters[adapter_key]
            preds[sid] = adapter(modality_latent[g])

        return {
            "h_cls": h_cls,
            "modality_latent": modality_latent,
            "pred": preds,
        }

    # ------------------------------------------------------------------
    # Checkpoint I/O — three explicit parts: backbone, modality_heads, output_adapters
    # ------------------------------------------------------------------
    def get_token_encoder_state_dict(self) -> dict:
        return self.tokens.state_dict()

    def get_backbone_state_dict(self) -> dict:
        return self.backbone.state_dict()

    def load_backbone_state_dict(self, state: dict, strict: bool = True):
        missing, unexpected = self.backbone.load_state_dict(state, strict=strict)
        return missing, unexpected

    def get_modality_heads_state_dict(self) -> dict:
        return self.modality_heads.state_dict()

    def load_token_encoder_state_dict(self, state: dict, strict: bool = True):
        # Same API pattern as backbone/heads/adapters:
        missing, unexpected = self.tokens.load_state_dict(state, strict=strict)
        return missing, unexpected

    def load_modality_heads_state_dict(self, state: dict, strict: bool = True):
        missing, unexpected = self.modality_heads.load_state_dict(state, strict=strict)
        return missing, unexpected

    def get_output_adapters_state_dict(self) -> dict:
        return self.output_adapters.state_dict()

    def load_output_adapters_state_dict(self, state: dict, strict: bool = True):
        missing, unexpected = self.output_adapters.load_state_dict(state, strict=strict)
        return missing, unexpected
