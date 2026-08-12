"""
Multi-Modal Transformer (MMT) model.

This module defines the main model used by the MMT pipeline. It consumes token batches produced by:

  Chunk → SelectValidWindows → TrimChunks → EmbedChunks → BuildTokens → MMTCollate

and produces predictions for all output signals of the task.

The architecture is intentionally modular:
- TokenEncoder: projects per-token embeddings into d_model and adds metadata
- Backbone: transformer encoder over the token sequence
- Modality heads: map CLS to modality-specific latent vectors
- Output adapters: per-signal heads mapping modality latent → output embedding
"""

from __future__ import annotations

from typing import Any, cast
from collections.abc import Mapping

import torch
import torch.nn as nn
import logging

from mmt.data.signal_spec import SignalSpecRegistry

from .token_encoder import TokenEncoder
from .modality_heads import ModalityHead
from .output_adapters import (
    OutputAdapter,
    OutputAdapterItem,
    ProbabilisticOutputAdapter,
    apply_output_residual,
    apply_output_adapters,
    resolve_gaussian_adapter_cfg,
    resolve_output_adapter_hiddens,
    zero_initialize_output_corrections,
)
from .backbone import Backbone
from .blocks import MMT_BLOCK_NAMES


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.Model")


# ======================================================================================================================
class MultiModalTransformer(nn.Module):
    """
     MultiModalTransformer: foundation + task model for the open-source MMT pipeline.

     This module implements the final stage of the MMT architecture: a lightweight, fully modular transformer that
     consumes *tokenized embeddings* produced by the MMT preprocessing pipeline:

         Chunk → SelectValidWindows → TrimChunks → EmbedChunks → BuildTokens → MMTCollate

     and generates predictions for all output signals of a task.

     The model is intentionally simple, explicit, and easy to reason about. It is composed of four cleanly separated
     blocks:

         TokenEncoder  →  Transformer backbone  →  Modality heads  →  Output adapters


     -------------------------------------------------------------------------------------------------------------------
     1. TokenEncoder
     -------------------------------------------------------------------------------------------------------------------

     The TokenEncoder receives packed token embeddings (by signal_id) and metadata produced by `MMTCollate`. For each
     token it:

       • selects the correct projection layer for its signal_id and maps the chunk-level embedding from
       `D_enc(signal)` → `d_model`,
       • adds positional, signal-id and role embeddings,
       • prepends a learned CLS token (`pos = 0`, `role = OUTPUT`).

     Internally, per-signal projection layers are keyed by a stable canonical string derived from the SignalSpec, so
     that module names remain independent of the numeric `signal_id` used in the preprocessing pipeline.

     Output:
         tokens : (B, L+1, d_model)
         attn_keep : (B, L+1)  — True where the token is real (including CLS)

     No raw chunks or raw signals enter the model; all preprocessing happens outside.


     -------------------------------------------------------------------------------------------------------------------
     2. Transformer backbone
     -------------------------------------------------------------------------------------------------------------------

     A standard PyTorch `nn.TransformerEncoder` (`batch_first=True`) processes the sequence of tokens and produces a
     contextualized representation for each token.

     Masking:
         • On CPU/CUDA, we use `src_key_padding_mask = ~attn_keep`.
         • On MPS (Apple Silicon), padding-only columns are pruned and the mask is dropped to avoid unsupported
         nested-tensor code paths.

     Output:
         h : (B, L+1, d_model)


     -------------------------------------------------------------------------------------------------------------------
     3. Modality heads
     -------------------------------------------------------------------------------------------------------------------

     Each modality (e.g., `"timeseries"`, `"profile"`, `"video"`) receives its own small MLP that maps the CLS token to
     a modality-specific latent vector `G_mod`:

         modality_latent[mod] = head_mod(h_cls)     # (B, G_mod)

     This corresponds to the shared “modality subspace” in the original MMT: inputs of the same modality share
     statistical structure.


     -------------------------------------------------------------------------------------------------------------------
     4. Output adapters
     -------------------------------------------------------------------------------------------------------------------

     Every output signal (role="output") receives:

       • an output dimension `K_t = SignalSpec.embedding_dim`,
       • an OutputAdapter: a small linear or MLP mapping `G_mod → K_t`.

         pred[sid] = adapter_sid(modality_latent[modality_of_sid])

     Internally, output adapters are stored in a ModuleDict keyed by a canonical string derived from the SignalSpec
     (e.g., `"output:pf_active-coil_current"`), while the public `pred` dictionary is still keyed by numeric
     `signal_id`. This ensures that warm-starting and checkpoint loading are driven by stable, human-readable keys
     rather than by internal ID ordering.

     This cleanly separates:
         • modality-level representation learning (shared),
         • per-signal heads (task-specific).


     -------------------------------------------------------------------------------------------------------------------
     Input format (from MMTCollate)
     -------------------------------------------------------------------------------------------------------------------

     The forward pass expects a batch dictionary containing at least:

         batch["emb"]           : dict[int, Tensor] packed by signal_id (sid)
         batch["emb_index"]     : dict[int, LongTensor] flat indices (b*L+t) aligned with emb[sid]
         batch["pos"]           : LongTensor (B, L)
         batch["id"]            : LongTensor (B, L)  — physical signal IDs
         batch["role"]          : LongTensor (B, L)
         batch["padding_mask"]  : BoolTensor (B, L)

     All fields are produced by BuildTokensTransform + MMTCollate. No raw arrays, no dicts of chunks, and no signal
     names are used here.


     -------------------------------------------------------------------------------------------------------------------
     Model initialization parameters
     -------------------------------------------------------------------------------------------------------------------

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


     -------------------------------------------------------------------------------------------------------------------
     Forward() return structure
     -------------------------------------------------------------------------------------------------------------------

     The forward method returns:

         {
             "h_cls"          : Tensor (B, d_model),
             "modality_latent": dict[str, Tensor(B, G_mod)],
             "pred"           : dict[signal_id, Tensor(B, K_t)],
         }

     • `h_cls` is the pooled representation (CLS token).
     • `modality_latent` contains one latent vector per modality.
     • `pred` maps each output signal_id → its prediction vector (dimension K_t).

     Downstream train code computes losses directly from `pred` (MSE, masked MSE, task-specific losses, etc.).

    Why preds --> signal_id?
    ------------------------
    Internally, the model uses integer signal IDs for fast routing, indexing, masking, and adapter selection. These IDs
    are stable within a given task and ensure that the transformer does not depend on user-facing string names.

    Why not return names here?
    ---------------------------
    Higher-level components (training loop, evaluation, trace saving, etc.) convert the model outputs from:

        signal_id → canonical output name

    using the SignalSpecRegistry.  This separation ensures:

        • the model stays efficient and ID-keyed internally,
        • user-facing APIs (metrics, CSVs, adapters, configs) remain name-keyed.

    Downstream training code computes all losses directly from the returned dict[int → Tensor], and evaluation performs
    ID→name conversion before decoding and destandardizing outputs.

    """

    # ------------------------------------------------------------------------------------------------------------------
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
        output_adapter_type: str = "deterministic",
        output_residual_enabled: bool = False,
        output_residual_zero_init: bool = True,
    ) -> None:
        """

        Initialization of class parameters.

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
        modality_heads_cfg : Mapping[str, Mapping[str, Any]]
            Configuration of modality heads. Example:
                {
                    "timeseries": {"hidden": 128, "out_dim": 128},
                    "profile":    {"hidden": 128, "out_dim": 128},
                    "video":      {"hidden": 192, "out_dim": 128},
                }
        output_adapters_cfg : Mapping[str, Any]
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
            Optional. Default: "relu".
        debug_tokens : bool
            Enable extra consistency checks in the TokenEncoder.
            Optional. Default: False.
        output_adapter_type : str
            Output adapter type. One of ``"deterministic"`` or ``"gaussian"``.
            Optional. Default: "deterministic".
        output_residual_enabled : bool
            Whether same-name outputs predict zero-initialized corrections on
            top of latest-input output-space baselines.
            Optional. Default: False.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors.

        Raises
        ------
        ValueError
            If `signal_specs` contains no outputs.
        KeyError
            If no modality head configuration in `modality_heads_cfg` for a given modality.
        AttributeError
            If `signal_specs` does not have an 'embedding_dim' field for outputs.

        """

        super().__init__()
        self.signal_specs = signal_specs
        self.output_adapter_type = str(output_adapter_type)
        self.output_residual_enabled = bool(output_residual_enabled)
        self.output_residual_zero_init = bool(output_residual_zero_init)
        if self.output_adapter_type not in {"deterministic", "gaussian"}:
            raise ValueError(
                f"Unsupported output_adapter_type={self.output_adapter_type!r}. Expected 'deterministic' or 'gaussian'."
            )

        # ..............................................................................................................
        # 1) Token encoder + backbone
        # ..............................................................................................................

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

        # ..............................................................................................................
        # 2) Output specs and modalities
        # ..............................................................................................................

        output_specs = [s for s in signal_specs.specs if s.role == "output"]
        if not output_specs:
            raise ValueError("SignalSpecRegistry `signal_specs` contains no outputs (role='output').")

        self.output_specs = output_specs

        # Map signal_id → modality name
        self.output2modality: dict[int, str] = {}
        modalities = sorted({s.modality for s in output_specs})
        for spec in output_specs:
            self.output2modality[spec.signal_id] = spec.modality

        # Per-modality head configuration (from model.modality_heads).
        per_mod_hidden: dict[str, int] = {}
        per_mod_dim: dict[str, int] = {}
        for mod in modalities:
            cfg = modality_heads_cfg.get(mod)
            if cfg is None:
                raise KeyError(f"No modality head configuration provided in `modality_heads_cfg` for modality={mod!r}.")

            per_mod_hidden[mod] = int(cfg.get("hidden", 0) or 0)
            per_mod_dim[mod] = int(cfg.get("out_dim", d_model) or d_model)

        self.modality_heads = nn.ModuleDict(
            {
                mod: ModalityHead(
                    in_dim=int(self.backbone_out_dim),
                    out_dim=per_mod_dim[mod],
                    hidden_dim=per_mod_hidden[mod],
                    layers=2,
                )
                for mod in modalities
            }
        )

        # ..............................................................................................................
        # 3) Per-output adapters (from model.output_adapters)
        # ..............................................................................................................

        hidden_dim_cfg = output_adapters_cfg.get("hidden_dim", None)
        hidden_by_name = resolve_output_adapter_hiddens(
            output_specs=output_specs, d_model=d_model, hidden_dim_cfg=hidden_dim_cfg
        )
        gaussian_adapter_cfg = resolve_gaussian_adapter_cfg(output_adapters_cfg=output_adapters_cfg)

        self.output_adapters = nn.ModuleDict()
        self.output_dims: dict[int, int] = {}
        self.output_hidden: dict[int, int] = {}
        # Mapping from signal_id → canonical adapter key "role:name".
        self.output_sid_to_key: dict[int, str] = {}

        for spec in output_specs:
            g = spec.modality
            head = cast(ModalityHead, self.modality_heads[g])
            G_mod = int(head.out_dim)  # noqa - Ignore lowercase warning

            # Target dimension: stored in SignalSpec.embedding_dim.
            K_t = getattr(spec, "embedding_dim", None)  # noqa - Ignore lowercase warning
            if K_t is None:
                raise AttributeError(
                    "`signal_specs` is expected to have an 'embedding_dim' field for outputs. "
                    "Please update SignalSpecRegistry to attach embedding_dim."
                )

            K_t = int(K_t)  # type: ignore[arg-type]

            hidden_dim = hidden_by_name.get(spec.name, 0)

            self.output_hidden[spec.signal_id] = int(hidden_dim)

            # Use a stable canonical key for the adapter name (role:name)
            adapter_key = spec.canonical_key
            if self.output_adapter_type == "gaussian":
                self.output_adapters[adapter_key] = ProbabilisticOutputAdapter(
                    in_dim=G_mod,
                    out_dim=K_t,
                    hidden_dim=hidden_dim,
                    **gaussian_adapter_cfg,
                )
            else:
                self.output_adapters[adapter_key] = OutputAdapter(in_dim=G_mod, out_dim=K_t, hidden_dim=hidden_dim)

            self.output_dims[spec.signal_id] = K_t
            self.output_sid_to_key[spec.signal_id] = adapter_key

        input_names = {str(spec.name) for spec in signal_specs.specs_for_role("input")}
        self.output_residual_ids: set[int] = (
            {int(spec.signal_id) for spec in output_specs if str(spec.name) in input_names}
            if self.output_residual_enabled
            else set()
        )
        if self.output_residual_enabled and not self.output_residual_ids:
            raise ValueError("Output residual mode requires at least one same-name input/output signal pair.")
        if self.output_residual_ids and self.output_residual_zero_init:
            zero_initialize_output_corrections(
                output_adapters=self.output_adapters,
                adapter_keys=(self.output_sid_to_key[sid] for sid in self.output_residual_ids),
            )

        self._print_init_summary(
            modalities=modalities,
            per_mod_hidden=per_mod_hidden,
            per_mod_dim=per_mod_dim,
        )

    # ------------------------------------------------------------------------------------------------------------------
    def _print_init_summary(
        self,
        modalities: list[str],
        per_mod_hidden: Mapping[str, int],
        per_mod_dim: Mapping[str, int],
    ) -> None:
        """
        Print init summary.

        Parameters
        ----------
        modalities : list[str]
            Target modality heads.
        per_mod_hidden : Mapping[str, int]
            Mapping for modalities' hidden status.
        per_mod_dim : Mapping[str, int]
            Mapping for modalities' dimension.

        Returns
        -------
        None

        """

        logger.info("MultiModalTransformer initialized:")
        logger.info(f"  backbone: d_model={self.backbone_out_dim}")
        logger.info(
            "  output residual: enabled=%s mapped_outputs=%d zero_init=%s",
            self.output_residual_enabled,
            len(self.output_residual_ids),
            self.output_residual_zero_init,
        )

        logger.info("  modality heads:")
        for mod in modalities:
            logger.info(f"    - {mod}: hidden={per_mod_hidden[mod]}, out_dim={per_mod_dim[mod]}")

        logger.info(f"  Output adapters (type={self.output_adapter_type}):")
        for spec in self.output_specs:
            sid = spec.signal_id
            mod = self.output2modality[sid]
            hidden = int(self.output_hidden.get(sid, 0))
            logger.info(f"    - {spec.name} (id={sid}, modality={mod}): dim={self.output_dims[sid]}, hidden={hidden}")

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """
        MMT's forward function.

        Parameters
        ----------
        batch : Mapping[str, Any]
            Output of MMTCollate, with at least:
              * "emb"           : packed embeddings (dict[int, Tensor])
              * "emb_index"     : packed indices (dict[int, LongTensor])
              * "pos", "id", "role" : LongTensor (B, L)
              * "padding_mask"  : BoolTensor (B, L)

        Returns
        -------
        dict[str, Any]
            Mapping with "h_cls" (CLS), "modality_latent" (per-modality heads), and "pred" (prediction vector) keys.
            Gaussian heads additionally return "pred_dist" (consumers sample from it via sample_gaussian_dist).

        """

        # ..............................................................................................................
        # 1) Tokens + attention mask (True = keep)
        # ..............................................................................................................

        tokens, attn_keep = self.tokens(batch)  # (B, L+1, d_model)
        src_key_padding_mask = ~attn_keep  # True = PAD (for Transformer)

        # --- MPS workaround (drop mask; keep any-present token columns) ---
        if tokens.device.type == "mps":
            keep_cols = attn_keep.any(dim=0)  # (L+1,)
            if bool((~keep_cols).any()):
                idx = keep_cols.nonzero(as_tuple=False).squeeze(1)
                tokens = tokens.index_select(dim=1, index=idx)
                _attn_keep = attn_keep.index_select(dim=1, index=idx)
            src_key_padding_mask = None  # Avoid problematic MPS code path

        # ..............................................................................................................
        # 2) Transformer backbone
        # ..............................................................................................................

        h = self.backbone(tokens, src_key_padding_mask=src_key_padding_mask)

        # ..............................................................................................................
        # 3) CLS
        # ..............................................................................................................

        h_cls = h[:, 0, :]  # (B, d_model)

        # ..............................................................................................................
        # 4) Per-modality heads
        # ..............................................................................................................

        modality_latent: dict[str, torch.Tensor] = {
            mod: self.modality_heads[mod](h_cls) for mod in self.modality_heads.keys()
        }

        # ..............................................................................................................
        # 5) Per-output adapters
        # ..............................................................................................................

        adapter_items: list[OutputAdapterItem] = []
        for spec in self.output_specs:
            sid = spec.signal_id
            g = self.output2modality[sid]
            adapter_key = self.output_sid_to_key[sid]
            adapter_items.append((sid, adapter_key, modality_latent[g]))

        # ..............................................................................................................
        # Return
        # ..............................................................................................................

        adapter_output = apply_output_adapters(
            output_adapters=self.output_adapters,
            items=adapter_items,
            output_adapter_type=self.output_adapter_type,
        )
        adapter_output = apply_output_residual(
            adapter_output=adapter_output,
            baseline_emb=batch.get("output_baseline_emb"),
            residual_output_ids=set(self.output_residual_ids),
        )

        out = {
            "h_cls": h_cls,
            "modality_latent": modality_latent,
            **adapter_output,
        }

        return out

    # ==================================================================================================================
    # Model block API
    # ==================================================================================================================

    # ------------------------------------------------------------------------------------------------------------------
    def get_named_blocks(self) -> dict[str, nn.Module]:
        """
        Return learnable model blocks keyed by stable checkpoint/config names.

        Returns
        -------
        dict[str, nn.Module]
            Mapping from block name to module.

        """

        blocks = {
            "token_encoder": self.tokens,
            "backbone": self.backbone,
            "modality_heads": self.modality_heads,
            "output_adapters": self.output_adapters,
        }
        return {name: blocks[name] for name in MMT_BLOCK_NAMES}

    # ------------------------------------------------------------------------------------------------------------------
