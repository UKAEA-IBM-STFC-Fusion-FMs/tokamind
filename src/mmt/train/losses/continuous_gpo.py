"""
Continuous GPO (Generalized Preference Optimization) loss term.

Unlike token-level DPO/IPO the model operates in continuous embedding
(coefficient) space, so log-probabilities are not available.  Instead we use
an embedding-space distance proxy.

Margins and objectives
----------------------
Without a reference model the implemented margin is the fixed collection gap minus policy error:

    margin = transform(d(y_w, y_l)) - d(ŷ, y_w)

Here transform applies log1p first, then an optional upper clip. With a frozen reference:

    margin = [d(ŷ, y_l) - d(ŷ, y_w)] - [d(ŷ_ref, y_l) - d(ŷ_ref, y_w)]

This is a relative distance margin, not a KL divergence or an absolute constraint on policy drift.
The reference cancels at initialization when policy and reference coincide, giving margin=0.
Collection-gap clip/log options are incompatible with this branch and are rejected.

    DPO:   L = -log σ(β margin),       ∂L/∂margin = -β σ(-β margin).
           At margin=0 the derivative is -β/2, not zero; it tends to zero as β margin → +∞.
    IPO:   L = (margin - 1/(2β))²,     ∂L/∂margin = 2(margin - 1/(2β)).
           Its derivative is zero at the target; overshoot is penalized along the margin direction.
    SLiC:  L = max(0, 1/(2β) - margin).
    Hinge: L = max(0, 1 - margin).

For squared distance, the reference-relative margin is linear in ŷ. Components orthogonal to y_w-y_l
are unconstrained by any of these margin objectives. IPO does not remove that freedom. The optional
sft_weight adds MSE on paired windows; a separate embed_mse term anchors every valid output window.
These are soft penalties whose strength must be selected on validation, not guarantees against drift.

where:
  ŷ        = current model prediction  (embedding, shape B×D)
  ŷ_ref    = frozen reference model prediction (same architecture, base weights)
  y_w      = preferred embedding (ground truth, from batch["y_w_emb"])
  y_l      = dispreferred distance anchor (collection-time pred, from batch["y_l_emb"])
  d()      = per-sample MSE (mean over D dimensions)
  β        = margin temperature
  σ        = sigmoid

Batch dict requirements
-----------------------
    batch["y_w_emb"]   : dict[signal_id → Tensor(B, D)]
    batch["y_l_emb"]   : dict[signal_id → Tensor(B, D)]
    batch["ref_preds"] : dict[signal_id → Tensor(B, D)]  ← optional

Config (inside ``train.loss.terms``)
--------------------------------------
::

    - type: continuous_gpo
      weight: 1.0
      beta: 0.1         # margin temperature; controls δ for slic/ipo (δ = 1/(2β))
      loss_type: dpo    # "dpo" | "ipo" | "slic" | "hinge"
      sft_weight: 0.5   # supervised MSE co-loss weight  (default 0.0 = off)
                        # doubles as the SLiC regularisation weight λ_reg
      # Reference model is enabled globally via train.use_reference_model: true
      # in gpo.yaml; no per-term flag needed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Hashable

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import BaseLoss, LossComputeContext

_DEFAULT_BETA: float = 0.1
_DEFAULT_SFT_WEIGHT: float = 0.0
_DEFAULT_LOSS_TYPE: str = "dpo"
_VALID_LOSS_TYPES: frozenset[str] = frozenset({"dpo", "ipo", "slic", "hinge"})

# Keys owned by this loss class (used in validate_term_cfg).
_OWN_CFG_KEYS: frozenset[str] = frozenset({"beta", "loss_type", "sft_weight", "mse_gap_clip", "log_mse_gap"})
# Note: use_reference_model is a train-level flag, not a per-term key.


# ======================================================================================================================
class ContinuousGPOLoss(BaseLoss):
    """
    Embedding-space GPO loss for continuous regression models.

    Uses squared Euclidean distance in embedding space as a proxy for
    log-probabilities in the Bradley-Terry preference model.

    Supports four preference-optimisation objectives selected via ``loss_type``:

    * ``"dpo"``   — DPO log-sigmoid loss (default).
    * ``"ipo"``   — IPO squared-margin loss (Azar et al., 2023).
    * ``"slic"``  — SLiC max-margin loss with hinge target δ = 1/(2β)
                    (Zhao et al., 2023).
    * ``"hinge"`` — Hard-margin hinge loss with fixed target of 1.

    Parameters
    ----------
    beta : float
        Margin temperature (default 0.1).  Higher β amplifies the preference
        signal; lower β smooths it toward zero.
    loss_type : str
        Preference-optimisation objective.  One of:

        ``"dpo"`` (default)
            DPO log-sigmoid loss: ``−log σ(β·margin)``.
            Gradient = ``−β·σ(−β·margin)``; at margin=0 it is ``−β/2``.
            The collection-gap beta rule is a scale heuristic, not a reference-margin calibration.

        ``"ipo"``
            IPO squared-margin loss: ``(margin − 1/(2β))²``.
            Gradient = ``2·(margin − 1/(2β))``; zero at its target.
            The unique fixed point is margin = 1/(2β).
            Does not constrain prediction components orthogonal to the pair direction.

        ``"slic"``
            SLiC max-margin loss: ``max(0, δ − margin)`` where δ = 1/(2β).
            Gradient = ``−1`` when margin < δ, ``0`` when margin ≥ δ.
            Unlike IPO the loss is exactly zero once the margin exceeds δ,
            so well-separated pairs receive no gradient.  More conservative
            than IPO; may be preferable when a fraction of pairs already
            have large margins and should not be disturbed.

        ``"hinge"``
            Hard-margin hinge loss: ``max(0, 1 − margin)``.
            The target margin is always 1 regardless of β.
            Gradient = ``−1`` for unsatisfied pairs (margin < 1), ``0``
            otherwise.  Use when you want an absolute margin constraint
            decoupled from β (e.g. β is used only for β-effectiveness
            diagnostics but not for the loss target).

    sft_weight : float
        Weight of the supervised fine-tuning co-loss (MSE between ŷ and y_w).
        When > 0 the total loss is::

            L = L_gpo + sft_weight × MSE(ŷ, y_w)

        This softly penalizes drift from the preferred response on paired windows.
        Default: 0.0 (disabled).
    mse_gap_clip : float | None
        If set, the per-sample collection-time MSE gap ``‖y_w − y_l‖²/D``
        is hard-clipped after optional log1p compression, only without a reference model.
        This changes the margin scale; it is not a general gradient-norm bound.
        Applied to ``d_l_collect = ‖y_w − y_l‖²/D`` used as the Bradley-Terry
        dispreferred distance approximation.  Default: ``None`` (no clip).
    log_mse_gap : bool
        If ``True``, replace the raw collection-time MSE gap with
        ``log(1 + ‖y_w − y_l‖²/D)`` before computing the GPO margin.
        Valid only without a reference model. Default: ``False``.
    output_filter : set[Hashable] | None
        Optional set of signal IDs supervised by this term.  ``None`` uses all
        signals present in the batch.
    """

    requires_native_target: bool = False
    requires_decode: bool = False
    requires_destandardize: bool = False

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        beta: float = _DEFAULT_BETA,
        loss_type: str = _DEFAULT_LOSS_TYPE,
        sft_weight: float = _DEFAULT_SFT_WEIGHT,
        mse_gap_clip: float | None = None,
        log_mse_gap: bool = False,
        output_filter: set[Hashable] | None = None,
    ) -> None:
        if beta <= 0.0:
            raise ValueError(f"ContinuousGPOLoss: beta must be > 0, got {beta!r}.")
        if loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(
                f"ContinuousGPOLoss: loss_type must be one of {sorted(_VALID_LOSS_TYPES)}, got {loss_type!r}."
            )
        if sft_weight < 0.0:
            raise ValueError(f"ContinuousGPOLoss: sft_weight must be >= 0, got {sft_weight!r}.")
        if mse_gap_clip is not None and float(mse_gap_clip) <= 0.0:
            raise ValueError(f"ContinuousGPOLoss: mse_gap_clip must be > 0, got {mse_gap_clip!r}.")

        self._beta = float(beta)
        self._loss_type: str = str(loss_type)
        self._sft_weight = float(sft_weight)
        self._mse_gap_clip: float | None = float(mse_gap_clip) if mse_gap_clip is not None else None
        self._log_mse_gap: bool = bool(log_mse_gap)
        self._output_filter = set(output_filter) if output_filter is not None else None

    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def validate_term_cfg(cls, term_def: Mapping[str, Any], path: str) -> None:
        """Validate GPO-specific config keys."""
        cls._validate_known_term_keys(term_def, path, allowed_specific_keys=_OWN_CFG_KEYS)

        beta = term_def.get("beta", _DEFAULT_BETA)
        if isinstance(beta, bool) or not isinstance(beta, (int, float)):
            raise TypeError(f"{path}.beta must be a positive number, got {beta!r}.")
        if float(beta) <= 0.0:
            raise ValueError(f"{path}.beta must be > 0, got {beta!r}.")

        loss_type = term_def.get("loss_type", _DEFAULT_LOSS_TYPE)
        if not isinstance(loss_type, str):
            raise TypeError(f"{path}.loss_type must be a string, got {loss_type!r}.")
        if loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(f"{path}.loss_type must be one of {sorted(_VALID_LOSS_TYPES)}, got {loss_type!r}.")

        sft_w = term_def.get("sft_weight", _DEFAULT_SFT_WEIGHT)
        if isinstance(sft_w, bool) or not isinstance(sft_w, (int, float)):
            raise TypeError(f"{path}.sft_weight must be a non-negative number, got {sft_w!r}.")
        if float(sft_w) < 0.0:
            raise ValueError(f"{path}.sft_weight must be >= 0, got {sft_w!r}.")

        clip = term_def.get("mse_gap_clip", None)
        if clip is not None:
            if isinstance(clip, bool) or not isinstance(clip, (int, float)):
                raise TypeError(f"{path}.mse_gap_clip must be a positive number or null, got {clip!r}.")
            if float(clip) <= 0.0:
                raise ValueError(f"{path}.mse_gap_clip must be > 0, got {clip!r}.")

        log_g = term_def.get("log_mse_gap", False)
        if not isinstance(log_g, bool):
            raise TypeError(f"{path}.log_mse_gap must be a boolean, got {log_g!r}.")

    # ------------------------------------------------------------------------------------------------------------------
    def compute(
        self,
        preds: Mapping[Hashable, Tensor],
        y_emb: Mapping[Hashable, Tensor],
        y_native: Mapping[Hashable, Tensor] | None,
        output_mask: Mapping[Hashable, Tensor],
        context: LossComputeContext | None = None,
        *,
        y_w_emb: Mapping[Hashable, Tensor] | None = None,
        y_l_emb: Mapping[Hashable, Tensor] | None = None,
        gpo_pair_mask: Mapping[Hashable, Tensor] | None = None,
        ref_preds: Mapping[Hashable, Tensor] | None = None,
    ) -> tuple[Tensor, dict[Hashable, float]]:
        """
        Compute the continuous GPO loss for one batch.

        ``y_w_emb``, ``y_l_emb``, and optionally ``ref_preds`` are passed as
        keyword-only arguments by the
        :class:`~mmt.train.losses.aggregator.LossAggregator` when this term is
        active — they are extracted from the batch dict which the GPO DataLoader
        injects before the forward pass.

        When ``y_l_emb`` is absent (e.g. a plain finetune batch without GPO
        pairs) the term silently returns zero, so it can coexist with other
        loss terms without crashing the pipeline.

        Parameters
        ----------
        preds : Mapping[Hashable, Tensor]
            Current model predictions in embedding space, shape (B, D) per signal.
        y_emb : Mapping[Hashable, Tensor]
            Ground-truth embeddings.  Used as the preferred anchor when
            ``y_w_emb`` is not given explicitly.
        y_native : Mapping[Hashable, Tensor] | None
            Unused; accepted for ``BaseLoss.compute`` signature compatibility.
        output_mask : Mapping[Hashable, Tensor]
            Boolean validity mask, shape (B,).
        context : LossComputeContext | None
            Optional metadata (unused in loss computation).
        y_w_emb : Mapping[Hashable, Tensor] | None
            Preferred embeddings keyed by signal_id.  Falls back to ``y_emb``
            when ``None``.
        y_l_emb : Mapping[Hashable, Tensor] | None
            Dispreferred embeddings keyed by signal_id.  When ``None`` or empty
            the term returns zero (no-op).
        gpo_pair_mask : Mapping[Hashable, Tensor] | None
            Boolean per-row mask identifying rows with an actual preference pair.
            When omitted, all rows selected by ``output_mask`` are treated as paired
            for backwards compatibility.
        ref_preds : Mapping[Hashable, Tensor] | None
            Frozen reference model predictions in embedding space, shape (B, D)
            per signal.  When provided, the DPO-style reference-anchored margin
            is used::

                margin = [d(ŷ, y_l) − d(ŷ_ref, y_l)] − [d(ŷ, y_w) − d(ŷ_ref, y_w)]

            This measures preference relative to the reference without bounding absolute policy drift.  When ``None``, the
            surrogate margin ``transform(d(y_w, y_l)) − d(ŷ, y_w)`` is used.

        Returns
        -------
        tuple[Tensor, dict[Hashable, float]]
            ``(scalar_loss, per_signal_logs)``

            Per-signal log keys produced:

            ``gpo/<signal_id>``        — raw GPO preference loss for that signal
            ``sft/<signal_id>``        — SFT co-loss value (0.0 when sft_weight=0)
            ``margin/<signal_id>``     — mean preference margin (higher = better)
            ``ref_margin/<signal_id>`` — mean reference model margin (diagnostic)
        """
        if not preds:
            raise RuntimeError("ContinuousGPOLoss received empty predictions from the model.")

        if ref_preds is not None and (self._mse_gap_clip is not None or self._log_mse_gap):
            raise ValueError("mse_gap_clip/log_mse_gap are incompatible with reference predictions.")

        # Fall back to y_emb as preferred anchor when y_w_emb not explicitly given.
        effective_y_w: Mapping[Hashable, Tensor] = y_w_emb if (y_w_emb is not None) else y_emb

        # If dispreferred embeddings are absent return zero (plain finetune batch).
        if not y_l_emb:
            ref = next(iter(preds.values()))
            return ref.sum() * 0.0, {}

        per_signal_losses: list[Tensor] = []
        logs: dict[Hashable, float] = {}

        for sig_id, y_pred in preds.items():
            if (self._output_filter is not None) and (sig_id not in self._output_filter):
                continue
            if sig_id not in effective_y_w:
                continue
            if sig_id not in y_l_emb:
                continue
            if sig_id not in output_mask:
                continue

            mask = output_mask[sig_id]
            if mask.dtype != torch.bool:
                raise RuntimeError(f"[{sig_id!r}] output_mask must be a bool tensor, got {mask.dtype}.")
            if gpo_pair_mask is not None and sig_id in gpo_pair_mask:
                pair_mask = gpo_pair_mask[sig_id]
                if pair_mask.dtype != torch.bool:
                    raise RuntimeError(f"[{sig_id!r}] gpo_pair_mask must be a bool tensor, got {pair_mask.dtype}.")
                if pair_mask.shape != mask.shape:
                    raise RuntimeError(
                        f"[{sig_id!r}] gpo_pair_mask shape {tuple(pair_mask.shape)} "
                        f"does not match output_mask shape {tuple(mask.shape)}."
                    )
                mask = mask & pair_mask
            if not bool(mask.any()):
                continue

            # Apply mask and cast to float32 for numerical stability.
            y_pred_f = y_pred[mask].to(torch.float32)  # (N, D)
            y_w_f = effective_y_w[sig_id][mask].to(torch.float32)  # (N, D)
            y_l_f = y_l_emb[sig_id][mask].to(torch.float32)  # (N, D)

            # Current model distances.
            d_w = (y_pred_f - y_w_f).square().mean(dim=1)  # (N,)
            d_l = (y_pred_f - y_l_f).square().mean(dim=1)  # (N,)

            # Collection-time fixed dispreferred distance ‖y_w − y_l‖²/D.
            # Used as the dispreferred anchor when no reference model is present.
            d_l_collect = (y_w_f - y_l_f).square().mean(dim=1)  # (N,) fixed
            if self._log_mse_gap:
                d_l_collect = torch.log1p(d_l_collect)
            if self._mse_gap_clip is not None:
                d_l_collect = d_l_collect.clamp(max=self._mse_gap_clip)

            # ------------------------------------------------------------------
            # Compute the GPO margin.
            #
            # With reference model (relative distance margin; not a KL constraint):
            #   margin = [d(ŷ, y_l) − d(ŷ_ref, y_l)] − [d(ŷ, y_w) − d(ŷ_ref, y_w)]
            #
            #   Orthogonal prediction drift is unconstrained by this margin.
            #
            # Without reference model (fallback, uses fixed collection-time anchor):
            #   margin = d_l_collect − d_w
            # ------------------------------------------------------------------
            if ref_preds is not None and sig_id in ref_preds:
                y_ref_f = ref_preds[sig_id][mask].to(torch.float32)  # (N, D)
                d_w_ref = (y_ref_f - y_w_f).square().mean(dim=1)  # (N,)
                d_l_ref = (y_ref_f - y_l_f).square().mean(dim=1)  # (N,)
                # margin = [d(ŷ,y_l) − d(ŷ,y_w)] − [d(ŷ_ref,y_l) − d(ŷ_ref,y_w)]
                margin = (d_l - d_w) - (d_l_ref - d_w_ref)  # (N,)
                logs[f"ref_margin/{sig_id}"] = float((d_l_ref - d_w_ref).mean().detach().cpu())
            else:
                # No reference model: use fixed collection-time anchor.
                margin = d_l_collect - d_w  # (N,)

            # ------------------------------------------------------------------
            # Apply the selected preference-optimisation objective.
            #
            # DPO: −log σ(β·margin)
            #   dL/dmargin = -β·σ(-β·margin); at zero it is -β/2.
            #   The collection-gap beta rule is a scale heuristic, not a reference-margin calibration.
            #
            # IPO: (margin − 1/(2β))²
            #   Gradient = 2·(margin − 1/(2β)); zero at the target.
            #   Fixed point: margin = 1/(2β).
            #   Its scale and target differ from DPO and must be assessed on validation.
            # ------------------------------------------------------------------
            if self._loss_type == "ipo":
                # IPO: (margin − 1/(2β))²  — constant gradient at margin=0
                ipo_target = 0.5 / self._beta  # 1/(2β)
                gpo_loss = (margin - ipo_target).square().mean()
            elif self._loss_type == "slic":
                # SLiC: max(0, δ − margin)  where δ = 1/(2β)
                # Zero gradient once the margin exceeds the target threshold.
                slic_target = 0.5 / self._beta  # 1/(2β)
                gpo_loss = F.relu(slic_target - margin).mean()
            elif self._loss_type == "hinge":
                # Hinge: max(0, 1 − margin)  — fixed target of 1, β-independent
                gpo_loss = F.relu(1.0 - margin).mean()
            else:  # "dpo"
                gpo_loss = -F.logsigmoid(self._beta * margin).mean()

            term_loss: Tensor = gpo_loss

            # Optional SFT co-loss: keep the model anchored to the preferred response.
            if self._sft_weight > 0.0:
                sft_loss = (y_pred_f - y_w_f).square().mean()
                term_loss = term_loss + self._sft_weight * sft_loss
                logs[f"sft/{sig_id}"] = float(sft_loss.detach().cpu())
            else:
                logs[f"sft/{sig_id}"] = 0.0

            logs[f"count/{sig_id}"] = int(mask.sum())
            logs[f"loss/{sig_id}"] = float(term_loss.detach().cpu())
            logs[f"pref_acc/{sig_id}"] = float((d_w < d_l).float().mean().cpu())
            logs[f"gpo/{sig_id}"] = float(gpo_loss.detach().cpu())
            logs[f"margin/{sig_id}"] = float(margin.mean().detach().cpu())
            # d_w: mean distance to ground truth on pair windows.
            # Rising d_w with rising pref_acc = reward hacking (moving away from y_w
            # while still beating y_l). Should ideally decrease alongside pref_acc rising.
            logs[f"d_w/{sig_id}"] = float(d_w.mean().detach().cpu())

            per_signal_losses.append(term_loss)

        if not per_signal_losses:
            ref = next(iter(preds.values()))
            return ref.sum() * 0.0, logs

        return torch.stack(per_signal_losses).mean(), logs


# ======================================================================================================================
# Standalone metric helper (not a loss term)
# ======================================================================================================================


def gpo_preference_accuracy(
    preds: Mapping[Hashable, Tensor],
    y_w_emb: Mapping[Hashable, Tensor],
    y_l_emb: Mapping[Hashable, Tensor],
    output_mask: Mapping[Hashable, Tensor],
    pair_mask: Mapping[Hashable, Tensor] | None = None,
) -> dict[Hashable, float]:
    """
    Compute per-signal preference accuracy.

    Returns the fraction of windows where the current model prediction is
    strictly closer to ``y_w_emb`` than to ``y_l_emb`` (measured by MSE).

    This is a pure diagnostic metric, not a differentiable loss term.  Use it
    during GPO validation to track how often the model already "prefers" the
    right answer before / after training.

    Parameters
    ----------
    preds : Mapping[Hashable, Tensor]
        Model predictions in embedding space, shape (B, D) per signal.
    y_w_emb : Mapping[Hashable, Tensor]
        Preferred embeddings, shape (B, D) per signal.
    y_l_emb : Mapping[Hashable, Tensor]
        Dispreferred embeddings, shape (B, D) per signal.
    output_mask : Mapping[Hashable, Tensor]
        Boolean validity mask, shape (B,) per signal.
    pair_mask : Mapping[Hashable, Tensor] | None
        Optional Boolean mask selecting rows with actual preference pairs.

    Returns
    -------
    dict[Hashable, float]
        Signal-id → preference accuracy in [0, 1].
    """
    results: dict[Hashable, float] = {}

    for sig_id, y_pred in preds.items():
        if sig_id not in y_w_emb or sig_id not in y_l_emb or sig_id not in output_mask:
            continue

        mask = output_mask[sig_id]
        if pair_mask is not None and sig_id in pair_mask:
            mask = mask & pair_mask[sig_id]
        if not bool(mask.any()):
            continue

        y_pred_f = y_pred[mask].to(torch.float32)
        y_w_f = y_w_emb[sig_id][mask].to(torch.float32)
        y_l_f = y_l_emb[sig_id][mask].to(torch.float32)

        d_w = (y_pred_f - y_w_f).square().mean(dim=1)
        d_l = (y_pred_f - y_l_f).square().mean(dim=1)

        results[sig_id] = float((d_w < d_l).float().mean().detach().cpu())

    return results
