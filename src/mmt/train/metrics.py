"""
metrics.py — Sample-counted epoch metrics and checkpoint selection for MMT training.

This module keeps epoch reporting separate from per-batch gradient computation. It provides:

    • Embedding MSE over every valid target, including windows without preference pairs.
    • Independent per-signal sums and observation counts for loss terms and diagnostics.
    • Separate window, batch and per-signal preference-pair coverage.
    • Explicit selection of validation MSE or the epoch objective for checkpointing and early stopping.

Aggregation rules
-----------------
MSE first averages coefficient errors within a window, then windows within a signal. The global MSE is the unweighted
mean of observed signal means. Loss terms use their own valid observations and configured signal weights; their
epoch means are combined using active term weights. A diagnostic uses only observations where it was recorded.
Changing batch partitioning preserves these reductions up to floating-point rounding; batch coverage may change.

Integration
-----------
run_one_epoch() creates one EpochMetrics instance and updates it after each forward pass. train_finetune() selects
the validation metric explicitly. These utilities are shared training infrastructure: they also support ordinary
MSE-only training and do not require CGPO pairs, a reference model or a separate data pipeline. They do not change
backpropagation or claim that optimization trajectories are invariant to batch size.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Hashable

import torch


# ======================================================================================================================
class EpochMetrics:
    """
    Accumulate per-signal epoch statistics independently of preference-pair availability.

    Parameters
    ----------
    terms : list[tuple[str, float, dict[str, float]]]
        Loss log prefixes, term weights and per-signal weights supplied by ``LossAggregator.epoch_terms``.
        Signal identifiers are stringified to match log keys.

    Notes
    -----
    Construct one accumulator per epoch. ``update()`` detaches diagnostic calculations from autograd;
    ``finish()`` reduces the accumulated observations without resetting them. Missing observations are not
    treated as zero loss, and an unobserved epoch objective is represented by NaN.
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, terms: list[tuple[str, float, dict[str, float]]]) -> None:
        """
        Initialize empty sum and count tables for the configured loss terms.

        Parameters
        ----------
        terms : list[tuple[str, float, dict[str, float]]]
            Term prefixes, scalar weights and per-signal weights for epoch reduction.
        """

        self.terms = terms
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)
        self.metric_counts = defaultdict(int)
        self.mse_sums = defaultdict(float)
        self.mse_counts = defaultdict(int)
        self.windows = self.pair_windows = self.batches = self.pair_batches = 0
        self.signal_pairs = defaultdict(int)
        self.signal_windows = defaultdict(int)

    # ------------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def update(
        self,
        preds: Mapping[Hashable, torch.Tensor],
        batch: Mapping[str, Any],
        logs: Mapping[str, float],
    ) -> None:
        """
        Accumulate one batch of predictions, targets, preference masks and loss diagnostics.

        Parameters
        ----------
        preds : Mapping[Hashable, torch.Tensor]
            Nonempty prediction mapping keyed by signal ID, with tensors shaped ``(B, D)``.
        batch : Mapping[str, Any]
            Collated targets and boolean validity masks under ``output_emb`` and ``output_mask``. Optional
            ``y_l_emb`` and ``gpo_pair_mask`` identify available preference pairs.
        logs : Mapping[str, float]
            Per-batch loss logs containing ``<term>/count/<signal>`` and per-signal means. Each diagnostic is
            accumulated only when its key is present.

        Returns
        -------
        None

        Notes
        -----
        Target MSE is calculated independently of active loss terms. CPU float64 accumulation supports predictions
        on MPS as well as CPU/CUDA. A paired window has at least one valid signal with a preference pair.
        """

        size = next(iter(preds.values())).shape[0]
        paired = torch.zeros(size, dtype=torch.bool, device=next(iter(preds.values())).device)
        self.batches += 1
        self.windows += size
        for sig, pred in preds.items():
            mask = batch.get("output_mask", {}).get(sig)
            target = batch.get("output_emb", {}).get(sig)
            if mask is None:
                continue
            n = int(mask.sum())
            self.signal_windows[str(sig)] += n
            if target is not None and n:
                # CPU float64 also works when predictions live on MPS (no float64 support).
                error = pred[mask].to(device="cpu", dtype=torch.float64) - target[mask].to(
                    device="cpu", dtype=torch.float64
                )
                values = error.square().flatten(1).mean(1)
                self.mse_sums[str(sig)] += float(values.sum())
                self.mse_counts[str(sig)] += n
            if sig in batch.get("y_l_emb", {}):
                pair_mask = batch.get("gpo_pair_mask", {}).get(sig, torch.ones_like(mask)) & mask
                paired |= pair_mask
                self.signal_pairs[str(sig)] += int(pair_mask.sum())
        self.pair_windows += int(paired.sum())
        self.pair_batches += int(paired.any())
        for prefix, _, _ in self.terms:
            count_prefix = f"{prefix}/count/"
            for key, value in logs.items():
                if not key.startswith(count_prefix):
                    continue
                sig = key[len(count_prefix) :]
                n = int(value)
                self.counts[f"{prefix}/{sig}"] += n
                for metric in ("loss", "gpo", "sft", "margin", "ref_margin", "d_w", "pref_acc"):
                    name = f"{prefix}/{metric}/{sig}"
                    if name in logs:
                        self.sums[name] += logs[name] * n
                        self.metric_counts[name] += n

    # ------------------------------------------------------------------------------------------------------------------
    def finish(self) -> tuple[float, dict[str, float]]:
        """
        Reduce accumulated observations to epoch means and an explicitly weighted objective.

        Returns
        -------
        tuple[float, dict[str, float]]
            Epoch objective and logs containing per-signal means/sums/counts, global MSE, term means and pair
            coverage. Global MSE is omitted when no valid targets were observed; the objective is NaN when no
            observed terms have positive combined weight. Counts are integers or integer-valued floats.

        Notes
        -----
        An unobserved term is excluded from objective normalization. A signal-weight mapping with nonpositive
        observed total weight falls back to a uniform signal mean, matching the loss-term reduction convention.
        """

        logs = {
            "batches": float(self.batches),
            "windows": float(self.windows),
            "pair_batches": float(self.pair_batches),
            "pair_windows": float(self.pair_windows),
            "pair_batch_coverage": self.pair_batches / max(1, self.batches),
            "pair_window_coverage": self.pair_windows / max(1, self.windows),
        }
        for sig, count in self.mse_counts.items():
            logs[f"mse/count/{sig}"] = count
            logs[f"mse/sum/{sig}"] = self.mse_sums[sig]
            logs[f"mse/{sig}"] = self.mse_sums[sig] / count
        if self.mse_counts:
            logs["mse"] = sum(self.mse_sums[s] / n for s, n in self.mse_counts.items()) / len(self.mse_counts)
        for sig, n in self.signal_windows.items():
            logs[f"pairs/count/{sig}"] = self.signal_pairs[sig]
            logs[f"outputs/count/{sig}"] = n
            logs[f"pairs/coverage/{sig}"] = self.signal_pairs[sig] / max(1, n)
        active = []
        for prefix, weight, output_weights in self.terms:
            signals = {k[len(prefix) + 1 :]: n for k, n in self.counts.items() if k.startswith(prefix + "/") and n}
            for sig, count in signals.items():
                logs[f"{prefix}/count/{sig}"] = count
            for key, total in self.sums.items():
                if key.startswith(prefix + "/"):
                    logs[key] = total / self.metric_counts[key]
                    logs[key.replace(prefix + "/", prefix + "/count/", 1)] = self.metric_counts[key]
                    logs[key.replace(prefix + "/", prefix + "/sum/", 1)] = total
            if not signals:
                continue
            weights = {s: float(output_weights.get(s, 1.0)) for s in signals}
            if sum(weights.values()) <= 0:
                weights = dict.fromkeys(signals, 1.0)
            term_mean = sum(logs[f"{prefix}/loss/{s}"] * weights[s] for s in signals) / sum(weights.values())
            logs[f"{prefix}/mean"] = term_mean
            active.append((prefix, weight, term_mean))
        denom = sum(weight for _, weight, _ in active)
        objective = sum(weight * mean for _, weight, mean in active) / denom if denom > 0 else float("nan")
        for prefix, weight, mean in active:
            logs[f"{prefix}/weighted"] = weight * mean / denom if denom > 0 else 0.0
        logs["objective"] = objective
        return objective, logs


# ----------------------------------------------------------------------------------------------------------------------
def select_metric(name: str, objective: float, logs: Mapping[str, float]) -> float:
    """
    Select the finite validation value used for best-checkpoint selection and early stopping.

    Parameters
    ----------
    name : str
        Configured metric: ``mse`` for macro-average embedding MSE, or ``objective`` for the epoch objective.
    objective : float
        Epoch objective returned by ``EpochMetrics.finish()``.
    logs : Mapping[str, float]
        Validation logs containing ``mse`` when at least one valid target was observed.

    Returns
    -------
    float
        Selected validation value; lower is better.

    Raises
    ------
    ValueError
        If the metric name is unsupported.
    RuntimeError
        If the requested metric has no observations or its value is non-finite.
    """

    if name not in {"mse", "objective"}:
        raise ValueError(f"Unknown checkpoint metric {name!r}; expected mse or objective.")
    value = objective if name == "objective" else logs.get("mse", float("nan"))
    if not math.isfinite(value):
        raise RuntimeError(f"Checkpoint metric {name!r} has no valid observations or is non-finite.")
    return value
