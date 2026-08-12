"""
Continuous Ranked Probability Score (CRPS) scoring rules.

Pure, dependency-light (torch only) CRPS formulas, shared by training losses (`mmt.train.losses`) and evaluation
(`mmt.eval`). CRPS is a *proper scoring rule*: the same formula is used whether it is a training objective or a
reported metric, so it belongs in a neutral scoring module rather than inside a loss term or an eval helper.

Three variants:
- ``gaussian_crps``  : closed-form CRPS of a univariate Gaussian (used by the closed-form embedding-space loss).
- ``sample_crps``    : fair (unbiased) marginal CRPS from a finite sample ensemble (sampled losses + eval).
- ``point_crps``     : CRPS of a point (deterministic) forecast, which is exactly the absolute error.

All are *marginal* (per-element): for multi-dimensional outputs each coordinate is scored independently and spatial
dependence between coordinates is not rewarded.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from mmt.constants import FLOAT_STABILITY_EPS


# Standard-normal constants.
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


# ----------------------------------------------------------------------------------------------------------------------
def gaussian_crps(mu: Tensor, sigma: Tensor, y: Tensor) -> Tensor:
    """
    Element-wise closed-form CRPS of a univariate Gaussian against an observation.

    For a Gaussian predictive distribution ``N(mu, sigma^2)`` the CRPS has the closed form

        CRPS(N(mu, sigma^2), y) = sigma * [ omega * (2 * Phi(omega) - 1) + 2 * phi(omega) - 1 / sqrt(pi) ]

    with ``omega = (y - mu) / sigma`` and ``Phi`` / ``phi`` the standard-normal CDF / PDF.

    Parameters
    ----------
    mu : Tensor
        Predictive means. Any shape.
    sigma : Tensor
        Predictive standard deviations, strictly positive. Broadcastable to `mu`.
    y : Tensor
        Observations. Broadcastable to `mu`.

    Returns
    -------
    Tensor
        Element-wise CRPS, same broadcast shape as the inputs. Non-negative.

    """

    # sigma is strictly positive (softplus + eps in the head); clamp defensively against any degenerate input.
    sigma = sigma.clamp_min(FLOAT_STABILITY_EPS)
    omega = (y - mu) / sigma
    cdf = 0.5 * (1.0 + torch.erf(omega * _INV_SQRT_2))  # Phi(omega)
    pdf = _INV_SQRT_2PI * torch.exp(-0.5 * omega * omega)  # phi(omega)
    return sigma * (omega * (2.0 * cdf - 1.0) + 2.0 * pdf - _INV_SQRT_PI)


# ----------------------------------------------------------------------------------------------------------------------
def sample_crps(samples: Tensor, y: Tensor, *, sample_dim: int = 1) -> Tensor:
    """
    Fair (unbiased) marginal CRPS estimator from a finite sample ensemble.

    The CRPS of a predictive distribution `F` against `y` admits the energy-form identity
    ``CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|`` (`X, X'` independent draws from `F`). Given `S >= 2` samples it is
    estimated unbiasedly ("fair" estimator) by

        CRPS ≈ (1/S) Σ_s |x_s - y| - (1 / (2 S (S - 1))) Σ_{s,t} |x_s - x_t|

    where the double sum runs over all ordered pairs (diagonal contributes 0). The biased `1/S^2` normalization is
    avoided because it under-penalizes spread and encourages sample collapse during training. The pairwise term is
    computed via a sort identity rather than the `(S, S)` matrix, keeping memory at the size of the sample tensor:
    for ascending order statistics ``x_(1) <= ... <= x_(S)``, ``Σ_{s<t} |x_s - x_t| = Σ_i (2 i - S - 1) x_(i)``.

    Parameters
    ----------
    samples : Tensor
        Predictive samples with an explicit sample axis at `sample_dim`. Shape: `(..., S, ...)` with `S >= 2`.
    y : Tensor
        Observations, matching `samples` with the `sample_dim` axis removed. Broadcastable to that shape.
    sample_dim : int
        Axis of `samples` holding the `S` draws.
        Optional. Default: 1.

    Returns
    -------
    Tensor
        Element-wise CRPS with the sample axis reduced (same shape as `y`). Non-negative up to estimator noise.

    Raises
    ------
    ValueError
        If `samples` has fewer than 2 entries along `sample_dim`.

    """

    n_samples = samples.shape[sample_dim]
    if n_samples < 2:
        raise ValueError(f"sample_crps requires at least 2 samples along sample_dim={sample_dim}, got {n_samples}.")

    # term1 = mean_s |x_s - y|
    abs_dev = (samples - y.unsqueeze(sample_dim)).abs()
    term1 = abs_dev.mean(dim=sample_dim)

    # term2 = Σ_{s<t} |x_s - x_t| / (S (S - 1)), via the sorted order-statistic identity.
    sorted_x, _ = torch.sort(samples, dim=sample_dim)
    idx = torch.arange(1, n_samples + 1, device=samples.device, dtype=samples.dtype)
    coeff = 2.0 * idx - n_samples - 1.0  # (S,)
    coeff_shape = [1] * samples.dim()
    coeff_shape[sample_dim] = n_samples
    pair_sum = (coeff.reshape(coeff_shape) * sorted_x).sum(dim=sample_dim)  # Σ_{s<t} |x_s - x_t|
    term2 = pair_sum / (n_samples * (n_samples - 1))

    return term1 - term2


# ----------------------------------------------------------------------------------------------------------------------
def point_crps(pred: Tensor, y: Tensor) -> Tensor:
    """
    Element-wise CRPS of a point (deterministic) forecast.

    A point forecast is a degenerate distribution (a Dirac delta at `pred`), and its CRPS is exactly the absolute
    error ``|pred - y|``. This makes CRPS a common axis for probabilistic and deterministic models.

    Parameters
    ----------
    pred : Tensor
        Point predictions. Any shape.
    y : Tensor
        Observations. Broadcastable to `pred`.

    Returns
    -------
    Tensor
        Element-wise ``|pred - y|``.

    """

    return (pred - y).abs()
