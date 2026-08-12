"""
Predictive-distribution contract shared by output heads, losses, and evaluation.

This is a dependency-light leaf module (torch only) so both ``mmt.models`` (the producer) and ``mmt.train.losses`` /
``mmt.eval`` (the consumers) can import it without pulling in the heavy ``mmt.models`` package init — which would
create an import cycle.

It defines:
- the family tag carried in each ``pred_dist`` entry, so a ``pred_dist``-consuming loss/metric can assert it is
  scoring the distribution family the head was built for, and
- ``sample_gaussian_dist``, the single reparameterized Gaussian sampler used by both training and evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


# Family tag carried in each pred_dist entry. Gaussian is currently the only probabilistic head.
PRED_DIST_FAMILY_KEY = "family"
GAUSSIAN_DIST_FAMILY = "gaussian"


# ----------------------------------------------------------------------------------------------------------------------
def sample_gaussian_dist(
    dist: Mapping[str, Any],
    n_samples: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Draw reparameterized samples from a Gaussian ``pred_dist`` entry.

    Single sampling primitive shared by the model forward (eval/inference) and the training loss, so both draw
    ``mu + sigma * eps`` identically. Operates on an already-computed distribution dict rather than recomputing it.
    This is the natural dispatch point once a second distribution family is added.

    Parameters
    ----------
    dist : Mapping[str, Any]
        Distribution parameters with ``"family"``, ``"mu"`` and ``"sigma"`` (each tensor shaped ``(B, K_t)``).
    n_samples : int
        Number of samples to draw. Must be >= 1.
    generator : torch.Generator | None
        Optional PyTorch random generator. Must live on the same device as ``mu``.

    Returns
    -------
    torch.Tensor
        Samples shaped ``(B, n_samples, K_t)``.

    Raises
    ------
    ValueError
        If ``n_samples`` is not positive.
    RuntimeError
        If ``dist`` is not a Gaussian distribution.
    """

    if int(n_samples) <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")

    family = dist.get(PRED_DIST_FAMILY_KEY)
    if family != GAUSSIAN_DIST_FAMILY:
        raise RuntimeError(f"sample_gaussian_dist expects a Gaussian distribution, got family={family!r}.")

    mu = dist["mu"]
    sigma = dist["sigma"]
    eps = torch.randn(
        (mu.shape[0], int(n_samples), mu.shape[1]),
        device=mu.device,
        dtype=mu.dtype,
        generator=generator,
    )
    return mu.unsqueeze(1) + sigma.unsqueeze(1) * eps
