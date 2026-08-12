"""
Output adapters for MMT.

Each output signal has a small adapter network that maps from the modality
latent space (G_mod) to the target output embedding dimension (K_t).

Adapters are lightweight (linear or a tiny MLP with one hidden layer) and are
keyed by stable canonical keys ("output:<name>") to ensure predictable
checkpoint loading and warm-start across tasks.
"""

from __future__ import annotations

from typing import Any, Hashable
from collections.abc import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmt.distributions import GAUSSIAN_DIST_FAMILY, PRED_DIST_FAMILY_KEY


OutputAdapterItem = tuple[Hashable, str, torch.Tensor]


# ----------------------------------------------------------------------------------------------------------------------
def _activation_layer(activation: str) -> nn.Module:
    """Build the requested nonlinearity for a small output adapter."""

    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported output-adapter activation={activation!r}.")


# ======================================================================================================================
class OutputAdapter(nn.Module):
    """
    Per-output adapter: group latent (G_mod) -> output embedding (K_t).
    Optionally with a tiny hidden layer.

    Attributes
    ----------
    net = nn.Sequential
        ModalityHead's network.
    out_dim : int
        Output dimension.

    Methods
    -------
    forward(h)
        OutputAdapter's forward function.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 0, activation: str = "relu"):
        """

        Initialize class attributes.

        Parameters
        ----------
        in_dim : int
            Input dimension.
        out_dim : int
            Output dimension.
        hidden_dim : int
            Dimension of hidden layer.
        activation : str
            Activation applied after the optional hidden projection. Supported
            values are ``"relu"`` and ``"gelu"``.
            Optional. Default: ``"relu"``.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking mistakes.

        """

        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_features=in_dim, out_features=hidden_dim),
                _activation_layer(activation),
                nn.Linear(in_features=hidden_dim, out_features=out_dim),
            )
        else:
            self.net = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.out_dim = int(out_dim)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        OutputAdapter's forward function.

        Parameters
        ----------
        h : torch.Tensor
            Input for the network.

        Returns
        -------
        torch.Tensor
            Forward pass over specified `h`.

        """

        return self.net(h)  # (B, K_t)


# ======================================================================================================================
class ProbabilisticOutputAdapter(nn.Module):
    """
    Per-output latent Gaussian adapter.

    The adapter emits two vectors for each output signal: a mean and a raw scale parameter. ``forward()`` returns only
    the mean so deterministic training/eval paths can keep using ``pred[sid]`` as ``(B, K_t)``. Call
    :meth:`distribution` to access the full Gaussian parameters; consumers draw samples from those parameters via
    :func:`sample_gaussian_dist`.

    Attributes
    ----------
    net : nn.Module
        Linear or tiny MLP that maps hidden features to the predictive mean. The attribute name intentionally matches
        ``OutputAdapter.net`` so deterministic checkpoints can warm-start the Gaussian mean head.
    raw_scale_net : nn.Module
        Linear or tiny MLP that maps hidden features to raw scale logits.
    out_dim : int
        Output embedding dimension ``K_t`` for one output signal.
    scale_eps : float
        Small positive constant added to ``softplus(raw_scale)``.
    raw_scale_min : float
        Lower clamp bound for raw scale logits before ``softplus``.
    raw_scale_max : float
        Upper clamp bound for raw scale logits before ``softplus``.

    Methods
    -------
    distribution(h)
        Return latent Gaussian parameters for hidden features ``h``.
    forward(h)
        Return the predictive mean, preserving the deterministic adapter contract.
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 0,
        scale_eps: float = 1e-6,
        raw_scale_min: float = -20.0,
        raw_scale_max: float = 20.0,
        raw_scale_bias_init: float = -5.0,
        activation: str = "relu",
    ) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        in_dim : int
            Input hidden dimension.
        out_dim : int
            Output embedding dimension ``K_t``.
        hidden_dim : int
            Optional hidden-layer dimension. If 0, use a single linear projection.
            Optional. Default: 0.
        scale_eps : float
            Small positive constant added to ``softplus(raw_scale)``.
            Optional. Default: 1e-6.
        raw_scale_min : float
            Lower clamp bound for raw scale logits before ``softplus``.
            Optional. Default: -20.0.
        raw_scale_max : float
            Upper clamp bound for raw scale logits before ``softplus``.
            Optional. Default: 20.0.
        raw_scale_bias_init : float
            Initial bias value for raw scale outputs.
            Optional. Default: -5.0.
        activation : str
            Activation applied after the optional hidden projection. Supported
            values are ``"relu"`` and ``"gelu"``.
            Optional. Default: ``"relu"``.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking mistakes.

        """

        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_features=in_dim, out_features=hidden_dim),
                _activation_layer(activation),
                nn.Linear(in_features=hidden_dim, out_features=out_dim),
            )
            self.raw_scale_net = nn.Sequential(
                nn.Linear(in_features=in_dim, out_features=hidden_dim),
                _activation_layer(activation),
                nn.Linear(in_features=hidden_dim, out_features=out_dim),
            )
        else:
            self.net = nn.Linear(in_features=in_dim, out_features=out_dim)
            self.raw_scale_net = nn.Linear(in_features=in_dim, out_features=out_dim)

        self.out_dim = int(out_dim)
        self.scale_eps = float(scale_eps)
        self.raw_scale_min = float(raw_scale_min)
        self.raw_scale_max = float(raw_scale_max)
        self._init_raw_scale_bias(raw_scale_bias_init=float(raw_scale_bias_init))

    # ------------------------------------------------------------------------------------------------------------------
    def _init_raw_scale_bias(self, raw_scale_bias_init: float) -> None:
        """
        Initialize the raw-scale half of the final bias to a small initial sigma.

        Parameters
        ----------
        raw_scale_bias_init : float
            Initial bias value for raw scale outputs.

        Returns
        -------
        None
        """

        last: nn.Linear | None = None
        net = self.raw_scale_net
        if isinstance(net, nn.Linear):
            last = net
        elif isinstance(net, nn.Sequential):
            candidate = net[-1]
            if isinstance(candidate, nn.Linear):
                last = candidate

        if last is not None and last.bias is not None:
            with torch.no_grad():
                last.bias.fill_(raw_scale_bias_init)

    # ------------------------------------------------------------------------------------------------------------------
    def distribution(self, h: torch.Tensor) -> dict[str, Any]:
        """
        Return latent Gaussian parameters for hidden features ``h``.

        Parameters
        ----------
        h : torch.Tensor
            Hidden features of shape ``(B, in_dim)``.

        Returns
        -------
        dict[str, Any]
            ``{"family", "mu", "raw_scale", "sigma", "log_sigma"}``. ``"family"`` is the string distribution tag
            (``"gaussian"``); the remaining values are tensors shaped ``(B, K_t)``.
        """

        mu = self.net(h)
        raw_scale = self.raw_scale_net(h)
        raw_scale = raw_scale.clamp(min=self.raw_scale_min, max=self.raw_scale_max)
        sigma = F.softplus(raw_scale) + self.scale_eps
        log_sigma = sigma.log()
        # The family tag lets a pred_dist-consuming loss assert it is scoring the distribution it was built for.
        return {
            PRED_DIST_FAMILY_KEY: GAUSSIAN_DIST_FAMILY,
            "mu": mu,
            "raw_scale": raw_scale,
            "sigma": sigma,
            "log_sigma": log_sigma,
        }

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Return the predictive mean, preserving the deterministic ``OutputAdapter`` contract.

        Parameters
        ----------
        h : torch.Tensor
            Hidden features of shape ``(B, in_dim)``.

        Returns
        -------
        torch.Tensor
            Predictive mean shaped ``(B, K_t)``.
        """

        return self.distribution(h)["mu"]


# ----------------------------------------------------------------------------------------------------------------------
def apply_output_adapters(
    *,
    output_adapters: nn.ModuleDict,
    items: Iterable[OutputAdapterItem],
    output_adapter_type: str,
) -> dict[str, Any]:
    """
    Apply deterministic or Gaussian output adapters to prepared output hidden states.

    The model emits only the predictive distribution; sampling is the consumer's responsibility (training loss /
    eval), which draws from ``pred_dist`` via :func:`sample_gaussian_dist`.

    Parameters
    ----------
    output_adapters : nn.ModuleDict
        Per-output adapters keyed by canonical output signal key.
    items : Iterable[OutputAdapterItem]
        Iterable of ``(output_key, adapter_key, h_out)`` tuples. ``output_key`` is usually the numeric signal id,
        ``adapter_key`` indexes ``output_adapters``, and ``h_out`` is the hidden tensor for that output.
    output_adapter_type : str
        Output adapter type. One of ``"deterministic"`` or ``"gaussian"``.

    Returns
    -------
    dict[str, Any]
        Mapping containing ``"pred"`` for all adapter types. Gaussian adapters additionally include ``"pred_dist"``.

    Raises
    ------
    TypeError
        If an adapter's runtime class does not match ``output_adapter_type``.
    ValueError
        If ``output_adapter_type`` is unsupported.
    """

    adapter_type = str(output_adapter_type)
    if adapter_type not in {"deterministic", "gaussian"}:
        raise ValueError(f"Unsupported output_adapter_type={adapter_type!r}. Expected 'deterministic' or 'gaussian'.")

    preds: dict[Hashable, torch.Tensor] = {}
    pred_dist: dict[Hashable, dict[str, Any]] = {}

    for out_key, adapter_key, h_out in items:
        adapter = output_adapters[adapter_key]

        if adapter_type == "gaussian":
            if not isinstance(adapter, ProbabilisticOutputAdapter):
                raise TypeError(f"Expected ProbabilisticOutputAdapter for adapter_key={adapter_key!r}.")

            dist = adapter.distribution(h_out)
            pred_dist[out_key] = dist
            preds[out_key] = dist["mu"]
        else:
            if not isinstance(adapter, OutputAdapter):
                raise TypeError(f"Expected OutputAdapter for adapter_key={adapter_key!r}.")

            preds[out_key] = adapter(h_out)

    out: dict[str, Any] = {"pred": preds}
    if adapter_type == "gaussian":
        out["pred_dist"] = pred_dist

    return out


# ----------------------------------------------------------------------------------------------------------------------
def apply_output_residual(
    *,
    adapter_output: dict[str, Any],
    baseline_emb: Mapping[Hashable, torch.Tensor] | None,
    residual_output_ids: set[Hashable],
) -> dict[str, Any]:
    """
    Add output-space persistence baselines to predicted corrections.

    Parameters
    ----------
    adapter_output : dict[str, Any]
        Result from :func:`apply_output_adapters`.
    baseline_emb : Mapping[Hashable, torch.Tensor] | None
        Batched baseline embeddings keyed by output signal ID.
    residual_output_ids : set[Hashable]
        Output IDs whose adapters represent residual corrections.

    Returns
    -------
    dict[str, Any]
        Adapter output with absolute predictions. Gaussian ``mu`` values are
        shifted by the same baseline while scale parameters remain unchanged.

    Raises
    ------
    KeyError
        If residual outputs are enabled but a required baseline is missing.
    ValueError
        If a baseline and prediction have different shapes.
    """

    if not residual_output_ids:
        return adapter_output
    if not isinstance(baseline_emb, Mapping):
        raise KeyError("Residual outputs require batch['output_baseline_emb'].")

    preds = adapter_output.get("pred") or {}
    pred_dist = adapter_output.get("pred_dist") or {}
    for output_id in residual_output_ids:
        if output_id not in preds:
            continue
        if output_id not in baseline_emb:
            raise KeyError(f"Missing residual baseline for output signal ID {output_id!r}.")

        pred = preds[output_id]
        baseline = baseline_emb[output_id].to(device=pred.device, dtype=pred.dtype)
        if baseline.shape != pred.shape:
            raise ValueError(
                f"Residual baseline shape {tuple(baseline.shape)} != prediction shape {tuple(pred.shape)} "
                f"for output signal ID {output_id!r}."
            )

        absolute = pred + baseline
        preds[output_id] = absolute
        if output_id in pred_dist:
            dist = dict(pred_dist[output_id])
            dist["mu"] = absolute
            pred_dist[output_id] = dist

    adapter_output["pred"] = preds
    if pred_dist:
        adapter_output["pred_dist"] = pred_dist
    return adapter_output


# ----------------------------------------------------------------------------------------------------------------------
def zero_initialize_output_corrections(*, output_adapters: nn.ModuleDict, adapter_keys: Iterable[str]) -> None:
    """
    Zero the final predictive-mean layer for residual output adapters.

    Parameters
    ----------
    output_adapters : nn.ModuleDict
        Output adapter modules keyed by canonical output key.
    adapter_keys : Iterable[str]
        Adapter keys whose predictions represent residual corrections.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If an adapter has no supported predictive-mean network.
    """

    for adapter_key in adapter_keys:
        adapter = output_adapters[adapter_key]
        net = getattr(adapter, "net", None)
        last: nn.Linear | None = None
        if isinstance(net, nn.Linear):
            last = net
        elif isinstance(net, nn.Sequential) and isinstance(net[-1], nn.Linear):
            last = net[-1]
        if last is None:
            raise TypeError(f"Output adapter {adapter_key!r} has no supported predictive-mean layer.")

        with torch.no_grad():
            last.weight.zero_()
            if last.bias is not None:
                last.bias.zero_()


# ----------------------------------------------------------------------------------------------------------------------
def resolve_gaussian_adapter_cfg(output_adapters_cfg: Mapping[str, Any] | None) -> dict[str, float]:
    """
    Return explicitly configured Gaussian output adapter settings.

    Parameters
    ----------
    output_adapters_cfg : Mapping[str, Any] | None
        ``model.output_adapters`` config block. Settings are read from the optional ``gaussian`` sub-block.

    Returns
    -------
    dict[str, float]
        User-provided Gaussian adapter keyword arguments. Missing keys are intentionally omitted so
        ``ProbabilisticOutputAdapter`` constructor defaults remain the single source of default values.
    """

    cfg = dict(output_adapters_cfg or {})
    gaussian_cfg = cfg.get("gaussian") or {}
    return {str(k): float(v) for k, v in gaussian_cfg.items()}


# ----------------------------------------------------------------------------------------------------------------------
def resolve_output_adapter_hiddens(  # NOSONAR - Ignore cognitive complexity
    *,
    output_specs: Sequence[Any],
    d_model: int,
    hidden_dim_cfg: Mapping[str, Any] | None,
) -> dict[str, int]:
    """
    Resolve per-output adapter hidden dims from config.

    Validation is done in the config validator. Manual overrides always win.

    Parameters
    ----------
    output_specs : Sequence[Any]
        List of output specifications used for resolution of adapter hidden dims.
    d_model : int
        The number of expected features in the input.
    hidden_dim_cfg : Mapping[str, Any]
        Mapping of adapter hidden dims.
        Optional. Default: None.

    Returns
    -------
    dict[str, int]
        Dictionary with resolved per-output adapter hidden dims.

    """

    # ..................................................................................................................
    def _to_hidden_dim(v: str | int):
        """Resolve a hidden dim value: return `d_model` if the value is the string "d_model", else cast to int."""
        return int(d_model) if (v == "d_model") else int(v)

    # ..................................................................................................................

    cfg = dict(hidden_dim_cfg or {})

    default_hidden_dim = int(cfg.get("default", 0) or 0)
    bucketed = cfg.get("bucketed") or {}
    bucket_enable = bool(bucketed.get("enable", False))
    rules = bucketed.get("rules") or []
    manual = {str(k): v for k, v in (cfg.get("manual") or {}).items()}

    out: dict[str, int] = {}
    for spec in output_specs:
        name = str(getattr(spec, "name"))
        out_dim = int(getattr(spec, "embedding_dim"))
        hidden_dim = default_hidden_dim

        if bucket_enable:
            for r in rules:
                max_out = r.get("max_out_dim")
                if (max_out is None) or (out_dim <= int(max_out)):
                    hidden_dim = _to_hidden_dim(v=r.get("hidden", default_hidden_dim))
                    break

        if name in manual:
            hidden_dim = _to_hidden_dim(v=manual[name])

        out[name] = int(hidden_dim)

    return out
