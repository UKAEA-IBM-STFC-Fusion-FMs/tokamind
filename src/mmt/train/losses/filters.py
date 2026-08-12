"""
Loss output filter helpers.

This module resolves train-loss output filters from user-facing output signal names to runtime signal IDs, and checks
that each configured loss term can supervise the outputs assigned to it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mmt.train.losses.constants import (
    ALL_LOSS_TYPES,
    DEFAULT_LOSS_TERMS,
    EMBED_MSE_LOSS_TYPE,
    EMBED_SPACE_LOSS_TYPES,
    GRAD_SHAFRANOV_LOSS_TYPE,
    GRAD_SHAFRANOV_WEAK_FORM_LOSS_TYPE,
    NATIVE_SPACE_LOSS_TYPES,
)


# ----------------------------------------------------------------------------------------------------------------------
def _resolve_named_outputs(names: Any, *, field: str, name_to_sid: Mapping[str, int]) -> set[int]:
    """
    Resolve configured output names to signal IDs.

    Parameters
    ----------
    names : Any
        Config value for a loss-term output filter.
    field : str
        Human-readable field path used in error messages.
    name_to_sid : Mapping[str, int]
        Mapping from output signal name to signal ID.

    Returns
    -------
    set[int]
        Resolved signal IDs.

    Raises
    ------
    TypeError
        If ``names`` is not a non-string sequence.
    ValueError
        If ``names`` is empty.
    KeyError
        If any configured name is not a known output signal.

    """

    if isinstance(names, str) or not isinstance(names, Sequence):
        raise TypeError(f"{field} must be a non-empty list of output signal names.")
    if len(names) == 0:
        raise ValueError(f"{field} must be non-empty when provided.")

    unknown = [str(name) for name in names if str(name) not in name_to_sid]
    if unknown:
        raise KeyError(
            f"Unknown {field} output names: {unknown}. "
            f"Expected output signal names among: {sorted(name_to_sid.keys())}."
        )

    return {int(name_to_sid[str(name)]) for name in names}


# ----------------------------------------------------------------------------------------------------------------------
def _resolve_loss_term_filter(
    term_def: Mapping[str, Any],
    *,
    term_index: int,
    name_to_sid: Mapping[str, int],
    all_output_ids: set[int],
    path: str,
) -> tuple[str | None, set[int] | None]:
    """
    Resolve one loss term's optional output filter.

    Parameters
    ----------
    term_def : Mapping[str, Any]
        Loss term config block.
    term_index : int
        Loss term index used in error messages.
    name_to_sid : Mapping[str, int]
        Mapping from output signal name to signal ID.
    all_output_ids : set[int]
        All output signal IDs for the task.
    path : str
        Human-readable loss config path used in error messages.

    Returns
    -------
    tuple[str | None, set[int] | None]
        Filter mode (``"include"``, ``"exclude"``, or None) and selected signal IDs before applying loss-term
        capability.

    Raises
    ------
    TypeError
        If ``outputs`` is not a mapping, or if include/exclude values are malformed.
    ValueError
        If both include and exclude are configured, or if either list is empty.
    KeyError
        If any configured output name is unknown.

    """

    outputs = term_def.get("outputs")
    if outputs is None:
        return None, None
    if not isinstance(outputs, Mapping):
        raise TypeError(f"{path}.terms[{term_index}].outputs must be a mapping when provided.")

    unknown_keys = sorted(str(key) for key in outputs.keys() if key not in {"include", "exclude"})
    if unknown_keys:
        raise KeyError(
            f"Unknown {path}.terms[{term_index}].outputs keys: {unknown_keys}. "
            "Supported keys are 'include' and 'exclude'."
        )

    has_include = outputs.get("include") is not None
    has_exclude = outputs.get("exclude") is not None
    if has_include and has_exclude:
        raise ValueError(f"{path}.terms[{term_index}].outputs must set only one of 'include' or 'exclude'.")
    if not has_include and not has_exclude:
        return None, None

    if has_include:
        selected = _resolve_named_outputs(
            outputs["include"],
            field=f"{path}.terms[{term_index}].outputs.include",
            name_to_sid=name_to_sid,
        )
        return "include", selected

    excluded = _resolve_named_outputs(
        outputs["exclude"],
        field=f"{path}.terms[{term_index}].outputs.exclude",
        name_to_sid=name_to_sid,
    )
    return "exclude", set(all_output_ids) - excluded


# ----------------------------------------------------------------------------------------------------------------------
def _term_capable_ids(
    term_type: str,
    *,
    all_output_ids: set[int],
    identity_ids: set[int],
    native_capable_ids: set[int],
) -> set[int]:
    """
    Return output IDs that a loss term type can supervise.

    Parameters
    ----------
    term_type : str
        Loss term type.
    all_output_ids : set[int]
        All output signal IDs for the task.
    identity_ids : set[int]
        Output IDs whose encoder is ``identity``.
    native_capable_ids : set[int]
        Output IDs with native-space decoders.

    Returns
    -------
    set[int]
        Output IDs that this loss term type can supervise.

    Raises
    ------
    ValueError
        If the term type is unknown, or if a native-space loss is requested without decoders.

    """

    if term_type in EMBED_SPACE_LOSS_TYPES:
        # Embedding-space terms (embed_mse, embed_gaussian_crps) operate on dense coeff vectors and exclude
        # identity-encoded outputs, which are sparse/NaN and belong to native-space terms instead.
        return all_output_ids - identity_ids
    if term_type in NATIVE_SPACE_LOSS_TYPES:
        if not native_capable_ids:
            raise ValueError(
                f"Loss term '{term_type}' requires decoders to be provided, but got None or empty dict. "
                "Build and pass a dict[signal_id, TorchDecoder] when using this term."
            )
        return native_capable_ids
    raise ValueError(f"Unknown loss term type '{term_type}'. Supported: {sorted(ALL_LOSS_TYPES)}.")


# ----------------------------------------------------------------------------------------------------------------------
def resolve_loss_output_filters(
    loss_cfg: Mapping[str, Any],
    *,
    output_specs: Sequence[Any],
    decoders: Mapping[int, Any] | None,
    path: str = "train.loss",
    require_all_outputs: bool = True,
) -> list[set[int] | None]:
    """
    Resolve and validate per-loss-term output filters.

    Parameters
    ----------
    loss_cfg : Mapping[str, Any]
        The ``train.loss`` config dict.
    output_specs : Sequence[Any]
        Model output signal specs. Each spec must expose ``name``, ``signal_id``, and ``encoder_name``.
    decoders : Mapping[int, Any] | None
        Per-output decoders keyed by signal ID, used to determine native-space loss capability.
    path : str
        Human-readable loss config path used in error messages.
    require_all_outputs : bool
        Whether to require every task output to be supervised by at least one configured term.
        Optional. Default: True.

    Returns
    -------
    list[set[int] | None]
        Per-term signal ID filters. ``None`` means the term uses its default capability.

    Raises
    ------
    KeyError
        If a configured output name is unknown.
    TypeError
        If an output filter block is malformed.
    ValueError
        If output filters are invalid, if a term selects no supervisable outputs, or if any task output is not
        supervised by at least one capable loss term.

    """

    name_to_sid = {str(spec.name): int(spec.signal_id) for spec in output_specs}
    sid_to_name = {int(spec.signal_id): str(spec.name) for spec in output_specs}
    all_output_ids = set(sid_to_name.keys())
    identity_ids = {
        int(spec.signal_id) for spec in output_specs if str(getattr(spec, "encoder_name", "")).lower() == "identity"
    }
    native_capable_ids = {int(sid) for sid in (decoders or {}).keys()}
    terms_cfg: list[Mapping[str, Any]] = list(loss_cfg.get("terms", DEFAULT_LOSS_TERMS))

    term_filters: list[set[int] | None] = []
    supervised_ids: set[int] = set()

    for term_index, term_def in enumerate(terms_cfg):
        term_type = str(term_def.get("type", EMBED_MSE_LOSS_TYPE))
        mode, raw_filter = _resolve_loss_term_filter(
            term_def=term_def,
            term_index=term_index,
            name_to_sid=name_to_sid,
            all_output_ids=all_output_ids,
            path=path,
        )

        capable_ids = _term_capable_ids(
            term_type=term_type,
            all_output_ids=all_output_ids,
            identity_ids=identity_ids,
            native_capable_ids=native_capable_ids,
        )

        if (term_type in EMBED_SPACE_LOSS_TYPES) and (mode == "include"):
            invalid_ids = sorted(set(raw_filter or set()) & identity_ids)
            if invalid_ids:
                invalid_names = [sid_to_name[sid] for sid in invalid_ids]
                raise ValueError(
                    f"{path}.terms[{term_index}] uses {term_type} but explicitly includes identity outputs "
                    f"{invalid_names}. Identity outputs are not written to output_emb; use a native-space loss "
                    "(e.g. native_sparse_mse) for those signals."
                )

        selected_ids = capable_ids if raw_filter is None else (set(raw_filter) & capable_ids)
        if not selected_ids:
            raise ValueError(
                f"{path}.terms[{term_index}] ({term_type}) selects no outputs that this loss can supervise."
            )

        if term_type in {
            GRAD_SHAFRANOV_LOSS_TYPE,
            GRAD_SHAFRANOV_WEAK_FORM_LOSS_TYPE,
        }:
            psi_ids = {sid for sid, name in sid_to_name.items() if name == "equilibrium-psi"}
            if not psi_ids:
                raise ValueError("Grad-Shafranov loss requires model output 'equilibrium-psi'.")
            if selected_ids.isdisjoint(psi_ids):
                raise ValueError(
                    f"{path}.terms[{term_index}] ({term_type}) must include 'equilibrium-psi' in its outputs."
                )
            if term_type == GRAD_SHAFRANOV_WEAK_FORM_LOSS_TYPE:
                j_tor_ids = {sid for sid, name in sid_to_name.items() if name == "equilibrium-j_tor"}
                if not j_tor_ids:
                    raise ValueError(f"{term_type} requires model output 'equilibrium-j_tor'.")
                if selected_ids.isdisjoint(j_tor_ids):
                    raise ValueError(
                        f"{path}.terms[{term_index}] ({term_type}) must include 'equilibrium-j_tor' in its outputs."
                    )

        supervised_ids.update(selected_ids)
        term_filters.append(None if raw_filter is None else selected_ids)

    unsupervised_ids = sorted(all_output_ids - supervised_ids)
    if require_all_outputs and unsupervised_ids:
        unsupervised_names = [sid_to_name[sid] for sid in unsupervised_ids]
        raise ValueError(
            "Every output must be supervised by at least one capable loss term. "
            f"Unsupervised outputs: {unsupervised_names}."
        )

    return term_filters


# ----------------------------------------------------------------------------------------------------------------------
def resolve_native_loss_output_ids(
    loss_cfg: Mapping[str, Any],
    *,
    output_specs: Sequence[Any],
    decoders: Mapping[int, Any] | None,
) -> set[int]:
    """
    Resolve output IDs whose native targets are required by native-space loss terms.

    Parameters
    ----------
    loss_cfg : Mapping[str, Any]
        The ``train.loss`` config dict.
    output_specs : Sequence[Any]
        Model output signal specs. Each spec must expose ``name`` and ``signal_id``.
    decoders : Mapping[int, Any] | None
        Per-output decoders keyed by signal ID.

    Returns
    -------
    set[int]
        Output signal IDs whose native values must be retained in the data pipeline.

    """

    filters = resolve_loss_output_filters(loss_cfg=loss_cfg, output_specs=output_specs, decoders=decoders)
    terms_cfg: list[Mapping[str, Any]] = list(loss_cfg.get("terms", DEFAULT_LOSS_TERMS))
    native_capable_ids = {int(sid) for sid in (decoders or {}).keys()}

    native_ids: set[int] = set()
    for term_def, output_filter in zip(terms_cfg, filters, strict=True):
        if str(term_def.get("type", EMBED_MSE_LOSS_TYPE)) not in NATIVE_SPACE_LOSS_TYPES:
            continue
        native_ids.update(native_capable_ids if output_filter is None else {int(sid) for sid in output_filter})

    return native_ids


# ----------------------------------------------------------------------------------------------------------------------
def resolve_native_loss_output_names(
    loss_cfg: Mapping[str, Any],
    *,
    output_specs: Sequence[Any],
    decoders: Mapping[int, Any] | None,
) -> set[str]:
    """
    Resolve output names whose native targets are required by native-space loss terms.

    Parameters
    ----------
    loss_cfg : Mapping[str, Any]
        The ``train.loss`` config dict.
    output_specs : Sequence[Any]
        Model output signal specs. Each spec must expose ``name`` and ``signal_id``.
    decoders : Mapping[int, Any] | None
        Per-output decoders keyed by signal ID.

    Returns
    -------
    set[str]
        Output signal names whose native values must be retained in the data pipeline.

    """

    sid_to_name = {int(spec.signal_id): str(spec.name) for spec in output_specs}
    native_ids = resolve_native_loss_output_ids(loss_cfg=loss_cfg, output_specs=output_specs, decoders=decoders)
    return {sid_to_name[sid] for sid in native_ids if sid in sid_to_name}
