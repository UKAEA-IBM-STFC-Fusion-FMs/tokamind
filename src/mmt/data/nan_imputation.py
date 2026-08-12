"""
Shared non-finite value imputation helpers for data transforms.

The helpers in this module are used before DCT3D encoding and before DCT3D
coefficient tuning so both paths apply the same NaN/inf policy.
"""

from __future__ import annotations

from typing import Literal
import numpy as np

NanImputationStrategy = Literal["zero", "interpolate"] | None
_VALID_NAN_IMPUTATION_STRATEGIES = {"zero", "interpolate", None}


def validate_nan_imputation_strategy(strategy: NanImputationStrategy, *, owner: str) -> None:
    """
    Validate a NaN/non-finite imputation strategy value.

    Parameters
    ----------
    strategy : NanImputationStrategy
        Strategy to validate. Allowed values are ``"zero"``, ``"interpolate"``, and ``None``.
    owner : str
        Name of the caller used in the error message.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``strategy`` is not one of ``"zero"``, ``"interpolate"``, or ``None``.

    """

    if strategy not in _VALID_NAN_IMPUTATION_STRATEGIES:
        raise ValueError(f"[{owner}] Invalid nan_imputation={strategy!r}. Must be one of: 'zero', 'interpolate', None.")


def interpolate_non_finite(arr: np.ndarray) -> np.ndarray:
    """
    Fill non-finite values via temporal then spatial interpolation, with zero fallback.

    Uses the same native-to-DCT view convention as ``DCT3DCodec``:
    ``(T,) -> (1, 1, T)``, ``(C, T) -> (C, 1, T)``, ``(H, W, T) -> (H, W, T)``.
    The passed array is modified in-place and returned with the same shape as the input.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape ``(T,)``, ``(C, T)``, or ``(H, W, T)``. The array must be writable.

    Returns
    -------
    np.ndarray
        Array with all non-finite values filled. The shape matches the input shape.

    Raises
    ------
    ValueError
        If ``arr`` is not 1D, 2D, or 3D.

    """

    original_shape = arr.shape
    if arr.ndim == 1:
        arr_view = arr.reshape(1, 1, arr.shape[0])
    elif arr.ndim == 2:
        arr_view = arr.reshape(arr.shape[0], 1, arr.shape[1])
    elif arr.ndim == 3:
        arr_view = arr
    else:
        raise ValueError(f"[nan_imputation] Interpolation supports 1D/2D/3D arrays, got shape={arr.shape}.")

    H, W, T = arr_view.shape

    # Temporal interpolation per (h, w).
    t_idx = np.arange(T)
    for h in range(H):
        for w in range(W):
            row = arr_view[h, w]
            non_finite_mask = ~np.isfinite(row)
            if non_finite_mask.any() and not non_finite_mask.all():
                valid = np.where(~non_finite_mask)[0]
                arr_view[h, w] = np.interp(t_idx, valid, row[valid])

    # Spatial interpolation along H per (w, t).
    if H > 1:
        h_idx = np.arange(H)
        for w in range(W):
            for t in range(T):
                col = arr_view[:, w, t]
                non_finite_mask = ~np.isfinite(col)
                if non_finite_mask.any() and not non_finite_mask.all():
                    valid = np.where(~non_finite_mask)[0]
                    arr_view[:, w, t] = np.interp(h_idx, valid, col[valid])

    # Last resort for any remaining NaN/inf values.
    np.nan_to_num(arr_view, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    return arr_view.reshape(original_shape)


def impute_non_finite(
    arr: np.ndarray,
    strategy: NanImputationStrategy,
    *,
    owner: str = "nan_imputation",
) -> np.ndarray:
    """
    Return an array with non-finite values handled according to the requested strategy.

    Parameters
    ----------
    arr : np.ndarray
        Input array. If it is already fully finite, the original array is returned unchanged.
    strategy : NanImputationStrategy
        Non-finite imputation strategy. ``"zero"`` zero-fills NaN/inf values, ``"interpolate"`` applies temporal
        then spatial interpolation with zero fallback, and ``None`` leaves the array unchanged.
    owner : str
        Name of the caller used in validation error messages.
        Optional. Default: ``"nan_imputation"``.

    Returns
    -------
    np.ndarray
        Imputed array. For ``"interpolate"``, a copied array is returned when imputation is required. For ``"zero"``,
        ``np.nan_to_num`` returns the imputed result. For ``None``, the original array is returned.

    Raises
    ------
    ValueError
        If ``strategy`` is invalid, or if interpolation receives an unsupported array dimensionality.

    """

    validate_nan_imputation_strategy(strategy, owner=owner)

    if np.isfinite(arr).all():
        return arr

    if strategy == "zero":
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if strategy == "interpolate":
        return interpolate_non_finite(arr.copy())

    return arr
