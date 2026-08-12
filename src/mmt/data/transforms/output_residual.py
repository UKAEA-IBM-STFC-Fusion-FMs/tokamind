"""Build output-space persistence baselines for residual prediction.

The transform derives same-name input/output signal pairs from a
``SignalSpecRegistry``. For each pair it selects the latest native input sample
and encodes it with the *output* codec. Using the output codec is essential:
input and output signals may use different ranked DCT coefficient selections,
even when their physical field names and native shapes match.

The resulting coefficient vectors are stored under
``window["output_baseline_emb"]`` and remain separate from the model targets.
Downstream collation and model code may use them as a persistence skip while
the existing output embeddings continue to represent absolute targets.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mmt.data.signal_spec import SignalSpecRegistry


class OutputResidualBaselineTransform:
    """Encode latest same-name inputs in their corresponding output spaces."""

    def __init__(self, *, signal_specs: SignalSpecRegistry, codecs: Mapping[int, Any]) -> None:
        """
        Initialize the residual-baseline transform.

        Parameters
        ----------
        signal_specs : SignalSpecRegistry
            Registry containing role-specific input and output signal specs.
        codecs : Mapping[int, Any]
            Codec mapping keyed by role-specific signal ID.

        Raises
        ------
        ValueError
            If no same-name input/output signals exist, or a mapped output does
            not use a supported DCT-style encoder.
        KeyError
            If a mapped output signal has no codec.
        """

        self.signal_specs = signal_specs
        self.codecs = codecs

        inputs_by_name = {str(spec.name): spec for spec in signal_specs.specs_for_role("input")}
        self._pairs: dict[int, tuple[str, int, Any, tuple[int, ...]]] = {}
        for output_spec in signal_specs.specs_for_role("output"):
            name = str(output_spec.name)
            input_spec = inputs_by_name.get(name)
            if input_spec is None:
                continue
            if str(output_spec.encoder_name) != "dct3d":
                raise ValueError(
                    "Output residual prediction currently supports DCT-style outputs only; "
                    f"got output:{name} encoder={output_spec.encoder_name!r}."
                )

            output_sid = int(output_spec.signal_id)
            codec = codecs.get(output_sid)
            if codec is None:
                raise KeyError(f"Missing output codec for residual signal output:{name} (id={output_sid}).")

            self._pairs[output_sid] = (
                name,
                int(input_spec.signal_id),
                codec,
                tuple(int(v) for v in output_spec.native_shape),
            )

        if not self._pairs:
            raise ValueError("Output residual prediction requires at least one same-name input/output signal pair.")

    def __call__(self, window: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Add output-space persistence baselines to one raw window.

        Parameters
        ----------
        window : dict[str, Any] | None
            Tokamind window containing native input signal payloads.

        Returns
        -------
        dict[str, Any] | None
            The input window with ``output_baseline_emb`` and
            ``output_baseline_source_id`` mappings, or ``None`` unchanged.

        Raises
        ------
        KeyError
            If a mapped input signal or its native values are missing.
        ValueError
            If the latest input slice does not match the output native shape or
            contains non-finite values.
        """

        if window is None:
            return None

        inputs = window.get("input") or {}
        baseline_emb: dict[int, np.ndarray] = {}
        baseline_source_id: dict[int, int] = {}

        for output_sid, (name, input_sid, codec, output_shape) in self._pairs.items():
            payload = inputs.get(name)
            if not isinstance(payload, Mapping) or payload.get("values") is None:
                raise KeyError(f"Residual source input:{name} is missing native values.")

            values = np.asarray(payload["values"])
            if values.ndim < 1 or values.shape[-1] < 1:
                raise ValueError(f"Residual source input:{name} has invalid shape {values.shape}.")

            latest = np.asarray(values[..., -1:], dtype=np.float32)
            if tuple(latest.shape) != output_shape:
                raise ValueError(
                    f"Residual source input:{name} latest shape {tuple(latest.shape)} does not match "
                    f"output native shape {output_shape}."
                )
            if not bool(np.isfinite(latest).all()):
                raise ValueError(f"Residual source input:{name} contains non-finite values in its latest sample.")

            baseline_emb[output_sid] = np.asarray(codec.encode(latest), dtype=np.float32)
            baseline_source_id[output_sid] = input_sid

        window["output_baseline_emb"] = baseline_emb
        window["output_baseline_source_id"] = baseline_source_id
        return window


__all__ = ["OutputResidualBaselineTransform"]
