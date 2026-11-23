"""
Batch collation for the MMT window-level dataloaders.

MMTCollate takes a list of per-window dictionaries (produced by the transforms pipeline) and builds a padded,
model-ready batch by:

- padding variable-length token sequences and packing token embeddings by signal_id,
- applying per-token and per-chunk dropout for inputs/actuators (and optional per-output dropout),
- producing masks for padding and dropped tokens,
- assembling output embeddings (and optionally native output tensors for eval).

This collate uses explicit PAD semantics (PAD ID/role/mod/pos) so padding/dropped slots are never confused with real
signals.

The returned batch dict is the standard input format expected by `MultiModalTransformer.forward()`.

Identity-encoded outputs
------------------------
Output signals configured with ``encoder_name: identity`` intentionally have no ``output_emb`` entry in the window
(``EmbedChunksTransform`` skips the embedding step to avoid duplicating large arrays in memory). The collate handles
this transparently:

- ``output_native`` is built from ``window["output"]`` when ``keep_output_native=True``, for mapped output
  signals present in the batch (missing per-window native values are represented as NaN, not supervised zeros).
- ``output_mask`` is built from ``output_emb`` presence for embedded outputs, and from native value presence for
  identity outputs — so ``NativeSparseMSELoss`` sees both correctly.
- ``output_emb`` only contains embedded outputs; loss config validation prevents using ``EmbedMSELoss`` to supervise
  identity outputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal
import logging
import numpy as np
import random

import torch

from mmt.constants import (
    ROLE_CONTEXT,
    ROLE_ACTUATOR,
    PAD_ID,
    PAD_ROLE,
    PAD_MOD,
    PAD_POS,
)


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.Collate")


# ----------------------------------------------------------------------------------------------------------------------
def _coerce_overrides_to_int_keys(d: Mapping[int, Any] | None, *, name: str) -> dict[int, float]:
    """
    Coerce a mapping {signal_id -> p} into a dict[int, float].

    The public YAML config is allowed to be name-keyed, but it must be converted to ID-keyed *once at startup* (see
    pipeline_ops.make_collate_fn).

    Keeping collate ID-keyed avoids storing per-token signal names in every window.

    Parameters
    ----------
    d : Mapping[int, Any] | None
        Input mapping (dict) to be coerced.
    name : str
        Name for the passed `d` mapping (dict).

    Returns
    -------
    dict[int, float]
        Coerced dictionary.

    Raises
    ------
    TypeError
        If `d` is not a mapping (dict).
        If `d` is not a valid mapping with keys of type int.

    """

    if d is None:
        return {}
    if not isinstance(d, dict):
        raise TypeError(f"`d` (mapping {name!r}) must be a dict of type {{signal_id: p}}, got {type(d).__name__}.")

    out: dict[int, float] = {}
    for k, v in d.items():
        if not isinstance(k, (int, np.integer)):
            raise TypeError(
                f"`d` must be a mapping (dict) keyed by int signal IDs, got key={k!r} type={type(k).__name__}. "
                "Convert name-based overrides to IDs once at startup."
            )

        out[int(k)] = float(v)

    return out


# ----------------------------------------------------------------------------------------------------------------------
def _get_native_val(window: dict[str, Any], name: str) -> np.ndarray | None:
    """
    Extract the native value array for output signal ``name`` from a window dict.

    Handles both the nested ``{"values": arr}`` form produced by the transform pipeline and a bare array.

    Parameters
    ----------
    window : dict[str, Any]
        Window dict.
    name : str
        Output signal name.

    Returns
    -------
    np.ndarray | None
        Value array, or None if absent.

    """

    info = (window.get("output") or {}).get(name)
    if isinstance(info, dict):
        return info.get("values")
    return info  # bare array or None


def _get_native_time(window: dict[str, Any], name: str) -> np.ndarray | None:
    """Extract the native output timestamps for one signal when available."""

    info = (window.get("output") or {}).get(name)
    if not isinstance(info, dict):
        return None
    time = info.get("time")
    if time is None:
        return None
    return np.asarray(time, dtype=np.float64).reshape(-1)


def _get_output_time(window: dict[str, Any], name: str) -> np.ndarray | None:
    """Extract compact finalized output timestamps, falling back to native output metadata."""

    compact = window.get("output_time")
    if isinstance(compact, dict):
        time = compact.get(name)
        if time is not None:
            return np.asarray(time, dtype=np.float64).reshape(-1)
    return _get_native_time(window, name)


# ======================================================================================================================
class MMTCollate:
    """
    Collate function for window-level MMT batches (pretraining + finetuning).

    Expected input
    --------------
    Each element in the batch is a single *window dict* produced by the preprocessing/transforms chain.

    The **minimal** required keys are:

    .code-block:: python

        {
            "emb_chunks": [np.ndarray(D_i), ...],  # ragged token embeddings
            "pos": np.ndarray(L,),                 # token positions
            "id": np.ndarray(L,),                  # signal IDs
            "mod": np.ndarray(L,),                 # modality IDs
            "role": np.ndarray(L,),                # role IDs
            "output_emb": {signal_id: np.ndarray(D_out), ...},

            # Optional (only if keep_output_native=True):
            # "output": {output_name: {"values": np.ndarray(...)}, ...}
        }

    Configuration
    -------------
    Public config still specifies override keys by *signal name* (human-friendly). The run pipeline converts those
    override dicts to be keyed by numeric `signal_id` once at startup (see `pipeline_ops.make_collate_fn`).

    This class expects the post-conversion form:

    .code-block:: yaml

        collate:
          # INPUT DROPOUT
          p_drop_inputs: 0.08
          p_drop_inputs_overrides: {}          # keyed by signal_id (after conversion)

          # OUTPUT DROPOUT
          p_drop_outputs: 0.0
          p_drop_outputs_overrides: {}         # keyed by output signal_id (after conversion)

          # ACTUATORS DROPOUT
          p_drop_actuators: 0.0
          p_drop_actuators_overrides: {}       # keyed by signal_id (after conversion)

          # CHUNK DROPOUT (coarse time-based masking)
          p_drop_inputs_chunks: 0.08
          p_drop_actuators_chunks: 0.0

          # EVAL-ONLY: include native output (Y_native)
          # keep_output_native: false

    Notes
    -----
    - Per-signal dropout overrides used by collate are keyed by **signal_id**. Name-keyed overrides should be converted
      once at startup (see `pipeline_ops.make_collate_fn`).

    - This collate keeps the dtype of token embeddings and output embeddings (e.g., float16 cached windows remain
      float16 through collation).

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, cfg_collate: Mapping[str, Any]) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        cfg_collate : Mapping[str, Any]
            Input mapping (dict) to be used to configure the MMTCollate instance.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors.

        Raises
        ------
        ValueError
            If `cfg_collate["output_id_to_name"]` is None when `cfg_collate['keep_output_native']=True`.
        TypeError
            If `cfg_collate["output_id_to_name"]` is not a mapping(dict) when `cfg_collate['keep_output_native']=True`.

        """

        self.cfg = dict(cfg_collate)
        self.keep_output_native = bool(self.cfg.get("keep_output_native", False))

        # Override dicts are expected to be keyed by signal_id (int).
        self.drop_inputs_overrides = _coerce_overrides_to_int_keys(
            d=self.cfg.get("p_drop_inputs_overrides", {}),
            name="p_drop_inputs_overrides",
        )
        self.drop_act_overrides = _coerce_overrides_to_int_keys(
            d=self.cfg.get("p_drop_actuators_overrides", {}),
            name="p_drop_actuators_overrides",
        )
        self.drop_outputs_overrides = _coerce_overrides_to_int_keys(
            d=self.cfg.get("p_drop_outputs_overrides", {}),
            name="p_drop_outputs_overrides",
        )

        # Optional output ID -> name mapping. Native targets require it, while coordinate-aware models use it to
        # collate compact output timestamp metadata even for embedding-only losses.
        self.output_id_to_name: dict[int, str] | None = None
        m = self.cfg.get("output_id_to_name")
        if m is None:
            if self.keep_output_native:
                raise ValueError(
                    "MMTCollate `cfg_collate['keep_output_native']=True` requires cfg_collate['output_id_to_name'] (a "
                    "dict {output_signal_id: output_name})."
                )
        elif not isinstance(m, dict):
            raise TypeError(f"`cfg_collate['output_id_to_name']` must be a dict, got {type(m).__name__}.")
        else:
            self.output_id_to_name = {int(k): str(v) for k, v in m.items()}

        # Reverse mapping name → id: used to discover native-only (identity) output sids from
        # window["output"] keys, which are signal names rather than IDs.
        self._output_name_to_id: dict[str, int] = (
            {name: sid for sid, name in self.output_id_to_name.items()} if self.output_id_to_name else {}
        )

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(  # NOSONAR - Ignore cognitive complexity
        self, batch: list[Any]
    ) -> dict[str, Any]:
        """
        Call method for the class instances to behave like a function.

        Parameters
        ----------
        batch : list[Any]
            Input batch on which the MMTCollate will be applied.

        Returns
        -------
        dict[str, Any]
            Assembled final batch.

        Raises
        ------
        TypeError
            If a batch element is not a single window dict.
            If a batch item does not have a key "output_emb" with mapping (dict) value.
        ValueError
            If an empty batch of windows is passed.
            If an inconsistent native output shape is identified.
            If resulting embeddings cannot be stacked.
        RuntimeError
            If `self.output_id_to_name` is None but `self.keep_output_native` is True.

        """

        # ..............................................................................................................
        def _drop_token(i_: int, t_: int, *, kind: Literal["input", "actuator"]) -> None:
            """
            Helper method: drop a token (set PAD metadata + update role mask).

            Parameters
            ----------
            i_ : int
                Sample index.
            t_ : int
                Token position index.
            kind : Literal["input", "actuator"]
                Drop kind. Valid options: ["input", "actuator"].

            Raises
            ------
            ValueError
                If unknown drop `kind` is passed.

            """
            if kind == "input":
                input_mask[i_, t_] = 0
            elif kind == "actuator":
                actuator_mask[i_, t_] = 0
            else:
                raise ValueError(f"Unknown drop kind: {kind!r}. Valid options: ['input', 'actuator'].")

            id_batch[i_, t_] = PAD_ID
            mod_batch[i_, t_] = PAD_MOD
            role_batch[i_, t_] = PAD_ROLE
            pos_batch[i_, t_] = PAD_POS
            token_time_batch[i_, t_] = np.nan

        # ..............................................................................................................
        # 0) Safety-check + filter None
        # ..............................................................................................................

        flat_windows: list[dict[str, Any]] = []
        for item in batch:
            if item is None:
                continue
            if not isinstance(item, dict):
                raise TypeError(
                    f"MMTCollate expects each batch element to be a single window dict, got {type(item)} instead."
                )
            flat_windows.append(item)

        B = len(flat_windows)
        if B == 0:
            raise ValueError("MMTCollate received an empty batch of windows.")

        # ..............................................................................................................
        # 1) Extract per-window arrays and discover output signal IDs
        # ..............................................................................................................

        emb_lists: list[list[np.ndarray]] = []
        pos_lists: list[np.ndarray] = []
        id_lists: list[np.ndarray] = []
        mod_lists: list[np.ndarray] = []
        role_lists: list[np.ndarray] = []
        token_time_lists: list[np.ndarray] = []
        space_grids: list[np.ndarray | None] = []

        # out_dicts[i] = window["output_emb"] — embedded output embeddings keyed by signal_id.
        out_dicts: list[dict[int, Any]] = []

        # Optional output-residual persistence baselines and their source input signal IDs.
        baseline_dicts: list[dict[int, Any]] = []
        baseline_source_dicts: list[dict[int, int]] = []
        all_baseline_sids: set[int] = set()

        # Signals with an output_emb entry (embed_mse targets).
        all_emb_sids: set[int] = set()
        # Signals present in window["output"] (native_sparse_mse targets; superset when identity outputs exist).
        all_native_sids: set[int] = set()

        for w in flat_windows:
            emb_lists.append(w["emb_chunks"])
            pos_lists.append(np.asarray(w["pos"], dtype=np.int32))
            id_lists.append(np.asarray(w["id"], dtype=np.int32))
            mod_lists.append(np.asarray(w["mod"], dtype=np.int16))
            role_lists.append(np.asarray(w["role"], dtype=np.int8))
            token_time = np.asarray(w.get("token_time", np.full(len(w["pos"]), np.nan)), dtype=np.float32).reshape(-1)
            if token_time.size != len(w["pos"]):
                raise ValueError(
                    "MMTCollate expects window['token_time'] to align with window['pos']; "
                    f"got {token_time.size} times for {len(w['pos'])} tokens."
                )
            token_time_lists.append(token_time)
            grid = w.get("space_grid")
            space_grids.append(None if grid is None else np.asarray(grid, dtype=np.float32))

            out_emb = w.get("output_emb")
            if not isinstance(out_emb, dict):
                raise TypeError(
                    "MMTCollate expects window['output_emb'] to be a dict of the form {signal_id: embedding}."
                )
            out_dicts.append(out_emb)
            all_emb_sids.update(int(k) for k in out_emb.keys())

            baseline = w.get("output_baseline_emb") or {}
            baseline_sources = w.get("output_baseline_source_id") or {}
            if not isinstance(baseline, dict) or not isinstance(baseline_sources, dict):
                raise TypeError("MMTCollate expects output residual baseline fields to be mappings when present.")
            baseline_keys = {int(k) for k in baseline}
            source_keys = {int(k) for k in baseline_sources}
            if baseline_keys != source_keys:
                raise ValueError(
                    "window['output_baseline_emb'] and window['output_baseline_source_id'] must have identical "
                    f"output signal IDs, got baselines={sorted(baseline_keys)} sources={sorted(source_keys)}."
                )
            baseline_dicts.append({int(k): v for k, v in baseline.items()})
            baseline_source_dicts.append({int(k): int(v) for k, v in baseline_sources.items()})
            all_baseline_sids.update(baseline_keys)

            if self.keep_output_native and self._output_name_to_id:
                for sname in w.get("output") or {}:
                    sid = self._output_name_to_id.get(sname)
                    if sid is not None:
                        all_native_sids.add(int(sid))

        # ..............................................................................................................
        # 2) Allocate padded token arrays
        # ..............................................................................................................

        lengths = [len(e) for e in emb_lists]
        L_max = max(lengths)  # NOSONAR # noqa

        pos_batch = np.full(shape=(B, L_max), fill_value=PAD_POS, dtype=np.int32)
        id_batch = np.full(shape=(B, L_max), fill_value=PAD_ID, dtype=np.int32)
        mod_batch = np.full(shape=(B, L_max), fill_value=PAD_MOD, dtype=np.int16)
        role_batch = np.full(shape=(B, L_max), fill_value=PAD_ROLE, dtype=np.int8)
        token_time_batch = np.full(shape=(B, L_max), fill_value=np.nan, dtype=np.float32)

        padding_mask = np.zeros(shape=(B, L_max), dtype=np.int8)
        input_mask = np.ones(shape=(B, L_max), dtype=np.int8)
        actuator_mask = np.ones(shape=(B, L_max), dtype=np.int8)

        # ..............................................................................................................
        # 3) Fill padded arrays
        # ..............................................................................................................

        for i in range(B):
            Li = lengths[i]  # NOSONAR # noqa
            if Li == 0:
                continue
            pos_batch[i, :Li] = pos_lists[i]
            id_batch[i, :Li] = id_lists[i]
            mod_batch[i, :Li] = mod_lists[i]
            role_batch[i, :Li] = role_lists[i]
            token_time_batch[i, :Li] = token_time_lists[i]
            padding_mask[i, :Li] = 1

        # ..............................................................................................................
        # 4) Input dropout (per-token)
        # ..............................................................................................................

        p_drop_in = float(self.cfg.get("p_drop_inputs", 0.0))
        if (p_drop_in > 0.0) or self.drop_inputs_overrides:
            for i in range(B):
                Li = lengths[i]
                if Li == 0:
                    continue
                idxs = np.nonzero(role_batch[i, :Li] == ROLE_CONTEXT)[0]  # noqa
                for t in idxs:
                    sid = int(id_batch[i, int(t)])
                    if sid == PAD_ID:
                        continue
                    p = float(self.drop_inputs_overrides.get(sid, p_drop_in))
                    if random.random() < p:
                        _drop_token(i_=i, t_=int(t), kind="input")

        # ..............................................................................................................
        # 5) Actuator dropout (per-token)
        # ..............................................................................................................

        p_drop_act = float(self.cfg.get("p_drop_actuators", 0.0))
        if (p_drop_act > 0.0) or self.drop_act_overrides:
            for i in range(B):
                Li = lengths[i]
                if Li == 0:
                    continue
                idxs = np.nonzero(role_batch[i, :Li] == ROLE_ACTUATOR)[0]  # noqa
                for t in idxs:
                    sid = int(id_batch[i, int(t)])
                    if sid == PAD_ID:
                        continue
                    p = float(self.drop_act_overrides.get(sid, p_drop_act))
                    if random.random() < p:
                        _drop_token(i_=i, t_=int(t), kind="actuator")

        # ..............................................................................................................
        # 6) Chunk dropout (coarse time masking, per-pos group)
        # ..............................................................................................................

        p_drop_inputs_chunks = float(self.cfg.get("p_drop_inputs_chunks", 0.0))
        p_drop_actuators_chunks = float(self.cfg.get("p_drop_actuators_chunks", 0.0))

        if (p_drop_inputs_chunks > 0.0) or (p_drop_actuators_chunks > 0.0):
            for i in range(B):
                Li = lengths[i]
                if Li == 0:
                    continue

                pos_i = pos_batch[i, :Li]
                order = np.argsort(pos_i)
                pos_sorted = pos_i[order]
                split = np.nonzero(np.diff(pos_sorted) != 0)[0] + 1  # noqa
                groups = np.split(order, split)

                for idxs in groups:
                    if idxs.size == 0:
                        continue
                    roles = role_batch[i, idxs]

                    if np.any(roles == ROLE_CONTEXT) and (random.random() < p_drop_inputs_chunks):
                        for t in idxs:
                            if role_batch[i, int(t)] == ROLE_CONTEXT:
                                _drop_token(i_=i, t_=int(t), kind="input")

                    if np.any(roles == ROLE_ACTUATOR) and (random.random() < p_drop_actuators_chunks):
                        for t in idxs:
                            if role_batch[i, int(t)] == ROLE_ACTUATOR:
                                _drop_token(i_=i, t_=int(t), kind="actuator")

        # ..............................................................................................................
        # Guard: ensure at least one valid token remains per sample
        # ..............................................................................................................

        # With stochastic per-token/per-chunk dropout (and some inherently missing signals), it is possible for a
        # sample to end up with *zero* valid tokens (all PAD_ID). That can trigger NaNs downstream (e.g., empty
        # attention sequences). We fix this deterministically by restoring a single original token (preferably a
        # context token).

        restored = 0
        for i in range(B):
            Li = lengths[i]
            if Li == 0:
                continue
            if np.any(id_batch[i, :Li] != PAD_ID):
                continue

            orig_roles = role_lists[i]
            candidates = np.nonzero(orig_roles[:Li] == ROLE_CONTEXT)[0]  # noqa
            if candidates.size == 0:
                candidates = np.arange(Li)

            t_restore = int(candidates[0])
            id_batch[i, t_restore] = int(id_lists[i][t_restore])
            mod_batch[i, t_restore] = int(mod_lists[i][t_restore])
            role_batch[i, t_restore] = int(role_lists[i][t_restore])
            pos_batch[i, t_restore] = int(pos_lists[i][t_restore])
            token_time_batch[i, t_restore] = float(token_time_lists[i][t_restore])
            input_mask[i, t_restore] = 1
            actuator_mask[i, t_restore] = 1
            restored += 1

        if restored > 0:
            logger.debug(
                "[CollateGuard] Restored 1 token for %d/%d samples after dropout.",
                restored,
                B,
            )

        # ..............................................................................................................
        # 7) Unified output loop
        #
        # One pass over every supervised output signal (embedded or native-only).
        # For each signal:
        #   - has_emb  → build output_emb + output_mask from output_emb presence
        #   - has_native → build output_native in a single scan (no double pass for shape)
        #   - native-only (identity outputs) → build output_mask from native value presence
        # ..............................................................................................................

        p_drop_outputs = float(self.cfg.get("p_drop_outputs", 0.0))

        output_emb_batch: dict[int, list[np.ndarray]] = {}
        output_mask_batch_np: dict[int, np.ndarray] = {}
        output_native_batch_np: dict[int, np.ndarray] = {}
        output_time_batch_np: dict[int, np.ndarray] = {}
        output_baseline_batch_np: dict[int, np.ndarray] = {}
        output_baseline_mask_np: dict[int, np.ndarray] = {}

        if self.keep_output_native and self.output_id_to_name is None:
            raise RuntimeError(
                "[MMTCollate] `self.output_id_to_name` is None but `self.keep_output_native` is True. "
                "This should have been caught in __init__."
            )

        all_output_sids = all_emb_sids | all_native_sids

        for sig_id in sorted(all_output_sids):
            has_emb = sig_id in all_emb_sids
            has_native = self.keep_output_native and (sig_id in all_native_sids)

            out_name = self.output_id_to_name.get(sig_id) if self.output_id_to_name is not None else None
            if out_name is not None:
                times: list[np.ndarray | None] = [_get_output_time(window, out_name) for window in flat_windows]
                reference_time = next((time for time in times if time is not None), None)
                if reference_time is not None:
                    n_steps = int(reference_time.size)
                    normalized_times: list[np.ndarray] = []
                    for time in times:
                        if time is None:
                            normalized_times.append(np.full(n_steps, np.nan, dtype=np.float64))
                            continue
                        if int(time.size) != n_steps:
                            raise ValueError(
                                f"Inconsistent output timestamp length for output={out_name!r}: "
                                f"expected {n_steps}, got {time.size}."
                            )
                        normalized_times.append(np.asarray(time, dtype=np.float64))
                    output_time_batch_np[sig_id] = np.stack(normalized_times, axis=0)

            # ----------------------------------------------------------------------------------------------------------
            # Embedded output: collect embeddings + mask, apply per-output dropout.
            # ----------------------------------------------------------------------------------------------------------
            if has_emb:
                # Infer ref shape/dtype from the first window that has this signal.
                ref_arr: np.ndarray | None = None
                for d in out_dicts:
                    emb = d.get(sig_id)
                    if emb is not None:
                        ref_arr = np.asarray(emb).reshape(-1)
                        break

                if ref_arr is not None:
                    ref_dtype = ref_arr.dtype if ref_arr.dtype in (np.float16, np.float32) else np.float32
                    ref_shape_emb = tuple(ref_arr.shape)

                    emb_list: list[np.ndarray] = []
                    emb_mask = np.ones(B, dtype=np.int8)

                    for i, d in enumerate(out_dicts):
                        emb = d.get(sig_id)
                        if emb is None:
                            emb_mask[i] = 0
                            emb_list.append(np.zeros(ref_shape_emb, dtype=ref_dtype))
                        else:
                            emb_list.append(np.asarray(emb, dtype=ref_dtype).reshape(-1))

                    # Per-output dropout.
                    p = float(self.drop_outputs_overrides.get(sig_id, p_drop_outputs))
                    if p > 0.0:
                        for i in range(B):
                            if emb_mask[i] and random.random() < p:
                                emb_mask[i] = 0
                                emb_list[i] = np.zeros(ref_shape_emb, dtype=ref_dtype)

                    output_emb_batch[sig_id] = emb_list
                    output_mask_batch_np[sig_id] = emb_mask

            # ----------------------------------------------------------------------------------------------------------
            # Native output: single scan — collect values and infer shape together.
            # Also builds output_mask for native-only (identity) outputs.
            # ----------------------------------------------------------------------------------------------------------
            if has_native:
                out_name = self.output_id_to_name.get(sig_id)  # type: ignore[union-attr]
                if out_name is None:
                    continue

                ref_shape_nat: tuple[int, ...] | None = None
                native_vals: list[np.ndarray | None] = []

                for w in flat_windows:
                    val = _get_native_val(w, out_name)
                    if val is None:
                        native_vals.append(None)
                        continue
                    arr = np.asarray(val, dtype=np.float32)
                    if ref_shape_nat is None:
                        ref_shape_nat = tuple(arr.shape)
                    elif tuple(arr.shape) != ref_shape_nat:
                        raise ValueError(
                            f"Inconsistent native output shape for output={out_name!r} (signal_id={sig_id}): "
                            f"expected {ref_shape_nat}, got {tuple(arr.shape)}."
                        )
                    native_vals.append(arr)

                # Skip if this signal is entirely absent from the batch.
                if ref_shape_nat is None:
                    continue

                output_native_batch_np[sig_id] = np.stack(
                    [v if v is not None else np.full(ref_shape_nat, np.nan, dtype=np.float32) for v in native_vals],
                    axis=0,
                )

                # Native-only (identity) outputs have no output_emb entry, so their output_mask
                # must be built here from value presence rather than embedding presence.
                # Per-output dropout is applied to the mask (no embedding to zero out, but
                # NativeSparseMSELoss gates on the mask so setting mask[i]=0 is sufficient).
                if not has_emb:
                    nat_mask = np.array([1 if v is not None else 0 for v in native_vals], dtype=np.int8)
                    p = float(self.drop_outputs_overrides.get(sig_id, p_drop_outputs))
                    if p > 0.0:
                        for i in range(B):
                            if nat_mask[i] and random.random() < p:
                                nat_mask[i] = 0
                    output_mask_batch_np[sig_id] = nat_mask

        # ..............................................................................................................
        # 8) Optional output-residual baselines
        #
        # A baseline is available only while the corresponding newest input token (pos=1) survives input/chunk
        # dropout. Unavailable baselines are represented by zeros plus an explicit mask, so the model cannot bypass a
        # forced missing-input ablation.
        # ..............................................................................................................

        for sig_id in sorted(all_baseline_sids):
            ref_arr: np.ndarray | None = None
            for d in baseline_dicts:
                value = d.get(sig_id)
                if value is not None:
                    ref_arr = np.asarray(value).reshape(-1)
                    break
            if ref_arr is None:
                continue

            ref_dtype = ref_arr.dtype if ref_arr.dtype in (np.float16, np.float32) else np.float32
            ref_shape = tuple(ref_arr.shape)
            values: list[np.ndarray] = []
            mask = np.zeros(B, dtype=np.int8)

            for i, (baseline, sources) in enumerate(zip(baseline_dicts, baseline_source_dicts)):
                value = baseline.get(sig_id)
                source_sid = sources.get(sig_id)
                source_kept = False
                if source_sid is not None:
                    Li = lengths[i]
                    source_kept = bool(
                        np.any(
                            (id_batch[i, :Li] == int(source_sid))
                            & (role_batch[i, :Li] == ROLE_CONTEXT)
                            & (pos_batch[i, :Li] == 1)
                        )
                    )

                if value is None or not source_kept:
                    values.append(np.zeros(ref_shape, dtype=ref_dtype))
                    continue

                arr = np.asarray(value, dtype=ref_dtype).reshape(-1)
                if tuple(arr.shape) != ref_shape:
                    raise ValueError(
                        f"Inconsistent output residual baseline shape for signal_id={sig_id}: "
                        f"expected {ref_shape}, got {tuple(arr.shape)}."
                    )
                values.append(arr)
                mask[i] = 1

            output_baseline_batch_np[sig_id] = np.stack(values, axis=0)
            output_baseline_mask_np[sig_id] = mask

        # ..............................................................................................................
        # 9) Convert arrays to torch
        # ..............................................................................................................

        pos_t = torch.from_numpy(pos_batch).long()
        id_t = torch.from_numpy(id_batch).long()
        mod_t = torch.from_numpy(mod_batch.astype(np.int64))
        role_t = torch.from_numpy(role_batch.astype(np.int64))
        token_time_t = torch.from_numpy(token_time_batch)

        padding_mask_t = torch.from_numpy(padding_mask.astype(bool))
        input_mask_t = torch.from_numpy(input_mask.astype(bool))
        actuator_mask_t = torch.from_numpy(actuator_mask.astype(bool))

        # ..............................................................................................................
        # 9) Pack token embeddings by signal_id
        # ..............................................................................................................

        emb_by_sid_np: dict[int, list[np.ndarray]] = {}
        emb_index_np: dict[int, list[int]] = {}

        for i in range(B):
            Li = lengths[i]
            if Li == 0:
                continue
            emb_list_i = emb_lists[i]
            for t in range(Li):
                sid_i = int(id_batch[i, t])
                if sid_i == PAD_ID:
                    continue
                arr = emb_list_i[t]
                if arr is None:
                    continue
                arr = np.asarray(arr)
                if arr.size == 0:
                    continue
                emb_by_sid_np.setdefault(sid_i, []).append(arr)
                emb_index_np.setdefault(sid_i, []).append(i * L_max + t)

        emb_by_sid_t: dict[int, torch.Tensor] = {}
        emb_index_t: dict[int, torch.Tensor] = {}

        for sid_i, arr_list in emb_by_sid_np.items():
            try:
                stacked = np.stack(arr_list, axis=0)
            except Exception as e:
                shapes = [tuple(a.shape) for a in arr_list]
                raise ValueError(f"Cannot stack embeddings for signal_id={sid_i}. shapes={shapes}.") from e

            emb_by_sid_t[sid_i] = torch.from_numpy(stacked)
            emb_index_t[sid_i] = torch.as_tensor(emb_index_np[sid_i], dtype=torch.long)

        # Output tensors — all keyed by signal_id.
        output_emb_t: dict[int, torch.Tensor] = {
            sig_id: torch.from_numpy(np.stack(emb_list, axis=0)) for sig_id, emb_list in output_emb_batch.items()
        }
        output_mask_t: dict[int, torch.Tensor] = {
            sig_id: torch.from_numpy(mask_np.astype(bool)) for sig_id, mask_np in output_mask_batch_np.items()
        }
        output_baseline_t: dict[int, torch.Tensor] = {
            sig_id: torch.from_numpy(arr) for sig_id, arr in output_baseline_batch_np.items()
        }
        output_baseline_mask_t: dict[int, torch.Tensor] = {
            sig_id: torch.from_numpy(mask_np.astype(bool)) for sig_id, mask_np in output_baseline_mask_np.items()
        }
        output_native_t: dict[int, torch.Tensor] = {}
        if self.keep_output_native:
            output_native_t = {sig_id: torch.from_numpy(arr) for sig_id, arr in output_native_batch_np.items()}
        output_time_t: dict[int, torch.Tensor] = {
            sig_id: torch.from_numpy(arr.astype(np.float32, copy=False)) for sig_id, arr in output_time_batch_np.items()
        }

        # ..............................................................................................................
        # 10) Assemble final batch dict
        # ..............................................................................................................

        batch_out: dict[str, Any] = {
            "emb": emb_by_sid_t,
            "emb_index": emb_index_t,
            "pos": pos_t,
            "id": id_t,
            "mod": mod_t,
            "role": role_t,
            "token_time": token_time_t,
            "padding_mask": padding_mask_t,
            "input_mask": input_mask_t,
            "actuator_mask": actuator_mask_t,
            "output_emb": output_emb_t,
            "output_mask": output_mask_t,
        }

        if any(grid is not None for grid in space_grids):
            if any(grid is None for grid in space_grids):
                raise ValueError("space_grid must be present for every window in a collated batch or for none.")
            reference_grid = space_grids[0]
            assert reference_grid is not None
            reference_shape = tuple(reference_grid.shape)
            if any(tuple(grid.shape) != reference_shape for grid in space_grids if grid is not None):
                raise ValueError("Inconsistent space_grid shapes within a collated batch.")
            batch_out["space_grid"] = torch.from_numpy(np.stack(space_grids, axis=0))

        if output_baseline_t:
            batch_out["output_baseline_emb"] = output_baseline_t
            batch_out["output_baseline_mask"] = output_baseline_mask_t

        if self.keep_output_native:
            batch_out["output_native"] = output_native_t
        if output_time_t:
            batch_out["output_time"] = output_time_t

        # Keep shot/window identifiers if present (useful for debug).
        first = flat_windows[0]
        if "shot_id" in first:
            batch_out["shot_id"] = [w.get("shot_id") for w in flat_windows]
        if "window_index" in first:
            batch_out["window_index"] = [w.get("window_index") for w in flat_windows]
        if "t_cut" in first:
            batch_out["t_cut"] = torch.as_tensor(
                [float(w.get("t_cut", np.nan)) for w in flat_windows], dtype=torch.float32
            )

        return batch_out
