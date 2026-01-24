"""
Batch collation for the MMT window-level dataloaders.

MMTCollate takes a list of per-window dictionaries (produced by the transforms
pipeline) and builds a padded, model-ready batch by:

- padding variable-length token sequences and packing token embeddings by signal_id,
- applying per-token and per-chunk dropout for inputs/actuators (and optional per-output dropout),
- producing masks for padding and dropped tokens,
- assembling output embeddings (and optionally native output tensors for eval).

This collate uses explicit PAD semantics (PAD id/role/mod/pos) so padding/dropped
slots are never confused with real signals.

The returned batch dict is the standard input format expected by
``MultiModalTransformer.forward()``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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


logger = logging.getLogger("mmt.Collate")


def _coerce_overrides_to_int_keys(d: Any, *, name: str) -> Dict[int, float]:
    """Coerce a mapping {signal_id -> p} into a Dict[int, float].

    The public YAML config is allowed to be name-keyed, but it must be
    converted to id-keyed *once at startup* (see pipeline_helpers.make_collate_fn).

    Keeping collate id-keyed avoids storing per-token signal names in every window.
    """

    if d is None:
        return {}
    if not isinstance(d, dict):
        raise TypeError(
            f"{name} must be a dict of {{signal_id: p}}, got {type(d).__name__}."
        )

    out: Dict[int, float] = {}
    for k, v in d.items():
        if not isinstance(k, (int, np.integer)):
            raise TypeError(
                f"{name} must be keyed by int signal_id (got key={k!r} type={type(k).__name__}). "
                "Convert name-based overrides to ids once at startup."
            )
        out[int(k)] = float(v)
    return out


class MMTCollate:
    """Collate function for window-level MMT batches (pretraining + finetuning).

    Expected input
    --------------
    Each element in the batch is a single *window dict* produced by the
    preprocessing/transforms chain.

    The **minimal** required keys are:

    .. code-block:: python

        {
            "emb_chunks": [np.ndarray(D_i), ...],   # ragged token embeddings
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
    Public config still specifies override keys by *signal name* (human-friendly).
    The run pipeline converts those override dicts to be keyed by numeric
    `signal_id` once at startup (see `pipeline_helpers.make_collate_fn`).

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
    - Per-signal dropout overrides used by collate are keyed by **signal_id**.
      Name-keyed overrides should be converted once at startup (see
      ``pipeline_helpers.make_collate_fn``).

    - This collate keeps the dtype of token embeddings and output embeddings
      (e.g. float16 cached windows remain float16 through collation).
    """

    def __init__(self, cfg_collate: Dict[str, Any]) -> None:
        self.cfg = dict(cfg_collate)
        self.keep_output_native = bool(self.cfg.get("keep_output_native", False))

        # Override dicts are expected to be keyed by signal_id (int).
        self.drop_inputs_overrides = _coerce_overrides_to_int_keys(
            self.cfg.get("p_drop_inputs_overrides", {}),
            name="p_drop_inputs_overrides",
        )
        self.drop_act_overrides = _coerce_overrides_to_int_keys(
            self.cfg.get("p_drop_actuators_overrides", {}),
            name="p_drop_actuators_overrides",
        )
        self.drop_outputs_overrides = _coerce_overrides_to_int_keys(
            self.cfg.get("p_drop_outputs_overrides", {}),
            name="p_drop_outputs_overrides",
        )

        # Optional: output_id -> output_name mapping (used only for output_native).
        # This mapping should be built once at startup from the SignalSpecRegistry.
        self.output_id_to_name: Optional[Dict[int, str]] = None
        if self.keep_output_native:
            m = self.cfg.get("output_id_to_name")
            if m is None:
                raise ValueError(
                    "MMTCollate keep_output_native=True requires cfg_collate['output_id_to_name'] "
                    "(a dict {output_signal_id: output_name})."
                )
            if not isinstance(m, dict):
                raise TypeError(
                    f"output_id_to_name must be a dict, got {type(m).__name__}."
                )
            self.output_id_to_name = {int(k): str(v) for k, v in m.items()}

    # ------------------------------------------------------------------
    def __call__(self, batch: List[Any]) -> Dict[str, Any]:
        # ---------------------------------------------------------------
        # 0) Sanity-check + filter None
        # ---------------------------------------------------------------
        flat_windows: List[Dict[str, Any]] = []
        for item in batch:
            if item is None:
                continue
            if not isinstance(item, dict):
                raise TypeError(
                    "MMTCollate expects each batch element to be a single window dict, "
                    f"got {type(item)} instead."
                )
            flat_windows.append(item)

        B = len(flat_windows)
        if B == 0:
            raise ValueError("MMTCollate received an empty batch of windows.")

        # ---------------------------------------------------------------
        # 1) Extract per-window arrays
        # ---------------------------------------------------------------
        emb_lists: List[List[np.ndarray]] = []
        pos_lists: List[np.ndarray] = []
        id_lists: List[np.ndarray] = []
        mod_lists: List[np.ndarray] = []
        role_lists: List[np.ndarray] = []

        out_dicts: List[Dict[int, Any]] = []
        all_target_ids: set[int] = set()

        for w in flat_windows:
            emb_lists.append(w["emb_chunks"])

            # Force signed dtypes (prevents accidental uint wrap)
            pos_lists.append(np.asarray(w["pos"], dtype=np.int32))
            id_lists.append(np.asarray(w["id"], dtype=np.int32))
            mod_lists.append(np.asarray(w["mod"], dtype=np.int16))
            role_lists.append(np.asarray(w["role"], dtype=np.int8))

            out_emb = w.get("output_emb")
            if not isinstance(out_emb, dict):
                raise TypeError(
                    "MMTCollate expects window['output_emb'] to be a dict {signal_id: embedding}."
                )
            out_dicts.append(out_emb)
            all_target_ids.update(int(k) for k in out_emb.keys())

        # ---------------------------------------------------------------
        # 2) Allocate padded token arrays
        # ---------------------------------------------------------------
        lengths = [len(e) for e in emb_lists]
        L_max = max(lengths)

        pos_batch = np.full((B, L_max), PAD_POS, dtype=np.int32)
        id_batch = np.full((B, L_max), PAD_ID, dtype=np.int32)
        mod_batch = np.full((B, L_max), PAD_MOD, dtype=np.int16)
        role_batch = np.full((B, L_max), PAD_ROLE, dtype=np.int8)

        padding_mask = np.zeros((B, L_max), dtype=np.int8)
        input_mask = np.ones((B, L_max), dtype=np.int8)
        actuator_mask = np.ones((B, L_max), dtype=np.int8)

        # ---------------------------------------------------------------
        # 3) Fill padded arrays
        # ---------------------------------------------------------------
        for i in range(B):
            Li = lengths[i]
            if Li == 0:
                continue
            pos_batch[i, :Li] = pos_lists[i]
            id_batch[i, :Li] = id_lists[i]
            mod_batch[i, :Li] = mod_lists[i]
            role_batch[i, :Li] = role_lists[i]
            padding_mask[i, :Li] = 1

        # ---------------------------------------------------------------
        # Helper: drop a token (set PAD metadata + update role mask)
        # ---------------------------------------------------------------
        def _drop_token(i: int, t: int, *, kind: str) -> None:
            if kind == "input":
                input_mask[i, t] = 0
            elif kind == "actuator":
                actuator_mask[i, t] = 0
            else:
                raise ValueError(f"Unknown drop kind: {kind!r}")

            id_batch[i, t] = PAD_ID
            mod_batch[i, t] = PAD_MOD
            role_batch[i, t] = PAD_ROLE
            pos_batch[i, t] = PAD_POS

        # ---------------------------------------------------------------
        # 4) Input dropout (per-token)
        # ---------------------------------------------------------------
        p_drop_in = float(self.cfg.get("p_drop_inputs", 0.0))
        if p_drop_in > 0.0 or self.drop_inputs_overrides:
            for i in range(B):
                Li = lengths[i]
                if Li == 0:
                    continue
                idxs = np.where(role_batch[i, :Li] == ROLE_CONTEXT)[0]
                for t in idxs:
                    sid = int(id_batch[i, int(t)])
                    if sid == PAD_ID:
                        continue
                    p = float(self.drop_inputs_overrides.get(sid, p_drop_in))
                    if random.random() < p:
                        _drop_token(i, int(t), kind="input")

        # ---------------------------------------------------------------
        # 5) Actuator dropout (per-token)
        # ---------------------------------------------------------------
        p_drop_act = float(self.cfg.get("p_drop_actuators", 0.0))
        if p_drop_act > 0.0 or self.drop_act_overrides:
            for i in range(B):
                Li = lengths[i]
                if Li == 0:
                    continue
                idxs = np.where(role_batch[i, :Li] == ROLE_ACTUATOR)[0]
                for t in idxs:
                    sid = int(id_batch[i, int(t)])
                    if sid == PAD_ID:
                        continue
                    p = float(self.drop_act_overrides.get(sid, p_drop_act))
                    if random.random() < p:
                        _drop_token(i, int(t), kind="actuator")

        # ---------------------------------------------------------------
        # 6) Chunk dropout (coarse time masking, per-pos group)
        # ---------------------------------------------------------------
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
                split = np.where(np.diff(pos_sorted) != 0)[0] + 1
                groups = np.split(order, split)

                for idxs in groups:
                    if idxs.size == 0:
                        continue

                    roles = role_batch[i, idxs]

                    # Drop input tokens in this chunk-position group
                    if (roles == ROLE_CONTEXT).any() and (
                        random.random() < p_drop_inputs_chunks
                    ):
                        for t in idxs:
                            if role_batch[i, int(t)] == ROLE_CONTEXT:
                                _drop_token(i, int(t), kind="input")

                    # Drop actuator tokens in this chunk-position group
                    if (roles == ROLE_ACTUATOR).any() and (
                        random.random() < p_drop_actuators_chunks
                    ):
                        for t in idxs:
                            if role_batch[i, int(t)] == ROLE_ACTUATOR:
                                _drop_token(i, int(t), kind="actuator")

        # ---------------------------------------------------------------
        # Guard: ensure at least one valid token remains per sample
        # ---------------------------------------------------------------
        # With stochastic per-token/per-chunk dropout (and some inherently
        # missing signals), it is possible for a sample to end up with *zero*
        # valid tokens (all PAD_ID). That can trigger NaNs downstream (e.g.,
        # empty attention sequences). We fix this deterministically by
        # restoring a single original token (preferably a context token).
        restored = 0
        for i in range(B):
            Li = lengths[i]
            if Li == 0:
                continue
            if np.any(id_batch[i, :Li] != PAD_ID):
                continue

            # Prefer restoring the first context token if one existed.
            orig_roles = role_lists[i]
            candidates = np.where(orig_roles[:Li] == ROLE_CONTEXT)[0]
            if candidates.size == 0:
                candidates = np.arange(Li)

            t_restore = int(candidates[0])

            # Restore original metadata for this token.
            id_batch[i, t_restore] = int(id_lists[i][t_restore])
            mod_batch[i, t_restore] = int(mod_lists[i][t_restore])
            role_batch[i, t_restore] = int(role_lists[i][t_restore])
            pos_batch[i, t_restore] = int(pos_lists[i][t_restore])

            # Mark token as kept (setting both to 1 is safe).
            input_mask[i, t_restore] = 1
            actuator_mask[i, t_restore] = 1
            restored += 1

        if restored > 0:
            logger.debug(
                "[CollateGuard] Restored 1 token for %d/%d samples after dropout.",
                restored,
                B,
            )

        # ---------------------------------------------------------------
        # 7) Output embeddings + dropout
        # ---------------------------------------------------------------
        p_drop_outputs = float(self.cfg.get("p_drop_outputs", 0.0))

        output_emb_batch: Dict[int, List[np.ndarray]] = {}
        output_mask_batch_np: Dict[int, np.ndarray] = {}

        for sig_id in sorted(all_target_ids):
            # Find a reference embedding to infer shape + dtype.
            ref_arr: Optional[np.ndarray] = None
            for i in range(B):
                emb = out_dicts[i].get(sig_id)
                if emb is None:
                    continue
                ref_arr = np.asarray(emb).reshape(-1)
                break

            # Given how all_target_ids is built, this should always exist.
            if ref_arr is None:
                continue

            ref_dtype = ref_arr.dtype
            if ref_dtype not in (np.float16, np.float32):
                ref_dtype = np.float32

            ref_shape = tuple(ref_arr.shape)

            emb_list: List[np.ndarray] = []
            mask = np.ones((B,), dtype=np.int8)

            for i in range(B):
                emb = out_dicts[i].get(sig_id)
                if emb is None:
                    mask[i] = 0
                    emb_list.append(np.zeros(ref_shape, dtype=ref_dtype))
                else:
                    arr = np.asarray(emb, dtype=ref_dtype).reshape(-1)
                    emb_list.append(arr)

            # Per-output dropout (mask + zero embedding)
            p = float(self.drop_outputs_overrides.get(sig_id, p_drop_outputs))
            if p > 0.0:
                for i in range(B):
                    if mask[i] == 0:
                        continue
                    if random.random() < p:
                        mask[i] = 0
                        emb_list[i] = np.zeros(ref_shape, dtype=ref_dtype)

            output_emb_batch[sig_id] = emb_list
            output_mask_batch_np[sig_id] = mask

        # ---------------------------------------------------------------
        # 8) Optional: native outputs (eval only)
        # ---------------------------------------------------------------
        output_native_batch_np: Dict[int, np.ndarray] = {}
        if self.keep_output_native:
            assert self.output_id_to_name is not None

            for sig_id in sorted(all_target_ids):
                out_name = self.output_id_to_name.get(sig_id)
                if not out_name:
                    # If mapping is incomplete, skip rather than crashing.
                    continue

                # Infer shape from the first present value in this batch.
                ref_shape: Optional[Tuple[int, ...]] = None
                for i in range(B):
                    w = flat_windows[i]
                    out_group = w.get("output") or {}
                    out_info = out_group.get(out_name)

                    val = None
                    if isinstance(out_info, dict):
                        val = out_info.get("values", None)
                    elif out_info is not None:
                        val = out_info

                    if val is None:
                        continue

                    arr = np.asarray(val)
                    ref_shape = tuple(arr.shape)
                    break

                # If this output is missing everywhere in the batch, skip.
                if ref_shape is None:
                    continue

                per_sig_vals: List[np.ndarray] = []
                for i in range(B):
                    w = flat_windows[i]
                    out_group = w.get("output") or {}
                    out_info = out_group.get(out_name)

                    val = None
                    if isinstance(out_info, dict):
                        val = out_info.get("values", None)
                    elif out_info is not None:
                        val = out_info

                    if val is None:
                        arr = np.zeros(ref_shape, dtype=np.float32)
                    else:
                        arr = np.asarray(val, dtype=np.float32)
                        if tuple(arr.shape) != ref_shape:
                            raise ValueError(
                                f"Inconsistent native output shape for output={out_name!r} "
                                f"(signal_id={sig_id}): expected {ref_shape}, got {tuple(arr.shape)}"
                            )

                    per_sig_vals.append(arr)

                output_native_batch_np[sig_id] = np.stack(per_sig_vals, axis=0)

        # ---------------------------------------------------------------
        # 9) Convert arrays to torch
        # ---------------------------------------------------------------
        pos_t = torch.from_numpy(pos_batch).long()
        id_t = torch.from_numpy(id_batch).long()
        mod_t = torch.from_numpy(mod_batch.astype(np.int64))
        role_t = torch.from_numpy(role_batch.astype(np.int64))

        padding_mask_t = torch.from_numpy(padding_mask.astype(bool))
        input_mask_t = torch.from_numpy(input_mask.astype(bool))
        actuator_mask_t = torch.from_numpy(actuator_mask.astype(bool))

        # ---------------------------------------------------------------
        # 10) Pack embeddings by signal_id
        # ---------------------------------------------------------------
        emb_by_sid_np: Dict[int, List[np.ndarray]] = {}
        emb_index_np: Dict[int, List[int]] = {}

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

        emb_by_sid_t: Dict[int, torch.Tensor] = {}
        emb_index_t: Dict[int, torch.Tensor] = {}

        for sid_i, arr_list in emb_by_sid_np.items():
            try:
                stacked = np.stack(arr_list, axis=0)
            except Exception as e:
                shapes = [tuple(a.shape) for a in arr_list]
                raise ValueError(
                    f"Cannot stack embeddings for signal_id={sid_i}. shapes={shapes}"
                ) from e

            emb_by_sid_t[sid_i] = torch.from_numpy(stacked)
            emb_index_t[sid_i] = torch.as_tensor(emb_index_np[sid_i], dtype=torch.long)

        # Outputs: dense tensors of shape (B, D)
        output_emb_t: Dict[int, torch.Tensor] = {}
        output_mask_t: Dict[int, torch.Tensor] = {}

        for sig_id, emb_list in output_emb_batch.items():
            emb_arr = np.stack(emb_list, axis=0)
            output_emb_t[sig_id] = torch.from_numpy(emb_arr)
            output_mask_t[sig_id] = torch.from_numpy(
                output_mask_batch_np[sig_id].astype(bool)
            )

        # native outputs (optional)
        output_native_t: Dict[int, torch.Tensor] = {}
        if self.keep_output_native:
            for sig_id, arr in output_native_batch_np.items():
                output_native_t[sig_id] = torch.from_numpy(arr)

        # ---------------------------------------------------------------
        # 11) Assemble final batch dict
        # ---------------------------------------------------------------
        batch_out: Dict[str, Any] = {
            "emb": emb_by_sid_t,
            "emb_index": emb_index_t,
            "pos": pos_t,
            "id": id_t,
            "mod": mod_t,
            "role": role_t,
            "padding_mask": padding_mask_t,
            "input_mask": input_mask_t,
            "actuator_mask": actuator_mask_t,
            "output_emb": output_emb_t,
            "output_mask": output_mask_t,
        }

        if self.keep_output_native:
            batch_out["output_native"] = output_native_t

        # Keep shot/window identifiers if present (useful for debug)
        first = flat_windows[0]
        if "shot_id" in first:
            batch_out["shot_id"] = [w.get("shot_id") for w in flat_windows]
        if "window_index" in first:
            batch_out["window_index"] = [w.get("window_index") for w in flat_windows]

        return batch_out
