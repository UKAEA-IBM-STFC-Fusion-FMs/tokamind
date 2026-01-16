"""
Batch collation for the MMT window-level dataloaders.

MMTCollate takes a list of per-window dictionaries (produced by the transforms
pipeline) and builds a padded, model-ready batch by:

x- padding variable-length token sequences and packing token embeddings by signal_id,
- applying per-token and per-chunk dropout for inputs/actuators (and optional
  per-output dropout),
- producing masks for padding and dropped tokens,
- assembling output embeddings (and optionally native output tensors for eval).

This collate uses explicit PAD semantics (PAD id/role/mod/pos) so padding/dropped
tokens are never confused with real signals.

The returned batch dict is the standard input format expected by
MultiModalTransformer.forward().
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
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


class MMTCollate:
    """
    Collate function for window-level MMT batches (pretraining + finetuning).

    This collate:
      • Pads variable-length token sequences
      • Applies input, actuator, chunk, and output dropout
      • Builds masks for padding and dropped tokens
      • Packs token embeddings by signal_id (per-token projection happens in the model)
      • Uses explicit PAD semantics so that PAD tokens are never confused
        with real signals (sid = -1, role = -1, mod = -1, pos = 0)

    Expected input
    --------------
    `batch` is a `List[window_dict]`, where **each element corresponds to a
    single model window**, as produced by `BuildTokensTransform` and coming
    from either:

      - `WindowStreamedDataset` (streaming path), or
      - `WindowCachedDataset` (RAM-cached path).

    Each `window_dict` should minimally contain:

    .. code-block:: python

        {
            "shot_id": ...,
            "window_index": ...,
            "emb_chunks": [np.ndarray(D_i), ...],   # ragged token embeddings
            "pos": np.ndarray(L,),                  # token positions
            "id": np.ndarray(L,),                   # signal IDs
            "mod": np.ndarray(L,),                  # modality IDs
            "role": np.ndarray(L,),                 # role IDs

            "output_emb": {signal_id: np.ndarray(D_out), ...},

            # Optionally (e.g. eval, if enabled in transforms):
            # "output": {... native output payloads ...}
        }

    Returned batch
    --------------
    A dictionary with the following keys:

    Token-level inputs
    ------------------
    "emb"            : Dict[int, torch.Tensor]
                       Packed token embeddings by signal_id (sid).
                       For each sid, emb[sid] has shape (N_sid, D_sid), where
                       N_sid is the number of kept tokens of that sid in this batch.

    "emb_index"      : Dict[int, LongTensor]
                       For each sid, emb_index[sid] has shape (N_sid,) and contains
                       flattened indices into the padded (B, L) token grid:
                           flat_index = b * L + t
                       aligned row-by-row with emb[sid].

                       This representation drastically reduces the number of
                       torch.Storage objects crossing process boundaries (fixes
                       "Too many open files" issues with multi-worker DataLoaders).

    "pos"            : LongTensor (B, L)
    "id"             : LongTensor (B, L)
    "mod"            : LongTensor (B, L)
    "role"           : LongTensor (B, L)
    "padding_mask"   : BoolTensor (B, L)
                       True where there is a *real* token (not padding),
                       i.e. where original length > t.

    "input_mask"     : BoolTensor (B, L)
                       True where the input token is kept (not dropped).

    "actuator_mask"  : BoolTensor (B, L)
                       True where the actuator token is kept (not dropped).

    Outputs
    -------
    "output_emb"    : Dict[int, torch.Tensor]
                       For each output `signal_id`:
                         tensor shape = (B, D_out)

    "output_mask"   : Dict[int, BoolTensor] with shape (B,)
                       Per-output presence/dropout mask.

    If `cfg_collate["keep_output_native"] == True`, also:

    "output_native"  : Dict[int, torch.Tensor]
                       Each tensor has shape (B, *orig_output_shape).

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
    """

    # ------------------------------------------------------------------ #
    def __init__(self, cfg_collate: Dict[str, Any]) -> None:
        self.cfg = cfg_collate
        self.keep_output_native = bool(cfg_collate.get("keep_output_native", False))

        def _require_int_keys(d: Any, *, field: str) -> Dict[int, float]:
            """Return a copy of `d` with int keys and float values.

            We keep config name-based at the boundary (YAML), but once we're in
            the hot path we want signal_id keys for compactness and speed.
            """

            if d is None:
                return {}
            if not isinstance(d, dict):
                raise TypeError(
                    f"MMTCollate expects cfg_collate['{field}'] to be a dict, got {type(d)}"
                )

            out: Dict[int, float] = {}
            for k, v in d.items():
                if isinstance(k, int):
                    sid = int(k)
                elif isinstance(k, str) and k.isdigit():
                    sid = int(k)
                else:
                    raise TypeError(
                        f"MMTCollate expects cfg_collate['{field}'] to be keyed by "
                        f"signal_id (int). Got key={k!r}. "
                        "Did you forget to pass the config through "
                        "pipeline_helpers.make_collate_fn (name->id conversion)?"
                    )

                out[sid] = float(v)

            return out

        # Override dicts (keyed by signal_id, post-conversion)
        self.drop_inputs_overrides = _require_int_keys(
            cfg_collate.get("p_drop_inputs_overrides", {}),
            field="p_drop_inputs_overrides",
        )
        self.drop_act_overrides = _require_int_keys(
            cfg_collate.get("p_drop_actuators_overrides", {}),
            field="p_drop_actuators_overrides",
        )
        self.drop_outputs_overrides = _require_int_keys(
            cfg_collate.get("p_drop_outputs_overrides", {}),
            field="p_drop_outputs_overrides",
        )

        # Needed only for `keep_output_native`: native outputs are still stored
        # by name in window["output"], while the training graph is keyed by id.
        id_to_name = cfg_collate.get("output_id_to_name", {}) or {}
        if not isinstance(id_to_name, dict):
            raise TypeError(
                "MMTCollate expects cfg_collate['output_id_to_name'] to be a dict "
                f"(signal_id -> output_name), got {type(id_to_name)}"
            )
        self.output_id_to_name: Dict[int, str] = {
            int(k): str(v) for k, v in id_to_name.items()
        }
        if self.keep_output_native and (not self.output_id_to_name):
            raise ValueError(
                "MMTCollate.keep_output_native=True requires cfg_collate['output_id_to_name'] "
                "(signal_id -> output_name)."
            )

        # Reusable empty embedding for padded slots
        self._empty_emb = np.empty((0,), dtype=np.float32)

    # ------------------------------------------------------------------ #
    def __call__(self, batch: List[Any]) -> Dict[str, Any]:
        """
        Collate a batch of MMT windows into padded model-ready tensors.

        Input
        -----
        batch: List[dict]
            Each element must be a single window dict as described in the
            class docstring.  `None` entries are skipped (defensive) but
            should not normally appear when using WindowStreamedDataset or
            WindowCachedDataset.
        """
        # --------------------------------------------------------------- #
        # 0. Sanity-check + filter Nones → flat list of window dicts
        # --------------------------------------------------------------- #
        flat_windows: List[Dict[str, Any]] = []
        for item in batch:
            if item is None:
                continue
            if not isinstance(item, dict):
                raise TypeError(
                    "MMTCollate expects each batch element to be a single "
                    f"window dict, got {type(item)} instead."
                )
            flat_windows.append(item)

        B = len(flat_windows)
        if B == 0:
            raise ValueError("MMTCollate received an empty batch of windows.")

        # --------------------------------------------------------------- #
        # 1. Extract per-window token arrays + output metadata
        # --------------------------------------------------------------- #
        emb_lists: List[List[np.ndarray]] = []
        pos_lists: List[np.ndarray] = []
        id_lists: List[np.ndarray] = []
        mod_lists: List[np.ndarray] = []
        role_lists: List[np.ndarray] = []
        out_dicts: List[Dict[int, Any]] = []
        all_target_ids: set[int] = set()

        for w in flat_windows:
            # Token embeddings are a ragged list-of-arrays per window (we pack them by sid at the end).
            emb_lists.append(w["emb_chunks"])

            # Force signed dtypes (prevents any accidental uint wrap)
            pos_lists.append(np.asarray(w["pos"], dtype=np.int32))
            id_lists.append(np.asarray(w["id"], dtype=np.int32))
            mod_lists.append(np.asarray(w["mod"], dtype=np.int16))
            role_lists.append(np.asarray(w["role"], dtype=np.int8))

            out_dict = w.get("output_emb") or {}
            out_dicts.append(out_dict)
            all_target_ids.update(out_dict.keys())

        # --------------------------------------------------------------- #
        # 2. Determine max token length + allocate padded arrays (NumPy)
        # --------------------------------------------------------------- #
        lengths = [len(e) for e in emb_lists]
        L_max = max(lengths)

        pos_batch = np.full((B, L_max), PAD_POS, dtype=np.int32)
        id_batch = np.full((B, L_max), PAD_ID, dtype=np.int32)
        mod_batch = np.full((B, L_max), PAD_MOD, dtype=np.int16)
        role_batch = np.full((B, L_max), PAD_ROLE, dtype=np.int8)

        padding_mask = np.zeros((B, L_max), dtype=np.int8)
        input_mask = np.ones((B, L_max), dtype=np.int8)
        actuator_mask = np.ones((B, L_max), dtype=np.int8)

        # Per-token embeddings (NumPy) during collation. These are packed by sid before returning.
        emb_batch: List[List[np.ndarray]] = [
            [self._empty_emb for _ in range(L_max)] for _ in range(B)
        ]

        # --------------------------------------------------------------- #
        # 3. Fill padded arrays
        # --------------------------------------------------------------- #
        for i in range(B):
            Li = lengths[i]
            pos_batch[i, :Li] = pos_lists[i]
            id_batch[i, :Li] = id_lists[i]
            mod_batch[i, :Li] = mod_lists[i]
            role_batch[i, :Li] = role_lists[i]

            # Fill token embeddings
            for t in range(Li):
                emb_batch[i][t] = emb_lists[i][t]

            padding_mask[i, :Li] = 1

        # --------------------------------------------------------------- #
        # Helper: drop a token (zero embedding + PAD metadata + role mask)
        # --------------------------------------------------------------- #
        def _drop_token(i: int, t: int, *, kind: str) -> None:
            if kind == "input":
                input_mask[i, t] = 0
            elif kind == "actuator":
                actuator_mask[i, t] = 0
            else:
                raise ValueError(f"Unknown drop kind: {kind!r}")

            emb_batch[i][t] = np.zeros(
                emb_batch[i][t].shape, dtype=emb_batch[i][t].dtype
            )
            id_batch[i, t] = PAD_ID
            mod_batch[i, t] = PAD_MOD
            role_batch[i, t] = PAD_ROLE
            pos_batch[i, t] = PAD_POS

        # --------------------------------------------------------------- #
        # 4. Input dropout (per-token)
        # --------------------------------------------------------------- #
        p_drop_in = float(self.cfg.get("p_drop_inputs", 0.0))
        for i in range(B):
            Li = lengths[i]
            if Li == 0:
                continue

            idxs = np.where(role_batch[i, :Li] == ROLE_CONTEXT)[0]
            for t in idxs:
                sid = int(id_batch[i, t])
                p = float(self.drop_inputs_overrides.get(sid, p_drop_in))
                if random.random() < p:
                    _drop_token(i, int(t), kind="input")

        # --------------------------------------------------------------- #
        # 5. Actuator dropout (per-token)
        # --------------------------------------------------------------- #
        p_drop_act = float(self.cfg.get("p_drop_actuators", 0.0))
        for i in range(B):
            Li = lengths[i]
            if Li == 0:
                continue

            idxs = np.where(role_batch[i, :Li] == ROLE_ACTUATOR)[0]
            for t in idxs:
                sid = int(id_batch[i, t])
                p = float(self.drop_act_overrides.get(sid, p_drop_act))
                if random.random() < p:
                    _drop_token(i, int(t), kind="actuator")

        # --------------------------------------------------------------- #
        # 6. Chunk dropout (per-pos group)
        # --------------------------------------------------------------- #
        p_drop_inputs_chunks = float(self.cfg.get("p_drop_inputs_chunks", 0.0))
        p_drop_actuators_chunks = float(self.cfg.get("p_drop_actuators_chunks", 0.0))

        if (p_drop_inputs_chunks > 0.0) or (p_drop_actuators_chunks > 0.0):
            for i in range(B):
                Li = lengths[i]
                if Li == 0:
                    continue

                # Group token indices by pos (stable, avoids repeated np.where)
                pos_i = pos_batch[i, :Li]
                order = np.argsort(pos_i)
                pos_sorted = pos_i[order]
                split = np.where(np.diff(pos_sorted) != 0)[0] + 1
                groups = np.split(order, split)

                for idxs in groups:
                    if idxs.size == 0:
                        continue

                    roles = role_batch[i, idxs]

                    # Input chunks
                    if (ROLE_CONTEXT in roles) and (
                        random.random() < p_drop_inputs_chunks
                    ):
                        for t in idxs:
                            if role_batch[i, t] == ROLE_CONTEXT:
                                _drop_token(i, int(t), kind="input")

                    # Actuator chunks
                    if (ROLE_ACTUATOR in roles) and (
                        random.random() < p_drop_actuators_chunks
                    ):
                        for t in idxs:
                            if role_batch[i, t] == ROLE_ACTUATOR:
                                _drop_token(i, int(t), kind="actuator")

        # --------------------------------------------------------------- #
        # 7. Output embeddings + dropout (still NumPy here)
        # --------------------------------------------------------------- #
        p_drop_outputs = float(self.cfg.get("p_drop_outputs", 0.0))

        output_emb_batch: Dict[int, List[np.ndarray]] = {}
        output_mask_batch_np: Dict[int, np.ndarray] = {}

        for sig_id in sorted(all_target_ids):
            raw_list: List[Optional[np.ndarray]] = []
            mask = np.ones((B,), dtype=np.int8)
            ref_shape: Optional[Tuple[int, ...]] = None
            ref_dtype: Optional[np.dtype] = None

            # First pass: collect embeddings and remember a reference shape/dtype.
            for i in range(B):
                emb = out_dicts[i].get(sig_id, None)
                if emb is None:
                    mask[i] = 0
                    raw_list.append(None)
                else:
                    # IMPORTANT: do NOT force float32 here.
                    # Keep the cached/streamed dtype (e.g. float16) all the way
                    # through collate to avoid worker/prefetch memory spikes.
                    arr = np.asarray(emb).reshape(-1)
                    if ref_dtype is None:
                        ref_dtype = arr.dtype
                    elif arr.dtype != ref_dtype:
                        arr = arr.astype(ref_dtype, copy=False)

                    raw_list.append(arr)
                    if ref_shape is None:
                        ref_shape = arr.shape

            # Given how all_target_ids is built, this should always hold.
            if ref_shape is None:
                raise AssertionError(
                    "MMTCollate invariant broken: no output embedding found for sig_id."
                )
            if ref_dtype is None:
                ref_dtype = np.dtype(np.float32)

            # Second pass: fill missing with zeros of the right shape/dtype.
            emb_list: List[np.ndarray] = []
            for arr in raw_list:
                if arr is None:
                    arr = np.zeros(ref_shape, dtype=ref_dtype)
                emb_list.append(arr)

            # Per-output dropout (mask only, embedding left as-is / zeros)
            for i in range(B):
                p = float(self.drop_outputs_overrides.get(int(sig_id), p_drop_outputs))
                if random.random() < p:
                    mask[i] = 0

            output_emb_batch[int(sig_id)] = emb_list
            output_mask_batch_np[int(sig_id)] = mask

        # --------------------------------------------------------------- #
        # 8. Optional: native outputs
        # --------------------------------------------------------------- #
        output_native_batch_np: Dict[int, np.ndarray] = {}
        if self.keep_output_native:
            for sig_id in sorted(all_target_ids):
                sid = int(sig_id)
                if sid not in self.output_id_to_name:
                    raise KeyError(
                        f"Missing output_id_to_name mapping for output signal_id={sid}."
                    )

                out_name = self.output_id_to_name[sid]

                # Infer a reference native shape from the first non-missing value.
                ref_shape: Optional[Tuple[int, ...]] = None
                for i in range(B):
                    out_group = flat_windows[i].get("output") or {}
                    out_info = out_group.get(out_name)
                    val = None
                    if isinstance(out_info, dict):
                        val = out_info.get("values", None)
                    elif out_info is not None:
                        val = out_info

                    if val is not None:
                        ref_shape = tuple(np.asarray(val).shape)
                        break

                if ref_shape is None:
                    raise ValueError(
                        f"Missing native output values for output signal_id={sid} (name={out_name!r}). "
                        "Make sure keep_output_native is enabled in the transforms pipeline."
                    )

                per_sig_vals: List[np.ndarray] = []
                for i in range(B):
                    out_group = flat_windows[i].get("output") or {}
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
                                f"Inconsistent native output shape for output signal_id={sid} (name={out_name!r}): "
                                f"expected {ref_shape}, got {tuple(arr.shape)}"
                            )

                    per_sig_vals.append(arr)

                output_native_batch_np[sid] = np.stack(per_sig_vals, axis=0)

        # --------------------------------------------------------------- #
        # 9. Convert everything to torch.Tensor
        # --------------------------------------------------------------- #
        pos_t = torch.from_numpy(pos_batch).long()
        id_t = torch.from_numpy(id_batch).long()
        mod_t = torch.from_numpy(mod_batch.astype(np.int64))
        role_t = torch.from_numpy(role_batch.astype(np.int64))

        padding_mask_t = torch.from_numpy(padding_mask.astype(bool))
        input_mask_t = torch.from_numpy(input_mask.astype(bool))
        actuator_mask_t = torch.from_numpy(actuator_mask.astype(bool))

        # Packed embeddings by signal_id (sid).
        #
        # Instead of returning a nested List[List[Tensor]] (one tensor per token),
        # we pack embeddings by sid to avoid creating thousands of small tensor
        # storages per batch (which can trigger "Too many open files" errors when
        # using multiprocessing DataLoaders).
        emb_by_sid_np: Dict[int, List[np.ndarray]] = {}
        emb_index_np: Dict[int, List[int]] = {}

        for i in range(B):
            for t in range(L_max):
                sid_i = int(id_batch[i, t])
                if sid_i == PAD_ID:
                    continue
                arr = emb_batch[i][t]
                if arr is None or arr.size == 0:
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

        # Outputs: embeddings (dense tensors of shape (B, D))
        output_emb_t: Dict[int, torch.Tensor] = {}
        output_mask_t: Dict[int, torch.Tensor] = {}

        for sig_id, emb_list in output_emb_batch.items():
            emb_arr = np.stack(emb_list, axis=0)  # (B, D)
            output_emb_t[sig_id] = torch.from_numpy(emb_arr)
            output_mask_t[sig_id] = torch.from_numpy(
                output_mask_batch_np[sig_id].astype(bool)
            )

        # native outputs (optional)
        output_native_t: Dict[int, torch.Tensor] = {}
        if self.keep_output_native:
            for sig_id, arr in output_native_batch_np.items():
                output_native_t[sig_id] = torch.from_numpy(arr)

        # --------------------------------------------------------------- #
        # 10. Assemble final batch dict (torch)
        # --------------------------------------------------------------- #
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

        # Adding shot id and window index to batch (kept from your version)
        first = flat_windows[0]
        if "shot_id" in first:
            batch_out["shot_id"] = [w["shot_id"] for w in flat_windows]
        if "window_index" in first:
            batch_out["window_index"] = [w["window_index"] for w in flat_windows]

        return batch_out
