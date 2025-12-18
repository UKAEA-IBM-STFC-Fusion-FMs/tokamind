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
      • Preserves ragged embeddings (per-token projection happens in the model)
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
            "signal_name": np.ndarray(L,),          # human-friendly names

            "output_emb": {signal_id: np.ndarray(D_out), ...},
            "output_shapes": {signal_id: shape, ...},
            "output_names": {signal_id: name, ...},

            # Optionally (e.g. eval, if enabled in transforms):
            # "output": {... native output payloads ...}
        }

    Returned batch
    --------------
    A dictionary with the following keys:

    Token-level inputs
    ------------------
    "emb"            : List[List[torch.Tensor]]
                       Ragged embeddings before projection:
                         emb[b][t] is a 1D tensor (D_i,)
                       Padded positions (t >= length_b) are left as empty
                       tensors (shape (0,)).

    "pos"            : LongTensor (B, L)
    "id"             : LongTensor (B, L)
    "mod"            : LongTensor (B, L)
    "role"           : LongTensor (B, L)
    "signal_name"    : List[List[str]]

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
    `cfg_collate` is expected to follow the structure in `finetune_default.yaml`:

    .code-block:: yaml

        collate:
          # INPUT DROPOUT
          p_drop_inputs: 0.08
          p_drop_inputs_overrides: {}          # keyed by signal_name

          # OUTPUT DROPOUT
          p_drop_outputs: 0.0
          p_drop_outputs_overrides: {}         # keyed by output signal_name

          # ACTUATORS DROPOUT
          p_drop_actuators: 0.0
          p_drop_actuators_overrides: {}       # keyed by signal_name

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

        # Override dicts (keyed by *names* now)
        self.drop_inputs_overrides = cfg_collate.get("p_drop_inputs_overrides", {})
        self.drop_act_overrides = cfg_collate.get("p_drop_actuators_overrides", {})
        self.drop_outputs_overrides = cfg_collate.get("p_drop_outputs_overrides", {})

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
        name_lists: List[List[str]] = []

        out_dicts: List[Dict[int, Any]] = []
        out_shapes_dicts: List[Dict[int, Any]] = []

        all_target_ids: set[int] = set()
        id_to_output_name: Dict[int, str] = {}

        for w in flat_windows:
            # Keep embeddings ragged (list-of-arrays)
            emb_lists.append(w["emb_chunks"])

            # Force signed dtypes (prevents any accidental uint wrap)
            pos_lists.append(np.asarray(w["pos"], dtype=np.int32))
            id_lists.append(np.asarray(w["id"], dtype=np.int32))
            mod_lists.append(np.asarray(w["mod"], dtype=np.int16))
            role_lists.append(np.asarray(w["role"], dtype=np.int8))
            name_lists.append(list(w["signal_name"]))

            out_dicts.append(w["output_emb"])
            out_shapes_dicts.append(w["output_shapes"])

            output_names = w.get("output_names", {})
            all_target_ids.update(w["output_emb"].keys())

            for sid, sname in output_names.items():
                if sid not in id_to_output_name:
                    id_to_output_name[sid] = sname
                elif id_to_output_name[sid] != sname:
                    raise ValueError(
                        f"Inconsistent output name for signal_id={sid}: "
                        f"'{id_to_output_name[sid]}' vs '{sname}'"
                    )

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

        # Ragged embeddings kept as Python nested lists of np.ndarrays
        emb_batch: List[List[np.ndarray]] = [
            [self._empty_emb for _ in range(L_max)] for _ in range(B)
        ]
        name_batch: List[List[str]] = [[""] * L_max for _ in range(B)]

        # --------------------------------------------------------------- #
        # 3. Fill padded arrays
        # --------------------------------------------------------------- #
        for i in range(B):
            Li = lengths[i]
            pos_batch[i, :Li] = pos_lists[i]
            id_batch[i, :Li] = id_lists[i]
            mod_batch[i, :Li] = mod_lists[i]
            role_batch[i, :Li] = role_lists[i]

            # Ragged fill
            for t in range(Li):
                emb_batch[i][t] = emb_lists[i][t]
                name_batch[i][t] = name_lists[i][t]

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
                sig_name = name_batch[i][t]
                p = float(self.drop_inputs_overrides.get(sig_name, p_drop_in))
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
                sig_name = name_batch[i][t]
                p = float(self.drop_act_overrides.get(sig_name, p_drop_act))
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
            out_name = id_to_output_name.get(sig_id, str(sig_id))

            raw_list: List[Optional[np.ndarray]] = []
            mask = np.ones((B,), dtype=np.int8)
            ref_shape: Optional[Tuple[int, ...]] = None

            # First pass: collect embeddings and remember a reference shape
            for i in range(B):
                emb = out_dicts[i].get(sig_id, None)
                if emb is None:
                    mask[i] = 0
                    raw_list.append(None)
                else:
                    arr = np.asarray(emb, dtype=np.float32).reshape(-1)
                    raw_list.append(arr)
                    if ref_shape is None:
                        ref_shape = arr.shape

            # Given how all_target_ids is built, this should always hold.
            if ref_shape is None:
                raise AssertionError(
                    "MMTCollate invariant broken: no output embedding found for sig_id."
                )

            # Second pass: fill missing with zeros of the right shape
            emb_list: List[np.ndarray] = []
            for arr in raw_list:
                if arr is None:
                    arr = np.zeros(ref_shape, dtype=np.float32)
                emb_list.append(arr)

            # Per-output dropout (mask only, embedding left as-is / zeros)
            for i in range(B):
                p = float(self.drop_outputs_overrides.get(out_name, p_drop_outputs))
                if random.random() < p:
                    mask[i] = 0

            output_emb_batch[sig_id] = emb_list
            output_mask_batch_np[sig_id] = mask

        # --------------------------------------------------------------- #
        # 8. Optional: native outputs
        # --------------------------------------------------------------- #
        output_native_batch_np: Dict[int, np.ndarray] = {}
        if self.keep_output_native:

            def _find_shape(sig_id: int) -> Any:
                # Prefer the shape from the same index; else fallback scan
                for sd in out_shapes_dicts:
                    if sig_id in sd:
                        return sd[sig_id]
                return None

            for sig_id in sorted(all_target_ids):
                out_name = id_to_output_name.get(sig_id, str(sig_id))
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
                        shape = _find_shape(sig_id)
                        if shape is None:
                            raise ValueError(
                                f"Missing shape for output signal_id={sig_id} "
                                f"while assembling output_native."
                            )
                        arr = np.zeros(shape, dtype=np.float32)
                    else:
                        arr = np.asarray(val, dtype=np.float32)

                    per_sig_vals.append(arr)

                output_native_batch_np[sig_id] = np.stack(per_sig_vals, axis=0)

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

        # Ragged embeddings
        emb_t: List[List[torch.Tensor]] = [
            [torch.empty(0) for _ in range(L_max)] for _ in range(B)
        ]
        for i in range(B):
            for t in range(L_max):
                arr = emb_batch[i][t]
                if arr is not None:
                    emb_t[i][t] = torch.from_numpy(arr)

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
            "emb": emb_t,
            "pos": pos_t,
            "id": id_t,
            "mod": mod_t,
            "role": role_t,
            "signal_name": name_batch,  # still List[List[str]]
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
