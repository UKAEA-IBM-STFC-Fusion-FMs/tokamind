from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import random
import torch

# Token roles (same as in BuildTokensTransform)
ROLE_CONTEXT = 0
ROLE_ACTUATOR = 1
ROLE_OUTPUT = 2

# Explicit PAD semantics for token-level fields
PAD_ID = -1  # never a real signal_id
PAD_ROLE = -1  # no semantic role
PAD_MOD = -1  # no modality
PAD_POS = 0  # arbitrary, unused by model for PAD


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
                    "MMTCollate now expects each batch element to be a single "
                    f"window dict, got {type(item)} instead."
                )
            flat_windows.append(item)

        B = len(flat_windows)
        if B == 0:
            raise ValueError("MMTCollate received an empty batch of windows.")

        # --------------------------------------------------------------- #
        # 1. Extract per-window token arrays + output metadata
        # --------------------------------------------------------------- #
        emb_lists = []
        pos_lists = []
        id_lists = []
        mod_lists = []
        role_lists = []
        name_lists = []

        out_dicts = []
        out_shapes_dicts = []
        out_names_dicts = []

        all_target_ids = set()
        id_to_output_name: Dict[int, str] = {}

        for w in flat_windows:
            emb_lists.append(w["emb_chunks"])
            pos_lists.append(w["pos"])
            id_lists.append(w["id"])
            mod_lists.append(w["mod"])
            role_lists.append(w["role"])
            name_lists.append(w["signal_name"])

            out_dicts.append(w["output_emb"])
            out_shapes_dicts.append(w["output_shapes"])
            output_names = w.get("output_names", {})
            out_names_dicts.append(output_names)

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

        # NOTE: we now initialise with explicit PAD values so that padded
        # tokens are never mistaken for real signals.
        pos_batch = np.full((B, L_max), PAD_POS, dtype=np.int32)
        id_batch = np.full((B, L_max), PAD_ID, dtype=np.int32)
        mod_batch = np.full((B, L_max), PAD_MOD, dtype=np.int16)
        role_batch = np.full((B, L_max), PAD_ROLE, dtype=np.int8)
        padding_mask = np.zeros((B, L_max), dtype=np.int8)

        input_mask = np.ones((B, L_max), dtype=np.int8)
        actuator_mask = np.ones((B, L_max), dtype=np.int8)

        # Ragged embeddings kept as Python nested lists of np.ndarrays
        emb_batch: List[List[np.ndarray]] = [
            [np.empty((0,), dtype=np.float32) for _ in range(L_max)] for _ in range(B)
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

            for t in range(Li):
                emb_batch[i][t] = emb_lists[i][t]
                name_batch[i][t] = name_lists[i][t]

            padding_mask[i, :Li] = 1

        # --------------------------------------------------------------- #
        # 4. Input dropout (per-token)
        # --------------------------------------------------------------- #
        p_drop_in = self.cfg.get("p_drop_inputs", 0.0)

        for i in range(B):
            for t in range(lengths[i]):
                if role_batch[i, t] == ROLE_CONTEXT:
                    sig_name = name_batch[i][t]
                    p = self.drop_inputs_overrides.get(sig_name, p_drop_in)
                    if random.random() < p:
                        input_mask[i, t] = 0
                        emb_batch[i][t] = np.zeros_like(emb_batch[i][t])

        # --------------------------------------------------------------- #
        # 5. Actuator dropout (per-token)
        # --------------------------------------------------------------- #
        p_drop_act = self.cfg.get("p_drop_actuators", 0.0)

        for i in range(B):
            for t in range(lengths[i]):
                if role_batch[i, t] == ROLE_ACTUATOR:
                    sig_name = name_batch[i][t]
                    p = self.drop_act_overrides.get(sig_name, p_drop_act)
                    if random.random() < p:
                        actuator_mask[i, t] = 0
                        emb_batch[i][t] = np.zeros_like(emb_batch[i][t])

        # --------------------------------------------------------------- #
        # 6. Chunk dropout (per-pos group)
        # --------------------------------------------------------------- #
        p_drop_inputs_chunks = self.cfg.get("p_drop_inputs_chunks", 0.0)
        p_drop_act_chunks = self.cfg.get("p_drop_actuators_chunks", 0.0)

        for i in range(B):
            Li = lengths[i]
            unique_pos = set(pos_batch[i, :Li])

            for pval in unique_pos:
                idxs = np.where(pos_batch[i, :Li] == pval)[0]
                if len(idxs) == 0:
                    continue

                # input chunks
                if any(role_batch[i, t] == ROLE_CONTEXT for t in idxs):
                    if random.random() < p_drop_inputs_chunks:
                        for t in idxs:
                            if role_batch[i, t] == ROLE_CONTEXT:
                                input_mask[i, t] = 0
                                emb_batch[i][t] = np.zeros_like(emb_batch[i][t])

                # actuator chunks
                if any(role_batch[i, t] == ROLE_ACTUATOR for t in idxs):
                    if random.random() < p_drop_act_chunks:
                        for t in idxs:
                            if role_batch[i, t] == ROLE_ACTUATOR:
                                actuator_mask[i, t] = 0
                                emb_batch[i][t] = np.zeros_like(emb_batch[i][t])

        # --------------------------------------------------------------- #
        # 7. Output embeddings + dropout (still NumPy here)
        # --------------------------------------------------------------- #
        p_drop_outputs = self.cfg.get("p_drop_outputs", 0.0)

        output_emb_batch: Dict[int, List[np.ndarray]] = {}
        output_mask_batch_np: Dict[int, np.ndarray] = {}

        for sig_id in sorted(all_target_ids):
            out_name = id_to_output_name[sig_id]

            raw_list: List[np.ndarray | None] = []
            mask = np.ones((B,), dtype=np.int8)
            ref_shape = None

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

            # Sanity: given how all_target_ids is built, this should always hold.
            assert ref_shape is not None, (
                "MMTCollate invariant broken: no emb for sig_id."
            )

            # Second pass: fill missing with zeros of the right shape
            emb_list: List[np.ndarray] = []
            for arr in raw_list:
                if arr is None:
                    arr = np.zeros(ref_shape, dtype=np.float32)
                emb_list.append(arr)

            # Per-output dropout
            for i in range(B):
                p = self.drop_outputs_overrides.get(out_name, p_drop_outputs)
                if random.random() < p:
                    mask[i] = 0

            output_emb_batch[sig_id] = emb_list
            output_mask_batch_np[sig_id] = mask

        # --------------------------------------------------------------- #
        # 8. Optional: native outputs
        # --------------------------------------------------------------- #
        output_native_batch_np: Dict[int, np.ndarray] = {}
        if self.keep_output_native:
            for sig_id in sorted(all_target_ids):
                out_name = id_to_output_name[sig_id]
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
                        # Use shapes dict to reconstruct shape
                        shape = None
                        shapes_dict = out_shapes_dicts[i]
                        if sig_id in shapes_dict:
                            shape = shapes_dict[sig_id]
                        else:
                            for sd in out_shapes_dicts:
                                if sig_id in sd:
                                    shape = sd[sig_id]
                                    break
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
        # Token-level metadata
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

        # Adding shot id and window index to batch
        first = flat_windows[0]
        if "shot_id" in first:
            batch_out["shot_id"] = [w["shot_id"] for w in flat_windows]
        if "window_index" in first:
            batch_out["window_index"] = [w["window_index"] for w in flat_windows]

        return batch_out
