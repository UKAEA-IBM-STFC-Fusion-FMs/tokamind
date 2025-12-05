from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import random


# Token roles (same as in BuildTokensTransform)
ROLE_CONTEXT = 0
ROLE_ACTUATOR = 1
ROLE_OUTPUT = 2


class MMTCollate:
    """
    Universal collate function for both pretraining and finetuning.

    This collate:
      • Pads variable-length sequences
      • Applies input, actuator, chunk, and output dropout
      • Builds masks for padding and dropped tokens
      • Preserves ragged embeddings (per-token projection happens in the model)

    Input:
      List[window_dict] where each window was produced by BuildTokensTransform.

    Output:
      A batch dictionary with:
        "emb"            : list[list[np.ndarray]]   # ragged before projection
        "pos"            : (B, L)
        "id"             : (B, L)
        "mod"            : (B, L)
        "role"           : (B, L)
        "signal_name"    : (B, L) list of strings
        "padding_mask"   : (B, L)
        "input_mask"     : (B, L)
        "actuator_mask"  : (B, L)
        "outputs_emb"    : dict[signal_id -> list[np.ndarray]]
        "outputs_mask"   : dict[signal_id -> (B,)]

      If cfg_collate["keep_output_native"] == True, also:
        "output_native"  : dict[signal_id -> np.ndarray of shape (B, *orig_shape)]
    """

    # ------------------------------------------------------------------ #
    def __init__(self, cfg_collate: Dict[str, Any]) -> None:
        """
        cfg_collate comes from finetune_default.yaml:

        collate:
          # INPUT DROPOUT
          p_drop_inputs: 0.08
          p_drop_inputs_overrides: {}          # keyed by signal_name

          # output DROPOUT
          p_drop_outputs: 0.0
          p_drop_outputs_overrides: {}         # keyed by output signal_name

          # ACTUATORS DROPOUT
          p_drop_actuators: 0.0
          p_drop_actuators_overrides: {}      # keyed by signal_name

          # CHUNK DROPOUT (coarse time-based masking)
          p_drop_inputs_chunks: 0.08
          p_drop_actuators_chunks: 0.0

          # EVAL-ONLY: include native outputs (Y_native)
          # keep_output_native: false
        """
        self.cfg = cfg_collate
        self.keep_output_native = bool(cfg_collate.get("keep_output_native", False))

        # Override dicts (keyed by *names* now)
        self.drop_inputs_overrides = cfg_collate.get("p_drop_inputs_overrides", {})
        self.drop_act_overrides = cfg_collate.get("p_drop_actuators_overrides", {})
        self.drop_outputs_overrides = cfg_collate.get("p_drop_outputs_overrides", {})

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def __call__(self, batch: List[Any]) -> Dict[str, Any]:
        """
        Collate a batch.

        The baseline TaskModelTransformWrapper returns, for each dataset
        index, an *iterable of windows* (one per window in that shot).

        DataLoader therefore passes a batch shaped like:

            batch = [windows_for_shot_0, windows_for_shot_1, ...]

        where each element is a list or generator of window dicts.

        Here we first flatten this into a single list of window dicts:

            flat_windows = [win_0, win_1, ..., win_N]

        and then apply padding + dropout logic on those windows.
        """
        # --------------------------------------------------------------- #
        # 0. Flatten shot-level sequences into a list of windows
        # --------------------------------------------------------------- #
        flat_windows: List[Dict[str, Any]] = []

        for item in batch:
            if item is None:
                continue

            # If it's already a dict, treat as a single window
            if isinstance(item, dict):
                flat_windows.append(item)
                continue

            # Otherwise assume it's an iterable of windows (list/generator)
            try:
                for w in item:
                    if w is None:
                        continue
                    if not isinstance(w, dict):
                        raise TypeError(
                            "MMTCollate expected inner elements to be window dicts, "
                            f"got {type(w)}"
                        )
                    flat_windows.append(w)
            except TypeError:
                raise TypeError(
                    "MMTCollate expected each batch element to be an iterable "
                    "of window dicts (as returned by TaskModelTransformWrapper), "
                    f"got {type(item)}"
                )

        B = len(flat_windows)
        if B == 0:
            raise ValueError("MMTCollate received an empty flattened batch.")

        # --------------------------------------------------------------- #
        # 1. Extract all fields per window
        # --------------------------------------------------------------- #
        emb_lists: List[List[np.ndarray]] = []
        pos_lists: List[np.ndarray] = []
        id_lists: List[np.ndarray] = []
        mod_lists: List[np.ndarray] = []
        role_lists: List[np.ndarray] = []
        name_lists: List[np.ndarray] = []

        tgt_dicts: List[Dict[int, np.ndarray]] = []
        tgt_shapes_dicts: List[Dict[int, Any]] = []
        tgt_names_dicts: List[Dict[int, str]] = []

        all_target_ids = set()
        id_to_output_name: Dict[int, str] = {}

        for w in flat_windows:
            emb_lists.append(w["emb_chunks"])
            pos_lists.append(w["pos"])
            id_lists.append(w["id"])
            mod_lists.append(w["mod"])
            role_lists.append(w["role"])
            name_lists.append(w["signal_name"])

            tgt_dicts.append(w["outputs_emb"])
            tgt_shapes_dicts.append(w["outputs_shapes"])
            outputs_names = w.get("outputs_names", {})
            tgt_names_dicts.append(outputs_names)

            all_target_ids.update(w["outputs_emb"].keys())

            for sid, sname in outputs_names.items():
                if sid not in id_to_output_name:
                    id_to_output_name[sid] = sname
                elif id_to_output_name[sid] != sname:
                    raise ValueError(
                        f"Inconsistent output name for signal_id={sid}: "
                        f"'{id_to_output_name[sid]}' vs '{sname}'"
                    )

        # --------------------------------------------------------------- #
        # 2. Determine batch lengths and pad to L_max
        # --------------------------------------------------------------- #
        lengths = [len(e) for e in emb_lists]
        L_max = max(lengths)

        pos_batch = np.zeros((B, L_max), dtype=np.int32)
        id_batch = np.zeros((B, L_max), dtype=np.int32)
        mod_batch = np.zeros((B, L_max), dtype=np.int16)
        role_batch = np.zeros((B, L_max), dtype=np.int8)

        padding_mask = np.zeros((B, L_max), dtype=np.int8)

        input_mask = np.ones((B, L_max), dtype=np.int8)
        actuator_mask = np.ones((B, L_max), dtype=np.int8)

        emb_batch: List[List[np.ndarray]] = [[None] * L_max for _ in range(B)]
        name_batch: List[List[str]] = [[""] * L_max for _ in range(B)]

        # --------------------------------------------------------------- #
        # 3. Insert tokens into padded batch
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
        # 4. Apply input dropout (per-token)
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
        # 5. Apply actuator dropout (per-token)
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
        # 6. Chunk dropout (per pos group)
        # --------------------------------------------------------------- #
        p_drop_inputs_chunks = self.cfg.get("p_drop_inputs_chunks", 0.0)
        p_drop_act_chunks = self.cfg.get("p_drop_actuators_chunks", 0.0)

        for i in range(B):
            Li = lengths[i]
            unique_pos = set(pos_batch[i, :Li])

            for pval in unique_pos:
                idx = np.where(pos_batch[i, :Li] == pval)[0]
                if len(idx) == 0:
                    continue

                if any(role_batch[i, t] == ROLE_CONTEXT for t in idx):
                    if random.random() < p_drop_inputs_chunks:
                        for t in idx:
                            if role_batch[i, t] == ROLE_CONTEXT:
                                input_mask[i, t] = 0
                                emb_batch[i][t] = np.zeros_like(emb_batch[i][t])

                if any(role_batch[i, t] == ROLE_ACTUATOR for t in idx):
                    if random.random() < p_drop_act_chunks:
                        for t in idx:
                            if role_batch[i, t] == ROLE_ACTUATOR:
                                actuator_mask[i, t] = 0
                                emb_batch[i][t] = np.zeros_like(emb_batch[i][t])

        # --------------------------------------------------------------- #
        # 7. Apply output dropout (per output name)
        # --------------------------------------------------------------- #
        p_drop_outputs = self.cfg.get("p_drop_outputs", 0.0)

        outputs_emb_batch: Dict[int, List[np.ndarray]] = {}
        outputs_mask_batch: Dict[int, np.ndarray] = {}

        for sig_id in sorted(all_target_ids):
            sig_embs = []
            sig_mask = np.ones((B,), dtype=np.int8)

            out_name = id_to_output_name.get(sig_id, str(sig_id))

            for i in range(B):
                emb = tgt_dicts[i].get(sig_id, None)

                if emb is None:
                    sig_mask[i] = 0
                    emb = np.zeros((1,), dtype=np.float32)

                sig_embs.append(emb)

            outputs_emb_batch[sig_id] = sig_embs

            for i in range(B):
                p = self.drop_outputs_overrides.get(out_name, p_drop_outputs)
                if random.random() < p:
                    sig_mask[i] = 0

            outputs_mask_batch[sig_id] = sig_mask

        # --------------------------------------------------------------- #
        # 8. Optionally collate native outputs (Y_native) for evaluation
        # --------------------------------------------------------------- #
        output_native_batch: Dict[int, np.ndarray] = {}
        if self.keep_output_native:
            # For each target signal, gather per-window native outputs.
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
                        # Allow shorthand "name: values"
                        val = out_info

                    if val is None:
                        # If the native value is missing (e.g. dropped by DropNa),
                        # create a zero array with the correct shape so that
                        # downstream code can use outputs_mask to ignore it.
                        shape = None
                        shapes_dict = tgt_shapes_dicts[i]
                        if shapes_dict and sig_id in shapes_dict:
                            shape = tuple(shapes_dict[sig_id])
                        else:
                            # Fall back to any other window that has this signal.
                            for sd in tgt_shapes_dicts:
                                if sig_id in sd:
                                    shape = tuple(sd[sig_id])
                                    break

                        if shape is None:
                            raise ValueError(
                                f"Missing shape for output signal_id={sig_id} "
                                f"(needed to build output_native_batch)"
                            )

                        arr = np.zeros(shape, dtype=np.float32)
                    else:
                        arr = np.asarray(val, dtype=np.float32)

                    per_sig_vals.append(arr)

                output_native_batch[sig_id] = np.stack(per_sig_vals, axis=0)

        batch_out = {
            "emb": emb_batch,
            "pos": pos_batch,
            "id": id_batch,
            "mod": mod_batch,
            "role": role_batch,
            "signal_name": name_batch,
            "padding_mask": padding_mask,
            "input_mask": input_mask,
            "actuator_mask": actuator_mask,
            "outputs_emb": outputs_emb_batch,
            "outputs_mask": outputs_mask_batch,
        }

        if self.keep_output_native:
            batch_out["output_native"] = output_native_batch

        return batch_out
