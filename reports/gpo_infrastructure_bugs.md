# GPO Training Infrastructure Bugs
## Bugs found and fixed during development — significant for reproducibility

> **Source:** `docs/gpo_implementation_log.md §§4–5` lines 229–328;
> commits `0bb0c4a`, `94ad59c`, `7365f4d`, `3230453`

These bugs are documented because they caused **systematic NRMSE degradation** in
early experiments (v2–v4) and their absence is required for correct results.
All are fixed in the current codebase.

---

## Bug A — Device mismatch (task 4-1)

**Symptom:** Runtime error during GPO training — boolean-mask indexing raised a CUDA
device error.

**Root cause:** `ContinuousGPOLoss` received `y_w_emb` / `y_l_emb` tensors on CPU
while `output_mask` was on CUDA.

**Fix:** Extended `move_batch_to_device` in `src/mmt/train/loop_utils.py` to also move
`batch["y_w_emb"]` and `batch["y_l_emb"]` (both `dict[int, Tensor]`) alongside the
standard batch fields.

**Commit:** `0bb0c4a`
*Source: `gpo_implementation_log.md §Bug A` lines 234–241*

---

## Bug B — Embedding shape mismatch at checkpoint load (tasks 4-2, 4-3)

**Symptom:** `load_best_weights` failed with a shape mismatch error when loading the
base finetune checkpoint into the GPO model.

**Root cause:** `resolve_finetune_embeddings` was re-tuning the output DCT3D embeddings
for the GPO run.  This produced a different `encoded_dim` from the source finetune run.

**Fix:** Added `gpo_inherit_all_embeddings = True` flag inside `_gpo_config_hook` in
`scripts_mast/run_gpo_finetune.py`.  The embedding resolution policy checks this flag
and routes all roles — including output — through the `source` branch, guaranteeing
the loaded checkpoint's embedding dimensions match.

**Commit:** `0bb0c4a`
*Source: `gpo_implementation_log.md §Bug B` lines 242–252*

---

## Bug C — Double cache materialisation (tasks 4-2, 4-3)

**Symptom:** Dataset was iterated twice during cache build — training was approximately
2× slower than expected.

**Root cause:** `build_window_data` was called with both `"train"` and `"val"` keys
pointing to the same MAST dataset, causing both splits to be iterated during materialisation.

**Fix:** Pass only `{"train": mast_context}` to `build_window_data`; build the val
loader from the already-cached dataset.

**Commit:** `0bb0c4a`
*Source: `gpo_implementation_log.md §Bug C` lines 253–261*

---

## Bug D — Wrong stages list after config merge (all tasks)

**Symptom:** GPO training ran 20 extra epochs before starting the GPO stage itself.

**Root cause:** `deep_merge` appended the `"gpo"` stage to the existing finetune
stages list `["ft_heads", "ft_full"]`, producing `["ft_heads", "ft_full", "gpo"]`.
The model ran two full finetune stages before entering GPO training.

**Fix:** Force-replace `train.stages` with just `["gpo"]` after merging `gpo.yaml`
instead of letting `deep_merge` append.

**Commit:** `0bb0c4a`
*Source: `gpo_implementation_log.md §Bug D` lines 262–275*

---

## Bug E — Reference model not on GPU (tasks 4-1, 4-3)

**Symptom:** Training crashed with a device error on the first batch.

**Root cause:** The frozen reference model copy was not moved to the GPU before the
first forward pass.

**Fix (commit `94ad59c`):** Move reference model to device in `_GpoBatchInjector.__init__`.

**Second GPU bug (commit `7365f4d`):** The batch itself was also not moved to device
before the reference model forward pass.  Fixed by calling `move_batch_to_device`
before the reference forward.

*Source: `gpo_implementation_log.md §Bug E` lines 276–283; §4 header lines 229–233*

---

## Bug F — Val-loss dilution (all tasks, discovered in v4)

**Symptom:** Training appeared to converge in <5 epochs with very low val loss; early
stopping fired prematurely.  Actual NRMSE showed no improvement or degradation.

**Root cause:** Val loss was averaged over **all** batches, including pair-absent
batches that return loss=0 from `ContinuousGPOLoss`.  With 46–85% pair-absent batches,
the apparent val loss was 17–37% lower than the true GPO val loss:

```
apparent_val  =  true_val × pair_hit_rate
             ≈  true_val × (0.15 to 0.54)
```

The diluted val loss formed an artificially low plateau from epoch 1, triggering early
stopping before the model had a chance to learn.

**Fix (commit `3230453`):** Val loss in `run_one_epoch` is now averaged only over batches
that contain ≥1 GPO pair (`gpo_pair_mask` is non-empty).

**Quantitative impact:**
- True val loss was `17–37%` higher than the pre-fix apparent val loss
- Early stopping was firing at epochs 1–3; after fix, training proceeds to epoch 10–200+

*Source: `gpo_implementation_log.md §5` lines 284–328*
