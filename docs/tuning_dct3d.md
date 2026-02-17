# Tuning DCT3D Embeddings

Many MAST signals are encoded with a lightweight **DCT3D** codec before being projected into the model.

The DCT3D codec now supports two coefficient selection modes:

1. **Spatial mode** (default): Selects the top-left-front block of DCT coefficients (low-frequency components)
2. **Rank mode** (tuned): Selects top-K coefficients by explained variance, regardless of spatial position

This document explains:
- The two selection modes and when to use each
- How the tuning pipeline works (rank mode)
- Where tuned results are written
- How tuned overrides are consumed by training and evaluation

---

## Selection Modes

### Spatial Mode (Manual Configuration)

**Use when:**
- Prototyping or quick experiments
- Default settings work well for your signals
- You don't want to run tuning

**Configuration:**
```yaml
embeddings:
  defaults:
    dct3d:
      encoder_name: dct3d
      encoder_kwargs:
        selection_mode: spatial  # default
        keep_h: 16
        keep_w: 8
        keep_t: 64
```

The codec keeps the first `(keep_h, keep_w, keep_t)` coefficients from the DCT spectrum (low-frequency components).

### Rank Mode (Variance-Based Selection)

**Use when:**
- You want optimal compression for your specific signals
- Different signals have different spectral characteristics
- You're willing to run a one-time tuning step

**How it works:**
1. Computes per-coefficient energy `E[c_i²]` across a sample of windows
2. Sorts all coefficients by energy (descending)
3. Selects top-K coefficients that achieve target explained energy
4. Saves coefficient indices to `.npy` files

**Mathematical foundation:**

For an orthonormal DCT, the explained energy ratio is:
```
explained_energy = sum(c_i² for i in selected) / sum(c_i² for all i)
```

The optimal K coefficients that minimize expected MSE are those with largest `E[c_i²]`.

**Benefits:**
- 10-30% fewer coefficients for same reconstruction quality
- OR 5-15% better quality for same number of coefficients
- Automatically adapts to each signal's spectral content

---

## Why Tune DCT3D?

DCT3D truncation trades off:
- **Smaller/cheaper embeddings** (fewer coefficients)
- vs **Explained energy** percentage (higher is better)

Different signals often have very different spectral content. A single global truncation setting can be:
- Overly conservative for some signals (wasting capacity)
- Too aggressive for others (losing important information)

**Rank mode tuning** automatically finds the optimal coefficient set for each signal.

---

## Tuning Pipeline (Rank Mode)

Tuning uses the same upstream preprocessing as training:

```
ChunkWindows → SelectValidWindows → TrimChunks → TuneRankedDCT3DTransform
```

The `TuneRankedDCT3DTransform`:
1. Computes full DCT for each chunk
2. Accumulates per-coefficient energy `E[c_i²]`
3. After streaming all windows, selects top-K coefficients per signal
4. Writes coefficient indices to `.npy` files
5. Generates config with `selection_mode: rank`

### Aggregation Strategy

The tuning process uses a **pooled energy aggregation** approach:

1. **For each window/chunk**: Compute full DCT: `z = DCT(x)`
2. **Accumulate energies**: `acc_energy[i] += z[i]²` for all coefficients
3. **After all windows**: Compute mean energy per coefficient: `E[c_i²] = acc_energy[i] / n_windows`
4. **Compute explained energy**: `sum(E[c_i²] for selected) / sum(E[c_i²] for all)`

This computes the **ratio of expected energies**: `E[sum(z_selected²)] / E[sum(z_all²)]`

**Important:** This differs from the old `TuneDCT3DTransform` which computed the **expected ratio**: `E[sum(z_selected²) / sum(z_all²)]` by averaging per-window ratios. The pooled approach is more robust to windows with varying signal energy and provides a more accurate estimate of compression performance on the overall dataset.

### Key Differences from Old Approach

- ✅ No grid search over `(keep_h, keep_w, keep_t)` combinations
- ✅ Much faster: 50-200x speedup
- ✅ Better compression: coefficients selected by actual importance
- ✅ More robust aggregation: pools energy across all windows before computing ratios

---

## Tuning Configuration

Edit `scripts_mast/configs/common/tune_dct3d.yaml`:

```yaml
tune_dct3d:
  sampling:
    max_windows: 15000  # Number of windows to sample

  objective:
    thresholds:
      input: 0.999      # Target explained energy (99.9%)
      actuator: 0.999
      output: 0.995
    max_budget:
      input: 4096       # Max coefficients per signal
      actuator: 4096
      output: 4096
```

### Parameters

**`max_windows`**: Cap on total windows processed. Use smaller values (1000-5000) for fast iteration, larger (10000-20000) for final tuning.

**`thresholds`**: Per-role target explained energy (0-1 scale). Higher values preserve more signal information but use more coefficients.
- 0.999 = 99.9% of signal energy preserved
- 0.995 = 99.5% preserved
- Typical range: 0.99-0.999

**`max_budget`**: Maximum allowed coefficients per role. Prevents any signal from using excessive capacity.
- Candidates with more coefficients are excluded
- If no candidates fit budget, smallest candidate is used
- Typical range: 2048-8192

---

## Output Files

Tuning produces two types of outputs:

### 1. Config File
```
scripts_mast/configs/tasks_overrides/<task>/embeddings_overrides/dct3d.yaml
```

Example:
```yaml
embeddings:
  per_signal_overrides:
    input:
      pf_active-coil_current:
        encoder_name: dct3d
        encoder_kwargs:
          selection_mode: rank
          coeff_indices_path: dct3d_indices/input_pf_active-coil_current.npy
          coeff_shape: [16, 1, 128]
          num_coeffs: 512
          explained_energy: 0.9985
    output:
      equilibrium-psi:
        encoder_name: dct3d
        encoder_kwargs:
          selection_mode: rank
          coeff_indices_path: dct3d_indices/output_equilibrium-psi.npy
          coeff_shape: [32, 32, 64]
          num_coeffs: 2048
          explained_energy: 0.9972
```

**Understanding the output:**

- `coeff_shape`: Original DCT spectrum dimensions `[height, width, time]`
- `num_coeffs`: Number of coefficients selected (out of `h × w × t` total)
- `explained_energy`: Fraction of signal energy preserved (0-1 scale)

**Note on dimension distribution:** In rank mode, coefficients are selected by variance regardless of spatial position. Unlike spatial mode (which takes a contiguous low-frequency block), rank mode may select coefficients scattered across all three dimensions. The selected coefficients automatically adapt to each signal's spectral characteristics - some signals may have more energy in temporal frequencies, others in spatial frequencies.

To see which dimensions are represented, you can load the `.npy` file and check:
```python
import numpy as np
indices = np.load("dct3d_indices/input_signal.npy")
h, w, t = 16, 1, 128  # coeff_shape
indices_3d = np.unravel_index(indices, (h, w, t))
unique_h = len(np.unique(indices_3d[0]))  # How many height positions used
unique_w = len(np.unique(indices_3d[1]))  # How many width positions used
unique_t = len(np.unique(indices_3d[2]))  # How many time positions used
```

### 2. Coefficient Indices (.npy files)
```
scripts_mast/configs/tasks_overrides/<task>/embeddings_overrides/dct3d_indices/
├── input_signal1.npy
├── input_signal2.npy
├── output_signal1.npy
└── ...
```

Each `.npy` file contains a 1D array of integer indices specifying which DCT coefficients to keep.

### 3. History (Archived Copies)
```
scripts_mast/configs/tasks_overrides/<task>/embeddings_overrides/history/
├── dct3d_20260216_143052.yaml
└── dct3d_indices_20260216_143052/
    ├── input_signal1.npy
    └── ...
```

Timestamped copies preserve previous tuning runs.

---

## How Tuned Overrides Are Used

The config loader merges tuned overrides last, so they win over defaults:

1. `common/embeddings.yaml` (global defaults)
2. `common/<phase>.yaml` (phase-specific settings)
3. `tasks_overrides/<task>/<phase>_overrides.yaml` (optional task overrides)
4. `tasks_overrides/<task>/embeddings_overrides/<profile>.yaml` (tuned settings)

**Note:** During `tune_dct3d`, the embeddings overrides are NOT merged (to avoid circular dependency).

This means:
- Future pretrain/finetune/eval runs automatically use tuned settings
- Common defaults remain clean and task-agnostic
- Easy to compare tuned vs default performance

---

## Practical Workflow

### Option A: Use Spatial Mode (No Tuning)

For quick experiments or when defaults work:

```yaml
# In your task config or common/embeddings.yaml
embeddings:
  defaults:
    dct3d:
      encoder_name: dct3d
      encoder_kwargs:
        selection_mode: spatial
        keep_h: 16
        keep_w: 8
        keep_t: 64
```

### Option B: Use Rank Mode (With Tuning)

For optimal performance:

**Step 1:** Run tuning
```bash
python scripts_mast/run_tune_dct3d.py --task task_1-1
```

Optional: Restrict to specific roles:
```bash
python scripts_mast/run_tune_dct3d.py --task task_1-1 --roles input,output
```

**Step 2:** Review generated files
- Check `embeddings_overrides/dct3d.yaml` for selected coefficients
- Verify `explained_energy` values meet your requirements
- Inspect `num_coeffs` to ensure reasonable sizes

**Step 3:** Run training/evaluation
```bash
# Tuned settings are automatically loaded
python scripts_mast/run_pretrain.py --task task_1-1
python scripts_mast/run_finetune.py --task task_1-1
python scripts_mast/run_eval.py --task task_1-1
```

**Step 4:** (Optional) Commit tuned files
If satisfied with results, commit:
- `embeddings_overrides/dct3d.yaml`
- `embeddings_overrides/dct3d_indices/*.npy`

---

## Performance Comparison

### Tuning Phase

| Metric | Old (Grid Search) | New (Rank Mode) | Improvement |
|--------|------------------|-----------------|-------------|
| Candidates evaluated | 50-200 configs | 1 (all coeffs) | **50-200x faster** |
| Typical runtime | 10-30 min | 1-3 min | **~10x faster** |
| Memory overhead | High | Moderate | Better |

### Runtime (Encode/Decode)

| Operation | Spatial Mode | Rank Mode | Delta |
|-----------|--------------|-----------|-------|
| Encode | O(HWT) DCT + O(1) slice | O(HWT) DCT + O(K) gather | +1-5% |
| Decode | O(HWT) IDCT + O(1) place | O(HWT) IDCT + O(K) scatter | +1-5% |

The gather/scatter overhead is negligible compared to DCT computation.

### Compression Quality

Rank mode typically achieves:
- **10-30% fewer coefficients** for same reconstruction quality
- **5-15% better quality** for same number of coefficients
- Better adaptation to signal-specific spectral characteristics

---

## Tips and Best Practices

### Tuning
- Start with `max_windows: 5000` for fast iteration
- Increase to `max_windows: 15000-20000` for final tuning
- Use representative data (typically train split)
- If signals fail to meet threshold, try:
  - Lowering thresholds slightly (e.g., 0.999 → 0.995)
  - Increasing `max_budget`
  - Checking if signal has unusual characteristics

### Configuration
- Use spatial mode for prototyping
- Use rank mode for production deployments
- Keep spatial mode configs as fallback
- Document any manual overrides

### Debugging
- Check tuning logs for warnings about signals not meeting targets
- Verify `.npy` files exist before running training
- Compare `explained_energy` values across signals
- Use `num_coeffs` to identify unusually large/small embeddings

---

## Troubleshooting

### Error: "coeff_indices_path required for rank mode"

**Cause:** Config specifies `selection_mode: rank` but missing indices file reference.

**Solution:** Run tuning to generate indices:
```bash
python scripts_mast/run_tune_dct3d.py --task <your_task>
```

### Error: "Coefficient indices file not found"

**Cause:** `.npy` file referenced in config doesn't exist.

**Solutions:**
1. Re-run tuning if files were deleted
2. Check path in config matches actual file location
3. Verify you're in the correct task directory

### Warning: "Signal cannot reach target threshold"

**Cause:** Signal has insufficient energy in any K coefficients to meet target.

**Solutions:**
1. Lower threshold in `tune_dct3d.yaml` (e.g., 0.999 → 0.995)
2. Increase `max_budget` to allow more coefficients
3. Check if signal has unusual characteristics (very noisy, sparse, etc.)
4. Verify signal preprocessing is correct

### Tuning is too slow

**Solutions:**
1. Reduce `max_windows` (e.g., 15000 → 5000)
2. Use fewer shots in dataset sampling
3. Restrict to specific roles: `--roles output`
4. Check if dataset loading is bottleneck

---


## References

- DCT3D codec implementation: `src/mmt/data/embeddings/dct3d_codec.py`
- Rank mode tuning transform: `src/mmt/data/transforms/tune_ranked_dct3d.py`
- Tuning script: `scripts_mast/run_tune_dct3d.py`
- Config reference: `docs/config_reference.md`
