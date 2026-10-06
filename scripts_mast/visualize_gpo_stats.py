"""
GPO Preference-Pair Dataset Statistics Visualizer
==================================================

Loads one or more GPO pairs directories (schema v1 through v3) and produces up
to three matplotlib figures per directory:

Figure 1 — Standard overview (12 panels)
-----------------------------------------
1.  **Window counts per signal** — bar chart of total windows stored, split
    by train / val based on the stored val_fraction.

2.  **Embedding dimension per signal** — confirms all shards share the same
    D; flags mismatches in red.

3.  **MSE preference gap per signal** — box-plot of ``‖y_w−y_l‖₂² / D``
    (the raw GPO loss input at collection time).  Summarises how far off the
    model was when the pairs were collected.

4.  **Preference margin distribution** — histogram of
    ``‖y_w_emb − y_l_emb‖₂`` (L2 norm in embedding space) per signal.
    Pairs with margin ≈ 0 are near-zero preference; very large margins may
    indicate outliers or data errors.

5.  **Cosine similarity y_w · y_l** — distribution of cosine similarity
    between preferred and dispreferred embeddings.  Values near 1.0 indicate
    similar predictions (easy pairs); values near −1 indicate extreme
    divergence.

6.  **Normalised margin distribution** — histogram of
    ``‖y_w−y_l‖₂ / (‖y_w‖₂ + ε)``; scale-invariant and comparable across
    tasks / signals.

7.  **Per-coefficient error profile** — mean absolute per-dimension error
    ``E[|y_w[d] − y_l[d]|]`` for each embedding coefficient d.  Reveals
    which DCT / Fourier modes the model struggles with most.

8.  **Window index distribution** — histogram of window_index values across
    all stored pairs.  Verifies temporal coverage: are early, mid, and late
    plasma phases all represented?

9.  **Windows per shot histogram** — distribution of window counts across
    shot IDs, useful for detecting heavily over-represented shots.

10. **Shard size distribution** — number of windows per shard file.

11. **Zero-margin fraction per signal** — bar chart of near-zero margin
    pairs (``‖y_w−y_l‖₂ < 1e−6``).  High fractions signal NaN/mask issues
    or that the model already predicts ground truth perfectly.

12. **Collection metadata summary table** — human-readable provenance from
    metadata.json + collection_config.json.

Figure 2 — Extended diagnostic (8 panels)
------------------------------------------
13. **Preference-strength CDF** — cumulative distribution of ``‖y_w−y_l‖₂²/D``
    (the raw GPO loss margin at collection time, before β scaling).  Shows
    what fraction of pairs are "soft" (margin ≈ 0) vs "hard" (large margin);
    a heavy tail means a few pairs dominate training.

14. **Pair hardness tiers × temporal coverage** — pairs split into quartile
    tiers (Q1=easy … Q4=hard) by normalised margin.  Stacked-area histogram
    by window_index shows whether hard pairs concentrate in specific plasma
    phases (e.g. disruptions, ramp-down).

15. **Embedding energy bias (‖y_w‖ vs ‖y_l‖)** — 2-D hexbin scatter of
    ground-truth L2 norm vs model-prediction L2 norm.  The diagonal is the
    identity line; systematic off-diagonal shift indicates the model
    consistently over- or under-predicts signal amplitude in DCT space.

16. **Direction vs magnitude error decomposition** — bar chart per signal:
    directional error = ``1 − cosine(y_w, y_l)`` (shape / angle mismatch)
    vs magnitude error = ``|‖y_l‖ − ‖y_w‖| / (‖y_w‖ + ε)`` (amplitude
    mismatch).  Identifies whether the model's primary failure mode is
    orientation or scale.

17. **Per-shot mean MSE gap** — histogram of shot-averaged MSE gaps.
    Detects pathological shots that dominate the preference loss.  The top-10
    worst shots by mean MSE are annotated.

18. **Coefficient error × time heatmap** — 2-D heatmap of
    ``|y_w[d] − y_l[d]|`` averaged over windows bucketed into 20 temporal
    bins (x-axis) vs the first 64 coefficient indices (y-axis).  Reveals
    whether specific DCT modes degrade during specific plasma phases.

19. **Ground-truth norm drift over time** — per-signal mean ``‖y_w‖₂`` as
    a function of window_index (temporal position).  A rising/falling trend
    indicates the signal amplitude evolves through the discharge; the model
    energy curve (mean ``‖y_l‖₂``) is overlaid to show tracking accuracy.

20. **Embedding PCA diversity** — 2-D PCA projection of a random sample of
    ``y_w_emb`` vectors coloured by ``window_index``.  A tight cluster means
    low diversity (GPO risk: over-fitting a narrow manifold); spread indicates
    the dataset spans a wide range of plasma states.

Figure 3 — Actionability diagnostics (3 panels, schema v2/v3 only)
----------------------------------------------------------------
21. **β-Effectiveness Score Distribution** — simulates the actual sigmoid gate
    value ``σ(β × MSE_gap)`` that each pair receives during GPO training.
    Values near 0.5 are ideally informative; values ≈ 0 or ≈ 1 are saturated
    (zero gradient).  The configured β from ``gpo_tasks.yaml`` is shown as a
    reference line on the MSE-gap axis.  Directly answers *"is my β
    calibrated?"*.

22. **Pair Quality 2-D Scatter** — normalised margin (x) vs cosine similarity
    (y) coloured by MSE gap.  Quadrant interpretation: top-right = good
    actionable pairs (correct shape, wrong amplitude); bottom-right = hard
    pairs (wrong shape); top-left = near-perfect predictions (wasted pairs);
    bottom-left = noise / mask artefacts.

23. **Shot-Level Outlier Table** — ranked top-15 worst shots by mean MSE gap,
    with columns n_windows, mean/p50/p95 MSE gap, mean normalised margin, and
    mean cosine similarity.  Directly identifies pathological shots (disruptions,
    unusual discharges) that dominate the preference loss and are candidates for
    blacklisting.

Figure 4 — Training-signal quality (6 panels, schema v2/v3 only)
--------------------------------------------------------------
24. **Lorenz Curve of MSE Gap** — cumulative gradient concentration.
    The Gini coefficient measures pair inequality (0 = uniform, 1 = one pair).
    A highly bowed curve means a tiny fraction of hard pairs drives almost all
    gradient signal — GPO is at risk of overtraining on a few windows.

25. **Signed Coefficient Bias** — ``E[y_l[d] − y_w[d]]`` per DCT coefficient.
    Positive bars = model systematically *over*-predicts that mode; negative =
    under-predicts.  Unlike panel 7 (absolute error), this reveals directionality:
    a zero mean despite high absolute error means the bias cancels across windows.

26. **Fisher-Weighted Effective N** — ``Σ 4σ(β·g)(1−σ(β·g))`` per pair.
    Pairs with gate ≈ 0.5 (the ideal) each contribute weight 1; saturated
    (gate ≈ 1) or dead (gate ≈ 0) pairs contribute ≈ 0.  ``eff_N / total_N``
    is the fraction of the dataset that actually provides useful gradient.

27. **Between-Shot vs Within-Shot MSE Variance (ANOVA)** — stacked bar showing
    what fraction of MSE variance is explained by shot identity vs temporal
    position within a shot.  High between-shot → a few pathological shots
    dominate; blacklisting them would substantially flatten the distribution.

28. **Filter Survival Curve** — fraction of pairs surviving each
    ``min_margin_mse`` threshold (log-spaced sweep up to p99).  The dotted
    vertical line marks the threshold that cuts 95% of pairs (keeps 5%).
    Guides the choice of ``min_margin_mse`` in the GPO dataset config.

29. **β Optimisation Table** — one row per signal with: current β, p50 MSE,
    β×p50 operating point, σ(β×p50) gate value, and ``β_opt = 1/p50_MSE``
    (the value that places the median pair at the ideal σ = 0.731 gate).
    Colour-coded: green = within 2× of β_opt, amber = 2–5×, red = >5×.

Multi-directory comparison mode (--compare)
--------------------------------------------
When two or more directories are passed with ``--compare``, the script
produces a single overlay figure with panels 3, 4, 7, and 5 overlaid for
all directories — useful for comparing before-/after re-collection or
across tasks.

Usage
-----
    python scripts_mast/visualize_gpo_stats.py \\
        --gpo_dir runs/ft-task_4-1-scratch-mmt/gpo_pairs

    # Multiple directories — one figure each
    python scripts_mast/visualize_gpo_stats.py \\
        --gpo_dir runs/ft-task_4-1-scratch-mmt/gpo_pairs \\
                  runs/ft-task_4-3-scratch-mmt/gpo_pairs

    # Overlay comparison figure across directories
    python scripts_mast/visualize_gpo_stats.py \\
        --gpo_dir runs/ft-task_4-1-scratch-mmt/gpo_pairs \\
                  runs/ft-task_4-3-scratch-mmt/gpo_pairs \\
        --compare

    # Save figures to a directory instead of displaying interactively
    python scripts_mast/visualize_gpo_stats.py \\
        --gpo_dir runs/ft-task_4-1-scratch-mmt/gpo_pairs \\
        --save_dir /tmp/gpo_stats

    # Skip the extended diagnostic figure
    python scripts_mast/visualize_gpo_stats.py \\
        --gpo_dir runs/ft-task_4-1-scratch-mmt/gpo_pairs \\
        --no_extended

    # Pass the configured β for β-effectiveness and quality panels
    python scripts_mast/visualize_gpo_stats.py \\
        --gpo_dir runs/ft-task_4-1-scratch-mmt/gpo_pairs \\
        --beta 10.0

Optional flags
--------------
    --signal          Only analyse the named signal (default: all signals).
    --max_windows     Maximum windows to load per signal for histogram computation
                      (default: 50000).  Set 0 to load all.
    --compare         Produce an additional cross-directory overlay figure when
                      multiple --gpo_dir paths are given.
    --no_extended     Skip Figure 2 (extended diagnostics) — useful for quick runs.
    --no_actionability Skip Figure 3 (β-effectiveness, pair quality, shot table).
    --no_quality      Skip Figure 4 (training-signal quality: Lorenz, bias, eff-N,
                      ANOVA, survival curve, β table).
    --beta            Configured β value used in panels 21, 26, and 29.
                      Default: 0.1 if not provided.
    --no_show         Do not call plt.show() — use with --save_dir for headless runs.
    --dpi             Figure DPI (default: 120).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("gpo.stats")


# ======================================================================================================================
# Shard discovery helpers
# ======================================================================================================================


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _discover_signals(gpo_dir: Path) -> dict[str, list[Path]]:
    """Return {safe_signal_name: [shard_path, ...]} sorted by name."""
    shards: dict[str, list[Path]] = {}
    for p in sorted(gpo_dir.glob("*.npz")):
        stem = p.stem
        if "__shard_" in stem:
            sname = stem[: stem.rfind("__shard_")]
        else:
            sname = stem
        shards.setdefault(sname, []).append(p)
    for v in shards.values():
        v.sort()
    return dict(sorted(shards.items()))


def _load_arrays(
    shards: list[Path],
    schema_version: int,
    max_windows: int,
) -> dict[str, np.ndarray]:
    """
    Load arrays from a list of shard files into concatenated numpy arrays.

    Returns a dict with keys:
        shot_id, window_index, y_w (shape N×D), y_l (shape N×D),
        emb_dim (scalar, v2/v3), n_rows_per_shard
    """
    parts_shot: list[np.ndarray] = []
    parts_widx: list[np.ndarray] = []
    parts_yw: list[np.ndarray] = []
    parts_yl: list[np.ndarray] = []
    rows_per_shard: list[int] = []
    emb_dims: list[int] = []
    total = 0

    for shard in shards:
        with np.load(shard, allow_pickle=False) as npz:
            n = len(npz["shot_id"])
            rows_per_shard.append(n)
            if max_windows > 0 and total + n > max_windows:
                n = max(0, max_windows - total)
            if n == 0:
                break

            parts_shot.append(npz["shot_id"][:n])
            parts_widx.append(npz["window_index"][:n])

            if schema_version >= 2:
                parts_yw.append(npz["y_w_emb"][:n].astype(np.float32))
                parts_yl.append(npz["y_l_emb"][:n].astype(np.float32))
                if "emb_dim" in npz:
                    emb_dims.append(int(npz["emb_dim"]))
            else:
                # v1: native-space arrays — flatten to 1D per window for L2/cosine
                yw_raw = npz["y_w"][:n].astype(np.float32)
                yl_raw = npz["y_l"][:n].astype(np.float32)
                parts_yw.append(yw_raw.reshape(n, -1))
                parts_yl.append(yl_raw.reshape(n, -1))

            total += n
            if max_windows > 0 and total >= max_windows:
                break

    if not parts_shot:
        return {}

    return {
        "shot_id": np.concatenate(parts_shot),
        "window_index": np.concatenate(parts_widx),
        "y_w": np.concatenate(parts_yw),
        "y_l": np.concatenate(parts_yl),
        "n_rows_per_shard": np.array(rows_per_shard, dtype=np.int64),
        "emb_dim": int(emb_dims[0]) if emb_dims else None,
    }


# ======================================================================================================================
# Per-signal stats computation
# ======================================================================================================================


def _compute_signal_stats(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """
    Compute per-signal stats from loaded arrays.

    Returns
    -------
    dict with keys:
        n_windows, emb_dim,
        margins              — L2 norms ‖y_w − y_l‖₂  (N,)
        margin_mean/std/p5/p95,
        margins_normalised   — ‖y_w − y_l‖₂ / (‖y_w‖₂ + 1e-9)  (N,)
        cosines              — cosine(y_w, y_l)  (N,)
        cosine_mean/std,
        mse_gap              — ‖y_w − y_l‖₂² / D  (N,)  — direct GPO loss input
        mse_gap_mean/p50/p95,
        coeff_abs_err        — mean abs per-dim error  (D,)
        shot_ids_unique,     windows_per_shot,
        window_indices,
        n_rows_per_shard,    zero_margin_fraction
    """
    yw = arrays["y_w"]  # (N, D)
    yl = arrays["y_l"]  # (N, D)
    n, D = yw.shape

    diff = yw - yl

    # L2 margin
    margins = np.linalg.norm(diff, axis=1)  # (N,)

    # Cosine similarity
    yw_norm = np.linalg.norm(yw, axis=1, keepdims=True)
    yl_norm = np.linalg.norm(yl, axis=1, keepdims=True)
    denom = np.maximum(yw_norm * yl_norm, 1e-9)
    cosines = (yw * yl).sum(axis=1) / denom.squeeze()  # (N,)

    # Normalised margin: ‖y_w − y_l‖₂ / (‖y_w‖₂ + ε)
    margins_normalised = margins / (yw_norm.squeeze() + 1e-9)

    # MSE preference gap: ‖y_w − y_l‖₂² / D (same as d_w at collection time when ŷ=y_l)
    mse_gap = (diff**2).mean(axis=1)  # (N,)

    # Per-coefficient absolute error: E[|y_w[d] − y_l[d]|] shape (D,)
    coeff_abs_err = np.abs(diff).mean(axis=0)

    shot_ids = arrays["shot_id"]
    unique_shots, shot_counts = np.unique(shot_ids, return_counts=True)

    zero_margin_frac = float((margins < 1e-6).mean())

    return {
        "n_windows": n,
        "emb_dim": arrays.get("emb_dim"),
        # margin
        "margins": margins,
        "margin_mean": float(margins.mean()),
        "margin_std": float(margins.std()),
        "margin_p5": float(np.percentile(margins, 5)),
        "margin_p95": float(np.percentile(margins, 95)),
        # normalised margin
        "margins_normalised": margins_normalised,
        "norm_margin_mean": float(margins_normalised.mean()),
        "norm_margin_p95": float(np.percentile(margins_normalised, 95)),
        # cosine
        "cosines": cosines,
        "cosine_mean": float(cosines.mean()),
        "cosine_std": float(cosines.std()),
        # mse gap
        "mse_gap": mse_gap,
        "mse_gap_mean": float(mse_gap.mean()),
        "mse_gap_p50": float(np.percentile(mse_gap, 50)),
        "mse_gap_p95": float(np.percentile(mse_gap, 95)),
        # per-coefficient error
        "coeff_abs_err": coeff_abs_err,
        # shot / window coverage
        "shot_ids_unique": unique_shots,
        "windows_per_shot": shot_counts,
        "window_indices": arrays["window_index"],
        # shard metadata
        "n_rows_per_shard": arrays["n_rows_per_shard"],
        "zero_margin_fraction": zero_margin_frac,
    }


# ======================================================================================================================
# Figure creation — single directory (12 panels)
# ======================================================================================================================


def _make_figure(
    gpo_dir: Path,
    signal_filter: str | None,
    max_windows: int,
    dpi: int,
) -> Figure:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------
    meta: dict[str, Any] = {}
    cc: dict[str, Any] = {}
    meta_path = gpo_dir / "metadata.json"
    cc_path = gpo_dir / "collection_config.json"
    if meta_path.is_file():
        meta = _load_json(meta_path)
    if cc_path.is_file():
        cc = _load_json(cc_path)

    schema_version = int(meta.get("schema_version", cc.get("schema_version", 1)))
    val_fraction = float(cc.get("val_fraction", meta.get("val_fraction", 0.0)))
    task = cc.get("task", meta.get("task", "unknown"))
    run_id = cc.get("run_id", meta.get("run_id", "unknown"))

    log.info(
        "Dataset: %s | schema_version=%d | task=%s | run_id=%s | val_fraction=%.2f",
        gpo_dir,
        schema_version,
        task,
        run_id,
        val_fraction,
    )

    # ------------------------------------------------------------------
    # Discover + load signal data
    # ------------------------------------------------------------------
    all_signals = _discover_signals(gpo_dir)
    if not all_signals:
        raise FileNotFoundError(f"No .npz shards found in {gpo_dir}")

    if signal_filter is not None:
        safe_filter = signal_filter.replace("/", "-").replace("\\", "-")
        all_signals = {k: v for k, v in all_signals.items() if k == safe_filter}
        if not all_signals:
            raise FileNotFoundError(
                f"Signal {signal_filter!r} not found in {gpo_dir}. Available: {list(_discover_signals(gpo_dir).keys())}"
            )

    per_signal: dict[str, dict[str, Any]] = {}
    for sname, shards in all_signals.items():
        log.info("  Loading signal %r (%d shards)…", sname, len(shards))
        arrays = _load_arrays(shards, schema_version=schema_version, max_windows=max_windows)
        if not arrays:
            log.warning("  Signal %r: no data loaded — skipping.", sname)
            continue
        per_signal[sname] = _compute_signal_stats(arrays)
        log.info(
            "  Signal %r: %d windows | L2 margin μ=%.3f σ=%.3f | "
            "cosine μ=%.3f | MSE gap μ=%.4f | norm_margin μ=%.3f | zero_margin=%.1f%%",
            sname,
            per_signal[sname]["n_windows"],
            per_signal[sname]["margin_mean"],
            per_signal[sname]["margin_std"],
            per_signal[sname]["cosine_mean"],
            per_signal[sname]["mse_gap_mean"],
            per_signal[sname]["norm_margin_mean"],
            per_signal[sname]["zero_margin_fraction"] * 100,
        )

    if not per_signal:
        raise RuntimeError("No valid signals found — nothing to plot.")

    signal_names = list(per_signal.keys())
    n_signals = len(signal_names)

    # ------------------------------------------------------------------
    # Layout: 4 rows × 3 cols = 12 panels
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 24), dpi=dpi)
    fig.suptitle(
        f"GPO Pair Dataset Statistics\n"
        f"dir: {gpo_dir.name}   task: {task}   run: {run_id}   "
        f"schema: v{schema_version}   val_frac: {val_fraction:.0%}",
        fontsize=13,
        y=0.995,
    )

    gs = gridspec.GridSpec(
        4,
        3,
        figure=fig,
        hspace=0.48,
        wspace=0.38,
        top=0.96,
        bottom=0.04,
        left=0.06,
        right=0.97,
    )

    ax_counts = fig.add_subplot(gs[0, 0])
    ax_emb_dim = fig.add_subplot(gs[0, 1])
    ax_mse_gap = fig.add_subplot(gs[0, 2])
    ax_margin = fig.add_subplot(gs[1, 0])
    ax_cosine = fig.add_subplot(gs[1, 1])
    ax_norm_marg = fig.add_subplot(gs[1, 2])
    ax_coeff_err = fig.add_subplot(gs[2, 0])
    ax_win_idx = fig.add_subplot(gs[2, 1])
    ax_shot_hist = fig.add_subplot(gs[2, 2])
    ax_shard = fig.add_subplot(gs[3, 0])
    ax_zero_marg = fig.add_subplot(gs[3, 1])
    ax_meta = fig.add_subplot(gs[3, 2])

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_signals, 1)))

    # ---- Panel 1: Window counts (train + val) ----
    train_counts = [int(round(per_signal[s]["n_windows"] * (1 - val_fraction))) for s in signal_names]
    val_counts = [int(round(per_signal[s]["n_windows"] * val_fraction)) for s in signal_names]
    x = np.arange(n_signals)
    w = 0.4
    bars_t = ax_counts.bar(x - w / 2, train_counts, width=w, label="train", color="steelblue")
    bars_v = ax_counts.bar(x + w / 2, val_counts, width=w, label="val", color="coral")
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(signal_names, rotation=25, ha="right", fontsize=8)
    ax_counts.set_ylabel("Windows")
    ax_counts.set_title("Window Counts per Signal (train / val split)")
    ax_counts.legend(fontsize=8)
    for bar in bars_t:
        h = bar.get_height()
        if h > 0:
            ax_counts.text(bar.get_x() + bar.get_width() / 2, h + 1, str(h), ha="center", fontsize=6)
    for bar in bars_v:
        h = bar.get_height()
        if h > 0:
            ax_counts.text(bar.get_x() + bar.get_width() / 2, h + 1, str(h), ha="center", fontsize=6)

    # ---- Panel 2: Embedding dimension ----
    emb_dims = [per_signal[s].get("emb_dim") for s in signal_names]
    valid_dims = [d for d in emb_dims if d is not None]
    if valid_dims:
        all_same = len(set(valid_dims)) == 1
        bar_colors = [
            "steelblue"
            if (d is not None and (all_same or d == max(set(valid_dims), key=valid_dims.count)))
            else "tomato"
            for d in emb_dims
        ]
        ax_emb_dim.bar(signal_names, [d or 0 for d in emb_dims], color=bar_colors)
        ax_emb_dim.set_xticks(range(n_signals))
        ax_emb_dim.set_xticklabels(signal_names, rotation=25, ha="right", fontsize=8)
        ax_emb_dim.set_ylabel("Embedding dim D")
        ax_emb_dim.set_title("Embedding Dimension per Signal" + ("  ✓ consistent" if all_same else "  ✗ MISMATCH"))
        for i, d in enumerate(emb_dims):
            if d:
                ax_emb_dim.text(i, d + 0.5, str(d), ha="center", fontsize=7)
    else:
        ax_emb_dim.text(
            0.5,
            0.5,
            "emb_dim not available\n(schema v1 data)",
            ha="center",
            va="center",
            transform=ax_emb_dim.transAxes,
            fontsize=10,
        )
        ax_emb_dim.set_title("Embedding Dimension (N/A for v1)")

    # ---- Panel 3: MSE preference gap ‖y_w−y_l‖²/D (GPO loss input at collection) ----
    # Box-plot with per-signal quartiles; annotate mean.
    bp_data = [per_signal[s]["mse_gap"] for s in signal_names]
    bp = ax_mse_gap.boxplot(
        bp_data,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="crimson", linewidth=1.5),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor((*c[:3], 0.5))
    ax_mse_gap.set_xticks(range(1, n_signals + 1))
    ax_mse_gap.set_xticklabels(signal_names, rotation=25, ha="right", fontsize=8)
    ax_mse_gap.set_ylabel("MSE = ‖y_w−y_l‖² / D")
    ax_mse_gap.set_title("MSE Preference Gap per Signal\n(d_w at collection — direct GPO loss input)")
    for i, sname in enumerate(signal_names):
        ax_mse_gap.text(
            i + 1,
            per_signal[sname]["mse_gap_mean"],
            f"μ={per_signal[sname]['mse_gap_mean']:.3f}",
            ha="center",
            va="bottom",
            fontsize=6,
            color="navy",
        )

    # ---- Panel 4: Preference margin distribution ----
    for i, sname in enumerate(signal_names):
        stats = per_signal[sname]
        ax_margin.hist(
            stats["margins"],
            bins=60,
            density=True,
            histtype="stepfilled",
            alpha=0.45,
            color=colors[i],
            label=f"{sname}\nμ={stats['margin_mean']:.2f} σ={stats['margin_std']:.2f}",
        )
    ax_margin.set_xlabel("‖y_w_emb − y_l_emb‖₂  (preference margin)")
    ax_margin.set_ylabel("Density")
    ax_margin.set_title("Preference Margin Distribution  (L2 in emb space)")
    ax_margin.legend(fontsize=6)

    # ---- Panel 5: Cosine similarity y_w · y_l ----
    for i, sname in enumerate(signal_names):
        stats = per_signal[sname]
        ax_cosine.hist(
            stats["cosines"],
            bins=60,
            density=True,
            histtype="stepfilled",
            alpha=0.45,
            color=colors[i],
            label=f"{sname} (μ={stats['cosine_mean']:.3f})",
        )
    ax_cosine.axvline(1.0, color="grey", linewidth=0.8, linestyle="--", label="sim=1 (identical)")
    ax_cosine.set_xlabel("cosine(y_w_emb, y_l_emb)")
    ax_cosine.set_ylabel("Density")
    ax_cosine.set_title("y_w / y_l Cosine Similarity")
    ax_cosine.legend(fontsize=7)

    # ---- Panel 6: Normalised margin distribution ----
    for i, sname in enumerate(signal_names):
        stats = per_signal[sname]
        ax_norm_marg.hist(
            stats["margins_normalised"],
            bins=60,
            density=True,
            histtype="stepfilled",
            alpha=0.45,
            color=colors[i],
            label=f"{sname} (μ={stats['norm_margin_mean']:.3f})",
        )
    ax_norm_marg.set_xlabel("‖y_w − y_l‖₂ / ‖y_w‖₂  (relative error)")
    ax_norm_marg.set_ylabel("Density")
    ax_norm_marg.set_title("Normalised Margin Distribution\n(scale-invariant, comparable across tasks)")
    ax_norm_marg.axvline(1.0, color="grey", linewidth=0.8, linestyle="--", label="relative err = 1")
    ax_norm_marg.legend(fontsize=7)

    # ---- Panel 7: Per-coefficient error profile ----
    # Show the first signal only if there are multiple (or merge them by averaging).
    # If all signals have the same D, we average across signals.
    coeff_errs = [per_signal[s]["coeff_abs_err"] for s in signal_names]
    valid_errs = [e for e in coeff_errs if e is not None and len(e) > 0]
    if valid_errs:
        common_D = valid_errs[0].shape[0]
        if all(e.shape[0] == common_D for e in valid_errs):
            mean_coeff_err = np.stack(valid_errs).mean(axis=0)
            label_txt = "avg over signals" if n_signals > 1 else signal_names[0]
        else:
            mean_coeff_err = valid_errs[0]
            label_txt = signal_names[0] + " (D mismatch — first signal only)"
        dims = np.arange(len(mean_coeff_err))
        ax_coeff_err.bar(dims, mean_coeff_err, width=1.0, color="steelblue", alpha=0.8)
        ax_coeff_err.set_xlabel("Embedding coefficient index d")
        ax_coeff_err.set_ylabel("E[|y_w[d] − y_l[d]|]")
        ax_coeff_err.set_title(
            f"Per-Coefficient Abs Error Profile  ({label_txt})\n"
            "Reveals which DCT/Fourier modes the model mispredicts most"
        )
        # Annotate top-5 worst coefficients
        top5_idx = np.argsort(mean_coeff_err)[::-1][:5]
        for idx in top5_idx:
            ax_coeff_err.annotate(
                str(idx),
                xy=(idx, mean_coeff_err[idx]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=6,
                color="tomato",
            )
    else:
        ax_coeff_err.text(0.5, 0.5, "No coefficient data", ha="center", va="center", transform=ax_coeff_err.transAxes)
        ax_coeff_err.set_title("Per-Coefficient Error Profile")

    # ---- Panel 8: Window index distribution (temporal coverage) ----
    all_win_idx: list[np.ndarray] = []
    for sname in signal_names:
        all_win_idx.append(per_signal[sname]["window_indices"])
    if all_win_idx:
        combined_win_idx = np.concatenate(all_win_idx)
        ax_win_idx.hist(combined_win_idx, bins=50, color="mediumseagreen", edgecolor="white", linewidth=0.4)
        ax_win_idx.set_xlabel("window_index  (chronological position within shot)")
        ax_win_idx.set_ylabel("Count (windows)")
        ax_win_idx.set_title(
            "Window Index Distribution (all signals)\nFlat = uniform temporal coverage; peaked = concentrated phase"
        )
        ax_win_idx.axvline(
            float(np.median(combined_win_idx)),
            color="tomato",
            linewidth=1.2,
            linestyle="--",
            label=f"median={np.median(combined_win_idx):.0f}",
        )
        ax_win_idx.legend(fontsize=8)

    # ---- Panel 9: Windows per shot histogram ----
    all_wps: list[int] = []
    for sname in signal_names:
        all_wps.extend(per_signal[sname]["windows_per_shot"].tolist())
    if all_wps:
        ax_shot_hist.hist(all_wps, bins=40, color="steelblue", edgecolor="white", linewidth=0.4)
        ax_shot_hist.set_xlabel("Windows per shot_id")
        ax_shot_hist.set_ylabel("Count (shots)")
        ax_shot_hist.set_title("Windows-per-Shot Distribution (all signals)")
        ax_shot_hist.axvline(
            float(np.median(all_wps)),
            color="tomato",
            linewidth=1.2,
            linestyle="--",
            label=f"median={np.median(all_wps):.0f}",
        )
        ax_shot_hist.legend(fontsize=8)

    # ---- Panel 10: Shard size distribution ----
    all_shard_sizes: list[int] = []
    for sname in signal_names:
        all_shard_sizes.extend(per_signal[sname]["n_rows_per_shard"].tolist())
    if all_shard_sizes:
        ax_shard.hist(all_shard_sizes, bins=30, color="mediumpurple", edgecolor="white", linewidth=0.4)
        ax_shard.set_xlabel("Windows per shard")
        ax_shard.set_ylabel("Count (shards)")
        ax_shard.set_title("Shard Size Distribution")
        ax_shard.axvline(
            float(np.median(all_shard_sizes)),
            color="tomato",
            linewidth=1.2,
            linestyle="--",
            label=f"median={np.median(all_shard_sizes):.0f}",
        )
        ax_shard.legend(fontsize=8)

    # ---- Panel 11: Zero-margin fraction (near-identical pairs) ----
    zero_fracs = [per_signal[s]["zero_margin_fraction"] * 100 for s in signal_names]
    bar_colors_z = ["tomato" if z > 5 else "steelblue" for z in zero_fracs]
    ax_zero_marg.bar(signal_names, zero_fracs, color=bar_colors_z)
    ax_zero_marg.set_xticks(range(n_signals))
    ax_zero_marg.set_xticklabels(signal_names, rotation=25, ha="right", fontsize=8)
    ax_zero_marg.set_ylabel("Zero-margin pairs (%)")
    ax_zero_marg.set_title("Near-Zero Margin Pairs\n(‖y_w−y_l‖₂ < 1e−6 — NaN/mask artefacts or perfect prediction)")
    ax_zero_marg.axhline(5, color="tomato", linewidth=0.8, linestyle="--", label=">5% threshold")
    ax_zero_marg.legend(fontsize=8)
    for i, z in enumerate(zero_fracs):
        ax_zero_marg.text(i, z + 0.1, f"{z:.1f}%", ha="center", fontsize=7)

    # ---- Panel 12: Metadata summary table ----
    ax_meta.axis("off")
    total_all = sum(per_signal[s]["n_windows"] for s in signal_names)
    n_shots_all = sum(len(per_signal[s]["shot_ids_unique"]) for s in signal_names)
    mse_gap_all = float(np.mean([per_signal[s]["mse_gap_mean"] for s in signal_names]))
    margin_all = float(np.mean([per_signal[s]["margin_mean"] for s in signal_names]))
    cos_all = float(np.mean([per_signal[s]["cosine_mean"] for s in signal_names]))
    nm_all = float(np.mean([per_signal[s]["norm_margin_mean"] for s in signal_names]))

    rows = [
        ["Field", "Value"],
        ["task", task],
        ["run_id", run_id],
        ["schema_version", str(schema_version)],
        ["split", cc.get("split", meta.get("split", "–"))],
        ["val_fraction", f"{val_fraction:.1%}"],
        ["pair_sources", ", ".join(cc.get("pair_sources", ["–"]))],
        ["multi_signal", cc.get("multi_signal", meta.get("multi_signal", "–"))],
        ["signals", str(n_signals)],
        ["total windows", str(total_all) + (f"  (≤{max_windows} loaded)" if max_windows > 0 else "")],
        ["unique shots", str(n_shots_all)],
        ["avg L2 margin μ", f"{margin_all:.4f}"],
        ["avg cosine μ", f"{cos_all:.4f}"],
        ["avg MSE gap μ", f"{mse_gap_all:.4f}"],
        ["avg norm margin μ", f"{nm_all:.4f}"],
        ["dir", gpo_dir.name],
    ]
    col_widths = [0.42, 0.58]
    y0, dy = 0.97, 0.060
    for r, row in enumerate(rows):
        is_header = r == 0
        for c, cell in enumerate(row):
            x_pos = sum(col_widths[:c])
            style = dict(fontsize=7.5, va="top", ha="left", transform=ax_meta.transAxes)
            if is_header:
                style["fontweight"] = "bold"
                style["color"] = "white"
                ax_meta.axhspan(
                    y0 - dy * 0.85,
                    y0 + dy * 0.05,
                    xmin=0,
                    xmax=1,
                    color="steelblue",
                    transform=ax_meta.transAxes,
                    clip_on=True,
                )
            elif r % 2 == 0:
                ax_meta.axhspan(
                    y0 - dy * (r + 0.85),
                    y0 - dy * (r - 0.05),
                    xmin=0,
                    xmax=1,
                    color="#f0f4f8",
                    transform=ax_meta.transAxes,
                    clip_on=True,
                )
            ax_meta.text(x_pos + 0.01, y0 - dy * r, cell, **style)
    ax_meta.set_title("Collection Metadata + Summary Stats", pad=6)

    return fig


# ======================================================================================================================
# Extended stats computation (requires schema v2/v3 arrays)
# ======================================================================================================================


def _compute_extended_stats(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """
    Compute additional per-signal statistics for the extended diagnostic figure.

    Requires embedding-space schema-v2/v3 arrays (keys: y_w, y_l, shot_id,
    window_index).

    Returns
    -------
    dict with keys:
        mse_gap             — ‖y_w − y_l‖²/D per window  (N,)
        norm_margin         — ‖y_w − y_l‖₂ / ‖y_w‖₂     (N,)
        yw_norm             — ‖y_w‖₂ per window           (N,)
        yl_norm             — ‖y_l‖₂ per window           (N,)
        dir_err_mean        — mean (1 − cosine)            float
        mag_err_mean        — mean |‖y_l‖ − ‖y_w‖| / (‖y_w‖ + ε)  float
        dir_err             — (1 − cosine) per window     (N,)
        mag_err             — mag error per window         (N,)
        shot_mse_gap        — per unique shot mean MSE gap dict {shot_id: float}
        coeff_err_by_winbin — (n_bins, min(D,64)) mean |Δ| per temporal bin
        n_win_bins          — number of temporal bins used
        window_indices      — raw window_index array       (N,)
        shot_id             — raw shot_id array            (N,)
        y_w_sample          — up to 4096 rows of y_w_emb for PCA
    """
    yw = arrays["y_w"]  # (N, D)
    yl = arrays["y_l"]  # (N, D)
    n, D = yw.shape
    win_idx = arrays["window_index"]  # (N,)
    shot_ids = arrays["shot_id"]  # (N,)

    diff = yw - yl

    # MSE gap per window
    mse_gap = (diff**2).mean(axis=1)

    # Norms
    yw_norm = np.linalg.norm(yw, axis=1)  # (N,)
    yl_norm = np.linalg.norm(yl, axis=1)  # (N,)

    # Normalised margin
    norm_margin = np.linalg.norm(diff, axis=1) / (yw_norm + 1e-9)

    # Direction vs magnitude decomposition
    # directional error: 1 − cosine(y_w, y_l)
    yw_n = yw_norm.reshape(-1, 1)
    yl_n = yl_norm.reshape(-1, 1)
    denom = np.maximum(yw_n * yl_n, 1e-9)
    cosines = (yw * yl).sum(axis=1) / denom.squeeze()
    dir_err = 1.0 - cosines  # (N,)
    # magnitude error: |‖y_l‖ − ‖y_w‖| / (‖y_w‖ + ε)
    mag_err = np.abs(yl_norm - yw_norm) / (yw_norm + 1e-9)  # (N,)

    # Per-shot MSE gap
    unique_shots = np.unique(shot_ids)
    shot_mse_gap: dict[int, float] = {}
    for sid in unique_shots:
        mask = shot_ids == sid
        shot_mse_gap[int(sid)] = float(mse_gap[mask].mean())

    # Coefficient error × temporal bin heatmap (first 64 coeff dims)
    n_bins = 20
    D_cap = min(D, 64)
    coeff_abs = np.abs(diff[:, :D_cap])  # (N, D_cap)
    win_min, win_max = int(win_idx.min()), int(win_idx.max())
    bin_edges = np.linspace(win_min, win_max + 1, n_bins + 1)
    coeff_err_by_winbin = np.zeros((n_bins, D_cap), dtype=np.float32)
    for b in range(n_bins):
        mask_b = (win_idx >= bin_edges[b]) & (win_idx < bin_edges[b + 1])
        if mask_b.any():
            coeff_err_by_winbin[b] = coeff_abs[mask_b].mean(axis=0)

    # PCA sample — cap at 4096 rows to keep it fast
    rng = np.random.default_rng(0)
    pca_n = min(n, 4096)
    idx_pca = rng.choice(n, size=pca_n, replace=False)
    y_w_sample = yw[idx_pca]  # (pca_n, D)
    win_idx_sample = win_idx[idx_pca]

    return {
        "mse_gap": mse_gap,
        "norm_margin": norm_margin,
        "yw_norm": yw_norm,
        "yl_norm": yl_norm,
        "dir_err": dir_err,
        "mag_err": mag_err,
        "dir_err_mean": float(dir_err.mean()),
        "mag_err_mean": float(mag_err.mean()),
        "shot_mse_gap": shot_mse_gap,
        "coeff_err_by_winbin": coeff_err_by_winbin,
        "n_win_bins": n_bins,
        "window_indices": win_idx,
        "shot_id": shot_ids,
        "y_w_sample": y_w_sample,
        "win_idx_sample": win_idx_sample,
    }


# ======================================================================================================================
# Figure creation — extended diagnostics (8 panels, schema v2/v3 only)
# ======================================================================================================================


def _make_figure_extended(
    gpo_dir: Path,
    signal_filter: str | None,
    max_windows: int,
    dpi: int,
) -> Figure:
    """
    Produce Figure 2: extended diagnostic panels (13–20).

    Panels
    ------
    13. Preference-strength CDF
    14. Pair hardness tiers × temporal coverage
    15. Embedding energy bias (‖y_w‖ vs ‖y_l‖ hexbin)
    16. Direction vs magnitude error decomposition
    17. Per-shot mean MSE gap histogram
    18. Coefficient error × time heatmap
    19. Ground-truth norm drift over time
    20. Embedding PCA diversity
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------
    meta: dict[str, Any] = {}
    cc: dict[str, Any] = {}
    meta_path = gpo_dir / "metadata.json"
    cc_path = gpo_dir / "collection_config.json"
    if meta_path.is_file():
        meta = _load_json(meta_path)
    if cc_path.is_file():
        cc = _load_json(cc_path)

    schema_version = int(meta.get("schema_version", cc.get("schema_version", 1)))
    task = cc.get("task", meta.get("task", "unknown"))
    run_id = cc.get("run_id", meta.get("run_id", "unknown"))

    if schema_version < 2:
        raise ValueError(
            "Extended diagnostics require schema v2 or v3 shards (y_w_emb / y_l_emb). "
            f"Dataset at {gpo_dir} has schema v{schema_version}.  "
            "Re-collect with run_collect_gpo_pairs.py."
        )

    # ------------------------------------------------------------------
    # Discover + load signal data
    # ------------------------------------------------------------------
    all_signals = _discover_signals(gpo_dir)
    if not all_signals:
        raise FileNotFoundError(f"No .npz shards found in {gpo_dir}")

    if signal_filter is not None:
        safe_filter = signal_filter.replace("/", "-").replace("\\", "-")
        all_signals = {k: v for k, v in all_signals.items() if k == safe_filter}
        if not all_signals:
            raise FileNotFoundError(
                f"Signal {signal_filter!r} not found in {gpo_dir}. Available: {list(_discover_signals(gpo_dir).keys())}"
            )

    per_signal: dict[str, dict[str, Any]] = {}
    for sname, shards in all_signals.items():
        log.info("  [extended] Loading signal %r (%d shards)…", sname, len(shards))
        arrays = _load_arrays(shards, schema_version=schema_version, max_windows=max_windows)
        if not arrays:
            continue
        per_signal[sname] = _compute_extended_stats(arrays)

    if not per_signal:
        raise RuntimeError("No valid signals found for extended diagnostics.")

    signal_names = list(per_signal.keys())
    n_signals = len(signal_names)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_signals, 1)))

    # ------------------------------------------------------------------
    # Layout: 4 rows × 2 cols = 8 panels
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 24), dpi=dpi)
    fig.suptitle(
        f"GPO Extended Diagnostics\ndir: {gpo_dir.name}   task: {task}   run: {run_id}",
        fontsize=13,
        y=0.995,
    )

    gs = gridspec.GridSpec(
        4,
        2,
        figure=fig,
        hspace=0.50,
        wspace=0.38,
        top=0.96,
        bottom=0.04,
        left=0.07,
        right=0.97,
    )

    ax_cdf = fig.add_subplot(gs[0, 0])
    ax_hardness = fig.add_subplot(gs[0, 1])
    ax_energy = fig.add_subplot(gs[1, 0])
    ax_decomp = fig.add_subplot(gs[1, 1])
    ax_shot_mse = fig.add_subplot(gs[2, 0])
    ax_heatmap = fig.add_subplot(gs[2, 1])
    ax_norm_drift = fig.add_subplot(gs[3, 0])
    ax_pca = fig.add_subplot(gs[3, 1])

    # ---- Panel 13: Preference-strength CDF ----
    for i, sname in enumerate(signal_names):
        ext = per_signal[sname]
        mse_sorted = np.sort(ext["mse_gap"])
        cdf = np.arange(1, len(mse_sorted) + 1) / len(mse_sorted)
        ax_cdf.plot(
            mse_sorted,
            cdf,
            color=colors[i],
            linewidth=1.5,
            label=f"{sname}  (p50={np.percentile(mse_sorted, 50):.4f})",
        )
        # Mark p50, p90, p99 ticks
        for pct, ls in [(50, "--"), (90, ":"), (99, "-.")]:
            val = float(np.percentile(mse_sorted, pct))
            ax_cdf.axvline(val, color=colors[i], linewidth=0.8, linestyle=ls, alpha=0.6)
    ax_cdf.set_xlabel("MSE gap  ‖y_w − y_l‖² / D  (β-unscaled)")
    ax_cdf.set_ylabel("Cumulative fraction of pairs")
    ax_cdf.set_title("Panel 13 — Preference-Strength CDF\nDashes: p50 | dots: p90 | dash-dot: p99")
    ax_cdf.set_xlim(left=0.0)
    ax_cdf.set_ylim(0.0, 1.0)
    ax_cdf.legend(fontsize=7)

    # ---- Panel 14: Pair hardness tiers × temporal coverage ----
    # Use the first (or only) signal; if multi-signal, pool all windows.
    all_norm_margins = np.concatenate([per_signal[s]["norm_margin"] for s in signal_names])
    all_win_indices = np.concatenate([per_signal[s]["window_indices"] for s in signal_names])
    q_edges = np.percentile(all_norm_margins, [0, 25, 50, 75, 100])
    tier_labels = ["Q1 easy (0–25%)", "Q2 (25–50%)", "Q3 (50–75%)", "Q4 hard (75–100%)"]
    tier_colors_h = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
    win_min_h, win_max_h = int(all_win_indices.min()), int(all_win_indices.max())
    bins_h = np.linspace(win_min_h, win_max_h + 1, 31)
    bottom_h = np.zeros(30)
    for t_idx in range(4):
        lo, hi = q_edges[t_idx], q_edges[t_idx + 1]
        mask_t = (all_norm_margins >= lo) & (all_norm_margins < hi)
        # include upper bound for Q4
        if t_idx == 3:
            mask_t = all_norm_margins >= lo
        counts, _ = np.histogram(all_win_indices[mask_t], bins=bins_h)
        ax_hardness.bar(
            (bins_h[:-1] + bins_h[1:]) / 2,
            counts,
            width=(bins_h[1] - bins_h[0]) * 0.9,
            bottom=bottom_h,
            color=tier_colors_h[t_idx],
            label=tier_labels[t_idx],
        )
        bottom_h = bottom_h + counts
    ax_hardness.set_xlabel("window_index (temporal position)")
    ax_hardness.set_ylabel("Pair count")
    ax_hardness.set_title(
        "Panel 14 — Hardness Tiers × Temporal Coverage\nHard pairs in specific phases → localised plasma events"
    )
    ax_hardness.legend(fontsize=7, loc="upper right")

    # ---- Panel 15: Embedding energy bias (hexbin scatter ‖y_w‖ vs ‖y_l‖) ----
    # Pool all signals for this overview panel.
    all_yw_norm = np.concatenate([per_signal[s]["yw_norm"] for s in signal_names])
    all_yl_norm = np.concatenate([per_signal[s]["yl_norm"] for s in signal_names])
    # Cap at 20k points for speed
    _cap = 20_000
    if len(all_yw_norm) > _cap:
        _idx = np.random.default_rng(1).choice(len(all_yw_norm), _cap, replace=False)
        all_yw_norm = all_yw_norm[_idx]
        all_yl_norm = all_yl_norm[_idx]
    hb = ax_energy.hexbin(
        all_yw_norm,
        all_yl_norm,
        gridsize=40,
        cmap="viridis",
        mincnt=1,
        bins="log",
    )
    ax_energy.figure.colorbar(hb, ax=ax_energy, label="log(count)")
    _diag_max = max(all_yw_norm.max(), all_yl_norm.max()) * 1.05
    ax_energy.plot([0, _diag_max], [0, _diag_max], "r--", linewidth=1.0, label="identity (no bias)")
    ax_energy.set_xlabel("‖y_w‖₂  (ground-truth L2 norm)")
    ax_energy.set_ylabel("‖y_l‖₂  (model-prediction L2 norm)")
    ax_energy.set_title("Panel 15 — Embedding Energy Bias\nPoints above diagonal: model over-predicts amplitude")
    ax_energy.legend(fontsize=8)

    # ---- Panel 16: Direction vs magnitude error decomposition ----
    dir_means = [per_signal[s]["dir_err_mean"] for s in signal_names]
    mag_means = [per_signal[s]["mag_err_mean"] for s in signal_names]
    x_decomp = np.arange(n_signals)
    w_decomp = 0.35
    ax_decomp.bar(x_decomp - w_decomp / 2, dir_means, width=w_decomp, label="Directional (1−cosine)", color="steelblue")
    ax_decomp.bar(
        x_decomp + w_decomp / 2, mag_means, width=w_decomp, label="Magnitude (|‖y_l‖−‖y_w‖|/‖y_w‖)", color="coral"
    )
    ax_decomp.set_xticks(x_decomp)
    ax_decomp.set_xticklabels(signal_names, rotation=25, ha="right", fontsize=8)
    ax_decomp.set_ylabel("Mean error component")
    ax_decomp.set_title(
        "Panel 16 — Direction vs Magnitude Error\n"
        "Directional ≫ magnitude → model gets shape wrong; vice-versa → wrong amplitude"
    )
    ax_decomp.legend(fontsize=8)
    for i, (dm, mm) in enumerate(zip(dir_means, mag_means)):
        ax_decomp.text(i - w_decomp / 2, dm + 0.001, f"{dm:.3f}", ha="center", fontsize=6, color="steelblue")
        ax_decomp.text(i + w_decomp / 2, mm + 0.001, f"{mm:.3f}", ha="center", fontsize=6, color="coral")

    # ---- Panel 17: Per-shot mean MSE gap histogram ----
    all_shot_mse: list[float] = []
    shot_id_list: list[int] = []
    for sname in signal_names:
        for sid, v in per_signal[sname]["shot_mse_gap"].items():
            all_shot_mse.append(v)
            shot_id_list.append(sid)
    if all_shot_mse:
        ax_shot_mse.hist(all_shot_mse, bins=40, color="mediumpurple", edgecolor="white", linewidth=0.4)
        ax_shot_mse.set_xlabel("Per-shot mean MSE gap")
        ax_shot_mse.set_ylabel("Count (shots)")
        ax_shot_mse.set_title(
            "Panel 17 — Per-Shot Mean MSE Gap\nTail shots dominate preference loss — annotated top-10"
        )
        ax_shot_mse.axvline(
            float(np.median(all_shot_mse)),
            color="tomato",
            linewidth=1.2,
            linestyle="--",
            label=f"median={np.median(all_shot_mse):.4f}",
        )
        ax_shot_mse.legend(fontsize=8)
        # Annotate top-10 worst shots
        sorted_idx = np.argsort(all_shot_mse)[::-1][:10]
        for rank, idx in enumerate(sorted_idx):
            ax_shot_mse.annotate(
                f"#{shot_id_list[idx]}",
                xy=(all_shot_mse[idx], 0),
                xytext=(all_shot_mse[idx], 0.5 + rank * 0.4),
                textcoords="data",
                fontsize=5,
                color="tomato",
                rotation=45,
                ha="right",
            )

    # ---- Panel 18: Coefficient error × temporal bin heatmap ----
    # Use first signal for clarity (or average if same D).
    coeff_maps = [per_signal[s]["coeff_err_by_winbin"] for s in signal_names]
    if len(coeff_maps) > 1 and all(m.shape == coeff_maps[0].shape for m in coeff_maps):
        heat_data = np.stack(coeff_maps).mean(axis=0)  # (n_bins, D_cap)
        heat_label = "avg across signals"
    else:
        heat_data = coeff_maps[0]
        heat_label = signal_names[0]
    n_bins_h = heat_data.shape[0]
    im = ax_heatmap.imshow(
        heat_data.T,  # (D_cap, n_bins) — coeff on y, time on x
        aspect="auto",
        origin="lower",
        cmap="hot",
        interpolation="nearest",
    )
    ax_heatmap.figure.colorbar(im, ax=ax_heatmap, label="mean |Δ coeff|")
    ax_heatmap.set_xlabel("Temporal bin (window_index)")
    ax_heatmap.set_ylabel("Coeff index d  (first 64)")
    ax_heatmap.set_xticks(np.linspace(0, n_bins_h - 1, min(n_bins_h, 5)).astype(int))
    ax_heatmap.set_title(
        f"Panel 18 — Coeff Error × Time Heatmap  ({heat_label})\n"
        "Bright rows: high-error DCT modes; bright cols: high-error plasma phases"
    )

    # ---- Panel 19: Ground-truth norm drift + model norm over time ----
    # Bin windows and compute mean ‖y_w‖ and ‖y_l‖ per bin.
    all_win_19 = np.concatenate([per_signal[s]["window_indices"] for s in signal_names])
    win_min_19, win_max_19 = int(all_win_19.min()), int(all_win_19.max())
    bins_19 = np.linspace(win_min_19, win_max_19 + 1, 31)
    bin_centres_19 = (bins_19[:-1] + bins_19[1:]) / 2

    for i, sname in enumerate(signal_names):
        ext = per_signal[sname]
        win_s = ext["window_indices"]
        yw_n_s = ext["yw_norm"]
        yl_n_s = ext["yl_norm"]
        # Compute mean per bin
        yw_bin = np.full(30, np.nan)
        yl_bin = np.full(30, np.nan)
        for b in range(30):
            mask_b = (win_s >= bins_19[b]) & (win_s < bins_19[b + 1])
            if mask_b.any():
                yw_bin[b] = yw_n_s[mask_b].mean()
                yl_bin[b] = yl_n_s[mask_b].mean()
        ax_norm_drift.plot(
            bin_centres_19,
            yw_bin,
            color=colors[i],
            linewidth=1.6,
            label=f"{sname} ‖y_w‖",
        )
        ax_norm_drift.plot(
            bin_centres_19,
            yl_bin,
            color=colors[i],
            linewidth=1.0,
            linestyle="--",
            label=f"{sname} ‖y_l‖ (model)",
        )
    ax_norm_drift.set_xlabel("window_index (temporal position)")
    ax_norm_drift.set_ylabel("Mean L2 norm")
    ax_norm_drift.set_title("Panel 19 — Norm Drift Over Time\nSolid: ground-truth ‖y_w‖₂;  dashed: model ‖y_l‖₂")
    ax_norm_drift.legend(fontsize=7)

    # ---- Panel 20: Embedding PCA diversity ----
    # Pool y_w samples from all signals (same D required, else first signal only).
    pca_samples_list = [per_signal[s]["y_w_sample"] for s in signal_names]
    win_idx_pca_list = [per_signal[s]["win_idx_sample"] for s in signal_names]
    if len(pca_samples_list) > 1 and all(p.shape[1] == pca_samples_list[0].shape[1] for p in pca_samples_list):
        pca_data = np.concatenate(pca_samples_list, axis=0)
        win_pca = np.concatenate(win_idx_pca_list)
        pca_source_label = "all signals merged"
    else:
        pca_data = pca_samples_list[0]
        win_pca = win_idx_pca_list[0]
        pca_source_label = signal_names[0]

    # Cap further if needed
    _pca_cap = 2048
    if len(pca_data) > _pca_cap:
        _rng = np.random.default_rng(2)
        _sel = _rng.choice(len(pca_data), _pca_cap, replace=False)
        pca_data = pca_data[_sel]
        win_pca = win_pca[_sel]

    # Centre + PCA via SVD
    pca_mean = pca_data.mean(axis=0, keepdims=True)
    pca_centred = pca_data - pca_mean
    try:
        _, sv, Vt = np.linalg.svd(pca_centred, full_matrices=False)
        proj = pca_centred @ Vt[:2].T  # (N, 2)
        var_explained = (sv[:2] ** 2) / (sv**2).sum() * 100
        pca_ok = True
    except Exception:
        pca_ok = False

    if pca_ok:
        sc = ax_pca.scatter(
            proj[:, 0],
            proj[:, 1],
            c=win_pca,
            cmap="plasma",
            s=6,
            alpha=0.7,
            linewidths=0,
        )
        ax_pca.figure.colorbar(sc, ax=ax_pca, label="window_index")
        ax_pca.set_xlabel(f"PC1  ({var_explained[0]:.1f}% var)")
        ax_pca.set_ylabel(f"PC2  ({var_explained[1]:.1f}% var)")
        ax_pca.set_title(
            f"Panel 20 — Embedding PCA Diversity  ({pca_source_label})\n"
            "Tight cluster = low diversity (GPO overfit risk); "
            "colour = temporal position"
        )
    else:
        ax_pca.text(
            0.5,
            0.5,
            "PCA failed",
            ha="center",
            va="center",
            transform=ax_pca.transAxes,
            fontsize=12,
        )
        ax_pca.set_title("Panel 20 — Embedding PCA Diversity  (error)")

    return fig


# ======================================================================================================================
# Figure creation — actionability diagnostics (3 panels, schema v2/v3 only)
# ======================================================================================================================


def _make_figure_actionability(
    gpo_dir: Path,
    signal_filter: str | None,
    max_windows: int,
    dpi: int,
    beta_override: float | None = None,
) -> Figure:
    """
    Produce Figure 3: actionability diagnostic panels (21–23).

    Panels
    ------
    21. β-Effectiveness Score Distribution — σ(β × MSE_gap) histogram
    22. Pair Quality 2-D Scatter — normalised margin vs cosine, coloured by MSE gap
    23. Shot-Level Outlier Table — top-15 worst shots by mean MSE gap
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------
    meta: dict[str, Any] = {}
    cc: dict[str, Any] = {}
    meta_path = gpo_dir / "metadata.json"
    cc_path = gpo_dir / "collection_config.json"
    if meta_path.is_file():
        meta = _load_json(meta_path)
    if cc_path.is_file():
        cc = _load_json(cc_path)

    schema_version = int(meta.get("schema_version", cc.get("schema_version", 1)))
    task = cc.get("task", meta.get("task", "unknown"))
    run_id = cc.get("run_id", meta.get("run_id", "unknown"))

    if schema_version < 2:
        raise ValueError(
            "Actionability diagnostics require schema v2 or v3 shards (y_w_emb / y_l_emb). "
            f"Dataset at {gpo_dir} has schema v{schema_version}.  "
            "Re-collect with run_collect_gpo_pairs.py."
        )

    # Resolve β: explicit override > collection_config (not stored there) > default 0.1.
    # The config does not persist β but gpo_tasks.yaml values are the authoritative source;
    # the user passes --beta on the CLI when they want the panel to be meaningful.
    beta = float(beta_override) if beta_override is not None else 0.1

    # ------------------------------------------------------------------
    # Discover + load signal data
    # ------------------------------------------------------------------
    all_signals = _discover_signals(gpo_dir)
    if not all_signals:
        raise FileNotFoundError(f"No .npz shards found in {gpo_dir}")

    if signal_filter is not None:
        safe_filter = signal_filter.replace("/", "-").replace("\\", "-")
        all_signals = {k: v for k, v in all_signals.items() if k == safe_filter}
        if not all_signals:
            raise FileNotFoundError(
                f"Signal {signal_filter!r} not found in {gpo_dir}. Available: {list(_discover_signals(gpo_dir).keys())}"
            )

    per_signal: dict[str, dict[str, Any]] = {}
    for sname, shards in all_signals.items():
        log.info("  [actionability] Loading signal %r (%d shards)…", sname, len(shards))
        arrays = _load_arrays(shards, schema_version=schema_version, max_windows=max_windows)
        if not arrays:
            continue
        per_signal[sname] = _compute_signal_stats(arrays)
        # Also compute extended stats for shot-level info
        per_signal[sname]["_ext"] = _compute_extended_stats(arrays)

    if not per_signal:
        raise RuntimeError("No valid signals found for actionability diagnostics.")

    signal_names = list(per_signal.keys())
    n_signals = len(signal_names)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_signals, 1)))

    # ------------------------------------------------------------------
    # Layout: 2 rows × 2 cols with the table spanning the full bottom row
    # We use gridspec with 3 rows: top-left and top-right for panels 21/22,
    # bottom row spanning both columns for panel 23.
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 18), dpi=dpi)
    fig.suptitle(
        f"GPO Actionability Diagnostics\ndir: {gpo_dir.name}   task: {task}   run: {run_id}   β={beta:.4g}",
        fontsize=13,
        y=0.995,
    )

    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.42,
        wspace=0.36,
        top=0.96,
        bottom=0.04,
        left=0.06,
        right=0.97,
        height_ratios=[1.1, 0.9],
    )

    ax_beta = fig.add_subplot(gs[0, 0])
    ax_quality = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])  # spans both columns

    # ---- Panel 21: β-Effectiveness Score Distribution ----
    # For each signal, plot the distribution of σ(β × mse_gap).
    # Also show a twin x-axis that marks the MSE gap at which σ=0.5 (1/β),
    # σ=0.73 (target operating point), and σ=0.95 (near-saturation).
    _sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731

    for i, sname in enumerate(signal_names):
        mse = per_signal[sname]["mse_gap"]  # (N,)
        gate = _sigmoid(beta * mse)
        ax_beta.hist(
            gate,
            bins=60,
            density=True,
            histtype="stepfilled",
            alpha=0.40,
            color=colors[i],
            label=(
                f"{sname}\n"
                f"μ={gate.mean():.3f}  "
                f"p50={np.percentile(gate, 50):.3f}  "
                f"frac<0.52={float((gate < 0.52).mean()):.1%}"
            ),
        )

    # Reference vertical lines at key gate values
    for gval, ls, lbl in [
        (0.50, "--", "σ=0.5 (zero-gradient)"),
        (0.73, "-", "σ=0.73 (ideal)"),
        (0.95, ":", "σ=0.95 (near-saturated)"),
    ]:
        ax_beta.axvline(gval, color="grey", linewidth=1.0, linestyle=ls, label=lbl)

    ax_beta.set_xlabel("σ(β × MSE_gap)  (sigmoid gate value per pair)")
    ax_beta.set_ylabel("Density")
    ax_beta.set_xlim(0.0, 1.0)
    ax_beta.set_title(
        f"Panel 21 — β-Effectiveness Score  (β={beta:.4g})\n"
        "Ideal: distribution centred near 0.73 | "
        "Peak at 0.5 → too small β | Peak at 1.0 → too large β"
    )
    ax_beta.legend(fontsize=6, loc="upper left")

    # Annotate β-calibration rule-of-thumb: β × p50(MSE) = target
    for i, sname in enumerate(signal_names):
        p50_mse = float(per_signal[sname]["mse_gap_p50"])
        product = beta * p50_mse
        ax_beta.annotate(
            f"β×p50={product:.3f}\n({sname})",
            xy=(_sigmoid(product), 0.0),
            xytext=(0, 14 + 28 * i),
            textcoords="offset points",
            fontsize=5.5,
            color=colors[i],
            ha="center",
            arrowprops=dict(arrowstyle="->", color=colors[i], lw=0.6),
        )

    # ---- Panel 22: Pair Quality 2-D Scatter ----
    # Pool all signals (cap at 20k for rendering speed).
    all_nm = np.concatenate([per_signal[s]["margins_normalised"] for s in signal_names])
    all_cos = np.concatenate([per_signal[s]["cosines"] for s in signal_names])
    all_mse = np.concatenate([per_signal[s]["mse_gap"] for s in signal_names])

    _cap22 = 20_000
    if len(all_nm) > _cap22:
        _rng22 = np.random.default_rng(3)
        _sel22 = _rng22.choice(len(all_nm), _cap22, replace=False)
        all_nm = all_nm[_sel22]
        all_cos = all_cos[_sel22]
        all_mse = all_mse[_sel22]

    sc = ax_quality.scatter(
        all_nm,
        all_cos,
        c=np.log1p(all_mse),  # log-scale colour to handle fat tails
        cmap="plasma",
        s=4,
        alpha=0.5,
        linewidths=0,
    )
    fig.colorbar(sc, ax=ax_quality, label="log(1 + MSE gap)")

    # Quadrant lines at median of each axis
    nm_med = float(np.median(all_nm))
    cos_med = float(np.median(all_cos))
    ax_quality.axvline(nm_med, color="white", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_quality.axhline(cos_med, color="white", linewidth=0.8, linestyle="--", alpha=0.7)

    # Annotate quadrant labels
    _xm, _xM = all_nm.min(), all_nm.max()
    _ym, _yM = all_cos.min(), all_cos.max()
    _qfont = dict(
        fontsize=7,
        ha="center",
        va="center",
        color="white",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#00000066", edgecolor="none"),
    )
    ax_quality.text((_xm + nm_med) / 2, (_yM + cos_med) / 2, "Near-perfect\n(wasted)", **_qfont)
    ax_quality.text((nm_med + _xM) / 2, (_yM + cos_med) / 2, "✓ Actionable\n(shape ok, amp off)", **_qfont)
    ax_quality.text((_xm + nm_med) / 2, (_ym + cos_med) / 2, "Noise / artefacts", **_qfont)
    ax_quality.text((nm_med + _xM) / 2, (_ym + cos_med) / 2, "Hard pairs\n(shape wrong)", **_qfont)

    ax_quality.set_xlabel("Normalised margin  ‖y_w − y_l‖₂ / ‖y_w‖₂")
    ax_quality.set_ylabel("Cosine similarity  cos(y_w, y_l)")
    ax_quality.set_title(
        "Panel 22 — Pair Quality 2-D Scatter\nDashed lines = median per axis | colour = log(1+MSE gap)"
    )

    # ---- Panel 23: Shot-Level Outlier Table (top-15 worst shots) ----
    ax_table.axis("off")

    # Aggregate shot-level stats across all signals
    # shot_id → {mse_vals, nm_vals, cos_vals, n_windows}
    shot_agg: dict[int, dict[str, list[float]]] = {}
    for sname in signal_names:
        ext = per_signal[sname]["_ext"]
        shot_ids_s = ext["shot_id"]
        mse_gap_s = ext["mse_gap"]
        nm_s = ext["norm_margin"]
        cos_s = 1.0 - ext["dir_err"]  # dir_err = 1 - cos, so cos = 1 - dir_err

        for sid in np.unique(shot_ids_s):
            mask_s = shot_ids_s == sid
            entry = shot_agg.setdefault(int(sid), {"mse": [], "nm": [], "cos": [], "n": 0})
            entry["mse"].extend(mse_gap_s[mask_s].tolist())
            entry["nm"].extend(nm_s[mask_s].tolist())
            entry["cos"].extend(cos_s[mask_s].tolist())
            entry["n"] += int(mask_s.sum())

    # Build rows: mean_mse, sort descending
    table_rows: list[tuple] = []
    for sid, e in shot_agg.items():
        mse_arr = np.asarray(e["mse"], dtype=np.float32)
        nm_arr = np.asarray(e["nm"], dtype=np.float32)
        cos_arr = np.asarray(e["cos"], dtype=np.float32)
        table_rows.append(
            (
                sid,
                e["n"],
                float(mse_arr.mean()),
                float(np.median(mse_arr)),
                float(np.percentile(mse_arr, 95)),
                float(nm_arr.mean()),
                float(cos_arr.mean()),
            )
        )

    table_rows.sort(key=lambda r: r[2], reverse=True)  # sort by mean MSE desc
    top_n = min(15, len(table_rows))

    col_headers = ["Rank", "shot_id", "n_windows", "mean MSE", "p50 MSE", "p95 MSE", "mean norm_margin", "mean cosine"]
    col_x = [0.0, 0.05, 0.16, 0.26, 0.36, 0.46, 0.60, 0.76]

    y0_tbl, dy_tbl = 0.97, 0.058

    # Header row background
    ax_table.axhspan(
        y0_tbl - dy_tbl * 0.85,
        y0_tbl + dy_tbl * 0.05,
        xmin=0,
        xmax=1,
        color="steelblue",
        transform=ax_table.transAxes,
        clip_on=True,
    )

    for c, (hdr, x) in enumerate(zip(col_headers, col_x)):
        ax_table.text(
            x + 0.005,
            y0_tbl,
            hdr,
            fontsize=7.5,
            va="top",
            ha="left",
            fontweight="bold",
            color="white",
            transform=ax_table.transAxes,
        )

    for r, row in enumerate(table_rows[:top_n]):
        rank, sid, n_win, mean_mse, p50_mse, p95_mse, mean_nm, mean_cos = r + 1, *row
        row_y = y0_tbl - dy_tbl * (r + 1)

        if r % 2 == 0:
            ax_table.axhspan(
                row_y - dy_tbl * 0.85,
                row_y + dy_tbl * 0.05,
                xmin=0,
                xmax=1,
                color="#f0f4f8",
                transform=ax_table.transAxes,
                clip_on=True,
            )

        # Highlight the worst 3 rows
        txt_color = "darkred" if r < 3 else "#1f2328"
        cells = [
            str(rank),
            str(sid),
            str(n_win),
            f"{mean_mse:.5f}",
            f"{p50_mse:.5f}",
            f"{p95_mse:.4f}",
            f"{mean_nm:.4f}",
            f"{mean_cos:.4f}",
        ]
        for c, (val, x) in enumerate(zip(cells, col_x)):
            ax_table.text(
                x + 0.005, row_y, val, fontsize=7, va="top", ha="left", color=txt_color, transform=ax_table.transAxes
            )

    ax_table.set_title(
        f"Panel 23 — Shot-Level Outlier Table  (top-{top_n} worst shots by mean MSE gap)\n"
        "Red rows = top-3 dominant shots — candidates for blacklisting in the GPO dataloader",
        pad=8,
    )

    return fig


# ======================================================================================================================
# Figure creation — multi-directory comparison (4-panel overlay)
# ======================================================================================================================


def _make_compare_figure(
    dirs_data: list[tuple[Path, dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]],
    max_windows: int,
    dpi: int,
) -> Figure:
    """
    Overlay comparison figure: margin, cosine, normalised margin, and MSE gap
    across multiple GPO directories.

    Parameters
    ----------
    dirs_data : list of (gpo_dir, meta, cc, per_signal_stats)
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_dirs = len(dirs_data)
    fig = plt.figure(figsize=(18, 12), dpi=dpi)
    fig.suptitle(
        "GPO Dataset Comparison\n" + "  |  ".join(d[0].name for d in dirs_data),
        fontsize=12,
        y=0.995,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35, top=0.94, bottom=0.06, left=0.07, right=0.97)
    ax_margin = fig.add_subplot(gs[0, 0])
    ax_cosine = fig.add_subplot(gs[0, 1])
    ax_norm_marg = fig.add_subplot(gs[1, 0])
    ax_mse_gap = fig.add_subplot(gs[1, 1])

    dir_colors = plt.cm.tab10(np.linspace(0, 1, max(n_dirs, 1)))

    for d_idx, (gpo_dir, meta, cc, per_signal) in enumerate(dirs_data):
        label_base = gpo_dir.name
        task = cc.get("task", meta.get("task", "?"))
        c = dir_colors[d_idx]

        for s_idx, (sname, stats) in enumerate(per_signal.items()):
            alpha = 0.55 if len(per_signal) > 1 else 0.65
            lw = 1.8

            label_margin = f"{label_base}/{sname}" if len(per_signal) > 1 else f"{label_base} (task={task})"

            ax_margin.hist(
                stats["margins"],
                bins=60,
                density=True,
                histtype="step",
                alpha=alpha,
                color=c,
                linewidth=lw,
                label=f"{label_margin}  μ={stats['margin_mean']:.2f}",
            )
            ax_cosine.hist(
                stats["cosines"],
                bins=60,
                density=True,
                histtype="step",
                alpha=alpha,
                color=c,
                linewidth=lw,
                label=f"{label_margin}  μ={stats['cosine_mean']:.3f}",
            )
            ax_norm_marg.hist(
                stats["margins_normalised"],
                bins=60,
                density=True,
                histtype="step",
                alpha=alpha,
                color=c,
                linewidth=lw,
                label=f"{label_margin}  μ={stats['norm_margin_mean']:.3f}",
            )
            ax_mse_gap.hist(
                stats["mse_gap"],
                bins=60,
                density=True,
                histtype="step",
                alpha=alpha,
                color=c,
                linewidth=lw,
                label=f"{label_margin}  μ={stats['mse_gap_mean']:.4f}",
            )

    ax_margin.set_xlabel("‖y_w − y_l‖₂")
    ax_margin.set_ylabel("Density")
    ax_margin.set_title("Preference Margin Distribution")
    ax_margin.legend(fontsize=6)

    ax_cosine.axvline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax_cosine.set_xlabel("cosine(y_w, y_l)")
    ax_cosine.set_ylabel("Density")
    ax_cosine.set_title("Cosine Similarity Distribution")
    ax_cosine.legend(fontsize=6)

    ax_norm_marg.axvline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax_norm_marg.set_xlabel("‖y_w − y_l‖₂ / ‖y_w‖₂")
    ax_norm_marg.set_ylabel("Density")
    ax_norm_marg.set_title("Normalised Margin Distribution")
    ax_norm_marg.legend(fontsize=6)

    ax_mse_gap.set_xlabel("‖y_w − y_l‖² / D  (MSE gap)")
    ax_mse_gap.set_ylabel("Density")
    ax_mse_gap.set_title("MSE Preference Gap Distribution")
    ax_mse_gap.legend(fontsize=6)

    return fig


# ======================================================================================================================
# ======================================================================================================================
# Figure — multi-task summary report (single figure, all dirs)
# ======================================================================================================================


def _make_report_figure(
    report_data: list[tuple[Path, dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]],
    beta: float,
    dpi: int,
) -> Figure:
    """
    Single-figure summary report across all processed GPO directories.

    Layout (3 rows):
      Row 0 — Summary table: one row per (task × signal) with key stats
      Row 1 — MSE gap distributions overlaid (step histograms)
      Row 2 — Cosine similarity and normalised margin distributions overlaid
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    _sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731

    # Build flat list of (label, task, signal, stats_dict) sorted by task then signal
    rows: list[tuple[str, str, str, dict[str, Any]]] = []
    for gpo_dir, meta, cc, per_signal in report_data:
        task = cc.get("task", meta.get("task", gpo_dir.parent.name))
        for sname, stats in per_signal.items():
            label = f"{task} / {sname}"
            rows.append((label, task, sname, stats))

    n_rows_data = len(rows)
    if n_rows_data == 0:
        raise RuntimeError("No data rows for report figure.")

    fig = plt.figure(figsize=(20, 6 + n_rows_data * 0.55 + 8), dpi=dpi)
    fig.suptitle(
        f"GPO Dataset Summary Report — {n_rows_data} signal(s) across {len(report_data)} task(s)",
        fontsize=14,
        y=0.995,
    )

    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        hspace=0.50,
        wspace=0.32,
        top=0.97,
        bottom=0.04,
        left=0.04,
        right=0.98,
        height_ratios=[max(1.8, n_rows_data * 0.42), 1.0, 1.0],
    )
    ax_table = fig.add_subplot(gs[0, :])  # full-width table
    ax_mse = fig.add_subplot(gs[1, 0])
    ax_cosine = fig.add_subplot(gs[1, 1])
    ax_norm = fig.add_subplot(gs[2, 0])
    ax_beta = fig.add_subplot(gs[2, 1])

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_rows_data, 1)))

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------
    ax_table.axis("off")

    col_headers = [
        "Task",
        "Signal",
        "n_windows",
        "MSE gap μ",
        "MSE gap p50",
        "cosine μ",
        "norm margin μ",
        f"β×p50  (β={beta:.3g})",
        "β-eff σ(β×p50)",
    ]
    col_x = [0.0, 0.10, 0.28, 0.37, 0.46, 0.55, 0.64, 0.74, 0.87]

    y0, dy = 0.97, min(0.90 / (n_rows_data + 1), 0.072)

    # Header background
    ax_table.axhspan(
        y0 - dy * 0.85,
        y0 + dy * 0.05,
        xmin=0,
        xmax=1,
        color="steelblue",
        transform=ax_table.transAxes,
        clip_on=True,
    )
    for hdr, x in zip(col_headers, col_x):
        ax_table.text(
            x + 0.004,
            y0,
            hdr,
            fontsize=7.5,
            va="top",
            ha="left",
            fontweight="bold",
            color="white",
            transform=ax_table.transAxes,
        )

    for r, (label, task, sname, stats) in enumerate(rows):
        row_y = y0 - dy * (r + 1)
        if r % 2 == 0:
            ax_table.axhspan(
                row_y - dy * 0.85,
                row_y + dy * 0.05,
                xmin=0,
                xmax=1,
                color="#f0f4f8",
                transform=ax_table.transAxes,
                clip_on=True,
            )
        p50_mse = stats["mse_gap_p50"]
        beta_p50 = beta * p50_mse
        beta_eff = float(_sigmoid(beta_p50))
        # Colour β-eff: green if 0.6–0.85 (ideal), amber if 0.5–0.6 or 0.85–0.95, red otherwise
        if 0.60 <= beta_eff <= 0.85:
            eff_color = "#15803d"
        elif 0.50 <= beta_eff <= 0.95:
            eff_color = "#92400e"
        else:
            eff_color = "#991b1b"

        cells = [
            task,
            sname,
            f"{stats['n_windows']:,}",
            f"{stats['mse_gap_mean']:.4f}",
            f"{p50_mse:.4f}",
            f"{stats['cosine_mean']:.4f}",
            f"{stats['norm_margin_mean']:.4f}",
            f"{beta_p50:.3f}",
            f"{beta_eff:.3f}",
        ]
        cell_colors = ["#1f2328"] * (len(cells) - 1) + [eff_color]
        for c, (val, x) in enumerate(zip(cells, col_x)):
            ax_table.text(
                x + 0.004,
                row_y,
                val,
                fontsize=7,
                va="top",
                ha="left",
                color=cell_colors[c],
                transform=ax_table.transAxes,
            )

    ax_table.set_title(
        "Per-Signal Statistics Summary  "
        "(β-eff colour: green=ideal 0.60–0.85 | amber=acceptable | red=saturated/zero-gradient)",
        pad=6,
        fontsize=9,
    )

    # ------------------------------------------------------------------
    # Distribution panels — overlaid step histograms per row entry
    # ------------------------------------------------------------------
    for r, (label, task, sname, stats) in enumerate(rows):
        c = colors[r]
        kw = dict(bins=60, density=True, histtype="step", linewidth=1.4, color=c, alpha=0.85)

        ax_mse.hist(stats["mse_gap"], label=f"{label}  μ={stats['mse_gap_mean']:.3f}", **kw)
        ax_cosine.hist(stats["cosines"], label=f"{label}  μ={stats['cosine_mean']:.3f}", **kw)
        ax_norm.hist(stats["margins_normalised"], label=f"{label}  μ={stats['norm_margin_mean']:.3f}", **kw)
        gate = _sigmoid(beta * stats["mse_gap"])
        ax_beta.hist(gate, label=f"{label}  p50={np.percentile(gate, 50):.3f}", **kw)

    ax_mse.set_xlabel("‖y_w − y_l‖² / D  (MSE gap)")
    ax_mse.set_ylabel("Density")
    ax_mse.set_title("MSE Preference Gap Distribution (all tasks)")
    ax_mse.legend(fontsize=6)

    ax_cosine.axvline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax_cosine.set_xlabel("cosine(y_w, y_l)")
    ax_cosine.set_ylabel("Density")
    ax_cosine.set_title("Cosine Similarity Distribution (all tasks)")
    ax_cosine.legend(fontsize=6)

    ax_norm.axvline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax_norm.set_xlabel("‖y_w − y_l‖₂ / ‖y_w‖₂  (normalised margin)")
    ax_norm.set_ylabel("Density")
    ax_norm.set_title("Normalised Margin Distribution (all tasks)")
    ax_norm.legend(fontsize=6)

    for gval, ls, lbl in [(0.50, "--", "σ=0.5"), (0.73, "-", "σ=0.73 ideal"), (0.95, ":", "σ=0.95")]:
        ax_beta.axvline(gval, color="grey", linewidth=0.9, linestyle=ls, label=lbl)
    ax_beta.set_xlabel(f"σ(β × MSE_gap)  (β={beta:.3g})")
    ax_beta.set_ylabel("Density")
    ax_beta.set_xlim(0.0, 1.0)
    ax_beta.set_title("β-Effectiveness Distribution (all tasks)")
    ax_beta.legend(fontsize=6)

    return fig


# ======================================================================================================================
# Training-signal quality stats computation (for Figure 4)
# ======================================================================================================================


def _compute_quality_stats(
    arrays: dict[str, np.ndarray],
    beta: float,
) -> dict[str, Any]:
    """
    Compute training-signal-quality statistics for Figure 4.

    Returns
    -------
    dict with keys:
        mse_gap             — ‖y_w − y_l‖²/D per window  (N,)
        mse_gap_sorted      — sorted ascending             (N,)
        lorenz_x            — cumulative fraction of pairs  (N+1,)
        lorenz_y            — cumulative fraction of total MSE  (N+1,)
        gini                — Gini coefficient of MSE distribution
        coeff_signed_bias   — E[y_l[d] − y_w[d]] per coeff  (D,)
        eff_n               — Fisher-weighted effective N: Σ 4σ(1-σ)  scalar
        total_n             — total pair count              int
        eff_n_frac          — eff_n / total_n               float
        between_shot_var    — between-shot MSE variance     float
        within_shot_var     — within-shot MSE variance      float
        total_var           — total MSE variance            float
        between_frac        — between_shot_var / total_var  float
        filter_thresholds   — log-spaced thresholds for survival plot  (T,)
        filter_survival     — fraction of pairs surviving each threshold (T,)
        beta_opt            — 1 / p50_MSE (recommended β)   float
        beta_current        — β passed in                   float
        beta_x_p50          — beta × p50_MSE                float
    """
    yw = arrays["y_w"]  # (N, D)
    yl = arrays["y_l"]  # (N, D)
    n, D = yw.shape
    shot_ids = arrays["shot_id"]  # (N,)

    diff = yw - yl
    mse_gap = (diff**2).mean(axis=1)  # (N,)

    # ---- Lorenz curve of MSE gap (pair gradient concentration) ----
    mse_sorted = np.sort(mse_gap)
    cumsum_mse = np.concatenate([[0.0], np.cumsum(mse_sorted)])
    total_mse = cumsum_mse[-1]
    lorenz_y = cumsum_mse / (total_mse + 1e-30)
    lorenz_x = np.linspace(0.0, 1.0, n + 1)

    # Gini = 1 − 2 × area under Lorenz curve  (0=perfect equality, 1=all in one pair)
    # np.trapezoid is the canonical name in NumPy ≥2.0; np.trapz is the deprecated alias.
    _trapz_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    area_lorenz = float(_trapz_fn(lorenz_y, lorenz_x))
    gini = float(1.0 - 2.0 * area_lorenz)

    # ---- Signed coefficient bias ----
    # E[y_l[d] − y_w[d]] — positive = model over-predicts that DCT mode
    coeff_signed_bias = (yl - yw).mean(axis=0)  # (D,)

    # ---- Fisher-weighted effective N ----
    # Each pair contributes 4σ(β·g)(1−σ(β·g)) to the Fisher information.
    # Sum gives "equivalent number of perfectly-informative (σ=0.73) pairs".
    sigmoid_gate = 1.0 / (1.0 + np.exp(-beta * mse_gap))
    fisher_weights = 4.0 * sigmoid_gate * (1.0 - sigmoid_gate)  # (N,)
    eff_n = float(fisher_weights.sum())
    eff_n_frac = eff_n / n if n > 0 else 0.0

    # ---- Between-shot vs within-shot MSE variance (one-way ANOVA) ----
    unique_shots = np.unique(shot_ids)
    grand_mean = float(mse_gap.mean())
    between_shot_var = 0.0
    within_shot_var = 0.0
    for sid in unique_shots:
        mask = shot_ids == sid
        shot_mse = mse_gap[mask]
        n_s = len(shot_mse)
        shot_mean = float(shot_mse.mean())
        between_shot_var += n_s * (shot_mean - grand_mean) ** 2
        within_shot_var += float(((shot_mse - shot_mean) ** 2).sum())
    between_shot_var /= n
    within_shot_var /= n
    total_var = float(mse_gap.var())
    between_frac = between_shot_var / (total_var + 1e-30)

    # ---- Filter survival curve ----
    # How many pairs survive a min_margin_mse threshold sweep
    t_min = float(mse_gap[mse_gap > 0].min()) if (mse_gap > 0).any() else 1e-6
    t_max = float(np.percentile(mse_gap, 99))
    filter_thresholds = np.logspace(np.log10(max(t_min, 1e-9)), np.log10(max(t_max, 1e-8)), 80)
    filter_survival = np.array([float((mse_gap >= thr).mean()) for thr in filter_thresholds])

    # ---- β-optimum ----
    p50_mse = float(np.percentile(mse_gap, 50))
    beta_opt = 1.0 / (p50_mse + 1e-30)
    beta_x_p50 = beta * p50_mse

    return {
        "mse_gap": mse_gap,
        "mse_sorted": mse_sorted,
        "lorenz_x": lorenz_x,
        "lorenz_y": lorenz_y,
        "gini": gini,
        "coeff_signed_bias": coeff_signed_bias,
        "eff_n": eff_n,
        "total_n": n,
        "eff_n_frac": eff_n_frac,
        "between_shot_var": between_shot_var,
        "within_shot_var": within_shot_var,
        "total_var": total_var,
        "between_frac": between_frac,
        "filter_thresholds": filter_thresholds,
        "filter_survival": filter_survival,
        "beta_opt": beta_opt,
        "beta_current": beta,
        "beta_x_p50": beta_x_p50,
        "p50_mse": p50_mse,
        "fisher_weights": fisher_weights,
    }


# ======================================================================================================================
# Figure 4 — Training Signal Quality (panels 24–29)
# ======================================================================================================================


def _make_figure_quality(
    gpo_dir: Path,
    signal_filter: str | None,
    max_windows: int,
    dpi: int,
    beta_override: float | None = None,
) -> Figure:
    """
    Produce Figure 4: training-signal-quality diagnostic panels (24–29).

    Panels
    ------
    24. **Lorenz Curve of MSE Gap** — cumulative gradient concentration.
        A diagonal line = all pairs equally informative; a concave bow =
        a small fraction of hard pairs drives almost all gradient signal.
        The Gini coefficient quantifies concentration (0 = uniform, 1 = one pair).

    25. **Signed Coefficient Bias** — ``E[y_l[d] − y_w[d]]`` per DCT coefficient.
        Positive bars = model systematically over-predicts that mode;
        negative = under-predicts.  Unlike the absolute error profile (panel 7),
        this reveals directionality — whether bias cancels across windows.

    26. **Fisher-Weighted Effective N** — per-signal bar chart of the Fisher-
        information-weighted effective pair count ``Σ 4σ(β·g)(1−σ(β·g))``.
        Pairs with gate near 0.5 (σ=0.5) contribute weight 1; saturated or
        zero-gradient pairs contribute ~0.  ``eff_N / total_N`` indicates
        what fraction of the dataset is actually informative at this β.

    27. **Between-Shot vs Within-Shot MSE Variance** — one-way ANOVA
        decomposition of MSE gap variance into between-shot (shot identity)
        and within-shot (temporal/random) components.  Shown as a stacked
        bar per signal.  High between-shot fraction → blacklisting a few
        dominant shots would substantially reduce variance; high within-shot
        → the residual error is spread uniformly across windows, and
        blacklisting would not help much.

    28. **Filter Survival Curve** — fraction of pairs surviving a
        ``min_margin_mse`` threshold sweep (log-spaced thresholds from
        near-zero to the 99th percentile).  Helps choose the right
        ``min_margin_mse`` value to cut near-zero pairs without discarding
        too many informative ones.  The vertical dashed line marks the
        recommended operating point (5% survival cutoff).

    29. **β Optimisation Table** — one row per signal: current β, p50 MSE,
        β×p50 (operating point), sigmoid gate at p50 (σ(β×p50)), and the
        recommended β_opt = 1/p50_MSE that places σ(β×p50) = 0.731 (the
        ideal operating point).  Colour-coded: green = already calibrated,
        amber = off by <3×, red = needs attention.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------
    meta: dict[str, Any] = {}
    cc: dict[str, Any] = {}
    meta_path = gpo_dir / "metadata.json"
    cc_path = gpo_dir / "collection_config.json"
    if meta_path.is_file():
        meta = _load_json(meta_path)
    if cc_path.is_file():
        cc = _load_json(cc_path)

    schema_version = int(meta.get("schema_version", cc.get("schema_version", 1)))
    task = cc.get("task", meta.get("task", "unknown"))
    run_id = cc.get("run_id", meta.get("run_id", "unknown"))

    if schema_version < 2:
        raise ValueError(
            "Training-signal quality diagnostics require schema v2 or v3 shards (y_w_emb / y_l_emb). "
            f"Dataset at {gpo_dir} has schema v{schema_version}.  "
            "Re-collect with run_collect_gpo_pairs.py."
        )

    beta = float(beta_override) if beta_override is not None else 0.1

    # ------------------------------------------------------------------
    # Discover + load signal data
    # ------------------------------------------------------------------
    all_signals = _discover_signals(gpo_dir)
    if not all_signals:
        raise FileNotFoundError(f"No .npz shards found in {gpo_dir}")

    if signal_filter is not None:
        safe_filter = signal_filter.replace("/", "-").replace("\\", "-")
        all_signals = {k: v for k, v in all_signals.items() if k == safe_filter}
        if not all_signals:
            raise FileNotFoundError(
                f"Signal {signal_filter!r} not found in {gpo_dir}. Available: {list(_discover_signals(gpo_dir).keys())}"
            )

    per_signal: dict[str, dict[str, Any]] = {}
    for sname, shards in all_signals.items():
        log.info("  [quality] Loading signal %r (%d shards)…", sname, len(shards))
        arrays = _load_arrays(shards, schema_version=schema_version, max_windows=max_windows)
        if not arrays:
            continue
        per_signal[sname] = _compute_quality_stats(arrays, beta=beta)
        q = per_signal[sname]
        log.info(
            "  Signal %r: Gini=%.3f | eff_N=%.0f (%.1f%% of %d) | between_shot_frac=%.1f%% | β_opt=%.2f | β×p50=%.3f",
            sname,
            q["gini"],
            q["eff_n"],
            q["eff_n_frac"] * 100,
            q["total_n"],
            q["between_frac"] * 100,
            q["beta_opt"],
            q["beta_x_p50"],
        )

    if not per_signal:
        raise RuntimeError("No valid signals found for quality diagnostics.")

    signal_names = list(per_signal.keys())
    n_signals = len(signal_names)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_signals, 1)))

    # ------------------------------------------------------------------
    # Layout: 3 rows × 2 cols = 6 panels (panels 24–29)
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 24), dpi=dpi)
    fig.suptitle(
        f"GPO Training-Signal Quality Diagnostics\ndir: {gpo_dir.name}   task: {task}   run: {run_id}   β={beta:.4g}",
        fontsize=13,
        y=0.995,
    )

    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        hspace=0.52,
        wspace=0.38,
        top=0.96,
        bottom=0.04,
        left=0.07,
        right=0.97,
    )

    ax_lorenz = fig.add_subplot(gs[0, 0])
    ax_bias = fig.add_subplot(gs[0, 1])
    ax_effn = fig.add_subplot(gs[1, 0])
    ax_anova = fig.add_subplot(gs[1, 1])
    ax_survival = fig.add_subplot(gs[2, 0])
    ax_beta_tbl = fig.add_subplot(gs[2, 1])

    # ---- Panel 24: Lorenz Curve of MSE Gap ----
    for i, sname in enumerate(signal_names):
        q = per_signal[sname]
        ax_lorenz.plot(
            q["lorenz_x"],
            q["lorenz_y"],
            color=colors[i],
            linewidth=1.8,
            label=f"{sname}  Gini={q['gini']:.3f}",
        )
    # Perfect-equality diagonal
    ax_lorenz.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5, label="perfect equality")
    ax_lorenz.fill_between([0, 1], [0, 1], [0, 0], alpha=0.04, color="grey")
    ax_lorenz.set_xlabel("Cumulative fraction of pairs (sorted by MSE gap, easiest first)")
    ax_lorenz.set_ylabel("Cumulative fraction of total MSE gap")
    ax_lorenz.set_title(
        "Panel 24 — Lorenz Curve of MSE Gap (pair gradient concentration)\n"
        "Diagonal = uniform gradient; bowed curve = hard-pair dominance"
    )
    ax_lorenz.set_xlim(0.0, 1.0)
    ax_lorenz.set_ylim(0.0, 1.05)
    ax_lorenz.legend(fontsize=7)

    # ---- Panel 25: Signed Coefficient Bias ----
    # Average signed bias across signals (if same D), else first signal
    bias_arrs = [per_signal[s]["coeff_signed_bias"] for s in signal_names]
    valid_biases = [b for b in bias_arrs if b is not None and len(b) > 0]
    if valid_biases:
        common_D = valid_biases[0].shape[0]
        if all(b.shape[0] == common_D for b in valid_biases):
            mean_bias = np.stack(valid_biases).mean(axis=0)
            bias_label = "avg over signals" if n_signals > 1 else signal_names[0]
        else:
            mean_bias = valid_biases[0]
            bias_label = signal_names[0] + " (D mismatch — first signal only)"
        dims = np.arange(len(mean_bias))
        bar_colors_b = ["coral" if v > 0 else "steelblue" for v in mean_bias]
        ax_bias.bar(dims, mean_bias, width=1.0, color=bar_colors_b, alpha=0.85)
        ax_bias.axhline(0.0, color="black", linewidth=0.8)
        ax_bias.set_xlabel("Embedding coefficient index d")
        ax_bias.set_ylabel("E[y_l[d] − y_w[d]]  (signed bias)")
        ax_bias.set_title(
            f"Panel 25 — Signed Coefficient Bias  ({bias_label})\n"
            "Coral = model over-predicts (y_l > y_w); blue = under-predicts"
        )
        # Annotate top-5 most biased coefficients (by magnitude)
        top5b = np.argsort(np.abs(mean_bias))[::-1][:5]
        for idx in top5b:
            ax_bias.annotate(
                str(idx),
                xy=(idx, mean_bias[idx]),
                xytext=(0, 5 if mean_bias[idx] >= 0 else -12),
                textcoords="offset points",
                ha="center",
                fontsize=6,
                color="darkred",
            )
    else:
        ax_bias.text(0.5, 0.5, "No coefficient data", ha="center", va="center", transform=ax_bias.transAxes)
        ax_bias.set_title("Panel 25 — Signed Coefficient Bias")

    # ---- Panel 26: Fisher-Weighted Effective N ----
    total_ns = [per_signal[s]["total_n"] for s in signal_names]
    eff_ns = [per_signal[s]["eff_n"] for s in signal_names]
    eff_fracs = [per_signal[s]["eff_n_frac"] * 100 for s in signal_names]
    x_en = np.arange(n_signals)
    w_en = 0.4
    ax_effn.bar(x_en - w_en / 2, total_ns, width=w_en, label="Total pairs", color="lightsteelblue")
    ax_effn.bar(x_en + w_en / 2, eff_ns, width=w_en, label="Effective N (Fisher-weighted)", color="steelblue")
    ax_effn.set_xticks(x_en)
    ax_effn.set_xticklabels(signal_names, rotation=25, ha="right", fontsize=8)
    ax_effn.set_ylabel("Pair count")
    ax_effn.set_title(
        f"Panel 26 — Fisher-Weighted Effective N  (β={beta:.4g})\n"
        "Σ 4σ(β·g)(1−σ(β·g))  |  near-saturation & zero-grad pairs count ≈0"
    )
    ax_effn.legend(fontsize=8)
    for i, (ef, ef_frac) in enumerate(zip(eff_ns, eff_fracs)):
        ax_effn.text(
            x_en[i] + w_en / 2,
            ef + max(total_ns) * 0.01,
            f"{ef:.0f}\n({ef_frac:.1f}%)",
            ha="center",
            fontsize=6,
            color="navy",
        )

    # ---- Panel 27: Between-shot vs Within-shot MSE Variance ----
    between_fracs = [per_signal[s]["between_frac"] * 100 for s in signal_names]
    within_fracs = [100.0 - min(f, 100.0) for f in between_fracs]
    x_av = np.arange(n_signals)
    ax_anova.bar(x_av, between_fracs, label="Between-shot (shot identity)", color="coral")
    ax_anova.bar(x_av, within_fracs, bottom=between_fracs, label="Within-shot (temporal/random)", color="steelblue")
    ax_anova.set_xticks(x_av)
    ax_anova.set_xticklabels(signal_names, rotation=25, ha="right", fontsize=8)
    ax_anova.set_ylabel("% of total MSE variance")
    ax_anova.set_ylim(0, 110)
    ax_anova.set_title(
        "Panel 27 — Between-Shot vs Within-Shot MSE Variance (one-way ANOVA)\n"
        "High between-shot → blacklisting dominant shots helps; "
        "high within-shot → distributed error"
    )
    ax_anova.legend(fontsize=8, loc="upper right")
    ax_anova.axhline(50, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    for i, bf in enumerate(between_fracs):
        ax_anova.text(i, min(bf, 108), f"{bf:.1f}%", ha="center", fontsize=7, color="darkred")

    # ---- Panel 28: Filter Survival Curve ----
    for i, sname in enumerate(signal_names):
        q = per_signal[sname]
        ax_survival.semilogx(
            q["filter_thresholds"],
            q["filter_survival"] * 100,
            color=colors[i],
            linewidth=1.8,
            label=sname,
        )
        # Mark the 5% survival point (suggested min threshold)
        survive_5pct_mask = q["filter_survival"] >= 0.05
        if survive_5pct_mask.any():
            idx_5 = np.where(survive_5pct_mask)[0][-1]
            thr_5 = float(q["filter_thresholds"][idx_5])
            ax_survival.axvline(
                thr_5,
                color=colors[i],
                linewidth=1.0,
                linestyle=":",
                alpha=0.7,
            )
            ax_survival.text(
                thr_5,
                6 + i * 5,
                f"5% cut\n{thr_5:.2e}",
                fontsize=5.5,
                color=colors[i],
                ha="left",
                rotation=90,
                va="bottom",
            )
    ax_survival.set_xlabel("min_margin_mse threshold (log scale)")
    ax_survival.set_ylabel("Pairs surviving threshold (%)")
    ax_survival.set_ylim(0, 105)
    ax_survival.set_title(
        "Panel 28 — Filter Survival Curve\n"
        "How many pairs remain at each min_margin_mse cutoff | dotted = 5% survival point"
    )
    ax_survival.legend(fontsize=7)
    ax_survival.axhline(5, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_survival.axhline(50, color="grey", linewidth=0.8, linestyle=":", alpha=0.4)

    # ---- Panel 29: β Optimisation Table ----
    ax_beta_tbl.axis("off")

    col_headers_b = ["Signal", "total N", "p50 MSE", "β_current", "β×p50", "σ(β×p50)", "β_opt (=1/p50)", "σ(β_opt×p50)"]
    col_x_b = [0.0, 0.18, 0.28, 0.39, 0.49, 0.59, 0.70, 0.85]

    y0_b, dy_b = 0.97, min(0.88 / (n_signals + 2), 0.10)

    # Header background
    ax_beta_tbl.axhspan(
        y0_b - dy_b * 0.85,
        y0_b + dy_b * 0.05,
        xmin=0,
        xmax=1,
        color="steelblue",
        transform=ax_beta_tbl.transAxes,
        clip_on=True,
    )
    for hdr, x in zip(col_headers_b, col_x_b):
        ax_beta_tbl.text(
            x + 0.005,
            y0_b,
            hdr,
            fontsize=7,
            va="top",
            ha="left",
            fontweight="bold",
            color="white",
            transform=ax_beta_tbl.transAxes,
        )

    _sig_fn = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731

    for r, sname in enumerate(signal_names):
        q = per_signal[sname]
        p50_mse = float(q["p50_mse"])
        beta_cur = float(beta)
        beta_opt = float(q["beta_opt"])
        bxp50 = beta_cur * p50_mse
        sigma_cur = float(_sig_fn(bxp50))
        sigma_opt = float(_sig_fn(1.0))  # β_opt×p50 = 1 by construction → σ(1)=0.731
        ratio = beta_cur / beta_opt if beta_opt > 0 else float("inf")

        # Colour by calibration quality: green if β is within 2× of β_opt
        if 0.5 <= ratio <= 2.0:
            row_color = "#15803d"  # green: well calibrated
        elif 0.2 <= ratio <= 5.0:
            row_color = "#92400e"  # amber: off by 2–5×
        else:
            row_color = "#991b1b"  # red: severely under/over-calibrated

        row_y = y0_b - dy_b * (r + 1)
        if r % 2 == 0:
            ax_beta_tbl.axhspan(
                row_y - dy_b * 0.85,
                row_y + dy_b * 0.05,
                xmin=0,
                xmax=1,
                color="#f0f4f8",
                transform=ax_beta_tbl.transAxes,
                clip_on=True,
            )

        cells_b = [
            sname,
            f"{q['total_n']:,}",
            f"{p50_mse:.5f}",
            f"{beta_cur:.4g}",
            f"{bxp50:.3f}",
            f"{sigma_cur:.3f}",
            f"{beta_opt:.2f}",
            f"{sigma_opt:.3f}",
        ]
        for c, (val, x) in enumerate(zip(cells_b, col_x_b)):
            txt_c = row_color if c in (3, 4, 5, 6, 7) else "#1f2328"
            ax_beta_tbl.text(
                x + 0.005,
                row_y,
                val,
                fontsize=7,
                va="top",
                ha="left",
                color=txt_c,
                transform=ax_beta_tbl.transAxes,
            )

    ax_beta_tbl.set_title(
        "Panel 29 — β Optimisation Table\n"
        "β_opt = 1/p50_MSE places σ(β×p50) at ideal 0.731  |  "
        "colour: green=within 2×, amber=2–5×, red=>5×",
        pad=6,
        fontsize=8,
    )

    return fig


# ======================================================================================================================
# CLI
# ======================================================================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize GPO preference-pair dataset statistics.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--gpo_dir",
        nargs="+",
        required=True,
        metavar="DIR",
        help="One or more GPO pairs directories (each must contain .npz shards and collection_config.json).",
    )
    parser.add_argument(
        "--signal",
        type=str,
        default=None,
        help="Only plot statistics for this signal name (default: all signals).",
    )
    parser.add_argument(
        "--max_windows",
        type=int,
        default=50_000,
        help=(
            "Maximum windows to load per signal when computing histograms (default: 50000). "
            "Set 0 to load all windows (may be slow for large datasets)."
        ),
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "When multiple --gpo_dir paths are given, produce an additional overlay "
            "comparison figure (margin, cosine, normalised margin, MSE gap across all directories)."
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save figures as PNG files (default: display interactively).",
    )
    parser.add_argument(
        "--no_extended",
        action="store_true",
        help=("Skip Figure 2 (extended diagnostics).  Useful for quick runs or schema-v1 datasets."),
    )
    parser.add_argument(
        "--no_actionability",
        action="store_true",
        help="Skip Figure 3 (β-effectiveness, pair quality, shot outlier table).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help=(
            "Configured β value to use when simulating σ(β × MSE_gap) in panel 21 "
            "(Figure 3).  Default: 0.1 if not provided."
        ),
    )
    parser.add_argument(
        "--no_quality",
        action="store_true",
        help=(
            "Skip Figure 4 (training-signal quality: Lorenz curve, signed bias, "
            "Fisher-weighted effective N, ANOVA variance, filter survival, β table)."
        ),
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not call plt.show() (useful for headless/scripted runs).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Figure DPI (default: 120).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        import matplotlib

        if args.no_show or args.save_dir:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.error("matplotlib is required.  Install it with: pip install matplotlib")
        sys.exit(1)

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Per-directory figures
    # ------------------------------------------------------------------
    compare_data: list[tuple[Path, dict, dict, dict]] = []
    report_data: list[tuple[Path, dict, dict, dict]] = []

    for raw_dir in args.gpo_dir:
        gpo_dir = Path(raw_dir)
        if not gpo_dir.is_dir():
            log.error("Directory not found: %s — skipping.", gpo_dir)
            continue

        log.info("=" * 60)
        log.info("Processing: %s", gpo_dir)

        try:
            fig = _make_figure(
                gpo_dir=gpo_dir,
                signal_filter=args.signal,
                max_windows=args.max_windows,
                dpi=args.dpi,
            )
        except Exception as exc:
            log.error("Failed to process %s: %s", gpo_dir, exc)
            continue

        # Build a unique stem: <run_id>__<pairs_tag>
        # e.g. ft-task_4-1-scratch-mmt-embed_mse__gpo_pairs_v1
        file_stem = f"{gpo_dir.parent.name}__{gpo_dir.name}"

        if save_dir:
            out_path = save_dir / f"gpo_stats__{file_stem}.png"
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            log.info("Saved: %s", out_path)
            plt.close(fig)

        # ------------------------------------------------------------------
        # Figure 2: extended diagnostics (panels 13–20, schema v2/v3 only)
        # ------------------------------------------------------------------
        if not args.no_extended:
            try:
                fig_ext = _make_figure_extended(
                    gpo_dir=gpo_dir,
                    signal_filter=args.signal,
                    max_windows=args.max_windows,
                    dpi=args.dpi,
                )
            except ValueError as exc:
                # Schema v1 datasets silently skip the extended figure.
                log.warning("Extended diagnostics skipped for %s: %s", gpo_dir, exc)
                fig_ext = None
            except Exception as exc:
                log.error("Extended diagnostics failed for %s: %s", gpo_dir, exc)
                fig_ext = None

            if fig_ext is not None and save_dir:
                out_path_ext = save_dir / f"gpo_extended__{file_stem}.png"
                fig_ext.savefig(out_path_ext, dpi=args.dpi, bbox_inches="tight")
                log.info("Saved extended figure: %s", out_path_ext)
                plt.close(fig_ext)

        # ------------------------------------------------------------------
        # Figure 3: actionability diagnostics (panels 21–23, schema v2/v3 only)
        # ------------------------------------------------------------------
        if not args.no_actionability:
            try:
                fig_act = _make_figure_actionability(
                    gpo_dir=gpo_dir,
                    signal_filter=args.signal,
                    max_windows=args.max_windows,
                    dpi=args.dpi,
                    beta_override=args.beta,
                )
            except ValueError as exc:
                log.warning("Actionability diagnostics skipped for %s: %s", gpo_dir, exc)
                fig_act = None
            except Exception as exc:
                log.error("Actionability diagnostics failed for %s: %s", gpo_dir, exc)
                fig_act = None

            if fig_act is not None and save_dir:
                out_path_act = save_dir / f"gpo_actionability__{file_stem}.png"
                fig_act.savefig(out_path_act, dpi=args.dpi, bbox_inches="tight")
                log.info("Saved actionability figure: %s", out_path_act)
                plt.close(fig_act)

        # ------------------------------------------------------------------
        # Figure 4: training-signal quality (panels 24–29, schema v2/v3 only)
        # ------------------------------------------------------------------
        if not args.no_quality:
            try:
                fig_qual = _make_figure_quality(
                    gpo_dir=gpo_dir,
                    signal_filter=args.signal,
                    max_windows=args.max_windows,
                    dpi=args.dpi,
                    beta_override=args.beta,
                )
            except ValueError as exc:
                log.warning("Quality diagnostics skipped for %s: %s", gpo_dir, exc)
                fig_qual = None
            except Exception as exc:
                log.error("Quality diagnostics failed for %s: %s", gpo_dir, exc)
                fig_qual = None

            if fig_qual is not None and save_dir:
                out_path_qual = save_dir / f"gpo_quality__{file_stem}.png"
                fig_qual.savefig(out_path_qual, dpi=args.dpi, bbox_inches="tight")
                log.info("Saved quality figure: %s", out_path_qual)
                plt.close(fig_qual)

        # Collect data for the report and optional comparison figures.
        # Always done so the all-tasks report is available without --compare.
        _meta: dict[str, Any] = {}
        _cc: dict[str, Any] = {}
        _meta_path = gpo_dir / "metadata.json"
        _cc_path = gpo_dir / "collection_config.json"
        if _meta_path.is_file():
            _meta = _load_json(_meta_path)
        if _cc_path.is_file():
            _cc = _load_json(_cc_path)

        _schema_version = int(_meta.get("schema_version", _cc.get("schema_version", 1)))
        _all_signals = _discover_signals(gpo_dir)
        if args.signal is not None:
            _safe_filter = args.signal.replace("/", "-").replace("\\", "-")
            _all_signals = {k: v for k, v in _all_signals.items() if k == _safe_filter}

        _per_signal: dict[str, dict[str, Any]] = {}
        for _sname, _shards in _all_signals.items():
            _arrays = _load_arrays(_shards, schema_version=_schema_version, max_windows=args.max_windows)
            if _arrays:
                _per_signal[_sname] = _compute_signal_stats(_arrays)

        if _per_signal:
            report_data.append((gpo_dir, _meta, _cc, _per_signal))
            if args.compare:
                compare_data.append((gpo_dir, _meta, _cc, _per_signal))

    # ------------------------------------------------------------------
    # All-tasks summary report (always produced when ≥2 dirs succeed)
    # ------------------------------------------------------------------
    if len(report_data) >= 2:
        log.info("=" * 60)
        log.info("Building all-tasks summary report for %d directories…", len(report_data))
        beta_report = float(args.beta) if args.beta is not None else 0.1
        try:
            fig_report = _make_report_figure(
                report_data=report_data,
                beta=beta_report,
                dpi=args.dpi,
            )
        except Exception as exc:
            log.error("Failed to build report figure: %s", exc)
        else:
            if save_dir:
                out_report = save_dir / "gpo_report__all_tasks.png"
                fig_report.savefig(out_report, dpi=args.dpi, bbox_inches="tight")
                log.info("Saved all-tasks report: %s", out_report)
                plt.close(fig_report)

    # ------------------------------------------------------------------
    # Optional cross-directory comparison figure
    # ------------------------------------------------------------------
    if args.compare and len(compare_data) >= 2:
        log.info("=" * 60)
        log.info("Building comparison figure for %d directories…", len(compare_data))
        try:
            fig_cmp = _make_compare_figure(
                dirs_data=compare_data,
                max_windows=args.max_windows,
                dpi=args.dpi,
            )
        except Exception as exc:
            log.error("Failed to build comparison figure: %s", exc)
        else:
            if save_dir:
                dir_names = "__vs__".join(d[0].name for d in compare_data)
                out_path = save_dir / f"gpo_compare__{dir_names}.png"
                fig_cmp.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
                log.info("Saved comparison figure: %s", out_path)
                plt.close(fig_cmp)
    elif args.compare and len(compare_data) < 2:
        log.warning("--compare requires at least 2 successfully loaded directories (got %d).", len(compare_data))

    if not args.no_show and not save_dir:
        plt.show()


# ======================================================================================================================
if __name__ == "__main__":
    main()
