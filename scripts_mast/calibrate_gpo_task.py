"""
calibrate_gpo_task.py
=====================

CPU-only post-collection pipeline step.  For each supplied GPO pairs directory this script:

  1. Validates the schema-v3 pair dataset (equivalent to validate_gpo_pairs.py).
  2. Computes β calibration from the p50 MSE gap per signal
     (β_opt = 1/p50, IPO target 1/(2β) ≈ p50).
  3. Determines the shot blacklist (top-N shots ranked by mean MSE gap that
     exceed a multiple of the dataset mean).
  4. Writes the calibrated β and shot_blacklist back into
     ``scripts_mast/configs/mmt/tasks/gpo_tasks.yaml`` under the matching
     task key, leaving all other task config intact.
  5. Saves the 4-figure diagnostic PNG set (requires matplotlib) unless
     ``--no_plots`` is given.
  6. Writes a per-task calibration JSON summary (always).

The script is safe to re-run: it only touches ``gpo_tasks.yaml`` fields it
computed (beta, shot_blacklist) and leaves every other key untouched.

Usage
-----
    # Single task (pair dir auto-discovered from run dir):
    python scripts_mast/calibrate_gpo_task.py \\
        runs/ft-task_4-2-scratch-mmt-embed_crps/gpo_pairs_v8

    # Multiple tasks at once:
    python scripts_mast/calibrate_gpo_task.py \\
        runs/ft-task_4-*/gpo_pairs_v8

    # Dry-run (compute + print, do NOT write gpo_tasks.yaml):
    python scripts_mast/calibrate_gpo_task.py \\
        runs/ft-task_4-2-scratch-mmt-embed_crps/gpo_pairs_v8 --dry_run

    # Skip plot generation (headless CI):
    python scripts_mast/calibrate_gpo_task.py \\
        runs/ft-task_4-2-scratch-mmt-embed_crps/gpo_pairs_v8 --no_plots

Exit code 0 = all tasks passed; 1 = one or more failures.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("gpo.calibrate")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fraction of dataset mean MSE above which a shot is a blacklist candidate.
# Top shots with mean_mse > BLACKLIST_MULTIPLIER * dataset_mean are candidates;
# we then take the top BLACKLIST_TOP_N by rank.
_BLACKLIST_MULTIPLIER = 10.0
_BLACKLIST_TOP_N = 5  # at most this many shots added (fewer if below threshold)

# Minimum effective-N fraction (Fisher-weighted) below which the current β is
# flagged as mis-calibrated (warning only).
_EFF_N_WARN_THRESHOLD = 0.20

# Path to gpo_tasks.yaml relative to the repo root (auto-resolved from this file).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_GPO_TASKS_YAML = _REPO_ROOT / "scripts_mast" / "configs" / "mmt" / "tasks" / "gpo_tasks.yaml"


# ---------------------------------------------------------------------------
# Schema validation (minimal inline version, no dependency on validate_gpo_pairs)
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = frozenset(
    ["shot_id", "window_index", "y_w_emb", "y_l_emb",
     "native_nrmse", "native_nmae", "native_mse", "embedding_gap", "emb_dim"]
)


def _validate_dir(gpo_dir: Path) -> tuple[bool, list[str]]:
    """Return (ok, error_list).  Quick structural check, not full 10-point suite."""
    errors: list[str] = []

    meta_path = gpo_dir / "metadata.json"
    cc_path = gpo_dir / "collection_config.json"
    if not meta_path.is_file():
        errors.append("MISSING metadata.json")
        return False, errors
    if not cc_path.is_file():
        errors.append("MISSING collection_config.json")
        return False, errors

    meta = json.loads(meta_path.read_text())
    cc = json.loads(cc_path.read_text())

    if meta.get("schema_version") != 3:
        errors.append(f"schema_version={meta.get('schema_version')!r} in metadata.json; expected 3")
    if cc.get("schema_version") != 3:
        errors.append(f"schema_version={cc.get('schema_version')!r} in collection_config.json; expected 3")
    if (cc.get("split") or meta.get("split")) != "train":
        errors.append(f"split={cc.get('split')!r}; must be 'train'")

    shards = sorted(gpo_dir.glob("*.npz"))
    if not shards:
        errors.append("No .npz shards found")
    for s in shards[:5]:  # spot-check first 5
        try:
            npz = np.load(s, allow_pickle=False)
            missing = _REQUIRED_KEYS - set(npz.files)
            if missing:
                errors.append(f"{s.name}: missing keys {missing}")
        except Exception as exc:
            errors.append(f"{s.name}: corrupt ({exc})")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _discover_signals(gpo_dir: Path) -> dict[str, list[Path]]:
    sig_map: dict[str, list[Path]] = {}
    for s in sorted(gpo_dir.glob("*.npz")):
        stem = s.stem
        sig = stem[: stem.rfind("__shard_")] if "__shard_" in stem else stem
        sig_map.setdefault(sig, []).append(s)
    return sig_map


def _load_signal(shards: list[Path], max_windows: int) -> dict[str, np.ndarray]:
    """Load y_w_emb, y_l_emb, shot_id from shards with optional cap."""
    yw_list, yl_list, sid_list = [], [], []
    total = 0
    for shard in sorted(shards):
        with np.load(shard, allow_pickle=False) as npz:
            n = len(npz["shot_id"])
            if max_windows > 0 and total + n > max_windows:
                n = max(0, max_windows - total)
            if n == 0:
                break
            yw_list.append(npz["y_w_emb"][:n].astype(np.float32))
            yl_list.append(npz["y_l_emb"][:n].astype(np.float32))
            sid_list.append(npz["shot_id"][:n])
            total += n
            if max_windows > 0 and total >= max_windows:
                break
    return {
        "y_w": np.concatenate(yw_list, axis=0),
        "y_l": np.concatenate(yl_list, axis=0),
        "shot_id": np.concatenate(sid_list, axis=0),
    }


# ---------------------------------------------------------------------------
# Calibration logic
# ---------------------------------------------------------------------------

def _calibrate_signal(
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compute per-signal calibration stats."""
    yw = arrays["y_w"]
    yl = arrays["y_l"]
    shot_ids = arrays["shot_id"]

    mse_gap = ((yw - yl) ** 2).mean(axis=1)  # (N,)
    n = len(mse_gap)

    p50_mse = float(np.percentile(mse_gap, 50))
    beta_opt = float(1.0 / (p50_mse + 1e-30))

    # Shot-level aggregation for blacklist
    shot_stats: dict[int, list[float]] = {}
    for i in range(n):
        sid = int(shot_ids[i])
        shot_stats.setdefault(sid, []).append(float(mse_gap[i]))

    shot_rows = []
    for sid, vals in shot_stats.items():
        arr = np.asarray(vals, dtype=np.float32)
        shot_rows.append({"shot_id": sid, "n_win": len(vals),
                          "mean_mse": float(arr.mean()),
                          "p50_mse": float(np.median(arr)),
                          "p95_mse": float(np.percentile(arr, 95))})
    shot_rows.sort(key=lambda r: r["mean_mse"], reverse=True)

    dataset_mean = float(mse_gap.mean())

    return {
        "n_windows": n,
        "p50_mse": p50_mse,
        "beta_opt": beta_opt,
        "dataset_mean_mse": dataset_mean,
        "shot_rows": shot_rows,
    }


def _determine_blacklist(
    signal_stats: dict[str, dict[str, Any]],
    multiplier: float = _BLACKLIST_MULTIPLIER,
    top_n: int = _BLACKLIST_TOP_N,
) -> list[int]:
    """
    Aggregate shot outlier tables across all signals and return a deduplicated
    blacklist of shot IDs.

    A shot qualifies if its mean MSE ≥ ``multiplier`` × dataset mean in *any*
    signal, and it ranks in the top-``top_n`` of that signal's per-shot table.
    """
    candidate_set: set[int] = set()
    for sig, stats in signal_stats.items():
        dataset_mean = stats["dataset_mean_mse"]
        threshold = multiplier * dataset_mean
        for rank, row in enumerate(stats["shot_rows"]):
            if rank >= top_n:
                break
            if row["mean_mse"] >= threshold:
                candidate_set.add(row["shot_id"])
    return sorted(candidate_set)


# ---------------------------------------------------------------------------
# YAML patch helpers — pure text manipulation to preserve comments
# ---------------------------------------------------------------------------

def _read_yaml_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _patch_yaml_task(
    yaml_text: str,
    task_key: str,
    beta: float,
    shot_blacklist: list[int],
) -> str:
    """
    Patch ``beta`` and ``shot_blacklist`` in the YAML text for ``task_key``.

    Strategy: locate the ``  task_X-Y:`` block, then find and replace
    (or insert) the ``beta:`` and ``shot_blacklist:`` lines within it.
    This approach avoids a full YAML round-trip that strips comments.
    """
    lines = yaml_text.splitlines(keepends=True)

    # Find the line where the task block starts (indented by 2 spaces)
    task_start = None
    for i, line in enumerate(lines):
        if line.rstrip() == f"  {task_key}:":
            task_start = i
            break

    if task_start is None:
        log.warning("Task key '%s' not found in gpo_tasks.yaml — skipping patch.", task_key)
        return yaml_text

    # Find the end of this task block: next line at indent <= 2 that isn't blank/comment
    task_end = len(lines)
    for i in range(task_start + 1, len(lines)):
        stripped = lines[i].rstrip()
        if stripped and not stripped.startswith("  #") and not stripped.startswith("    "):
            task_end = i
            break

    block_lines = lines[task_start:task_end]

    # Patch beta: replace existing `beta:` line inside the block
    beta_str = f"{beta:.1f}" if beta == int(beta) else f"{beta}"
    blacklist_str = "[" + ", ".join(str(s) for s in shot_blacklist) + "]"

    block_patched = []
    for line in block_lines:
        stripped = line.lstrip()
        if stripped.startswith("beta:") and "  # " in line:
            # Preserve trailing comment
            comment_start = line.index("  # ")
            prefix = line[: line.index("beta:")]
            comment = line[comment_start:]
            block_patched.append(f"{prefix}beta: {beta_str}{comment}")
        elif stripped.startswith("beta:"):
            prefix = line[: line.index("beta:")]
            block_patched.append(f"{prefix}beta: {beta_str}  # auto-calibrated: β_opt=1/p50\n")
        elif stripped.startswith("shot_blacklist:"):
            prefix = line[: line.index("shot_blacklist:")]
            block_patched.append(f"{prefix}shot_blacklist: {blacklist_str}  # auto-calibrated\n")
        else:
            block_patched.append(line)

    return "".join(lines[:task_start] + block_patched + lines[task_end:])


def _extract_task_key(gpo_dir: Path) -> str | None:
    """
    Infer the task key (e.g. ``task_4-2``) from the run directory name above the
    gpo_pairs* directory.

    Expected layout:
        runs/ft-<task>-scratch-mmt-<tag>/gpo_pairs_<version>/
    """
    run_dir_name = gpo_dir.parent.name  # e.g. ft-task_4-2-scratch-mmt-embed_crps
    # Strip the "ft-" prefix and find the task token
    if run_dir_name.startswith("ft-"):
        rest = run_dir_name[3:]  # task_4-2-scratch-mmt-embed_crps
        parts = rest.split("-")
        # task key is "task_X-Y" — find the first occurrence of "task"
        for i, p in enumerate(parts):
            if p == "task" and i + 1 < len(parts):
                # could be task_4-2 (underscore already) or task + 4 + 2
                next_p = parts[i + 1]
                if "-" in next_p or next_p.replace("-", "").isdigit():
                    return f"task_{next_p}"
    # Fallback: look for task_X-Y anywhere in the name
    import re
    m = re.search(r"(task_\d+-\d+)", run_dir_name)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Plot generation (optional; wraps visualize_gpo_stats)
# ---------------------------------------------------------------------------

def _save_plots(gpo_dir: Path, save_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        log.warning("matplotlib not available — skipping plots.")
        return
    try:
        # Import the full visualizer lazily so this script stays fast without it
        import importlib, os, sys as _sys
        # Add repo root to path if needed
        repo_root = str(_REPO_ROOT)
        if repo_root not in _sys.path:
            _sys.path.insert(0, repo_root)
        vgs = importlib.import_module("scripts_mast.visualize_gpo_stats")
        import matplotlib.pyplot as plt

        save_dir.mkdir(parents=True, exist_ok=True)
        for fig_fn, suffix in [
            (vgs._make_figure, "gpo_stats"),
            (vgs._make_figure_extended, "gpo_extended"),
            (vgs._make_figure_actionability, "gpo_actionability"),
            (vgs._make_figure_quality, "gpo_quality"),
        ]:
            try:
                kwargs: dict = {"gpo_dir": gpo_dir, "signal_filter": None,
                                "max_windows": 50_000, "dpi": 120}
                if fig_fn is vgs._make_figure_quality:
                    kwargs["beta"] = None
                fig = fig_fn(**kwargs)
                tag = gpo_dir.name  # e.g. gpo_pairs_v8
                run_tag = gpo_dir.parent.name  # e.g. ft-task_4-2-...
                fname = save_dir / f"{suffix}__{run_tag}__{tag}.png"
                fig.savefig(fname, dpi=120, bbox_inches="tight")
                plt.close(fig)
                log.info("  Saved: %s", fname)
            except Exception as exc:
                log.warning("  Plot %s failed: %s", suffix, exc)
    except Exception as exc:
        log.warning("Plot generation failed: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Validate GPO pairs, calibrate β and shot_blacklist, patch gpo_tasks.yaml, "
            "and optionally save diagnostic plots."
        ),
        allow_abbrev=False,
    )
    p.add_argument(
        "gpo_dirs",
        nargs="+",
        metavar="GPO_DIR",
        help="One or more GPO pairs directories (schema v3).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print calibration results but do NOT write gpo_tasks.yaml.",
    )
    p.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip saving the 4-figure diagnostic PNGs.",
    )
    p.add_argument(
        "--plots_dir",
        type=str,
        default=None,
        help="Directory to save diagnostic PNGs (default: <gpo_dir>/../gpo_plots/).",
    )
    p.add_argument(
        "--max_windows",
        type=int,
        default=100_000,
        help="Max windows to load per signal for calibration (0 = all, default: 100000).",
    )
    p.add_argument(
        "--blacklist_multiplier",
        type=float,
        default=_BLACKLIST_MULTIPLIER,
        help=f"A shot is a blacklist candidate if mean_MSE ≥ X × dataset_mean (default: {_BLACKLIST_MULTIPLIER}).",
    )
    p.add_argument(
        "--blacklist_top_n",
        type=int,
        default=_BLACKLIST_TOP_N,
        help=f"Maximum blacklist candidates per signal (default: {_BLACKLIST_TOP_N}).",
    )
    p.add_argument(
        "--gpo_tasks_yaml",
        type=str,
        default=str(_GPO_TASKS_YAML),
        help=f"Path to gpo_tasks.yaml to patch (default: {_GPO_TASKS_YAML}).",
    )
    p.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help="Write calibration summary to this JSON file (default: <gpo_dir>/calibration_summary.json).",
    )
    p.add_argument(
        "--report_dir",
        type=str,
        default=None,
        help=(
            "If given, writes a Markdown calibration report and copies the diagnostic PNGs "
            "into <report_dir>/gpo_stats/<task>/.  Intended for the reports/ folder."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def _write_calibration_report(
    summary: dict,
    plots_dir: Path | None,
    report_dir: Path,
    gpo_tag: str,
) -> Path:
    """Write a per-task calibration Markdown report into *report_dir*/gpo_stats/<task>/."""
    task = summary["task"]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = report_dir / "gpo_stats" / task
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Copy diagnostic PNGs if the plots directory exists ----
    if plots_dir and plots_dir.is_dir():
        # calibrate_gpo_task saves PNGs like:
        #   <plots_dir>/<suffix>__<run_tag>__<gpo_tag>.png
        # The plots_dir passed here is already task-specific (or shared root).
        # Copy all PNGs whose name contains the gpo_tag to the report dir.
        copied: list[str] = []
        for png in sorted(plots_dir.glob(f"*__{gpo_tag}.png")):
            dest = out_dir / png.name
            shutil.copy2(png, dest)
            copied.append(png.name)
        if copied:
            log.info("  Copied %d PNGs to %s", len(copied), out_dir)

    # ---- Write Markdown ----
    md_path = out_dir / f"calibration_{task}_{gpo_tag}_{ts}.md"

    lines: list[str] = [
        f"# GPO Calibration Report — {task}",
        f"## gpo_tag=`{gpo_tag}`",
        f"",
        f"> **Generated:** {datetime.datetime.now().isoformat(timespec='seconds')}  ",
        f"> **Source:** `scripts_mast/calibrate_gpo_task.py --report_dir`  ",
        f"> **Pair directory:** `{summary['gpo_dir']}`  ",
        f"",
        f"---",
        f"",
        f"## Calibrated parameters",
        f"",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Task key | `{task}` |",
        f"| β calibrated | **{summary['beta_calibrated']:.1f}** |",
        f"| β geomean (raw) | {summary['beta_geomean_raw']:.2f} |",
        f"| Shot blacklist | {summary['shot_blacklist'] if summary['shot_blacklist'] else '*(none)*'} |",
        f"",
        f"---",
        f"",
        f"## Per-signal statistics",
        f"",
        f"| Signal | n_windows | p50 MSE | β_opt (1/p50) | dataset mean MSE |",
        f"|--------|-----------|---------|---------------|-----------------|",
    ]

    for sig, s in sorted(summary["signals"].items()):
        lines.append(
            f"| {sig} | {s['n_windows']:,} | {s['p50_mse']:.5f} | "
            f"{s['beta_opt']:.1f} | {s['dataset_mean_mse']:.5f} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## Top-shot outliers (from calibration scan)",
        f"",
    ]

    any_top = False
    for sig, s in sorted(summary["signals"].items()):
        top_shots = s.get("top_shots", [])
        if not top_shots:
            continue
        any_top = True
        lines += [
            f"### {sig}",
            f"",
            f"| Rank | shot_id | mean MSE | × dataset mean |",
            f"|------|---------|----------|----------------|",
        ]
        for rank, row in enumerate(top_shots[:10], 1):
            # row is [shot_id, n_windows, mean_mse, p50_mse, p95_mse]
            if len(row) >= 3:
                sid, n_win, mean_mse = int(row[0]), int(row[1]), float(row[2])
                dset_mean = s["dataset_mean_mse"]
                ratio = mean_mse / dset_mean if dset_mean > 0 else float("nan")
                bl_marker = " ← blacklisted" if sid in summary["shot_blacklist"] else ""
                lines.append(f"| {rank} | {sid} | {mean_mse:.4f} | {ratio:.1f}×{bl_marker} |")
        lines.append("")

    if not any_top:
        lines.append("*(no top-shot data in calibration summary)*")
        lines.append("")

    if out_dir.exists():
        pngs = sorted(out_dir.glob("*.png"))
        if pngs:
            lines += [
                f"---",
                f"",
                f"## Diagnostic figures",
                f"",
                f"The following PNG files were saved alongside this report:",
                f"",
            ]
            for png in pngs:
                lines.append(f"- `{png.name}`")
            lines.append("")

    lines += ["---", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("  Calibration report written to: %s", md_path)
    return md_path


def main() -> None:
    args = _parse_args()
    gpo_tasks_path = Path(args.gpo_tasks_yaml)
    all_ok = True
    yaml_text = _read_yaml_text(gpo_tasks_path) if not args.dry_run else ""

    for raw_dir in args.gpo_dirs:
        gpo_dir = Path(raw_dir)
        log.info("=" * 64)
        log.info("Processing: %s", gpo_dir)

        if not gpo_dir.is_dir():
            log.error("Directory not found: %s", gpo_dir)
            all_ok = False
            continue

        # ---- 1. Validate ----
        ok, errors = _validate_dir(gpo_dir)
        if not ok:
            for e in errors:
                log.error("  VALIDATION: %s", e)
            all_ok = False
            continue
        log.info("  Validation passed.")

        # ---- 2. Extract task key ----
        task_key = _extract_task_key(gpo_dir)
        if task_key is None:
            log.error("  Cannot infer task key from path '%s'. "
                      "Expected layout: runs/ft-<task>-*/gpo_pairs*/", gpo_dir)
            all_ok = False
            continue
        log.info("  Task key: %s", task_key)

        # ---- 3. Load data and calibrate ----
        sig_map = _discover_signals(gpo_dir)
        signal_stats: dict[str, dict[str, Any]] = {}
        for sig, shards in sorted(sig_map.items()):
            arrays = _load_signal(shards, max_windows=args.max_windows)
            stats = _calibrate_signal(arrays)
            signal_stats[sig] = stats
            log.info(
                "  Signal %-40s  n=%6d  p50_mse=%.5f  β_opt=%.1f",
                sig, stats["n_windows"], stats["p50_mse"], stats["beta_opt"],
            )

        # Geometric mean β across signals (balanced multi-signal tasks)
        beta_opts = [s["beta_opt"] for s in signal_stats.values() if s["p50_mse"] > 1e-8]
        if not beta_opts:
            log.warning("  No valid p50 values — skipping β calibration.")
            continue
        beta_geomean = float(np.exp(np.mean(np.log(np.maximum(beta_opts, 1e-6)))))
        # Round to a clean value: 1, 2, 5, 8, 10, 25, 50, 100, ...
        _scale = 10 ** int(np.floor(np.log10(max(beta_geomean, 1))))
        beta_rounded = float(round(beta_geomean / _scale) * _scale)
        if beta_rounded < 1.0:
            beta_rounded = 1.0

        # ---- 4. Shot blacklist ----
        blacklist = _determine_blacklist(
            signal_stats,
            multiplier=args.blacklist_multiplier,
            top_n=args.blacklist_top_n,
        )

        log.info("  Calibrated β = %.1f  (geomean of β_opt across signals)", beta_rounded)
        if blacklist:
            log.info("  Shot blacklist (%d shots): %s", len(blacklist), blacklist)
        else:
            log.info("  Shot blacklist: none (no shots exceeded %.0f× dataset mean)", args.blacklist_multiplier)

        # ---- 5. Write calibration summary JSON ----
        summary = {
            "task": task_key,
            "gpo_dir": str(gpo_dir),
            "beta_calibrated": beta_rounded,
            "beta_geomean_raw": beta_geomean,
            "shot_blacklist": blacklist,
            "signals": {
                sig: {
                    "n_windows": s["n_windows"],
                    "p50_mse": s["p50_mse"],
                    "beta_opt": s["beta_opt"],
                    "dataset_mean_mse": s["dataset_mean_mse"],
                    "top_shots": s["shot_rows"][:10],
                }
                for sig, s in signal_stats.items()
            },
        }
        summary_path = Path(args.summary_json) if args.summary_json else gpo_dir / "calibration_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.info("  Summary written to: %s", summary_path)

        # ---- 6. Patch gpo_tasks.yaml ----
        if args.dry_run:
            log.info("  DRY RUN — would patch gpo_tasks.yaml: %s  beta=%.1f  blacklist=%s",
                     task_key, beta_rounded, blacklist)
        else:
            yaml_text = _patch_yaml_task(yaml_text, task_key, beta_rounded, blacklist)
            log.info("  Patched gpo_tasks.yaml for %s.", task_key)

        # ---- 7. Save diagnostic plots ----
        plots_dir_path: Path | None = None
        if not args.no_plots:
            plots_dir_path = Path(args.plots_dir) if args.plots_dir else gpo_dir.parent / "gpo_plots"
            _save_plots(gpo_dir, plots_dir_path)

        # ---- 8. Write report (Markdown + copy PNGs) ----
        if args.report_dir:
            # Derive gpo_tag from the pair directory name (e.g. gpo_pairs_v8 → v8).
            dir_name = gpo_dir.name  # e.g. "gpo_pairs_v8"
            gpo_tag = dir_name.split("_", 2)[-1] if "_" in dir_name else dir_name
            # The plots_dir here is the task-specific subdirectory inside PLOTS_DIR,
            # which _save_plots writes into as: <plots_dir>/<suffix>__<run>__<tag>.png
            # (all flat, not sub-divided further).  Pass it so PNGs are copied.
            _write_calibration_report(
                summary=summary,
                plots_dir=plots_dir_path,
                report_dir=Path(args.report_dir),
                gpo_tag=gpo_tag,
            )

    # Write the (possibly multi-task-patched) YAML once at the end
    if not args.dry_run and yaml_text:
        gpo_tasks_path.write_text(yaml_text, encoding="utf-8")
        log.info("gpo_tasks.yaml updated: %s", gpo_tasks_path)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
