"""
calibrate_gpo_task.py — Immutable, run-specific CGPO calibration from training shots only.

Reads the same phase/task/local configuration stack as CGPO training and writes one dedicated task YAML plus
calibration_summary.json under <output_dir>/<task>/. The shared input recipe is never edited. Existing output
artifacts are refused. --dry_run computes and reports the plan without writing summaries, plots or reports.

Beta = 1 / median(collection gap), geometrically aggregated across signals, is a scale heuristic. It does not
estimate reference-relative margins: policy and reference initially coincide, so those margins start at zero.
For IPO the target is 1/(2 beta), i.e. half the representative gap under this heuristic. MSE-only recipes remain
MSE-only. Source/pair hashes, recipe hashes, calibration settings and the target run tag are saved for verification.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import shutil
import os
import re

import yaml
from mast_utils.gpo.protocol import config_fingerprint, digest, file_digest, source_fingerprint, validate_collection
from mmt.utils.config.experiment.merge import deep_merge, load_yaml, _load_task_block
from mmt.utils.config.validator import _validate_loss_terms
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

# Path to gpo_tasks.yaml relative to the repo root (auto-resolved from this file).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_GPO_TASKS_YAML = _REPO_ROOT / "scripts_mast" / "configs" / "mmt" / "tasks" / "gpo_tasks.yaml"


# ---------------------------------------------------------------------------
# Schema validation (minimal inline version, no dependency on validate_gpo_pairs)
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = frozenset(
    [
        "shot_id",
        "window_index",
        "y_w_emb",
        "y_l_emb",
        "native_nrmse",
        "native_nmae",
        "native_mse",
        "embedding_gap",
        "emb_dim",
    ]
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
    if (cc.get("split") or meta.get("split")) != "train" and cc.get("protocol") != "mast-shot-split-v1":
        errors.append(f"split={cc.get('split')!r}; must be 'train'")

    if cc.get("protocol") == "mast-shot-split-v1":
        from mast_utils.gpo.protocol import validate_collection

        try:
            validate_collection(cc, cc["contract"], gpo_dir)
        except (KeyError, ValueError) as exc:
            errors.append(f"PROTOCOL: {exc}")

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


def _load_signal(shards: list[Path], max_windows: int, train_shots: set[int] | None = None) -> dict[str, np.ndarray]:
    """Load y_w_emb, y_l_emb, shot_id from shards with optional cap."""
    yw_list, yl_list, sid_list = [], [], []
    total = 0
    for shard in sorted(shards):
        with np.load(shard, allow_pickle=False) as npz:
            rows = (
                np.flatnonzero(np.isin(npz["shot_id"], list(train_shots)))
                if train_shots is not None
                else np.arange(len(npz["shot_id"]))
            )
            n = len(rows)
            if max_windows > 0 and total + n > max_windows:
                n = max(0, max_windows - total)
            if n == 0:
                continue
            yw_list.append(npz["y_w_emb"][rows[:n]].astype(np.float32))
            yl_list.append(npz["y_l_emb"][rows[:n]].astype(np.float32))
            sid_list.append(npz["shot_id"][rows[:n]])
            total += n
            if max_windows > 0 and total >= max_windows:
                break
    return {
        "y_w": np.concatenate(yw_list, axis=0) if yw_list else np.empty((0, 0)),
        "y_l": np.concatenate(yl_list, axis=0) if yl_list else np.empty((0, 0)),
        "shot_id": np.concatenate(sid_list, axis=0) if sid_list else np.empty(0, dtype=np.int64),
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

    if n == 0 or not np.isfinite(mse_gap).all():
        raise ValueError("Calibration requires finite, nonempty training-pair gaps for every signal.")
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
        shot_rows.append(
            {
                "shot_id": sid,
                "n_win": len(vals),
                "mean_mse": float(arr.mean()),
                "p50_mse": float(np.median(arr)),
                "p95_mse": float(np.percentile(arr, 95)),
            }
        )
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


# ----------------------------------------------------------------------------------------------------------------------
def _effective_recipe(task: str, input_path: Path, configs_root: Path) -> dict:
    """
    Merge phase, selected task and local overrides without creating a run directory.

    Parameters
    ----------
    task : str
        Collection task identifier.
    input_path, configs_root : pathlib.Path
        Input task recipe and configuration root, also used by training.

    Returns
    -------
    dict
        Effective recipe to freeze and calibrate. Missing task entries inherit phase defaults explicitly.
    """
    phase = load_yaml(configs_root / "mmt/phases/gpo.yaml")
    overrides = _load_task_block(path=input_path, task=task, phase="gpo")
    if not overrides:
        log.info("Task %s has no input overrides; freezing the GPO phase defaults explicitly.", task)
    recipe = deep_merge(base=phase, override=overrides or {})
    local = configs_root / "local_overrides.yaml"
    if local.is_file():
        recipe = deep_merge(base=recipe, override=load_yaml(local))
    recipe.pop("run_id", None)
    _validate_loss_terms({"train": recipe["train"]})
    return recipe


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
        import importlib
        import sys as _sys

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
                kwargs: dict = {"gpo_dir": gpo_dir, "signal_filter": None, "max_windows": 50_000, "dpi": 120}
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
            "Validate GPO pairs, calibrate β and shot_blacklist into a dedicated run artifact, "
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
        help="Compute calibration without any file or directory writes.",
    )
    p.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip saving the 4-figure diagnostic PNGs.",
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
        default=os.environ.get("GPO_TASKS_YAML", str(_GPO_TASKS_YAML)),
        help="Read-only task recipe; shared with training via the generated artifact.",
    )

    p.add_argument("--report", action="store_true", help="Write a Markdown report inside the run artifact.")
    p.add_argument("--output_dir", required=True, help="New run-specific artifact root, outside configs/.")
    p.add_argument("--run_tag", required=True, help="Target GPO training tag; must match training --tag.")
    p.add_argument("--configs_root", default=str(_REPO_ROOT / "scripts_mast/configs"))
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
        "",
        f"> **Generated:** {datetime.datetime.now().isoformat(timespec='seconds')}  ",
        "> **Source:** `scripts_mast/calibrate_gpo_task.py --report`  ",
        f"> **Pair directory:** `{summary['gpo_dir']}`  ",
        "",
        "---",
        "",
        "## Calibrated parameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Task key | `{task}` |",
        f"| β calibrated | **{summary['beta_calibrated']:.1f}** |",
        f"| β geomean (raw) | {summary['beta_geomean_raw']:.2f} |",
        f"| Shot blacklist | {summary['shot_blacklist'] if summary['shot_blacklist'] else '*(none)*'} |",
        "",
        "---",
        "",
        "## Per-signal statistics",
        "",
        "| Signal | n_windows | p50 MSE | β_opt (1/p50) | dataset mean MSE |",
        "|--------|-----------|---------|---------------|-----------------|",
    ]

    for sig, s in sorted(summary["signals"].items()):
        lines.append(
            f"| {sig} | {s['n_windows']:,} | {s['p50_mse']:.5f} | {s['beta_opt']:.1f} | {s['dataset_mean_mse']:.5f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Top-shot outliers (from calibration scan)",
        "",
    ]

    any_top = False
    for sig, s in sorted(summary["signals"].items()):
        top_shots = s.get("top_shots", [])
        if not top_shots:
            continue
        any_top = True
        lines += [
            f"### {sig}",
            "",
            "| Rank | shot_id | mean MSE | × dataset mean |",
            "|------|---------|----------|----------------|",
        ]
        for rank, row in enumerate(top_shots[:10], 1):
            # Shot statistics are keyed mappings in the immutable summary.
            if len(row) >= 3:
                sid, mean_mse = int(row["shot_id"]), float(row["mean_mse"])
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
                "---",
                "",
                "## Diagnostic figures",
                "",
                "The following PNG files were saved alongside this report:",
                "",
            ]
            for png in pngs:
                lines.append(f"- `{png.name}`")
            lines.append("")

    lines += ["---", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("  Calibration report written to: %s", md_path)
    return md_path


def main() -> None:
    """
    Validate and calibrate all requested collections, publishing only immutable run artifacts.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        Nonzero if any requested task fails; dry-run performs the same validation without writes.
    """
    args = _parse_args()
    configs_root = Path(args.configs_root).resolve()
    output_root = Path(args.output_dir).resolve()
    input_path = Path(args.gpo_tasks_yaml).resolve()
    if output_root.is_relative_to(configs_root):
        raise SystemExit("Calibration output must be outside the shared configuration tree.")
    if (
        args.max_windows < 0
        or args.blacklist_top_n < 0
        or not np.isfinite(args.blacklist_multiplier)
        or args.blacklist_multiplier <= 0
    ):
        raise SystemExit("Invalid calibration sampling/blacklist settings.")
    all_ok = True
    seen = set()
    for raw_dir in args.gpo_dirs:
        try:
            gpo_dir = Path(raw_dir).resolve()
            ok, errors = _validate_dir(gpo_dir)
            if not ok:
                raise ValueError("; ".join(errors))
            cc = json.loads((gpo_dir / "collection_config.json").read_text())
            validate_collection(cc, cc["contract"], gpo_dir)
            if source_fingerprint(cc["run_dir"]) != cc["contract"]["source"]:
                raise ValueError("Source checkpoint/embeddings changed after collection; recollect with a new tag.")
            task = cc["contract"]["task"]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", task) or task in {".", ".."}:
                raise ValueError(f"Invalid collection task identifier: {task!r}")
            if task in seen:
                raise ValueError(f"Duplicate task {task}; provide one collection per task.")
            seen.add(task)
            recipe = _effective_recipe(task, input_path, configs_root)
            train = recipe["train"]
            term_blocks = [train["loss"]] + [stage["loss"] for stage in train["stages"] if "loss" in stage]
            preference_terms = [
                term for block in term_blocks for term in block["terms"] if term["type"] == "continuous_gpo"
            ]
            if any(t.get("log_mse_gap") or t.get("mse_gap_clip") is not None for t in preference_terms):
                raise ValueError(
                    "Automatic beta calibration requires raw gaps; disable clip/log or use a reviewed recipe."
                )
            shots = cc["contract"]["split_manifest"]["shots"]
            stats = {
                sig: _calibrate_signal(_load_signal(shards, args.max_windows, set(shots["train"])))
                for sig, shards in _discover_signals(gpo_dir).items()
            }
            beta_opts = [value["beta_opt"] for value in stats.values() if value["p50_mse"] > 1e-8]
            if not beta_opts:
                raise ValueError("No positive training-pair gap: calibration is incomplete.")
            beta = float(np.exp(np.mean(np.log(beta_opts))))
            if not np.isfinite(beta) or beta <= 0:
                raise ValueError("Non-finite calibration beta.")
            for term in preference_terms:
                term["beta"] = beta
            dataset = train.setdefault("gpo_dataset", {})
            blacklist = sorted(
                set(dataset.get("shot_blacklist") or [])
                | set(_determine_blacklist(stats, args.blacklist_multiplier, args.blacklist_top_n))
            )
            dataset["shot_blacklist"] = blacklist
            # Local loss overrides must not undo calibrated beta or blacklist during training.
            from mast_utils.gpo.protocol import require_config_subset

            local = configs_root / "local_overrides.yaml"
            if local.is_file():
                effective = deep_merge(base=recipe, override=load_yaml(local))
                require_config_subset(recipe["train"], effective["train"], "train")
            artifact = {
                "calibration": {
                    "format": "cgpo-calibration-v1",
                    "run_tag": args.run_tag,
                    "collection_digest": digest(cc),
                    "inputs": config_fingerprint(configs_root),
                    "input_recipe": str(input_path),
                    "input_recipe_digest": file_digest(input_path),
                    "recipe_digest": digest(recipe),
                    "task": task,
                    "max_windows": args.max_windows,
                    "blacklist_multiplier": args.blacklist_multiplier,
                    "blacklist_top_n": args.blacklist_top_n,
                },
                "tasks": {task: recipe},
            }
            summary = {
                "task": task,
                "gpo_dir": str(gpo_dir),
                "beta_calibrated": beta,
                "beta_geomean_raw": beta,
                "shot_blacklist": blacklist,
                "preference_terms": len(preference_terms),
                "calibration": artifact["calibration"],
                "signals": {
                    sig: {**{k: v for k, v in value.items() if k != "shot_rows"}, "top_shots": value["shot_rows"][:10]}
                    for sig, value in stats.items()
                },
            }
            out = output_root / task
            if out.exists():
                raise FileExistsError(f"Calibration artifact already exists: {out}; use a new run tag.")
            log.info(
                "%s: beta=%g, preference_terms=%d, blacklist=%s; artifact=%s",
                task,
                beta,
                len(preference_terms),
                blacklist,
                out / "gpo_tasks.yaml",
            )
            if args.dry_run:
                log.info("DRY RUN: no YAML, summary, plots or reports written.")
                continue
            out.mkdir(parents=True, exist_ok=False)
            with (out / "gpo_tasks.yaml").open("x") as stream:
                yaml.safe_dump(artifact, stream, sort_keys=False)
            (out / "calibration_summary.json").write_text(json.dumps(summary, indent=2))
            if not args.no_plots:
                _save_plots(gpo_dir, out / "plots")
            if args.report:
                _write_calibration_report(summary, out / "plots", out, args.run_tag)
        except (OSError, ValueError, KeyError, TypeError) as error:
            log.error("INCOMPLETE calibration for %s: %s", raw_dir, error)
            all_ok = False
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
