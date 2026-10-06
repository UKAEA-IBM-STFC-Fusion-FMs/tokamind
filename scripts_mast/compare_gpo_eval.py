"""
compare_gpo_eval.py
===================

Compare evaluation metrics between GPO fine-tuned models and their base
finetune counterparts.  Reads the CSV files written by ``run_eval.py`` under
``runs/<run_id>/eval/metrics/<task>/`` and prints a side-by-side table of key
metrics with absolute and relative deltas (GPO − base).

Output layout expected from run_eval.py
----------------------------------------
    runs/<run_id>/eval/metrics/<task>/task_metrics.csv

    # if --tag was passed to run_eval.py:
    runs/<run_id>/eval_<tag>/metrics/<task>/task_metrics.csv

task_metrics.csv columns (from tokamark.evaluator.compute_metrics):
    index=task_name   NRMSE_mean  NRMSE_std_pop
                      NMAE_mean   NMAE_std_pop
                      RMSE_mean   RMSE_std_pop
                      MAE_mean    MAE_std_pop
                      nan_fraction_mean  nan_fraction_std_pop
                      n_shots

Usage
-----
Auto-derive run IDs from task names (standard naming convention):

    python scripts_mast/compare_gpo_eval.py \\
        --tasks task_4-1 task_4-2 task_4-3 task_4-4 task_4-5

    # GPO-v2 is the default (--gpo_version v2); for v1 runs use:
    python scripts_mast/compare_gpo_eval.py \\
        --tasks task_4-1 task_4-2 task_4-3 task_4-4 task_4-5 \\
        --gpo_version gpo-v1

Explicit base + GPO run IDs for one task:

    python scripts_mast/compare_gpo_eval.py \\
        --task task_4-3 \\
        --base_run ft-task_4-3-scratch-mmt-embed_mse \\
        --gpo_run  ft-task_4-3-ws-ft-task_4-3-scratch-mmt-embed_mse-mmt-v2

Optional flags
--------------
    --runs_dir DIR        Root directory containing run sub-folders.
                          Default: runs/  (relative to cwd, or $REPO_ROOT/runs/).
    --eval_tag TAG        Eval sub-folder suffix; uses eval_<TAG>/ instead of eval/.
    --base_tag TAG        Separate eval tag for the base run (default: same as --eval_tag).
    --gpo_tag  TAG        Separate eval tag for the GPO run (default: same as --eval_tag).
    --gpo_version VERSION Tag appended to the auto-derived GPO run ID.
                          Run ID = ft-<task>-ws-ft-<task>-scratch-mmt-embed_mse-mmt-<VERSION>.
                          Default: v2  (matches run_gpo_finetune.py --tag v2).
                          Use 'gpo-v1' for the older v1 runs.
    --metrics METRIC      One or more metric columns to compare.
                          Default: NRMSE_mean NMAE_mean RMSE_mean MAE_mean n_shots
    --save_dir DIR        If given, writes comparison_<task>.json files there.
    --report_dir DIR      If given, writes a Markdown summary report to that directory.
                          File name: gpo_comparison_<base_ft_tag>_<gpo_version>_<timestamp>.md
                          Intended for use with the reports/ folder.
    --csv                 Also print raw CSV rows for each run (for debugging).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("gpo.compare")

# Metrics shown by default (subset of task_metrics.csv columns).
_DEFAULT_METRICS = ["NRMSE_mean", "NMAE_mean", "RMSE_mean", "MAE_mean", "n_shots"]

# Improvement direction: True = lower is better, False = higher is better.
_LOWER_IS_BETTER: dict[str, bool] = {
    "NRMSE_mean": True,
    "NMAE_mean": True,
    "RMSE_mean": True,
    "MAE_mean": True,
    "n_shots": False,  # higher n_shots is not "better" — neutral
}


# ======================================================================================================================
# Run-ID construction
# ======================================================================================================================


def _gpo_run_id(
    task: str,
    version: str = "v2",
    base_ft_tag: str = "embed_mse",
    model_profile: str = "mmt",
) -> str:
    """Standard GPO warmstart run ID for a task.

    The run ID produced by run_gpo_finetune.py --tag <version> is
    ``ft-<task>-ws-ft-<task>-scratch-<model_profile>-<base_ft_tag>-<model_profile>-<version>``.
    Note: there is NO 'gpo-' infix — the tag is appended literally.
    """
    return (
        f"ft-{task}-ws-ft-{task}-scratch-{model_profile}-{base_ft_tag}"
        f"-{model_profile}-{version}"
    )


def _base_run_id(task: str, base_ft_tag: str = "embed_mse", model_profile: str = "mmt") -> str:
    """Standard base finetune run ID for a task."""
    return f"ft-{task}-scratch-{model_profile}-{base_ft_tag}"


# ======================================================================================================================
# Metric file discovery
# ======================================================================================================================


def _eval_subdir(run_dir: Path, tag: str | None) -> Path:
    """Return the eval sub-directory path (eval/ or eval_<tag>/)."""
    return run_dir / (f"eval_{tag}" if tag else "eval")


def _find_task_metrics_csv(run_dir: Path, task: str, tag: str | None) -> Path | None:
    """
    Locate task_metrics.csv for a run.

    Returns the path if it exists, else None.
    """
    candidate = _eval_subdir(run_dir, tag) / "metrics" / task / "task_metrics.csv"
    return candidate if candidate.is_file() else None


# ======================================================================================================================
# CSV reading
# ======================================================================================================================


def _read_task_row(csv_path: Path, task: str) -> dict[str, float]:
    """
    Read the task-level metrics row from a task_metrics.csv file.

    Parameters
    ----------
    csv_path : Path
        Path to the task_metrics.csv written by tokamark.evaluator.compute_metrics.
    task : str
        Task name used as the DataFrame index (e.g. "task_4-1").

    Returns
    -------
    dict[str, float]
        Column-name → float value for the task row.

    Raises
    ------
    ValueError
        If the task row is not found in the CSV.
    """
    try:
        import pandas as pd  # pandas is available in the CCC environment
    except ImportError:
        return _read_task_row_no_pandas(csv_path, task)

    df = pd.read_csv(csv_path, index_col=0)
    if task not in df.index:
        # Some task_metrics.csv files use the task name as the only row but
        # without an explicit index label — fall back to the first row.
        if len(df) == 1:
            row = df.iloc[0]
            log.debug("task_metrics.csv: task %r not in index; using first row.", task)
        else:
            raise ValueError(f"Task row {task!r} not found in {csv_path}.\nAvailable index values: {list(df.index)}")
    else:
        row = df.loc[task]

    return {k: float(v) for k, v in row.to_dict().items() if not _is_nan(v)}


def _read_task_row_no_pandas(csv_path: Path, task: str) -> dict[str, float]:
    """Fallback CSV reader when pandas is not available."""
    with csv_path.open(encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"task_metrics.csv at {csv_path} has fewer than 2 lines.")
    headers = [h.strip() for h in lines[0].split(",")]
    for line in lines[1:]:
        cols = [c.strip() for c in line.split(",")]
        if not cols:
            continue
        row_label = cols[0].strip('"')
        values = cols[1:]
        if row_label == task or len(lines) == 2:
            result: dict[str, float] = {}
            for h, v in zip(headers[1:], values):
                try:
                    result[h] = float(v)
                except (ValueError, TypeError):
                    pass
            return result
    raise ValueError(f"Task row {task!r} not found in {csv_path}.")


def _is_nan(v: Any) -> bool:
    try:
        import math

        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


# ======================================================================================================================
# Core comparison logic
# ======================================================================================================================


def _load_run_metrics(
    run_dir: Path,
    task: str,
    tag: str | None,
    run_label: str,
) -> dict[str, float] | None:
    """
    Load all available metrics for one run into a flat dict.

    Returns None if task_metrics.csv is not found.
    """
    csv_path = _find_task_metrics_csv(run_dir, task, tag)
    if csv_path is None:
        log.warning(
            "%s — task_metrics.csv not found at expected path: %s",
            run_label,
            _eval_subdir(run_dir, tag) / "metrics" / task / "task_metrics.csv",
        )
        return None

    try:
        metrics = _read_task_row(csv_path, task)
    except Exception as exc:
        log.error("%s — failed to read task_metrics.csv: %s", run_label, exc)
        return None

    return metrics


def _compare_task(
    task: str,
    base_run_id: str,
    gpo_run_id: str,
    runs_root: Path,
    base_tag: str | None,
    gpo_tag: str | None,
    metric_cols: list[str],
    show_csv: bool,
) -> dict[str, Any] | None:
    """
    Load metrics for one task's base and GPO runs, print a comparison table,
    and return the comparison dict.
    """
    base_dir = runs_root / base_run_id
    gpo_dir = runs_root / gpo_run_id

    for label, d in [("base", base_dir), ("GPO", gpo_dir)]:
        if not d.is_dir():
            log.error("Run directory not found for %s (%s): %s", label, task, d)
            return None

    base_metrics = _load_run_metrics(base_dir, task, base_tag, f"base ({base_run_id})")
    gpo_metrics = _load_run_metrics(gpo_dir, task, gpo_tag, f"GPO  ({gpo_run_id})")

    if base_metrics is None or gpo_metrics is None:
        return None

    # Build comparison for every requested metric column.
    comparison: dict[str, Any] = {
        "task": task,
        "base_run_id": base_run_id,
        "gpo_run_id": gpo_run_id,
        "base_metrics": base_metrics,
        "gpo_metrics": gpo_metrics,
        "deltas": {},
    }

    # Collect all available columns (requested + always show n_shots).
    display_cols = list(dict.fromkeys(metric_cols + ["n_shots"]))

    # ---- Print table ----
    col_w = 14
    name_w = max(len(c) for c in display_cols) + 2

    print()
    print(f"{'─' * (name_w + col_w * 4 + 6)}")
    print(f"  Task: {task}   base: {base_run_id}")
    print(f"                  gpo:  {gpo_run_id}")
    print(f"{'─' * (name_w + col_w * 4 + 6)}")
    header = f"  {'Metric':<{name_w}} {'Base':>{col_w}} {'GPO':>{col_w}} {'Δ (GPO−Base)':>{col_w}} {'Δ %':>{col_w}}"
    print(header)
    print(f"  {'─' * (name_w + col_w * 4 + 4)}")

    for col in display_cols:
        base_v = base_metrics.get(col)
        gpo_v = gpo_metrics.get(col)

        if base_v is None and gpo_v is None:
            continue

        base_str = f"{base_v:.6g}" if base_v is not None else "N/A"
        gpo_str = f"{gpo_v:.6g}" if gpo_v is not None else "N/A"

        if base_v is not None and gpo_v is not None:
            delta = gpo_v - base_v
            delta_pct = delta / abs(base_v) * 100.0 if base_v != 0.0 else float("nan")
            lower_better = _LOWER_IS_BETTER.get(col, True)

            # Improvement = delta < 0 if lower is better, delta > 0 if higher is better.
            improved = (delta < -1e-9) if lower_better else (delta > 1e-9)
            marker = " ✓" if improved else (" ✗" if abs(delta_pct) > 0.1 else "  ")

            delta_str = f"{delta:+.6g}"
            pct_str = f"{delta_pct:+.2f}%{marker}"

            comparison["deltas"][col] = {
                "base": base_v,
                "gpo": gpo_v,
                "delta": delta,
                "delta_pct": delta_pct,
                "improved": improved,
            }
        else:
            delta_str = "N/A"
            pct_str = "N/A"

        print(f"  {col:<{name_w}} {base_str:>{col_w}} {gpo_str:>{col_w}} {delta_str:>{col_w}} {pct_str:>{col_w}}")

    print(f"  {'─' * (name_w + col_w * 4 + 4)}")

    # Summary verdict
    improved_cols = [c for c, d in comparison["deltas"].items() if d.get("improved")]
    degraded_cols = [
        c
        for c, d in comparison["deltas"].items()
        if not d.get("improved") and abs(d.get("delta_pct", 0)) > 0.1 and c != "n_shots"
    ]
    if improved_cols and not degraded_cols:
        verdict = "✓ GPO improved all metrics"
    elif improved_cols:
        verdict = f"~ Mixed: improved {improved_cols}, degraded {degraded_cols}"
    elif degraded_cols:
        verdict = f"✗ GPO degraded: {degraded_cols}"
    else:
        verdict = "≈ No meaningful change (all Δ < 0.1%)"

    print(f"\n  Verdict: {verdict}")

    if show_csv:
        print("\n  [debug] base CSV row:", base_metrics)
        print("  [debug]  gpo CSV row:", gpo_metrics)

    comparison["verdict"] = verdict
    return comparison


# ======================================================================================================================
# CLI
# ======================================================================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare GPO vs base finetune eval metrics.",
        allow_abbrev=False,
    )

    # Task / run specification (two modes)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tasks",
        nargs="+",
        metavar="TASK",
        help=(
            "One or more task identifiers (e.g. task_4-1 task_4-3). "
            "Run IDs are auto-derived using the standard naming convention."
        ),
    )
    group.add_argument(
        "--task",
        type=str,
        metavar="TASK",
        help="Single task identifier. Use with --base_run and --gpo_run for explicit run IDs.",
    )

    p.add_argument(
        "--base_run",
        type=str,
        default=None,
        help="Base finetune run ID (use with --task; auto-derived when --tasks is given).",
    )
    p.add_argument(
        "--gpo_run", type=str, default=None, help="GPO run ID (use with --task; auto-derived when --tasks is given)."
    )
    p.add_argument(
        "--gpo_version",
        type=str,
        default="v2",
        help=(
            "GPO run tag used to auto-derive the GPO run ID when --tasks is given. "
            "The run ID is ft-<task>-ws-ft-<task>-scratch-<model_profile>-<base_ft_tag>-<model_profile>-<VERSION>. "
            "Default: v2  (matches --tag v2 passed to run_gpo_finetune.py). "
            "Use 'gpo-v1' to compare against the older v1 runs."
        ),
    )
    p.add_argument(
        "--base_ft_tag",
        type=str,
        default="embed_mse",
        help=(
            "Fine-tune tag used to auto-derive both the base and GPO run IDs when --tasks "
            "is given.  Run IDs are ft-<task>-scratch-<model_profile>-<TAG> (base) and "
            "ft-<task>-ws-...-<model_profile>-<TAG>-<model_profile>-<gpo_version> (GPO).  "
            "Default: embed_mse (legacy).  Use 'dct3d-embed-mse' for the new pipeline."
        ),
    )
    p.add_argument(
        "--model_profile",
        type=str,
        default="mmt",
        help=(
            "Model architecture profile embedded in run IDs. "
            "Must match the --model_profile used during fine-tuning. "
            "Default: mmt."
        ),
    )

    # Directory / tag
    p.add_argument(
        "--runs_dir",
        type=str,
        default=None,
        help="Root directory containing run sub-folders (default: 'runs/' relative to cwd).",
    )
    p.add_argument(
        "--eval_tag",
        type=str,
        default=None,
        help="Eval sub-folder suffix used for both runs: eval_<TAG>/ (default: eval/).",
    )
    p.add_argument(
        "--base_tag", type=str, default=None, help="Eval tag for the base run only (overrides --eval_tag for base)."
    )
    p.add_argument(
        "--gpo_tag", type=str, default=None, help="Eval tag for the GPO run only (overrides --eval_tag for GPO)."
    )

    # Metrics
    p.add_argument(
        "--metrics",
        nargs="+",
        default=_DEFAULT_METRICS,
        metavar="COL",
        help=(
            f"Metric columns to compare (default: {' '.join(_DEFAULT_METRICS)}). "
            "Any column from task_metrics.csv is valid."
        ),
    )

    # Output
    p.add_argument("--save_dir", type=str, default=None, help="Directory to write comparison_<task>.json files.")
    p.add_argument(
        "--report_dir",
        type=str,
        default=None,
        help=(
            "Directory to write a Markdown summary report. "
            "File name: gpo_comparison_<base_ft_tag>_<gpo_version>_<timestamp>.md. "
            "Intended for use with the reports/ folder."
        ),
    )
    p.add_argument("--csv", action="store_true", help="Also print the raw CSV row for each run (debug).")

    return p.parse_args()


# ======================================================================================================================
# Markdown report
# ======================================================================================================================


def _write_markdown_report(
    results: list[dict[str, Any]],
    metric_cols: list[str],
    base_ft_tag: str,
    gpo_version: str,
    report_dir: Path,
) -> Path:
    """Write a Markdown summary of all task comparisons to *report_dir*.

    File name: ``gpo_comparison_<base_ft_tag>_<gpo_version>_<YYYYMMDD_HHMMSS>.md``

    The file is self-contained: it includes a provenance header, a summary
    table across all tasks, and a per-task detail section.  Intended to be
    committed to the ``reports/`` folder for paper reference.
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gpo_comparison_{base_ft_tag}_{gpo_version}_{ts}.md"
    out_path = report_dir / filename

    # Collect display columns (preserve order, n_shots always last)
    display_cols = list(dict.fromkeys(metric_cols))
    metric_cols_no_shots = [c for c in display_cols if c != "n_shots"]

    lines: list[str] = []

    # ---- Header ----
    lines += [
        f"# GPO vs Base Fine-Tune Comparison",
        f"## base_ft_tag=`{base_ft_tag}`  ·  gpo_version=`{gpo_version}`",
        f"",
        f"> **Generated:** {datetime.datetime.now().isoformat(timespec='seconds')}  ",
        f"> **Source:** `scripts_mast/compare_gpo_eval.py --report_dir`  ",
        f"> **Metrics:** {', '.join(display_cols)}  ",
        f"> **Key:** ✓ improvement · ✗ degradation · ≈ no meaningful change (|Δ| < 0.1%)  ",
        f"",
        f"---",
        f"",
    ]

    # ---- Summary table ----
    lines += [
        "## Summary",
        "",
    ]

    # Header row
    col_headers = " | ".join(f"{c} Δ%" for c in metric_cols_no_shots)
    lines.append(f"| Task | {col_headers} | n_shots | Verdict |")
    lines.append(f"|------|{'|'.join(['---'] * len(metric_cols_no_shots))}|---------|---------|")

    for r in results:
        task = r["task"]
        deltas = r.get("deltas", {})
        verdict = r.get("verdict", "—")

        # Shorten verdict for the table
        if verdict.startswith("✓"):
            verdict_short = "✓ all improved"
        elif verdict.startswith("✗"):
            verdict_short = "✗ degraded"
        elif verdict.startswith("~"):
            verdict_short = "~ mixed"
        else:
            verdict_short = "≈ flat"

        n_shots_str = "—"
        if "n_shots" in deltas:
            n_shots_str = str(int(deltas["n_shots"]["base"]))
        elif "n_shots" in r.get("base_metrics", {}):
            v = r["base_metrics"]["n_shots"]
            n_shots_str = str(int(v)) if v is not None else "—"

        cells = []
        for col in metric_cols_no_shots:
            d = deltas.get(col)
            if d is None:
                cells.append("—")
            else:
                pct = d.get("delta_pct")
                improved = d.get("improved", False)
                marker = " ✓" if improved else (" ✗" if abs(pct or 0) > 0.1 else " ≈")
                cells.append(f"{pct:+.2f}%{marker}" if pct is not None else "—")

        row = f"| {task} | {' | '.join(cells)} | {n_shots_str} | {verdict_short} |"
        lines.append(row)

    lines += ["", "---", ""]

    # ---- Per-task detail ----
    lines += ["## Per-task detail", ""]

    for r in results:
        task = r["task"]
        base_id = r.get("base_run_id", "—")
        gpo_id = r.get("gpo_run_id", "—")
        deltas = r.get("deltas", {})
        verdict = r.get("verdict", "—")

        lines += [
            f"### {task}",
            f"",
            f"- **Base run:** `{base_id}`",
            f"- **GPO run:**  `{gpo_id}`",
            f"- **Verdict:**  {verdict}",
            f"",
        ]

        # Detail table
        lines.append(f"| Metric | Base | GPO | Δ (GPO−Base) | Δ % |")
        lines.append(f"|--------|------|-----|--------------|-----|")

        all_cols = list(dict.fromkeys(display_cols + ["n_shots"]))
        for col in all_cols:
            d = deltas.get(col)
            bm = r.get("base_metrics", {})
            gm = r.get("gpo_metrics", {})

            base_v = bm.get(col)
            gpo_v = gm.get(col)

            if base_v is None and gpo_v is None:
                continue

            base_str = f"{base_v:.6g}" if base_v is not None else "N/A"
            gpo_str = f"{gpo_v:.6g}" if gpo_v is not None else "N/A"

            if d is not None:
                delta = d.get("delta")
                pct = d.get("delta_pct")
                improved = d.get("improved", False)
                marker = " ✓" if improved else (" ✗" if abs(pct or 0) > 0.1 else "")
                delta_str = f"{delta:+.6g}" if delta is not None else "—"
                pct_str = (f"{pct:+.2f}%{marker}" if pct is not None else "—")
            else:
                delta_str = "—"
                pct_str = "—"

            lines.append(f"| {col} | {base_str} | {gpo_str} | {delta_str} | {pct_str} |")

        lines += [""]

    lines += ["---", ""]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ======================================================================================================================
# Main
# ======================================================================================================================


def main() -> None:
    args = _parse_args()

    # ---- Resolve runs root ----
    if args.runs_dir:
        runs_root = Path(args.runs_dir)
    else:
        # Default: 'runs/' relative to cwd; fall back to REPO_ROOT env var.
        import os

        repo_root = Path(os.environ.get("REPO_ROOT", "."))
        runs_root = repo_root / "runs"

    if not runs_root.is_dir():
        log.error("Runs directory not found: %s", runs_root)
        sys.exit(1)

    # ---- Resolve eval tags ----
    base_tag = args.base_tag if args.base_tag is not None else args.eval_tag
    gpo_tag = args.gpo_tag if args.gpo_tag is not None else args.eval_tag

    # ---- Resolve task list and run IDs ----
    if args.tasks:
        # Multi-task mode: auto-derive run IDs.
        task_pairs: list[tuple[str, str, str]] = [
            (
                task,
                _base_run_id(task, base_ft_tag=args.base_ft_tag, model_profile=args.model_profile),
                _gpo_run_id(task, version=args.gpo_version, base_ft_tag=args.base_ft_tag, model_profile=args.model_profile),
            )
            for task in args.tasks
        ]
    else:
        # Single-task mode: explicit run IDs.
        task = args.task
        base = args.base_run or _base_run_id(task, base_ft_tag=args.base_ft_tag, model_profile=args.model_profile)
        gpo = args.gpo_run or _gpo_run_id(task, version=args.gpo_version, base_ft_tag=args.base_ft_tag, model_profile=args.model_profile)
        task_pairs = [(task, base, gpo)]

    # ---- Save dir ----
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Report dir ----
    report_dir = Path(args.report_dir) if args.report_dir else None

    # ---- Run comparisons ----
    all_results: list[dict[str, Any]] = []
    n_ok = 0

    print("\nGPO vs Base Finetune Evaluation Comparison")
    print(f"Runs root : {runs_root}")
    print(f"Metrics   : {args.metrics}")
    print(f"Eval tags : base={base_tag or 'eval'}, gpo={gpo_tag or 'eval'}")

    for task, base_id, gpo_id in task_pairs:
        result = _compare_task(
            task=task,
            base_run_id=base_id,
            gpo_run_id=gpo_id,
            runs_root=runs_root,
            base_tag=base_tag,
            gpo_tag=gpo_tag,
            metric_cols=args.metrics,
            show_csv=args.csv,
        )
        if result is not None:
            all_results.append(result)
            n_ok += 1

            if save_dir:
                out_path = save_dir / f"comparison_{task}.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                log.info("Saved: %s", out_path)

    print()
    print(f"Compared {n_ok}/{len(task_pairs)} tasks successfully.")

    # ---- Write Markdown report ----
    if report_dir and all_results:
        report_path = _write_markdown_report(
            results=all_results,
            metric_cols=args.metrics,
            base_ft_tag=args.base_ft_tag,
            gpo_version=args.gpo_version,
            report_dir=report_dir,
        )
        print(f"Report    : {report_path}")

    if n_ok < len(task_pairs):
        sys.exit(1)


# ======================================================================================================================
if __name__ == "__main__":
    main()
