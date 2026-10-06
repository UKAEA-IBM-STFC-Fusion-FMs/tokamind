"""
print_gpo_shot_outliers.py
==========================

Print the top-N shot outlier table from a GPO pair directory to stdout.

This is the panel-23 data (shot-level MSE gap ranking) without any matplotlib
dependency — suitable for running on a login node and reading directly from the
job log or terminal.

Usage
-----
    python scripts_mast/print_gpo_shot_outliers.py \
        runs/ft-task_2-3-scratch-mmt-embed_mse/gpo_pairs \
        [--top 20] [--max_windows 0]

    # Multiple directories at once
    python scripts_mast/print_gpo_shot_outliers.py \
        runs/ft-task_1-1-scratch-mmt-embed_mse/gpo_pairs \
        runs/ft-task_2-3-scratch-mmt-embed_mse/gpo_pairs

Exit code 0 always (informational only).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _shot_outlier_table(gpo_dir: Path, top: int, max_windows: int) -> None:
    # Inline minimal versions of _discover_signals and _load_arrays
    # so this script has no import dependency on visualize_gpo_stats.
    shards = sorted(gpo_dir.glob("*.npz"))
    if not shards:
        print(f"  ERROR: no .npz shards in {gpo_dir}", file=sys.stderr)
        return

    # Group by signal
    signal_shards: dict[str, list[Path]] = {}
    for s in shards:
        stem = s.stem
        sig = stem[: stem.rfind("__shard_")] if "__shard_" in stem else stem
        signal_shards.setdefault(sig, []).append(s)

    # Aggregate shot stats across all signals
    # shot_id → {mse: [], n_windows: int}
    shot_agg: dict[int, dict] = {}

    for sig, sig_shards in sorted(signal_shards.items()):
        total = 0
        for shard in sorted(sig_shards):
            with np.load(shard, allow_pickle=False) as npz:
                n = len(npz["shot_id"])
                if max_windows > 0 and total + n > max_windows:
                    n = max(0, max_windows - total)
                if n == 0:
                    break

                shot_ids = npz["shot_id"][:n]
                yw = npz["y_w_emb"][:n].astype(np.float32)
                yl = npz["y_l_emb"][:n].astype(np.float32)
                mse = ((yw - yl) ** 2).mean(axis=1)  # (n,)

                for i in range(n):
                    sid = int(shot_ids[i])
                    entry = shot_agg.setdefault(sid, {"mse": [], "n": 0})
                    entry["mse"].append(float(mse[i]))
                    entry["n"] += 1

                total += n
                if max_windows > 0 and total >= max_windows:
                    break

    if not shot_agg:
        print(f"  ERROR: no data loaded from {gpo_dir}", file=sys.stderr)
        return

    # Build rows sorted by mean MSE descending
    rows = []
    for sid, e in shot_agg.items():
        arr = np.asarray(e["mse"], dtype=np.float32)
        rows.append((sid, e["n"], float(arr.mean()), float(np.median(arr)), float(np.percentile(arr, 95))))
    rows.sort(key=lambda r: r[2], reverse=True)

    print(f"\n{'=' * 72}")
    print(f"Shot outlier table: {gpo_dir}")
    print(f"Total unique shots: {len(rows)} | showing top {min(top, len(rows))}")
    print(f"{'Rank':>4}  {'shot_id':>8}  {'n_win':>6}  {'mean_MSE':>10}  {'p50_MSE':>10}  {'p95_MSE':>10}")
    print(f"{'-' * 4}  {'-' * 8}  {'-' * 6}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    for rank, (sid, n_win, mean_mse, p50_mse, p95_mse) in enumerate(rows[:top], 1):
        marker = "  ← BLACKLIST CANDIDATE" if rank <= 3 else ""
        print(f"{rank:>4}  {sid:>8}  {n_win:>6}  {mean_mse:>10.4f}  {p50_mse:>10.4f}  {p95_mse:>10.4f}{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print shot-level MSE outlier table from GPO pair shards (panel-23 data).",
        allow_abbrev=False,
    )
    parser.add_argument("dirs", nargs="+", metavar="GPO_DIR")
    parser.add_argument("--top", type=int, default=20, help="Number of worst shots to show (default: 20).")
    parser.add_argument(
        "--max_windows", type=int, default=0, help="Cap windows loaded per signal (0 = all, default: 0)."
    )
    args = parser.parse_args()

    for raw_dir in args.dirs:
        _shot_outlier_table(Path(raw_dir), top=args.top, max_windows=args.max_windows)


if __name__ == "__main__":
    main()
