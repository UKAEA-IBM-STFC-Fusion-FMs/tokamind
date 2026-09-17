"""
validate_gpo_pairs.py
=====================

Validate a GPO preference-pair directory before launching training.

Checks (in order):
  1. metadata.json and collection_config.json are present and consistent.
  2. schema_version == 3 in both files (embedding-space pairs with native diagnostics).
  3. Verified official train/validation shot protocol and shard hashes (test excluded).
  4. At least one .npz shard is present.
  5. Every shard has keys: shot_id, window_index, y_w_emb, y_l_emb, native_nrmse,
     native_nmae, native_mse, embedding_gap, emb_dim. Keys y_w / y_l (schema v1
     native arrays) must NOT be present.
  6. Embeddings are float32 with shape (B, D), D == emb_dim; diagnostics are
     float32 with shape (B,).
  7. No NaN or Inf values in embeddings or diagnostics.
  8. y_w_emb != y_l_emb for every row (non-zero MSE gap — a zero gap means the
     model already predicted perfectly, which produces a useless preference pair).
  9. emb_dim is consistent across all shards for the same signal.
 10. total_windows in metadata.json matches the actual shard row counts.

Usage
-----
    python scripts_mast/validate_gpo_pairs.py path/to/gpo_pairs [path2 ...]

Exit code 0 = all checks passed; 1 = one or more checks failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def validate_gpo_dir(gpo_dir: Path) -> tuple[list[str], list[str]]:
    """
    Validate a single GPO pairs directory.

    Returns (errors, warnings).  errors is non-empty when the directory
    has a genuine problem that would cause training to fail or produce
    wrong results.  warnings is informational only — the directory is
    still usable for training.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Metadata files present
    # ------------------------------------------------------------------
    meta_path = gpo_dir / "metadata.json"
    cc_path = gpo_dir / "collection_config.json"

    if not meta_path.is_file():
        errors.append("MISSING  metadata.json")
    if not cc_path.is_file():
        errors.append("MISSING  collection_config.json")
    if errors:
        return errors, warnings  # nothing more to check without these files

    meta = _load_json(meta_path)
    cc = _load_json(cc_path)

    # ------------------------------------------------------------------
    # 2. Schema version == 3 in both files
    # ------------------------------------------------------------------
    meta_sv = meta.get("schema_version")
    cc_sv = cc.get("schema_version")
    if meta_sv != 3:
        errors.append(
            f"SCHEMA   metadata.json schema_version={meta_sv!r} — must be 3. "
            "Re-collect with run_collect_gpo_pairs.py (current code)."
        )
    if cc_sv != 3:
        errors.append(f"SCHEMA   collection_config.json schema_version={cc_sv!r} — must be 3.")
    if meta_sv != cc_sv:
        errors.append(f"MISMATCH metadata schema_version={meta_sv} != collection_config schema_version={cc_sv}")

    # ------------------------------------------------------------------
    # 3. Verify the official shot protocol; legacy row-split collections are rejected.
    # ------------------------------------------------------------------
    if cc.get("protocol") == "mast-shot-split-v1":
        from mast_utils.gpo.protocol import validate_collection

        try:
            validate_collection(cc, cc["contract"], gpo_dir)
        except (KeyError, ValueError) as exc:
            errors.append(f"PROTOCOL {exc}")
    else:
        errors.append("PROTOCOL Legacy collection: recollect with --split both --val_fraction 0 and a new tag.")

    # ------------------------------------------------------------------
    # 4. At least one shard
    # ------------------------------------------------------------------
    shards = sorted(gpo_dir.glob("*.npz"))
    if not shards:
        errors.append("EMPTY    No .npz shard files found in directory.")
        return errors, warnings

    # ------------------------------------------------------------------
    # 5–9. Per-shard checks
    # ------------------------------------------------------------------
    # Group shards by signal name (prefix before __shard_)
    signal_shards: dict[str, list[Path]] = {}
    for s in shards:
        stem = s.stem
        sig = stem[: stem.rfind("__shard_")] if "__shard_" in stem else stem
        signal_shards.setdefault(sig, []).append(s)

    actual_totals: dict[str, int] = {}

    for sig, sig_shards in sorted(signal_shards.items()):
        emb_dim_seen: int | None = None
        sig_rows = 0

        for shard in sig_shards:
            try:
                npz = np.load(shard, allow_pickle=False)
            except Exception as exc:
                errors.append(f"CORRUPT  {shard.name}: {exc}")
                continue

            keys = set(npz.files)

            # 5a. Required keys present
            for req in (
                "shot_id",
                "window_index",
                "y_w_emb",
                "y_l_emb",
                "native_nrmse",
                "native_nmae",
                "native_mse",
                "embedding_gap",
                "emb_dim",
            ):
                if req not in keys:
                    errors.append(f"MISSING_KEY  {shard.name}: key '{req}' not found")

            # 5b. Schema-v1 native keys must NOT be present
            for bad in ("y_w", "y_l"):
                if bad in keys:
                    errors.append(
                        f"SCHEMA_V1  {shard.name}: key '{bad}' found — this is a "
                        "schema-v1 shard (native space). Re-collect with current code."
                    )

            if "y_w_emb" not in keys or "y_l_emb" not in keys:
                continue  # already reported; skip further checks on this shard

            y_w = npz["y_w_emb"]
            y_l = npz["y_l_emb"]
            emb_dim = int(npz["emb_dim"])
            B = y_w.shape[0]

            # 6. dtype and shape
            for name, arr in (("y_w_emb", y_w), ("y_l_emb", y_l)):
                if arr.dtype != np.float32:
                    errors.append(f"DTYPE    {shard.name} {name}: dtype={arr.dtype}, expected float32")
                if arr.ndim != 2:
                    errors.append(f"SHAPE    {shard.name} {name}: ndim={arr.ndim}, expected 2 (B, D)")
                elif arr.shape[1] != emb_dim:
                    errors.append(f"EMB_DIM  {shard.name} {name}: shape[1]={arr.shape[1]} != emb_dim={emb_dim}")
            for name in ("native_nrmse", "native_nmae", "native_mse", "embedding_gap"):
                if name not in keys:
                    continue
                arr = npz[name]
                if arr.dtype != np.float32:
                    errors.append(f"DTYPE    {shard.name} {name}: dtype={arr.dtype}, expected float32")
                if arr.shape != (B,):
                    errors.append(f"SHAPE    {shard.name} {name}: shape={arr.shape}, expected ({B},)")

            # 9. Consistent emb_dim across shards
            if emb_dim_seen is None:
                emb_dim_seen = emb_dim
            elif emb_dim != emb_dim_seen:
                errors.append(
                    f"EMB_DIM  {shard.name}: emb_dim={emb_dim} inconsistent with "
                    f"earlier shards (emb_dim={emb_dim_seen}) for signal '{sig}'"
                )

            sig_rows += B

            # 7. No NaN / Inf
            for name, arr in (("y_w_emb", y_w), ("y_l_emb", y_l)):
                n_bad = int((~np.isfinite(arr).all(axis=1)).sum())  # rows with any bad value
                if n_bad > 0:
                    errors.append(f"NON_FINITE {shard.name} {name}: {n_bad}/{B} rows contain NaN or Inf values")

            for name in ("native_nrmse", "native_nmae", "native_mse", "embedding_gap"):
                if name not in keys:
                    continue
                arr = npz[name]
                n_bad = int((~np.isfinite(arr)).sum())
                if n_bad > 0:
                    errors.append(f"NON_FINITE {shard.name} {name}: {n_bad}/{B} rows contain NaN or Inf values")

            # 8. Non-zero MSE gap (y_w != y_l per row).
            # Threshold 1e-9 (mean squared error per coefficient) is well
            # above float32 machine-epsilon (~1e-14) and catches genuine
            # all-zeros / NaN-mask failures without over-flagging near-
            # perfect predictions on D=1 identity-encoded scalar outputs.
            mse_gap = ((y_w - y_l) ** 2).mean(axis=1)  # (B,)
            n_zero = int((mse_gap < 1e-9).sum())
            if n_zero > 0:
                frac = n_zero / max(1, B)
                msg = (
                    f"ZERO_GAP  {shard.name}: {n_zero}/{B} ({frac:.1%}) rows "
                    "have near-zero MSE gap (y_w ≈ y_l). These pairs carry no GPO signal."
                )
                if frac >= 0.05:
                    errors.append("ERROR_" + msg)
                else:
                    warnings.append("WARN_" + msg)

        actual_totals[sig] = sig_rows

    # ------------------------------------------------------------------
    # 10. total_windows consistency
    # ------------------------------------------------------------------
    meta_totals: dict[str, int] = meta.get("total_windows", {})
    for sig, actual in actual_totals.items():
        expected = meta_totals.get(sig)
        if expected is None:
            errors.append(f"TOTALS   signal '{sig}' not in metadata.json total_windows")
        elif actual != expected:
            errors.append(f"TOTALS   signal '{sig}': metadata says {expected} windows, shards contain {actual}")
    for sig in meta_totals:
        if sig not in actual_totals:
            errors.append(f"TOTALS   signal '{sig}' in metadata.json but no shards found for it")

    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate GPO preference-pair directories before training.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "dirs",
        nargs="+",
        metavar="GPO_DIR",
        help="One or more GPO pair directories to validate.",
    )
    args = parser.parse_args()

    all_ok = True
    for raw_dir in args.dirs:
        gpo_dir = Path(raw_dir)
        print(f"\n{'=' * 60}")
        print(f"Validating: {gpo_dir}")
        if not gpo_dir.is_dir():
            print("  ERROR: directory does not exist.")
            all_ok = False
            continue

        errs, warns = validate_gpo_dir(gpo_dir)
        if warns:
            for w in warns:
                print(f"  ~ {w}")
        if errs:
            all_ok = False
            for e in errs:
                print(f"  ✗ {e}")
            print(f"  → FAILED ({len(errs)} error(s), {len(warns)} warning(s))")
        else:
            # Print summary from metadata
            meta = _load_json(gpo_dir / "metadata.json")
            cc = _load_json(gpo_dir / "collection_config.json")
            print(f"  ✓ schema_version : {meta['schema_version']}")
            print(f"  ✓ task           : {meta.get('task')}")
            print(f"  ✓ split          : {cc.get('split')} (val_fraction={cc.get('val_fraction')})")
            for sig, n in sorted(meta.get("total_windows", {}).items()):
                print(f"  ✓ {sig}: {n} windows")
            if warns:
                print(f"  → PASSED with {len(warns)} warning(s)")
            else:
                print("  → PASSED")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
