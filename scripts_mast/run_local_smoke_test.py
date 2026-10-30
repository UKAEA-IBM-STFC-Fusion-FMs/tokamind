"""
run_local_smoke_test.py
=======================

Automated end-to-end local smoke test for Continuous Generalized Preference Optimization (CGPO).
Designed for running on local workstations (Mac with MPS or CPU, Linux) without access to an LSF
compute cluster or large datasets.

Executes all 5 stages in order:
  Stage 0: Base model fine-tuning (scratch init)
  Stage 1: GPO pair collection (official train & test splits)
  Stage 2: CGPO task calibration (computes beta scale heuristic & blacklists)
  Stage 3: CGPO fine-tuning (warmstarted from base model, inherits all embeddings)
  Stage 4: Verified evaluation of both Base and CGPO models (test split)
  Stage 5: Comparison and delta report (verifies complete test coverage)

Prerequisites:
  - Python environment with PyTorch, tokamark, zarr, etc.
  - At least 1-3 MAST shot .zarr directories configured in scripts_mast/configs/local_overrides.yaml
    with `data.local: true` and `data.subset_size: 1`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_stage(cmd: list[str], stage_name: str, env: dict[str, str] | None = None) -> None:
    """Execute a stage command with real-time output streaming and status checking."""
    print(f"\n{'=' * 80}")
    print(f"▶ STARTING: {stage_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")
    start = time.perf_counter()

    full_env = os.environ.copy()
    full_env["PYTHONPATH"] = f"src:scripts_mast:{full_env.get('PYTHONPATH', '')}"
    if env:
        full_env.update(env)

    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=full_env)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        print(f"\n❌ FAILED: {stage_name} (exit code {proc.returncode}) after {elapsed:.1f}s")
        sys.exit(proc.returncode)

    print(f"\n✔ COMPLETED: {stage_name} in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run complete local CGPO smoke test pipeline on Mac/workstation.",
        allow_abbrev=False,
    )
    parser.add_argument("--task", type=str, default="task_1-1", help="Task ID (e.g. task_1-1).")
    parser.add_argument("--tag", type=str, default="local_smoke", help="Run tag for local testing.")
    parser.add_argument("--emb_profile", type=str, default="dct3d", help="Embedding profile.")
    parser.add_argument("--model_profile", type=str, default="mmt", help="Model profile.")
    parser.add_argument("--clean", action="store_true", help="Clean prior runs with this tag before starting.")
    parser.add_argument("--skip_base", action="store_true", help="Skip Stage 0 base fine-tuning if already done.")
    parser.add_argument("--eval_only", action="store_true", help="Run only Stage 4a, 4b, and 5 (skip Stage 0-3 training).")
    args = parser.parse_args()

    py = sys.executable
    task = args.task
    tag = args.tag
    base_run_id = f"ft-{task}-scratch-{args.model_profile}-{tag}"
    gpo_run_id = f"ft-{task}-ws-{base_run_id}-{args.model_profile}-{tag}"
    pairs_dir = REPO_ROOT / "runs" / base_run_id / f"gpo_pairs_{tag}"
    artifact_dir = REPO_ROOT / "cgpo_artifacts" / tag
    calibrated_tasks_yaml = artifact_dir / task / "gpo_tasks.yaml"

    if args.clean and not args.eval_only:
        print(f"Cleaning prior runs for tag={tag}...")
        for p in [
            REPO_ROOT / "runs" / base_run_id,
            REPO_ROOT / "runs" / gpo_run_id,
            artifact_dir,
        ]:
            if p.exists():
                shutil.rmtree(p)
                print(f"Removed {p}")

    if not args.eval_only:
        # Stage 0: Base Finetuning
        if not args.skip_base:
            run_stage(
                cmd=[
                    py,
                    "scripts_mast/run_finetune.py",
                    "--task", task,
                    "--init", "scratch",
                    "--emb_profile", args.emb_profile,
                    "--tag", tag,
                ],
                stage_name="Stage 0: Base Model Scratch Fine-Tuning",
            )

        # Stage 1: Collect GPO Pairs
        run_stage(
            cmd=[
                py,
                "scripts_mast/run_collect_gpo_pairs.py",
                "--task", task,
                "--model_source", base_run_id,
                "--split", "both",
                "--val_fraction", "0",
                "--tag", tag,
                "--overwrite",
            ],
            stage_name="Stage 1: GPO Pair Collection (Train + Test splits)",
        )

        # Stage 2: Calibration
        run_stage(
            cmd=[
                py,
                "scripts_mast/calibrate_gpo_task.py",
                str(pairs_dir),
                "--output_dir", str(artifact_dir),
                "--run_tag", tag,
                "--no_plots",
            ],
            stage_name="Stage 2: CGPO Task Calibration (Beta & Blacklist)",
        )

        # Stage 3: GPO Finetuning
        run_stage(
            cmd=[
                py,
                "scripts_mast/run_gpo_finetune.py",
                "--task", task,
                "--model_source", base_run_id,
                "--gpo_dir", str(pairs_dir),
                "--model_profile", args.model_profile,
                "--emb_profile", args.emb_profile,
                "--tag", tag,
            ],
            stage_name="Stage 3: CGPO Fine-Tuning",
            env={"GPO_TASKS_YAML": str(calibrated_tasks_yaml)},
        )

    # Stage 4a: Evaluate Base Model
    # Remove any unverified eval folder from Stage 0 if present
    unverified_eval = REPO_ROOT / "runs" / base_run_id / f"eval_{tag}"
    if unverified_eval.exists():
        shutil.rmtree(unverified_eval)

    run_stage(
        cmd=[
            py,
            "scripts_mast/run_eval.py",
            "--task", task,
            "--model_source", base_run_id,
            "--tag", tag,
            "--cgpo_protocol", str(pairs_dir),
        ],
        stage_name="Stage 4a: Base Model Verified Test Evaluation",
    )

    # Stage 4b: Evaluate GPO Model
    run_stage(
        cmd=[
            py,
            "scripts_mast/run_eval.py",
            "--task", task,
            "--model_source", gpo_run_id,
            "--tag", tag,
            "--cgpo_protocol", str(pairs_dir),
        ],
        stage_name="Stage 4b: GPO Model Verified Test Evaluation",
    )

    # Stage 5: Comparison
    run_stage(
        cmd=[
            py,
            "scripts_mast/compare_gpo_eval.py",
            "--tasks", task,
            "--base_ft_tag", tag,
            "--gpo_version", tag,
            "--eval_tag", tag,
            "--require_verified_test",
        ],
        stage_name="Stage 5: Official GPO vs Base Evaluation Comparison",
    )

    print("\n" + "=" * 80)
    print("🎉 ALL STAGES PASSED! Local CGPO smoke test completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
