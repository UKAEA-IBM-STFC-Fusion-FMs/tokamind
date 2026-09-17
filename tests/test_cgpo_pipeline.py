"""
CGPO launcher regression tests using simulated LSF and immutable synthetic artifacts.

No test invokes a real scheduler or MAST experiment. Subprocess tests exercise the externally supplied Bash
entrypoints, dry-run guarantees and scheduler output parsing; artifact tests reject incomplete proofs.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import gpo_pipeline as pipeline
from mast_utils.gpo.protocol import (
    file_digest,
    source_fingerprint,
    validate_evaluation_manifest,
    verified_test_batches,
)
from mmt.utils.config.validator import _validate_loss_terms

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_DIR = os.environ.get("CGPO_LAUNCHER_DIR")
EXTERNAL_LAUNCHERS = pytest.mark.skipif(
    not LAUNCHER_DIR, reason="Set CGPO_LAUNCHER_DIR to test external Bash launchers."
)


@pytest.mark.parametrize("state", ["DONE", "EXIT", "RUN", "PEND", "PSUSP", "USUSP", "SSUSP", "WAIT", "PROV"])
def test_lsf_known_states(monkeypatch, state):
    monkeypatch.setattr(
        pipeline.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=f"123 {state}\n")
    )
    assert pipeline.job_status("123") == state


@pytest.mark.parametrize("output", ["", "123 UNKWN\n", "456 DONE\n", "123 DONE\n123 EXIT\n"])
def test_lsf_unknown_is_not_success(monkeypatch, output):
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=output))
    with pytest.raises(RuntimeError, match="UNKNOWN"):
        pipeline.job_status("123")


@pytest.mark.parametrize(
    ("text", "expected"), [("Job <123>\nDone successfully.", "DONE"), ("Job <123>\nExited with exit code 1.", "EXIT")]
)
def test_lsf_accounting_fallback(monkeypatch, text, expected):
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda cmd, **kw: SimpleNamespace(
            returncode=1 if cmd[0] == "bjobs" else 0, stdout="" if cmd[0] == "bjobs" else text
        ),
    )
    assert pipeline.job_status("123") == expected


@pytest.mark.parametrize("job", ["", "0", "123; touch bad", "-123", "12 13"])
def test_invalid_job_id_never_queries_scheduler(monkeypatch, job):
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *a, **k: pytest.fail("invalid job queried"))
    with pytest.raises(RuntimeError, match="Invalid job"):
        pipeline.job_status(job)


@pytest.mark.parametrize(
    ("output", "accepted"),
    [("Job <123> is submitted", True), ("", False), ("Job <0>", False), ("Job <123>\nJob <124>", False)],
)
def test_submission_id_and_quoted_worker_environment(tmp_path, monkeypatch, output, accepted):
    for key, value in {
        "TASKS": "task_1-1",
        "ARTIFACT_DIR": str(tmp_path / "artifacts with spaces"),
        "LSF_QUEUE": "q",
        "LSF_NCPUS": "1",
        "LSF_WALLTIME": "1:00",
        "LSF_MEM_GB": "2",
    }.items():
        monkeypatch.setenv(key, value)
    commands = []

    def submit(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", submit)
    p = pipeline.Pipeline()
    if accepted:
        assert p.submit("gpo", "task_1-1", [sys.executable, "worker.py"]) == "123"
        import shlex

        worker = shlex.split(commands[0][-1])
        assert f"GPO_TASKS_YAML={p.calibration('task_1-1')}" in worker
    else:
        with pytest.raises(RuntimeError, match="SUBMISSION"):
            p.submit("gpo", "task_1-1", ["python", "worker.py"])


@pytest.mark.parametrize("state", ["EXIT", "UNKWN"])
def test_failed_stage_prevents_all_downstream_work(monkeypatch, state):
    monkeypatch.setenv("TASKS", "task_1-1")
    monkeypatch.setenv("RUN_FINETUNE", "1")
    p = pipeline.Pipeline()
    monkeypatch.setattr(p, "submit", lambda *a: "123")
    monkeypatch.setattr(
        pipeline, "job_status", lambda job: state if state == "EXIT" else (_ for _ in ()).throw(RuntimeError("UNKNOWN"))
    )
    monkeypatch.setattr(p, "collect", lambda: pytest.fail("downstream stage started"))
    with pytest.raises(RuntimeError):
        p.complete()


@EXTERNAL_LAUNCHERS
def test_full_bash_dry_run_has_no_subprocesses_or_writes(tmp_path):
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    marker = tmp_path / "scheduler-was-called"
    for name in ("bsub", "bjobs", "bhist"):
        file = fakebin / name
        file.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 99\n')
        file.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(fakebin) + os.pathsep + os.environ["PATH"],
        "PYTHON_BIN": sys.executable,
        "REPO_ROOT": str(ROOT),
        "TASKS": "task_1-1 task_4-5",
        "RUNS_DIR": str(tmp_path / "runs"),
        "ARTIFACT_DIR": str(tmp_path / "artifacts"),
        "LOG_DIR": str(tmp_path / "logs"),
        "RUN_FINETUNE": "1",
    }
    before = set(tmp_path.rglob("*"))
    result = subprocess.run(
        ["bash", str(Path(LAUNCHER_DIR) / "gpo_pipeline_all.sh"), "--dry_run"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert set(tmp_path.rglob("*")) == before
    assert all(f"Stage {i}" in result.stdout for i in range(5))
    assert "DRY RUN COMPLETE" in result.stdout
    assert "base_eval-task_1-1" in result.stdout and "gpo_eval-task_4-5" in result.stdout
    assert "--gpo_tasks_yaml" in result.stdout and "--require_verified_test" in result.stdout


@EXTERNAL_LAUNCHERS
def test_compare_only_never_calibrates_or_submits(tmp_path):
    env = {
        **os.environ,
        "PYTHON_BIN": sys.executable,
        "REPO_ROOT": str(ROOT),
        "TASKS": "task_1-1",
        "ARTIFACT_DIR": str(tmp_path / "never-created"),
    }
    result = subprocess.run(
        ["bash", str(Path(LAUNCHER_DIR) / "gpo_compare_all.sh"), "--compare_only", "--dry_run"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "compare_gpo_eval.py" in result.stdout
    assert "calibrate_gpo_task.py" not in result.stdout and "bsub" not in result.stdout
    assert not (tmp_path / "never-created").exists()


@EXTERNAL_LAUNCHERS
@pytest.mark.parametrize(
    "script",
    [
        "gpo_pipeline_all.sh",
        "gpo_compare_all.sh",
        "ccc_collect_gpo_all.sh",
        "ccc_collect_gpo_pairs.sh",
        "ccc_gpo_all.sh",
        "ccc_gpo_finetune.sh",
        "ccc_finetune_all.sh",
        "ccc_finetune_task.sh",
        "ccc_eval_task.sh",
    ],
)
def test_bash_syntax(script):
    subprocess.run(["bash", "-n", str(Path(LAUNCHER_DIR) / script)], check=True)


@pytest.mark.parametrize("option", [{"mse_gap_clip": 2.0}, {"log_mse_gap": True}])
@pytest.mark.parametrize("stage", [False, True])
def test_reference_rejects_ineffective_options(option, stage):
    loss = {"terms": [{"type": "continuous_gpo", **option}]}
    cfg = {
        "train": {
            "use_reference_model": True,
            "loss": loss if not stage else {"terms": [{"type": "embed_mse"}]},
            "stages": [{"name": "gpo", "loss": loss}] if stage else [],
        }
    }
    with pytest.raises(ValueError, match="incompatible"):
        _validate_loss_terms(cfg)


def test_actual_test_batches_reject_leakage_and_count_windows():
    counts = {}
    batches = [{"shot_id": [4, 4, 5]}]
    assert list(verified_test_batches(batches, [4, 5], counts)) == batches
    assert counts == {"4": 2, "5": 1}
    with pytest.raises(ValueError, match="non-test shot"):
        list(verified_test_batches([{"shot_id": [1]}], [4, 5], counts))


def test_evaluation_proof_rejects_stale_metrics_and_missing_shots(tmp_path):
    checkpoint = tmp_path / "checkpoints/best"
    checkpoint.mkdir(parents=True)
    (checkpoint / "fake.pt").write_bytes(b"weights")
    evaluation = tmp_path / "eval_frozen"
    metrics = evaluation / "metrics/task_1-1/task_metrics.csv"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(",n_shots\ntask_1-1,1\n")
    source_config = tmp_path / f"{tmp_path.name}.yaml"
    source_config.write_text("task: task_1-1\n")
    snapshot = evaluation / "evaluation_config.yaml"
    snapshot.write_text("task: task_1-1\n")
    proof = {
        "source_config_digest": file_digest(source_config),
        "config_digest": file_digest(snapshot),
        "format": "cgpo-evaluation-v1",
        "task": "task_1-1",
        "complete": True,
        "split_manifest": {"shots": {"train": [1], "val": [2], "test": [3]}},
        "test_window_counts": {"3": 2},
        "checkpoint": source_fingerprint(tmp_path),
        "metrics_digest": file_digest(metrics),
    }
    (evaluation / "cgpo_evaluation.json").write_text(json.dumps(proof))
    assert validate_evaluation_manifest(tmp_path, evaluation, "task_1-1") == proof
    metrics.write_text(",n_shots\ntask_1-1,2\n")
    with pytest.raises(ValueError, match="stale"):
        validate_evaluation_manifest(tmp_path, evaluation, "task_1-1")
    proof["metrics_digest"] = file_digest(metrics)
    proof["test_window_counts"] = {}
    (evaluation / "cgpo_evaluation.json").write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="stale"):
        validate_evaluation_manifest(tmp_path, evaluation, "task_1-1")


def test_pair_validator_reports_missing_artifacts_without_unpacking_error(tmp_path):
    from validate_gpo_pairs import validate_gpo_dir

    errors, warnings = validate_gpo_dir(tmp_path)
    assert len(errors) == 2 and not warnings
    (tmp_path / "metadata.json").write_text('{"schema_version":3}')
    (tmp_path / "collection_config.json").write_text('{"schema_version":3,"split":"train"}')
    errors, warnings = validate_gpo_dir(tmp_path)
    assert any("Legacy" in message for message in errors)
    assert any("No .npz" in message for message in errors)


@pytest.mark.parametrize(("kind", "expected"), [("dpo", 1.0), ("ipo", 0.5)])
def test_documented_reference_gradients_and_orthogonal_freedom(kind, expected):
    import torch
    from mmt.train.losses.continuous_gpo import ContinuousGPOLoss

    term = ContinuousGPOLoss(beta=2.0, loss_type=kind)
    losses = []
    for orthogonal in (0.0, 100.0):
        pred = torch.tensor([[1.0, orthogonal]], requires_grad=True)
        loss, _ = term.compute(
            {1: pred},
            {1: torch.zeros(1, 2)},
            None,
            {1: torch.ones(1, dtype=torch.bool)},
            y_l_emb={1: torch.tensor([[1.0, 0.0]])},
            ref_preds={1: torch.tensor([[1.0, 0.0]])},
        )
        loss.backward()
        assert pred.grad.tolist()[0] == pytest.approx([expected, 0.0])
        losses.append(loss.item())
    assert losses[0] == losses[1]


def test_runtime_reference_clip_guard():
    import torch
    from mmt.train.losses.continuous_gpo import ContinuousGPOLoss

    term = ContinuousGPOLoss(mse_gap_clip=1.0)
    with pytest.raises(ValueError, match="incompatible"):
        term.compute({1: torch.ones(1, 2)}, {}, None, {}, ref_preds={})


@pytest.mark.parametrize("state", ["DONE", "EXIT", "MISSING"])
def test_complete_coordinator_with_fake_lsf_executables(tmp_path, monkeypatch, state):
    """Exercise real bsub/bjobs subprocess boundaries without executing any submitted worker."""
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    ledger = tmp_path / "jobs.json"
    bsub = fakebin / "bsub"
    bsub.write_text(f"""#!{sys.executable}
import json, os, sys
from pathlib import Path
path=Path(os.environ['SIM_LEDGER'])
jobs=json.loads(path.read_text()) if path.exists() else []
jobs.append(sys.argv[1:])
path.write_text(json.dumps(jobs))
print(f'Job <{{len(jobs)}}> is submitted')
""")
    bjobs = fakebin / "bjobs"
    bjobs.write_text(f"""#!{sys.executable}
import os, sys
state=os.environ['SIM_STATE']
if state != 'MISSING': print(sys.argv[-1], state)
""")
    bhist = fakebin / "bhist"
    bhist.write_text("#!/bin/sh\nexit 1\n")
    for path in (bsub, bjobs, bhist):
        path.chmod(0o755)
    for key, value in {
        "PATH": str(fakebin) + os.pathsep + os.environ["PATH"],
        "SIM_LEDGER": str(ledger),
        "SIM_STATE": state,
        "TASKS": "task_1-1",
        "RUN_FINETUNE": "1",
        "RUN_COLLECT": "1",
        "RUN_GPO": "1",
        "RUNS_DIR": str(tmp_path / "runs"),
        "ARTIFACT_DIR": str(tmp_path / "artifacts"),
        "LSF_QUEUE": "simulated",
        "LSF_NCPUS": "1",
        "LSF_MEM_GB": "1",
        "LSF_WALLTIME": "0:01",
    }.items():
        monkeypatch.setenv(key, value)
    p = pipeline.Pipeline()
    calls = []
    monkeypatch.setattr(p, "run", lambda command: calls.append(command))
    monkeypatch.setattr(p, "require_base", lambda task: None)
    monkeypatch.setattr(p, "require_calibration", lambda task: None)
    import mast_utils.gpo.protocol as protocol

    monkeypatch.setattr(protocol, "validate_evaluation_manifest", lambda *args: {})
    if state == "DONE":
        p.complete()
        jobs = json.loads(ledger.read_text())
        stages = [job[job.index("-J") + 1].split("-task_")[0] for job in jobs]
        assert stages == ["finetune", "collect", "base_eval", "gpo", "gpo_eval"]
        assert "GPO_TASKS_YAML=" + str(p.calibration("task_1-1")) in jobs[3][-1]
        scripts = [Path(command[1]).name for command in calls]
        assert scripts[-1] == "compare_gpo_eval.py"
        assert scripts.count("calibrate_gpo_task.py") == 1
    else:
        with pytest.raises(RuntimeError, match="FAILED|UNKNOWN"):
            p.complete()
        assert len(json.loads(ledger.read_text())) == 1
        assert not calls


def test_frozen_calibration_rejects_changed_external_input_recipe(tmp_path, monkeypatch):
    import yaml
    from mast_utils.gpo.protocol import config_fingerprint, digest

    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("TASKS", "task_1-1")
    input_recipe = tmp_path / "input.yaml"
    input_recipe.write_text("tasks: {}\n")
    monkeypatch.setenv("GPO_TASKS_YAML", str(input_recipe))
    p = pipeline.Pipeline()
    pairs = p.pairs("task_1-1")
    pairs.mkdir(parents=True)
    (pairs / "collection_config.json").write_text("{}")
    meta = {
        "format": "cgpo-calibration-v1",
        "task": "task_1-1",
        "run_tag": p.tag,
        "collection_digest": digest({}),
        "input_recipe": str(input_recipe),
        "input_recipe_digest": file_digest(input_recipe),
        "recipe_digest": digest({}),
        "inputs": config_fingerprint(tmp_path / "scripts_mast/configs"),
    }
    path = p.calibration("task_1-1")
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump({"calibration": meta, "tasks": {"task_1-1": {}}}))
    (path.parent / "calibration_summary.json").write_text(json.dumps({"calibration": meta}))
    p.require_calibration("task_1-1")
    input_recipe.write_text("tasks: {task_1-1: {train: {use_reference_model: false}}}\n")
    with pytest.raises(ValueError, match="mismatch"):
        p.require_calibration("task_1-1")


def test_calibration_preserves_mse_only_recipe():
    from calibrate_gpo_task import _effective_recipe

    recipe = _effective_recipe(
        "task_4-2", ROOT / "scripts_mast/configs/mmt/tasks/gpo_tasks_embed_mse_only.yaml", ROOT / "scripts_mast/configs"
    )
    assert [term["type"] for term in recipe["train"]["loss"]["terms"]] == ["embed_mse"]


def test_frozen_null_field_cannot_disappear():
    from mast_utils.gpo.protocol import require_config_subset

    with pytest.raises(ValueError, match="field missing"):
        require_config_subset({"optional": None}, {})


def test_all_missing_comparisons_produce_explicit_incomplete_report(tmp_path, monkeypatch):
    import compare_gpo_eval

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_gpo_eval.py",
            "--task",
            "task_1-1",
            "--base_run",
            "missing-base",
            "--gpo_run",
            "missing-gpo",
            "--runs_dir",
            str(tmp_path),
            "--report_dir",
            str(tmp_path / "reports"),
        ],
    )
    with pytest.raises(SystemExit) as status:
        compare_gpo_eval.main()
    assert status.value.code == 1
    reports = list((tmp_path / "reports").glob("*.md"))
    assert len(reports) == 1
    assert "INCOMPLETE: 0/1" in reports[0].read_text()
