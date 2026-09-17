"""Regression tests for CGPO configuration and shot-disjoint supervision."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from calibrate_gpo_task import _load_signal
from mast_utils.gpo.dataset import GpoPairDataset
from mast_utils.gpo.protocol import (
    PROTOCOL,
    apply_source_contract,
    save_run_snapshot,
    validate_collection,
    validate_shots,
)
from mast_utils.gpo.writer import GpoPairWriter
from run_gpo_finetune import _GpoBatchInjector, apply_gpo_config

from mmt.train.losses import build_loss_aggregator


@pytest.fixture
def source(tmp_path):
    run = tmp_path / "source"
    run.mkdir()
    cfg = {
        "task": "task_1-1",
        "model": {"name": "mmt"},
        "embeddings": {"x": 1},
        "preprocess": {"chunks": {"input": {"max_chunks": 7}}, "valid_windows": {"window_stride_sec": 0.04}},
        "data": {"split": "temporal", "subset_size": None},
    }
    (run / "source.yaml").write_text(yaml.safe_dump(cfg))
    return run, cfg


@pytest.mark.parametrize("split", ["random", "temporal"])
def test_real_gpo_hook_preserves_source_and_local_settings(tmp_path, source, split):
    run, src = source
    src["data"]["split"] = split
    (run / "source.yaml").write_text(yaml.safe_dump(src))
    import shutil

    configs = tmp_path / "configs"
    shutil.copytree(Path("scripts_mast/configs/mmt"), configs / "mmt")
    (configs / "local_overrides.yaml").write_text(
        "data:\n  local: true\n  local_path: /local/data\nloader:\n  batch_size: 3\n"
    )
    cfg = {
        "task": "task_1-1",
        "model_profile": "mmt",
        "run_id": "new",
        "data": {},
        "model_source": {"run_dir": str(run)},
        "train": {},
    }
    apply_gpo_config(cfg, "finetune", configs_root=str(configs), task="task_1-1", model_profile="mmt")
    assert cfg["data"]["split"] == cfg["model_source"]["data_split"] == split
    assert cfg["data"]["local"] and cfg["data"]["local_path"] == "/local/data"
    assert cfg["preprocess"] == src["preprocess"]
    assert cfg["embeddings"] == src["embeddings"]
    assert cfg["loader"]["batch_size"] == 3
    assert [s["name"] for s in cfg["train"]["stages"]] == ["gpo"]
    assert cfg["run_id"] == "new"


@pytest.mark.parametrize(
    ("override", "key"),
    [
        ({"data": {"split": "random"}}, "data.split"),
        ({"data": {"subset_size": 10}}, "data.subset_size"),
        ({"embeddings": {"dct3d": {"tuning": {"enable": True}}}}, "embeddings.dct3d"),
    ],
)
def test_incompatible_local_override_rejected(tmp_path, source, override, key):
    run, _ = source
    (tmp_path / "local_overrides.yaml").write_text(yaml.safe_dump(override))
    with pytest.raises(ValueError) as error:
        apply_source_contract({"task": "task_1-1", "model_source": {"run_dir": str(run)}}, tmp_path)
    assert key in str(error.value)
    assert "source=" in str(error.value) and "override=" in str(error.value)


@pytest.mark.parametrize("repeat_subset", [False, True])
def test_omitted_or_identical_local_subset_preserves_source(tmp_path, source, repeat_subset):
    run, src = source
    src["data"]["subset_size"] = 10
    (run / "source.yaml").write_text(yaml.safe_dump(src))
    local = {"data": {"local": True, "local_path": "/other-machine"}}
    if repeat_subset:
        local["data"]["subset_size"] = 10
    (tmp_path / "local_overrides.yaml").write_text(yaml.safe_dump(local))
    cfg = {"task": "task_1-1", "model_source": {"run_dir": str(run)}, "data": {"cache": {}}}
    apply_source_contract(cfg, tmp_path)
    assert cfg["data"]["subset_size"] == 10
    assert cfg["data"]["local_path"] == "/other-machine"


@pytest.mark.parametrize("resume", [False, True])
def test_incomplete_run_preserved_with_actionable_error(tmp_path, resume):
    root = tmp_path / "incomplete"
    root.mkdir()
    log = root / "run.log"
    log.write_text("previous attempt")
    cfg = SimpleNamespace(paths={"run_dir": str(root)}, run_id="incomplete", raw={})
    with pytest.raises((ValueError, FileExistsError)) as error:
        save_run_snapshot(cfg, resume=resume)
    assert "gpo_request.yaml" in str(error.value)
    assert "new tag" in str(error.value)
    assert log.read_text() == "previous attempt"
    assert not (root / "gpo_request.yaml").exists()


def make_pairs(tmp_path):
    shots = {"train": [1, 2], "val": [3], "test": [4]}
    contract = {"split_manifest": {"shots": shots}}
    writer = GpoPairWriter(
        tmp_path / "pairs", {"split": "both", "val_fraction": 0.0, "protocol": PROTOCOL, "contract": contract}
    )
    for sig in ["a", "b"]:
        writer.add_batch(
            output_name=sig,
            y_w_emb=np.zeros((6, 2)),
            y_l_emb=np.array([[1.0, 1.0]] * 4 + [[1000.0, 1000.0]] * 2),
            mask=np.ones(6, bool),
            shot_ids=np.array([1, 1, 2, 2, 3, 3]),
            window_indices=np.array([0, 1, 0, 1, 0, 1]),
            native_nrmse=np.array([1, 2, 3, 4, 1000, 1000], dtype=np.float32),
        )
    writer.finalize()
    return tmp_path / "pairs", contract


def test_split_precedes_filters_and_all_signals_stay_together(tmp_path):
    pairs, _ = make_pairs(tmp_path)
    tr = GpoPairDataset(pairs, "train", native_nrmse_percentiles=(0, 100), max_pairs_per_shot_percentile=90)
    va = GpoPairDataset(
        pairs, "val", native_nrmse_percentiles=(0, 100), max_pairs_per_shot_percentile=90, filter_state=tr.filter_state
    )
    assert {x["shot_id"] for x in tr} == {1, 2}
    assert tr.filter_state["native_bounds"] == {"a": (1.0, 4.0), "b": (1.0, 4.0)}
    assert len(va) == 0  # Validation outliers do not widen training-fitted thresholds.
    unfiltered = GpoPairDataset(pairs, "val", filter_state={"native_bounds": {}, "caps": {}})
    assert {x["shot_id"] for x in unfiltered} == {3}
    for sig in ["a", "b"]:
        only = GpoPairDataset(pairs, "train", signal_name=sig)
        assert {x["shot_id"] for x in only} == {1, 2}


def test_calibration_ignores_extreme_validation_pairs(tmp_path):
    pairs, _ = make_pairs(tmp_path)
    arrays = _load_signal(sorted(pairs.glob("a*.npz")), max_windows=0, train_shots={1, 2})
    assert arrays["shot_id"].tolist() == [1, 1, 2, 2]
    assert np.max(arrays["y_l"]) == 1.0


def test_injection_does_not_expose_validation_targets_to_training(tmp_path):
    pairs, _ = make_pairs(tmp_path)
    tr = GpoPairDataset(pairs, "train")
    va = GpoPairDataset(pairs, "val", filter_state=tr.filter_state)

    def batch(shot):
        return {
            "shot_id": [shot],
            "window_index": [0],
            "output_emb": {1: torch.zeros(1, 2)},
            "output_mask": {1: torch.ones(1, dtype=torch.bool)},
        }

    train_loader = _GpoBatchInjector([batch(1), batch(2)], tr, {"a": 1})
    val_loader = _GpoBatchInjector([batch(3)], va, {"a": 1})
    trained = set()
    agg = build_loss_aggregator({"terms": [{"type": "embed_mse", "weight": 0.2}]})
    for b in train_loader:
        p = torch.ones(1, 2, requires_grad=True)
        loss, _ = agg.compute({1: p}, b)
        loss.backward()
        assert p.grad.abs().sum() > 0
        trained.update(b["shot_id"])
    assert trained.isdisjoint(s for b in val_loader for s in b["shot_id"])


def test_contract_and_snapshot_are_strict(tmp_path):
    pairs, expected = make_pairs(tmp_path)
    cc = json.loads((pairs / "collection_config.json").read_text())
    validate_collection(cc, expected)
    with pytest.raises(ValueError, match="Legacy"):
        validate_collection({}, expected)
    changed = copy.deepcopy(expected)
    changed["task"] = "other"
    with pytest.raises(ValueError, match="differs"):
        validate_collection(cc, changed)
    with pytest.raises(ValueError, match="disjoint"):
        validate_shots({"train": [1], "val": [1], "test": [2]})
    root = tmp_path / "run"
    root.mkdir()
    cfg = SimpleNamespace(paths={"run_dir": str(root)}, run_id="run", raw={"task": "task_1-1"})
    save_run_snapshot(cfg)
    original = (root / "gpo_request.yaml").read_bytes()
    with pytest.raises(FileExistsError):
        save_run_snapshot(cfg)
    save_run_snapshot(cfg, resume=True)
    cfg.raw["task"] = "other"
    with pytest.raises(ValueError):
        save_run_snapshot(cfg, resume=True)
    assert (root / "gpo_request.yaml").read_bytes() == original


def test_shard_tampering_is_detected(tmp_path):
    pairs, expected = make_pairs(tmp_path)
    cc = json.loads((pairs / "collection_config.json").read_text())
    validate_collection(cc, expected, pairs)
    with next(pairs.glob("*.npz")).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="shard contents"):
        validate_collection(cc, expected, pairs)


def test_resolved_snapshot_is_not_overwritten(tmp_path):
    from mast_utils.embedding_resolution.resolve import save_config_snapshot

    cfg = SimpleNamespace(run_id="run", raw={"gpo_provenance": {"protocol": PROTOCOL}, "train": {"resume": False}})
    path = save_config_snapshot(cfg, tmp_path)
    before = path.read_bytes()
    cfg.raw["train"]["resume"] = True
    save_config_snapshot(cfg, tmp_path)
    cfg.raw["train"]["new_parameter"] = 2
    with pytest.raises(ValueError, match="differs"):
        save_config_snapshot(cfg, tmp_path)
    assert path.read_bytes() == before


def test_entrypoints_route_distinct_mast_loaders(tmp_path, monkeypatch):
    """Use the real config loader and real pair writer/dataset; stub only costly MAST/model work."""
    import logging

    import run_collect_gpo_pairs as collector
    import run_gpo_finetune as trainer
    import run_eval as evaluator
    import calibrate_gpo_task as calibrator
    from mast_utils import load_experiment_config
    from mast_utils.gpo import protocol

    from mmt.utils.config.experiment import finalize

    monkeypatch.setattr(finalize, "REPO_ROOT", tmp_path)
    src = load_experiment_config(
        task="task_1-1",
        phase="finetune",
        finetune_init="scratch",
        model_profile="mmt",
        embeddings_profile="dct3d",
        tag="source",
        save_config=False,
    )
    root = Path(src.paths["run_dir"])
    (root / f"{src.run_id}.yaml").write_text(yaml.safe_dump(src.raw))
    (root / "checkpoints" / "best").mkdir(parents=True)
    (root / "checkpoints" / "best" / "fake.pt").write_bytes(b"model")
    manifest = {"protocol": PROTOCOL, "shots": {"train": [1, 2], "val": [3], "test": [4]}, "assets": {}}
    monkeypatch.setattr(protocol, "benchmark_manifest", lambda data: manifest)
    spec = SimpleNamespace(signal_id=1, name="a")
    registry = SimpleNamespace(specs_for_role=lambda role: [spec])

    def batch(shot):
        return {
            "shot_id": [shot],
            "window_index": [0],
            "output_emb": {1: torch.zeros(1, 2)},
            "output_mask": {1: torch.ones(1, dtype=torch.bool)},
            "output_native": {1: torch.zeros(1, 2)},
        }

    captured = []

    def windows(**kw):
        assert kw["mast_datasets"] == {"train": "train_shots", "val": "val_shots"}
        captured.append(kw)
        return {"train": {"loader": [batch(1), batch(2)]}, "val": {"loader": [batch(3)]}}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, b):
            return {"pred": {1: torch.ones(1, 2) * self.weight}}

    for module in (collector, trainer, evaluator):
        monkeypatch.setattr(module, "init_run_context", lambda **kw: (torch.device("cpu"), logging.getLogger("test")))
        monkeypatch.setattr(module, "build_mast_datasets", lambda **kw: ({}, "train_shots", "val_shots", None))
        monkeypatch.setattr(module, "load_task_definition", lambda **kw: {"name": "task_1-1"})
        monkeypatch.setattr(module, "build_signals_by_role_from_task_definition", lambda **kw: {})
        monkeypatch.setattr(module, "build_window_data", windows)
        monkeypatch.setattr(module, "build_model_and_optional_warmstart", lambda **kw: Model())
        monkeypatch.setattr(module, "load_best_weights", lambda **kw: (0, 0.0, {}))
        monkeypatch.setattr(module, "build_decoders", lambda **kw: {1: torch.nn.Identity()})
        monkeypatch.setattr(module, "extract_signal_stats", lambda **kw: {"a": {"mean": 0.0, "std": 1.0}})
    monkeypatch.setattr(collector, "resolve_eval_embeddings", lambda **kw: (registry, {}))

    def resolve_training_embeddings(**kw):
        cfg = kw["cfg_mmt"]
        Path(cfg.paths["run_dir"], f"{cfg.run_id}.yaml").write_text(yaml.safe_dump(cfg.raw))
        return registry, {}

    monkeypatch.setattr(trainer, "resolve_finetune_embeddings", resolve_training_embeddings)
    monkeypatch.setattr(evaluator, "resolve_eval_embeddings", lambda **kw: (registry, {}))
    monkeypatch.setattr(
        collector,
        "_parse_args",
        lambda: SimpleNamespace(
            task="task_1-1",
            model_source=str(root),
            tag="shots-v1",
            split="both",
            val_fraction=0.0,
            train_fraction=1.0,
            multi_signal="joint",
            shard_size=2,
            overwrite=False,
        ),
    )
    collector.main()
    from calibrate_gpo_task import _validate_dir
    from validate_gpo_pairs import validate_gpo_dir

    pair_path = root / "gpo_pairs_shots-v1"
    assert not validate_gpo_dir(pair_path)[0]
    assert _validate_dir(pair_path)[0]
    assert captured[0]["window_test_mode"] is False
    alternate = tmp_path / "tasks.yaml"
    alternate.write_text(
        "tasks:\n  task_1-1:\n    train:\n      gpo_dataset:\n        native_nrmse_percentiles: null\n        max_pairs_per_shot_percentile: null\n"
    )
    original_recipe = alternate.read_bytes()
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate_gpo_task.py",
            str(pair_path),
            "--gpo_tasks_yaml",
            str(alternate),
            "--output_dir",
            str(tmp_path / "calibration"),
            "--run_tag",
            "corrected",
            "--no_plots",
            "--report",
        ],
    )
    command = list(sys.argv)
    monkeypatch.setattr(sys, "argv", [*command, "--dry_run"])
    before_dry = set(tmp_path.rglob("*"))
    with pytest.raises(SystemExit) as dry_status:
        calibrator.main()
    assert dry_status.value.code == 0
    assert set(tmp_path.rglob("*")) == before_dry
    monkeypatch.setattr(sys, "argv", command)
    with pytest.raises(SystemExit) as status:
        calibrator.main()
    assert status.value.code == 0
    assert alternate.read_bytes() == original_recipe
    frozen = tmp_path / "calibration/task_1-1/gpo_tasks.yaml"
    artifact = yaml.safe_load(frozen.read_text())
    assert artifact["calibration"]["run_tag"] == "corrected"
    assert artifact["tasks"]["task_1-1"]["train"]["loss"]["terms"][0]["beta"] == pytest.approx(1.0)
    monkeypatch.setenv("GPO_TASKS_YAML", str(frozen))
    monkeypatch.delenv("GPO_RESUME", raising=False)
    monkeypatch.setattr(
        trainer,
        "_parse_args",
        lambda: SimpleNamespace(
            task="task_1-1",
            model_source=str(root),
            tag="corrected",
            gpo_dir=str(root / "gpo_pairs_shots-v1"),
            model_profile="mmt",
            emb_profile="dct3d",
        ),
    )

    def train(**kw):
        train_shots = {s for b in kw["train_loader"] for s in b["shot_id"]}
        val_shots = {s for b in kw["val_loader"] for s in b["shot_id"]}
        assert train_shots == {1, 2}
        assert val_shots == {3}
        assert not train_shots & val_shots
        best = Path(kw["run_dir"]) / "checkpoints/best"
        best.mkdir(parents=True)
        (best / "fake.pt").write_bytes(b"trained model")
        return {"ok": True}

    monkeypatch.setattr(trainer, "train_finetune", train)
    trainer.main()
    with pytest.raises(FileExistsError, match="already exists"):
        trainer.main()
    # Execute both evaluation entrypoints and the strict comparison on a synthetic test shot.
    from compare_gpo_eval import _compare_task

    gpo_root = root.parent / f"ft-task_1-1-ws-{root.name}-mmt-corrected"
    monkeypatch.setattr(evaluator, "build_window_data", lambda **kw: {"test": {"loader": [batch(4)]}})

    def benchmark(**kw):
        assert [s for b in kw["dataloader"] for s in b["shot_id"]] == [4]
        path = Path(kw["run_dir"]) / "metrics/task_1-1/task_metrics.csv"
        path.parent.mkdir(parents=True)
        path.write_text(",NRMSE_mean,NMAE_mean,RMSE_mean,MAE_mean,n_shots\ntask_1-1,1,1,1,1,1\n")
        return {"metrics_task_dir": str(path.parent)}

    monkeypatch.setattr(evaluator, "evaluate_benchmark_and_diagnostics", benchmark)
    for model_root in (root, gpo_root):
        monkeypatch.setattr(
            evaluator,
            "parse_args_eval",
            lambda model_root=model_root: SimpleNamespace(
                task="task_1-1", model_source=str(model_root), tag="verified", cgpo_protocol=pair_path
            ),
        )
        evaluator.main()
    comparison = _compare_task(
        "task_1-1",
        str(root),
        str(gpo_root),
        root.parent,
        "verified",
        "verified",
        ["NRMSE_mean"],
        False,
        require_verified_test=True,
    )
    assert comparison is not None
    (gpo_root / "eval_verified/metrics/task_1-1/task_metrics.csv").write_text(",NRMSE_mean,n_shots\ntask_1-1,9,1\n")
    assert (
        _compare_task(
            "task_1-1",
            str(root),
            str(gpo_root),
            root.parent,
            "verified",
            "verified",
            ["NRMSE_mean"],
            False,
            require_verified_test=True,
        )
        is None
    )

    # Resume validation must precede even the construction of expensive MAST datasets.
    monkeypatch.setenv("GPO_RESUME", "1")

    def forbidden_datasets(**kwargs):
        """Assert that invalid resume state fails before MAST dataset construction."""
        pytest.fail("Resume preflight should fail before preparing datasets")

    monkeypatch.setattr(trainer, "build_mast_datasets", forbidden_datasets)
    with pytest.raises(FileNotFoundError, match="Missing verified resume checkpoint"):
        trainer.main()
    (root / "checkpoints" / "best" / "fake.pt").write_bytes(b"changed model")
    with pytest.raises(ValueError, match="contract differs"):
        trainer.main()


@pytest.mark.parametrize(
    "task", [f"task_{g}-{n}" for g, count in [(1, 3), (2, 3), (3, 3), (4, 5)] for n in range(1, count + 1)]
)
def test_all_benchmark_gpo_configs_resolve_with_source_contract(tmp_path, monkeypatch, task):
    from mast_utils import load_experiment_config, validate_mast_config

    from mmt.utils import validate_config
    from mmt.utils.config.experiment import finalize

    monkeypatch.setattr(finalize, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("GPO_TASKS_YAML", raising=False)
    src = load_experiment_config(task=task, phase="finetune", finetune_init="scratch", save_config=False)
    root = Path(src.paths["run_dir"])
    (root / f"{src.run_id}.yaml").write_text(yaml.safe_dump(src.raw))
    (root / "checkpoints" / "best").mkdir(parents=True)
    cfg = load_experiment_config(
        task=task,
        phase="finetune",
        finetune_init="warmstart",
        model_source=str(root),
        tag="cgpo-shots",
        save_config=False,
        integration_hook=lambda merged, phase: apply_gpo_config(
            merged, phase, configs_root="scripts_mast/configs", task=task, model_profile="mmt"
        ),
    )
    validate_config(cfg=cfg)
    validate_mast_config(cfg=cfg)
    assert cfg.data["split"] == src.data["split"]
    assert cfg.raw["preprocess"] == src.raw["preprocess"]
    assert [s["name"] for s in cfg.train["stages"]] == ["gpo"]
