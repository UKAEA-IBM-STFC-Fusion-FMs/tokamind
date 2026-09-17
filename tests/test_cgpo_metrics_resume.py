"""
CGPO regression tests for epoch metrics, checkpoint selection and strict training resume.

Synthetic CPU predictions exercise masked signals, sparse preference pairs, MSE-only ablations and batch partitioning.
A small model with dropout, shuffled batches and Python/NumPy randomness compares continuous training against resume
within and between stages. Additional cases reject missing/corrupt checkpoint files and changes to the saved training
contract. These deterministic tests do not replace a full MAST run or establish GPU bitwise reproducibility.
"""

import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from mmt.checkpoints.strict import inspect_resume, training_signature
from mmt.train import loop
from mmt.train.loop_utils import run_one_epoch
from mmt.train.losses import build_loss_aggregator
from mmt.train.metrics import select_metric


class PredictionModel(nn.Module):
    """
    Return supplied predictions through a trainable scalar for epoch-metric tests.
    """

    def __init__(self):
        """
        Initialize the synthetic model parameters and output contract.
        """

        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, batch):
        """
        Produce the synthetic prediction mapping expected by the training loop.

        Parameters
        ----------
        batch : dict
            Collated synthetic inputs.

        Returns
        -------
        dict
            Prediction tensors under the ``pred`` key.
        """

        return {"pred": {key: value * self.scale for key, value in batch["pred"].items()}}


def evaluate(rows, batch_size, terms):
    """
    Evaluate synthetic rows using the production epoch runner.

    Parameters
    ----------
    rows : list[dict]
        Per-window predictions, targets and masks.
    batch_size : int
        Number of rows per batch.
    terms : list[dict]
        Loss definitions passed to the aggregator.

    Returns
    -------
    tuple[float, dict[str, float]]
        Epoch objective and reduced metric logs.
    """

    return run_one_epoch(
        PredictionModel(),
        DataLoader(rows, batch_size=batch_size),
        None,
        None,
        None,
        device=torch.device("cpu"),
        amp_enabled=False,
        loss_aggregator=build_loss_aggregator({"terms": terms}),
        grad_accum_steps=1,
        train=False,
        global_step=0,
    )[:2]


def metric_rows():
    """
    Construct two signals with unequal validity and sparse preference-pair coverage.

    Returns
    -------
    list[dict]
        Five deterministic windows with predictions, masks, targets and fixed reference predictions.
    """

    return [
        {
            "pred": {1: torch.tensor([value]), 2: torch.tensor([2.0 * value])},
            "output_emb": {1: torch.zeros(1), 2: torch.zeros(1)},
            "output_mask": {1: torch.tensor(True), 2: torch.tensor(index != 2)},
            "y_l_emb": {1: torch.ones(1), 2: torch.ones(1)},
            "gpo_pair_mask": {1: torch.tensor(index % 2 == 0), 2: torch.tensor(index == 0)},
            "ref_preds": {1: torch.zeros(1), 2: torch.zeros(1)},
        }
        for index, value in enumerate([1.0, 10.0, 2.0, 3.0, 4.0])
    ]


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("with_pairs", [False, True])
def test_mse_only_1_100_is_50_5(batch_size, with_pairs):
    """
    Verify MSE-only averaging includes unpaired windows and is independent of batch size.
    """

    rows = metric_rows()[:2]
    for row in rows:
        for key in row:
            row[key].pop(2)
        if not with_pairs:
            row.pop("y_l_emb")
            row.pop("gpo_pair_mask")
    objective, logs = evaluate(rows, batch_size, [{"type": "embed_mse", "weight": 1.0}])
    assert objective == pytest.approx(50.5)
    assert logs["mse"] == 50.5
    assert logs["mse/count/1"] == logs["EmbedMSELoss/count/1"] == 2


@pytest.mark.parametrize("batch_size", [2, 3, 5])
def test_epoch_metrics_independent_of_batch_partition(batch_size):
    """
    Verify per-signal reductions stay fixed while batch coverage can change.
    """

    terms = [{"type": "continuous_gpo", "weight": 0.8, "sft_weight": 0.3}, {"type": "embed_mse", "weight": 0.2}]
    expected, reference = evaluate(metric_rows(), 1, terms)
    actual, logs = evaluate(metric_rows(), batch_size, terms)
    assert actual == pytest.approx(expected)
    # Batch coverage deliberately depends on packing. All sample metrics must not.
    for key in reference.keys() - {"batches", "pair_batches", "pair_batch_coverage"}:
        assert logs[key] == pytest.approx(reference[key]), key
    assert logs["pair_windows"] == 3
    assert logs["pair_window_coverage"] == 0.6
    assert logs["ContinuousGPOLoss_0/count/1"] == 3
    assert logs["ContinuousGPOLoss_0/count/2"] == 1
    assert logs["mse/count/2"] == 4
    assert logs["pair_batch_coverage"] == 1.0
    assert reference["pair_batch_coverage"] == 0.6


@pytest.mark.parametrize("metric", ["mse", "objective"])
def test_requested_metric_without_observations_fails(metric):
    """
    Reject checkpoint selection when validation has no valid targets.
    """

    rows = metric_rows()
    for row in rows:
        row["output_mask"] = {1: torch.tensor(False), 2: torch.tensor(False)}
    objective, logs = evaluate(rows, 2, [{"type": "embed_mse"}])
    with pytest.raises(RuntimeError, match="no valid observations"):
        select_metric(metric, objective, logs)


class TrainingModel(nn.Module):
    """
    Exercise Torch dropout and Python/NumPy random streams in a checkpointable CPU model.
    """

    def __init__(self):
        """
        Initialize the synthetic model parameters and output contract.
        """

        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(2, 4), nn.Dropout(0.3), nn.Linear(4, 1))
        self.output_specs = [SimpleNamespace(name="a", signal_id=1)]

    def get_named_blocks(self):
        """
        Expose the model block used by the production checkpoint API.

        Returns
        -------
        dict[str, torch.nn.Module]
            The trainable backbone block.
        """

        return {"backbone": self.backbone}

    def forward(self, batch):
        """
        Produce the synthetic prediction mapping expected by the training loop.

        Parameters
        ----------
        batch : dict
            Collated synthetic inputs.

        Returns
        -------
        dict
            Prediction tensors under the ``pred`` key.
        """

        value = self.backbone(batch["x"])
        if self.training:
            value = value + 0.01 * (random.random() + np.random.random())
        return {"pred": {1: value}}


def seeded_setup(seed=123):
    """
    Initialize a small model and loaders with reproducible independent random streams.

    Parameters
    ----------
    seed : int
        Seed for Python, NumPy, Torch and loader generators. Optional. Default: 123.

    Returns
    -------
    tuple[TrainingModel, list[DataLoader]]
        CPU model and separate shuffled training/unshuffled validation loaders.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = TrainingModel()
    rows = [
        {
            "x": torch.tensor([i / 10.0, 1.0]),
            "output_emb": {1: torch.tensor([i / 20.0])},
            "output_mask": {1: torch.tensor(True)},
            "y_l_emb": {1: torch.tensor([i / 20.0 + 1.0])},
            "ref_preds": {1: torch.tensor([0.0])},
            "gpo_pair_mask": {1: torch.tensor(i % 2 == 0)},
        }
        for i in range(8)
    ]
    loaders = [
        DataLoader(rows, batch_size=2, shuffle=shuffle, generator=torch.Generator().manual_seed(seed))
        for shuffle in (True, False)
    ]
    return model, loaders


def config():
    """
    Build a two-stage schedule that changes the loss while retaining MSE selection.

    Returns
    -------
    dict
        Training configuration with checkpoint provenance and an enabled stochastic model.
    """

    stage = {
        "name": "first",
        "epochs": 2,
        "freeze": {"backbone": False},
        "optimizer": {"lr": {"backbone": 0.01}, "wd": {"backbone": 0.01}},
        "scheduler": {"grad_accum_steps": 1, "warmup_steps_fraction": 0.25},
    }
    second = copy.deepcopy(stage)
    second["name"] = "second"
    second["loss"] = {"terms": [{"type": "embed_mse"}]}
    return {
        "stages": [stage, second],
        "resume": False,
        "checkpoint_metric": "mse",
        "early_stop": {"patience": 0, "delta": 0.0},
        "amp": {"enable": False},
        "optimizer": {"use_adamw": True},
        "resume_identity": "fixed-original-reference",
        "loss": {
            "terms": [
                {"type": "continuous_gpo", "weight": 0.8, "sft_weight": 0.3},
                {"type": "embed_mse", "weight": 0.2},
            ]
        },
    }


def train(path, model, loaders, cfg):
    """
    Run the production training loop with synthetic fixtures.

    Parameters
    ----------
    path : pathlib.Path
        Run directory.
    model : TrainingModel
        Model whose state is trained or resumed.
    loaders : list[DataLoader]
        Training and validation loaders, in that order.
    cfg : dict
        Effective training configuration.

    Returns
    -------
    dict
        Per-stage history and final training counters.
    """

    return loop.train_finetune(model, *loaders, run_dir=str(path), train_cfg=cfg, loader_cfg={})


def assert_state_equal(left, right):
    """
    Compare nested checkpoint states exactly, ignoring the RNG capture timestamp.

    Parameters
    ----------
    left, right : Any
        Corresponding state objects containing mappings, sequences, arrays, tensors or scalars.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any state value other than the diagnostic capture time differs.
    """

    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left.keys() - {"time"}:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_state_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("interrupt_epoch", [1, 2, 3])
def test_resume_exactly_matches_continuous_training(tmp_path, monkeypatch, interrupt_epoch):
    """
    Compare all training state after interruptions within and between stages.
    """

    cfg = config()
    continuous, loaders = seeded_setup()
    full_history = train(tmp_path / "full", continuous, loaders, cfg)
    interrupted, loaders = seeded_setup()
    original_save = loop.save_latest

    class Interrupted(Exception):
        """
        Signal a simulated process interruption after a completed checkpoint save.
        """

        pass

    def stop_after_save(**kwargs):
        """
        Persist the real checkpoint, then interrupt at the configured epoch.
        """

        original_save(**kwargs)
        if kwargs["epoch"] == interrupt_epoch:
            raise Interrupted

    monkeypatch.setattr(loop, "save_latest", stop_after_save)
    with pytest.raises(Interrupted):
        train(tmp_path / "resumed", interrupted, loaders, cfg)
    monkeypatch.setattr(loop, "save_latest", original_save)
    resumed, loaders = seeded_setup(seed=999)
    cfg["resume"] = True
    actual_history = train(tmp_path / "resumed", resumed, loaders, cfg)
    assert actual_history == full_history
    assert_state_equal(continuous.state_dict(), resumed.state_dict())
    for filename in ["optimizer.pt", "scheduler.pt", "scaler.pt", "rng.pt", "training_state.pt"]:
        states = [
            torch.load(tmp_path / name / "checkpoints/latest" / filename, weights_only=False)
            for name in ("full", "resumed")
        ]
        assert_state_equal(*states)
    for name in ("full", "resumed"):
        best = json.loads((tmp_path / name / "checkpoints/best/meta.json").read_text())
        assert best["checkpoint_metric"] == "mse"
        assert best["best_val"] == full_history["best_val"]
    # Resuming a finished run is idempotent, with no further updates.
    assert train(tmp_path / "resumed", resumed, loaders, cfg) == full_history


@pytest.fixture
def checkpoint(tmp_path):
    """
    Create a complete training checkpoint for integrity and compatibility tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated run directory provided by pytest.

    Returns
    -------
    tuple[pathlib.Path, dict]
        Completed run directory and its unchanged training configuration.
    """

    model, loaders = seeded_setup()
    cfg = config()
    train(tmp_path, model, loaders, cfg)
    return tmp_path, cfg


@pytest.mark.parametrize(
    "filename", ["backbone.pt", "optimizer.pt", "scheduler.pt", "scaler.pt", "rng.pt", "training_state.pt", "meta.json"]
)
@pytest.mark.parametrize("corrupt", [False, True])
def test_resume_rejects_missing_and_corrupt_checkpoint(checkpoint, filename, corrupt):
    """
    Reject incomplete resume state before changing the newly constructed model.
    """

    path, cfg = checkpoint
    file = path / "checkpoints/latest" / filename
    if corrupt:
        file.write_bytes(b"broken checkpoint")
    else:
        file.unlink()
    cfg["resume"] = True
    model, loaders = seeded_setup()
    before = copy.deepcopy(model.state_dict())
    with pytest.raises((ValueError, FileNotFoundError)):
        train(path, model, loaders, cfg)
    assert_state_equal(before, model.state_dict())


@pytest.mark.parametrize("change", ["loss", "protocol", "metric", "schedule", "loader"])
def test_resume_requires_unchanged_protocol_and_objective(checkpoint, change):
    """
    Reject changes to loss, provenance, selection metric, schedule or loader settings.
    """

    path, cfg = checkpoint
    loader = {}
    if change == "loss":
        cfg["loss"]["terms"][0]["beta"] = 10.0
    elif change == "protocol":
        cfg["resume_identity"] = "different-reference-or-pairs"
    elif change == "metric":
        cfg["checkpoint_metric"] = "objective"
    elif change == "schedule":
        cfg["stages"][0]["epochs"] += 1
    else:
        loader["batch_size"] = 3
    with pytest.raises(ValueError, match="new run tag"):
        inspect_resume(path, signature=training_signature(cfg, loader), block_names=["backbone"])


def test_missing_resume_never_starts_from_scratch(tmp_path):
    """
    Require an existing verified checkpoint whenever resume is requested.
    """

    cfg = config()
    cfg["resume"] = True
    model, loaders = seeded_setup()
    with pytest.raises(FileNotFoundError, match="Missing verified resume"):
        train(tmp_path, model, loaders, cfg)


def test_checkpoint_selection_uses_mse_and_preserves_early_stop_state(tmp_path, monkeypatch):
    """
    Select checkpoints by MSE even when the training objective improves.
    """

    def epoch(**kwargs):
        """
        Supply deterministic epoch logs to isolate checkpoint-selection behavior.
        """

        index = kwargs["epoch_global"]
        # Objective improves, independent MSE gets worse after epoch 1.
        return 10.0 / index, {"mse": float(index)}, index

    monkeypatch.setattr(loop, "run_one_epoch", epoch)
    cfg = config()
    cfg["early_stop"]["patience"] = 1
    cfg["stages"][0]["epochs"] = 3
    model, loaders = seeded_setup()
    result = train(tmp_path, model, loaders, cfg)
    assert result["epochs_run"] == 2
    assert result["best_val"] == 1.0
    meta = json.loads((tmp_path / "checkpoints/best/meta.json").read_text())
    assert meta["epoch_best"] == 1
    latest, _ = inspect_resume(tmp_path, block_names=["backbone"])
    assert latest["training_complete"] is True
    assert latest["stage_index"] == 0
    assert latest["epoch_in_stage"] < cfg["stages"][0]["epochs"]
    assert "second" not in result["stages"]
    cfg["resume"] = True
    assert train(tmp_path, model, loaders, cfg) == result


@pytest.mark.parametrize("filename", ["backbone.pt", "meta.json"])
def test_resume_rejects_best_checkpoint_inconsistent_with_latest(checkpoint, filename):
    """
    Reject a best checkpoint that was changed after the latest committed manifest.
    """

    path, _ = checkpoint
    (path / "checkpoints/best" / filename).write_bytes(b"incomplete next save")
    with pytest.raises(ValueError, match="Best checkpoint is corrupt or inconsistent"):
        inspect_resume(path, block_names=["backbone"])


def test_missing_pairs_still_allow_mse_checkpoint_selection():
    """
    Allow target-based MSE selection when preference diagnostics have no observations.
    """

    rows = metric_rows()
    for row in rows:
        row.pop("y_l_emb")
        row.pop("gpo_pair_mask")
    objective, logs = evaluate(rows, 2, [{"type": "continuous_gpo"}])
    assert select_metric("mse", objective, logs) > 0
    with pytest.raises(RuntimeError, match="no valid observations"):
        select_metric("objective", objective, logs)
    assert logs["pair_windows"] == logs["pair_batches"] == 0


def test_reference_diagnostics_use_their_own_observation_counts():
    """
    Exclude batches without reference predictions from reference diagnostic denominators.
    """

    from mmt.train.metrics import EpochMetrics

    aggregator = build_loss_aggregator({"terms": [{"type": "continuous_gpo"}]})
    metrics = EpochMetrics(aggregator.epoch_terms)
    for index, batch in enumerate(DataLoader(metric_rows()[:3], batch_size=1)):
        if index == 2:
            batch.pop("ref_preds")
        _, logs = aggregator.compute(batch["pred"], batch)
        metrics.update(batch["pred"], batch, logs)
    _, logs = metrics.finish()
    assert logs["ContinuousGPOLoss/count/1"] == 2
    assert logs["ContinuousGPOLoss/count/ref_margin/1"] == 1
    assert logs["ContinuousGPOLoss/ref_margin/1"] == 1.0


def test_no_validation_observations_aborts_before_checkpointing(tmp_path, monkeypatch):
    """
    Fail before writing a checkpoint if the selected validation metric is unobserved.
    """

    def epoch(**kwargs):
        """
        Supply deterministic epoch logs to isolate checkpoint-selection behavior.
        """

        return float("nan"), {}, 1

    monkeypatch.setattr(loop, "run_one_epoch", epoch)
    model, loaders = seeded_setup()
    with pytest.raises(RuntimeError, match="no valid observations"):
        train(tmp_path, model, loaders, config())
    assert not (tmp_path / "checkpoints").exists()


def test_validation_log_is_compact_and_history_keeps_diagnostics(tmp_path, monkeypatch, caplog):
    """
    Keep hundreds of per-signal diagnostic keys in history without printing them at INFO.
    """

    logs = {f"diagnostic/{index}": float(index) for index in range(400)}
    logs.update(
        {
            "mse": 1.0,
            "objective": 2.0,
            "EmbedMSELoss_1/mean": 1.0,
            "ContinuousGPOLoss_0/mean": 3.0,
            "pair_window_coverage": 0.25,
            "pair_batch_coverage": 0.5,
        }
    )

    def epoch(**kwargs):
        """
        Supply a large diagnostic record to the production checkpoint and logging path.
        """

        return 2.0, dict(logs), kwargs["epoch_global"]

    monkeypatch.setattr(loop, "run_one_epoch", epoch)
    caplog.set_level("INFO", logger="mmt.Train")
    model, loaders = seeded_setup()
    history = train(tmp_path, model, loaders, config())
    messages = [record.getMessage() for record in caplog.records if "validation metrics:" in record.getMessage()]
    assert len(messages) == history["epochs_run"]
    for message in messages:
        assert len(message) < 300
        assert "diagnostic/" not in message
        for key in (
            "mse",
            "objective",
            "EmbedMSELoss_1/mean",
            "ContinuousGPOLoss_0/mean",
            "pair_window_coverage",
            "pair_batch_coverage",
        ):
            assert f"{key}=" in message
    _, saved = inspect_resume(tmp_path, block_names=["backbone"])
    for records in saved["history"]["stages"].values():
        for record in records:
            for key, value in logs.items():
                assert record[f"val_{key}"] == value


@pytest.mark.parametrize("loader_kind", ["streaming", "persistent_workers"])
def test_unsupported_loader_warns_at_first_save_and_resume_still_fails(tmp_path, monkeypatch, caplog, loader_kind):
    """
    Warn once per unsupported loader while saving checkpoints, before any resume request.
    """

    def epoch(**kwargs):
        """
        Isolate checkpoint reporting without iterating streams or creating worker processes.
        """

        return 1.0, {"mse": 1.0}, kwargs["epoch_global"]

    monkeypatch.setattr(loop, "run_one_epoch", epoch)
    caplog.set_level("WARNING", logger="mmt.Train")
    model, loaders = seeded_setup()
    loaders[1] = SimpleNamespace(
        dataset=torch.utils.data.ChainDataset([]) if loader_kind == "streaming" else [],
        persistent_workers=loader_kind == "persistent_workers",
    )
    history = train(tmp_path, model, loaders, config())
    assert history["epochs_run"] == 4
    warnings = [record.getMessage() for record in caplog.records if "cannot be resumed" in record.getMessage()]
    assert len(warnings) == 1
    assert "val loader" in warnings[0]
    assert str(tmp_path) in warnings[0]
    assert ("streaming" if loader_kind == "streaming" else "persistent_workers=false") in warnings[0]
    with pytest.raises(ValueError, match="Cannot resume val loader"):
        inspect_resume(tmp_path, block_names=["backbone"])
