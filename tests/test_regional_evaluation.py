"""Regression tests for regional MSE and backwards-compatible full-grid error maps.

Run with src/, scripts_mast/ and a GS-compatible TokaMark revision on PYTHONPATH.
No dataset download, training or GPU is required.
"""

import csv
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from mast_utils.config import _validate_regional_diagnostics_config
from mast_utils.eval.grad_shafranov.error_map import FieldErrorMapMetric
from mast_utils.eval.grad_shafranov.grid import GSEvalGrid
from mast_utils.eval.grad_shafranov.psi_error_map import PsiErrorMapMetric
from mast_utils.eval.grad_shafranov.regional_mse import RegionalMSEMetric
from mast_utils.eval.grad_shafranov.topology_provider import TopologyProvider


@pytest.fixture
def grid():
    result = GSEvalGrid.__new__(GSEvalGrid)
    result.n_r, result.n_z = 5, 7
    result.r, result.z = np.meshgrid(np.arange(1.0, 6.0), np.arange(-3.0, 4.0), indexing="ij")
    result.limiter_region = np.zeros((5, 7), bool)
    result.limiter_region[1:4, 1:6] = True
    result.region = result._erode_one_cell(result.limiter_region)
    return result


def native(fields):
    """(B,T,R,Z) -> (B,Z,R,T), deliberately non-square to catch transposes."""
    return np.asarray(fields).transpose(0, 3, 2, 1)


def polygon():
    return np.array([[2.0, -1.0], [4.0, -1.0], [4.0, 1.0], [2.0, 1.0]])


def feed(metric, fields, ids=None, indices=None, selected=None, signal="psi", true=None):
    batch = len(fields)
    metric.add_batch(
        signal=signal,
        pred_native=native(fields),
        true_native=native(np.zeros_like(fields) if true is None else true),
        shot_ids=np.ones(batch, dtype=int) if ids is None else np.asarray(ids),
        window_indices=np.arange(batch) if indices is None else np.asarray(indices),
        window_mask=np.ones(batch, bool) if selected is None else np.asarray(selected),
    )


def rows(metric, tmp_path):
    result = metric.write_csvs(metrics_task_dir=tmp_path, task_name="test")
    return {(row["signal"], row["region"]): row for row in result["rows"]}


def test_limiter_asset_preserves_original_region():
    actual = GSEvalGrid()
    assert actual.limiter_region.sum() > actual.region.sum()
    np.testing.assert_array_equal(actual.region, actual._erode_one_cell(actual.limiter_region))


def test_maps_preserve_psi_and_add_current_mse(grid, tmp_path):
    values = np.arange(2 * 2 * 5 * 7, dtype=float).reshape(2, 2, 5, 7) / 10
    true = np.zeros_like(values)
    true[0, 0, 0, 0] = np.nan
    for signal in ("psi", "jtor"):
        metric = PsiErrorMapMetric(grid=grid) if signal == "psi" else FieldErrorMapMetric(grid=grid, signal=signal)
        kwargs = (
            {"psi_native": native(values), "psi_gt_native": native(true)}
            if signal == "psi"
            else {
                "pred_native": native(values),
                "true_native": native(true),
            }
        )
        metric.add_batch(**kwargs, window_mask=np.array([True, False]))
        result = metric.write_npz(metrics_task_dir=tmp_path, task_name="test")
        data = np.load(result[f"{signal}_error_map_npz"])
        expected = np.where(np.isfinite(true[0]), values[0], np.nan)
        np.testing.assert_allclose(data["mse"], np.nanmean(expected**2, axis=0))
        np.testing.assert_allclose(data["rmse"] ** 2, data["mse"])
        np.testing.assert_allclose(data["mae"], np.nanmean(abs(expected), axis=0))
        np.testing.assert_allclose(data["bias"], np.nanmean(expected, axis=0))
        np.testing.assert_array_equal(data["count"], np.isfinite(expected).sum(axis=0))
        assert data["n_fields_contributed"] == 2
    empty = FieldErrorMapMetric(grid=grid, signal="jtor")
    empty.add_batch(pred_native=native(values), true_native=None, window_mask=None)
    assert empty.is_empty()


def test_limiter_errors_and_independent_signals(grid, tmp_path):
    metric = RegionalMSEMetric(grid=grid)
    # A non-square grid and large exterior errors expose transpose/mask errors.
    fields = np.where(grid.limiter_region, 2.0, 100.0)[None, None]
    feed(metric, fields)
    feed(metric, fields * 3, signal="jtor", selected=[False])
    result = rows(metric, tmp_path)
    assert set(result) == {("psi", r) for r in ("full_grid", "limiter", "outside_limiter")}
    assert result["psi", "limiter"]["mse"] == 4
    assert result["psi", "limiter"]["n_cells"] == 15  # Includes un-eroded boundary.
    protocol = json.loads((tmp_path / "regional_mse_protocol.json").read_text())
    assert protocol["schema_version"] == 3
    assert protocol["regions"]["limiter"] == "MAST_lim_mask_rz, no erosion"
    assert not any("lcfs" in key for key in result["psi", "limiter"])


def test_output_times_pool_valid_limiter_cells(grid, tmp_path):
    metric = RegionalMSEMetric(grid=grid)
    fields = np.ones((1, 2, 5, 7))
    fields[0, 0] = 2
    true = np.zeros_like(fields)
    true[0, 1, 1, 1] = np.nan
    feed(metric, fields, true=true)
    result = rows(metric, tmp_path)["psi", "limiter"]
    assert result["mse"] == pytest.approx((15 * 4 + 14) / 29)
    assert result["n_cells"] == 29
    assert result["n_reference_nonfinite"] == 1


def test_hierarchical_mean_and_batch_invariance(grid, tmp_path):
    fields = np.broadcast_to(np.array([1.0, 3.0, 2.0])[:, None, None, None], (3, 2, 5, 7))
    one, split = RegionalMSEMetric(grid=grid), RegionalMSEMetric(grid=grid)
    feed(one, fields, ids=[10, 10, 20])
    for i, sid in enumerate([10, 10, 20]):
        feed(split, fields[i : i + 1], ids=[sid], indices=[i])
    a, b = rows(one, tmp_path / "one"), rows(split, tmp_path / "split")
    assert a == b
    assert a["psi", "limiter"]["mse"] == 4.5  # ((1+9)/2 + 4)/2, not 14/3.
    for row in a.values():
        assert row["mse_pooled"] == pytest.approx(14 / 3)
    assert (tmp_path / "one/regional_mse_shots.csv").read_text() == (
        tmp_path / "split/regional_mse_shots.csv"
    ).read_text()


def test_empty_limiter_reports_nan(grid, tmp_path):
    grid.limiter_region[:] = False
    metric = RegionalMSEMetric(grid=grid)
    feed(metric, np.ones((1, 2, 5, 7)))
    result = rows(metric, tmp_path)["psi", "limiter"]
    assert np.isnan(result["mse"])
    assert np.isnan(result["mse_pooled"])
    assert result["n_cells"] == 0
    assert result["n_fields_empty_region"] == 2


def test_nonfinite_predictions_outside_limiter_raise(grid):
    metric = RegionalMSEMetric(grid=grid)
    fields = np.where(grid.limiter_region, 2.0, np.nan)[None, None]
    with pytest.raises(ValueError, match="Non-finite"):
        feed(metric, fields)


def test_bad_prediction_fails_instead_of_exclusion(grid):
    values = np.ones((1, 1, 5, 7))
    values[0, 0, 1, 1] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        feed(RegionalMSEMetric(grid=grid), values)
    feed(RegionalMSEMetric(grid=grid), values, selected=[False])


def test_shape_mismatch_fails(grid):
    with pytest.raises(ValueError, match="shapes differ"):
        feed(RegionalMSEMetric(grid=grid), np.ones((1, 2, 5, 7)), true=np.zeros((1, 1, 5, 7)))


@pytest.mark.parametrize("value", [np.inf, 1e308])
def test_invalid_or_overflowed_limiter_prediction_fails(grid, value):
    fields = np.ones((1, 1, 5, 7))
    fields[0, 0, 1, 1] = value
    with pytest.raises(ValueError, match="Non-finite"):
        feed(RegionalMSEMetric(grid=grid), fields)


@pytest.mark.parametrize("config", [{"enable": "true"}, [], {"enable": True, "mask": "predicted"}])
def test_config_rejects_bad_options(config):
    for key in ("jtor_error_map", "regional_mse"):
        with pytest.raises((ValueError, TypeError)):
            _validate_regional_diagnostics_config({"eval": {"compute_metrics": {key: config}}})


def test_gs_configs_enabled():
    config = yaml.safe_load((Path(__file__).parents[1] / "scripts_mast/configs/mmt/tasks/eval_tasks.yaml").read_text())
    for task, spec in config["tasks"].items():
        if task.endswith(("-data", "-strong", "-weak")):
            _validate_regional_diagnostics_config(spec)
            assert spec["eval"]["compute_metrics"]["regional_mse"]["enable"]


@pytest.mark.parametrize("with_current", [False, True])
@pytest.mark.parametrize("with_topology", [False, True])
def test_evaluation_wiring_without_auxiliary_data(grid, tmp_path, monkeypatch, with_current, with_topology):
    runner = importlib.import_module("mast_utils.eval.benchmark_eval")
    monkeypatch.setattr(runner, "GSEvalGrid", lambda _: grid)
    fields = native(np.ones((1, 2, 5, 7)))
    predictions = {"equilibrium-psi": fields}
    if with_current:
        predictions["equilibrium-j_tor"] = native(np.full((1, 3, 5, 7), 2.0))
    references = {key: np.zeros_like(value) for key, value in predictions.items()}
    masks = {key: np.ones(1, bool) for key in predictions}
    # Limiter MSE does not need timestamps; the independent topology metric does.
    times = {"equilibrium-psi": np.array([[0.1, 0.2]])} if with_topology else None
    monkeypatch.setattr(
        runner,
        "forward_decode_native",
        lambda **_: (references, predictions, masks, np.array([30421]), np.array([4]), times),
    )

    def forbidden_provider(**kwargs):
        raise AssertionError("Limiter-only evaluation accessed auxiliary geometry")

    monkeypatch.setattr(runner, "TopologyProvider", forbidden_provider)
    topology_calls = []
    if with_topology:

        class Provider:
            def __init__(self, **kwargs):
                assert kwargs["requested_fields"] == {"lcfs_r", "lcfs_z"}

            def query(self, ids, query_times):
                np.testing.assert_array_equal(query_times, times["equilibrium-psi"])
                topology_calls.append("query")
                return [[None, None]]

        class Metric:
            required_auxiliary_fields = frozenset({"lcfs_r", "lcfs_z"})

            def __init__(self, **kwargs):
                pass

            def add_batch(self, **kwargs):
                assert kwargs["topology"] == [[None, None]]
                topology_calls.append("metric")

            def write_csvs(self, **kwargs):
                return {"n_shots": 1}

        monkeypatch.setattr(runner, "TopologyProvider", Provider)
        monkeypatch.setattr(runner, "GSTopologyMetric", Metric)

    config = {key: {"enable": True} for key in ("regional_mse", "psi_error_map", "jtor_error_map")}
    config["gs_topology"] = {"enable": with_topology}
    result = runner.evaluate_benchmark_and_diagnostics(
        model=torch.nn.Identity(),
        amp_enabled=False,
        dataloader=[{}],
        device=torch.device("cpu"),
        stats={},
        decoders={},
        id_to_name={},
        run_dir=tmp_path,
        task_name="test",
        source_zarr_root="unused" if with_topology else None,
        compute_metrics_cfg=config,
    )
    assert topology_calls == (["query", "metric"] if with_topology else [])
    assert ("gs_topology" in result) == with_topology
    assert len(result["regional_mse"]["rows"]) == 3 * len(predictions)
    assert {row["region"] for row in result["regional_mse"]["rows"]} == {"full_grid", "limiter", "outside_limiter"}
    assert "psi_error_map" in result
    assert ("jtor_error_map" in result) == with_current
    with open(result["regional_mse"]["regional_mse_shots_csv"], newline="") as stream:
        assert all(float(row["mse"]) in (1.0, 4.0) for row in csv.DictReader(stream))


def test_provider_alignment_rejects_distant_times(monkeypatch):
    provider = TopologyProvider(requested_fields={"lcfs_r", "lcfs_z"}, store_root="unused")
    points = polygon()
    arrays = {
        "time": np.array([0.0, 0.1, 0.2]),
        "lcfs_r": np.tile(points[:, 0, None], (1, 3)),
        "lcfs_z": np.tile(points[:, 1, None], (1, 3)),
    }
    monkeypatch.setattr(provider, "_load_shot", lambda _: arrays)
    result = provider.query([30421], np.array([[0.1, 10.0]]))
    assert result[0][0]["source_index"] == 1
    assert result[0][1] is None


def test_limiter_matches_benchmark_on_same_selected_cells(grid, tmp_path, monkeypatch):
    from tokamark import evaluator

    monkeypatch.setattr(evaluator, "get_signals_metadata", lambda: {"equilibrium-psi": {"std": 1.0}})
    fields = np.broadcast_to(np.array([1.0, 3.0, 2.0])[:, None, None, None], (3, 2, 5, 7))
    ids = np.array([10, 10, 20])
    df = evaluator.compute_windows_metrics(
        y_target=np.zeros((3, 30)),
        y_pred=fields[:, :, grid.limiter_region].reshape(3, -1),
        shot_ids=ids,
        window_indices=np.arange(3),
        feature_name="equilibrium-psi",
    )
    _, _, benchmark_shots = evaluator.aggregate_windows_metrics(df)
    metric = RegionalMSEMetric(grid=grid)
    feed(metric, fields, ids=ids)
    rows(metric, tmp_path)
    with (tmp_path / "regional_mse_shots.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["region"] == "limiter":
                rmse = benchmark_shots.loc[benchmark_shots.shot_id == int(row["shot_id"]), "RMSE"].iloc[0]
                assert float(row["mse"]) == pytest.approx(rmse**2)


def test_disabled_diagnostics_do_not_load_geometry(tmp_path, monkeypatch):
    runner = importlib.import_module("mast_utils.eval.benchmark_eval")

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled diagnostic accessed geometry")

    monkeypatch.setattr(runner, "GSEvalGrid", forbidden)
    monkeypatch.setattr(runner, "TopologyProvider", forbidden)
    monkeypatch.setattr(runner, "forward_decode_native", lambda **_: ({}, {}, {}, np.array([1]), np.array([0]), None))
    result = runner.evaluate_benchmark_and_diagnostics(
        model=torch.nn.Identity(),
        amp_enabled=False,
        dataloader=[{}],
        device=torch.device("cpu"),
        stats={},
        decoders={},
        id_to_name={},
        run_dir=tmp_path,
        task_name="test",
        compute_metrics_cfg={},
    )
    assert set(result) == {"metrics_task_dir", "metrics_root_dir", "task"}
    assert not (tmp_path / "metrics").exists()


def test_all_missing_reference_and_masked_nan(grid, tmp_path):
    metric = RegionalMSEMetric(grid=grid)
    feed(metric, np.ones((1, 1, 5, 7)), true=np.full((1, 1, 5, 7), np.nan))
    result = rows(metric, tmp_path)
    assert all(np.isnan(row["mse"]) and row["n_cells"] == 0 for row in result.values())
    assert result["psi", "limiter"]["n_reference_nonfinite"] == 15


@pytest.mark.parametrize("signal", ["psi", "jtor"])
def test_regions_partition_and_match_maps(grid, tmp_path, signal):
    fields = np.arange(3 * 2 * 5 * 7, dtype=float).reshape(3, 2, 5, 7) / 10
    true = np.zeros_like(fields)
    true[0, 0, 1, 1] = np.nan
    true[1, 1, 0, 0] = np.nan
    metric = RegionalMSEMetric(grid=grid)
    feed(metric, fields, true=true, signal=signal, ids=[10, 20, 30], selected=[True, True, False])
    result = rows(metric, tmp_path)
    full, inside, outside = [result[signal, name] for name in ("full_grid", "limiter", "outside_limiter")]
    assert full["n_cells"] == inside["n_cells"] + outside["n_cells"]
    assert full["squared_error_sum"] == pytest.approx(inside["squared_error_sum"] + outside["squared_error_sum"])
    maps = FieldErrorMapMetric(grid=grid, signal=signal)
    maps.add_batch(pred_native=native(fields), true_native=native(true), window_mask=np.array([True, True, False]))
    output = maps.write_npz(metrics_task_dir=tmp_path, task_name="test")
    with np.load(output[f"{signal}_error_map_npz"]) as data:
        for region, mask in metric._regions.items():
            valid = mask & (data["count"] > 0)
            expected = np.sum(data["mse"][valid] * data["count"][valid]) / np.sum(data["count"][valid])
            assert result[signal, region]["mse_pooled"] == pytest.approx(expected)


def test_empty_exterior_reports_nan(grid, tmp_path):
    grid.limiter_region[:] = True
    metric = RegionalMSEMetric(grid=grid)
    feed(metric, np.ones((1, 1, 5, 7)))
    result = rows(metric, tmp_path)
    assert np.isnan(result["psi", "outside_limiter"]["mse_pooled"])
    assert result["psi", "full_grid"]["mse_pooled"] == result["psi", "limiter"]["mse_pooled"] == 1
