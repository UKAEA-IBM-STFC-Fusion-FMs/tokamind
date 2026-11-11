"""Regression tests for regional MSE and backwards-compatible full-grid error maps.

Run with src/, scripts_mast/ and a GS-compatible TokaMark revision on PYTHONPATH.
No dataset download, training or GPU is required.
"""

import csv
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from mast_utils.config import _validate_regional_diagnostics_config
from mast_utils.eval.grad_shafranov.error_map import FieldErrorMapMetric
from mast_utils.eval.grad_shafranov.grid import GSEvalGrid
from mast_utils.eval.grad_shafranov.psi_error_map import PsiErrorMapMetric
from mast_utils.eval.grad_shafranov.regional_mse import RegionalMSEMetric, _has_self_intersections, lcfs_mask
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


def topology(batch, times):
    return [[{"lcfs": polygon()} for _ in range(times)] for _ in range(batch)]


def feed(metric, fields, ids=None, indices=None, selected=None, topo=None, signal="psi", true=None):
    batch = len(fields)
    metric.add_batch(
        signal=signal,
        pred_native=native(fields),
        true_native=native(np.zeros_like(fields) if true is None else true),
        shot_ids=np.ones(batch, dtype=int) if ids is None else np.asarray(ids),
        window_indices=np.arange(batch) if indices is None else np.asarray(indices),
        window_mask=np.ones(batch, bool) if selected is None else np.asarray(selected),
        topology=topo,
    )


def rows(metric, tmp_path):
    result = metric.write_csvs(metrics_task_dir=tmp_path, task_name="test")
    return {(row["signal"], row["region"]): row for row in result["rows"]}


def test_limiter_asset_preserves_original_region():
    actual = GSEvalGrid()
    assert actual.limiter_region.sum() > actual.region.sum()
    np.testing.assert_array_equal(actual.region, actual._erode_one_cell(actual.limiter_region))


def test_lcfs_orientation_boundary_and_order(grid):
    assert not _has_self_intersections(polygon())
    mask = lcfs_mask(grid, polygon())
    expected = (grid.r >= 2) & (grid.r <= 4) & (abs(grid.z) <= 1)
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(lcfs_mask(grid, polygon()[::-1]), expected)
    np.testing.assert_array_equal(lcfs_mask(grid, np.vstack([polygon(), polygon()[0]])), expected)


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((2, 2)),
        np.array([[2, 0], [3, 0], [4, 0]]),
        np.array([[2, -1], [4, 1], [2, 1], [4, -1]]),
        np.array([[0, 0], [2, 1], [3, -1]]),
        np.array([[2, 0], [3, 1], [np.nan, 0]]),
    ],
)
def test_invalid_polygons(grid, points):
    with pytest.raises(ValueError):
        lcfs_mask(grid, points)


@pytest.mark.parametrize("reverse", [False, True])
def test_self_intersection_with_nonzero_area(grid, reverse):
    # Asymmetric bow tie: unlike the earlier test, its signed area is nonzero,
    # so rejection must exercise the intersection check rather than the area check.
    points = np.array([[2.0, -1.0], [4.0, 1.0], [2.0, 1.0], [3.0, -1.0]])
    if reverse:
        points = points[::-1]
    end = np.roll(points, -1, axis=0)
    twice_area = np.sum(points[:, 0] * end[:, 1] - points[:, 1] * end[:, 0])
    assert abs(twice_area) == 2.0
    assert _has_self_intersections(points)
    with pytest.raises(ValueError, match="LCFS is self-intersecting"):
        lcfs_mask(grid, points)


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


def test_regional_errors_and_independent_signals(grid, tmp_path):
    metric = RegionalMSEMetric(grid=grid)
    inside = lcfs_mask(grid, polygon())
    fields = np.where(inside, 2.0, 1.0)[None, None]
    feed(metric, fields, topo=topology(1, 1))
    feed(metric, fields * 3, topo=topology(1, 1), signal="jtor", selected=[False])
    result = rows(metric, tmp_path)
    assert len(result) == 3  # No invented J_tor zeroes for an unscored signal.
    assert result["psi", "lcfs"]["mse"] == 4
    assert result["psi", "limiter"]["mse"] == pytest.approx((9 * 4 + 6) / 15)
    assert result["psi", "full_grid"]["mse"] == pytest.approx((9 * 4 + 26) / 35)


def test_time_dependent_mask_before_aggregation(grid, tmp_path):
    metric = RegionalMSEMetric(grid=grid)
    fields = np.zeros((1, 2, 5, 7))
    fields[0, 0, 1:4, 2:5] = 2
    fields[0, 1] = 1
    topo = topology(1, 2)
    topo[0][1] = {"lcfs": np.array([[2, 0], [3, 0], [3, 1], [2, 1]])}
    feed(metric, fields, topo=topo)
    result = rows(metric, tmp_path)
    assert result["psi", "lcfs"]["mse"] == pytest.approx((9 * 4 + 4) / 13)
    assert result["psi", "lcfs"]["n_cells"] == 13


def test_hierarchical_mean_and_batch_invariance(grid, tmp_path):
    fields = np.broadcast_to(np.array([1.0, 3.0, 2.0])[:, None, None, None], (3, 2, 5, 7))
    one, split = RegionalMSEMetric(grid=grid), RegionalMSEMetric(grid=grid)
    feed(one, fields, ids=[10, 10, 20], topo=topology(3, 2))
    for i, sid in enumerate([10, 10, 20]):
        feed(split, fields[i : i + 1], ids=[sid], indices=[i], topo=topology(1, 2))
    a, b = rows(one, tmp_path / "one"), rows(split, tmp_path / "split")
    assert a == b
    assert a["psi", "full_grid"]["mse"] == 4.5  # ((1+9)/2 + 4)/2, not 14/3.
    assert (tmp_path / "one/regional_mse_shots.csv").read_text() == (
        tmp_path / "split/regional_mse_shots.csv"
    ).read_text()


def test_missing_invalid_empty_lcfs_and_nonfinite_reference(grid, tmp_path):
    metric = RegionalMSEMetric(grid=grid)
    fields = np.ones((1, 3, 5, 7))
    true = np.zeros_like(fields)
    true[0, 0, 0, 0] = np.nan
    topo = [[None, {"lcfs": [[2, 0], [3, 0]]}, {"lcfs": [[2.1, 0.1], [2.2, 0.1], [2.2, 0.2], [2.1, 0.2]]}]]
    feed(metric, fields, true=true, topo=topo)
    result = rows(metric, tmp_path)
    lcfs = result["psi", "lcfs"]
    assert np.isnan(lcfs["mse"])
    assert lcfs["n_fields_missing_lcfs"] == lcfs["n_fields_invalid_lcfs"] == lcfs["n_fields_empty_region"] == 1
    assert result["psi", "full_grid"]["mse"] == 1
    assert result["psi", "full_grid"]["n_cells"] == 104


def test_bad_prediction_fails_instead_of_exclusion(grid):
    values = np.ones((1, 1, 5, 7))
    values[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        feed(RegionalMSEMetric(grid=grid), values)
    feed(RegionalMSEMetric(grid=grid), values, selected=[False])


def test_shape_mismatch_fails(grid):
    with pytest.raises(ValueError, match="topology"):
        feed(RegionalMSEMetric(grid=grid), np.ones((1, 2, 5, 7)), topo=topology(1, 1))


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
def test_evaluation_wiring_and_signal_times(grid, tmp_path, monkeypatch, with_current):
    runner = importlib.import_module("mast_utils.eval.benchmark_eval")
    monkeypatch.setattr(runner, "GSEvalGrid", lambda _: grid)
    fields = native(np.ones((1, 2, 5, 7)))
    predictions = {"equilibrium-psi": fields}
    if with_current:
        predictions["equilibrium-j_tor"] = 2 * fields
    references = {key: np.zeros_like(value) for key, value in predictions.items()}
    masks = {key: np.ones(1, bool) for key in predictions}
    times = {key: np.array([[0.1, 0.2]]) + i for i, key in enumerate(predictions)}
    monkeypatch.setattr(
        runner,
        "forward_decode_native",
        lambda **_: (references, predictions, masks, np.array([30421]), np.array([4]), times),
    )
    calls = []

    class Provider:
        def __init__(self, **kwargs):
            assert kwargs["requested_fields"] == {"lcfs_r", "lcfs_z"}

        def query(self, ids, query_times):
            calls.append(query_times.copy())
            return topology(1, 2)

    monkeypatch.setattr(runner, "TopologyProvider", Provider)
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
        source_zarr_root="unused",
        compute_metrics_cfg={key: {"enable": True} for key in ("regional_mse", "psi_error_map", "jtor_error_map")},
    )
    assert len(calls) == len(predictions)
    for observed, expected in zip(calls, times.values(), strict=True):
        np.testing.assert_array_equal(observed, expected)
    assert len(result["regional_mse"]["rows"]) == 3 * len(predictions)
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


def test_full_grid_matches_benchmark_shot_rmse_squared(grid, tmp_path, monkeypatch):
    from tokamark import evaluator

    monkeypatch.setattr(evaluator, "get_signals_metadata", lambda: {"equilibrium-psi": {"std": 1.0}})
    fields = np.broadcast_to(np.array([1.0, 3.0, 2.0])[:, None, None, None], (3, 2, 5, 7))
    ids = np.array([10, 10, 20])
    df = evaluator.compute_windows_metrics(
        y_target=np.zeros((3, 70)),
        y_pred=native(fields).reshape(3, -1),
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
            if row["region"] == "full_grid":
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
    feed(metric, np.ones((1, 1, 5, 7)), true=np.full((1, 1, 5, 7), np.nan), topo=topology(1, 1))
    result = rows(metric, tmp_path)
    assert all(np.isnan(row["mse"]) and row["n_cells"] == 0 for row in result.values())
    assert result["psi", "full_grid"]["n_reference_nonfinite"] == 35
