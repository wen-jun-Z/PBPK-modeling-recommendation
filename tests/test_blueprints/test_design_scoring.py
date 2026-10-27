"""Tiny numerical fixtures only; one optional CPU/CUDA parity calculation."""
from __future__ import annotations

import copy
import json
import math

import pytest
import torch

from pbpk_kb.blueprints.design_scoring import score_calibration_curves


def _run(index: int = 0, values: list | None = None, **kwargs) -> dict:
    return {"vector_index": index, "time_min": [0.0, 2.0, 4.0],
            "concentration": [1.0, 3.0, 5.0] if values is None else values,
            "time_unit": "min", "unit": "µmol/l", **kwargs}


def _observations() -> list[dict]:
    return [{"time_min": time, "concentration_umol_per_l": value}
            for time, value in ((0.0, 1.0), (1.0, 2.0), (3.0, 4.0), (4.0, 5.0))]


def test_interpolates_on_actual_observation_times_and_keeps_input_order() -> None:
    runs = [_run(9, [2, 6, 10]), _run(2)]
    before = copy.deepcopy(runs)
    result = score_calibration_curves(runs=runs, observations=_observations(), device="cpu")
    assert result["device"] == "cpu" and not result["gpu_used"]
    assert result["observation_count"] == 4
    doubled, exact = result["rows"]
    assert doubled["vector_index"] == 9 and exact["vector_index"] == 2
    assert doubled["predictions_umol_per_l"] == [2, 4, 8, 10]
    assert doubled["log_mse"] == pytest.approx(math.log(2) ** 2)
    assert doubled["log_rmse"] == pytest.approx(math.log(2))
    assert doubled["gmfe"] == pytest.approx(2)
    assert exact["valid"] and exact["error"] is None
    assert exact["log_mse"] == exact["log_rmse"] == 0
    assert exact["gmfe"] == 1
    assert runs == before
    assert not any("cmax" in key.lower() or "auc" in key.lower() for row in result["rows"] for key in row)


def test_gmfe_uses_absolute_log_errors_not_absolute_mean_log_error() -> None:
    observations = [{"time_min": 0, "concentration_umol_per_l": 1},
                    {"time_min": 4, "concentration_umol_per_l": 1}]
    row = score_calibration_curves(runs=[_run(values=[2, 1, 0.5])], observations=observations, device="cpu")["rows"][0]
    assert row["gmfe"] == pytest.approx(2)
    assert row["log_mse"] == pytest.approx(math.log(2) ** 2)


def test_batching_supports_different_time_grids_lengths_and_observation_order() -> None:
    observations = list(reversed(_observations()))
    runs = [_run(0), _run(1, [1, 2, 5], time_min=[0, 1, 4]),
            _run(2, [1, 5], time_min=[0, 4])]
    result = score_calibration_curves(runs=runs, observations=observations, device="cpu")
    assert all(row["valid"] for row in result["rows"])
    assert all(row["predictions_umol_per_l"] == [5, 4, 2, 1] for row in result["rows"])


@pytest.mark.parametrize("unit", ["µmol/l", "μmol/l", "umol/l"])
def test_explicit_micro_unit_spellings_are_equivalent(unit: str) -> None:
    result = score_calibration_curves(runs=[_run(unit=unit)], observations=_observations(), device="cpu")
    assert result["rows"][0]["valid"]


@pytest.mark.parametrize("change,reason", [
    ({"time_unit": "h"}, "unsupported_time_unit"),
    ({"unit": "mg/l"}, "unsupported_concentration_unit"),
    ({"unit": None}, "unsupported_concentration_unit"),
    ({"time_min": [0, 2, 2]}, "invalid_time_grid"),
    ({"time_min": [0, 4, 2]}, "invalid_time_grid"),
    ({"time_min": [1, 2, 4]}, "observation_outside_curve"),
    ({"time_min": [0, 1, 3]}, "observation_outside_curve"),
    ({"concentration": [1, 2]}, "invalid_curve_shape"),
    ({"concentration": [1, float("nan"), 5]}, "finite"),
    ({"concentration": [1, 0, 0]}, "nonpositive_prediction_at_observation"),
    ({"concentration": [-1, 3, 5]}, "nonpositive_prediction_at_observation"),
])
def test_invalid_curve_is_retained_with_null_scores(change: dict, reason: str) -> None:
    runs = [_run(3, **change), _run(8)]
    result = score_calibration_curves(runs=runs, observations=_observations(), device="cpu")
    invalid, valid = result["rows"]
    assert len(result["rows"]) == 2
    assert invalid["vector_index"] == 3 and not invalid["valid"]
    assert reason in invalid["error"]
    for key in ("log_mse", "log_rmse", "gmfe", "predictions_umol_per_l"):
        assert invalid[key] is None
    assert valid["valid"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "2"])
def test_bad_observations_reject_entire_scoring_call(value: object) -> None:
    with pytest.raises(ValueError):
        score_calibration_curves(runs=[_run()], observations=[{"time_min": 1, "concentration_umol_per_l": value}], device="cpu")


def test_does_not_extrapolate_even_by_a_small_amount() -> None:
    result = score_calibration_curves(runs=[_run()], observations=[{"time_min": 4 + 1e-12, "concentration_umol_per_l": 5}], device="cpu")
    assert not result["rows"][0]["valid"]


def test_ambiguous_vector_mapping_is_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        score_calibration_curves(runs=[_run(1), _run(1)], observations=_observations(), device="cpu")
    with pytest.raises(ValueError, match="integer vector_index"):
        score_calibration_curves(runs=[_run(True)], observations=_observations(), device="cpu")


def test_unavailable_cuda_never_silently_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
        score_calibration_curves(runs=[_run()], observations=_observations())


def test_extreme_positive_values_use_log_differences_without_ratio_underflow() -> None:
    result = score_calibration_curves(runs=[_run(values=[1e-300] * 3)],
                                     observations=[{"time_min": 1, "concentration_umol_per_l": 1e-200}], device="cpu")
    row = result["rows"][0]
    assert row["valid"]
    assert row["gmfe"] == pytest.approx(1e100)


def test_unrepresentable_fold_error_is_invalid_not_infinite_json() -> None:
    result = score_calibration_curves(runs=[_run(values=[1e300] * 3)],
                                     observations=[{"time_min": 1, "concentration_umol_per_l": 1e-300}], device="cpu")
    assert result["rows"][0]["error"] == "nonfinite_score"
    assert result["rows"][0]["gmfe"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; production requests still raise")
def test_tiny_cuda_cpu_float64_parity() -> None:
    runs = [_run(0), _run(1, [2, 6, 10]), _run(2, [1, 0, 0])]
    cpu = score_calibration_curves(runs=runs, observations=_observations(), device="cpu")
    cuda = score_calibration_curves(runs=runs, observations=_observations(), device="cuda")
    assert cuda["gpu_used"] and cuda["device"].startswith("cuda:")
    for expected, actual in zip(cpu["rows"], cuda["rows"]):
        assert expected["vector_index"] == actual["vector_index"]
        assert expected["valid"] == actual["valid"] and expected["error"] == actual["error"]
        if expected["valid"]:
            for key in ("log_mse", "log_rmse", "gmfe", "predictions_umol_per_l"):
                assert actual[key] == pytest.approx(expected[key], rel=1e-12, abs=1e-14)
