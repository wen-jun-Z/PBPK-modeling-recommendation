"""Research gates must retain failures and missing-reference uncertainty."""
import copy
import json
import math

import pytest

from pbpk_kb.blueprints.evidence_evaluation import score_case
from pbpk_kb.blueprints.prediction_quality import assess_prediction_quality, curve_diagnostics


def _score(prediction_scale=1.0):
    times = [0., 60., 120., 180., 240., 300., 360.]
    values = [8 * math.exp(-t / 120) for t in times]
    curve = {"time_min": times, "concentration": [v * prediction_scale for v in values],
             "time_unit": "min", "unit": "umol/l"}
    observations = {"observations": [{"time_min": t, "concentration_umol_per_l": v}
                                    for t, v in zip(times, values)]}
    request = {"evidence": [{"parameter": "molecular_weight", "value": 200.}],
               "dosing": {"route": "intravenous", "dose_mg": 10., "start_h": 0., "infusion_minutes": 0.},
               "target": {"terminal_fit": {"start_h": 3., "end_h": 6., "min_points": 4,
                    "min_adjusted_r2": .98, "min_span_half_lives": 1., "max_auc_extrapolated_fraction": .2,
                    "max_aumc_extrapolated_fraction": .2}}}
    return score_case(curve, observations, request)


@pytest.mark.parametrize("scale,screen,target", [(1., "pass", "pass"), (1.5, "pass", "pass"),
    (2., "pass", "fail"), (.5, "pass", "fail"), (3., "fail", "fail")])
def test_real_score_case_integration_and_inclusive_boundaries(scale, screen, target):
    metrics = _score(scale)
    before = copy.deepcopy(metrics)
    result = assess_prediction_quality(metrics)
    for entry in [result["curve"], *(result["endpoints"][k] for k in ("auc", "sampled_cmax", "c_last"))]:
        assert entry["research_screening"]["status"] == screen
        assert entry["optimization_target"]["status"] == target
    assert result["curve"]["diagnostics"]["afe"] == pytest.approx(scale)
    assert result["endpoints"]["terminal_half_life"]["value"] == pytest.approx(1.)
    assert result["clinical_readiness"] == "not_assessed"
    assert metrics == before
    json.dumps(result, allow_nan=False)


def test_directional_bias_does_not_hide_opposing_large_curve_errors():
    result = curve_diagnostics([1., 4.], [4., 1.])
    assert result["afe"] == pytest.approx(1.)
    assert result["gmfe"] == pytest.approx(4.)
    assert result["log_rmse"] == pytest.approx(math.log(4))
    assert result["within_twofold_count"] == 0
    assert result["within_twofold_fraction"] == 0.


@pytest.mark.parametrize("observed,predicted", [([], []), ([1.], []), (None, [1.]),
    ([0.], [1.]), ([1.], [0.]), ([-1.], [1.]), ([1.], [float("nan")]),
    ([float("inf")], [1.]), ([True], [1.]), ([1., 2.], [1., None]),
    ("12", [1., 2.])])
def test_invalid_curve_pairs_are_not_dropped_or_counted_as_zero_error(observed, predicted):
    result = curve_diagnostics(observed, predicted)
    assert result["status"] == "unassessable"
    assert result["gmfe"] is None
    assert result["within_twofold_fraction"] is None
    json.dumps(result, allow_nan=False)


def test_missing_references_cannot_pass_even_if_reported_error_is_perfect():
    result = assess_prediction_quality({"gmfe": 1., "point_count": 10, "auc": {"fold_error": 1.}})
    assert result["curve"]["research_screening"]["status"] == "unassessable"
    for entry in result["endpoints"].values():
        assert entry["research_screening"]["status"] == "unassessable"
        assert entry["value"] is None


@pytest.mark.parametrize("bad", [0., -1., float("nan"), float("inf"), True, None])
def test_invalid_scalar_reference_does_not_pass(bad):
    result = assess_prediction_quality({"auc": {"observed_umol_h_l": bad,
        "predicted_umol_h_l": 1., "fold_error": 1.}})
    assert result["endpoints"]["auc"]["value"] is None
    assert result["endpoints"]["auc"]["research_screening"]["status"] == "unassessable"
    json.dumps(result, allow_nan=False)


def test_tmax_zero_error_is_retained_without_inventing_a_tolerance():
    metrics = _score()
    result = assess_prediction_quality(metrics)["endpoints"]["sampled_tmax"]
    assert result["value"] == 0.
    assert result["research_screening"]["status"] == "unassessable"
    assert result["reason"] == "no_predeclared_time_tolerance"
    assert assess_prediction_quality(metrics, tmax_tolerance_h=0.)["endpoints"]["sampled_tmax"]["research_screening"]["status"] == "pass"


def test_tmax_uses_absolute_time_not_fold_error():
    metrics = {"sampled_tmax": {"observed_h": 0., "predicted_h": .5, "absolute_error_h": .5}}
    assert assess_prediction_quality(metrics, tmax_tolerance_h=.5)["endpoints"]["sampled_tmax"]["research_screening"]["status"] == "pass"
    assert assess_prediction_quality(metrics, tmax_tolerance_h=.25)["endpoints"]["sampled_tmax"]["research_screening"]["status"] == "fail"


@pytest.mark.parametrize("bad", [-1., float("nan"), float("inf"), True])
def test_invalid_tmax_tolerance_is_rejected(bad):
    with pytest.raises(ValueError, match="tmax_tolerance"):
        assess_prediction_quality({}, tmax_tolerance_h=bad)


def test_unidentified_terminal_slope_does_not_borrow_native_prediction():
    metrics = _score()
    metrics["terminal_half_life"]["status"] = "not_estimable"
    result = assess_prediction_quality(metrics)["endpoints"]["terminal_half_life"]
    assert result["value"] is None
    assert result["research_screening"]["status"] == "unassessable"


@pytest.mark.parametrize("path,bad", [("gmfe", 0.), ("gmfe", float("nan")),
    ("gmfe", 1.4), ("point_count", 99), ("point_count", True)])
def test_corrupt_stored_curve_metrics_do_not_pass(path, bad):
    metrics = _score()
    metrics[path] = bad
    assert assess_prediction_quality(metrics)["curve"]["research_screening"]["status"] == "unassessable"


def test_corrupt_stored_endpoint_error_does_not_pass():
    metrics = _score(3.)
    metrics["auc"]["fold_error"] = 1.
    metrics["sampled_tmax"]["absolute_error_h"] = float("nan")
    result = assess_prediction_quality(metrics)["endpoints"]
    assert result["auc"]["value"] is None
    assert result["sampled_tmax"]["value"] is None


def test_extreme_finite_values_do_not_crash_or_emit_infinite_errors():
    diagnostics = curve_diagnostics([1e-300], [1e300])
    assert diagnostics["status"] == "unassessable"
    result = assess_prediction_quality({"auc": {"observed_umol_h_l": 1e-300,
        "predicted_umol_h_l": 1e300}})
    assert result["endpoints"]["auc"]["value"] is None
    json.dumps(result, allow_nan=False)
