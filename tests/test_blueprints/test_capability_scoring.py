"""Independent facts and numerical contracts for the capability experiment.

The expectations below come from the frozen source records, not a policy's
``expected_ready`` labels or its assertions that evidence was protected.
Repeated development subsets are deliberately not counted as new patients or
independent clinical studies.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "experiments/blueprint_design/vascular_tissue_ten_drug_20260930/protocol.json"

# Independent assay/regulatory summaries, with unit conversions shown here.
# Their construction records say "derived": selecting only basis="measured"
# would incorrectly make the evidence-protection denominator empty.
EMPIRICAL_FU = {
    "Dexmedetomidine": 1 - .94,
    "Moxifloxacin": 1 - .52,
    "Nemonoxacin": 1 - .16,
    "Remdesivir": 12.1 / 100,
}
EMPIRICAL_RENAL = {
    "Moxifloxacin": 23 / 72,
    "Remdesivir": 128.5 / 70,
}


def _checked(reference):
    path = ROOT / reference["path"]
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == reference["sha256"]
    return json.loads(payload)


@pytest.fixture(scope="module")
def cohort():
    protocol = json.loads(COHORT.read_text())
    return [
        (case, _checked(case["requests"]["evidence"]["development"]),
         _checked(case["requests"]["rf"]["development"]))
        for case in protocol["cases"]
    ]


def _rows(request):
    return {row["parameter"]: row for row in request["evidence"]}


def test_protection_denominators_exclude_predictions_and_absent_renal_evidence(cohort):
    observed_fu, observed_renal = {}, {}
    for case, evidence, _ in cohort:
        rows = _rows(evidence)
        fu = rows["fraction_unbound"]
        if fu["basis"] in {"measured", "derived"}:
            observed_fu[case["drug"]] = fu["value"]
            assert fu["context"]["species"] == "human"
            assert fu["context"]["matrix"] == "plasma"
        renal = rows.get("renal_clearance")
        if renal and renal["basis"] in {"measured", "derived"}:
            observed_renal[case["drug"]] = renal["value"]
            assert renal["unit"] == "ml/min/kg"
    assert observed_fu == pytest.approx(EMPIRICAL_FU)
    assert observed_renal == pytest.approx(EMPIRICAL_RENAL)
    assert len(observed_fu) == 4
    assert len(observed_renal) == 2


def test_fu_fill_effect_denominator_is_four_changed_drugs_and_six_aliases(cohort):
    changed, aliases = set(), set()
    for case, evidence, rf in cohort:
        selected = _rows(evidence)["fraction_unbound"]
        fallback = _rows(rf)["fraction_unbound"]
        assert fallback["basis"] == "predicted"
        metadata = fallback["prediction_method"]
        assert metadata["name"] == "random_forest"
        assert metadata["version"]
        assert metadata["training_overlap"] is False
        assert metadata["applicability"]["human_IV_parameter_model"] is True
        assert metadata["applicability"]["target_structure_and_scaffold_excluded"] is True
        _checked(fallback["source"])
        (aliases if selected["value"] == fallback["value"] else changed).add(case["drug"])
    assert changed == set(EMPIRICAL_FU)
    assert len(aliases) == 6
    assert len(changed | aliases) == 10


def test_sampling_and_two_drugs_in_one_study_do_not_increase_independent_n(cohort):
    assert len(cohort) == 10
    clusters = {case["cluster_id"] for case, _, _ in cohort}
    assert len(clusters) == 9
    pletz = {case["drug"] for case, _, _ in cohort if case["cluster_id"] == "Pletz2004"}
    assert pletz == {"Ertapenem", "Ceftriaxone"}
    for case, _, _ in cohort:
        full = _checked(case["sampling_sources"]["full"])
        five = _checked(case["sampling_sources"]["five"])
        assert full["study_id"] == five["study_id"]
        assert len(five["observations"]) == 5
        full_pairs = {(row["time_min"], row["concentration_umol_per_l"])
                      for row in full["observations"]}
        assert {(row["time_min"], row["concentration_umol_per_l"])
                for row in five["observations"]} <= full_pairs


@pytest.fixture
def selector():
    from pbpk_kb.blueprint_policy.capability_calibration import select_active
    return select_active


def _orthogonal(n, scales=(1., 1., 1.)):
    result = np.zeros((n, 3))
    for index, scale in enumerate(scales[:min(n, 3)]):
        result[index, index] = scale
    return result


@pytest.mark.parametrize("n, expected", [(1, []), (3, [0]), (5, [0, 1]), (6, [0, 1, 2])])
def test_actual_observation_budget_limits_active_axis_count(selector, n, expected):
    result = selector(_orthogonal(n), n)
    assert result["active_indices"] == expected
    assert len(result["active_parameters"]) == len(expected)
    assert len(result["frozen_parameters"]) == 3 - len(expected)


def test_rank_deficiency_skips_dependent_axis_without_discarding_later_axis(selector):
    jacobian = _orthogonal(6)
    jacobian[:, 1] = 2 * jacobian[:, 0]
    original = jacobian.copy()
    selected = selector(jacobian, 6)
    assert selected["active_indices"] == [0, 2]
    assert selected["active_parameters"] == ["clearance_scale", "bidirectional_exchange_scale"]
    np.testing.assert_array_equal(jacobian, original)


def test_uninformative_clearance_does_not_preclude_independent_tissue_axes(selector):
    jacobian = _orthogonal(6)
    jacobian[:, 0] = 0.
    assert selector(jacobian, 6)["active_indices"] == [1, 2]


def test_negligible_absolute_sensitivity_cannot_pass_by_column_normalization(selector):
    assert selector(_orthogonal(6, (9.9e-7, 1., 1.)), 6)["active_indices"] == [1, 2]
    assert selector(_orthogonal(6, (1e-6, 0., 0.)), 6)["active_indices"] == [0]
    result = selector(np.zeros((6, 3)), 6)
    assert result["active_indices"] == []
    assert len(result["frozen_parameters"]) == 3


def test_relative_singular_value_threshold_has_frozen_inclusive_boundary(selector):
    assert selector(_orthogonal(6, (1., .000999, 1.)), 6)["active_indices"] == [0, 2]
    assert selector(_orthogonal(6, (1., .001, 1.)), 6)["active_indices"] == [0, 1, 2]


@pytest.mark.parametrize("jacobian, n", [
    (np.zeros((5, 3)), 6),
    (np.zeros((6, 2)), 6),
    (np.full((6, 3), np.nan), 6),
    (np.full((6, 3), np.inf), 6),
    (np.zeros((6, 3)), True),
])
def test_invalid_jacobians_cannot_silently_become_safe_to_calibrate(selector, jacobian, n):
    with pytest.raises((ValueError, TypeError)):
        selector(jacobian, n)


def test_identity_jacobian_uses_six_model_calls_and_no_concentration_outcomes(monkeypatch):
    from pbpk_kb.blueprint_policy import capability_calibration as cal
    times = np.arange(1., 7.)
    derivative = np.column_stack((-times, np.ones(6), times**2 / 10))
    calls = []

    def analytic_model(physiology, request, actual_times, variant, keys, x):
        np.testing.assert_array_equal(actual_times, times)
        calls.append(np.asarray(x).copy())
        # Closed-form independent reference: d(log P)/dx = derivative.
        return {"concentration_mg_l": np.exp(.5 - times / 10 + derivative @ x)}

    monkeypatch.setattr(cal, "_simulate", analytic_model)
    result = cal.diagnose_axes({}, {}, times)
    np.testing.assert_allclose(result["data_jacobian"], derivative / np.sqrt(6), atol=1e-11)
    assert result["linear_solver_calls"] == len(calls) == 6
    for index in range(3):
        expected = np.zeros(3)
        expected[index] = .001
        np.testing.assert_array_equal(calls[2 * index], expected)
        np.testing.assert_array_equal(calls[2 * index + 1], -expected)


def test_constant_prediction_has_no_information_even_though_penalty_has_full_rank(monkeypatch):
    from pbpk_kb.blueprint_policy import capability_calibration as cal
    monkeypatch.setattr(cal, "_simulate", lambda *args: {"concentration_mg_l": np.ones(6)})
    result = cal.diagnose_axes({}, {}, np.arange(1., 7.))
    np.testing.assert_array_equal(result["data_jacobian"], np.zeros((6, 3)))
    assert result["active_indices"] == []


_spec = importlib.util.spec_from_file_location(
    "_capability_inherited_contracts", Path(__file__).with_name("test_perfusion_calibration.py"))
_contracts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_contracts)


@pytest.fixture
def fitting(tmp_path, monkeypatch):
    from pbpk_kb.blueprint_policy import capability_calibration as cal
    monkeypatch.setattr(_contracts, "cal", cal)
    original = cal.fit_mapping

    def fit(**kwargs):
        kwargs["variant"] = "flow_exchange"
        return original(**kwargs)

    monkeypatch.setattr(cal, "fit_mapping", fit)
    return _contracts.fitting.__wrapped__(tmp_path, monkeypatch)


@pytest.mark.parametrize("contract", [
    "test_sealed_fit_selection_resume_and_cost",
    "test_fit_refuses_target_observations",
    "test_completed_fit_refuses_source_or_code_changes",
    "test_subset_conditions_must_match_request",
    "test_mapping_cannot_rewrite_both_start_scores_and_selected_decision",
])
def test_adaptive_fit_preserves_source_and_target_isolation(fitting, contract):
    getattr(_contracts, contract)(fitting)


def test_fitted_axes_project_same_starts_and_preserve_original_full_penalty(fitting):
    from pbpk_kb.blueprint_policy import capability_calibration as cal
    from pbpk_kb.blueprint_policy import transport_calibration as frozen
    result = fitting[-1]()
    assert result["active_parameters"] == ["clearance_scale", "partition_scale"]
    assert result["axis_selection"]["frozen_parameters"] == ["bidirectional_exchange_scale"]
    starts = frozen.parameter_starts("flow_exchange")[3]
    assert len(result["starts"]) == 16
    for index, row in enumerate(result["starts"]):
        expected_start = starts[index].copy()
        expected_start[2] = 0.
        np.testing.assert_array_equal(row["initial_log_parameters"], expected_start)
        assert row["log_parameters"][2] == 0.
        assert row["parameters"]["bidirectional_exchange_scale"] == 1.
        x = np.asarray(row["log_parameters"])
        objective = (row["data_only_ln_rmse"]**2 + .1 * (x[0] / np.log(2))**2
                     + .01 * (x[1] / np.log(10))**2 + .01 * (x[2] / np.log(100))**2)
        assert row["objective"] == pytest.approx(objective)
    assert result["axis_selection"]["linear_solver_calls"] == 6
    assert result["linear_solver_calls"] == 6 + sum(row["linear_solver_calls"] for row in result["starts"])
    assert cal.CONFIG["loss"] == frozen.CONFIG["loss"]


def test_no_supported_axis_retains_identity_without_calling_optimizer(fitting, monkeypatch):
    from pbpk_kb.blueprint_policy import capability_calibration as cal
    monkeypatch.setattr(cal, "_simulate", lambda *args: {"concentration_mg_l": np.ones(5)})

    def forbidden_optimizer(*args, **kwargs):
        pytest.fail("An optimizer cannot create information for a zero-sensitivity model")

    monkeypatch.setattr(cal, "least_squares", forbidden_optimizer)
    result = fitting[-1]()
    assert result["active_parameters"] == []
    assert result["eligible"]
    assert all(row["log_parameters"] == [0., 0., 0.] for row in result["starts"])
    assert all(set(row["parameters"].values()) == {1.} for row in result["starts"])


def _tail_row(error, warning, *, reference=True, method="always_three", sampling="full"):
    return {"method": method, "sampling": sampling, "reference_available": reference,
            "matched_fold_error": error, "warning": warning}


def test_tail_confusion_counts_use_strict_twofold_and_distinct_denominators():
    from pbpk_kb.blueprint_policy.capability_evaluation import score_tail_rows
    rows = [_tail_row(3., True), _tail_row(4., False),
            _tail_row(1.5, True), _tail_row(2., False)]
    original = deepcopy(rows)
    result = score_tail_rows(rows)["always_three__full"]
    assert result["requested_tail_endpoints"] == result["matched_error_available"] == 4
    assert result["large_matched_errors"] == 2
    assert result["flagged_large_errors"] == result["missed_large_errors"] == 1
    assert result["false_warnings"] == 1
    assert result["large_error_detection_rate"] == .5
    assert result["false_warning_rate"] == .5
    assert result["matched_error_above_two_among_passed"] == .5
    assert result["coverage_of_scored_references"] == .5
    assert rows == original


def test_all_warning_tail_rule_has_zero_coverage_and_undefined_passed_error():
    from pbpk_kb.blueprint_policy.capability_evaluation import score_tail_rows
    result = score_tail_rows([_tail_row(3., True), _tail_row(1., True)])["always_three__full"]
    assert result["large_error_detection_rate"] == 1.
    assert result["false_warning_rate"] == 1.
    assert result["coverage_of_scored_references"] == 0.
    assert result["matched_error_above_two_among_passed"] is None


def test_missing_and_nonfinite_tail_errors_are_neither_correct_predictions_nor_rejections():
    from pbpk_kb.blueprint_policy.capability_evaluation import score_tail_rows
    rows = [_tail_row(3., True, reference=False), _tail_row(None, True, reference=False),
            _tail_row(None, True), _tail_row(float("nan"), True),
            _tail_row(float("inf"), True), _tail_row(0., True), _tail_row(2., False)]
    result = score_tail_rows(rows)["always_three__full"]
    assert result["requested_tail_endpoints"] == 7
    assert result["reference_available"] == 5
    assert result["matched_error_available"] == 1
    assert result["warnings_with_reference_and_matched_error"] == 0
    assert result["false_warnings"] == 0
    assert result["coverage_of_scored_references"] == 1.
    assert result["large_error_detection_rate"] is None
    assert result["matched_error_above_two_among_passed"] == 0.


def test_no_eligible_tail_error_keeps_every_rate_undefined():
    from pbpk_kb.blueprint_policy.capability_evaluation import score_tail_rows
    result = score_tail_rows([_tail_row(None, True), _tail_row(float("nan"), False)])["always_three__full"]
    assert result["matched_error_available"] == 0
    for field in ("large_error_detection_rate", "false_warning_rate",
                  "matched_error_above_two_among_passed", "coverage_of_scored_references"):
        assert result[field] is None


def test_tail_summaries_do_not_pool_methods_or_correlated_sampling_regimes():
    from pbpk_kb.blueprint_policy.capability_evaluation import score_tail_rows
    result = score_tail_rows([
        _tail_row(3., True), _tail_row(3., False, method="adaptive_axes"),
        _tail_row(1., False, sampling="five")])
    assert set(result) == {"always_three__full", "adaptive_axes__full", "always_three__five"}
    assert result["always_three__full"]["large_error_detection_rate"] == 1.
    assert result["adaptive_axes__full"]["large_error_detection_rate"] == 0.
    assert result["always_three__five"]["large_error_detection_rate"] is None


@pytest.fixture
def locked_tail(tmp_path):
    from pbpk_kb.blueprint_policy.capability_evaluation import TAIL_ENDPOINTS
    folder = tmp_path / "prediction"
    folder.mkdir()

    def write(path, value):
        path.write_text(json.dumps(value, sort_keys=True))
        return {"path": str(path.relative_to(tmp_path)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    prediction = write(folder / "prediction_locked.json", {"time_min": [0., 1.], "concentration": [1., .5]})
    dense = write(folder / "endpoints.json", {
        "endpoint_estimator_uses_observations": False, "comparison_observations_used": False,
        "endpoints": {key: {"status": "completed"} for key in TAIL_ENDPOINTS}})
    receipt = write(folder / "completion.json", {"artifact_pins": {dense["path"]: dense["sha256"]}})
    computation = {"status": "predicted", "prediction": prediction,
                   "dense_endpoints": dense, "completion": receipt}
    record = {"scenario": "calibration", "method": "always_three", "sampling": "full",
              "computation_id": "locked", "case_id": "synthetic", "drug": "Synthetic",
              "endpoints": {key: {"reference": 1., "error": 3., "error_metric": "fold_error"}
                            for key in TAIL_ENDPOINTS}}
    return tmp_path, record, computation, write


def test_tail_warning_cannot_be_derived_from_target_error(locked_tail):
    from pbpk_kb.blueprint_policy.capability_evaluation import evaluate_tail_gate
    root, record, computation, _ = locked_tail
    bad = evaluate_tail_gate([record], {"locked": computation}, root)
    better = deepcopy(record)
    for endpoint in better["endpoints"].values():
        endpoint["error"] = 1.
    good = evaluate_tail_gate([better], {"locked": computation}, root)
    assert [row["warning"] for row in bad["records"]] == [False] * 6
    assert [row["warning"] for row in good["records"]] == [False] * 6
    assert bad["by_method_sampling"]["always_three__full"]["missed_large_errors"] == 6
    assert good["by_method_sampling"]["always_three__full"]["large_matched_errors"] == 0


def test_tail_gate_cannot_reseal_changed_endpoint_and_receipt_after_prediction_lock(locked_tail):
    from pbpk_kb.blueprint_policy.capability_evaluation import evaluate_tail_gate
    root, record, computation, write = locked_tail
    path = root / computation["dense_endpoints"]["path"]
    data = json.loads(path.read_text())
    for endpoint in data["endpoints"].values():
        endpoint["status"] = "not_estimable"
    altered = write(path, data)
    write(root / computation["completion"]["path"], {
        "artifact_pins": {altered["path"]: altered["sha256"]}})
    with pytest.raises(ValueError):
        evaluate_tail_gate([record], {"locked": computation}, root)
