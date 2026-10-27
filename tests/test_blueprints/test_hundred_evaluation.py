"""Post-lock held-out scoring: identity, labels, common coverage and nulls."""
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

import pytest

from pbpk_kb.blueprint_policy import hundred_evaluation as scoring

ROOT = Path(__file__).resolve().parents[2]


def _seal(row):
    row["inline_prediction_sha256"] = {p["scenario_id"]: scoring._digest(p) for p in row["predictions"]}
    baseline = next((p for p in row["predictions"] if p["scenario_id"] == "baseline"), None)
    row["fixedauto_baseline"] = {"scenario_id": "baseline", "same_nominal_prediction": True,
                                 "prediction_sha256": scoring._digest(baseline) if baseline else None}


@pytest.fixture
def cohort(tmp_path):
    if not tmp_path.resolve().is_relative_to(ROOT):
        pytest.skip("Use --basetemp inside the project runtime directory")
    cases, outcomes = [], []
    for i, prediction in enumerate([10., 30., 15., None, 12.]):
        cid = f"case_{i}"
        identity = "C" * (i + 1)
        path = tmp_path / f"{cid}_evaluation.json"
        evaluation = {
            "compound_id": cid, "identity_smiles": identity, "evaluation_scope": scoring.SCOPE,
            "clinical_curve": None, "weight_kg": 70.,
            "endpoints": {"clearance": {"value": 10. if i < 4 else None, "unit": "L/h"},
                          "vss": {"value": 20. if i < 4 else None, "unit": "L"},
                          "mrt": {"value": None, "unit": "h"},
                          "terminal_half_life": {"value": None, "unit": "h"}},
            "source": {"dataset": "synthetic aggregated human-IV labels", "row": i},
        }
        path.write_text(json.dumps(evaluation))
        case = {"case_id": cid, "compound_id": cid, "identity_smiles": identity,
                "drug_name": f"Synthetic {i}", "evaluation_scope": scoring.SCOPE,
                "evaluation": {"path": str(path.relative_to(ROOT)), "sha256": sha256(path.read_bytes()).hexdigest()}}
        cases.append(case)
        status = ["within_declared_limits", "within_declared_limits", "sensitive", "unassessed", "within_declared_limits"][i]
        pred = {"scenario_id": "baseline", "status": "predicted" if prediction is not None else "failed",
                "endpoints": {}, "curve": {"time_h": [0., 1., 2.], "concentration_mg_l": [1., .5, .2]}}
        if prediction is not None:
            pred["endpoints"] = {
                "clearance": {"value": prediction, "unit": "l/h", "status": "completed"},
                "vss": {"value": prediction * 2, "unit": "l", "status": "completed"},
                "tmax": {"value": 0., "unit": "h", "status": "completed"},
            }
        else:
            pred["reason"] = "synthetic solver failure"
        outcome = {k: case[k] for k in ("case_id", "compound_id", "identity_smiles", "evaluation_scope")}
        outcome.update(status="locked", construction_status="predicted" if prediction is not None else "failed",
                       baseline_scenario_id="baseline", predictions=[pred],
                       input_only_score=[2., 1., 3., None, 2.][i],
                       failures=[] if prediction is not None else [{"reason": "synthetic solver failure"}],
                       assessment={"endpoint_assessments": {
                           name: {"status": status, "max_change": [1., 4., 2., None, 3.][i], "unbounded_change": False}
                           for name in scoring.ENDPOINT_UNITS}})
        _seal(outcome)
        outcomes.append(outcome)
    return {"cases": cases, "expected_compounds": len(cases)}, outcomes


def _rewrite_evaluation(protocol, index, mutate):
    reference = protocol["cases"][index]["evaluation"]
    path = ROOT / reference["path"]
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value))
    reference["sha256"] = sha256(path.read_bytes()).hexdigest()


def test_all_compounds_and_eleven_endpoints_retained_with_exact_error_denominators(cohort):
    protocol, outcomes = cohort
    report = scoring.score_hundred(protocol, outcomes, ROOT)
    assert report["total_compounds"] == 5 and len(report["records"]) == 55
    assert report["construction"]["predicted"] == 4
    assert report["construction"]["failed"] == 1
    assert report["construction"]["failed_scenarios"] == 1
    assert report["construction"]["failures"][0]["case_id"] == "case_3"
    metric = report["endpoint_metrics"]["clearance"]
    assert metric["reference_available_n"] == 4
    assert metric["prediction_available_n"] == 4
    assert metric["scorable_n"] == 3
    assert metric["missing_reference_n"] == 1
    assert metric["reference_available_prediction_missing_n"] == 1
    assert metric["nominal"]["gmfe"] == pytest.approx((3 * 1.5) ** (1 / 3))
    assert metric["nominal"]["within_twofold_rate"] == pytest.approx(2/3)
    assert metric["accepted_n"] == 3 and metric["accepted_coverage_all_compounds"] == .6
    assert metric["accepted_scorable_n"] == 2 and metric["false_acceptance_n"] == 1
    assert metric["false_acceptance_rate_among_accepted_scorable"] == .5
    assert metric["false_rejection_n"] == 1
    assert metric["false_rejection_rate_among_within_twofold"] == .5
    assert metric["fixed_auto_baseline"]["nominal"] == metric["nominal"]
    assert metric["fixed_auto_baseline"]["accepted_coverage_all_compounds"] == .8
    curve = report["endpoint_metrics"]["plasma_concentration_time"]
    assert curve["prediction_available_n"] == 4 and curve["scorable_n"] == 0
    assert report["clinical_curve_validation_n"] == 0
    rows = [r for r in report["records"] if r["endpoint"] not in {"clearance", "vss"}]
    assert all(r["reference_value"] is None and r["fold_error"] is None for r in rows)
    assert report["bootstrap"]["unit"] == "compound" and not report["bootstrap"]["study_clustered"]
    assert report["bootstrap"]["repeats"] == 2000
    json.dumps(report, allow_nan=False)


def test_ranking_compares_same_pool_and_coverage_with_paired_deterministic_bootstrap(cohort):
    protocol, outcomes = cohort
    original = scoring.score_hundred(protocol, outcomes, ROOT)
    risk = original["risk_coverage"]["clearance"]
    assert risk["common_scorable_pool_n"] == 3 and risk["excluded_from_numeric_risk_n"] == 2
    first = risk["points"][0]
    assert first["selected_n"] == 1
    assert first["actual_coverage_scorable_pool"] == pytest.approx(1/3)
    assert first["actual_coverage_registered_compounds"] == .2
    methods = first["methods"]
    assert methods["endpoint_sensitivity"]["selected_case_ids"] == ["case_0"]
    assert methods["input_only"]["selected_case_ids"] == ["case_1"]
    assert methods["endpoint_sensitivity"]["over_twofold_rate"] == 0.
    assert methods["input_only"]["over_twofold_rate"] == 1.
    assert first["paired_difference_sensitivity_minus_input_only"]["mean_absolute_ln_error"] == pytest.approx(-math.log(3))
    last = risk["points"][-1]
    assert last["paired_difference_sensitivity_minus_input_only"]["mean_absolute_ln_error"] == 0.
    assert last["paired_difference_sensitivity_minus_input_only"]["mean_absolute_ln_error_ci95"] == [0., 0.]
    reordered = scoring.score_hundred({**protocol, "cases": list(reversed(protocol["cases"]))}, list(reversed(outcomes)), ROOT)
    assert reordered["risk_coverage"] == original["risk_coverage"]
    assert reordered["endpoint_metrics"] == original["endpoint_metrics"]
    assert reordered["records"] == original["records"]


@pytest.mark.parametrize("break_kind", ["missing", "unlocked_last", "duplicate", "prediction_tampered", "fixed_different", "identity"])
def test_lock_barrier_is_global_and_precedes_any_target_read(cohort, monkeypatch, break_kind):
    protocol, outcomes = deepcopy(cohort)
    def no_read(*args):
        raise AssertionError("Held-out reference was read before the full lock barrier")
    monkeypatch.setattr(scoring, "_checked_evaluation", no_read)
    if break_kind == "missing":
        outcomes.pop()
    elif break_kind == "unlocked_last":
        outcomes[-1]["status"] = "running"
    elif break_kind == "duplicate":
        outcomes[-1] = outcomes[0]
    elif break_kind == "prediction_tampered":
        outcomes[-1]["predictions"][0]["endpoints"]["clearance"]["value"] *= 2
    elif break_kind == "fixed_different":
        outcomes[-1]["fixedauto_baseline"]["prediction_sha256"] = "0" * 64
    else:
        outcomes[-1]["identity_smiles"] = "N"
    with pytest.raises(ValueError):
        scoring.score_hundred(protocol, outcomes, ROOT)


def test_default_requires_one_hundred_not_silent_small_batch(cohort):
    protocol, outcomes = cohort
    protocol.pop("expected_compounds")
    with pytest.raises(ValueError, match="All registered compounds"):
        scoring.score_hundred(protocol, outcomes, ROOT)


@pytest.mark.parametrize("kind", ["hash", "unit", "identity", "curve", "nonfinite", "extra_endpoint"])
def test_reference_hash_units_identity_scope_are_enforced(cohort, kind):
    protocol, outcomes = cohort
    if kind == "hash":
        protocol["cases"][0]["evaluation"]["sha256"] = "0" * 64
    elif kind == "unit":
        _rewrite_evaluation(protocol, 0, lambda x: x["endpoints"]["clearance"].update(unit="ml/min/kg"))
    elif kind == "identity":
        _rewrite_evaluation(protocol, 0, lambda x: x.update(identity_smiles="NN"))
    elif kind == "curve":
        _rewrite_evaluation(protocol, 0, lambda x: x.update(clinical_curve={"time_h": [1, 2]}))
    elif kind == "nonfinite":
        _rewrite_evaluation(protocol, 0, lambda x: x["endpoints"]["clearance"].update(value=float("nan")))
    else:
        _rewrite_evaluation(protocol, 0, lambda x: x["endpoints"].update(cmax={"value": 2, "unit": "mg/l"}))
    with pytest.raises(ValueError):
        scoring.score_hundred(protocol, outcomes, ROOT)


def test_missing_scores_and_unassessed_status_do_not_become_correct_rejections(cohort):
    protocol, outcomes = cohort
    # A partial successful stress run may give a small finite maximum, but a
    # failed/missing scenario makes this an unassessed range, not a low risk.
    outcomes[0]["assessment"]["endpoint_assessments"]["clearance"].update(status="unassessed", max_change=1.)
    outcomes[0]["input_only_score"] = None
    outcomes[2]["assessment"]["endpoint_assessments"]["clearance"].update(unbounded_change=True)
    report = scoring.score_hundred(protocol, outcomes, ROOT)
    metric = report["endpoint_metrics"]["clearance"]
    assert metric["unassessed_scorable_n"] == 1 and metric["unassessed_within_twofold_n"] == 1
    assert metric["false_rejection_n"] == 1
    risk = report["risk_coverage"]["clearance"]
    assert risk["unknown_rank_score_n"] == {"endpoint_sensitivity": 2, "input_only": 1}
    assert risk["points"][0]["methods"]["endpoint_sensitivity"]["selected_case_ids"] == ["case_1"]


def test_invalid_prediction_kept_as_failure_to_score_not_silently_converted(cohort):
    protocol, outcomes = cohort
    outcomes[0]["predictions"][0]["endpoints"]["clearance"]["unit"] = "ml/min/kg"
    _seal(outcomes[0])
    report = scoring.score_hundred(protocol, outcomes, ROOT)
    metric = report["endpoint_metrics"]["clearance"]
    assert metric["scorable_n"] == 2 and metric["reference_available_prediction_missing_n"] == 2
    row = next(r for r in report["records"] if r["case_id"] == "case_0" and r["endpoint"] == "clearance")
    assert row["prediction_missing_reason"] == "noncanonical_prediction_unit"
    assert row["reference_value"] == 10. and row["fold_error"] is None


def test_target_values_change_scores_not_predictions_or_sensitivity_ranking(cohort):
    protocol, outcomes = cohort
    before_predictions = deepcopy(outcomes)
    before = scoring.score_hundred(protocol, outcomes, ROOT)
    _rewrite_evaluation(protocol, 0, lambda x: x["endpoints"]["clearance"].update(value=100.))
    after = scoring.score_hundred(protocol, outcomes, ROOT)
    assert outcomes == before_predictions
    assert before["endpoint_metrics"]["clearance"]["nominal"] != after["endpoint_metrics"]["clearance"]["nominal"]
    for left, right in zip(before["risk_coverage"]["clearance"]["points"], after["risk_coverage"]["clearance"]["points"]):
        for method in ("endpoint_sensitivity", "input_only"):
            assert left["methods"][method]["selected_case_ids"] == right["methods"][method]["selected_case_ids"]


def test_tied_scores_use_predeclared_case_hash_order(cohort):
    protocol, outcomes = cohort
    for row in outcomes:
        row["input_only_score"] = 2.
        for item in row["assessment"]["endpoint_assessments"].values():
            item["max_change"] = 2.
    report = scoring.score_hundred(protocol, outcomes, ROOT)
    expected = sorted(["case_0", "case_1", "case_2"], key=lambda x: sha256(x.encode()).hexdigest())
    point = report["risk_coverage"]["clearance"]["points"][-1]
    assert point["methods"]["endpoint_sensitivity"]["selected_case_ids"] == expected
    assert point["methods"]["input_only"]["selected_case_ids"] == expected
