"""Four-method comparisons retain failures and cannot replace frozen labels."""
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import uuid

import pytest

from pbpk_kb.blueprint_policy import purpose_refinement_evaluation as scoring

PROJECT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cohort():
    root = PROJECT / "runtime/cache/purpose_design_tests" / ("refinement_" + uuid.uuid4().hex)
    root.mkdir(parents=True)
    cases = []
    for i in range(4):
        cid = f"case_{i}"
        label = 10. if i != 2 else None
        evaluation = {"compound_id": cid, "identity_smiles": "C" * (i + 1),
            "evaluation_scope": scoring.SCOPE, "weight_kg": 70., "clinical_curve": None,
            "endpoints": {ep: {"value": label, "unit": scoring.ENDPOINT_UNITS[ep]} for ep in scoring.ENDPOINTS}}
        path = root / (cid + ".json")
        path.write_text(json.dumps(evaluation))
        cases.append({"case_id": cid, "compound_id": cid, "identity_smiles": evaluation["identity_smiles"],
            "evaluation_scope": scoring.SCOPE, "drug_name": "Synthetic " + str(i),
            "evaluation": {"path": str(path.relative_to(PROJECT)), "sha256": sha256(path.read_bytes()).hexdigest()}})
    predictions = {"baseline": [10., 40., 10., None], "parameter_only": [10., 20., 10., 10.],
                   "capacity_only": [None, 10., 10., 20.], "combined": [10., 10., 10., 10.]}
    results = {}
    for method in scoring.METHODS:
        rows = []
        for i, case in enumerate(cases):
            for ep in scoring.ENDPOINTS:
                rows.append({**{k: case[k] for k in ("case_id", "compound_id", "identity_smiles", "evaluation_scope")},
                    "construction_status": "failed" if method == "baseline" and i == 3 else "predicted",
                    "endpoint": ep, "unit": scoring.ENDPOINT_UNITS[ep], "prediction_value": predictions[method][i],
                    "reference_value": 10. if i != 2 else None, "reference_artifact": case["evaluation"],
                    "prediction_missing_reason": "synthetic_nonestimable" if predictions[method][i] is None else None,
                    "policy_status": "unassessed", "sensitivity_score": None, "input_only_score": None})
        results[method] = {"all_predictions_locked_before_scoring": True, "records": rows}
    yield {"expected_compounds": 4, "cases": cases}, results
    shutil.rmtree(root)


def score(cohort, **kwargs):
    return scoring.score_refinement(*cohort, PROJECT, bootstrap_repeats=200, **kwargs)


def test_known_errors_common_cases_and_failure_denominators(cohort):
    result = score(cohort)
    ep = result["endpoint_metrics"]["clearance"]
    baseline = ep["methods"]["baseline"]
    assert baseline["registered_n"] == 4
    assert baseline["reference_available_n"] == 3
    assert baseline["prediction_available_n"] == 3
    assert baseline["reference_available_prediction_missing_n"] == 1
    assert baseline["construction_status_counts"] == {"predicted": 3, "failed": 1}
    assert baseline["own_scorable"]["gmfe"] == pytest.approx(2.)
    assert baseline["own_scorable"]["within_twofold_n"] == 1
    assert ep["all_methods_common_scorable_case_ids"] == ["case_1"]
    assert baseline["all_methods_common_scorable"]["gmfe"] == pytest.approx(4.)
    pair = ep["paired_comparisons"]["parameter_only_vs_baseline"]
    assert pair["common_scorable_case_ids"] == ["case_0", "case_1"]
    assert pair["right_only_scorable_n"] == 1 and pair["neither_scorable_n"] == 1
    assert pair["gmfe_ratio_right_over_left"] == pytest.approx(2 ** -.5)
    assert len(ep["paired_comparisons"]) == 6
    assert all(len(rr) == 16 for rr in result["records_by_method"].values())
    assert result["clinical_curve_validation_n"] == 0
    json.dumps(result, allow_nan=False)


def test_paired_bootstrap_constant_improvement_has_degenerate_correct_interval(cohort):
    protocol, results = deepcopy(cohort)
    for method, value in (("baseline", 40.), ("parameter_only", 20.)):
        for row in results[method]["records"]:
            if row["case_id"] in {"case_0", "case_1"}:
                row["prediction_value"] = value
    pair = score((protocol, results))["endpoint_metrics"]["clearance"]["paired_comparisons"]["parameter_only_vs_baseline"]
    assert pair["gmfe_ratio_right_over_left"] == pytest.approx(.5)
    assert pair["gmfe_ratio_ci95"] == pytest.approx([.5, .5])
    assert pair["within_twofold_rate_difference_ci95"] == [1., 1.]


def test_missing_endpoint_becomes_explicit_missing_without_dropping_case(cohort):
    protocol, results = deepcopy(cohort)
    results["combined"]["records"] = [r for r in results["combined"]["records"] if not (r["case_id"] == "case_0" and r["endpoint"] == "vss")]
    result = score((protocol, results))
    endpoint = result["endpoint_metrics"]["vss"]["methods"]["combined"]
    assert endpoint["registered_n"] == 4 and endpoint["own_scorable"]["n"] == 2
    assert endpoint["prediction_missing_reason_counts"] == {"missing_endpoint_record": 1}


def test_all_failed_method_is_retained_and_paired_metrics_stay_null(cohort):
    protocol, results = deepcopy(cohort)
    for row in results["combined"]["records"]:
        row.update(construction_status="failed", prediction_value=None)
    result = score((protocol, results))
    ep = result["endpoint_metrics"]["mrt"]
    assert ep["all_methods_common_scorable_n"] == 0
    assert ep["methods"]["combined"]["construction_status_counts"] == {"failed": 4}
    assert ep["methods"]["combined"]["reference_available_prediction_missing_n"] == 3
    assert ep["paired_comparisons"]["combined_vs_baseline"]["gmfe_ratio_right_over_left"] is None


@pytest.mark.parametrize("problem", ["missing_method", "missing_case", "unlocked", "identity", "duplicate", "pending"])
def test_global_completion_barrier_precedes_any_reference_read(cohort, monkeypatch, problem):
    protocol, results = deepcopy(cohort)
    if problem == "missing_method":
        del results["combined"]
    elif problem == "missing_case":
        results["combined"]["records"] = [r for r in results["combined"]["records"] if r["case_id"] != "case_3"]
    elif problem == "unlocked":
        results["combined"]["all_predictions_locked_before_scoring"] = False
    elif problem == "identity":
        results["combined"]["records"][0]["identity_smiles"] = "wrong"
    elif problem == "duplicate":
        results["combined"]["records"].append(deepcopy(results["combined"]["records"][0]))
    else:
        results["combined"]["records"][-1]["construction_status"] = "pending"
    def no_reference(*args):
        pytest.fail("An evaluation was read before the four-method completion barrier")
    monkeypatch.setattr(scoring, "_checked_evaluation", no_reference)
    with pytest.raises(ValueError):
        score((protocol, results))


def test_original_evaluation_hash_is_enforced(cohort):
    protocol, results = cohort
    path = PROJECT / protocol["cases"][0]["evaluation"]["path"]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="SHA256"):
        score((protocol, results))


@pytest.mark.parametrize("problem", ["changed_value", "invented_unresolved_label", "changed_source", "unit", "failed_has_value", "nan"])
def test_canonical_records_cannot_change_original_reference_or_claim_failed_prediction(cohort, problem):
    protocol, results = deepcopy(cohort)
    row = results["combined"]["records"][0]
    if problem == "changed_value":
        row["reference_value"] = 11.
    elif problem == "invented_unresolved_label":
        results["combined"]["records"][8]["reference_value"] = 10.
    elif problem == "changed_source":
        row["reference_artifact"] = {"path": "fake.json", "sha256": "0" * 64}
    elif problem == "unit":
        row["unit"] = "ml/min"
    elif problem == "failed_has_value":
        for row in results["combined"]["records"][:4]:
            row["construction_status"] = "failed"
    else:
        row["prediction_value"] = math.nan
    with pytest.raises(ValueError):
        score((protocol, results))


def test_errors_are_recomputed_instead_of_trusting_precomputed_scores(cohort):
    original = score(cohort)
    protocol, results = deepcopy(cohort)
    for result in results.values():
        for row in result["records"]:
            row.update(absolute_ln_error=0., within_twofold=True, fold_error=1., scorable=True)
    assert score((protocol, results)) == original


def test_absent_risk_scores_do_not_fabricate_a_ranking(cohort):
    result = score(cohort)
    for ep in result["endpoint_metrics"].values():
        for method in ep["methods"].values():
            risk = method["risk_coverage"]
            assert risk["status"] == "not_assessed_no_comparable_risk_scores"
            assert risk["points"] == []


def test_risk_near_ties_use_case_hash_and_ignore_float_noise():
    scores = {"a": 43.17322342821363, "b": 43.1732234282136, "c": 43.17322342821361, "d": None}
    expected = sorted(["a", "b", "c"], key=lambda x: sha256(x.encode()).hexdigest()) + ["d"]
    assert scoring.tolerant_risk_order(scores) == expected
    assert scoring.tolerant_risk_order(dict(reversed(list(scores.items())))) == expected
    assert scoring.tolerant_risk_order({"a": 43.1732234282136, "b": 43.17322342821363, "c": 43.17322342821362, "d": None}) == expected


def test_tolerance_groups_are_anchored_not_transitively_chained():
    scores = {"a": 1., "b": 1. + .75e-8, "c": 1. + 1.5e-8}
    expected = sorted(["a", "b"], key=lambda x: sha256(x.encode()).hexdigest()) + ["c"]
    assert scoring.tolerant_risk_order(scores) == expected


def test_identical_risk_order_has_zero_paired_difference_and_deterministic_bootstrap(cohort):
    protocol, results = deepcopy(cohort)
    for result in results.values():
        for row in result["records"]:
            row.update(policy_status="sensitive", sensitivity_score=43.1732234282136,
                       input_only_score=43.17322342821363)
    result = score((protocol, results))
    risk = result["endpoint_metrics"]["clearance"]["methods"]["parameter_only"]["risk_coverage"]
    assert risk["status"] == "assessed"
    for point in risk["points"]:
        assert point["difference_sensitivity_minus_input_only"]["mean_absolute_ln_error"] == 0.
        assert point["difference_sensitivity_minus_input_only"]["mean_absolute_ln_error_ci95"] == [0., 0.]
    for method in results.values():
        method["records"].reverse()
    protocol["cases"].reverse()
    assert score((protocol, results))["endpoint_metrics"] == result["endpoint_metrics"]


def test_input_records_are_not_mutated(cohort):
    original = deepcopy(cohort)
    score(cohort)
    assert cohort == original
