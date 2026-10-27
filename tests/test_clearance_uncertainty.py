from copy import deepcopy
import math

import pytest

from pbpk_kb.blueprint_policy import clearance_uncertainty as m


def request(basis="predicted"):
    return {"compound_id": "drug:a", "human": {"weight_kg": 70.},
        "dosing": {"route": "intravenous", "dose_mg": 100.},
        "target": {"endpoints": list(m.SUPPORTED)}, "evidence": [{"parameter": "total_clearance",
            "compound_id": "drug:a", "unit": "ml/min/kg", "value": 2., "basis": basis,
            "context": {"independent_of_target": True, "species": "human", "matrix": "plasma",
                "study_role": "independent_parameters", "route": "intravenous"}, "source": {"path": "a"}}]}


def interval():
    return {"lower": .5, "median": 2., "upper": 8., "unit": "mL/min/kg",
        "calibration": "split_CQR", "clinical_coverage_guaranteed": False}


def test_preserves_inputs_and_only_supported_uncertainty_claims():
    q, ci = request(), interval(); before = deepcopy([q, ci]); d = m.decide(q, ci)
    assert [q, ci] == before
    assert d["action"] == "request_clearance_evidence"
    assert d["endpoint_decisions"]["auc_inf"] == "request_clearance_evidence"
    assert d["endpoint_decisions"]["auc"] == "uncertainty_unassessed"
    assert d["endpoint_decisions"]["plasma_concentration_time"] == "uncertainty_unassessed"
    assert d["systemic_auc_inf_cl_only_range"]["lower"] == pytest.approx(100/(8*.06*70))
    assert d["systemic_auc_inf_cl_only_range"]["upper"] == pytest.approx(100/(.5*.06*70))
    assert not d["systemic_auc_inf_cl_only_range"]["used_as_predicted_NCA_endpoint"]


def test_independent_cl_kept_without_claiming_zero_uncertainty():
    d = m.decide(request("measured"), None)
    assert d["risk_score"] is None and d["interval"] is None
    assert d["action"] == "preserve_independent_clearance"
    assert set(d["endpoint_decisions"].values()) == {"uncertainty_unassessed"}


@pytest.mark.parametrize("field,value", [("median", 3.), ("lower", 0.), ("upper", 1.),
    ("unit", "L/h"), ("calibration", "in_sample"), ("clinical_coverage_guaranteed", True)])
def test_rejects_invalid_or_wrong_interval(field, value):
    ci = interval();ci[field] = value
    with pytest.raises(ValueError):m.decide(request(), ci)


def test_forbids_heldout_clearance():
    q = request();q["evidence"][0]["context"]["study_role"] = "target"
    with pytest.raises(ValueError):m.decide(q, interval())


def test_selection_ties_are_not_broken_using_case_name_or_outcome():
    rows = [{"score": 1., "gmfe": y} for y in [1., 2., 4., 8.]]
    summary = m._selective_curve(rows)
    for point in summary["risk_coverage"]:
        assert point["gmfe"] == pytest.approx(math.sqrt(8.))
    assert summary["high_error_auroc"] == .5


def test_failure_remains_in_coverage_and_measured_not_lowest_risk():
    rows = [{"score": 1., "gmfe": 1.}, {"score": 2., "gmfe": 4.},
            {"score": 3., "gmfe": None}, {"score": None, "gmfe": 1.}]
    s = m._selective_curve(rows)
    assert s["eligible_predicted_cl_attempts"] == 3 and s["unscored_or_failed"] == 1
    assert s["risk_coverage"][-1]["coverage_of_eligible_attempts"] == 1.
    assert s["risk_coverage"][-1]["scored_coverage_of_eligible_attempts"] == 2/3
    assert s["high_error_auroc"] == 1.


def test_narrow_parameter_interval_is_not_clinical_acceptance():
    ci = interval();ci.update(lower=1.5, upper=2.5)
    d = m.decide(request(), ci)
    assert d["action"] == "research_parameter_range_within_tolerance"
    assert not d["clinical_coverage_guaranteed"]
    assert d["endpoint_decisions"]["cmax"] == "uncertainty_unassessed"


def test_legacy_context_preserved_only_with_exact_record_and_receipts():
    q = request("measured"); r = q["evidence"][0]
    r["context"].pop("independent_of_target"); r["context"]["study_role"] = "development"
    proof = {"record": deepcopy(r), "source_qualified_protocol": {"path": "protocol.json", "sha256": "a"*64},
        "source_outcome": {"path": "outcome.json", "sha256": "b"*64}}
    before = deepcopy(q)
    d = m.decide(q, None, reviewed_legacy_source=proof)
    assert d["action"] == "preserve_independent_clearance" and q == before
    proof["record"]["value"] = 9.
    with pytest.raises(ValueError): m.decide(q, None, reviewed_legacy_source=proof)


def test_blocked_construction_has_no_new_prior_or_measured_cl_claim():
    d = m.decide(request(), None, construction_blocked="Unsupported pKa sites")
    assert d["action"] == "construction_blocked" and d["risk_score"] is None
    out = m.summarize([{"fingerprint": "x", "case_id": "a", "condition": "all_ml"}], {"x": d})
    assert out["conditions"]["all_ml"]["construction_blocked"] == 1
    assert out["conditions"]["all_ml"]["measured_cl_uncertainty_not_evaluated"] == 0


def test_lowest_risk_failure_is_selected_before_successful_high_risk_cases():
    rows = [{"score": i, "gmfe": (None if i == 0 else float(i+1))} for i in range(4)]
    points = m._selective_curve(rows)["risk_coverage"]
    assert points[0]["selected_attempt_weight"] == 1
    assert points[0]["selected_failed_or_unscored_weight"] == 1
    assert points[0]["gmfe"] is None
    assert points[1]["selected_attempt_weight"] == 2
    assert points[1]["selected_scored_weight"] == 1 and points[1]["gmfe"] == 2.
    rows[0]["gmfe"] = 20.
    changed = m._selective_curve(rows)["risk_coverage"]
    assert [p["cutoff_log_cl_interval_width"] for p in points] == [p["cutoff_log_cl_interval_width"] for p in changed]
