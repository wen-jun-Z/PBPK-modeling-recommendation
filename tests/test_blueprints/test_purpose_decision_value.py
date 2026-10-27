"""Synthetic, outcome-free checks of whole-model choice and evidence gates."""
from copy import deepcopy
import math

import pytest

from pbpk_kb.blueprint_policy import evidence_decision as old
from pbpk_kb.blueprint_policy.purpose_decision_value import (
    METHODS, decide_value, validate_request,
)
from test_evidence_decision import request, fallback, record, evidence


def candidates(native=10., capacity=20.):
    def prediction(value):
        return {"status": "predicted", "endpoints": {
            "clearance": {"value": value, "unit": "l/h", "status": "completed"},
            "vss": {"value": 1000. / value, "unit": "l", "status": "completed"},
            "tmax": {"value": 0., "unit": "h", "status": "completed"}},
            "curve": {"time_h": [0., 1., 2.], "concentration_mg_l": [0., value, value / 2]}}
    return {"parameter_only": prediction(native), "combined": prediction(capacity)}


def references(local=(10., 10., 10.), globally=(20., 20., 20.)):
    return {"clearance": {"unit": "l/h", "role": "training_reference_distribution",
            "local_log_reference_values": list(map(math.log, local)),
            "global_log_reference_values": list(map(math.log, globally)),
            "source": {"path": "synthetic/training.json", "sha256": "a" * 64}}}


def decide(method="purpose_decision", *, req=None, pool=None, distributions=None, primary="clearance"):
    return decide_value(request() if req is None else req, candidates() if pool is None else pool,
                        references() if distributions is None else distributions,
                        primary_endpoint=primary, method=method)


def test_requested_primary_selects_one_complete_model_without_mutation_or_endpoint_mixing():
    req, pool, distributions = request(), candidates(), references()
    before = deepcopy((req, pool, distributions))
    result = decide(req=req, pool=pool, distributions=distributions)
    assert result["selected_candidate"] == "parameter_only"
    assert result["primary_endpoint"] == "clearance"
    assert result["requested_endpoints"] == req["target"]["endpoints"]
    assert result["whole_candidate_selection"] is True
    assert "endpoints" not in result and "selected_prediction" not in result
    assert (req, pool, distributions) == before


def test_global_local_and_fixed_methods_share_candidates_but_keep_declared_objectives():
    expected = {"fixed_native": "parameter_only", "fixed_capacity": "combined",
                "global_decision": "combined", "local_risk_only": "combined",
                "purpose_decision": "parameter_only"}
    for method, choice in expected.items():
        result = decide(method)
        assert result["selected_candidate"] == choice
        local = method in {"local_risk_only", "purpose_decision"}
        assert result["reference_kind"] == ("local" if local else "global") + "_log_reference_values"
    assert decide("purpose_decision")["expected_log_error"] == 0.
    assert decide("global_decision")["expected_log_error"] == 0.


@pytest.mark.parametrize("method", METHODS)
def test_every_method_has_legal_fallback_and_keeps_failure(method):
    pool = candidates()
    pool["combined"] = {"status": "failed", "reason": "capacity_floor", "endpoints": {}}
    result = decide(method, pool=pool)
    assert result["selected_candidate"] == "parameter_only"
    assert result["candidate_assessments"]["combined"]["feasible"] is False
    assert result["candidate_assessments"]["combined"]["reasons"] == ["candidate_not_predicted"]


def test_missing_secondary_endpoint_never_borrows_other_candidates_value():
    pool = candidates()
    pool["parameter_only"]["endpoints"].pop("vss")
    result = decide(pool=pool)
    assert result["selected_candidate"] == "parameter_only"
    assert "vss" not in pool[result["selected_candidate"]]["endpoints"]


@pytest.mark.parametrize("fault", ["missing", "failed", "negative", "nan", "wrong_unit", "bool", "zero"])
def test_invalid_primary_cannot_be_selected(fault):
    pool = candidates()
    endpoint = pool["combined"]["endpoints"]["clearance"]
    if fault == "missing":
        endpoint["value"] = None
    elif fault == "failed":
        endpoint["status"] = "not_estimable"
    elif fault == "wrong_unit":
        endpoint["unit"] = "ml/min/kg"
    else:
        endpoint["value"] = {"negative": -1., "nan": math.nan, "bool": True, "zero": 0.}[fault]
    result = decide("fixed_capacity", pool=pool)
    assert result["selected_candidate"] == "parameter_only"
    assert result["candidate_assessments"]["combined"]["feasible"] is False


def test_no_complete_prediction_is_blocked_not_missing_reference_success():
    pool = candidates()
    for value in pool.values():
        value.pop("curve")
    result = decide(pool=pool)
    assert result["status"] == "blocked" and result["selected_candidate"] is None
    assert result["risk"] is None


@pytest.mark.parametrize("method", METHODS)
def test_no_reference_is_explicitly_unassessed(method):
    result = decide(method, distributions={})
    assert result["status"] == "unassessed"
    assert result["selected_candidate"] in candidates()
    assert result["risk"] is None and result["expected_log_error"] is None
    assert result["review_required"] is None


def test_review_frequency_is_not_a_calibrated_clinical_probability():
    result = decide("fixed_capacity", distributions=references(globally=(1., 100.)))
    assert result["risk"] == 1. and result["review_required"] is True
    assert result["status"] == "review_required"
    assert result["clinical_validation"] is False
    assert "uncalibrated" in result["score_interpretation"]
    assert "not total" in result["conditional_fu_interpretation"]
    assert "review_threshold_not_a_clinical_safety_boundary" in result["limitations"]


def test_exact_half_review_threshold_and_strictly_greater_than_twofold():
    result = decide("fixed_native", distributions=references(globally=(10., 1.)))
    assert result["risk"] == .5 and result["review_required"] is True
    equal = decide("fixed_native", distributions=references(globally=(20.,)))
    assert equal["risk"] == 0.


@pytest.mark.parametrize("method", ["purpose_decision", "global_decision"])
def test_numerical_ties_prefer_capacity(method):
    pool = candidates(10., 10. * math.exp(1e-10))
    result = decide(method, pool=pool, distributions=references(globally=(10.,)))
    assert result["selected_candidate"] == "combined"


@pytest.mark.parametrize("method", METHODS)
def test_unsupported_mechanism_gates_all_methods(method):
    req = request()
    req["purpose"] = {"mechanism": "ddi"}
    result = decide(method, req=req)
    assert result["status"] == "blocked" and result["selected_candidate"] is None
    assert "unsupported_mechanism_for_this_pilot" in result["reasons"]


def test_target_evidence_is_rejected_before_candidate_scoring():
    req = request()
    record(req, "total_clearance")["context"]["study_role"] = "evaluation"
    result = decide(req=req)
    assert result["status"] == "blocked"
    assert any("held_out_outcome_evidence" in reason for reason in result["reasons"])
    assert result["candidate_assessments"] == {}


def test_extratrees_is_accepted_without_mutating_legacy_or_renaming_predictor():
    req, spare = request(), fallback()
    req["evidence"] = [r for r in req["evidence"] if r["parameter"] not in {"fraction_unbound", "total_clearance"}]
    for row in spare:
        row["prediction_method"]["name"] = "extra_trees"
    old_globals = old.decide.__globals__["_rf_qualified"]
    before = old.decide(req, spare)
    assert before["status"] == "blocked"
    result = validate_request(req, spare)
    assert result["status"] == "ready"
    assert all(row["prediction_method"]["name"] == "extra_trees" for row in result["selected_evidence"]
               if row["parameter"] in {"fraction_unbound", "total_clearance"})
    assert old.decide(req, spare) == before
    assert old.decide.__globals__["_rf_qualified"] is old_globals


@pytest.mark.parametrize("change", [{"training_overlap": True}, {"training_overlap": 0},
    {"applicability": {"human_IV_parameter_model": True}},
    {"applicability": {"human_IV_parameter_model": 1, "target_structure_and_scaffold_excluded": True}},
    {"name": "xgboost"}, {"version": " "}])
def test_extended_predictor_qualification_is_strict(change):
    req, spare = request(), fallback()
    req["evidence"] = [r for r in req["evidence"] if r["parameter"] != "fraction_unbound"]
    spare[0]["prediction_method"].update(name="extra_trees")
    spare[0]["prediction_method"].update(change)
    assert validate_request(req, spare)["status"] == "blocked"


def test_measured_fu_and_renal_inputs_are_preserved():
    req = request()
    original = deepcopy(record(req, "fraction_unbound"))
    record(req, "elimination_scope")["value"] = "hepatic_and_renal"
    renal = evidence("renal_clearance", 1., "ml/min/kg", context={
        "species": "human", "matrix": "plasma", "route": "intravenous"})
    req["evidence"].append(renal)
    result = validate_request(req, fallback())
    assert next(row for row in result["selected_evidence"] if row["parameter"] == "fraction_unbound") == original
    assert next(row for row in result["selected_evidence"] if row["parameter"] == "renal_clearance") == renal
    assert result["calibration"]["protect_net_renal"] is True
    assert result["calibration"]["clearance_scale_scope"] == "nonrenal_only"


def test_missing_primary_outside_purpose_and_unknown_candidates_are_rejected():
    req = request()
    req["target"]["endpoints"] = ["vss"]
    req["target"]["primary_endpoint"] = "vss"
    with pytest.raises(ValueError, match="belong"):
        decide(req=req)
    with pytest.raises(ValueError, match="Only"):
        decide(pool={"unregistered": {}})


@pytest.mark.parametrize("change", [{"unit": "ml/min/kg"},
    {"role": "evaluation"},
    {"local_log_reference_values": [math.inf]}, {"local_log_reference_values": [True]},
    {"local_log_reference_values": "1,2"}])
def test_bad_reference_distribution_never_silently_changes_units_or_values(change):
    distributions = references()
    distributions["clearance"].update(change)
    with pytest.raises(ValueError):
        decide(distributions=distributions)


def test_zero_tmax_and_curve_primary_are_research_outputs_without_log_scoring():
    for primary, unit in (("tmax", "h"), ("plasma_concentration_time", "mg/l")):
        distributions = {primary: {"unit": unit, "local_log_reference_values": [0.], "global_log_reference_values": [0.]}}
        result = decide(primary=primary, distributions=distributions)
        assert result["status"] == "unassessed" and result["selected_candidate"] == "combined"
        assert result["risk"] is None


def test_sources_and_qualification_are_preserved_for_audit():
    req, distributions = request(), references()
    result = decide(req=req, distributions=distributions)
    assert result["distribution_provenance"]["source"] == distributions["clearance"]["source"]
    assert result["input_gate"]["selected_evidence"]
    assert result["input_gate"]["source_verification"] == "required_by_caller_not_performed_here"
    result["input_gate"]["selected_evidence"][0]["source"]["path"] = "changed"
    assert req == request()


@pytest.mark.parametrize("same", ["path", "sha256"])
def test_explicit_forbidden_target_source_blocks_even_when_context_omits_role(same):
    req = request()
    actual = record(req, "total_clearance")["source"]
    forbidden = {"path": "synthetic/outcomes.json", "sha256": "f" * 64, "location": "target"}
    forbidden[same] = actual[same]
    req["forbidden_sources"] = [forbidden]
    result = decide(req=req)
    assert result["status"] == "blocked"
    assert any(reason.startswith("forbidden_outcome_source:") for reason in result["reasons"])
