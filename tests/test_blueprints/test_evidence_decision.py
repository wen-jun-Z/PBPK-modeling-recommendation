"""Input-policy tests use synthetic records, not clinical outcome labels."""
from copy import deepcopy

import pytest

from pbpk_kb.blueprint_policy.evidence_decision import decide


def evidence(parameter, value, unit, basis="measured", *, context=None, prediction=None):
    result = {"parameter": parameter, "compound_id": "synthetic:1", "value": value,
              "unit": unit, "basis": basis,
              "source": {"path": "synthetic/visible_evidence.json", "sha256": "a" * 64,
                         "location": parameter}, "context": context or {}, "notes": "synthetic fixture"}
    if prediction:
        result["prediction_method"] = prediction
    return result


def request():
    records = [evidence("molecular_weight", 300., "g/mol", "derived"),
               evidence("molecular_formula", "C12H20N2O4", "text", "derived"),
               evidence("logp", 1.5, "Log Units"),
               evidence("pka_types", [{"Type": "Base", "Pka": 7.5}], "1"),
               evidence("solubility", 1., "mg/ml", "assumption"),
               evidence("solubility_ph", 7.4, "1", "assumption"),
               evidence("binding_partner", "Albumin", "text", "assumption"),
               evidence("elimination_scope", "hepatic_only", "text", "assumption"),
               evidence("fraction_unbound", .1, "fraction", context={"species": "human", "matrix": "plasma"}),
               evidence("total_clearance", 5., "ml/min/kg", context={"species": "human", "matrix": "plasma", "route": "intravenous"})]
    return {"entry_reason": "model_missing", "compound_id": "synthetic:1", "compound_name": "Example",
            "human": {"age_years": 30., "weight_kg": 70., "height_cm": 175., "sex": "MALE", "population": "European_ICRP_2002"},
            "dosing": {"route": "intravenous", "dose_mg": 100., "infusion_minutes": 30.},
            "target": {"endpoints": ["plasma_concentration_time", "auc", "cmax", "tmax", "c_last", "terminal_half_life", "auc_inf", "clearance", "vz", "mrt", "vss"],
                       "primary_endpoint": "plasma_concentration_time", "window_start_h": 0., "window_end_h": 24.},
            "partition_method": "PKSimStandard", "evidence": records,
            "evidence_ledger": {"path": "synthetic/visible_evidence.json", "sha256": "a" * 64, "location": "fixture"}}


def fallback():
    pm = {"name": "random_forest", "version": "fixture-1", "training_overlap": False,
          "applicability": {"target_structure_and_scaffold_excluded": True, "human_IV_parameter_model": True}}
    return [evidence("fraction_unbound", .2, "fraction", "predicted", context={"species": "human", "matrix": "plasma"}, prediction=deepcopy(pm)),
            evidence("total_clearance", 6., "ml/min/kg", "predicted", context={"species": "human", "matrix": "plasma", "route": "intravenous"}, prediction=deepcopy(pm))]


def record(req, parameter):
    return next(e for e in req["evidence"] if e["parameter"] == parameter)


def selected(result, parameter):
    return next(e for e in result["selected_evidence"] if e["parameter"] == parameter)


@pytest.mark.parametrize("method", ["evidence_policy", "fixed_template", "rf_fill", "no_conflict_guard"])
def test_complete_input_is_ready_and_never_mutated(method):
    req, backup = request(), fallback()
    original, original_backup = deepcopy(req), deepcopy(backup)
    result = decide(req, backup, method=method)
    assert result["status"] == "ready"
    assert req == original and backup == original_backup
    assert len(result["supported_endpoints"]) == 11
    assert result["unsupported_endpoints"] == []
    result["selected_evidence"][0]["source"]["path"] = "changed"
    assert req == original and backup == original_backup


def test_evidence_priority_retains_measured_fu_and_soft_clearance_meaning():
    req = request()
    result = decide(req, fallback())
    assert selected(result, "fraction_unbound") == record(req, "fraction_unbound")
    assert selected(result, "total_clearance")["value"] == 5.
    assert result["calibration"]["allowed_parameters"] == ["lipophilicity", "clearance_scale"]
    assert result["calibration"]["clearance_evidence_role"] == "construction_prior_not_target_patient_truth"
    assert result["calibration"]["lipophilicity_role"] == "effective_logMA_prior_not_fixed_measured_logP"
    assert result["calibration"]["fraction_unbound_role"] == "retain_selected_construction_input"


def test_rf_baseline_explicitly_uses_given_backups():
    result = decide(request(), fallback(), method="rf_fill")
    assert selected(result, "fraction_unbound")["value"] == .2
    assert selected(result, "total_clearance")["value"] == 6.
    assert selected(result, "fraction_unbound")["source"] == fallback()[0]["source"]


@pytest.mark.parametrize("parameter", ["fraction_unbound", "total_clearance"])
def test_only_explicit_fallback_fills_missing_pk_inputs(parameter):
    req = request()
    req["evidence"] = [e for e in req["evidence"] if e["parameter"] != parameter]
    assert decide(req, [])["status"] == "blocked"
    assert decide(req, fallback())["status"] == "ready"
    assert decide(req, fallback(), method="fixed_template")["status"] == "blocked"


def test_missing_formula_is_not_repaired_from_unpermitted_backups():
    req = request()
    removed = record(req, "molecular_formula")
    req["evidence"].remove(removed)
    result = decide(req, fallback() + [removed])
    assert result["status"] == "blocked"
    assert "missing_required_parameter:molecular_formula" in result["reasons"]


@pytest.mark.parametrize("basis", ["measured", "derived", "predicted"])
def test_equal_priority_conflict_blocks_even_with_rf_backup(basis):
    req = request()
    if basis == "predicted":
        req["evidence"] = [e for e in req["evidence"] if e["parameter"] != "fraction_unbound"] + [fallback()[0]]
    else:
        record(req, "fraction_unbound")["basis"] = basis
    other = deepcopy(record(req, "fraction_unbound"))
    other["value"] *= 2
    req["evidence"].append(other)
    result = decide(req, fallback())
    assert result["status"] == "blocked"
    assert "fraction_unbound:equal_priority_conflict" in result["reasons"]
    assert decide(req, fallback(), method="fixed_template")["status"] == "blocked"
    ablation = decide(req, fallback(), method="no_conflict_guard")
    assert ablation["status"] == "ready"
    assert selected(ablation, "fraction_unbound")["value"] == record(req, "fraction_unbound")["value"]
    assert any(a.get("reason") == "conflict_guard_disabled_first_record" for a in ablation["actions"])


def test_measured_and_derived_records_have_same_priority():
    req = request()
    other = deepcopy(record(req, "fraction_unbound"))
    other.update(basis="derived", value=.15)
    req["evidence"].append(other)
    assert "fraction_unbound:equal_priority_conflict" in decide(req, fallback())["reasons"]


def test_identical_duplicates_can_be_resolved_but_fixed_template_rejects_them():
    req = request()
    req["evidence"].append(deepcopy(record(req, "fraction_unbound")))
    assert decide(req, fallback())["status"] == "ready"
    assert decide(req, fallback(), method="fixed_template")["status"] == "blocked"


def test_nonhuman_fu_is_quarantined_then_explicit_rf_can_replace_it():
    req = request()
    record(req, "fraction_unbound")["context"]["species"] = "rat"
    result = decide(req, fallback())
    assert result["status"] == "ready"
    assert selected(result, "fraction_unbound")["basis"] == "predicted"
    assert any(a.get("reason") == "not_human_evidence" for a in result["actions"])
    assert decide(req, [])["status"] == "blocked"
    assert decide(req, fallback(), method="fixed_template")["status"] == "blocked"


@pytest.mark.parametrize("context_change,reason", [({"route": "oral"}, "not_iv_clearance"),
                ({"matrix": "blood"}, "not_plasma_evidence"),
                ({"clearance_type": "CL/F"}, "not_systemic_plasma_clearance"),
                ({"analyte": "total_radioactivity"}, "not_parent_clearance")])
def test_clearance_semantics_are_checked_before_use(context_change, reason):
    req = request()
    record(req, "total_clearance")["context"].update(context_change)
    result = decide(req, fallback())
    assert result["status"] == "ready"
    assert selected(result, "total_clearance")["basis"] == "predicted"
    assert any(a.get("reason") == reason for a in result["actions"])


@pytest.mark.parametrize("value", [0., 1.1, float("nan"), float("inf"), True])
def test_bad_fu_blocks_instead_of_being_silently_replaced(value):
    req = request()
    record(req, "fraction_unbound")["value"] = value
    assert decide(req, fallback())["status"] == "blocked"


@pytest.mark.parametrize("field,value,reason", [("unit", "L/h", "noncanonical_unit"),
                                            ("compound_id", "synthetic:2", "compound_mismatch")])
def test_identity_and_unit_errors_cannot_be_hidden_by_fallback(field, value, reason):
    req = request()
    record(req, "total_clearance")[field] = value
    result = decide(req, fallback())
    assert result["status"] == "blocked"
    assert any(reason in x for x in result["reasons"])


@pytest.mark.parametrize("change", [{"training_overlap": True}, {"training_overlap": "unknown"},
                                  {"applicability": {}}, {"name": "unverified_predictor"}])
def test_unqualified_prediction_is_not_a_valid_fallback(change):
    req, backup = request(), fallback()
    req["evidence"] = [e for e in req["evidence"] if e["parameter"] != "fraction_unbound"]
    backup[0]["prediction_method"].update(change)
    assert decide(req, backup)["status"] == "blocked"


def test_physchem_prediction_does_not_need_human_pk_metadata():
    req = request()
    record(req, "logp").update(basis="predicted", prediction_method={
        "name": "chemical_descriptor", "version": "1", "training_overlap": "unknown",
        "applicability": "molecular property, not clinical labels"})
    assert decide(req, fallback())["status"] == "ready"


@pytest.mark.parametrize("purpose,reason", [({"mechanism": "ddi"}, "unsupported_mechanism"),
        ({"population_condition": "renal_impairment"}, "unsupported_population"),
        ({"endpoints": ["brain_auc"]}, "unsupported_endpoints"),
        ({"dosing_regimen": "multiple_dose"}, "unsupported_regimen")])
@pytest.mark.parametrize("method", ["evidence_policy", "fixed_template", "rf_fill", "no_conflict_guard"])
def test_all_methods_share_use_case_gate(purpose, reason, method):
    result = decide(request(), fallback(), purpose=purpose, method=method)
    assert result["status"] == "blocked"
    assert any(reason in x for x in result["reasons"])


def test_net_renal_input_is_preserved_and_not_in_shared_clearance_scale():
    req = request()
    record(req, "elimination_scope")["value"] = "hepatic_and_renal"
    renal = evidence("renal_clearance", 1.0, "ml/min/kg", "derived",
                     context={"species": "human", "matrix": "plasma", "route": "intravenous", "analyte": "unchanged_parent"})
    req["evidence"].append(renal)
    result = decide(req, fallback())
    assert result["status"] == "ready"
    assert selected(result, "renal_clearance") == renal
    assert result["calibration"]["protect_net_renal"] is True
    assert result["calibration"]["clearance_scale_scope"] == "nonrenal_only"
    record(req, "renal_clearance")["value"] = 100.
    assert "invalid_clearance_allocation" in decide(req, fallback())["reasons"]


def test_clinical_target_evidence_is_rejected_and_not_read_as_a_feature():
    req = request()
    record(req, "fraction_unbound")["context"]["study_role"] = "target"
    result = decide(req, fallback())
    assert result["status"] == "blocked"
    assert any("held_out_outcome_evidence" in x for x in result["reasons"])


def test_unknown_method_is_a_programming_error():
    with pytest.raises(ValueError, match="Unknown"):
        decide(request(), fallback(), method="oracle")
