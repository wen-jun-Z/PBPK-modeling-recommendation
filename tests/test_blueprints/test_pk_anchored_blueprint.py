"""Evidence-to-native mapping guards; all native execution is replaced by fixtures."""
from copy import deepcopy
import inspect
import json

import pytest

from pbpk_kb.blueprints import evidence_blueprint as eb
from pbpk_kb.blueprints import pk_anchored_blueprint as pk


@pytest.fixture
def evidence_case(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "inputs").mkdir(parents=True)
    source = root / "inputs/source.json"
    source.write_text('{"purpose":"synthetic test fixture, not clinical evidence"}')
    values = {"molecular_weight": 194.19, "molecular_formula": "C8H10N4O2", "logp": -.1,
              "fraction_unbound": .7, "pka_types": [], "solubility": 20., "solubility_ph": 7.,
              "binding_partner": "Albumin", "elimination_scope": "hepatic_and_renal",
              "total_clearance": 1., "renal_clearance": .25}
    evidence = [{"parameter": k, "compound_id": "test:1", "value": v,
                 "unit": eb.UNITS[k], "basis": "measured",
                 "source": {"path": "inputs/source.json", "sha256": pk.sha256(source), "location": k},
                 "context": {"species": "human", "matrix": "plasma", "route": "iv"}}
                for k, v in values.items()]
    ledger = root / "inputs/ledger.json"
    ledger.write_text(json.dumps({"evidence": evidence}))
    request = {"entry_reason": "model_missing", "compound_id": "test:1", "compound_name": "TestDrug",
               "human": {"age_years": 30., "weight_kg": 70., "height_cm": 175., "sex": "MALE",
                         "population": "European_ICRP_2002"},
               "dosing": {"route": "intravenous", "dose_mg": 10., "infusion_minutes": 15.},
               "target": {"endpoints": ["auc", "cmax"], "primary_endpoint": "auc",
                          "window_start_h": 0., "window_end_h": 12.},
               "partition_method": "PKSimStandard", "evidence": evidence,
               "evidence_ledger": {"path": "inputs/ledger.json", "sha256": pk.sha256(ledger),
                                   "location": "evidence"}}
    monkeypatch.setattr(eb, "load_standard_human_model", lambda _: {"purpose": "generic unit fixture"})
    anchor = {"value": 1.7, "unit": "L/kg", "basis": "measured", "study_id": "test:development",
              "compound_id": "test:1",
              "source": {"path": "inputs/source.json", "sha256": pk.sha256(source), "location": "Vss"},
              "context": {"species": "human", "route": "intravenous", "matrix": "plasma",
                          "definition": "steady_state_volume", "study_role": "development"}}
    return root, request, anchor


def test_mapping_preserves_evidence_physicochemistry_physiology_and_original_grid(evidence_case):
    root, request, _ = evidence_case
    original_request = deepcopy(request)
    blueprint = eb.assemble_evidence_blueprint(request, root)
    original = deepcopy(blueprint)
    mapped = pk.mapped_snapshot(blueprint, .4, .5)
    assert blueprint == original
    assert request == original_request
    assert mapped["Individuals"] == original["snapshot"]["Individuals"]
    before = original["snapshot"]["Compounds"][0]
    after = mapped["Compounds"][0]
    for key in ("PkaTypes", "FractionUnbound", "PlasmaProteinBindingPartner", "CalculationMethods"):
        assert after[key] == before[key]
    for old, new in zip(before["Processes"], after["Processes"]):
        assert old["InternalName"] == new["InternalName"]
        old_values = {p["Name"]: p["Value"] for p in old["Parameters"]}
        new_values = {p["Name"]: p["Value"] for p in new["Parameters"]}
        assert new_values["Plasma clearance"] == old_values["Plasma clearance"] * .5
        assert new_values["Fraction unbound (experiment)"] == old_values["Fraction unbound (experiment)"]
    original_grid = original["snapshot"]["Simulations"][0]["OutputSchema"]
    grid = mapped["Simulations"][0]["OutputSchema"]
    assert grid[:len(original_grid)] == original_grid
    assert next(p["Value"] for p in grid[-1]["Parameters"] if p["Name"] == "End time") == 1344.
    assert "EvidenceLogP" not in json.dumps(mapped["Simulations"])
    assert "PKAnchorEffectiveMembraneAffinity" in json.dumps(mapped["Simulations"])


@pytest.mark.parametrize("affinity,scale", [(2.4001, 1.), (-2.6001, 1.), (.1, .249),
                                         (.1, 4.001), (float("nan"), 1.), (.1, True)])
def test_out_of_range_mapping_is_rejected(evidence_case, affinity, scale):
    root, request, _ = evidence_case
    blueprint = eb.assemble_evidence_blueprint(request, root)
    with pytest.raises(ValueError):
        pk.mapped_snapshot(blueprint, affinity, scale)


def test_absolute_vss_is_not_normalized_by_assumed_weight(evidence_case):
    root, request, anchor = evidence_case
    assert pk._target_vss(anchor, request, root) == pytest.approx(119.)
    anchor.update(value=73.5, unit="L")
    assert pk._target_vss(anchor, request, root) == 73.5
    request["human"]["weight_kg"] = 95.
    assert pk._target_vss(anchor, request, root) == 73.5
    assert pk._target_vss(None, request, root) is None


@pytest.mark.parametrize("change", ["oral", "blood", "mouse", "vz", "predicted", "oral_unit",
                                   "changed_hash", "forbidden_path", "forbidden_hash", "negative",
                                   "wrong_drug", "missing_drug", "comparison_role", "missing_role"])
def test_incompatible_or_comparison_vss_sources_rejected(evidence_case, change):
    root, request, anchor = evidence_case
    if change == "oral": anchor["context"]["route"] = "oral"
    elif change == "blood": anchor["context"]["matrix"] = "blood"
    elif change == "mouse": anchor["context"]["species"] = "mouse"
    elif change == "vz": anchor["context"]["definition"] = "terminal_distribution_volume"
    elif change == "predicted": anchor["basis"] = "predicted"
    elif change == "oral_unit": anchor["unit"] = "L/F"
    elif change == "negative": anchor["value"] = -1.
    elif change == "wrong_drug": anchor["compound_id"] = "test:other_drug"
    elif change == "missing_drug": anchor.pop("compound_id")
    elif change == "comparison_role": anchor["context"]["study_role"] = "comparison"
    elif change == "missing_role": anchor["context"].pop("study_role")
    elif change == "changed_hash": anchor["source"]["sha256"] = "0" * 64
    elif change == "forbidden_path":
        request["forbidden_sources"] = [{**anchor["source"], "sha256": "0" * 64}]
    elif change == "forbidden_hash":
        request["forbidden_sources"] = [{**anchor["source"], "path": "inputs/other_study.json"}]
    with pytest.raises(ValueError):
        pk._target_vss(anchor, request, root)


def test_vss_source_cannot_escape_project(evidence_case):
    root, request, anchor = evidence_case
    outside = root.parent / "external.json"
    outside.write_text("{}")
    anchor["source"].update(path="../external.json", sha256=pk.sha256(outside))
    with pytest.raises(ValueError, match="inside"):
        pk._target_vss(anchor, request, root)


def test_signature_allows_new_dose_and_human_but_rejects_new_evidence(evidence_case):
    _, request, _ = evidence_case
    signature = pk._drug_signature(request)
    changed = deepcopy(request)
    changed["human"]["sex"] = "FEMALE"
    changed["human"]["weight_kg"] = 60.
    changed["dosing"]["dose_mg"] = 25.
    changed["target"]["window_end_h"] = 48.
    assert pk._drug_signature(changed) == signature
    for field in ("compound_id", "compound_name", "partition_method", "evidence", "evidence_ledger"):
        changed = deepcopy(request)
        if field == "evidence": changed[field][0]["value"] += 1.
        elif field == "evidence_ledger": changed[field]["sha256"] = "0" * 64
        else: changed[field] += "Different"
        assert pk._drug_signature(changed) != signature


def test_fit_uses_supplied_cl_units_and_does_not_mutate_ledger(evidence_case, monkeypatch):
    root, request, _ = evidence_case
    original_request = deepcopy(request)
    ledger_hash = pk.sha256(root / request["evidence_ledger"]["path"])
    calls = []

    def simulated(snapshot, req, output, project_root, *, audit):
        calls.append((snapshot, req))
        return {"status": "predicted", "clearance_l_h": 4.2, "vss_l": 70.}

    monkeypatch.setattr(pk, "_run", simulated)
    result = pk.fit_pk_mapping(reference_request=request, vss_anchor=None,
                               output_dir=root / "fit", project_root=root, max_evaluations=1)
    assert result["converged"] is True
    assert result["target_clearance_l_h"] == pytest.approx(4.2)  # 1 mL/min/kg * 70 kg * .06
    assert result["target_vss_l"] is None
    assert len(calls) == 1
    assert result["observations_used"] is False
    assert result["independent_validation"] is False
    assert result["decision"]["lipophilicity"] == -.1
    assert pk.sha256(root / request["evidence_ledger"]["path"]) == ledger_hash
    assert request == original_request
    assert not any("observ" in name or "curve" in name for name in inspect.signature(pk.fit_pk_mapping).parameters)


@pytest.mark.parametrize("mutation", ["unconverged", "wrong_drug", "changed_evidence"])
def test_prediction_rejects_unqualified_mapping_before_native_execution(evidence_case, monkeypatch, mutation):
    root, request, _ = evidence_case
    normalized = eb.EvidenceBlueprintRequest.model_validate(request).model_dump(mode="json", exclude_none=True)
    mapping = {"converged": True, "decision": {"lipophilicity": -.1, "clearance_scale": 1.},
               "drug_signature": pk._drug_signature(normalized), "target_vss_anchor": None}
    if mutation == "unconverged": mapping["converged"] = False
    elif mutation == "wrong_drug": request["compound_name"] = "AnotherDrug"
    elif mutation == "changed_evidence": request["evidence"][0]["value"] += 1.
    path = root / "mapping.json"
    path.write_text(json.dumps(mapping))
    monkeypatch.setattr(pk, "_run", lambda *a, **k: pytest.fail("Unqualified mapping reached native execution"))
    with pytest.raises(ValueError, match="unconverged or belongs"):
        pk.predict_with_pk_mapping(request=request, mapping_path=path, output_dir=root / "prediction", project_root=root)
    assert not (root / "prediction").exists()
