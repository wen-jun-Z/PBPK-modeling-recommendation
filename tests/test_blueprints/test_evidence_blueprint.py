"""Evidence assembly contracts; no native solver or same-drug model is loaded."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pbpk_kb.blueprints import evidence_blueprint as eb


@pytest.fixture
def request_data(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "docs").mkdir(parents=True)
    source = root / "docs/evidence.json"
    source.write_text(json.dumps({"kind": "synthetic unit fixture; not clinical evidence"}))
    values = {"molecular_weight": 194.19, "molecular_formula": "C8H10N4O2", "logp": -.1,
              "fraction_unbound": .7, "pka_types": [], "solubility": 20., "solubility_ph": 7.,
              "binding_partner": "Albumin", "elimination_scope": "hepatic_only", "total_clearance": 1.}
    evidence = [{"parameter": key, "compound_id": "test:1", "value": value,
                 "unit": eb.UNITS[key], "basis": "measured",
                 "source": {"path": "docs/evidence.json", "sha256": eb.sha256(source), "location": key},
                 "context": {"species": "human", "matrix": "plasma", "route": "iv"}}
                for key, value in values.items()]
    request = {"entry_reason": "model_missing", "compound_id": "test:1", "compound_name": "ToyDrug",
               "human": {"age_years": 30., "weight_kg": 70., "height_cm": 175., "sex": "MALE",
                         "population": "European_ICRP_2002"},
               "dosing": {"route": "intravenous", "dose_mg": 10., "infusion_minutes": 15.},
               "target": {"endpoints": ["auc", "cmax", "plasma_concentration_time"], "primary_endpoint": "auc",
                          "window_start_h": 0., "window_end_h": 12.},
               "evidence": evidence, "assumptions": ["Synthetic unit fixture only"]}
    monkeypatch.setattr(eb, "load_standard_human_model", lambda p: {"checksum": "a" * 64, "model": "generic physiology fixture"})
    _sync_ledger(root, request)
    return root, request


def _sync_ledger(root, request, shape="object"):
    path = root / "docs/curated.json"
    rows = deepcopy(request["evidence"])
    path.write_text(json.dumps({"evidence": rows} if shape == "object" else rows))
    request["evidence_ledger"] = {"path": "docs/curated.json", "sha256": eb.sha256(path), "location": "evidence"}
    return path


def _entry(request, key):
    return next(e for e in request["evidence"] if e["parameter"] == key)


def _replace_source(root, request, content, suffix=".json"):
    path = root / ("docs/replacement" + suffix)
    path.write_text(content)
    for e in request["evidence"]:
        e["source"].update(path=str(path.relative_to(root)), sha256=eb.sha256(path))
    _sync_ledger(root, request)
    return path


def test_assemble_new_standard_snapshot_without_same_drug_model(request_data):
    root, request = request_data
    original = deepcopy(request)
    result = eb.assemble_evidence_blueprint(request, root)
    assert result["status"] == "assembled" and result["manifest"]["used_complete_drug_model"] is False
    snapshot = result["snapshot"]
    assert snapshot["Simulations"][0]["Model"] == "4Comp"
    assert snapshot["Simulations"][0]["OutputSelections"] == ["Organism|PeripheralVenousBlood|ToyDrug|Plasma (Peripheral Venous Blood)"]
    assert snapshot["Compounds"][0]["Processes"][0]["InternalName"] == "LiverClearance"
    assert result["manifest"]["clearance_allocation"]["hepatic_ml_min_kg"] == 1.
    assert request == original


def test_missing_evidence_produces_draft_without_physiology_load(request_data, monkeypatch):
    root, request = request_data
    request["evidence"] = []
    _sync_ledger(root, request)
    monkeypatch.setattr(eb, "load_standard_human_model", lambda p: pytest.fail("Draft must not load physiology"))
    result = eb.assemble_evidence_blueprint(request, root)
    assert result["status"] == "draft_with_gaps" and "fraction_unbound" in result["gaps"]
    assert "snapshot" not in result


def test_missing_second_clearance_for_split_is_a_gap(request_data):
    root, request = request_data
    _entry(request, "elimination_scope")["value"] = "hepatic_and_renal"
    _sync_ledger(root, request)
    result = eb.assemble_evidence_blueprint(request, root)
    assert result["status"] == "draft_with_gaps" and result["gaps"] == ["clearance_inputs_for_hepatic_and_renal"]


def test_total_and_renal_are_subtracted_once(request_data):
    root, request = request_data
    _entry(request, "elimination_scope")["value"] = "hepatic_and_renal"
    renal = deepcopy(_entry(request, "total_clearance"))
    renal.update(parameter="renal_clearance", value=.25)
    request["evidence"].append(renal)
    _sync_ledger(root, request)
    result = eb.assemble_evidence_blueprint(request, root)
    allocation = result["manifest"]["clearance_allocation"]
    assert allocation["hepatic_ml_min_kg"] == .75 and allocation["renal_ml_min_kg"] == .25
    assert len(result["snapshot"]["Compounds"][0]["Processes"]) == 2


@pytest.mark.parametrize("value", [.1, 1.])
def test_total_plus_explicit_hepatic_is_ambiguous(request_data, value):
    root, request = request_data
    extra = deepcopy(_entry(request, "total_clearance"));extra.update(parameter="hepatic_clearance", value=value)
    request["evidence"].append(extra)
    _sync_ledger(root, request)
    with pytest.raises(ValueError, match="overlapping"):
        eb.assemble_evidence_blueprint(request, root)


@pytest.mark.parametrize("kind", ["hash", "identity", "unit", "primary", "oral_cl", "negative_clearance", "renal_above_total", "bool_value"])
def test_invalid_evidence_or_request_rejected(request_data, kind):
    root, request = request_data
    if kind == "hash": _entry(request, "logp")["source"]["sha256"] = "0" * 64
    elif kind == "identity": _entry(request, "logp")["compound_id"] = "different:1"
    elif kind == "unit": _entry(request, "total_clearance")["unit"] = "l/h"
    elif kind == "primary": request["target"]["primary_endpoint"] = "half_life"
    elif kind == "oral_cl": _entry(request, "total_clearance")["context"]["route"] = "oral"
    elif kind == "negative_clearance": _entry(request, "total_clearance")["value"] = -1.
    elif kind == "bool_value": _entry(request, "fraction_unbound")["value"] = True
    elif kind == "renal_above_total":
        _entry(request, "elimination_scope")["value"] = "hepatic_and_renal"
        extra = deepcopy(_entry(request, "total_clearance"));extra.update(parameter="renal_clearance", value=2.)
        request["evidence"].append(extra)
    _sync_ledger(root, request)
    with pytest.raises(ValueError): eb.assemble_evidence_blueprint(request, root)


def test_project_escape_via_path_or_symlink_rejected(request_data):
    root, request = request_data
    outside = root.parent / "external.json";outside.write_text("{}")
    for mode in ["../external.json", "docs/symlink.json"]:
        if "symlink" in mode: (root / mode).symlink_to(outside)
        changed = deepcopy(request)
        for e in changed["evidence"]:e["source"].update(path=mode, sha256=eb.sha256(outside))
        _sync_ledger(root, changed)
        with pytest.raises(ValueError): eb.assemble_evidence_blueprint(changed, root)


@pytest.mark.parametrize("content,suffix", [
    ('{"Version":80,"Compounds":[]}', '.json'),
    ('[{"snapshot":{"Version":80,"Simulations":[]}}]', '.json'),
    ('{"wrapped":{"Individuals":[],"Simulations":[],"Compounds":[]}}', '.json'),
    ('<SimulationTransfer><Simulation/></SimulationTransfer>', '.xml'),
    ('<Simulation name="native"/>', '.xml'),
])
def test_complete_or_nested_native_models_are_not_evidence(request_data, content, suffix):
    root, request = request_data
    _replace_source(root, request, content, suffix)
    with pytest.raises(ValueError): eb.assemble_evidence_blueprint(request, root)


def test_forbidden_hash_also_blocks_a_renamed_copy(request_data):
    root, request = request_data
    request["forbidden_sources"] = [dict(request["evidence"][0]["source"], path="docs/heldout_original.json")]
    with pytest.raises(ValueError, match="forbidden"):
        eb.assemble_evidence_blueprint(request, root)


def test_mutated_prevalidated_request_is_revalidated(request_data):
    root, request = request_data
    validated = eb.EvidenceBlueprintRequest.model_validate(request)
    validated.evidence[0].compound_id = "wrong"
    with pytest.raises(ValueError, match="identity"):
        eb.assemble_evidence_blueprint(validated, root)


def test_unproven_prediction_requires_method_metadata(request_data):
    root, request = request_data
    _entry(request, "fraction_unbound")["basis"] = "predicted"
    with pytest.raises(ValueError, match="Predicted"):
        eb.assemble_evidence_blueprint(request, root)


def test_oral_requires_defined_native_pint_not_caco2(request_data):
    root, request = request_data
    request["dosing"].update(route="oral", infusion_minutes=0., formulation="solution", water_ml_per_kg=3.)
    out = eb.assemble_evidence_blueprint(request, root)
    assert out["status"] == "draft_with_gaps" and "intestinal_permeability" in out["gaps"]
    pint = deepcopy(_entry(request, "fraction_unbound"));pint.update(parameter="intestinal_permeability", value=1e-5, unit="cm/s")
    pint["context"]["definition"] = "caco2_apparent_permeability"
    request["evidence"].append(pint)
    _sync_ledger(root, request)
    with pytest.raises(ValueError, match="Caco2/MDCK"):
        eb.assemble_evidence_blueprint(request, root)
    pint["context"]["definition"] = "pksim_specific_intestinal_permeability"
    _sync_ledger(root, request)
    assert eb.assemble_evidence_blueprint(request, root)["snapshot"]["Formulations"][0]["FormulationType"] == "Formulation_Dissolved"


def test_draft_execute_does_not_call_solver(request_data, monkeypatch):
    from pbpk_kb.blueprints import design_runtime
    root, request = request_data
    request["evidence"] = []
    _sync_ledger(root, request)
    monkeypatch.setattr(design_runtime, "execute_snapshot", lambda **kw: pytest.fail("Draft must not execute"))
    result = eb.run_evidence_blueprint(request=request, output_dir=root / "run", project_root=root, execute=True)
    assert result["status"] == "draft_with_gaps" and result["native_executed"] is False
    assert not (root / "run/snapshot.json").exists()


def test_curated_ledger_array_shape_and_order_are_supported(request_data):
    root, request = request_data
    request["evidence"].reverse()
    _sync_ledger(root, request, shape="array")
    request["evidence"].reverse()
    result = eb.assemble_evidence_blueprint(request, root)
    assert result["status"] == "assembled"
    assert result["manifest"]["values_match_curated_evidence_ledger"] is True
    assert result["manifest"]["source_value_extraction"] == "curated_not_automatically_verified"


def test_value_cannot_change_while_ledger_and_raw_hashes_stay_same(request_data):
    root, request = request_data
    _entry(request, "total_clearance")["value"] *= 2
    with pytest.raises(ValueError, match="checksum-bound curated evidence ledger"):
        eb.assemble_evidence_blueprint(request, root)


@pytest.mark.parametrize("mutation", ["missing", "bad_hash", "wrong_array", "oversized", "source_sha"])
def test_invalid_curated_ledger_rejected(request_data, mutation):
    root, request = request_data
    ledger = root / request["evidence_ledger"]["path"]
    if mutation == "missing":
        del request["evidence_ledger"]
    elif mutation == "bad_hash":
        request["evidence_ledger"]["sha256"] = "0" * 64
    elif mutation == "source_sha":
        _entry(request, "logp")["source"]["sha256"] = "0" * 64
        _sync_ledger(root, request)
    else:
        ledger.write_text(json.dumps({"evidence": {}}) if mutation == "wrong_array" else " " * (2 * 1024**2 + 1))
        request["evidence_ledger"]["sha256"] = eb.sha256(ledger)
    with pytest.raises(ValueError):
        eb.assemble_evidence_blueprint(request, root)


@pytest.mark.parametrize("value", [True, "1.0", float("nan"), float("inf"), -1.])
def test_numeric_clearance_is_strict_even_in_bound_ledger(request_data, value):
    root, request = request_data
    _entry(request, "total_clearance")["value"] = value
    _sync_ledger(root, request)
    with pytest.raises(ValueError):
        eb.assemble_evidence_blueprint(request, root)


def test_oral_cl_over_f_cannot_be_used_as_iv_clearance(request_data):
    root, request = request_data
    _entry(request, "total_clearance")["context"].update(route="oral", definition="CL/F")
    _sync_ledger(root, request)
    with pytest.raises(ValueError, match="human IV plasma evidence"):
        eb.assemble_evidence_blueprint(request, root)


@pytest.mark.parametrize("native_payload", [
    {"snapshot": {"Version": 80}},
    {"wrapped": {"Simulations": [], "Compounds": []}},
    {"other": [{"Individuals": []}]},
])
def test_ledger_itself_cannot_carry_complete_model_payload(request_data, native_payload):
    root, request = request_data
    ledger = root / request["evidence_ledger"]["path"]
    ledger.write_text(json.dumps({"evidence": request["evidence"], **native_payload}))
    request["evidence_ledger"]["sha256"] = eb.sha256(ledger)
    with pytest.raises(ValueError, match="complete/native model"):
        eb.assemble_evidence_blueprint(request, root)
