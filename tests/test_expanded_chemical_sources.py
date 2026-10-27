"""Independent chemical provenance, prohibited inputs, and unsupported sites."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("expanded_chemical_sources", ROOT / "scripts/prepare_expanded_chemical_sources.py")
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


@pytest.fixture(scope="module")
def artifact():
    return c.make_artifact()


def test_twelve_clinical_candidates_retained_with_explicit_blocks(artifact):
    assert artifact["counts"] == {"chemical_reviewed_compounds": 12, "chemical_ready": 10,
                                   "chemical_blocked": 2, "with_independent_fu": 5}
    assert set(artifact["compounds"]) == set(c.DRUGS)
    for drug in ("Amikacin", "Vancomycin"):
        row = artifact["compounds"][drug]
        assert row["construction_status"] == "blocked" and row["missing_reasons"]
        assert not any(e["parameter"] == "pka_types" for e in row["evidence"])
        assert len(row["excluded_evidence"][0]["reported_pka_values"]) > 3


def test_only_three_curated_sources_are_read_not_clinical_or_models(monkeypatch):
    seen = []
    original = Path.read_text
    def observe(path, *args, **kwargs):
        seen.append(path.name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", observe)
    out = c.make_artifact()
    assert seen == ["pubchem_properties.json", "pubchem_annotations.json", "chemical_literature_sources.json"]
    assert out["target_observations_read"] is False and out["osp_compound_parameters_used"] is False
    assert out["network_requests_during_regeneration"] == 0


def test_derived_fu_never_midpoints_bounds_or_relabels_montel_blood(artifact):
    rows = artifact["compounds"]
    assert {d: r["independent_evidence"][0]["value"] for d, r in rows.items() if r["independent_evidence"]} == {
        "Clarithromycin": .3, "Oxycodone": .55, "Paroxetine": .05, "Rifampicin": .2, "Rosuvastatin": .12}
    for drug in ("Amikacin", "Mexiletine", "Ethinylestradiol", "Montelukast", "Pitavastatin", "Probenecid", "Vancomycin"):
        assert not rows[drug]["independent_evidence"]
    excluded = rows["Montelukast"]["excluded_evidence"]
    assert any(e.get("value") == .0018 and "blood" in e["reason"] for e in excluded)
    for row in rows.values():
        for evidence in row["independent_evidence"]:
            assert evidence["context"]["underlying_study_overlap"] == "unknown"
            assert evidence["basis"] == "derived"


def test_properties_have_correct_prediction_roles_and_bound_source_paths(artifact):
    from pbpk_kb.blueprints.evidence_blueprint import Evidence
    for compound in artifact["compounds"].values():
        for row in compound["evidence"] + compound["independent_evidence"]:
            Evidence.model_validate(row)
            source = row["source"]
            path = c.inside(ROOT, source["path"])
            assert c.sha(path) == source["sha256"]
            c.pointer(json.loads(path.read_text()), source["location"])
            assert row["parameter"] not in {"total_clearance", "renal_clearance", "hepatic_clearance", "vss", "terminal_half_life", "mrt"}
            if row["parameter"] == "logp":
                assert row["basis"] == "predicted"
            if row["basis"] == "predicted":
                method = row["prediction_method"]
                assert all(key in method for key in ("name", "version", "training_overlap", "applicability"))
                assert method["training_overlap"] == "unknown"
    ee = next(x for x in artifact["compounds"]["Ethinylestradiol"]["evidence"] if x["parameter"] == "pka_types")
    assert ee["basis"] == "predicted"
    assert ee["value"] == [{"Type": "Acid", "Pka": 10.33}, {"Type": "Base", "Pka": -1.7}]


def test_changed_binding_text_cannot_turn_interval_into_point(monkeypatch):
    original = c.json.loads
    def changed(text, *args, **kwargs):
        value = original(text, *args, **kwargs)
        if value.get("version") == "bounded-pubchem-annotation-retrieval-1":
            info = c.pointer(value["records"]["Clarithromycin|Protein Binding"]["response"], c.BINDING_PATH)
            info["Value"]["StringWithMarkup"][0]["String"] = "60-80% protein bound"
        return value
    monkeypatch.setattr(c.json, "loads", changed)
    with pytest.raises(ValueError, match="Reviewed protein binding statement changed"):
        c.make_artifact()


def test_outside_project_paths_rejected():
    with pytest.raises(ValueError, match="stay in the project"):
        c.inside(ROOT, "../outside.json")


def test_prediction_metadata_passes_complete_native_request_schema(artifact):
    from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest
    for drug, compound in artifact["compounds"].items():
        req = {"entry_reason": "model_missing", "compound_id": compound["compound_id"], "compound_name": drug,
               "human": {"age_years": 30., "weight_kg": 70., "height_cm": 175., "sex": "MALE", "population": "European"},
               "dosing": {"route": "intravenous", "dose_mg": 1., "start_h": 0., "infusion_minutes": 1.},
               "target": {"endpoints": ["plasma_concentration_time"], "primary_endpoint": "plasma_concentration_time",
                          "window_start_h": 0., "window_end_h": 24.},
               "evidence_ledger": compound["evidence"][0]["source"], "evidence": compound["evidence"]}
        EvidenceBlueprintRequest.model_validate(req)
