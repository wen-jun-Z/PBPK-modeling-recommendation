"""Clinical execution inventory preserves scope, source identity and sealed outcomes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/build_recommendation_execution_cohort.py"
spec = importlib.util.spec_from_file_location("execution_cohort", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def put(root, name, value, jsonl=False):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r) for r in value) + "\n" if jsonl else json.dumps(value)
    path.write_text(text)
    return {"path": name, "sha256": m.digest(path.read_bytes())}


def test_exact_aliases_do_not_strip_salts_or_match_substrings():
    assert m.normalize_compound(" Acetaminophen ") == "paracetamol"
    assert m.normalize_compound("Ethinyl-estradiol") == "ethinylestradiol"
    assert m.normalize_compound("metformin hydrochloride") != "metformin"
    assert m.normalize_compound("hydroxy itraconazole") != "itraconazole"


def test_outcome_barrier_allows_dose_but_not_measured_arrays():
    m.outcome_free({"dose": {"value": 25}, "sampling_times_h": [0, 1], "observed_age_range_years": [18, 50]})
    with pytest.raises(ValueError, match="Outcome-bearing"):
        m.outcome_free({"nested": [{"observed_values": [1, 2]}]})


def test_read_refuses_sealed_and_symlink_escape(tmp_path):
    inv = m.Inventory(tmp_path)
    put(tmp_path, "sealed/data.json", {"observed_values": [1]})
    with pytest.raises(ValueError, match="Refusing"):
        inv.read("sealed/data.json")
    (tmp_path / "alias.json").symlink_to(tmp_path / "sealed/data.json")
    with pytest.raises(ValueError, match="Refusing"):
        inv.read("alias.json")
    (tmp_path / "escape").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError, match="inside"):
        inv.path("escape/outside")


def test_normalize_minutes_preserves_negative_predose_and_salt_basis():
    d = m.normalized_design({"protocol": {"dose": 250, "dose_unit": "mg", "dose_basis": "hydrochloride",
        "active_moiety_dose_mg": 195, "route": "oral", "sampling_times_min": [-30, 0, 80],
        "administrations": [{"time_min": 0, "dose": 250}]},
        "evaluation_target": {"endpoints": ["Cmax"]}})
    assert d["dose"]["value"] == 250 and d["dose"]["active_moiety_dose_mg"] == 195
    assert d["sampling"]["times_h"] == [-.5, 0, 80 / 60]
    assert d["regimen"]["dose_times_h"] == [0]
    assert d["regimen"]["observation_origin_h"] == 0
    assert d["endpoint_contracts"] == [{"endpoint": "Cmax", "aggregation": None}]


def test_normalize_infusion_and_preserve_rounded_sampling():
    d = m.normalized_design({"dose": {"value": 4, "unit": "mg/kg", "route": "IV",
        "administrations_h": [0], "infusion_duration_min": 80},
        "sampling": {"times_h_from_observation_origin": [0, 1.33, 1.42], "observation_origin_h_after_first_dose": 0},
        "endpoint_contracts": []})
    assert d["route"] == "intravenous"
    assert d["regimen"]["infusion_duration_h"] == 80 / 60
    assert d["sampling"]["times_h"][1] == 1.33


def test_repeated_dose_sampling_origin_not_reset_to_zero():
    d = m.normalized_design({"dose": {"value": 200, "unit": "mg", "administrations_h": [0, 12, 24]},
        "observation_origin": "last dose,24h after first dose", "nominal_sampling_hours_from_observation_origin": [0, 2]})
    assert d["regimen"]["observation_origin_h"] == 24
    missing = m.normalized_design({"dose": {"value": 5, "unit": "mg", "frequency": "twice daily"}})
    assert missing["regimen"]["dose_times_h"] is None
    assert missing["regimen"]["observation_origin_h"] is None


def test_exact_audit_not_inherited_by_other_candidate_or_path():
    scenario = {"scenario_id": "s", "case_id": "c", "compound": "metformin",
        "source_independence": {"bound_snapshot_references": [{"path": "model/a", "sha256": "aaa"}],
                                "audit_references": [{"path": "audit.json"}]}}
    candidate = {"candidate_id": "1", "compounds": ["metformin"], "family_id": "f"}
    asset = {"path": "model/a", "actual_sha256": "aaa", "sha256_matches": True, "state": "verified"}
    assert m.bind_candidate(scenario, candidate, asset)["independence_status"] == "previously_audited_pending_recheck"
    other = m.bind_candidate(scenario, candidate, {**asset, "path": "model/b"})
    assert other["independence_status"] == "not_audited_for_this_candidate"
    assert not other["independence_verified"]
    wrong = m.bind_candidate(scenario, {**candidate, "compounds": ["phenformin"]}, asset)
    assert wrong["execution_status"] == "excluded_parent_identity_mismatch"
    assert wrong["scientific_applicability"] == "unknown"


def test_asset_hash_failure_is_engineering_not_scientific_negative():
    scenario = {"scenario_id": "s", "case_id": "c", "compound": "a",
        "source_independence": {"bound_snapshot_references": [], "audit_references": []}}
    row = m.bind_candidate(scenario, {"candidate_id": "1", "compounds": ["a"]},
        {"state": "source_file_missing", "actual_sha256": None, "path": "a", "sha256_matches": False})
    assert row["execution_status"] == "blocked_source_integrity"
    assert row["scientific_applicability"] == "unknown"
    assert not row["execution_failure_is_scientific_nonapplicability"]


def test_frozen_write_does_not_overwrite(tmp_path):
    p = tmp_path / "cohort.json"
    m.frozen_write(p, b"one")
    m.frozen_write(p, b"one")
    with pytest.raises(ValueError, match="overwrite"):
        m.frozen_write(p, b"two")
    assert p.read_bytes() == b"one"


def miniature(tmp_path):
    asset = tmp_path / "models/a.json"
    asset.parent.mkdir()
    asset.write_text('{"ObservedData": ["not parsed by builder"]}')
    candidates = [{"candidate_id": "a", "family_id": "a", "compounds": ["a"],
                   "provenance": {"source_path": "models/a.json", "source_sha256": m.digest(asset.read_bytes())}}]
    put(tmp_path, str(m.CORE / "candidates.jsonl"), candidates, True)
    put(tmp_path, str(m.CORE / "queries.jsonl"), [{"query_id": "qa", "compounds": ["a"]}], True)
    protocol = {"case_id": "case_b", "compound": "b", "study_id": "study1",
        "dose": {"value": 5, "unit": "mg", "route": "PO", "administrations_h": [0]},
        "sampling": {"times_h_from_observation_origin": [0, 1]}, "endpoint_contracts": [{"endpoint": "Cmax"}],
        "sealed_observation_reference": {"path": "sealed/b.json", "sha256": "never_open_this"}}
    put(tmp_path, "batch/protocols/case_b.json", protocol)
    put(tmp_path, "handoff.json", {"drug_names": ["b"], "clinical_protocol_metadata_total": 1,
                                   "qualifying_protocols_directory": "batch/protocols"})
    return protocol


def test_complete_build_keeps_zero_candidate_case_without_opening_sealed(tmp_path):
    miniature(tmp_path)
    report = m.build(tmp_path, "out", latest="handoff.json", include_caffeine=False)
    assert report["scope"]["all_scenarios"] == 1
    assert report["scope"]["identity_matching_pairs"] == 0
    assert report["scope"]["sealed_outcome_files_read"] == 0
    scenario = json.loads((tmp_path / "out/scenarios.jsonl").read_text())
    assert "no_exact_parent_in_frozen_catalog" in scenario["blockers"]
    assert scenario["sealed_references"][0]["sha256"] == "never_open_this"
    assert not (tmp_path / "sealed/b.json").exists()
    m.build(tmp_path, "out", latest="handoff.json", include_caffeine=False)


def test_incomplete_manifest_fails_instead_of_silently_subsetting(tmp_path):
    miniature(tmp_path)
    put(tmp_path, "handoff.json", {"drug_names": ["b", "missing"], "clinical_protocol_metadata_total": 2,
                                   "qualifying_protocols_directory": "batch/protocols"})
    with pytest.raises(ValueError, match="no silent partial"):
        m.build(tmp_path, "out", latest="handoff.json", include_caffeine=False)


def test_hash_mismatch_recorded_and_case_retained(tmp_path):
    miniature(tmp_path)
    inv = m.Inventory(tmp_path)
    d = inv.read({"path": "handoff.json", "sha256": "wrong"})
    assert d["drug_names"] == ["b"]
    assert len(inv.hash_mismatches) == 1
    assert inv.hash_mismatches[0]["declared_sha256"] == "wrong"


def test_same_publication_different_declared_names_has_shared_cluster():
    a = {"identifiers": {"doi": "10.1007/S13318-025-00961-4", "pmid": "40839056"},
         "publication_cluster_id": "pmid:40839056"}
    b = {"source_identifiers": {"doi": "https://doi.org/10.1007/s13318-025-00961-4", "pmcid": "PMC12394267"},
         "publication_cluster_id": "PMC12394267"}
    assert m.publication_cluster(a) == m.publication_cluster(b)
