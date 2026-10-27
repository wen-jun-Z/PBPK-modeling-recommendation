from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pbpk_kb.processing import structured_evidence_validation as validation
from pbpk_kb.processing.auto_quality_stage import deterministic_repair, evaluate_evidence


def write_source(path: Path, payload: dict, *, source: str, url: str, compound: str = "caffeine") -> dict:
    path.write_text(json.dumps(payload), encoding="utf-8")
    metadata = {"asset_id": "asset:" + path.name, "checksum": hashlib.sha256(path.read_bytes()).hexdigest(),
                "byte_count": path.stat().st_size, "storage_path": str(path), "source_id": source,
                "compound": compound, "status_code": 200, "url": url}
    Path(str(path) + ".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def source_row(tmp_path: Path, monkeypatch, source="pubchem", field="MolecularWeight") -> dict:
    monkeypatch.setattr(validation, "PROJECT_ROOT", tmp_path)
    values = {"MolecularWeight": "194.19", "TPSA": 58.4, "XLogP": -0.1,
              "full_mwt": "194.19", "psa": "58.44", "alogp": "-0.07"}
    path = tmp_path / ("properties_by_name.json" if source == "pubchem" else "molecules.json")
    if source == "pubchem":
        payload = {"PropertyTable": {"Properties": [{"CID": 2519, field: values[field]}]}}
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/caffeine/property/{field}/JSON"
        pointer = f"$.PropertyTable.Properties[0].{field}"
        mapping = {"pubchem_cid": 2519, "json_field": field}
        write_source(tmp_path / "synonyms_by_name_abc.json", {"InformationList": {"Information": [
            {"CID": 2519, "Synonym": ["CAFFEINE", "1,3,7-trimethylxanthine"]}]}}, source="pubchem",
            url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/caffeine/synonyms/JSON")
    else:
        payload = {"molecules": [{"molecule_chembl_id": "CHEMBL113", "pref_name": "CAFFEINE",
                                 "molecule_properties": {field: values[field]}, "molecule_synonyms": []}]}
        url = "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q=caffeine&limit=25&offset=0"
        pointer = f"$.molecules[0].molecule_properties.{field}"
        mapping = {"molecule_chembl_id": "CHEMBL113", "chembl_identity_gate": "exact", "json_field": field}
    sidecar = write_source(path, payload, source=source, url=url)
    text = str(values[field])
    return {"evidence_id": "e1", "compound_id": "c1", "compound_name": "caffeine",
            "attributed_compound_name": "caffeine", "compound_role": "target_compound",
            "evidence_type": "physicochemical", "source_id": source, "raw_asset_id": sidecar["asset_id"],
            "storage_path": str(path), "source_url": url, "checksum": sidecar["checksum"],
            "source_location": f"{source}:{path.name}#json_path={pointer}", "source_text": text,
            "source_text_hash": hashlib.sha256(text.encode()).hexdigest(), "parameter": field,
            "value": float(values[field]), "value_text": text, "unit": None,
            "extraction_method": "source_structured_field", "extractor_version": "1",
            "extractor_name": "pubchem_property_mapper" if source == "pubchem" else "chembl_exact_molecule_mapper",
            "metadata_json": mapping}


@pytest.mark.parametrize("source,field,unit,dimension", [
    ("pubchem", "MolecularWeight", "g/mol", "molar_mass"),
    ("pubchem", "TPSA", "angstrom^2", "area"),
    ("pubchem", "XLogP", "dimensionless", "dimensionless"),
    ("chembl", "full_mwt", "g/mol", "molar_mass"),
    ("chembl", "psa", "angstrom^2", "area"),
    ("chembl", "alogp", "dimensionless", "dimensionless"),
])
def test_verified_field_survives_quality_gate_without_modifying_quote_or_value(tmp_path, monkeypatch, source, field, unit, dimension):
    row = source_row(tmp_path, monkeypatch, source, field)
    before = json.dumps(row, sort_keys=True)
    repaired, repairs = deterministic_repair(row)
    decision = evaluate_evidence(repaired, repairs)
    assert decision["quality_status"] == "auto_validated"
    assert repaired["unit"] == repaired["canonical_unit"] == unit
    assert repaired["unit_dimension"] == dimension
    assert repaired["source_text"] == row["source_text"]
    assert repaired["value"] == row["value"] == repaired["canonical_value"]
    assert repaired["metadata_json"]["structured_source_validation"]["accepted"]
    assert decision["structured_source_validation"]["source"]["sha256"] == row["checksum"]
    assert json.dumps(row, sort_keys=True) == before


@pytest.mark.parametrize("changes,reason", [
    ({"value": 1}, "property_numeric_value_mismatch"),
    ({"canonical_value": 1}, "property_numeric_value_mismatch"),
    ({"value": True}, "non_numeric_property"),
    ({"unit": "mg/L"}, "property_unit_conflicts_with_schema"),
    ({"canonical_unit": "kg/mol"}, "property_unit_conflicts_with_schema"),
    ({"value_low": 1, "value_high": 2}, "scalar_property_has_range"),
    ({"compound_name": "theophylline"}, "candidate_compound_name_mismatch"),
    ({"attributed_compound_name": "theophylline"}, "candidate_compound_name_mismatch"),
    ({"raw_asset_id": "other"}, "candidate_source_identity_mismatch"),
    ({"checksum": "0" * 64}, "candidate_source_identity_mismatch"),
    ({"extractor_name": "unknown"}, "unrecognized_property_mapper"),
    ({"source_location": "pubchem:properties_by_name.json#json_path=$.PropertyTable.Properties[-1].MolecularWeight"}, "unsupported_property_json_path"),
])
def test_mismatched_candidate_is_not_rescued_by_cached_verdict(tmp_path, monkeypatch, changes, reason):
    row = source_row(tmp_path, monkeypatch)
    row["metadata_json"]["structured_source_validation"] = {"accepted": True}
    row.update(changes)
    repaired, repairs = deterministic_repair(row)
    decision = evaluate_evidence(repaired, repairs)
    assert decision["quality_status"] == "quarantined"
    assert "structured_property_" + reason in decision["fatal_blockers"]


def test_source_modified_after_repair_is_rechecked(tmp_path, monkeypatch):
    row = source_row(tmp_path, monkeypatch)
    repaired, repairs = deterministic_repair(row)
    Path(row["storage_path"]).write_text("{}")
    assert evaluate_evidence(repaired, repairs)["quality_status"] == "quarantined"


@pytest.mark.parametrize("mutation", ["missing", "wrong_cid", "wrong_name", "checksum", "wrong_host"])
def test_pubchem_requires_independent_cid_name_binding(tmp_path, monkeypatch, mutation):
    row = source_row(tmp_path, monkeypatch)
    p = tmp_path / "synonyms_by_name_abc.json"
    if mutation == "missing":
        p.unlink()
    elif mutation == "checksum":
        p.write_text("{}")
    else:
        cid = 999 if mutation == "wrong_cid" else 2519
        name = "theophylline" if mutation == "wrong_name" else "caffeine"
        host = "invalid.example" if mutation == "wrong_host" else "pubchem.ncbi.nlm.nih.gov"
        write_source(p, {"InformationList": {"Information": [{"CID": cid, "Synonym": [name]}]}},
                     source="pubchem", url=f"https://{host}/rest/pug/compound/name/caffeine/synonyms/JSON")
    result = validation.validate_structured_evidence(row)
    assert not result["accepted"]
    assert result["error"].endswith("pubchem_cid_name_not_proven")


def test_same_value_from_other_chembl_molecule_is_not_supported(tmp_path, monkeypatch):
    row = source_row(tmp_path, monkeypatch, "chembl", "full_mwt")
    p = Path(row["storage_path"])
    payload = json.loads(p.read_text())
    payload["molecules"][0]["pref_name"] = "THEOPHYLLINE"
    metadata = write_source(p, payload, source="chembl", url=row["source_url"])
    row["checksum"] = metadata["checksum"]
    assert validation.validate_structured_evidence(row)["error"].endswith("chembl_target_name_not_proven")


@pytest.mark.parametrize("source,field,id_field,wrong_id", [
    ("pubchem", "MolecularWeight", "pubchem_cid", 999),
    ("chembl", "full_mwt", "molecule_chembl_id", "CHEMBL999"),
])
def test_candidate_molecular_identifier_must_match_source(tmp_path, monkeypatch, source, field, id_field, wrong_id):
    row = source_row(tmp_path, monkeypatch, source, field)
    row["metadata_json"][id_field] = wrong_id
    assert not validation.validate_structured_evidence(row)["accepted"]


def test_ambiguous_name_response_is_not_rescued(tmp_path, monkeypatch):
    row = source_row(tmp_path, monkeypatch, "chembl", "full_mwt")
    p = Path(row["storage_path"])
    payload = json.loads(p.read_text())
    payload["molecules"].append({**payload["molecules"][0], "molecule_chembl_id": "CHEMBL999"})
    metadata = write_source(p, payload, source="chembl", url=row["source_url"])
    row["checksum"] = metadata["checksum"]
    assert not validation.validate_structured_evidence(row)["accepted"]


@pytest.mark.parametrize("same_structure", [True, False])
def test_shared_alias_requires_full_cross_source_inchi_key(tmp_path, monkeypatch, same_structure):
    chembl = tmp_path / "chembl"
    chembl.mkdir()
    row = source_row(chembl, monkeypatch, "chembl", "full_mwt")
    monkeypatch.setattr(validation, "PROJECT_ROOT", tmp_path)
    p = Path(row["storage_path"])
    payload = json.loads(p.read_text())
    key = "AAAAAAAAAAAAAA-BBBBBBBBBB-C"
    other_key = "AAAAAAAAAAAAAA-DDDDDDDDDD-C"
    payload["molecules"][0]["molecule_structures"] = {"standard_inchi_key": key}
    payload["molecules"].append({**payload["molecules"][0], "molecule_chembl_id": "CHEMBL999",
                                 "molecule_structures": {"standard_inchi_key": other_key}})
    metadata = write_source(p, payload, source="chembl", url=row["source_url"])
    row["checksum"] = metadata["checksum"]
    pubchem = tmp_path / "pubchem"
    pubchem.mkdir()
    write_source(pubchem / "properties_by_name_cross.json", {"PropertyTable": {"Properties": [
        {"CID": 2519, "InChIKey": key if same_structure else other_key}]}}, source="pubchem",
        url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/caffeine/property/InChIKey/JSON")
    write_source(pubchem / "synonyms_by_name_cross.json", {"InformationList": {"Information": [
        {"CID": 2519, "Synonym": ["caffeine"]}]}}, source="pubchem",
        url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/caffeine/synonyms/JSON")
    result = validation.validate_structured_evidence(row)
    assert result["accepted"] is same_structure
    if same_structure:
        assert result["audit"]["cross_source_identity"]["inchi_key"] == key
    else:
        assert result["error"].endswith("chembl_cross_source_structure_mismatch")


def test_chembl_salt_does_not_count_as_parent_identity(tmp_path, monkeypatch):
    row = source_row(tmp_path, monkeypatch, "chembl", "full_mwt")
    p = Path(row["storage_path"])
    payload = json.loads(p.read_text())
    payload["molecules"][0]["pref_name"] = "CAFFEINE CITRATE"
    metadata = write_source(p, payload, source="chembl", url=row["source_url"])
    row["checksum"] = metadata["checksum"]
    assert not validation.validate_structured_evidence(row)["accepted"]


def test_source_outside_project_is_rejected_before_reading(tmp_path, monkeypatch):
    row = source_row(tmp_path, monkeypatch)
    row["storage_path"] = str(tmp_path.parent / "not-in-project.json")
    assert validation.validate_structured_evidence(row)["error"].endswith("source_outside_project")


def test_plain_sentence_cannot_use_structured_metadata_to_bypass_numeric_guard(tmp_path, monkeypatch):
    row = source_row(tmp_path, monkeypatch)
    text = "Caffeine Cmax was 10 ng/mL and Tmax was 2 hours."
    row.update({"extraction_method": "llm_assisted", "source_id": "pubmed", "evidence_type": "clinical_pk",
                "parameter": "Cmax", "value": 2, "value_text": "2", "unit": "ng/mL",
                "source_text": text, "source_text_hash": hashlib.sha256(text.encode()).hexdigest()})
    assert validation.validate_structured_evidence(row) is None
    # The existing sentence guard is still invoked; this amendment never treats
    # a detached numerical quote or cached typed-source marker as text support.
    from pbpk_kb.processing import run_llm_extraction
    calls = []
    original = run_llm_extraction._numeric_parameter_owner_error
    def guard(candidate, target):
        calls.append(target)
        return original(candidate, target)
    monkeypatch.setattr(run_llm_extraction, "_numeric_parameter_owner_error", guard)
    decision = evaluate_evidence(row, [])
    assert calls == ["caffeine"]
    assert decision["quality_status"] == "quarantined"
