"""Exact publication/source qualification; no outcomes or native execution."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("source_audit", ROOT / "scripts/recommendation_execution_source_audit.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def bundle(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value).encode() if isinstance(value, dict) else value.encode()
        path.write_bytes(payload)
        return {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}
    source = write("models/drug.json", {"compound": "drug", "parameter": 1})
    document = write("models/clinical_sources.md", "Development used Old Study 2000.")
    checked = {"compound": "drug", "target_identifiers": {"doi": "10.1000/new"},
               "clinical_source_disjoint_at_audited_source_level": True,
               "overlaps": [], "repository_source_documents_checked": [document]}
    auditref = write("audits/source.json", checked)
    protocol = {"case_id": "case", "compound": "drug", "identifiers": {"doi": "10.1000/new"}, "source_audit": auditref}
    protocolref = write("protocol.json", protocol)
    scenario = {"scenario_id": "clinical:case", "case_id": "case", "compound": "drug", "catalog_candidate_ids": ["c"],
                "source_protocol": protocolref, "source_independence": {
                    "audit_references": [auditref], "bound_snapshot_references": [source]}}
    audit = {"candidate_id": "c", "scenario_id": "clinical:case", "case_id": "case", "compound": "drug",
             "identity_match": True, "bound_snapshot_exact_match": True,
             "independence_status": "previously_audited_pending_recheck", "exact_bound_references": [source],
             "catalog_provenance": {"source_path": source["path"], "source_sha256": source["sha256"]},
             "source_asset": {"state": "verified", "sha256_matches": True, "path": source["path"], "actual_sha256": source["sha256"]}}
    def update_document(changes):
        checked.update(changes)
        updated = write("audits/source.json", checked)
        protocol["source_audit"] = updated
        scenario["source_protocol"] = write("protocol.json", protocol)
        scenario["source_independence"]["audit_references"] = [updated]
    return tmp_path, scenario, audit, update_document, write


def result(bundle):
    root, scenario, audit, _, _ = bundle
    return m.audit_candidate_source(scenario, audit, project_root=root)


def test_boolean_pass_without_status_retains_only_declared_scope(bundle):
    answer = result(bundle)
    assert answer["status"] == "scoped_source_audit_pass"
    assert answer["clinical_outcome_values_parsed"] is False
    assert answer["scientific_applicability_established"] is False
    assert answer["strict_target_match_established"] is False
    assert len(answer["verified_references"]) == 4


@pytest.mark.parametrize("path", ["models/drug.json", "models/clinical_sources.md", "audits/source.json", "protocol.json"])
def test_model_audit_protocol_or_reviewed_document_drift_blocks(bundle, path):
    (bundle[0] / path).write_text("changed")
    answer = result(bundle)
    assert answer["status"] == "unknown"
    assert "sha256_changed" in answer["reasons"][0]


def test_other_candidate_cannot_inherit_same_drug_audit(bundle):
    bundle[2]["independence_status"] = "not_audited_for_this_candidate"
    assert "no_previously_frozen_exact" in result(bundle)["reasons"][0]


def test_wrong_case_or_target_publication_blocks(bundle):
    bundle[3]({"target_identifiers": {"doi": "10.1000/another-study"}})
    assert "target_publication_mismatch" in result(bundle)["reasons"][0]
    bundle[2]["case_id"] = "another"
    assert "case_or_compound_mismatch" in result(bundle)["reasons"][0]


def test_nonempty_target_match_cannot_be_overruled_by_pass_boolean(bundle):
    bundle[3]({"audit_documents": [{"target_identity_matches": ["same trial"]}]})
    assert "unresolved_target_match" in result(bundle)["reasons"][0]


def test_explicit_negative_cannot_be_overruled_by_legacy_pass(bundle):
    bundle[3]({"status": m.PASS_STATUS, "clinical_source_disjoint_at_audited_source_level": False})
    assert "uncontradicted_scoped_pass" in result(bundle)["reasons"][0]


def test_legacy_protocol_link_and_general_limitations_are_not_blanket_blockers(bundle):
    bundle[3]({"target_identifiers": {}, "status": m.PASS_STATUS,
               "unknowns": ["No participant-level identity guarantee; supplement not locally available."]})
    answer = result(bundle)
    assert answer["status"] == "scoped_source_audit_pass"
    assert answer["audit_results"][0]["target_binding"] == "frozen_protocol_to_legacy_audit_link"
    assert "supplement" in answer["limitations"][0]


def test_protocol_must_point_to_the_exact_audit(bundle):
    root, scenario, _, _, write = bundle
    p = json.loads((root / "protocol.json").read_text())
    p["source_audit"]["sha256"] = "0" * 64
    scenario["source_protocol"] = write("protocol.json", p)
    assert "protocol_to_audit_path_or_sha256_mismatch" in result(bundle)["reasons"][0]


def test_sealed_model_paper_is_hashed_not_parsed_and_sealed_metadata_rejected(bundle):
    # Non-JSON bytes prove the helper never parses this reviewed source.
    paper = bundle[4]("sealed/model_papers/original.xml", "<original-model-paper>metadata only</original-model-paper>")
    bundle[3]({"original_model_papers_checked": [paper]})
    assert result(bundle)["status"] == "scoped_source_audit_pass"
    bundle[1]["source_protocol"] = paper
    assert "sealed_resource_cannot_be_parsed" in result(bundle)["reasons"][0]


def test_outside_project_reference_never_opens(bundle):
    bundle[3]({"repository_source_documents_checked": [{"path": "../forbidden-source", "sha256": "0" * 64}]})
    assert result(bundle)["status"] == "unknown"


def test_current_lisinopril_audit_passes_without_legacy_status_field():
    base = ROOT / "experiments/model_recommendation/execution_validation_20261001/inputs"
    scenario = next(json.loads(l) for l in (base / "scenarios.jsonl").read_text().splitlines() if json.loads(l)["compound"] == "lisinopril")
    audit = next(json.loads(l) for l in (base / "candidate_audit.jsonl").read_text().splitlines()
                 if json.loads(l)["compound"] == "lisinopril" and json.loads(l)["independence_status"] == "previously_audited_pending_recheck")
    answer = m.audit_candidate_source(scenario, audit)
    assert answer["status"] == "scoped_source_audit_pass", answer["reasons"]
    assert answer["audit_results"][0]["reviewed_reference_count"] == 12


@pytest.fixture
def overlap_bundle(bundle, monkeypatch):
    root, scenario, audit, _, write = bundle
    audit.update(independence_status="not_audited_for_this_candidate",
                 bound_snapshot_exact_match=False, exact_bound_references=[])
    source = {"path": audit["catalog_provenance"]["source_path"],
              "sha256": audit["catalog_provenance"]["source_sha256"]}
    bibliography = write("models/References.md", "Yamazaki 2017. doi:10.1000/new")
    data = write("models/Data.md", "The following studies were used for model verification:\nYamazaki 2017")
    evidence = [dict(bibliography, lines_1based=[1, 1], text="Yamazaki 2017. doi:10.1000/new"),
                dict(data, lines_1based=[1, 2], text="The following studies were used for model verification:\nYamazaki 2017")]
    frozen_audit = write("frozen_audit.json", audit)
    review = {"clinical_design_sources": [scenario["source_protocol"]],
              "source_study_overlap": {"verified_overlap": True,
                 "scenario_id": scenario["scenario_id"], "candidate_id": audit["candidate_id"],
                 "source_sha256": source["sha256"], "target_doi": "10.1000/new",
                 "original_model_role": "DGI Model Verification", "existing_frozen_audit_ref": frozen_audit},
              "dextromethorphan": {"original_author_case_mapping": {
                 "candidate_id": audit["candidate_id"], "source": source,
                 "study_identity_evidence": evidence}}}

    def register():
        ref = write("review.json", review)
        monkeypatch.setattr(m, "KNOWN_OVERLAP_REVIEWS", {(scenario["scenario_id"], audit["candidate_id"]): ref})
    register()
    return bundle, review, register


def test_documented_overlap_supersedes_unknown_without_changing_frozen_rows(overlap_bundle):
    bundle, _, _ = overlap_bundle
    before = deepcopy(bundle[1:3])
    answer = result(bundle)
    assert answer["status"] == "documented_source_study_overlap", answer
    assert answer["source_independence_confirmed_at_declared_scope"] is False
    assert answer["strict_unseen_study_stratum_eligible"] is False
    assert answer["source_study_overlap_verified"] is True
    assert answer["study_usage_role"] == "verification"
    assert answer["used_for_parameter_fitting"] == "not_established_from_reviewed_metadata"
    assert answer["clinical_outcome_values_parsed"] is False
    assert bundle[1:3] == before


@pytest.mark.parametrize("filename", ["review.json", "models/drug.json", "models/References.md", "models/Data.md", "frozen_audit.json"])
def test_known_overlap_all_hash_links_are_mandatory(overlap_bundle, filename):
    bundle, _, _ = overlap_bundle
    (bundle[0] / filename).write_text("changed")
    answer = result(bundle)
    assert answer["status"] == "unknown"
    assert "sha256_changed" in answer["reasons"][0]


@pytest.mark.parametrize("field,value", [("source_sha256", "0" * 64), ("candidate_id", "other"),
                                         ("scenario_id", "clinical:other"), ("verified_overlap", False)])
def test_known_overlap_cannot_inherit_another_identity_or_source(overlap_bundle, field, value):
    bundle, review, register = overlap_bundle
    review["source_study_overlap"][field] = value
    register()
    assert "candidate_scenario_or_source_mismatch" in result(bundle)["reasons"][0]


def test_known_overlap_requires_actual_doi_in_source_text_not_just_review_assertion(overlap_bundle):
    bundle, review, register = overlap_bundle
    evidence = review["dextromethorphan"]["original_author_case_mapping"]["study_identity_evidence"][0]
    evidence.update(bundle[4]("models/References.md", "Yamazaki 2017. doi:10.1000/other"))
    evidence["text"] = "Yamazaki 2017. doi:10.1000/other"
    register()
    assert "exact_reference_doi_or_verification_role_unconfirmed" in result(bundle)["reasons"][0]


def test_known_overlap_requires_exact_bound_excerpt(overlap_bundle):
    bundle, review, register = overlap_bundle
    review["dextromethorphan"]["original_author_case_mapping"]["study_identity_evidence"][0]["text"] += " invented"
    register()
    assert "repository_excerpt_mismatch" in result(bundle)["reasons"][0]


def test_overlap_cannot_transfer_to_other_candidate(overlap_bundle):
    bundle, _, _ = overlap_bundle
    bundle[2]["candidate_id"] = "unreviewed"
    assert "no_previously_frozen_exact" in result(bundle)["reasons"][0]


def test_known_overlap_forbids_sealed_evidence_parse(overlap_bundle):
    bundle, review, register = overlap_bundle
    evidence = review["dextromethorphan"]["original_author_case_mapping"]["study_identity_evidence"][0]
    evidence.update(bundle[4]("sealed/References.md", evidence["text"]))
    register()
    assert "sealed_resource_cannot_be_parsed" in result(bundle)["reasons"][0]


def test_current_dextromethorphan_target_overlap_overrides_prior_unknown():
    base = ROOT / "experiments/model_recommendation/execution_validation_20261001/inputs"
    scenario_id, candidate_id = next(iter(m.KNOWN_OVERLAP_REVIEWS))
    scenario = next(row for line in (base / "scenarios.jsonl").open()
                    if (row := json.loads(line))["scenario_id"] == scenario_id)
    audit = next(row for line in (base / "candidate_audit.jsonl").open()
                 if (row := json.loads(line))["scenario_id"] == scenario_id and row["candidate_id"] == candidate_id)
    assert audit["independence_status"] == "not_audited_for_this_candidate"
    answer = m.audit_candidate_source(scenario, audit)
    assert answer["status"] == "documented_source_study_overlap", answer["reasons"]
    assert answer["strict_unseen_study_stratum_eligible"] is False
    assert answer["review_overlay"]["target_doi"] == "10.1002/cpdd.281"


def test_known_overlap_freeze_refs_are_only_review_and_repository_excerpts(overlap_bundle):
    bundle, _, _ = overlap_bundle
    refs = m.known_overlap_freeze_refs(project_root=bundle[0])
    assert {ref['path'] for ref in refs} == {'review.json', 'models/References.md', 'models/Data.md'}
    assert all(len(ref['sha256']) == 64 for ref in refs)


@pytest.mark.parametrize('name', ['review.json', 'models/References.md', 'models/Data.md'])
def test_known_overlap_freeze_fails_before_scoring_on_missing_or_changed_evidence(overlap_bundle, name):
    bundle, _, _ = overlap_bundle
    (bundle[0] / name).write_text('changed')
    with pytest.raises(ValueError, match='sha256_changed'):
        m.known_overlap_freeze_refs(project_root=bundle[0])


def test_known_overlap_freeze_rejects_rehashed_false_source_excerpt(overlap_bundle):
    bundle, review, register = overlap_bundle
    review['dextromethorphan']['original_author_case_mapping']['study_identity_evidence'][0]['text'] += ' invented'
    register()
    with pytest.raises(ValueError, match='excerpt_differs'):
        m.known_overlap_freeze_refs(project_root=bundle[0])


def test_actual_overlap_freeze_dependencies_are_three_unsealed_files():
    refs = m.known_overlap_freeze_refs()
    assert len(refs) == 3
    assert all('sealed' not in Path(ref['path']).parts for ref in refs)
