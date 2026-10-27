"""Licence receipts must remain bound to original source and frozen commit."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("execution_license", ROOT / "scripts/recommendation_execution_repairs/license_audit.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

GRANT = "# Example\n\n## License\nThe model is distributed under the [GPLv2 Lincense](https://example.invalid/license).\n\n## References\nReferences.\n"


@pytest.fixture
def receipt(tmp_path):
    def write(path, value):
        path = tmp_path / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if isinstance(value, dict) else value)
        return {"path": str(path.relative_to(tmp_path)), "sha256": m._sha(path)}
    source = write("assets/source/model.json", "original model")
    commit = "a" * 40
    repository = "Open-Systems-Pharmacology/Example"
    asset = {"repository": repository, "commit": commit, "source_directory": "assets/source",
             "files": [{"path": "model.json", "sha256": source["sha256"]}]}
    write("assets/asset_manifest.json", asset)
    candidate = {"candidate_id": "candidate", "provenance": {"source_path": source["path"],
                 "source_sha256": source["sha256"], "repository_commit": commit, "license_status": "unresolved"}}
    audit = write("inputs/audit.jsonl", "frozen")
    doc = write("receipts/README.md", GRANT)
    doc.update(name="README.md", repository=repository, commit=commit,
               url=m._url(repository, commit, "README.md"), grant=m._grant(GRANT, "README.md"))
    entry = {"candidate_id": "candidate", "catalog_provenance": candidate["provenance"],
             "repository_binding": m._repository_binding(candidate, tmp_path), "status": "license_verified",
             "evidence_paths": [doc["path"]], "license_id": "GPL-2.0"}
    manifest = {"version": m.VERSION, "candidate_audit": audit, "candidates": [entry],
                "documents": [doc], "limitations": ["No clinical-data rights implied."]}
    write("receipts/manifest.json", manifest)
    return tmp_path, candidate, manifest


def verify(receipt, manifest=None):
    root, candidate, data = receipt
    if manifest is not None:
        (root / "receipts/manifest.json").write_text(json.dumps(manifest))
    return m.audit_candidate_license(candidate, project_root=root, receipt_path="receipts/manifest.json")


def test_explicit_same_repository_model_grant_is_accepted(receipt):
    answer = verify(receipt)
    assert answer["status"] == "license_verified"
    assert answer["scientific_qualification"] is False
    assert answer["license_id"] == "GPL-2.0"


@pytest.mark.parametrize("path", ["assets/source/model.json", "assets/asset_manifest.json", "inputs/audit.jsonl", "receipts/README.md"])
def test_each_source_or_evidence_drift_is_rejected(receipt, path):
    (receipt[0] / path).write_text("changed")
    assert verify(receipt)["status"] == "unknown"


@pytest.mark.parametrize("text", ["GPLv2", "## License\nSee Suite's GPLv2 license.",
                                      "## References\nThe model is distributed under GPLv2.",
                                      "## License\nPublic repository."])
def test_mentions_and_other_repository_license_are_not_model_grants(text):
    assert m._grant(text, "README.md") is None


@pytest.mark.parametrize("field,value", [("commit", "b" * 40), ("repository", "Open-Systems-Pharmacology/Other"),
                                          ("url", "https://example.invalid/README.md")])
def test_license_from_other_commit_or_repository_rejected(receipt, field, value):
    data = deepcopy(receipt[2])
    data["documents"][0][field] = value
    assert verify(receipt, data)["status"] == "unknown"


def test_catalogue_identity_cannot_be_changed_to_match_a_known_license(receipt):
    receipt[1]["provenance"]["source_sha256"] = "b" * 64
    assert verify(receipt)["status"] == "unknown"


def test_claimed_accepted_without_explicit_grant_rejected_even_if_hash_updated(receipt):
    root, _, data = receipt
    text = "## License\nPublicly available, no licence grant.\n"
    (root / "receipts/README.md").write_text(text)
    data["documents"][0]["sha256"] = m._sha(root / "receipts/README.md")
    assert verify(receipt, data)["status"] == "unknown"


def test_no_following_sealed_or_outside_project_license_paths(receipt):
    data = deepcopy(receipt[2])
    for unsafe in ["sealed/README.md", "../README.md"]:
        data["candidates"][0]["evidence_paths"] = [unsafe]
        data["documents"][0]["path"] = unsafe
        answer = verify(receipt, data)
        assert answer["status"] == "unknown"
        assert answer["reason"] == "PATH_OUTSIDE_PROJECT_OR_SEALED"
