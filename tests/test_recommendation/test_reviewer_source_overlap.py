import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("reviewer_source_overlap", ROOT / "scripts/reviewer_source_overlap.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_no_hit_never_proves_independence():
    assert audit.classify([], "unknown") == ("not_established", "unknown")
    assert audit.classify([], "unknown", True)[1] == "no_overlap_identified_in_previously_audited_scope"


def test_reference_alone_does_not_prove_model_validation():
    docs = [{"source": {"path": "References.md"}, "lines": ["**Someone 2025** a study doi:10.1000/ABC"]}]
    hits = audit.scan_documents(docs, {"doi": "10.1000/abc"})
    assert audit.classify(hits, "unknown") == ("mentioned_only", "documented_mention_usage_not_established")


def test_explicit_role_link_must_share_bibliography_alias():
    docs = [{"source": {"path": "References.md"}, "lines": ["**Someone 2025** doi:10.1000/ABC"]},
            {"source": {"path": "Data.md"}, "lines": ["### Model building", "used for model building (training data)",
                                                        "[Different 2020]", "### Model verification", "used for model verification", "[Someone 2025]"]}]
    hits = audit.scan_documents(docs, {"doi": "10.1000/abc"})
    assert audit.classify(hits, "unknown")[0] == "validation"


def test_training_and_validation_not_conflated():
    assert audit.section_role(["### Building", "used for model building (training data)", "[Study 2000]"], 2)[0] == "calibration"
    assert audit.section_role(["### Verification", "used for model verification", "[Study 2000]"], 2)[0] == "validation"


def test_neighbouring_bibliography_entry_cannot_inherit_identity():
    docs = [{"source": {"path": "References.md"}, "lines": ["**Wrong 2000** different study", "", "**Correct 2020** doi:10.1000/xyz"]},
            {"source": {"path": "Data.md"}, "lines": ["### Calibration", "used for model calibration", "[Wrong 2000]"]}]
    hits = audit.scan_documents(docs, {"doi": "10.1000/xyz"})
    assert audit.classify(hits, "unknown")[0] == "mentioned_only"


def test_pmid_substring_is_not_exact_identity():
    assert not audit.identifier_hits("123456789", {"pmid": "2345678"})
    assert audit.identifier_hits("PMID 2345678", {"pmid": "2345678"})


def test_long_exact_title_works_when_reference_omits_doi():
    title = "Clinical pharmacokinetics of test compound in healthy participants"
    assert audit.identifier_hits(title.upper(), {"title": title})
    assert not audit.identifier_hits("Clinical pharmacokinetics of a different compound", {"title": title})


def test_known_positive_is_not_erased_by_no_hit():
    assert audit.classify([], "documented_source_study_overlap") == ("validation", "documented_source_study_overlap")


def test_document_outside_project_rejected(tmp_path):
    (tmp_path / "inside").mkdir()
    other = tmp_path / "outside.txt"
    other.write_text("test")
    with pytest.raises(ValueError):
        audit.safe_path(other, root=tmp_path / "inside")
