"""Source provenance must remain verifiable and separate from ranking labels."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/build_frontier_assessment_sources.py"
spec = importlib.util.spec_from_file_location("assessment_sources", SCRIPT)
sources = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sources)


def test_exact_source_spans_and_stable_identifiers():
    text = "# Results\n\nThe model reproduced observed plasma concentrations.\n\nPredicted AUC ratios were within 2-fold for 95% of studies.\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    rows = sources.text_segments(text, path="report.md", file_sha=digest)
    assert rows == sources.text_segments(text, path="report.md", file_sha=digest)
    for row in rows:
        loc = row["location"]
        assert text[loc["char_start"]:loc["char_end"]] == row["excerpt"]
        assert hashlib.sha256(row["excerpt"].encode()).hexdigest() == row["excerpt_sha256"]
        assert loc["line_start"] >= 1
    assert "reported_quantitative_performance" in {r["evidence_kind"] for r in rows}


@pytest.mark.parametrize("text", [
    "Predictions within 2-fold were used as an acceptance criterion.",
    "The figure shows predicted and observed ratios with dashed lines at 2-fold.",
    "The predicted AUC ratios should be within 2-fold.",
])
def test_evaluation_criteria_are_not_observed_performance(text):
    assert sources.classify(text)["evidence_kind"] == "evaluation_method_only"


def test_claims_are_not_independent_validation_by_default():
    generic = sources.classify("The model was validated using clinical studies.")
    assert not generic["flags"]["external_holdout_explicit_in_excerpt"]
    independent = sources.classify("Independent clinical validation data were not used for model building.")
    assert independent["flags"]["external_holdout_explicit_in_excerpt"]
    limitation = sources.classify("The model cannot predict outcomes in children and was not validated in pregnancy.")
    assert limitation["evidence_kind"] == "limitation"


@pytest.mark.parametrize("text", [
    "About 70% of fluconazole is excreted unchanged in urine during 120 hours.",
    "To accurately predict concentrations, expression was increased 3.57-fold.",
    "The observer noted urinary fractions of 74% and 75%.",
    "Schnider TW, Minto CF, Shafer SL. Anesthesiology. 1998;88(5):1170-82.",
    "The hepatic ratio of enzyme expression was increased 2-fold.",
])
def test_physiological_values_are_not_prediction_errors(text):
    assert sources.classify(text)["evidence_kind"] != "reported_quantitative_performance"


def test_bibliography_not_performance_and_real_metric_recognized():
    text = "Predicted AUC ratios were within 2-fold of observations."
    assert sources.classify(text, "References")["evidence_kind"] == "bibliographic_reference"
    assert sources.classify("The GMFE was 1.32.")["evidence_kind"] == "reported_quantitative_performance"
    assert sources.classify("Predictions for oral administration were reasonable (GMFE < 2).")["evidence_kind"] == "reported_quantitative_performance"


def test_numerical_report_tables_keep_partition_and_group():
    excerpt = "| Group | GMFE |\n|:---|:---|\n| Model building oral | 1.09 |\n| Model verification iv | 1.14 |\n| All | 1.18 |\n"
    result = sources.classify(excerpt, "Diagnostics")
    assert result["evidence_kind"] == "reported_quantitative_performance"
    metrics = result["reported_metrics"]
    assert [m["value"] for m in metrics] == [1.09, 1.14, 1.18]
    assert [m["evaluation_partition"] for m in metrics] == ["model_building", "model_verification_reported", "unspecified"]
    assert all(not m["comparable_across_different_study_groups"] for m in metrics)


def test_source_traversal_and_external_symlink_rejected(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    with pytest.raises(ValueError):
        sources.local_path(root, "../other_project/model.json")
    (root / "link").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        sources.local_path(root, "link/model.json")


def fixture_candidate(root, *, document=True):
    model = root / "assets/repository/source/model.json"
    model.parent.mkdir(parents=True)
    model.write_text('{"Name":"Model"}')
    if document:
        (model.parent / "README.md").write_text("# Model\n\nThe PBPK model describes oral dosing in healthy adults.\n")
    out = root / "experiment"
    (out / "inputs").mkdir(parents=True)
    candidate = {"candidate_id": "candidate-a", "family_id": "family-a", "family_name": "Model", "compounds": ["compound"],
                 "features": {"routes": ["oral"]}, "provenance": {"source_path": str(model.relative_to(root)),
                 "source_sha256": sources.sha256_file(model)}}
    (out / "inputs/candidates.jsonl").write_text(json.dumps(candidate) + "\n")
    return out, model


def test_build_without_scores_or_labels_and_keep_unknown(tmp_path):
    out, model = fixture_candidate(tmp_path, document=False)
    manifest = sources.build(tmp_path, out)
    row = json.loads((out / "assessment/sources.jsonl").read_text())
    assert manifest["source_artifact_hashes_verified"] == 1
    assert not manifest["policy"]["labels_created"]
    assert not row["documents"]
    assert row["coverage"]["scientific_applicability"] == "not_adjudicated"
    assert row["coverage"]["no_document_means"] == "unknown_not_negative"
    assert row["parsed_structure"]["evidence_role"] == "descriptive_only_not_independent_gold"


def test_changed_source_fails_before_overwriting_output(tmp_path):
    out, model = fixture_candidate(tmp_path)
    sources.build(tmp_path, out)
    before = (out / "assessment/sources.jsonl").read_bytes()
    model.write_text('{"Name":"changed"}')
    with pytest.raises(ValueError, match="SHA mismatch"):
        sources.build(tmp_path, out)
    assert (out / "assessment/sources.jsonl").read_bytes() == before


def test_duplicate_markdown_pdf_is_traced_but_not_double_counted(tmp_path):
    doc = tmp_path / "report.md"
    doc.write_text("The PBPK model reproduced clinical observations.")
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"counterpart existence suffices; no fake PDF parsing")
    entry = sources.load_document(pdf, tmp_path)
    assert entry["status"] == "markdown_counterpart_used"
    assert entry["counterpart"] == "report.md"
    assert entry["sha256"] == sources.sha256_file(pdf)
    assert not entry["segments"]


def test_boilerplate_excluded_and_long_spans_bounded():
    text = "# License\n\nThe model is licensed.\n\n# Results\n\n" + "The model describes oral dosing. " * 130
    rows = sources.text_segments(text, path="report.md", file_sha="x")
    assert rows
    assert all("licensed" not in row["excerpt"] for row in rows)
    assert all(len(row["excerpt"]) <= sources.MAX_SEGMENT_CHARS for row in rows)
    assert any(row["paragraph_split"] for row in rows)
