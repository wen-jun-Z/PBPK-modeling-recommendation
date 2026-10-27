"""Guard paired reports against changed controls, pooled stages and false gains.

All source/prediction values below are synthetic software fixtures, not experiments.
Temporary files stay inside this project's runtime directory and are removed.
"""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import render_pbpk_method_comparison as report


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def synthetic_documents(folder, count=1):
    """Explicitly synthetic inputs for validation/layout, never final results."""
    original = folder/"synthetic_source.txt"
    original.write_text("Synthetic renderer fixture only; NOT clinical observations.\n")
    provenance = {"path": str(original), "sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                  "citation": "Synthetic software fixture; not clinical research"}
    baseline = {"schema_version": report.SCHEMA, "title": "Synthetic renderer verification", "cases": []}
    optimized = deepcopy(baseline)
    for i in range(count):
        identity = {"case_id": f"synthetic_case_{i}", "study_id": f"SyntheticStudy_{i}"}
        ep = {}
        for key in report.ENDPOINTS:
            unit = ("h" if key in {"tmax", "terminal_half_life", "mrt"} else
                    "L/h" if key == "clearance" else "L" if key in {"vz", "vss"} else
                    "ng.h/mL" if key in {"auc", "auc_inf"} else "ng/mL")
            ep[key] = {"value": .5 if key == "tmax" else 10., "unit": unit,
                       "definition": "Synthetic "+key, "statistic": "arithmetic_mean", "matrix": "plasma",
                       "kind": "clinical_study_reported", "location": "Synthetic fixture"}
        clinical_curve = {"time_h": [1., 2.], "concentration": [10., 5.], "unit": "ng/mL",
                          "matrix": "plasma", "statistic": "arithmetic_mean",
                          "kind": "clinical_study_reported", "extraction": "reported_table",
                          "location": "Synthetic fixture"}
        case = {**identity, "drug": f"SyntheticDrug{i+1:02d}",
                "conditions": "Synthetic software fixture; no drug or clinical prediction was run.",
                "approximations": ["representative_adult_vs_group_summary", "standard_nca_from_reported_sampling"],
                "clinical_reference_quality_notes": ["Synthetic reference used only for renderer validation."],
                "nca_policy": {"sampling_time_h": [0., 1., 2.], "lloq_ng_ml": .1,
                               "integration": "linear_up_log_down"},
                "clinical_reference": {**identity, "primary_source": provenance,
                    "verified_against_primary_source": True, "reviewer": "Synthetic test fixture, not an actual source review",
                    "curve": clinical_curve, "endpoints": deepcopy(ep)}}
        predictions = {}
        for method, stage, factor in [("rf", "shared", 2.5), ("ours", "baseline", 2.), ("ours", "optimized", 1.2)]:
            endpoints = deepcopy(ep)
            for key, item in endpoints.items():
                item["statistic"] = "representative_adult"
                item["value"] *= factor
            if stage == "optimized":
                endpoints["cmax"]["value"] = 30.  # A real numerical regression in the fixture.
            pred = {"schema_version": report.PREDICTION_SCHEMA, **identity, "method": method,
                    "curve": {**clinical_curve, "statistic": "representative_adult",
                              "concentration": [10*factor, 5*factor]}, "endpoints": endpoints,
                    "nca_diagnostics": {"auc_extrapolated_fraction": .1, "aumc_extrapolated_fraction": .1,
                                        "terminal_span_in_half_lives": 2.}}
            if i == 1 and stage == "optimized":
                pred["nca_diagnostics"]["auc_extrapolated_fraction"] = .4
            if i == 3 and stage == "optimized":
                pred = {"schema_version": report.PREDICTION_SCHEMA, **identity, "method": method,
                        "status": "failed", "reason": "Synthetic unavailable-method fixture"}
            predictions[stage] = save_json(folder/f"{i}_{stage}_{method}.json", pred)
        if i == 2:
            case["clinical_reference"]["endpoints"]["auc"] = {
                "value": None, "unit": "ng.h/mL", "reason": "Synthetic missing-reference fixture"}
        for stage, document in [("baseline", baseline), ("optimized", optimized)]:
            copy = deepcopy(case)
            copy["predictions"] = {"ours": predictions[stage], "rf": predictions["shared"]}
            document["cases"].append(copy)
    paths = [folder/"baseline.json", folder/"optimized.json"]
    for path, document in zip(paths, (baseline, optimized)):
        save_json(path, document)
    return paths


@pytest.fixture
def fixture_dir():
    with tempfile.TemporaryDirectory(prefix="paired_report_test_", dir=ROOT/"runtime") as tmp:
        yield Path(tmp)


def qualified_pairs(folder, count=4):
    before_path, after_path = synthetic_documents(folder, count)
    before, after = [report.read_json(p) for p in (before_path, after_path)]
    b, be = report.qualify_document(before)
    a, ae = report.qualify_document(after)
    assert not be and not ae
    return before, after, b, a, report.stage_pairs(before, after, b, a)


def test_missing_and_weak_tail_are_not_improvements(fixture_dir):
    _, _, _, _, pairs = qualified_pairs(fixture_dir)
    assert pairs[0]["errors"]["curve"]["status"] == "improved"
    assert pairs[0]["errors"]["cmax"]["status"] == "worsened"
    assert pairs[0]["errors"]["tmax"]["baseline"] == .5  # Absolute hours, not fold error.
    assert pairs[1]["errors"]["auc_inf"]["status"] == "descriptive_weak_tail"
    assert pairs[2]["errors"]["auc"]["status"] == "not_comparable"
    assert pairs[3]["errors"]["curve"]["status"] == "not_comparable"
    for pair, key in [(pairs[1], "auc_inf"), (pairs[2], "auc"), (pairs[3], "curve")]:
        item = pair["errors"][key]
        assert item["status"] not in {"improved", "worsened"}


@pytest.mark.parametrize("field", ["study_id", "conditions", "clinical_reference", "nca_policy", "approximations", "evaluation_input_metadata"])
def test_changed_evaluation_is_rejected(fixture_dir, field):
    before, after, b, a, _ = qualified_pairs(fixture_dir, 1)
    case = after["cases"][0]
    if field == "clinical_reference":
        case[field]["endpoints"]["clearance"]["value"] = 11
    elif field == "nca_policy":
        case[field]["lloq_ng_ml"] = 1
    elif field == "approximations":
        case[field].append("serum_as_plasma")
    elif field == "evaluation_input_metadata":
        case[field] = {"dose_mg": 1.5}
    else:
        case[field] += " changed"
    with pytest.raises(ValueError, match="changed"):
        report.stage_pairs(before, after, b, a)


def test_changed_rf_aborts_before_replacing_existing_report(fixture_dir):
    before, after, b, a, _ = qualified_pairs(fixture_dir, 1)
    rf = report.read_json(report.source(after["cases"][0]["predictions"]["rf"]))
    rf["endpoints"]["clearance"]["value"] *= 2
    after["cases"][0]["predictions"]["rf"] = save_json(fixture_dir/"changed_rf.json", rf)
    save_json(fixture_dir/"optimized.json", after)
    output = fixture_dir/"output"
    output.mkdir()
    (output/"comparison.pdf").write_bytes(b"existing report must survive")
    with pytest.raises(ValueError, match="RF prediction changed"):
        report.render(fixture_dir/"optimized.json", output, baseline_input=fixture_dir/"baseline.json")
    assert (output/"comparison.pdf").read_bytes() == b"existing report must survive"


def test_rf_identical_content_at_different_path_is_allowed(fixture_dir):
    before, after, b, a, _ = qualified_pairs(fixture_dir, 1)
    rf = report.read_json(report.source(after["cases"][0]["predictions"]["rf"]))
    other = fixture_dir/"identical_rf.json"
    other.write_text(json.dumps(rf, sort_keys=True))
    after["cases"][0]["predictions"]["rf"] = {
        "path": str(other), "sha256": hashlib.sha256(other.read_bytes()).hexdigest()}
    pairs = report.stage_pairs(before, after, b, a)
    assert pairs[0]["rf_control"] == "identical_prediction_content"


def test_summary_csv_and_single_stage_compatibility(fixture_dir, monkeypatch):
    before, after = synthetic_documents(fixture_dir, 10)
    pages = []
    def fake_compile(tex, expected_pages, *, preview):
        pages.append(expected_pages)
        tex.with_suffix(".pdf").write_bytes(b"Synthetic placeholder; real layout checked separately")
        return {"pages": expected_pages}
    monkeypatch.setattr(report, "compile_pdf", fake_compile)
    output = fixture_dir/"output"
    single = report.render(before, output)
    assert single["qualified_cases"] == 10 and single["scalar_rows"] == 100
    assert "stages" not in single
    paired = report.render(after, output, baseline_input=before)
    assert paired["qualified_cases"] == paired["unique_drugs"] == 10
    assert paired["stage_case_records"] == 20 and paired["scalar_rows"] == 200
    assert paired["stages"]["baseline"]["prediction_counts"]["failed"] == 0
    assert paired["stages"]["optimized"]["prediction_counts"]["failed"] == 1
    assert "prediction_counts" not in paired
    assert pages == [10, 21]
    rows = list(csv.DictReader((output/"comparison.csv").open()))
    assert len(rows) == 200
    assert {r["stage"] for r in rows} == {"baseline", "optimized"}
    assert len({(r["stage"], r["case_id"], r["endpoint"]) for r in rows}) == 200
    assert set(p.name for p in output.iterdir()) == {"comparison.csv", "comparison.tex", "comparison.pdf", "summary.json"}
    tex = (output/"comparison.tex").read_text()
    assert tex.count(r"\newpage") == 20
    assert tex.count(r"\cellcolor{Improved}") > 0
    assert tex.count(r"\cellcolor{Worsened}") > 0
