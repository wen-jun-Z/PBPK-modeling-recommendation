"""Source-freeze regression checks; synthetic files only, no native simulations."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import refine_clinical_blueprints as refinement


@pytest.fixture
def synthetic_baseline(tmp_path):
    if not tmp_path.resolve().is_relative_to(ROOT):
        pytest.skip("Use --basetemp inside the project's runtime directory")
    baseline = tmp_path / "baseline"
    baseline.mkdir()

    def artifact(name, value):
        path = baseline / name
        path.write_text(json.dumps(value) + "\n")
        return refinement.ref(path, "Synthetic regression fixture, not clinical data")

    source = artifact("source.json", {"synthetic": "source"})
    model = artifact("parameter_models.json", {"synthetic": "model artifact"})
    split = artifact("parameter_split.json", {"synthetic": "training split"})
    request = artifact("development_request.json", {"synthetic": "development inputs"})
    mapping = artifact("mapping_locked.json", {"synthetic": "development mapping", "decision": 1})
    rf = artifact("rf_prediction.json", {"synthetic": "RF native prediction"})
    ours = artifact("ours_prediction.json", {"synthetic": "original native prediction"})
    artifact("comparison_input.json", {"synthetic": True, "cases": []})
    document = {
        "status": "reported", "all_prediction_attempts_locked_before_scoring": True,
        "protocol": {"pins": {source["path"]: source["sha256"]},
                     "cases": [{"case_id": "synthetic_case"}]},
        "locks": {
            "inputs": {r["path"]: r["sha256"] for r in (model, split, request)},
            "mappings": {mapping["path"]: mapping["sha256"]},
            "predictions": {"synthetic_case": {"rf": rf, "ours": ours}},
        },
        "fits": {"synthetic_case": {"PKSimStandard": {
            "eligible": True, "mapping_path": mapping["path"]}}},
        "results": {"synthetic_case": {
            "rf": {"status": "predicted", "prediction_path": rf["path"]},
            "ours": {"status": "predicted", "prediction_path": ours["path"]}}},
    }
    refinement.save(baseline / "summary.json", document)
    return baseline, tmp_path / "refinement", document


def test_prepare_preserves_source_mapping_prediction_and_training_locks(synthetic_baseline):
    baseline, output, original = synthetic_baseline
    protocol = refinement.prepare(baseline, output, workers=1)
    prepared = refinement.read(output / "summary.json")
    expected = {**original["protocol"]["pins"], **original["locks"]["mappings"]}
    for methods in original["locks"]["predictions"].values():
        expected.update({r["path"]: r["sha256"] for r in methods.values()})
    assert all(protocol["pins"][path] == digest for path, digest in expected.items())
    assert prepared["locks"]["inputs"] == original["locks"]["inputs"]
    assert protocol["baseline_summary"]["sha256"] == refinement.sha256(baseline / "summary.json")
    assert prepared["parameter_training_reused"] is True
    refinement.verify(protocol["pins"])
    refinement.verify(prepared["locks"]["inputs"])


def test_changed_old_mapping_is_rejected_by_prepared_protocol(synthetic_baseline):
    baseline, output, original = synthetic_baseline
    protocol = refinement.prepare(baseline, output, workers=1)
    mapping = ROOT / next(iter(original["locks"]["mappings"]))
    mapping.write_text('{"synthetic": "altered decision after preparation"}\n')
    with pytest.raises(ValueError, match="Frozen input changed"):
        refinement.verify(protocol["pins"])


def test_preparation_references_results_without_copying_or_modifying_artifacts(synthetic_baseline):
    baseline, output, original = synthetic_baseline
    before = {str(p.relative_to(baseline)): p.read_bytes() for p in baseline.rglob("*") if p.is_file()}
    refinement.prepare(baseline, output, workers=1)
    after = {str(p.relative_to(baseline)): p.read_bytes() for p in baseline.rglob("*") if p.is_file()}
    assert after == before
    assert {p.name for p in output.iterdir()} == {"summary.json"}
    prepared = refinement.read(output / "summary.json")
    assert prepared["results"] == {"synthetic_case": {"rf": original["results"]["synthetic_case"]["rf"]}}
    assert prepared["fits"] == {}
    assert (ROOT / prepared["results"]["synthetic_case"]["rf"]["prediction_path"]).is_relative_to(baseline)


def test_unreported_baseline_and_duplicate_destination_are_rejected(synthetic_baseline):
    baseline, output, original = synthetic_baseline
    unfinished = deepcopy(original)
    unfinished["status"] = "predictions_locked"
    refinement.save(baseline / "summary.json", unfinished)
    with pytest.raises(ValueError, match="Complete and report the baseline"):
        refinement.prepare(baseline, output, workers=1)
    assert not output.exists()
    refinement.save(baseline / "summary.json", original)
    refinement.prepare(baseline, output, workers=1)
    prepared_bytes = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        refinement.prepare(baseline, output, workers=1)
    assert (output / "summary.json").read_bytes() == prepared_bytes
