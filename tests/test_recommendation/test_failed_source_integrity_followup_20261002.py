"""Failed source history preserves evidence without allowing verified reuse."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
BASE = ROOT / "experiments/model_recommendation/execution_validation_20261001/engineering_amendments/snapshot_input_followup_20261002"
LIVE = ROOT / "scripts/recommendation_execution_repairs/historical_source_proofs.py"
STAGED = BASE / "staged/scripts/recommendation_execution_repairs/historical_source_proofs.py"


@pytest.fixture
def history():
    module = types.ModuleType("staged_failed_source_integrity")
    module.__file__ = str(LIVE)
    exec(compile(STAGED.read_text(), str(LIVE), "exec"), module.__dict__)
    return module


def failed_entry(history):
    manifest = json.loads((ROOT / history.FAILED_MANIFEST).read_text())
    return manifest["entries"][0]


def test_real_failed_receipt_archives_original_producer_but_is_not_reusable(history):
    entry = failed_entry(history)
    receipt_path = ROOT / entry["receipt"]["path"]
    receipt = json.loads(receipt_path.read_text())
    verify = history.historical_checker(receipt_path, receipt, ROOT)
    assert receipt["status"] == "failed"
    assert receipt["scientific_inapplicability_inferred"] is False
    assert receipt["ode_run"] is receipt["sealed_observations_accessed"] is False
    for ref in receipt["implementation"]:
        archived = verify(ref)
        assert archived.is_relative_to(BASE / "failed_source_checkpoint/implementation")
        assert hashlib.sha256(archived.read_bytes()).hexdigest() == ref["sha256"]
    assert history.original_receipt(entry, ROOT) is None
    verified = [row for record in history.records(ROOT) for row in record["entries"]]
    assert len(verified) == 80
    assert all(row["source_sha256"] != entry["source_sha256"] for row in verified)


def test_failed_receipt_verifies_only_when_acceptance_is_not_required(history, monkeypatch):
    from recommendation_execution_repairs import snapshot_repair as repair

    entry = failed_entry(history)
    original_reference = repair.reference
    design = "src/pbpk_kb/blueprints/design_runtime.py"
    staged_sha = hashlib.sha256((BASE / "staged" / design).read_bytes()).hexdigest()

    def future_reference(path, root):
        ref = original_reference(path, root)
        return dict(ref, sha256=staged_sha) if ref["path"] == design else ref

    monkeypatch.setattr(repair, "reference", future_reference)
    monkeypatch.setitem(sys.modules, "recommendation_execution_repairs.historical_source_proofs", history)
    path = ROOT / entry["receipt"]["path"]
    result = repair._verify_receipt(path, entry, ROOT, require_accepted=False)
    assert result["status"] == "failed"
    with pytest.raises(ValueError, match="SNAPSHOT_REPAIR_NOT_VERIFIED"):
        repair._verify_receipt(path, entry, ROOT, require_accepted=True)


@pytest.mark.parametrize("change", ["status", "reason", "scientific_label"])
def test_registered_failure_cannot_be_upgraded_or_edited(history, change):
    entry = failed_entry(history)
    path = ROOT / entry["receipt"]["path"]
    value = json.loads(path.read_text())
    if change == "status":
        value.update(status="verified", comparison={"accepted": True, "exact_normalized_document_equality": True})
    elif change == "reason":
        value["reasons"] = []
    else:
        value["scientific_inapplicability_inferred"] = True
    with pytest.raises(ValueError, match="failed receipt cannot become"):
        history.historical_checker(path, value, ROOT)


@pytest.mark.parametrize("corruption", ["implementation", "preparation", "manifest"])
def test_archive_corruption_rejected_without_changing_real_archive(history, tmp_path, corruption):
    manifest = json.loads((ROOT / history.FAILED_MANIFEST).read_text())
    paths = [history.FAILED_MANIFEST, manifest["entries"][0]["receipt"]["path"]]
    paths += [row["preserved"]["path"] for key in ("implementation_archive", "preparation_archive") for row in manifest[key]]
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    receipt = tmp_path / manifest["entries"][0]["receipt"]["path"]
    value = json.loads(receipt.read_text())
    assert callable(history.historical_checker(receipt, value, tmp_path))
    if corruption == "manifest":
        target = tmp_path / history.FAILED_MANIFEST
    else:
        key = "implementation_archive" if corruption == "implementation" else "preparation_archive"
        target = tmp_path / manifest[key][0]["preserved"]["path"]
    target.write_text("tampered")
    with pytest.raises(ValueError, match="MIGRATION_BOUND_RESOURCE_CHANGED"):
        history.historical_checker(receipt, value, tmp_path)
