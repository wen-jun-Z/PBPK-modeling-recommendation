"""Synthetic contracts for evidence-value interventions; no native simulation."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from tests.test_blueprints.test_evidence_decision import request, fallback, record, evidence

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_blueprint_value.py"
SPEC = importlib.util.spec_from_file_location("blueprint_value_test_module", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def rich_request():
    base = request()
    predicted = fallback()
    for row in predicted:
        row["prediction_method"].update(name="shared_scaffold_excluded_parameter_model", version=runner.old.VERSION)
    base["evidence"] = [r for r in base["evidence"] if r["parameter"] not in runner.source_policy.FALLBACK_PARAMETERS] + predicted
    base["forbidden_sources"] = [{"path": "target.json", "sha256": "b"*64, "location": "heldout clinical source"}]
    return base


def test_rich_adapter_keeps_real_name_and_frozen_policy_unchanged():
    base = rich_request(); before = deepcopy(base)
    assert runner.source_policy.decide(base, [])["status"] == "blocked"
    assert runner.policy(base, [])["status"] == "ready"
    assert runner.source_policy.decide(base, [])["status"] == "blocked"
    assert record(base, "total_clearance")["prediction_method"]["name"] == "shared_scaffold_excluded_parameter_model"
    assert base == before
    record(base, "binding_partner")["value"] = "Glycoprotein"
    assert runner.policy(base, [])["status"] == "ready"
    assert record(base, "binding_partner")["value"] == "Glycoprotein"
    record(base, "total_clearance")["prediction_method"]["training_overlap"] = True
    assert runner.policy(base, [])["status"] == "blocked"


def test_missing_clearance_masks_renal_family_and_does_not_touch_fu():
    base = rich_request()
    independent = [deepcopy(record(request(), "total_clearance")), deepcopy(record(request(), "fraction_unbound")),
        evidence("renal_clearance", 1., "ml/min/kg", context={"species": "human", "matrix": "plasma", "route": "iv"})]
    available, decision = runner.select_request(base, independent, "available")
    assert decision["status"] == "ready"
    assert record(available, "renal_clearance")["value"] == 1.
    assert record(available, "fraction_unbound")["value"] == .1
    masked, decision = runner.select_request(base, independent, "missing_clearance")
    assert decision["status"] == "ready"
    assert not any(r["parameter"] in {"renal_clearance", "hepatic_clearance"} for r in masked["evidence"])
    assert record(masked, "total_clearance")["value"] == 6.
    assert record(masked, "fraction_unbound")["value"] == .1
    both, _ = runner.select_request(base, independent, "missing_both")
    assert runner.semantic_request(both) == runner.semantic_request(base)
    reordered = deepcopy(both); reordered["evidence"].reverse()
    reordered["evidence"][0]["source"]["location"] = "different spelling"
    assert runner.semantic_request(reordered) == runner.semantic_request(base)


def test_mixed_engineering_challenges_and_invalid_rich_values():
    base = rich_request()
    cases = [{"case_id": "synthetic"}]
    results = runner.challenges(cases, {"synthetic": base})
    assert len(results) == 11 and all(r["correct"] for r in results)
    assert sum(r["actual_ready"] for r in results) == 2
    record(base, "fraction_unbound")["value"] = 1.1
    assert runner.policy(base, [])["status"] == "blocked"


def test_scoring_cannot_open_targets_until_predictions_locked(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "own", lambda path: Path(path))
    monkeypatch.setattr(runner, "verify", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.old, "checked", lambda ref: {"tasks": {}})
    monkeypatch.setattr(runner.evaluation, "clinical_truth", lambda *args: pytest.fail("target opened before lock"))
    (tmp_path / "state.json").write_text(json.dumps({"protocol": {}, "outcomes": {}, "all_predictions_locked_before_scoring": False}))
    with pytest.raises(ValueError, match="terminal and locked"):
        runner.report(tmp_path, locked=True)


def test_publication_snapshot_does_not_invalidate_completed_report(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    paper = tmp_path / "paper.pdf"; paper.write_bytes(b"original")
    protocol = {"source_pins": {}, "code_pins": {}, "publication_snapshot": {"paper.pdf": runner.sha256(paper)}}
    paper.write_bytes(b"authorised later publication revision")
    runner.verify(protocol)
    with pytest.raises(runner.old.Pause, match="Publication changed"):
        runner.verify(protocol, publication=True)
