"""Reporting contract tests: lock-first scoring and honest study denominators."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import report_blueprint_evidence_policy as reporting


def write_ref(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture_protocol(tmp_path):
    root = tmp_path / "project"
    run = root / "run"
    run.mkdir(parents=True)
    source = write_ref(root, "source.json", {"provenance": "fixture"})
    cases = [{"case_id": name, "drug": drug, "study_id": "Pletz2004", "cluster_id": "Pletz2004",
              "reference_source": source, "physchem": "source.json"}
             for name, drug in (("a", "Ertapenem"), ("b", "Ceftriaxone"))]
    protocol = {"version": reporting.PROTOCOL_VERSION,
                "methods": list(reporting.METHODS), "scenarios": list(reporting.SCENARIOS),
                "sampling": list(reporting.SAMPLING), "cases": cases,
                "source_pins": {source["path"]: source["sha256"]}, "decisions": [], "units": [], "computations": {}}
    state = {"status": "predictions_locked", "all_attempts_locked_before_scoring": True, "computations": {}}
    for case in cases:
        cid = case["case_id"]
        for scenario in reporting.SCENARIOS:
            for method in reporting.METHODS:
                expected = reporting._expected_ready(scenario, method)
                actual = expected
                # Control without a conflict guard demonstrates an incorrect
                # allowance. The oracle stays separate from actual behavior.
                if scenario == "conflicting_fu" and method == "no_conflict_guard":
                    actual = True
                if scenario == "missing_fu" and method == "fixed_template":
                    actual = False
                status = "ready" if actual else "blocked"
                protocol["decisions"].append({"id": "__".join((cid, scenario, method)), "case_id": cid,
                    "scenario": scenario, "method": method, "status": status, "expected_ready": expected,
                    "decision": {"status": status, "actions": [], "reasons": []}})
                for sampling in reporting.SAMPLING:
                    unit = {"id": "__".join((cid, scenario, sampling, method)), "case_id": cid,
                            "scenario": scenario, "sampling": sampling, "method": method,
                            "status": "planned" if actual else "blocked"}
                    if actual:
                        # Exact numerical input identity allows scenario aliases.
                        identity = "__".join((cid, sampling, method))
                        unit["computation_id"] = identity
                        if identity not in protocol["computations"]:
                            protocol["computations"][identity] = {"kind": "native", "case_id": cid,
                                "sampling": sampling, "method": method}
                            prediction = write_ref(root, f"predictions/{identity}.json", {"identity": identity})
                            request = write_ref(root, f"requests/{identity}.json", {"identity": identity})
                            state["computations"][identity] = {"status": "predicted", "prediction": prediction,
                                "request": request, "resources": {"actual_new_native_calls": 1, "wall_seconds": 2}}
                    protocol["units"].append(unit)
    return root, run, protocol, state


def save_run(run, protocol, state):
    (run / "protocol.json").write_text(json.dumps(protocol))
    (run / "state.json").write_text(json.dumps(state))


def fake_reference(case, store):
    endpoints = {name: {"value": 10.0, "unit": unit, "location": "clinical fixture"}
                 for name, unit in reporting.UNITS.items()}
    curve = {"unit": "mg/l", "time_h": [1, 2, 3], "concentration": [3, 2, 1], "location": "clinical fixture"}
    return {"case_id": case["case_id"]}, curve, endpoints


def fake_prediction(case, clinical, curve, endpoints, unit, store):
    score = 1.5 if case["case_id"] == "a" else 2.0
    return {"curve_metrics": {"gmfe": score, "ln_rmse": .3, "n": 3},
            "endpoints": {name: {"value": 12.0, "reference": 10.0, "unit": observed["unit"],
                                "error": 2.0 if name == "tmax" else 1.2,
                                "error_metric": "absolute_error_h" if name == "tmax" else "fold_error",
                                "weak_tail": False} for name, observed in endpoints.items()},
            "curve": {"large": "not duplicated"}}, {}, 100.0


@pytest.mark.parametrize("status,locked", [("running", False), ("paused", True), ("prepared", True),
                                          ("predictions_locked", False), ("complete", False)])
def test_target_outcomes_are_never_read_before_lock(tmp_path, monkeypatch, status, locked):
    root, run, protocol, state = fixture_protocol(tmp_path)
    state.update(status=status, all_attempts_locked_before_scoring=locked)
    save_run(run, protocol, state)
    def forbidden(*args):
        pytest.fail("Clinical targets were decoded before the lock")
    monkeypatch.setattr(reporting, "_case_reference", forbidden)
    with pytest.raises(ValueError, match="locked"):
        reporting.report(run, project_root=root)
    assert not (run / "results.json").exists()


def test_incomplete_attempt_matrix_blocks_target_reads(tmp_path, monkeypatch):
    root, run, protocol, state = fixture_protocol(tmp_path)
    protocol["units"].pop()
    save_run(run, protocol, state)
    monkeypatch.setattr(reporting, "_case_reference", lambda *args: pytest.fail("Unexpected target read"))
    with pytest.raises(ValueError, match="attempt matrix"):
        reporting.report(run, project_root=root)


def test_source_tampering_blocks_target_reads(tmp_path, monkeypatch):
    root, run, protocol, state = fixture_protocol(tmp_path)
    save_run(run, protocol, state)
    (root / "source.json").write_text("changed")
    monkeypatch.setattr(reporting, "_case_reference", lambda *args: pytest.fail("Unexpected target read"))
    with pytest.raises(ValueError, match="Source lock mismatch"):
        reporting.report(run, project_root=root)


def test_blocked_requests_keep_denominators_and_shared_study_cluster(tmp_path, monkeypatch):
    root, run, protocol, state = fixture_protocol(tmp_path)
    failed = "b__five__fixed_template"
    state["computations"][failed].update(status="failed", failure="Native execution failure")
    save_run(run, protocol, state)
    calls = []
    def track(*args):
        calls.append(args[4]["prediction"]["path"])
        return fake_prediction(*args)
    monkeypatch.setattr(reporting, "_case_reference", fake_reference)
    monkeypatch.setattr(reporting, "_score_prediction", track)
    output = reporting.report(run, project_root=root)
    assert len(calls) == len(state["computations"]) - 1
    assert len(set(calls)) == len(calls), "Aliased scenarios must reuse the exact same score"
    assert output["statistics"]["independent_drugs"] == 2
    assert output["statistics"]["independent_study_clusters"] == 1
    available = output["statistics"]["groups"]["available"]["full"]["evidence_policy"]["curve"]
    assert available["n_cases"] == 2
    assert available["n_clusters"] == 1
    assert available["ci95"] is None
    assert available["estimate"] == pytest.approx((1.5 * 2.0) ** .5)
    conflict = output["statistics"]["groups"]["conflicting_fu"]["full"]["evidence_policy"]["curve"]
    assert conflict["planned_requests"] == 2
    assert conflict["blocked"] == 2
    assert conflict["scored"] == 0 and conflict["coverage"] == 0
    failure_group = output["statistics"]["groups"]["available"]["five"]["fixed_template"]
    assert failure_group["curve"]["failed"] == 1 and failure_group["curve"]["coverage"] == .5
    assert failure_group["endpoints"]["cmax"]["reference_available"] == 2
    assert failure_group["endpoints"]["cmax"]["scored"] == 1
    assert failure_group["endpoints"]["cmax"]["coverage_of_observed"] == .5
    engineering = output["engineering"]["by_scenario"]
    assert engineering["conflicting_fu"]["no_conflict_guard"]["incorrect_allow"] == 2
    assert engineering["missing_fu"]["fixed_template"]["incorrect_block"] == 2
    assert engineering["missing_required_pbpk_formula"]["two_compartment"]["correct_allow"] == 2
    assert engineering["unsupported_ddi"]["two_compartment"]["correct_block"] == 2
    assert output["engineering"]["by_method"]["evidence_policy"]["requests"] == 12
    assert output["resources"]["actual_new_native_calls"]["sum"] == len(state["computations"])
    assert all(row["scoring_status"] == "not_applicable" and not row["clinical_references"]
               for row in output["units"] if row["scenario"] == "unsupported_ddi")
    assert "unsupported_ddi" not in output["statistics"]["groups"]
    assert all("curve" not in row for row in output["units"])
    assert set(path.name for path in run.iterdir()) == {"protocol.json", "state.json", "results.json", "results.csv"}
    import csv
    with (run / "results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(protocol["units"]) * (len(reporting.UNITS) + 1)
    row = next(row for row in rows if row["case_id"] == "b" and row["scenario"] == "available" and
               row["sampling"] == "five" and row["method"] == "fixed_template" and row["endpoint"] == "cmax")
    assert row["prediction"] == "" and row["error"] == "" and row["clinical_reference"] == "10.0"


def test_pletz_drugs_must_share_study_cluster(tmp_path):
    _, _, protocol, state = fixture_protocol(tmp_path)
    protocol["cases"][1]["cluster_id"] = "incorrect_independent_study"
    with pytest.raises(ValueError, match="share one clinical study"):
        reporting._validate_matrix(protocol, state)


def test_capability_oracle_cannot_punish_classical_for_missing_formula(tmp_path):
    _, _, protocol, state = fixture_protocol(tmp_path)
    decision = next(d for d in protocol["decisions"]
                    if d["scenario"] == "missing_required_pbpk_formula" and d["method"] == "two_compartment")
    decision["expected_ready"] = False
    with pytest.raises(ValueError, match="oracle"):
        reporting._validate_matrix(protocol, state)


def test_alias_cannot_cross_compounds_or_sampling(tmp_path):
    _, _, protocol, state = fixture_protocol(tmp_path)
    item = next(unit for unit in protocol["units"] if unit.get("computation_id"))
    protocol["computations"][item["computation_id"]]["case_id"] = "different_compound"
    with pytest.raises(ValueError, match="different drug or sampling"):
        reporting._validate_matrix(protocol, state)


def test_alias_requires_matching_numerical_signature(tmp_path):
    _, _, protocol, state = fixture_protocol(tmp_path)
    item = next(unit for unit in protocol["units"] if unit.get("computation_id"))
    protocol["computations"][item["computation_id"]]["signature"] = {"frozen_input": 1}
    item["computation_signature_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="numerical input signature"):
        reporting._validate_matrix(protocol, state)


def test_completed_report_reuses_locked_result_without_self_referential_hash(tmp_path, monkeypatch):
    root, run, protocol, state = fixture_protocol(tmp_path)
    save_run(run, protocol, state)
    monkeypatch.setattr(reporting, "_case_reference", fake_reference)
    monkeypatch.setattr(reporting, "_score_prediction", fake_prediction)
    first = reporting.report(run, project_root=root)
    before = (run / "results.json").read_bytes()
    state.update(status="complete", results={"path": "run/results.json", "sha256": hashlib.sha256(before).hexdigest()})
    save_run(run, protocol, state)
    monkeypatch.setattr(reporting, "_case_reference", lambda *args: pytest.fail("Completed report should reuse locked result"))
    again = reporting.report(run, project_root=root)
    assert first == again
    assert (run / "results.json").read_bytes() == before
    assert "run/state.json" not in again["source_pins"]
    assert "run/results.json" not in again["source_pins"]


def test_recorded_source_field_completeness_does_not_claim_truth():
    source = {"path": "visible.json", "sha256": "a" * 64, "location": "/evidence/0"}
    decisions = [{"id": "a", "case_id": "a", "scenario": "available", "method": "evidence_policy",
                  "status": "ready", "expected_ready": True, "decision": {
                      "selected_evidence": [{"source": source}, {"source": {"path": "incomplete"}}],
                      "actions": [{"action": "reject", "reason": "wrong species"}]}}]
    summary = reporting.engineering_summary(decisions)
    entry = summary["by_method"]["evidence_policy"]
    assert entry["source_field_completeness"] == .5
    assert entry["rejection_reason_counts"] == {"wrong species": 1}
    assert "not correctness" in summary["interpretation"]
