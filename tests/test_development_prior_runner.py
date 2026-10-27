"""Synthetic runner contracts; never read clinical targets or experiment results."""
from copy import deepcopy
import csv
import importlib.util
import math
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("development_prior_runner_test", ROOT / "scripts/run_development_prior_selection.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN", tmp_path / "run")
    module.RUN.mkdir()
    monkeypatch.setattr(module, "verify", lambda pins: None)
    monkeypatch.setattr(module.base, "normalized", deepcopy)
    monkeypatch.setattr(module.base, "pins_in", lambda value, pins: None)
    monkeypatch.setattr(module, "ref", lambda p: {"path": str(Path(p).relative_to(tmp_path)), "sha256": module.base.sha256(p)})
    def checked(reference):
        path = tmp_path / reference["path"]
        if module.base.sha256(path) != reference["sha256"]:
            raise ValueError("Synthetic source changed")
        return module.read(path)
    monkeypatch.setattr(module, "checked", checked)
    return module


def setup_execution(runner, monkeypatch, n=5, selected="direct_relaxed"):
    cid, key = "synthetic", "synthetic__five"
    development = {"human": {"weight_kg": 70}, "dosing": {"dose_mg": 10}, "evidence": []}
    request_file = runner.RUN / "development.json"
    runner.save(request_file, development)
    physiology = runner.RUN / "physiology.json"
    runner.save(physiology, {"synthetic": True})
    job = {"case_id": cid, "sampling": "five", "training_source": {"sentinel": "visible-five-only"}}
    protocol = {"source_pins": {}, "cases": [{"case_id": cid, "requests": {"evidence": {"development": runner.ref(request_file)}}}],
                "physiology_cache": {f"{cid}__development__{f}": runner.ref(physiology) for f in runner.base.FORMULAS},
                "final_candidates": {}}
    for sampling in ("five", "full"):
        for method in runner.METHODS:
            protocol["final_candidates"][f"{cid}__{sampling}__{method}"] = {
                "status": "predicted", "prediction": {"sentinel": f"{sampling}-{method}"},
                "request": {"sentinel": f"{sampling}-{method}-request"}}
    runner.save(runner.RUN / "protocol.json", protocol)
    calls = []
    def training(source, request, root):
        assert source == job["training_source"] and request == development and root == runner.ROOT
        return {"observations": [{"time_min": i, "concentration_umol_per_l": 1} for i in range(n)]}
    monkeypatch.setattr(runner, "_training_document", training)
    def select(phys, request, visible, fitters, simulate, **kwargs):
        calls.append(deepcopy(visible))
        assert set(fitters) == set(runner.METHODS)
        assert kwargs["budget"] == 4096 and kwargs["starts"] == 16
        return {"selected_method": selected, "fitting_calls": 70, "validation_calls": 10}
    monkeypatch.setattr(runner.selector, "select", select)
    return protocol, key, job, calls


def test_execute_uses_only_five_visible_points_and_exact_frozen_candidate(runner, monkeypatch):
    protocol, key, job, calls = setup_execution(runner, monkeypatch)
    result = runner.execute(protocol, key, job)
    assert len(calls) == 1 and len(calls[0]["observations"]) == 5
    assert result["chosen_method"] == "direct_relaxed"
    assert result["prediction"] == {"sentinel": "five-direct_relaxed"}
    assert result["request"] == {"sentinel": "five-direct_relaxed-request"}
    assert result["status"] == "selected"
    # Resume reuses this same locked decision, without another CV fit.
    assert runner.execute(protocol, key, job) == result
    assert len(calls) == 1


def test_five_job_rejects_larger_curve_before_selection(runner, monkeypatch):
    protocol, key, job, calls = setup_execution(runner, monkeypatch, n=6)
    with pytest.raises(ValueError, match="Five-point"):
        runner.execute(protocol, key, job)
    assert calls == []


def test_failed_selection_keeps_blocked_unit_without_frozen_prediction(runner, monkeypatch):
    protocol, key, job, calls = setup_execution(runner, monkeypatch, selected=None)
    result = runner.execute(protocol, key, job)
    assert result["status"] == "blocked" and result["chosen_method"] is None
    assert "prediction" not in result and "request" not in result
    assert len(calls) == 1


def test_resume_rejects_protocol_mutation_before_reusing_decision(runner, monkeypatch):
    protocol, key, job, calls = setup_execution(runner, monkeypatch)
    runner.execute(protocol, key, job)
    runner.save(runner.RUN / "protocol.json", protocol | {"changed": True})
    with pytest.raises(ValueError, match="Resume protocol changed"):
        runner.execute(protocol, key, job)
    assert len(calls) == 1


def setup_report(runner, monkeypatch, state_status="selections_locked", missing=False, blocked=False):
    cases = [{"case_id": f"drug-{i}", "cluster_id": f"study-{max(i, 1)}"} for i in range(10)]
    jobs, units, records = {}, {}, []
    for i, case in enumerate(cases):
        for sampling in ("full", "five"):
            cid = case["case_id"]
            key = f"{cid}__{sampling}"
            choice = None if blocked and i == 0 and sampling == "five" else ("direct_relaxed" if i % 2 else "fixed_prior")
            selection_file = runner.RUN / f"{key}.json"
            runner.save(selection_file, {"selected_method": choice, "fitting_calls": 70, "fitting_failures": 2, "validation_calls": 10, "validation_failures": 0, "calls": 80, "failures": 2,
                                         "methods": {method: {"folds": [{"formula_fits": [{}, {}]} for _ in range(5)]} for method in runner.METHODS}})
            jobs[key] = {"case_id": cid, "sampling": sampling}
            units[key] = {"case_id": cid, "sampling": sampling, "chosen_method": choice,
                          "status": "selected" if choice else "blocked", "selection": runner.ref(selection_file), "output_pins": {}}
            for j, method in enumerate(runner.METHODS):
                value = float(1 + i + j / 10 + (sampling == "five") / 100)
                records.append({"case_id": cid, "drug": cid, "sampling": sampling, "method": method, "status": "predicted",
                                "curve_metrics": {"gmfe": value, "ln_rmse": value / 7}, "curve": {"marker": value},
                                "endpoints": {f"endpoint-{k}": {"value": value + k, "reference": 9 + k, "unit": "synthetic", "error": value / 2, "error_metric": "fold_error", "weak_tail": k == 9} for k in range(10)},
                                "nca_diagnostics": {"marker": value}, "nca_status": "ok", "weak_endpoints": ["endpoint-9"], "checks": {"valid": True}})
    score_file = runner.RUN / "synthetic-scores.json"
    runner.save(score_file, {"records": records})
    protocol = {"jobs": jobs, "cases": cases, "source_pins": {}, "score_reference": runner.ref(score_file)}
    runner.save(runner.RUN / "protocol.json", protocol)
    if missing:
        units.pop(next(iter(units)))
    state = {"status": state_status, "protocol": runner.ref(runner.RUN / "protocol.json"), "units": units, "locked_at": 100}
    runner.save(runner.RUN / "state.json", state)
    checked = runner.checked
    opened = []
    def observed_checked(reference):
        if reference == protocol["score_reference"]:
            opened.append("scores")
        return checked(reference)
    monkeypatch.setattr(runner, "checked", observed_checked)
    def gm(values, clusters):
        return {"estimate": math.exp(sum(map(math.log, values.values())) / len(values)) if values else None,
                "n_cases": len(values), "n_clusters": len(set(clusters.values()))}
    def paired(a, b, clusters):
        valid = [k for k in clusters if a.get(k) is not None and b.get(k) is not None]
        return {"attempted_pairs": len(clusters), "complete_pairs": len(valid)}
    monkeypatch.setitem(sys.modules, "report_blueprint_validation", types.SimpleNamespace(clustered_geomean=gm, paired_ratio=paired))
    return records, units, opened


@pytest.mark.parametrize("state_status,missing", [("running", False), ("selections_locked", True)])
def test_report_never_opens_scores_until_every_selection_locked(runner, monkeypatch, state_status, missing):
    _, _, opened = setup_report(runner, monkeypatch, state_status, missing)
    with pytest.raises(ValueError, match="Lock every selection"):
        runner.report()
    assert opened == []


def test_source_hash_failure_stops_before_opening_target_scores(runner, monkeypatch):
    _, _, opened = setup_report(runner, monkeypatch)
    def changed_source(_pins):
        raise ValueError("Synthetic source hash changed")
    monkeypatch.setattr(runner, "verify", changed_source)
    with pytest.raises(ValueError, match="source hash changed"):
        runner.report()
    assert opened == []


def test_report_copies_exact_matching_candidate_and_keeps_all_11_endpoints(runner, monkeypatch):
    original, units, opened = setup_report(runner, monkeypatch)
    runner.report()
    assert opened == ["scores"]
    results = runner.read(runner.RUN / "results.json")
    originals = {(r["case_id"], r["sampling"], r["method"]): r for r in original}
    assert len(results["records"]) == 60
    assert results["additional_cv_resources"] == {"fitting_calls": 1400, "fitting_failures": 40,
        "validation_calls": 200, "validation_failures": 0, "formula_fits": 400,
        "new_final_predictions": 0, "reused_final_candidates": 40}
    for record in results["records"]:
        chosen = units[f'{record["case_id"]}__{record["sampling"]}']["chosen_method"] if record["method"] == "adaptive" else record["method"]
        expected = originals[record["case_id"], record["sampling"], chosen]
        for field in ("curve_metrics", "endpoints", "curve", "nca_diagnostics"):
            assert record[field] == expected[field]
        assert record["prediction_reused"] is True
    with (runner.RUN / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 60 * 11
    assert set(r["endpoint"] for r in rows) == {"concentration_time_curve"} | {f"endpoint-{i}" for i in range(10)}


def test_blocked_selection_keeps_failure_denominator_and_only_reference_values(runner, monkeypatch):
    _, _, _ = setup_report(runner, monkeypatch, blocked=True)
    runner.report()
    result = runner.read(runner.RUN / "results.json")
    failed = next(r for r in result["records"] if r["status"] == "blocked")
    assert failed["curve_metrics"].get("gmfe") is None
    assert not failed.get("curve") and not failed.get("nca_diagnostics")
    assert all(v["value"] is None and v["error"] is None and v["reference"] is not None for v in failed["endpoints"].values())
    summary = result["summary"]["five"]["adaptive"]
    assert summary["attempted"] == 10 and summary["failed_or_unscorable"] == 1 and summary["n_cases"] == 9
    assert result["comparisons"]["five"]["fixed_prior"] == {"attempted_pairs": 10, "complete_pairs": 9}
