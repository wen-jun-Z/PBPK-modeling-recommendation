"""Synthetic runner contracts; no clinical target observations are opened."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(PROJECT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "blood_partition_runner_test", PROJECT / "scripts/run_blood_partition_blueprint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN", tmp_path / "run")
    module.RUN.mkdir()
    monkeypatch.setattr(module.base, "normalized", deepcopy)
    monkeypatch.setattr(module.base, "pins_in", lambda value, pins: None)
    monkeypatch.setattr(module, "verify", lambda pins: None)
    monkeypatch.setattr(module, "ref", lambda p: {
        "path": str(Path(p).relative_to(tmp_path)), "sha256": module.base.sha256(p)})

    def checked(value):
        if isinstance(value, dict) and "path" in value and "sha256" in value:
            path = tmp_path / value["path"]
            if module.base.sha256(path) != value["sha256"]:
                raise ValueError("Source changed: " + value["path"])
            return module.read(path)
        return deepcopy(value)

    monkeypatch.setattr(module, "checked", checked)
    return module


def request(weight=50.):
    return {"compound_id": "synthetic", "human": {"weight_kg": weight},
            "dosing": {"dose_mg": 10., "infusion_minutes": 30.},
            "evidence": [{"parameter": "molecular_weight", "value": 500.}]}


def setup_execution(runner, monkeypatch, *, n=5, eligible=True):
    source = {"synthetic_visible_sampling": "five"}
    dev, target = request(), request(100.)
    case = {"case_id": "synthetic", "requests": {
        "evidence": {"development": dev, "target": target}}}
    protocol = {"source_pins": {}, "cases": [case], "physiology_cache": {}}
    runner.save(runner.RUN / "protocol.json", protocol)
    observed = {"human": dev["human"], "dosing": dev["dosing"],
                "observations": [{"time_min": float(i+1), "concentration_umol_per_l": 1.}
                                 for i in range(n)]}
    def training(given, req, root):
        assert given == source
        assert req == dev
        return deepcopy(observed)
    monkeypatch.setattr(runner.base, "_training_document", training)
    physiology_path = runner.RUN / "synthetic_physiology.json"
    runner.save(physiology_path, {"synthetic_uncalibrated_physiology": True})
    phys_ref = runner.ref(physiology_path)
    monkeypatch.setattr(runner, "physiology", lambda *args: ({}, phys_ref, 0))
    fits = []
    def fit(predict, req, training, physiology, *, variant, budget, starts, before_trial):
        assert variant == "blood_partition"
        assert budget == runner.BUDGET and starts == runner.STARTS
        assert req["human"] == dev["human"]
        fits.append(deepcopy(training))
        before_trial()
        return {"best": {"objective": float(len(fits)), "log_parameters": [0.]*4}
                if eligible else None, "calls": 3, "failures": 1,
                "call_limit": budget, "starts": [{"calls": 0} for _ in range(starts)]}
    monkeypatch.setattr(runner.optimizer, "fit", fit)
    monkeypatch.setattr(runner.base.engine, "prediction_grid", lambda req: np.array([.1, 1.]))
    monkeypatch.setattr(runner.base, "summarize_extended_prediction", lambda *args: {})
    monkeypatch.setattr(runner, "diagnostic", lambda *args: {"calls": 0, "synthetic": True})
    if hasattr(runner, "partition_sensitivity"):
        monkeypatch.setattr(runner, "partition_sensitivity", lambda *args, **kwargs: {"calls": 0})
    job = {"case_id": "synthetic", "method": "blood_partition", "sampling": "five",
           "training_source": source}
    return protocol, job, fits, phys_ref


def synthetic_prediction(times):
    return {"concentration_mg_l": np.ones(len(times)), "observation": "synthetic",
            "mass_balance": {}, "effective_total_clearance_l_h": 1.,
            "blood_partition_audit": {"synthetic_audit": True}}


def test_five_point_fit_receives_only_visible_data_and_locks_selection_before_prediction(runner, monkeypatch):
    protocol, job, fits, _ = setup_execution(runner, monkeypatch)
    def predict(method, phys, req, times, params):
        selection = runner.read(runner.RUN / "work/unit/selection_locked.json")
        assert selection["basis"] == "development_only"
        assert selection["formula"] == "PKSimStandard"
        assert req["human"]["weight_kg"] == 100.
        return synthetic_prediction(times)
    monkeypatch.setattr(runner, "simulate", predict)
    result = runner.execute(protocol, "unit", job)
    assert len(fits) == 2
    assert all(len(fit["observations"]) == 5 for fit in fits)
    assert result["status"] == "predicted"
    assert result["target_observations_used"] is False
    assert result["resources"]["objective_calls"] == 6
    assert result.get("blood_partition_mapping", result.get("blood_partition_audit")) == {"synthetic_audit": True}


def test_five_point_branch_rejects_full_curve_before_fitting(runner, monkeypatch):
    protocol, job, fits, _ = setup_execution(runner, monkeypatch, n=6)
    with pytest.raises(ValueError, match="Five-point isolation"):
        runner.execute(protocol, "unit", job)
    assert fits == []


def test_ineligible_fits_are_retained_as_failed_attempts(runner, monkeypatch):
    protocol, job, fits, _ = setup_execution(runner, monkeypatch, eligible=False)
    monkeypatch.setattr(runner, "simulate", lambda *args: pytest.fail("An ineligible fit predicted"))
    unit = runner.execute(protocol, "unit", job)
    assert unit["status"] == "failed"
    assert unit["resources"]["failed_calls"] == 2
    assert len(unit["formula_scores"]) == 2
    assert not (runner.RUN / "work/unit/prediction_locked.json").exists()


def test_source_guard_prevents_fitting_and_prediction(runner, monkeypatch):
    protocol, job, fits, _ = setup_execution(runner, monkeypatch)
    def verify(pins):
        raise ValueError("Source changed: original-request.json")
    monkeypatch.setattr(runner, "verify", verify)
    with pytest.raises(RuntimeError, match="Source.*guard.*Source changed"):
        runner.execute(protocol, "unit", job)
    assert fits == []
    assert not (runner.RUN / "work/unit/result.json").exists()


def test_completed_resume_rejects_protocol_change_without_new_fit(runner, monkeypatch):
    protocol, job, fits, _ = setup_execution(runner, monkeypatch)
    folder = runner.RUN / "work/unit"; folder.mkdir(parents=True)
    runner.save(folder / "result.json", {"protocol_sha256": "changed", "output_pins": {}})
    with pytest.raises(ValueError, match="Changed resumed protocol"):
        runner.execute(protocol, "unit", job)
    assert fits == []


def test_completed_resume_checks_output_hashes(runner, monkeypatch):
    protocol, job, fits, _ = setup_execution(runner, monkeypatch)
    folder = runner.RUN / "work/unit"; folder.mkdir(parents=True)
    result = dict(job, protocol_sha256=runner.base.sha256(runner.RUN / "protocol.json"),
                  output_pins={"synthetic-output": "broken"})
    runner.save(folder / "result.json", result)
    def verify(pins):
        if pins: raise ValueError("Changed output hash")
    monkeypatch.setattr(runner, "verify", verify)
    with pytest.raises(ValueError, match="Changed output hash"):
        runner.execute(protocol, "unit", job)
    assert fits == []


def test_partial_resume_rejects_changed_training_source(runner, monkeypatch):
    protocol, job, fits, _ = setup_execution(runner, monkeypatch)
    folder = runner.RUN / "work/unit"; folder.mkdir(parents=True)
    req = request(); req["partition_method"] = "PKSimStandard"
    runner.save(folder / "PKSimStandard.json", {
        "protocol_sha256": runner.base.sha256(runner.RUN / "protocol.json"),
        "request": req, "training_source": {"synthetic_visible_sampling": "full"}})
    with pytest.raises(ValueError, match="Changed resumed formula"):
        runner.execute(protocol, "unit", job)
    assert fits == []


@pytest.mark.parametrize("changed", [
    {"method": "exchange_shape"}, {"sampling": "full"},
    {"training_source": {"synthetic_visible_sampling": "full"}},
])
def test_completed_resume_rejects_a_different_unit_identity(runner, monkeypatch, changed):
    protocol, job, fits, _ = setup_execution(runner, monkeypatch)
    folder = runner.RUN / "work/unit"; folder.mkdir(parents=True)
    previous = dict(job, protocol_sha256=runner.base.sha256(runner.RUN / "protocol.json"), output_pins={})
    previous.update(changed)
    runner.save(folder / "result.json", previous)
    with pytest.raises(ValueError, match="Changed resumed"):
        runner.execute(protocol, "unit", job)
    assert fits == []


@pytest.mark.parametrize("changed", [
    {"calls": 16385}, {"call_limit": 1}, {"starts": []},
    {"no_warm_start": False}, {"development_physiology": {"path": "unrelated"}},
    {"variant": "exchange_shape"},
])
def test_partial_resume_rejects_changed_budget_or_physiology(runner, monkeypatch, changed):
    protocol, job, fits, phys_ref = setup_execution(runner, monkeypatch)
    folder = runner.RUN / "work/unit"; folder.mkdir(parents=True)
    req = request(); req["partition_method"] = "PKSimStandard"
    previous = dict(protocol_sha256=runner.base.sha256(runner.RUN / "protocol.json"),
                    request=req, training_source=job["training_source"],
                    development_physiology=phys_ref, variant=job["method"],
                    call_limit=runner.BUDGET, calls=3, starts=[{}]*runner.STARTS, no_warm_start=True)
    previous.update(changed)
    runner.save(folder / "PKSimStandard.json", previous)
    with pytest.raises(ValueError, match="Changed resumed formula"):
        runner.execute(protocol, "unit", job)
    assert fits == []


def test_known_target_native_failure_stays_a_terminal_failed_unit(runner, monkeypatch):
    protocol, job, fits, phys_ref = setup_execution(runner, monkeypatch)
    def physiology(protocol, case, phase, formula, req, guard):
        if phase == "target":
            raise runner.base.NativeBuildFailed("Incomplete native construction evidence")
        return {}, phys_ref, 0
    monkeypatch.setattr(runner, "physiology", physiology)
    monkeypatch.setattr(runner, "simulate", lambda *args: pytest.fail("Failed native source predicted"))
    result = runner.execute(protocol, "unit", job)
    assert result["status"] == "failed"
    assert result["failure_stage"] == "native_target_construction"
    assert len(fits) == 2
    assert (runner.RUN / "work/unit/selection_locked.json").exists()
    assert not (runner.RUN / "work/unit/prediction_locked.json").exists()


def test_source_failure_during_target_export_is_not_a_scientific_failure(runner, monkeypatch):
    protocol, job, fits, phys_ref = setup_execution(runner, monkeypatch)
    def physiology(protocol, case, phase, formula, req, guard):
        if phase == "target": raise ValueError("Source changed during native export")
        return {}, phys_ref, 0
    monkeypatch.setattr(runner, "physiology", physiology)
    monkeypatch.setattr(runner, "simulate", lambda *args: pytest.fail("Broken source predicted"))
    with pytest.raises(ValueError, match="Source changed"):
        runner.execute(protocol, "unit", job)
    assert not (runner.RUN / "work/unit/result.json").exists()


@pytest.mark.parametrize("state", [
    {"status": "running", "units": {}},
    {"status": "predictions_locked", "units": {}},
    {"status": "predictions_locked", "units": {"unit": {"status": "fitting"}}},
])
def test_scoring_never_opens_clinical_targets_before_terminal_matrix(runner, monkeypatch, state):
    import report_blueprint_validation as reporting
    protocol = {"jobs": {"unit": {}}, "cases": [], "source_pins": {}}
    runner.save(runner.RUN / "protocol.json", protocol)
    runner.save(runner.RUN / "state.json", dict(state, protocol=runner.ref(runner.RUN / "protocol.json")))
    monkeypatch.setattr(reporting, "_case_reference", lambda *args: pytest.fail("Clinical targets opened early"))
    with pytest.raises(ValueError, match="lock|Nonterminal"):
        runner.report()


def test_pause_file_stops_guard_before_source_scan(runner, monkeypatch):
    (runner.RUN / "PAUSE").write_text("synthetic stop")
    monkeypatch.setattr(runner, "verify", lambda *args: pytest.fail("Source scan ran after explicit pause"))
    with pytest.raises(RuntimeError, match="paused"):
        runner.Guard({"source_pins": {}})(force=True)


def test_cached_physiology_rejects_a_different_native_request(runner, monkeypatch):
    native = runner.RUN / "native"
    (native / "physiology").mkdir(parents=True)
    (native / "base").mkdir()
    runner.save(native / "physiology/perfusion_physiology.json", {"native": True})
    runner.save(native / "base/completion.json", {"request": request(70.)})
    key = "synthetic__development__PKSimStandard"
    protocol = {"physiology_cache": {key: runner.ref(native / "physiology/perfusion_physiology.json")}}
    monkeypatch.setattr(runner.base.engine, "_physiology", lambda *args: pytest.fail("Mismatching native request reused"))
    with pytest.raises(ValueError, match="request|condition|differs|cache|Native"):
        runner.physiology(protocol, {"case_id": "synthetic"}, "development", "PKSimStandard", request(), lambda: None)


def test_cached_physiology_reuses_exactly_matching_native_request(runner, monkeypatch):
    native = runner.RUN / "native"
    (native / "physiology").mkdir(parents=True)
    (native / "base").mkdir()
    runner.save(native / "physiology/perfusion_physiology.json", {"native": True})
    runner.save(native / "base/completion.json", {"request": request()})
    key = "synthetic__development__PKSimStandard"
    reference = runner.ref(native / "physiology/perfusion_physiology.json")
    protocol = {"physiology_cache": {key: reference}}
    monkeypatch.setattr(runner.base.engine, "_physiology", lambda *args: pytest.fail("Exact cached source rebuilt"))
    phy, ref, new = runner.physiology(protocol, {"case_id": "synthetic"}, "development", "PKSimStandard", request(), lambda: None)
    assert phy == {"native": True} and ref == reference and new == 0


def test_report_retains_all_failed_attempts_and_renders_missing_scores(runner, monkeypatch):
    import report_blueprint_validation as reporting
    cases = [{"case_id": f"case_{i}", "drug": f"Synthetic {i}", "cluster_id": f"study_{i}"}
             for i in range(10)]
    jobs, units, controls = {}, {}, {}
    for case in cases:
        for sampling in runner.base.SAMPLINGS:
            for method in runner.ALL_METHODS:
                key = f'{case["case_id"]}__{sampling}__{method}'
                entry = {"status": "failed", "reason": "Synthetic infeasible construction", "resources": {}}
                if method in runner.METHODS:
                    jobs[key] = {}; units[key] = entry
                else:
                    controls[key] = entry
    protocol = {"jobs": jobs, "cases": cases, "controls": controls, "source_pins": {},
                "limitations": ["Synthetic test only"], "preflight": {}}
    runner.save(runner.RUN / "protocol.json", protocol)
    runner.save(runner.RUN / "state.json", {"status": "predictions_locked", "locked_at": 1.,
        "protocol": runner.ref(runner.RUN / "protocol.json"), "units": units, "output_pins": {}})
    endpoint_names = ("auc", "cmax", "tmax", "c_last", "terminal_half_life", "auc_inf",
                      "clearance", "vz", "mrt", "vss")
    opened = []
    def reference(case, store):
        opened.append(case["case_id"])
        return {}, {"unit": "mg/l"}, {name: {"value": 1., "unit": "synthetic"} for name in endpoint_names}
    monkeypatch.setattr(reporting, "_case_reference", reference)
    monkeypatch.setattr(reporting, "_score_prediction", lambda *args: pytest.fail("Failed attempt scored as prediction"))
    result = runner.report()
    assert len(opened) == 10
    assert result["status"] == "complete"
    assert result["rows"] == 880
    assert result["development_adoption_rule_passed"] is False
    assert all(result["summary"][sampling][method]["failed_or_unscorable"] == 10
               for sampling in runner.base.SAMPLINGS for method in runner.ALL_METHODS)
    assert "—" in (runner.RUN / "README.md").read_text()
