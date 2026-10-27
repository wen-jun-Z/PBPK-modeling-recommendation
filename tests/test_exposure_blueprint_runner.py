"""Runner contracts using synthetic records only; never open clinical targets."""
from copy import deepcopy
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("exposure_runner_test", ROOT / "scripts/run_exposure_blueprint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN", tmp_path / "run")
    module.RUN.mkdir()
    monkeypatch.setattr(module, "normalized", deepcopy)
    monkeypatch.setattr(module, "ref", lambda p: {"path": str(Path(p).relative_to(tmp_path)), "sha256": module.sha256(p)})
    monkeypatch.setattr(module, "pins_in", lambda value, pins: None)
    monkeypatch.setattr(module, "verify", lambda pins: None)
    return module


def request(weight=50.):
    context = {"species": "human", "matrix": "plasma", "route": "intravenous"}
    return {"compound_id": "synthetic", "human": {"weight_kg": weight},
            "dosing": {"dose_mg": 10., "infusion_minutes": 30.},
            "evidence": [{"parameter": "molecular_weight", "value": 500.},
                         {"parameter": "total_clearance", "value": 10., "unit": "ml/min/kg", "context": context},
                         {"parameter": "renal_clearance", "value": 2., "unit": "ml/min/kg", "context": context}]}


def training(n=5):
    req = request()
    return {"human": req["human"], "dosing": req["dosing"],
            "observations": [{"time_min": float(i + 1), "concentration_umol_per_l": 2. * math.exp(-i / 10)} for i in range(n)]}


def hungry(fun, x0, **kwargs):
    for i in range(10000):
        fun(np.asarray(x0) + [i * 1e-10, 0., 0.])
    pytest.fail("Actual model-call budget did not stop finite differences")


def test_relaxed_budget_counts_failed_trials_and_finite_differences(runner, monkeypatch):
    calls = []
    starts = []
    def simulate(_phys, _req, times, *_):
        calls.append(times.copy())
        if len(calls) % 3 == 0:
            raise ValueError("Synthetic numerical failure")
        return {"concentration_mg_l": np.ones(len(times))}
    monkeypatch.setattr(runner.engine, "_simulate", simulate)
    def optimizer(fun, x0, **kwargs):
        starts.append(np.asarray(x0).copy())
        return hungry(fun, x0, **kwargs)
    monkeypatch.setattr(runner, "least_squares", optimizer)
    result = runner.relaxed_fit({"circulation": {"cardiac_flow_l_h": 300.}}, request(), training(), budget=17, starts=3)
    assert result["calls"] == len(calls) == 17
    assert result["failures"] == 5
    assert [s["calls"] for s in result["starts"]] == [6, 6, 5]
    assert all(s["status"] == "call_budget_exhausted" for s in result["starts"])
    assert result["no_warm_start"] is True
    np.testing.assert_array_equal(np.asarray(starts)[:, 1:], runner.exposure_profile.initial_starts(3))


def test_relaxed_removes_only_cl_penalty_and_uses_common_physical_bounds(runner, monkeypatch):
    captured = {}
    x = np.log([2., 3., 4.])
    monkeypatch.setattr(runner.engine, "_simulate", lambda _p, _r, times, *_: {"concentration_mg_l": np.ones(len(times))})
    def optimizer(fun, x0, **kwargs):
        captured["bounds"] = kwargs["bounds"]
        captured["residual"] = fun(x)
        return types.SimpleNamespace(success=True)
    monkeypatch.setattr(runner, "least_squares", optimizer)
    runner.relaxed_fit({"circulation": {"cardiac_flow_l_h": 300.}}, request(), training(), budget=4, starts=1)
    lo, hi = np.exp(captured["bounds"])
    assert lo == pytest.approx([1e-4 / 24., .1, .01])
    assert hi == pytest.approx([3000. / 24., 10., 100.])
    assert len(captured["residual"]) == 7
    assert captured["residual"][-2:] == pytest.approx([.1 * x[1] / math.log(10), .1 * x[2] / math.log(100)])


def test_relaxed_guard_failure_is_not_counted_as_numerical_failure(runner, monkeypatch):
    monkeypatch.setattr(runner, "least_squares", hungry)
    monkeypatch.setattr(runner.engine, "_simulate", lambda *_: pytest.fail("Model evaluated after source guard stopped"))
    def stop():
        raise RuntimeError("Source changed")
    with pytest.raises(RuntimeError, match="Source changed"):
        runner.relaxed_fit({"circulation": {"cardiac_flow_l_h": 300.}}, request(), training(), budget=4, starts=1, before_trial=stop)


def test_cached_export_rejects_changed_request_before_native_export(runner, monkeypatch):
    base = runner.RUN / "cached/base"
    base.mkdir(parents=True)
    runner.save(base / "completion.json", {"request": request(40.)})
    monkeypatch.setattr(runner, "export_physiology", lambda **_: pytest.fail("Mismatching native source reused"))
    with pytest.raises(ValueError, match="Native base differs"):
        runner.cached_export(base.parent, request())


def setup_execution(runner, monkeypatch, n=5, eligible=True):
    dev, target = request(), request(100.)
    source = {"synthetic_sampling": "five"}
    case = {"case_id": "synthetic", "requests": {"evidence": {"development": dev, "target": target}}}
    protocol = {"cases": [case], "physiology_cache": {}}
    runner.save(runner.RUN / "protocol.json", protocol)
    monkeypatch.setattr(runner, "checked", deepcopy)
    def visible_only(given, req, root):
        assert given == source
        return training(n)
    monkeypatch.setattr(runner, "_training_document", visible_only)
    monkeypatch.setattr(runner, "physiology", lambda *_: ({}, {"new_native": 0, "cached_native": True}))
    fits = []
    def fit(_phys, req, observations, budget, starts, guard):
        fits.append((deepcopy(req), deepcopy(observations)))
        assert budget == runner.BUDGET and starts == runner.STARTS
        return {"best": {"objective": float(len(fits)), "log_parameters": [math.log(2.), 0., 0.]} if eligible else None,
                "calls": 7, "failures": 2, "starts": [{} for _ in range(runner.STARTS)]}
    monkeypatch.setattr(runner, "fit_cell", fit)
    job = {"case_id": "synthetic", "method": "fixed_prior", "sampling": "five", "training_source": source}
    return protocol, job, fits


def test_five_points_only_lock_selection_before_prediction_and_scale_cl_at_target_weight(runner, monkeypatch):
    protocol, job, fits = setup_execution(runner, monkeypatch)
    monkeypatch.setattr(runner.engine, "prediction_grid", lambda _: np.array([.1, 1.]))
    def simulate(_phys, req, times, _variant, _keys, x):
        selected = runner.read(runner.RUN / "work/unit/selection_locked.json")
        assert selected["formula"] == "PKSimStandard"
        assert selected["basis"] == "development_only"
        cl = runner.engine.clearance_inputs(req)
        effective = cl["renal_clearance_l_h"] + (cl["total_clearance_l_h"] - cl["renal_clearance_l_h"]) * math.exp(x[0])
        assert effective == pytest.approx(108.)
        assert cl["renal_clearance_l_h"] == 12.
        return {"concentration_mg_l": np.ones(len(times)), "observation": "synthetic", "mass_balance": {}, "effective_total_clearance_l_h": effective}
    monkeypatch.setattr(runner.engine, "_simulate", simulate)
    monkeypatch.setattr(runner, "summarize_extended_prediction", lambda *_: {})
    result = runner.execute(protocol, "unit", job, lambda: None)
    assert len(fits) == 2 and all(len(observed["observations"]) == 5 for _, observed in fits)
    assert result["status"] == "predicted" and result["target_observations_used"] is False
    assert result["resources"]["objective_calls"] == 14
    assert result["resources"]["new_native"] == 0


def test_five_point_job_rejects_full_curve_before_fitting(runner, monkeypatch):
    protocol, job, fits = setup_execution(runner, monkeypatch, n=6)
    with pytest.raises(ValueError, match="Five-point isolation"):
        runner.execute(protocol, "unit", job, lambda: None)
    assert fits == []


def test_all_ineligible_formula_fits_remain_terminal_failed_unit(runner, monkeypatch):
    protocol, job, fits = setup_execution(runner, monkeypatch, eligible=False)
    monkeypatch.setattr(runner.engine, "_simulate", lambda *_: pytest.fail("No eligible fit may predict"))
    result = runner.execute(protocol, "unit", job, lambda: None)
    assert result["status"] == "failed"
    assert len(result["formula_scores"]) == 2
    assert result["resources"]["failed_calls"] == 4
    assert runner.read(runner.RUN / "work/unit/result.json")["status"] == "failed"


@pytest.mark.parametrize("message", [
    "Incomplete native construction evidence",
    "Bounded native export failed; inspect retained process log",
])
def test_known_target_native_failure_is_retained_as_terminal_failed_unit(runner, monkeypatch, message):
    native_physiology = runner.physiology
    protocol, job, fits = setup_execution(runner, monkeypatch)
    monkeypatch.setattr(runner, "physiology", native_physiology)
    def native(req, _folder, _root, _guard):
        if req["human"]["weight_kg"] == 100.:
            raise ValueError(message)
        return {}
    monkeypatch.setattr(runner.engine, "_physiology", native)
    monkeypatch.setattr(runner.engine, "_simulate", lambda *_: pytest.fail("Failed native model must not predict"))
    result = runner.execute(protocol, "unit", job, lambda: None)
    assert result["status"] == "failed"
    assert result["failure_stage"] == "native_target_construction"
    assert result["reason"] == message
    assert len(fits) == len(result["formula_scores"]) == 2
    assert result["resources"]["objective_calls"] == 14
    assert runner.read(runner.RUN / "work/unit/result.json")["status"] == "failed"
    assert (runner.RUN / "work/unit/selection_locked.json").exists()
    assert not (runner.RUN / "work/unit/prediction_locked.json").exists()


def test_source_guard_value_error_during_target_native_construction_propagates(runner, monkeypatch):
    native_physiology = runner.physiology
    protocol, job, _ = setup_execution(runner, monkeypatch)
    monkeypatch.setattr(runner, "physiology", native_physiology)
    corrupted = False
    def guard():
        if corrupted:
            raise ValueError("Source changed: synthetic-input.json")
    def native(req, _folder, _root, before_trial):
        nonlocal corrupted
        if req["human"]["weight_kg"] == 100.:
            corrupted = True
            before_trial()
        return {}
    monkeypatch.setattr(runner.engine, "_physiology", native)
    monkeypatch.setattr(runner.engine, "_simulate", lambda *_: pytest.fail("Model predicted after integrity failure"))
    with pytest.raises(ValueError, match="Source changed: synthetic-input.json"):
        runner.execute(protocol, "unit", job, guard)
    assert not (runner.RUN / "work/unit/result.json").exists()
    assert not (runner.RUN / "work/unit/prediction_locked.json").exists()


@pytest.mark.parametrize("state", [
    {"status": "running", "all_predictions_locked_before_scoring": False, "units": {}},
    {"status": "predictions_locked", "all_predictions_locked_before_scoring": True, "units": {}},
    {"status": "predictions_locked", "all_predictions_locked_before_scoring": True, "units": {"unit": {"status": "fitting"}}},
])
def test_report_never_opens_target_before_complete_terminal_matrix(runner, monkeypatch, state):
    report = types.ModuleType("report_blueprint_validation")
    def forbidden(*_, **__):
        pytest.fail("Target/reference opened before all predictions locked")
    for name in ("Store", "_case_reference", "_score_prediction", "paired_ratio", "clustered_geomean"):
        setattr(report, name, forbidden)
    monkeypatch.setitem(sys.modules, "report_blueprint_validation", report)
    monkeypatch.setattr(runner, "checked", lambda _: None)
    runner.save(runner.RUN / "protocol.json", {"jobs": {"unit": {}}})
    runner.save(runner.RUN / "state.json", {"protocol": {}, **state})
    with pytest.raises(ValueError):
        runner.report()


def synthetic_report(runner, monkeypatch, *, all_failed=False):
    """Exercise real aggregation/bootstrap around wholly invented score records."""
    import report_blueprint_validation as statistics
    report = types.ModuleType("report_blueprint_validation")
    report.Store = lambda _: object()
    report.clustered_geomean = lambda values, clusters: statistics.clustered_geomean(values, clusters, resamples=40)
    report.paired_ratio = lambda numerator, denominator, clusters: statistics.paired_ratio(numerator, denominator, clusters, resamples=40)
    cases = [{"case_id": f"case{i}", "drug": name, "cluster_id": cluster}
             for i, (name, cluster) in enumerate([
                 ("Dexmedetomidine", "shared_study"), ("SyntheticB", "shared_study"),
                 ("SyntheticC", "other_study"), ("SyntheticD", "failed_study")])]
    def references(case, _store):
        i = int(case["case_id"][-1])
        return {}, {"unit": "ng/ml"}, {
            "auc_inf": {"value": None if i == 2 else 3., "unit": "ng*h/ml"},
            "tmax": {"value": 1., "unit": "h"}}
    report._case_reference = references
    def score(case, _clinical, _curve, endpoints, unit, _store):
        assert not all_failed
        i = int(case["case_id"][-1])
        scale = {"exposure_constrained": 1., "fixed_prior": 2., "direct_relaxed": 4.,
                 "paper_pbpk_reference": 3., "two_compartment_reference": .5}[unit["method"]]
        gmfe = [2., 1., 4.][i] * scale * (2. if unit["sampling"] == "five" else 1.)
        return {"curve_metrics": {"gmfe": gmfe}, "endpoints": {
            "auc_inf": {"value": 6., "reference": endpoints["auc_inf"]["value"], "unit": "ng*h/ml",
                        "error": [2., 8., None][i], "error_metric": "fold_error", "weak_tail": i == 1},
            "tmax": {"value": 1. + i / 2., "reference": 1., "unit": "h",
                     "error": i / 2., "error_metric": "absolute_error_h", "weak_tail": False}}}, {}, 500.
    report._score_prediction = score
    monkeypatch.setitem(sys.modules, "report_blueprint_validation", report)
    methods = runner.METHODS + ("paper_pbpk_reference", "two_compartment_reference")
    units, controls, jobs = {}, {}, {}
    for i, case in enumerate(cases):
        for sampling in runner.SAMPLINGS:
            for method in methods:
                uid = f'{case["case_id"]}__{sampling}__{method}'
                failure = all_failed or i == 3 or (i == 1 and sampling == "five" and method == "exposure_constrained")
                unit = {"case_id": case["case_id"], "method": method, "sampling": sampling,
                        "status": "failed" if failure else "predicted", "reason": "Synthetic failure",
                        "resources": {"objective_calls": 10 * (i + 1), "failed_calls": i % 2,
                                      "new_native": int(i == 0 and sampling == "full" and method == "fixed_prior"),
                                      "wall_seconds": i + 1.},
                        "formula_scores": [{"fit": {"synthetic_fit": f"{uid}:{formula}"}} for formula in runner.FORMULAS]}
                if method in runner.METHODS:
                    units[uid] = unit
                    jobs[uid] = {}
                else:
                    unit["resources"] = {"wall_seconds": 333.}
                    controls[uid] = unit
    def checked(reference):
        if "synthetic_fit" in reference:
            return {"starts": [{"status": status} for status in
                               ("converged", "call_budget_exhausted", "optimizer_limit", "infeasible")]}
        return None
    monkeypatch.setattr(runner, "checked", checked)
    runner.save(runner.RUN / "protocol.json", {"cases": cases, "jobs": jobs, "controls": controls, "source_pins": {}})
    runner.save(runner.RUN / "state.json", {"status": "predictions_locked", "protocol": {}, "units": units,
                "all_predictions_locked_before_scoring": True, "output_pins": {}, "locked_at": 123.})
    runner.report()
    return runner.read(runner.RUN / "results.json")


def test_report_pairs_by_study_and_retains_failures_in_all_sampling_denominators(runner, monkeypatch):
    doc = synthetic_report(runner, monkeypatch)
    full = doc["summary"]["full"]["exposure_constrained"]
    assert (full["attempted"], full["n_cases"], full["n_clusters"], full["failed_or_unscorable"]) == (4, 3, 2, 1)
    assert full["estimate"] == pytest.approx(2.)
    pair = doc["comparisons"]["full"]["fixed_prior"]
    assert pair["estimate"] == pytest.approx(.5)
    assert pair["ci95"] == pytest.approx([.5, .5])
    assert (pair["attempted_pairs"], pair["complete_pairs"], pair["n_clusters"]) == (4, 3, 2)
    assert pair["incomplete_case_ids"] == ["case3"]
    sparse = doc["five_over_full_gmfe"]["exposure_constrained"]
    assert sparse["estimate"] == pytest.approx(2.)
    assert sparse["per_case_ratio"] == {"case0": 2., "case2": 2.}
    assert sparse["attempted_pairs"] == 4 and sparse["complete_pairs"] == 2
    assert sparse["incomplete_case_ids"] == ["case1", "case3"]
    sensitivity = doc["source_sensitivity_excluding_nemonoxacin_dexmedetomidine_midazolam"]["full"]["fixed_prior"]
    assert sensitivity["attempted_pairs"] == 3 and sensitivity["complete_pairs"] == 2
    assert "case0" not in sensitivity["per_case_ratio"]


def test_report_endpoint_units_weak_scores_missing_references_and_resource_accounting(runner, monkeypatch):
    doc = synthetic_report(runner, monkeypatch)
    endpoints = doc["endpoint_summary"]["full"]["fixed_prior"]
    auc = endpoints["auc_inf"]
    assert (auc["attempted"], auc["reference_available"], auc["predicted"], auc["scored"], auc["weak_scored"]) == (4, 3, 3, 2, 1)
    assert auc["gmfe"] == pytest.approx(4.) and auc["within_twofold"] == 1
    assert endpoints["tmax"]["mean_absolute_error_h"] == pytest.approx(.5)
    assert "gmfe" not in endpoints["tmax"]
    resources = doc["resources"]["fixed_prior"]
    assert resources["objective_calls"] == 200 and resources["failed_calls"] == 4
    assert resources["new_native"] == 1 and resources["wall_seconds"] == 20.
    assert resources["starts"] == 64 and resources["budget_exhausted_starts"] == resources["converged_starts"] == 16
    assert resources["reused"] is False
    # Historical reference timing stays explicitly distinguishable from new
    # work. It must not be interpreted as this run's wall-clock expenditure.
    assert doc["resources"]["paper_pbpk_reference"]["reused"] is True
    assert doc["resources"]["paper_pbpk_reference"]["objective_calls"] == 0
    assert doc["resources"]["paper_pbpk_reference"]["wall_seconds"] == 8 * 333.
    with (runner.RUN / "results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 * 2 * 5 * 3
    failed = [r for r in rows if r["drug"] == "SyntheticD" and r["endpoint"] == "tmax"]
    assert len(failed) == 10
    assert all(r["status"] == "failed" and r["reference"] == "1.0" and r["prediction"] == "" for r in failed)


def test_report_all_failed_cells_still_complete_with_explicit_zero_coverage(runner, monkeypatch):
    doc = synthetic_report(runner, monkeypatch, all_failed=True)
    for sampling in runner.SAMPLINGS:
        for stats in doc["summary"][sampling].values():
            assert stats["estimate"] is None and stats["n_cases"] == 0
            assert stats["attempted"] == stats["failed_or_unscorable"] == 4
    assert runner.read(runner.RUN / "state.json")["status"] == "complete"
    assert "—" in (runner.RUN / "README.md").read_text()
