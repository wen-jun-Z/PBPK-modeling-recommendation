"""Learning orchestration contracts; synthetic project-local artifacts only."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_blueprint_learning as runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return runner.ref(path)


@pytest.fixture
def batch(tmp_path, monkeypatch):
    if not tmp_path.resolve().is_relative_to(ROOT):
        pytest.skip("Use --basetemp inside this project's runtime directory")
    base, run = tmp_path / "frozen", tmp_path / "new_run"
    monkeypatch.setattr(runner, "BASE", base)

    def own_run(path):
        path = Path(path)
        if path != run:
            raise ValueError("Synthetic test only owns its new_run directory")
        return path

    monkeypatch.setattr(runner, "own_run", own_run)
    cases, fits, units, pins = [], {}, {}, {}
    target = tmp_path / "target_observations_must_not_be_decoded.json"
    target.write_text("Not JSON: target labels are forbidden during learning")
    target_ref = runner.ref(target)
    pins[target_ref["path"]] = target_ref["sha256"]
    for index in range(3):
        cid = "synthetic_" + str(index)
        request = {"compound_id": cid, "human": {"weight_kg": 70.}, "evidence": []}
        request_ref = write(base / cid / "request.json", request)
        case = {"case_id": cid, "compound_id": cid, "drug": cid,
                "study_id": cid + "_target", "cluster_id": cid + "_target",
                "reference_source": target_ref, "sampling_sources": {},
                "requests": {"evidence": {"development": request_ref, "target": request_ref}}}
        for sampling, count in (("full", 7), ("five", 5)):
            doc_ref = write(base / cid / (sampling + ".json"), {
                "compound_id": cid, "study_id": cid + "_development",
                "observations": [{"time_min": k + 1., "concentration_umol_per_l": 1.}
                                 for k in range(count)]})
            case["sampling_sources"][sampling] = doc_ref
            for formula in runner.FORMULAS:
                mapping = write(base / cid / sampling / (formula + ".json"), {
                    "eligible": True, "decision": {"lipophilicity": 1., "clearance_scale": 1.}})
                fits[runner.key(case, sampling, "ours") + "__" + formula] = {"mapping": mapping}
            for method in runner.CLASSICAL[:2]:
                folder = base / cid / sampling / method
                fit_ref = write(folder / "fit.json", {"development_parameters": {
                    "CL_L_h": 1. + index, "Vc_L": 2., "Q_L_h": 3., "Vp_L": 4.}})
                pred_ref = write(folder / "prediction.json", {"time_min": [0, 1], "concentration": [0, 1]})
                units[runner.key(case, sampling, method)] = {
                    "status": "predicted", "prediction": pred_ref, "request": request_ref, "fit": fit_ref}
        cases.append(case)
    write(base / "protocol.json", {"cases": cases, "source_pins": pins})
    write(base / "summary.json", {"fits": fits, "units": units})
    return base, run, cases


def test_prepare_pins_teachers_and_subsets_without_decoding_target(batch):
    base, run, cases = batch
    before = {str(p): p.read_bytes() for p in base.rglob("*") if p.is_file()}
    protocol = runner.prepare(run, workers=14, rounds=1)
    assert len(protocol["teacher_rows"]) == len(cases) * 2 * 2
    assert protocol["learning"]["unique_development_compounds"] == 3
    assert protocol["learning"]["all_target_outcomes_excluded_from_learning"] is True
    for row in protocol["teacher_rows"]:
        assert row["training_source"] == next(c for c in cases if c["compound_id"] == row["compound_id"])["sampling_sources"][row["sampling"]]
        for field in ("training_source", "request_source", "teacher_source"):
            assert protocol["source_pins"][row[field]["path"]] == row[field]["sha256"]
    assert {str(p): p.read_bytes() for p in base.rglob("*") if p.is_file()} == before
    teacher = ROOT / protocol["teacher_rows"][0]["teacher_source"]["path"]
    teacher.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="Changed"):
        runner.verify(protocol["source_pins"])
    with pytest.raises(FileExistsError):
        runner.prepare(run)


def test_training_uses_compound_folds_for_both_sampling_settings(batch, monkeypatch):
    import numpy as np
    import torch
    from sklearn import ensemble
    from pbpk_kb.blueprint_learning import curve_prior

    _, run, cases = batch
    protocol = runner.prepare(run, rounds=1)
    state = runner.read(run / "state.json")
    controller = runner.Controller(run, protocol, state)
    monkeypatch.setattr(controller, "check", lambda **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls = []

    def fit_prior(rows, *, excluded_compound, algorithm, **kwargs):
        retained = [r for r in rows if r["compound_id"] != excluded_compound]
        calls.append((excluded_compound, algorithm, retained))
        return {"training_compound_ids": sorted({r["compound_id"] for r in retained})}

    class DummyRF:
        def __init__(self, **kwargs):
            pass

        def fit(self, features, labels):
            assert features.shape == (4, 2) and labels.shape == (4, 4)
            return self

    monkeypatch.setattr(curve_prior, "fit_prior", fit_prior)
    monkeypatch.setattr(curve_prior, "curve_features", lambda doc, req: np.array([
        int(doc["compound_id"].rsplit("_", 1)[1]), len(doc["observations"])]))
    monkeypatch.setattr(ensemble, "RandomForestRegressor", DummyRF)
    # Keep this test's dummy estimator out of pickle; the real serializer is unrelated to fold isolation.
    import joblib
    monkeypatch.setattr(joblib, "dump", lambda value, path: Path(path).write_bytes(b"synthetic bundle"))
    bundle = runner.train_priors(controller)
    assert len(calls) == len(cases) * 2
    for excluded, _, rows in calls:
        assert all(r["compound_id"] != excluded for r in rows)
        assert {r["sampling"] for r in rows} == {"full", "five"}
        assert len(rows) == 8
        assert excluded not in bundle["classical"][excluded]["training_compound_ids"]
    assert all(v["heldout_compound_teacher_used"] is False for v in bundle["folds"].values())


def test_scoring_is_gated_before_target_reader(batch, monkeypatch):
    import report_blueprint_validation as report
    _, run, _ = batch
    protocol = runner.prepare(run, rounds=1)
    controller = runner.Controller(run, protocol, runner.read(run / "state.json"))
    monkeypatch.setattr(report, "_case_reference", lambda *args: pytest.fail("Target was decoded before locking"))
    with pytest.raises(ValueError, match="before all prediction"):
        runner.score_round(controller, 1, {"units": {}})


def score_records(cases):
    rows = []
    for case in cases:
        for sampling in runner.SAMPLING:
            for method in runner.METHODS:
                error = 1.1 if method in ("pbpk_rf", "pbpk_mlp") else 2.
                rows.append({"case_id": case["case_id"], "sampling": sampling, "method": method,
                    "attempt_status": "predicted", "scoring_status": "scored",
                    "curve_metrics": {"gmfe": error}, "endpoints": {
                        "auc": {"reference": None if case["case_id"] == "synthetic_2" else 1.,
                                "reference_reason": "Original AUC window is ambiguous" if case["case_id"] == "synthetic_2" else "",
                                "value": error, "error": None if case["case_id"] == "synthetic_2" else error},
                        "cmax": {"reference": 1., "value": error, "error": error}}})
    return rows


def test_promotion_uses_existing_reference_set_and_never_drops_failures(batch):
    _, _, cases = batch
    rows = score_records(cases)
    outcome = runner.aggregate(rows, cases)
    assert set(outcome["development_goal_methods"]) == {"pbpk_rf", "pbpk_mlp"}
    damaged = deepcopy(rows)
    row = next(r for r in damaged if r["method"] == "pbpk_rf" and r["sampling"] == "five")
    row.update(attempt_status="failed", scoring_status="failed", curve_metrics={"gmfe": None})
    row["endpoints"]["auc"]["error"] = None
    assert "pbpk_rf" not in runner.aggregate(damaged, cases)["development_goal_methods"]


def rejected_run(batch):
    _, run, _ = batch
    runner.prepare(run, rounds=1)
    record = {"round": 1, "statistics": {"development_goal_methods": []}, "records": [
        {"attempt_status": "failed", "failure": "synthetic native failure", "endpoints": {
            "auc": {"value": None, "reference": None, "reason": "Not reported"}}}], "search_history": []}
    write(run / "iterations.json", {"version": runner.VERSION, "rounds": {"1": record}})
    return run, record


def test_cleanup_rejects_round_directory_symlink(batch, monkeypatch):
    run, _ = rejected_run(batch)
    victim = run.parent / "unrelated_prior_results"
    (victim / "work").mkdir(parents=True)
    (victim / "work" / "must_keep.txt").write_text("old artifact")
    (run / "round_01").symlink_to(victim, target_is_directory=True)
    removals = []
    monkeypatch.setattr(runner.shutil, "rmtree", lambda path: removals.append(path))
    with pytest.raises(ValueError, match="[Ss]ymlink|[Uu]nsafe|[Cc]leanup"):
        runner.cleanup_round(run, 1)
    assert removals == [] and (victim / "work" / "must_keep.txt").exists()


def test_cleanup_preserves_committed_metrics_and_prior_sources(batch):
    base, _, _ = batch
    run, record = rejected_run(batch)
    work = run / "round_01" / "work"
    work.mkdir(parents=True)
    (work / "regenerable.pkml").write_text("reproducible")
    old_hash = runner.sha256(base / "summary.json")
    assert runner.cleanup_round(run, 1) > 0
    assert not work.exists()
    retained = runner.read(run / "iterations.json")["rounds"]["1"]
    assert retained["records"] == record["records"]
    assert retained["cleanup"]["status"] == "complete"
    assert runner.sha256(base / "summary.json") == old_hash
    assert runner.cleanup_round(run, 1) == 0


def test_rejected_round_keeps_every_endpoint_failure_and_classical_fit(batch, monkeypatch):
    import report_blueprint_validation as report
    _, run, cases = batch
    protocol = runner.prepare(run, rounds=1)
    state = runner.read(run / "state.json")
    state["learning_artifact"] = write(run / "synthetic_learning.json", {"trained": True})
    controller = runner.Controller(run, protocol, state)

    class Store:
        def __init__(self, root):
            pass

        def unchanged(self):
            pass

    def reference(case, store):
        endpoints = {name: {"value": 1., "unit": unit, "location": "Original synthetic table"}
                     for name, unit in report.UNITS.items()}
        if case["case_id"] == "synthetic_2":
            endpoints["auc"].update(value=None, reason="Original AUC window is ambiguous")
        return {}, {"time_h": [1., 2.], "unit": "ng/mL", "location": "Original synthetic figure"}, endpoints

    def score(case, clinical, curve, endpoints, unit, store):
        return {"endpoints": {name: {"value": 1.4, "reference": original["value"],
            "error": 1.4 if original["value"] else None, "unit": original["unit"],
            "reference_reason": original.get("reason", ""), "weak_tail": False}
            for name, original in endpoints.items()}, "curve_metrics": {"gmfe": 1.4, "n": 2}}, {}, 400.

    monkeypatch.setattr(report, "Store", Store)
    monkeypatch.setattr(report, "_case_reference", reference)
    monkeypatch.setattr(report, "_score_prediction", score)
    current = {"all_predictions_locked": True, "predictions_locked_at": runner.now(),
               "configuration": protocol["rounds"][0], "fits": {}, "units": {}}
    fit_ref = write(run / "round_01" / "work" / "classical_fit.json", {
        "attempts": [{"start": k, "objective": .2, "converged": k != 0} for k in range(16)],
        "target_outcomes_used": False})
    for case in cases:
        for sampling in runner.SAMPLING:
            for method in runner.METHODS:
                unit = {"status": "predicted"}
                if method == "two_compartment_learned":
                    unit["fit"] = fit_ref
                current["units"][runner.key(case, sampling, method)] = unit
    failed_id = runner.key(cases[0], "five", "pbpk_rf")
    current["units"][failed_id] = {"status": "failed", "failure": "synthetic native failure"}
    state["rounds"]["1"] = current
    result = runner.score_round(controller, 1, current)
    assert len(result["records"]) == len(cases) * 2 * len(runner.METHODS)
    assert len(result["endpoint_rows"]) == len(result["records"]) * 11
    failure = next(r for r in result["endpoint_rows"] if r["unit_id"] == failed_id and r["endpoint"] == "cmax")
    assert failure["clinical_reference"] == 1. and failure["prediction"] is None
    assert failure["prediction_missing_reason"] == "synthetic native failure"
    missing = [r for r in result["endpoint_rows"] if r["case_id"] == "synthetic_2" and r["endpoint"] == "auc"]
    assert len(missing) == 2 * len(runner.METHODS)
    assert all(r["clinical_reference"] is None and r["reference_missing_reason"] for r in missing)
    assert len(result["learned_classical_fits"]) == len(cases) * 2
    assert all(len(v["attempts"]) == 16 for v in result["learned_classical_fits"].values())
    assert result["statistics"]["development_goal_methods"] == []
    runner.cleanup_round(run, 1)
    retained = runner.read(run / "iterations.json")["rounds"]["1"]
    assert retained["endpoint_rows"] == result["endpoint_rows"]
    assert retained["learned_classical_fits"] == result["learned_classical_fits"]


def test_terminal_resume_checks_sources_before_removing_pause_marker(batch, monkeypatch):
    _, run, _ = batch
    protocol = runner.prepare(run, rounds=1)
    marker = run / "PAUSE"
    marker.write_text("user pause")
    calls = []
    monkeypatch.setattr(runner, "execute", lambda path: calls.append(path))
    monkeypatch.setattr(sys, "argv", ["run_blueprint_learning.py", "resume", "--run-dir", str(run)])
    runner.main()
    assert calls == [run] and not marker.exists()
    marker.write_text("pause after source change")
    teacher = ROOT / protocol["teacher_rows"][0]["teacher_source"]["path"]
    teacher.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="Changed"):
        runner.main()
    assert marker.exists() and calls == [run]


def test_learned_classical_recovers_synthetic_pk_and_preserves_dose_auc(tmp_path):
    """Exercise real fitting/export against an independent matrix exponential."""
    import math
    import numpy as np
    from scipy.integrate import quad
    from scipy.linalg import expm
    import blueprint_validation_classical as classical

    if not tmp_path.resolve().is_relative_to(ROOT / "runtime" / "test-tmp"):
        pytest.skip("This numerical test requires --basetemp under project runtime/test-tmp")
    parameters = np.asarray([5., 10., 15., 30.])  # CL, Vc, Q, Vp
    weight, mw = 70., 194.19
    compound = "synthetic:learned-pk"

    def independent(values, times, dose, duration, start):
        cl, vc, q, vp = values
        system = np.asarray([[-(cl+q)/vc, q/vp], [q/vc, -q/vp]])
        augmented = np.zeros((3,3)); augmented[:2,:2] = system
        augmented[0,2] = dose/duration
        output = []
        for time_h in times:
            elapsed = max(0., time_h-start)
            amount = (expm(augmented*min(elapsed,duration)) @ np.asarray([0.,0.,1.]))[:2]
            if elapsed > duration:
                amount = expm(system*(elapsed-duration)) @ amount
            output.append(amount[0]/vc)
        return np.asarray(output)

    human = {"age_years":30., "weight_kg":weight, "height_cm":175., "sex":"MALE", "population":"European_ICRP_2002"}
    dosing = {"route":"intravenous", "dose_mg":10., "infusion_minutes":30., "start_h":.2}
    source = {"path":"synthetic/input","sha256":"a"*64}
    evidence_values = [("molecular_weight",mw,"g/mol"),("logp",-.1,"Log Units"),
        ("fraction_unbound",.7,"fraction"),("total_clearance",5/(weight*.06),"ml/min/kg")]
    request = {"compound_id":compound,"human":human,"dosing":dosing,
        "evidence":[{"compound_id":compound,"parameter":name,"value":value,"unit":unit,"source":source}
                    for name,value,unit in evidence_values],
        "target":{"window_start_h":0.,"window_end_h":24.}}
    times = np.asarray([.25,.4,.7,.9,1.5,3.,6.,12.,24.])
    measured = independent(parameters,times,10.,.5,.2)
    document = {"compound_id":compound,"study_id":"synthetic-development-only",
        "role":"calibration_data_reconstruction","matrix":"plasma","time_unit":"min","unit":"umol/l",
        "human":deepcopy(human),"dosing":deepcopy(dosing),"molecular_weight_g_mol":mw,
        "observations":[{"time_min":float(t*60),"concentration_umol_per_l":float(c*1000/mw)}
                        for t,c in zip(times,measured)]}
    req_ref = write(tmp_path/"request.json",request)
    dev_ref = write(tmp_path/"development.json",document)
    target_human = {**human,"weight_kg":84.}
    target_dosing = {"route":"intravenous","dose_mg":20.,"infusion_minutes":60.,"start_h":.1}
    nominal = [0.,.1,.2,.5,1.1,2.,4.,12.,24.]
    case = {"case_id":"synthetic_numeric","compound_id":compound,
        "requests":{"evidence":{"development":req_ref}},"sampling_sources":{"full":dev_ref},
        "target_request_metadata":{"human":target_human,"dosing":target_dosing,"window_end_h":24.},
        "nca_policy":{"sampling_time_h":nominal}}

    class ExactPrior:
        def predict(self, values):
            return np.tile(np.log(parameters/weight),(len(values),1))

    bundle = {"classical":{compound:{"model":ExactPrior(),"training_compound_ids":["synthetic:other"]}}}
    first = runner.learned_classical(case,"full",bundle,tmp_path/"dose20")
    fitted = runner.checked(first["fit"])
    assert len(fitted["attempts"]) == 16
    assert fitted["target_outcomes_used"] is False
    assert fitted["selected_objective"] < 1e-14
    np.testing.assert_allclose(fitted["development_parameters"],parameters,rtol=1e-5,atol=1e-6)
    ratio = 84./70.
    target_parameters = parameters*np.asarray([ratio**.75,ratio,ratio**.75,ratio])
    np.testing.assert_allclose(fitted["target_parameters"],target_parameters,rtol=1e-5,atol=1e-6)
    predicted = runner.checked(first["prediction"])
    hours = np.asarray(predicted["time_min"])/60
    concentration = np.asarray(predicted["concentration"])*mw/1000
    np.testing.assert_allclose(np.interp(nominal,hours,concentration),
        independent(target_parameters,nominal,20.,1.,.1),rtol=1e-5,atol=1e-9)
    args = {"dose_mg":20.,"infusion_h":1.,"start_h":.1}
    concentration_fn = lambda t: float(classical.concentration_mg_l(fitted["target_parameters"], [t], **args)[0])
    auc = quad(concentration_fn,.1,1.1,epsabs=1e-10)[0]+quad(concentration_fn,1.1,np.inf,epsabs=1e-10)[0]
    assert auc == pytest.approx(20./fitted["target_parameters"][0],rel=1e-8)
    checksum = runner.sha256(ROOT/first["prediction"]["path"])
    assert runner.learned_classical(case,"full",bundle,tmp_path/"dose20") == first
    assert runner.sha256(ROOT/first["prediction"]["path"]) == checksum
    higher = deepcopy(case);higher["target_request_metadata"]["dosing"]["dose_mg"] = 40.
    second = runner.learned_classical(higher,"full",bundle,tmp_path/"dose40")
    doubled = runner.checked(second["prediction"])
    np.testing.assert_array_equal(doubled["time_min"],predicted["time_min"])
    np.testing.assert_allclose(doubled["concentration"],2*np.asarray(predicted["concentration"]),rtol=1e-8,atol=1e-12)
    assert first["resources"]["actual_new_native_calls"] == second["resources"]["actual_new_native_calls"] == 0
