#!/usr/bin/env python3
"""Multi-compound clinical-curve learning and bounded native PBPK development.

This is an iterative development experiment, not a new blinded evaluation.
Every rejected round keeps its configuration, all case/endpoint scores and
search history before its reproducible simulation artifacts are removed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

BASE = ROOT / "experiments/blueprint_design/validation_20260929"
DEFAULT = ROOT / "experiments/blueprint_design/learning_20260929"
VERSION = "clinical-curve-learned-initialization-1"
NATIVE = ("pbpk_unlearned", "pbpk_rf", "pbpk_mlp")
CLASSICAL = ("two_compartment", "two_compartment_prior", "two_compartment_learned")
METHODS = NATIVE + CLASSICAL
FORMULAS = ("PKSimStandard", "RodgersRowland")
SAMPLING = ("full", "five")
LIMIT = 10 * 1024**3


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(inside(path, ROOT, exists=True).read_text())


def ref(path):
    path = inside(path, ROOT, exists=True)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def checked(record):
    path = inside(record["path"], ROOT, exists=True)
    if sha256(path) != record["sha256"]:
        raise ValueError("Changed frozen source: " + str(path))
    return read(path)


def verify(pins):
    for path, digest in pins.items():
        if sha256(inside(path, ROOT, exists=True)) != digest:
            raise ValueError("Changed source/code: " + path)


def key(case, sampling, method):
    return "__".join((case["case_id"], sampling, method))


def own_run(path):
    raw = Path(path) if Path(path).is_absolute() else ROOT / path
    if raw.is_symlink():
        raise ValueError("Learning run aliases are not allowed")
    run = inside(path, ROOT)
    allowed = ROOT / "experiments/blueprint_design"
    if run.parent != allowed or run.is_symlink():
        raise ValueError("Learning runs must be direct, non-symlink blueprint experiment folders")
    return run


def prepare(run, workers=14, rounds=3):
    run = own_run(run)
    if run.exists():
        raise FileExistsError("Keep existing run; use run/status to resume")
    if not 1 <= workers <= 14 or not 1 <= rounds <= 3:
        raise ValueError("Use 1–14 native workers and 1–3 development rounds")
    protocol, summary = read(BASE / "protocol.json"), read(BASE / "summary.json")
    verify(protocol["source_pins"])
    rows, pins = [], dict(protocol["source_pins"])
    for case in protocol["cases"]:
        for sampling in SAMPLING:
            training = case["sampling_sources"][sampling]
            checked(training)
            request_ref = case["requests"]["evidence"]["development"]
            checked(request_ref)
            for formula in FORMULAS:
                fit = summary["fits"][key(case, sampling, "ours") + "__" + formula]
                mapping = checked(fit["mapping"])
                if not mapping["eligible"]:
                    raise ValueError("Teacher fit is ineligible: " + case["case_id"])
                rows.append({"compound_id": case["compound_id"], "sampling": sampling,
                    "partition_method": formula, "training_source": training,
                    "request_source": request_ref, "teacher_source": fit["mapping"]})
                for record in (training, request_ref, fit["mapping"]):
                    pins[record["path"]] = record["sha256"]
            for method in CLASSICAL[:2]:
                unit = summary["units"][key(case, sampling, method)]
                for field in ("prediction", "fit", "request"):
                    checked(unit[field]); pins[unit[field]["path"]] = unit[field]["sha256"]
    code = [Path(__file__), *sorted((ROOT / "src/pbpk_kb/blueprint_learning").glob("*.py")),
        ROOT / "scripts/report_blueprint_validation.py", ROOT / "scripts/blueprint_validation_classical.py",
        ROOT / "scripts/run_classical_pk_comparator.py"]
    for path in code + [BASE / "protocol.json", BASE / "summary.json"]:
        pins[str(path.relative_to(ROOT))] = sha256(path)
    configs = [{"round": 1, "max_evaluations_per_formula": 48, "prior_weight": .1,
                "reason": "Learned initialization and a larger shared search budget"},
               {"round": 2, "max_evaluations_per_formula": 80, "prior_weight": .1,
                "reason": "Check search sufficiency under the same parameterization"},
               {"round": 3, "max_evaluations_per_formula": 80, "prior_weight": .03,
                "reason": "Development sensitivity to clearance regularization; not a physiological interval"}][:rounds]
    document = {"version": VERSION, "created_at": now(), "source_pins": pins,
        "source_protocol": ref(BASE / "protocol.json"), "source_summary": ref(BASE / "summary.json"),
        "teacher_rows": rows, "cases": protocol["cases"], "methods": list(METHODS),
        "sampling": list(SAMPLING), "formulas": list(FORMULAS), "rounds": configs,
        "workers": workers, "native_threads": 2, "native_timeout_seconds": 180,
        "monitor_seconds": 1200, "directory_limit_bytes": LIMIT,
        "learning": {"algorithms": ["rf", "mlp"], "seed": 20260929,
            "epochs": 300, "outer_split": "leave-one-compound-out, all studies and subsets grouped",
            "teacher": "Development-fitted PBPK inputs, not measured pharmacokinetic parameters",
            "unique_development_compounds": len({r["compound_id"] for r in rows}),
            "unique_development_studies": len({checked(c["sampling_sources"]["full"])["study_id"] for c in protocol["cases"]}),
            "all_target_outcomes_excluded_from_learning": True,
            "classical_shared_learning": "Same curve features; leave-compound-out RF predicts four starting parameters"},
        "interpretation": "Previously viewed ten-drug development cohort; iterative selection is not independent testing",
        "future_independent_test": {"used": False, "available_in_this_run": False,
            "policy": "Acquire untouched studies only after method freeze; never iterate on a final test"},
        "promotion_rule": "Complete predictions; full AND five drug-equal curve GMFE below every classical comparator, and AUC/Cmax GMFE no worse than every comparator; exploratory only",
        "retention": "Keep every round's scores, method, source/code hashes, fit candidate records and failure diagnostics; remove rejected reproducible work only after ledger commit"}
    run.mkdir()
    save(run / "protocol.json", document)
    save(run / "state.json", {"version": VERSION, "status": "prepared", "created_at": now(),
        "protocol": ref(run / "protocol.json"), "rounds": {}, "learning": {}, "monitor": {}})
    return document


def load_rows(protocol):
    rows = []
    for row in protocol["teacher_rows"]:
        item = deepcopy(row)
        item.update(document=checked(row["training_source"]), request=checked(row["request_source"]),
                    decision=checked(row["teacher_source"])["decision"])
        item["source_refs"] = {"development": row["training_source"], "teacher": row["teacher_source"],
                               "request": row["request_source"]}
        item["request"]["partition_method"] = item["partition_method"]
        rows.append(item)
    return rows


class Paused(RuntimeError):
    pass


class Controller:
    def __init__(self, run, protocol, state):
        self.run, self.protocol, self.state = run, protocol, state
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.halted = threading.Event()
        self.last_size = self.last_hash = 0.

    def persist(self, **values):
        with self.lock:
            self.state.update(values, updated_at=now())
            save(self.run / "state.json", self.state)

    def pause(self, reason):
        self.halted.set()
        self.persist(status="paused", pause_reason=str(reason))

    def check(self, force=False):
        with self.lock:
            if self.halted.is_set() or (self.run / "PAUSE").exists():
                raise Paused(self.state.get("pause_reason", "Pause requested"))
            timestamp = time.monotonic()
            if force or timestamp-self.last_size >= 60:
                size = 0
                for path in self.run.rglob("*"):
                    try:
                        if path.is_symlink():
                            raise Paused("Unexpected symlink in learning run")
                        if path.is_file(): size += path.stat().st_size
                    except FileNotFoundError:
                        continue
                monitor = self.state["monitor"]
                monitor.update(bytes=size, growth_bytes=size-monitor.get("bytes", 0), checked_at=now())
                self.last_size = timestamp
                if size > LIMIT:
                    self.pause("Run exceeds 10 GiB; no new native calls submitted")
                    raise Paused("Directory size limit")
            if force or timestamp-self.last_hash >= 1200:
                verify(self.protocol["source_pins"])
                checked(self.state["protocol"])
                if self.state.get("learning_artifact"):
                    artifact = self.state["learning_artifact"]
                    if sha256(ROOT / artifact["path"]) != artifact["sha256"]:
                        raise Paused("Learning artifact changed")
                self.state["monitor"]["hashes_checked_at"] = now()
                self.last_hash = timestamp
            self.persist()

    def watch(self):
        while not self.closed.wait(15):
            try:
                import psutil
                parent = psutil.Process(os.getpid())
                children = [parent] + parent.children(recursive=True)
                rss, count = 0, 0
                for process in children:
                    try:
                        if process.uids().real == os.getuid():
                            rss += process.memory_info().rss; count += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                with self.lock:
                    self.state["monitor"].update(own_processes=count, own_rss_bytes=rss)
                self.check()
            except Exception as exc:
                self.pause(str(exc))
                return


def train_priors(controller):
    import joblib
    import numpy as np
    import torch
    from sklearn.ensemble import RandomForestRegressor
    from pbpk_kb.blueprint_learning.curve_prior import fit_prior, curve_features
    run, protocol = controller.run, controller.protocol
    bundle_path = run / "learning.joblib"
    if controller.state.get("learning_artifact"):
        artifact = controller.state["learning_artifact"]
        if sha256(bundle_path) != artifact["sha256"]:
            raise ValueError("Learning checkpoint changed")
        return joblib.load(bundle_path)
    rows = load_rows(protocol)
    old = checked(protocol["source_summary"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    bundle = {"version": VERSION, "device": device, "models": {}, "classical": {}, "folds": {}}
    ids = sorted({row["compound_id"] for row in rows})
    controller.persist(status="training_shared_priors")
    for excluded in ids:
        for algorithm in ("rf", "mlp"):
            controller.check()
            started = time.monotonic()
            model = fit_prior(rows, excluded_compound=excluded, algorithm=algorithm,
                              seed=20260929, device=device, epochs=protocol["learning"]["epochs"])
            bundle["models"][excluded+"__"+algorithm] = model
            controller.state["learning"][excluded+"__"+algorithm] = {
                "status": "trained", "seconds": time.monotonic()-started,
                "training_compounds": [cid for cid in ids if cid != excluded]}
            controller.persist()
        features, labels = [], []
        for case in protocol["cases"]:
            if case["compound_id"] == excluded: continue
            request = checked(case["requests"]["evidence"]["development"])
            for sampling in SAMPLING:
                doc = checked(case["sampling_sources"][sampling])
                fit = checked(old["units"][key(case, sampling, "two_compartment")]["fit"])
                features.append(curve_features(doc, request))
                weight = request["human"]["weight_kg"]
                labels.append(np.log([fit["development_parameters"][name]/weight
                    for name in ("CL_L_h", "Vc_L", "Q_L_h", "Vp_L")]))
        model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, max_features=.8,
                                      random_state=20260929, n_jobs=2)
        model.fit(np.asarray(features), np.asarray(labels))
        bundle["classical"][excluded] = {"model": model, "training_compound_ids": [cid for cid in ids if cid != excluded]}
        bundle["folds"][excluded] = {"excluded_compound": excluded,
            "training_compound_ids": [cid for cid in ids if cid != excluded],
            "heldout_compound_teacher_used": False, "target_outcomes_used": False}
    temp = bundle_path.with_suffix(".tmp")
    joblib.dump(bundle, temp); temp.replace(bundle_path)
    controller.persist(learning_artifact=ref(bundle_path), learning_completed_at=now())
    return bundle


def learned_classical(case, sampling, bundle, output):
    """A pooled-data initialization control; same 16 starts/400-step fitting cap."""
    import numpy as np
    from scipy.optimize import least_squares
    import blueprint_validation_classical as classical
    from pbpk_kb.blueprint_learning.curve_prior import curve_features
    if (output / "unit.json").exists():
        unit = read(output / "unit.json")
        for field in ("prediction", "request", "fit"): checked(unit[field])
        return unit
    output.mkdir(parents=True, exist_ok=True)
    request = checked(case["requests"]["evidence"]["development"])
    doc = checked(case["sampling_sources"][sampling])
    weight, mw = request["human"]["weight_kg"], doc["molecular_weight_g_mol"]
    times = np.asarray([row["time_min"]/60 for row in doc["observations"]])
    observed = np.asarray([row["concentration_umol_per_l"]*mw/1000 for row in doc["observations"]])
    kwargs = classical.dosing_kwargs(request["dosing"])
    starts, lower, upper, _ = classical._initial_parameters(times, observed, kwargs, weight, case["case_id"])
    learned = bundle["classical"][case["compound_id"]]
    if case["compound_id"] in learned["training_compound_ids"]:
        raise ValueError("Classical learning fold contains heldout compound")
    proposed = np.exp(learned["model"].predict(curve_features(doc, request)[None])[0])*weight
    starts[-1] = np.clip(proposed, lower*1.00001, upper/1.00001)
    attempts, best, started = [], None, time.monotonic()
    for index, start in enumerate(starts):
        calls = 0
        def residual(parameters):
            nonlocal calls
            calls += 1
            return classical._residual_vector(parameters, times, observed, kwargs, None)
        try:
            result = least_squares(residual, np.log(start),
                bounds=(np.log(lower), np.log(upper)),
                max_nfev=400, ftol=1e-10, xtol=1e-10, gtol=1e-8)
            objective = float(np.dot(result.fun, result.fun))
            if not math.isfinite(objective) or not np.isfinite(result.x).all():
                raise ArithmeticError("Nonfinite learned classical fit")
            attempts.append({"start": index, "initial": start.tolist(), "objective": objective,
                "nfev": int(result.nfev), "converged": bool(result.success),
                "message": str(result.message), "parameters": np.exp(result.x).tolist(),
                "residual_calls_including_finite_differences": calls})
            if best is None or objective < best[0]:
                best = (objective, np.exp(result.x), bool(result.success))
        except (ValueError, ArithmeticError, FloatingPointError) as exc:
            attempts.append({"start": index, "initial": start.tolist(), "error": str(exc),
                "converged": False, "residual_calls_including_finite_differences": calls})
    if best is None:
        save(output / "fit.json", {"status": "failed", "attempts": attempts,
            "failure": "No finite learned classical fit", "target_outcomes_used": False})
        raise ValueError("No finite learned classical fit")
    target = case["target_request_metadata"]
    ratio = target["human"]["weight_kg"]/weight
    parameters = best[1]*np.asarray([ratio**.75, ratio, ratio**.75, ratio])
    hours, dosing = classical._target_grid(target, case["nca_policy"])
    concentrations = classical.concentration_mg_l(parameters, hours, **dosing)
    if not np.isfinite(concentrations).all() or np.any(concentrations < 0):
        raise ValueError("Invalid learned classical trajectory")
    alpha, beta, a, b = classical.classical.macro_parameters(parameters)
    analytic_auc = dosing["dose_mg"]/parameters[1]*(a/alpha+b/beta)
    auc_error = abs(analytic_auc/(dosing["dose_mg"]/parameters[0])-1)
    if auc_error > 1e-8 or not (np.diff(hours) > 0).all():
        raise ArithmeticError("Analytical mass balance or time grid check failed")
    target_request = deepcopy(request)
    target_request.update(human=target["human"], dosing=target["dosing"])
    target_request["target"].update(window_start_h=0., window_end_h=target["window_end_h"], terminal_fit=None)
    save(output / "request.json", target_request)
    save(output / "prediction_locked.json", {"time_min": (hours*60).tolist(),
        "concentration": (concentrations*1000/mw).tolist(), "time_unit": "min", "unit": "umol/l"})
    save(output / "fit.json", {"method": "two_compartment_learned", "attempts": attempts,
        "training_compound_ids": learned["training_compound_ids"], "excluded_compound": case["compound_id"],
        "selected_objective": best[0], "development_parameters": best[1].tolist(),
        "target_parameters": parameters.tolist(), "converged": best[2], "target_outcomes_used": False,
        "analytic_auc_mass_balance_relative_error": float(auc_error)})
    unit = {"status": "predicted", "prediction": ref(output / "prediction_locked.json"),
        "request": ref(output / "request.json"), "fit": ref(output / "fit.json"),
        "resources": {"wall_seconds": time.monotonic()-started, "actual_new_native_calls": 0,
            "analytic_cpu_residual_calls": sum(a["residual_calls_including_finite_differences"] for a in attempts),
            "optimizer_nfev": sum(a.get("nfev", 0) for a in attempts)}}
    save(output / "unit.json", unit)
    return unit


def aggregate(records, cases):
    from report_blueprint_validation import clustered_geomean, paired_ratio, positive
    clusters = {case["case_id"]: case["cluster_id"] for case in cases}
    metrics, ratios = {}, {}
    for sampling in SAMPLING:
        metrics[sampling], ratios[sampling] = {}, {}
        for method in METHODS:
            rows = [r for r in records if r["sampling"] == sampling and r["method"] == method]
            values = {r["case_id"]: r.get("curve_metrics", {}).get("gmfe") for r in rows}
            finite = {cid: value for cid, value in values.items() if positive(value)}
            endpoints = {}
            for endpoint in ("auc", "cmax"):
                available = sorted(r["case_id"] for r in rows
                    if positive(r.get("endpoints", {}).get(endpoint, {}).get("reference")))
                scored = {r["case_id"]: r.get("endpoints", {}).get(endpoint, {}).get("error") for r in rows
                    if positive(r.get("endpoints", {}).get(endpoint, {}).get("error"))}
                errors = list(scored.values())
                endpoints[endpoint] = {"n": len(errors), "reference_case_ids": available,
                    "scored_case_ids": sorted(scored),
                    "gmfe": math.exp(sum(map(math.log, errors))/len(errors)) if errors else None}
            metrics[sampling][method] = {"curve": clustered_geomean(finite, {cid: clusters[cid] for cid in finite}),
                "curve_by_case": values, "failures": len(cases)-len(finite), "endpoints": endpoints}
        for method in NATIVE:
            ratios[sampling][method] = {other: paired_ratio(
                metrics[sampling][method]["curve_by_case"], metrics[sampling][other]["curve_by_case"],
                clusters, expected_ids=list(clusters)) for other in CLASSICAL}
    promoted = []
    for method in NATIVE[1:]:
        accepted = True
        for sampling in SAMPLING:
            ours = metrics[sampling][method]
            if ours["failures"]: accepted = False
            for other in CLASSICAL:
                baseline = metrics[sampling][other]
                a, b = ours["curve"]["estimate"], baseline["curve"]["estimate"]
                if baseline["failures"] or a is None or b is None or a >= b: accepted = False
                for endpoint in ("auc", "cmax"):
                    first, second = ours["endpoints"][endpoint], baseline["endpoints"][endpoint]
                    if (not second["reference_case_ids"] or first["reference_case_ids"] != second["reference_case_ids"]
                            or first["scored_case_ids"] != second["reference_case_ids"]
                            or second["scored_case_ids"] != second["reference_case_ids"]
                            or first["gmfe"] is None or second["gmfe"] is None or first["gmfe"] > second["gmfe"]):
                        accepted = False
        if accepted: promoted.append(method)
    return {"metrics": metrics, "paired_ratios": ratios, "development_goal_methods": promoted,
            "independent_superiority_established": False}


def score_round(controller, number, current):
    from report_blueprint_validation import Store, _case_reference, _score_prediction, _unit_row, UNITS, CURVE
    if not current.get("all_predictions_locked"):
        raise ValueError("Scoring is forbidden before all prediction attempts are locked")
    store, records, table = Store(ROOT), [], []
    for case in controller.protocol["cases"]:
        clinical, reference_curve, endpoints = _case_reference(case, store)
        for sampling in SAMPLING:
            for method in METHODS:
                unit = current["units"][key(case, sampling, method)]
                record = {"unit_id": key(case, sampling, method), "case_id": case["case_id"],
                    "drug": case["drug"], "study_id": case["study_id"], "cluster_id": case["cluster_id"],
                    "sampling": sampling, "method": method, "attempt_status": unit["status"],
                    "scoring_status": "failed", "resources": unit.get("resources", {}),
                    "prediction_source": unit.get("prediction", {}), "failure": unit.get("failure", "")}
                if unit["status"] == "predicted":
                    scores, _, _ = _score_prediction(case, clinical, reference_curve, endpoints, unit, store)
                    scores.pop("curve", None)
                    record.update(scores, scoring_status="scored")
                records.append(record)
                references = {"curve": reference_curve, "endpoints": endpoints}
                table.extend(_unit_row(record, endpoint, references) for endpoint in (CURVE, *UNITS))
    store.unchanged()
    diagnostics = []
    for fit_id, fit in current["fits"].items():
        mapping = checked(fit["mapping"])
        diagnostics.append({"fit_id": fit_id, "status": mapping["status"], "converged": mapping.get("converged"),
            "decision": mapping.get("decision"), "evaluations": mapping.get("evaluations", []),
            "resource_and_optimizer_record": {name: mapping.get(name) for name in (
                "total_native_calls", "actual_new_calls", "reused_calls", "elapsed_seconds",
                "optimizer", "failed_evaluation_count", "source_pins", "input_sha256")},
            "mapping_sha256": fit["mapping"]["sha256"]})
    classical_diagnostics = {identity: checked(unit["fit"]) for identity, unit in current["units"].items()
                             if identity.endswith("__two_compartment_learned") and unit.get("fit")}
    result = {"round": number, "configuration": current["configuration"], "scored_at": now(),
        "predictions_locked_at": current["predictions_locked_at"], "records": records, "endpoint_rows": table,
        "statistics": aggregate(records, controller.protocol["cases"]), "search_history": diagnostics,
        "learned_classical_fits": classical_diagnostics,
        "protocol_sha256": controller.state["protocol"]["sha256"],
        "interpretation": controller.protocol["interpretation"],
        "learning_artifact": controller.state["learning_artifact"], "all_case_results_retained": True}
    ledger_path = controller.run / "iterations.json"
    ledger = read(ledger_path) if ledger_path.exists() else {"version": VERSION, "rounds": {}}
    ledger["rounds"][str(number)] = result
    save(ledger_path, ledger)
    if result["statistics"]["development_goal_methods"]:
        with (controller.run / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader(); writer.writerows(table)
    current.update(status="scored", promoted_methods=result["statistics"]["development_goal_methods"],
                   record_sha256=sha256(ledger_path), statistics=result["statistics"])
    controller.persist()
    return result


def cleanup_round(run, number):
    """Delete only reproducible artifacts from a committed rejected own round."""
    run = own_run(run)
    protocol, state = read(run / "protocol.json"), read(run / "state.json")
    if protocol.get("version") != VERSION or state.get("version") != VERSION:
        raise ValueError("Not an owned learning run")
    ledger = read(run / "iterations.json")
    record = ledger["rounds"].get(str(number))
    if not record or record["statistics"]["development_goal_methods"]:
        raise ValueError("No committed rejected round; artifacts retained")
    parent = run / f"round_{number:02d}"
    work = parent / "work"
    if parent.is_symlink() or work.is_symlink() or not work.resolve().is_relative_to(run.resolve()):
        raise ValueError("Unsafe cleanup parent or target")
    if not work.exists():
        if record.get("cleanup", {}).get("status") == "planned":
            record["cleanup"].update(status="complete", completed_at=now())
            save(run / "iterations.json", ledger)
        return 0
    if work.is_symlink() or work.resolve().parent != (run / f"round_{number:02d}").resolve():
        raise ValueError("Unsafe cleanup path")
    for path in work.rglob("*"):
        if path.is_symlink(): raise ValueError("Cleanup refuses symlinks")
    size = sum(p.stat().st_size for p in work.rglob("*") if p.is_file())
    receipt = {"completed_at": now(), "removed_bytes": size, "path": str(work.relative_to(ROOT)),
               "reason": "Development goal not reached; full scalar scores and search records retained"}
    # Commit the exact removal scope before mutation; a restart can finish deletion.
    record["cleanup"] = {**receipt, "status": "planned"}
    save(run / "iterations.json", ledger)
    shutil.rmtree(work)
    record["cleanup"]["status"] = "complete"
    save(run / "iterations.json", ledger)
    return size


def execute(run):
    from pbpk_kb.blueprint_learning.curve_prior import predict_prior
    from pbpk_kb.blueprint_learning.native_calibration import fit_mapping, predict_mapping, cleanup_unselected
    run = own_run(run)
    protocol, state = read(run / "protocol.json"), read(run / "state.json")
    with (run / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state["status"] in ("development_goal_reached", "round_limit_reached"):
            return state
        controller = Controller(run, protocol, state)
        controller.check(force=True)
        monitor = threading.Thread(target=controller.watch, daemon=True)
        monitor.start()
        try:
            bundle = train_priors(controller)
            old = checked(protocol["source_summary"])
            for config in protocol["rounds"]:
                number = config["round"]
                current = state["rounds"].setdefault(str(number), {"configuration": config, "fits": {}, "units": {}, "status": "running"})
                if current["status"] in ("scored", "cleaned"):
                    if current.get("promoted_methods"):
                        controller.persist(status="development_goal_reached"); break
                    cleanup_round(run, number)
                    current["status"] = "cleaned"; controller.persist(); continue
                work = run / f"round_{number:02d}" / "work"
                controller.persist(status="running_native_calibration", active_round=number)

                def fit_one(case, sampling, method, formula):
                    identity = key(case, sampling, method) + "__" + formula
                    if identity in current["fits"]:
                        checked(current["fits"][identity]["mapping"]); return
                    controller.check()
                    request = checked(case["requests"]["evidence"]["development"])
                    request["partition_method"] = formula
                    initial = []
                    if method != "pbpk_unlearned":
                        algorithm = method.removeprefix("pbpk_")
                        model = bundle["models"][case["compound_id"]+"__"+algorithm]
                        proposal = predict_prior(model, checked(case["sampling_sources"][sampling]), request, formula)
                        if case["compound_id"] in proposal["training_compound_ids"]:
                            raise ValueError("Heldout compound leaked into learned initializer")
                        initial.append(proposal)
                    folder = work / "fits" / case["case_id"] / sampling / method / formula
                    result = fit_mapping(reference_request=request, training_source=case["sampling_sources"][sampling],
                        output_dir=folder, project_root=ROOT, max_evaluations=config["max_evaluations_per_formula"],
                        initial_points=initial, before_trial=controller.check, prior_weight=config["prior_weight"])
                    mapping = ref(ROOT / result["mapping_path"])
                    with controller.lock:
                        current["fits"][identity] = {"mapping": mapping, "eligible": result["eligible"],
                            "decision": result.get("decision"), "status": result["status"]}
                        controller.persist()
                    cleanup = cleanup_unselected(output_dir=folder, project_root=ROOT)
                    with controller.lock:
                        current["fits"][identity]["cleanup"] = cleanup
                        controller.persist()

                planned = [(case, sampling, method, formula) for case in protocol["cases"]
                    for sampling in SAMPLING for method in NATIVE for formula in FORMULAS]
                with ThreadPoolExecutor(max_workers=protocol["workers"]) as pool:
                    futures = [pool.submit(fit_one, *job) for job in planned]
                    for future in as_completed(futures):
                        try: future.result()
                        except Exception as exc: controller.pause(type(exc).__name__+": "+str(exc))
                controller.check()
                controller.persist(status="locking_predictions")

                def predict_one(case, sampling, method):
                    identity = key(case, sampling, method)
                    if identity in current["units"]:
                        unit = current["units"][identity]
                        if unit.get("prediction"): checked(unit["prediction"])
                        return
                    controller.check()
                    folder = work / "predictions" / case["case_id"] / sampling / method
                    if method in CLASSICAL[:2]:
                        unit = deepcopy(old["units"][identity])
                        unit["reused_frozen_same_input_baseline"] = True
                    elif method == "two_compartment_learned":
                        unit = learned_classical(case, sampling, bundle, folder)
                    else:
                        fits = [current["fits"][identity+"__"+formula] for formula in FORMULAS]
                        valid = [fit for fit in fits if fit["eligible"]]
                        if not valid:
                            unit = {"status": "failed", "failure": "No eligible native development fit"}
                        else:
                            chosen = min(valid, key=lambda fit: fit["decision"]["objective"])
                            mapping = checked(chosen["mapping"])
                            request = checked(case["requests"]["evidence"]["target"])
                            request["partition_method"] = mapping["construction_partition_method"]
                            started = time.monotonic()
                            result = predict_mapping(request=request, mapping_path=ROOT/chosen["mapping"]["path"],
                                output_dir=folder, project_root=ROOT)
                            from pbpk_kb.blueprints.evidence_preserving_calibration import _infrastructure_problem
                            if _infrastructure_problem(result): raise Paused("Prediction infrastructure failure")
                            unit = {"status": result["status"], "mapping": chosen["mapping"],
                                    "request": ref(folder / "request.json")}
                            fit_records = [checked(item["mapping"]) for item in fits]
                            unit["resources"] = {"wall_seconds": time.monotonic()-started+sum(m.get("elapsed_seconds", 0) for m in fit_records),
                                "total_native_calls": 1+sum(m.get("total_native_calls", 0) for m in fit_records),
                                "actual_new_native_calls": result.get("actual_new_calls", 1)+sum(m.get("actual_new_calls", 0) for m in fit_records),
                                "target_execution": result.get("resources", {}),
                                "failed_native_trials": sum(sum(r.get("status") != "ok" for r in m["evaluations"]) for m in fit_records)}
                            if result["status"] == "predicted": unit["prediction"] = ref(folder / "prediction_locked.json")
                            else: unit["failure"] = result
                    with controller.lock:
                        current["units"][identity] = unit
                        controller.persist()

                with ThreadPoolExecutor(max_workers=protocol["workers"]) as pool:
                    futures = [pool.submit(predict_one, case, sampling, method) for case in protocol["cases"]
                               for sampling in SAMPLING for method in METHODS]
                    for future in as_completed(futures):
                        try: future.result()
                        except Exception as exc: controller.pause(type(exc).__name__+": "+str(exc))
                controller.check(force=True)
                if len(current["units"]) != len(protocol["cases"])*len(SAMPLING)*len(METHODS):
                    raise ValueError("Incomplete comparison matrix")
                current.update(all_predictions_locked=True, predictions_locked_at=now())
                controller.persist(status="scoring_development_round")
                result = score_round(controller, number, current)
                if result["statistics"]["development_goal_methods"]:
                    controller.persist(status="development_goal_reached", finished_at=now())
                    break
                cleanup_round(run, number)
                current.update(status="cleaned", reproducible_work_removed=True)
                controller.persist()
            else:
                controller.persist(status="round_limit_reached", finished_at=now(),
                    next_action="All three predeclared development variants exhausted; inspect retained records before changing mechanisms or data")
        except Exception as exc:
            controller.pause(type(exc).__name__+": "+str(exc))
            raise
        finally:
            controller.closed.set(); monitor.join(timeout=2)
        return state


def status(run):
    run = own_run(run)
    state, protocol = read(run / "state.json"), read(run / "protocol.json")
    active = state.get("active_round")
    current = state["rounds"].get(str(active), {})
    count = len(protocol["cases"])
    alive = False
    if (run / "pid.json").exists():
        try:
            import psutil
            identity = read(run / "pid.json")
            process = psutil.Process(identity["pid"])
            alive = (process.uids().real == os.getuid() and abs(process.create_time()-identity["created"]) < .1
                     and process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
    result = {"status": state["status"], "runner_alive": alive, "active_round": active,
        "maximum_rounds": len(protocol["rounds"]), "learning_folds_completed": len(state["learning"]),
        "learning_folds_planned": count*2, "fits_completed": len(current.get("fits", {})),
        "fits_planned_this_round": count*2*len(NATIVE)*len(FORMULAS),
        "predictions_completed": len(current.get("units", {})), "predictions_planned": count*2*len(METHODS),
        "monitor": state["monitor"], "pause_reason": state.get("pause_reason"), "updated_at": state.get("updated_at")}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "resume", "status", "report", "cleanup", "pause"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args(); run = own_run(args.run_dir)
    if args.command == "resume":
        with (run / ".run.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            verify(read(run / "protocol.json")["source_pins"])
            (run / "PAUSE").unlink(missing_ok=True)
        args.command = "run"
    if args.command == "prepare":
        document = prepare(run, args.workers, args.rounds)
        print(json.dumps({"status": "prepared", "compounds": len(document["cases"]), "rounds": len(document["rounds"])}))
    elif args.command == "run" and args.background:
        import psutil
        # A short lock check avoids knowingly starting a duplicate controller.
        with (run / ".run.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (run / "run.log").open("ab", buffering=0) as log:
            process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "run", "--run-dir", str(run)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
        save(run / "pid.json", {"pid": process.pid, "created": psutil.Process(process.pid).create_time(), "purpose": VERSION})
        print(json.dumps({"pid": process.pid, "log": str(run / "run.log")}))
    elif args.command == "run": execute(run)
    elif args.command == "status": status(run)
    elif args.command == "pause":
        (run / "PAUSE").write_text("Pause before next native call\n")
        print("Pause requested; in-flight native calls finish within their timeouts.")
    elif args.command == "report":
        ledger = read(run / "iterations.json")
        print(json.dumps({k: {"configuration": v["configuration"], "statistics": v["statistics"],
                             "cleanup": v.get("cleanup")} for k,v in ledger["rounds"].items()}, ensure_ascii=False, indent=2))
    else:
        with (run / ".run.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ledger = read(run / "iterations.json")
            print(json.dumps({k: cleanup_round(run, int(k)) for k,v in ledger["rounds"].items()
                              if not v["statistics"]["development_goal_methods"]}))


if __name__ == "__main__":
    main()
