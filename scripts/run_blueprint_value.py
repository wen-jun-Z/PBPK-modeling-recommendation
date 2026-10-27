#!/usr/bin/env python3
"""Value of evidence selection on the frozen twenty-drug learned-exchange model.

No predictor training, own-drug curve fitting, or new physiological algorithm.
The eight manually reviewed evidence sources are an input intervention; masked
replicas and engineering challenges are not additional independent patients.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_clinical_pbpk_comparison as old
import report_blueprint_validation as scores
import evaluate_existing_clinical_pbpk as evaluation
from score_cross_drug_blueprint import _native_fields, score_curve_reference
from render_current_clinical_table import native_value
from render_pbpk_method_comparison import interpolate
from pbpk_kb.blueprint_policy import evidence_decision as source_policy
from pbpk_kb.blueprint_policy import clinical_pbpk_comparison as model
from pbpk_kb.blueprint_policy.learned_exchange_design import VERSION as DESIGN_VERSION
from pbpk_kb.blueprints.clinical_prediction_nca import UNITS
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

VERSION = "clinical-blueprint-value-1"
DEFAULT = ROOT / "experiments/blueprint_design/blueprint_value_20261004"
ORIGINAL = old.DEFAULT
EVIDENCE_PROTOCOL = ROOT / "experiments/blueprint_design/expanded_tail_20260930/protocol.json"
SCENARIOS = ("available", "missing_clearance", "missing_fu", "missing_both")
CLEARANCES = {"total_clearance", "hepatic_clearance", "renal_clearance"}
ALLOW = {
    "total_clearance": {"Dexmedetomidine", "Moxifloxacin", "Remdesivir", "Alprazolam", "Fluconazole", "Furosemide"},
    "fraction_unbound": {"Dexmedetomidine", "Moxifloxacin", "Remdesivir", "Furosemide", "Caffeine", "Nemonoxacin"},
    "renal_clearance": {"Moxifloxacin", "Remdesivir", "Fluconazole", "Furosemide"},
}
CHANGED_DRUGS = set.union(*ALLOW.values())
NEW_CODE = ("scripts/run_blueprint_value.py", "scripts/evaluate_existing_clinical_pbpk.py",
    "scripts/run_clinical_method_comparison.py", "scripts/summarize_additional_clinical_baselines.py",
    "src/pbpk_kb/blueprint_policy/evidence_decision.py")


def own(path):
    path = inside(path, ROOT)
    if path.parent != DEFAULT.parent or not path.name.startswith("blueprint_value_"):
        raise ValueError("Use a project blueprint_value_ experiment directory")
    return path


def rich_qualified(record):
    """An explicit rich-prior qualification; never relabel a rich model as RF."""
    method = record.get("prediction_method", {})
    applicability = method.get("applicability", {})
    return (record.get("basis") == "predicted"
        and method.get("name") == "shared_scaffold_excluded_parameter_model"
        and method.get("version") == old.VERSION and method.get("training_overlap") is False
        and applicability.get("human_IV_parameter_model") is True
        and applicability.get("target_structure_and_scaffold_excluded") is True)


def policy(request, fallback, *, purpose=None):
    """Reuse frozen source rules with a declared predictor qualification adapter.

    Function-local namespaces avoid mutating the frozen RF implementation or its
    globals. Only the qualification predicate changes; validation, priority and
    conflict semantics are the original policy. No parameter is calibrated.
    """
    qualified_issue = FunctionType(source_policy._record_issue.__code__,
        {**source_policy._record_issue.__globals__, "_rf_qualified": rich_qualified})
    def issues(record, compound_id):
        issue, fatal = qualified_issue(record, compound_id)
        if (issue == "invalid_binding_partner" and record.get("parameter") == "binding_partner"
                and record.get("value") == "Glycoprotein"):
            # This is the already-used PK-Sim spelling; retain it unchanged.
            return None, False
        return issue, fatal
    decide = FunctionType(source_policy.decide.__code__, {**source_policy.decide.__globals__,
        "_record_issue": issues, "_rf_qualified": rich_qualified}, argdefs=source_policy.decide.__defaults__)
    result = decide(request, fallback, purpose=purpose, method="evidence_policy")
    forbidden = request.get("forbidden_sources", [])
    paths = {r.get("path") for r in forbidden}
    hashes = {r.get("sha256") for r in forbidden}
    reasons = list(result["reasons"])
    for row in request.get("evidence", []) + fallback:
        source = row.get("source", {})
        if source.get("path") in paths or source.get("sha256") in hashes:
            reasons.append("source_is_held_out_target:" + row.get("parameter", "unknown"))
    if request.get("dosing", {}).get("single_dose", True) is not True:
        reasons.append("repeated_dose_not_supported")
    result.update(policy_version=VERSION, source_rule_version=source_policy.VERSION,
        predictor_qualification="shared_scaffold_excluded_parameter_model; original name retained",
        native_binding_alias="Glycoprotein accepted without modifying original native input",
        status="blocked" if reasons else "ready", reasons=list(dict.fromkeys(reasons)),
        calibration={"performed": False, "allowed_parameters": [], "core_design_version": DESIGN_VERSION},
        clinical_validation=False)
    return result


def select_request(base, independent, scenario, *, scope_source=None):
    if scenario not in SCENARIOS:
        raise ValueError("Unknown masking scenario")
    request = deepcopy(base)
    rows = deepcopy(independent)
    if scenario in {"missing_clearance", "missing_both"}:
        rows = [r for r in rows if r["parameter"] not in CLEARANCES]
    if scenario in {"missing_fu", "missing_both"}:
        rows = [r for r in rows if r["parameter"] != "fraction_unbound"]
    request["evidence"].extend(rows)
    # Organ allocation follows available independent split evidence. Masking
    # removes the entire clearance family, not just its total scalar.
    has_renal = any(r["parameter"] == "renal_clearance" for r in rows)
    for record in request["evidence"]:
        if record["parameter"] == "elimination_scope":
            value = "hepatic_and_renal" if has_renal else "hepatic_only"
            if value != record["value"]:
                record["value"] = value
                record["basis"] = "assumption"
                if scope_source is not None: record["source"] = deepcopy(scope_source)
                record["notes"] = "Derived construction allocation: fixed net renal input plus nonrenal remainder in liver; not a reported organ mechanism"
    fallback = [deepcopy(r) for r in base["evidence"] if r["parameter"] in source_policy.FALLBACK_PARAMETERS]
    decision = policy(request, fallback)
    request["evidence"] = deepcopy(decision["selected_evidence"])
    return request, decision


def semantic_request(request):
    """Auditable numerical-equivalence key, retaining biological context.

    Source paths, ledgers, free text and item ordering do not enter the solver.
    They are separately hash-verified; identity, values, units, context and
    predictor declarations do enter this key.
    """
    keys = ("parameter", "compound_id", "value", "unit", "basis", "context", "prediction_method")
    return {**{k: request[k] for k in ("compound_id", "human", "dosing", "target", "partition_method")},
        "evidence": sorted([{k: r[k] for k in keys if k in r} for r in request["evidence"]],
                           key=lambda r: r["parameter"])}


def fingerprint(request, sidecars, core):
    return model._digest({"request": semantic_request(request), "sidecars": sidecars, "core": core})


def verify(protocol, *, publication=False):
    for name, expected in {**protocol["source_pins"], **protocol["code_pins"]}.items():
        if sha256(inside(name, ROOT, exists=True)) != expected:
            raise old.Pause("Source or code changed: " + name)
    if publication:
        for name, expected in protocol["publication_snapshot"].items():
            if sha256(ROOT / name) != expected:
                raise old.Pause("Publication changed before predictions locked: " + name)


def challenges(cases, requests):
    records = []
    for case in cases:
        base = requests[case["case_id"]]
        fallback = [r for r in base["evidence"] if r["parameter"] in source_policy.FALLBACK_PARAMETERS and rich_qualified(r)]
        for name, expected in (("valid", True), ("unsupported_organ", False), ("ddi", False),
                ("oral", False), ("repeated", False), ("special_population", False),
                ("target_source", False), ("wrong_unit", False), ("conflicting_fu", False),
                ("missing_formula", False), ("valid_reordered", True)):
            request, purpose = deepcopy(base), {}
            if name == "unsupported_organ": purpose["endpoints"] = ["brain_auc"]
            elif name == "ddi": purpose["mechanism"] = "drug_drug_interaction"
            elif name == "oral": request["dosing"]["route"] = "oral"
            elif name == "repeated": request["dosing"]["single_dose"] = False
            elif name == "special_population": purpose["population_condition"] = "renal_impairment"
            elif name == "missing_formula": request["evidence"] = [r for r in request["evidence"] if r["parameter"] != "molecular_formula"]
            elif name == "valid_reordered": request["evidence"].reverse()
            else:
                fu = next(r for r in request["evidence"] if r["parameter"] == "fraction_unbound")
                if name == "target_source": fu["source"] = deepcopy(request["forbidden_sources"][0])
                elif name == "wrong_unit": fu["unit"] = "percent"
                elif name == "conflicting_fu":
                    conflict = deepcopy(fu); conflict["value"] = float(fu["value"]) / 2
                    request["evidence"].append(conflict)
            result = policy(request, deepcopy(fallback), purpose=purpose)
            actual = result["status"] == "ready"
            records.append({"case_id": case["case_id"], "challenge": name, "expected_ready": expected,
                "actual_ready": actual, "correct": actual == expected, "reasons": result["reasons"],
                "engineering_only": True, "new_native_calls": 0})
    return records


def normalization_review(drug, row):
    if row["parameter"] == "fraction_unbound":
        return {"weight_kg": None, "basis": "Dimensionless binding fraction; no weight conversion"}
    source = old.read(ROOT / row["source"]["path"])
    for case in source.get("cases", []):
        if case.get("drug") == drug:
            return {"weight_kg": case.get("normalization_weight_kg"),
                    "basis": case.get("normalization_weight_basis"),
                    "independence": case.get("independence", {})}
    if drug in {"Fluconazole", "Furosemide"}:
        return {"weight_kg": 70., "basis": "Explicit representative-adult assumption, not measured mean cohort body weight"}
    return {"weight_kg": None, "basis": "Source already reports weight-normalized clearance; no new body-weight conversion"}


def prepare(run=DEFAULT, workers=14):
    run = own(run)
    if run.exists(): raise FileExistsError("Preserve existing batch; resume with run")
    if type(workers) is not int or not 1 <= workers <= 14: raise ValueError("Workers must be 1..14")
    state = old.read(ORIGINAL / "state.json"); previous = old.checked(state["protocol"])
    old.verify(previous, state.get("runtime_code_patches"))
    if state["status"] != "complete" or not state.get("all_predictions_locked_before_scoring"):
        raise ValueError("Previous twenty-drug predictions must be complete and locked")
    cases = deepcopy(previous["cases"])
    if len(cases) != 20 or len({c["compound_id"] for c in cases}) != 20: raise ValueError("Exactly twenty drugs required")
    prior = old.checked(state["parameter_predictions"])
    if (set(prior["parameters"]) != {c["compound_id"] for c in cases}
            or any(prior["audit"].get(k) is not False for k in ("target_labels_used", "target_observation_files_opened",
                "training_structure_overlap", "training_generic_scaffold_overlap"))
            or sha256(inside(prior["model_path"], ROOT, exists=True)) != prior["model_sha256"]):
        raise ValueError("Shared prior source, exclusion cohort or training audit changed")
    candidates = old.read(EVIDENCE_PROTOCOL)
    evidence_cases = {c["compound_id"]: c for c in candidates["cases"]}
    pins = dict(previous["source_pins"])
    for path in (EVIDENCE_PROTOCOL, ORIGINAL / "protocol.json", ROOT / state["parameter_predictions"]["path"],
                 inside(prior["model_path"], ROOT, exists=True)):
        pins[str(path.relative_to(ROOT))] = sha256(path)
    code = {name: sha256(ROOT / name) for name in (*previous["code_pins"], *NEW_CODE)}
    core = {"adapter_version": model.VERSION, "design_version": DESIGN_VERSION,
            "code_pins": {name: digest for name, digest in code.items() if name.startswith("src/")},
            "parameter_prior": state["parameter_predictions"]}
    tasks, units, review, baseline, baseline_requests = {}, [], [], {}, {}
    run.mkdir(); (run / "inputs").mkdir()
    rule_path = run / "inputs/selection_rules.json"
    save(rule_path, {"version": VERSION, "rule": "A qualified net-renal input selects hepatic_and_renal; all other clearance is a lumped hepatic remainder. Without a split, hepatic_only remains an assumption.",
        "organ_mechanism_demonstrated": False, "clearance_calibration": False})
    scope_source = {**old.ref(rule_path), "location": "Explicit rule-derived elimination_scope; not a clinical observation"}
    pins[str(rule_path.relative_to(ROOT))] = sha256(rule_path)
    for case in cases:
        cid, drug = case["case_id"], case["drug"]
        receipt = state["outcomes"][cid]; bundle = old.checked(receipt)
        old.collect_refs(bundle, pins); pins[receipt["path"]] = receipt["sha256"]
        entry = bundle["entries"]["ours"]
        if (bundle["case_id"] != cid or bundle["compound_id"] != case["compound_id"] or
                entry["status"] not in {"predicted", "failed"} or entry.get("target_observations_used") is not False):
            raise ValueError("Cached prediction identity or terminal status changed")
        if entry["status"] == "predicted": _native_fields(entry)
        base = deepcopy(entry["request"]); baseline_requests[cid] = base
        values = prior["parameters"][case["compound_id"]]
        sidecars = {"vss": old.prior_evidence(case, values, state["parameter_predictions"], "vss", "l/kg"),
                   "half": old.prior_evidence(case, values, state["parameter_predictions"], "terminal_half_life", "h")}
        baseline_key = fingerprint(base, sidecars, core)
        baseline[cid] = {**receipt, "json_pointer": "/entries/ours", "fingerprint": baseline_key}
        tasks[baseline_key] = {"key": baseline_key, "case_id": cid, "request": base, "sidecars": sidecars,
            "reuse": {**receipt, "json_pointer": "/entries/ours"}, "core": core}
        evidence_case = evidence_cases[case["compound_id"]]
        source_request_ref = evidence_case["requests"]["evidence"]["target"]
        evidence_request = old.checked(source_request_ref)
        pins[source_request_ref["path"]] = source_request_ref["sha256"]
        independent = []
        for row in evidence_request["evidence"]:
            if row["basis"] == "predicted" or row["parameter"] not in ALLOW: continue
            allowed = drug in ALLOW[row["parameter"]]
            review.append({"case_id": cid, "drug": drug, "parameter": row["parameter"],
                "value": row["value"], "unit": row["unit"], "source": row["source"],
                "original_context": row.get("context", {}), "accepted": allowed,
                "reason": "Conservative reviewed non-target source; original role retained" if allowed else
                    "Underlying first-hand source not established; retain shared ML prior",
                "original_request": source_request_ref, "target_request_human": base["human"],
                "source_weight_normalization": normalization_review(drug, row),
                "target_study_id": case["study_id"], "independent_of_target_reviewed": allowed,
                "caution": "Source-body conditions and any 70 kg conversion assumption are not target-patient truth; no development curve fitted"})
            if allowed:
                independent.append(deepcopy(row)); old.collect_refs(row, pins)
        for scenario in SCENARIOS:
            request, decision = select_request(base, independent, scenario, scope_source=scope_source)
            if decision["status"] != "ready":
                raise ValueError(drug + "/" + scenario + ": " + str(decision["reasons"]))
            key = fingerprint(request, sidecars, core)
            if key not in tasks:
                ledger = run / "inputs" / (key[:20] + ".json")
                save(ledger, {"evidence": request["evidence"]})
                request["evidence_ledger"] = {**old.ref(ledger), "location": "Selected inputs; no target observations"}
                pins[str(ledger.relative_to(ROOT))] = sha256(ledger)
                tasks[key] = {"key": key, "case_id": cid, "request": request, "sidecars": sidecars, "reuse": None, "core": core}
            units.append({"case_id": cid, "compound_id": case["compound_id"], "scenario": scenario, "key": key,
                "baseline_key": baseline_key, "input_changed_from_all_ml": key != baseline_key,
                "decision": decision})
    new_tasks = sum(task["reuse"] is None for task in tasks.values())
    changed = {u["case_id"] for u in units if u["scenario"] == "available" and u["input_changed_from_all_ml"]}
    if new_tasks != 16 or len(changed) != 8 or len(units) != 80:
        raise ValueError(f"Unexpected conservative matrix: new={new_tasks}, changed={len(changed)}, units={len(units)}")
    protocol = {"version": VERSION, "created_at": old.now(), "cases": cases, "scenarios": list(SCENARIOS),
        "tasks": tasks, "units": units, "baselines": baseline, "source_review": review,
        "engineering_challenges": challenges(cases, baseline_requests),
        "core": core, "source_pins": pins, "code_pins": code,
        "publication_snapshot": old.paper_pins(), "paper_guard": {},
        "publication_role": "Publication assets are checked during prediction only, never scientific input or post-completion status/report dependency",
        "source_protocol": state["protocol"], "parameter_prior": state["parameter_predictions"],
        "changed_case_ids": sorted(changed), "logical_units": 80, "new_unique_predictions": new_tasks,
        "workers": workers, "native_threads": 2, "native_timeout_seconds": 180,
        "directory_limit_bytes": 10 * 1024**3, "monitor_seconds": 1200, "linear_calls_per_prediction": 33,
        "seed": 20261004, "bootstrap_resamples": 2000,
        "masking": "missing_clearance removes independent total/hepatic/renal together; all missing inputs use the frozen shared prior",
        "information_matching": "Evidence-first and always-ML policies have the same candidate evidence pool, molecular data and frozen model; the latter deliberately ignores independent CL/fu/renal",
        "fit_policy": "No clinical curve or clinical target endpoint used to fit; Vss/half remain shared-prior predictions",
        "limitations": ["Previously viewed twenty-drug development cohort, not a new blind test",
            "Only eight drugs have accepted independent input interventions; unchanged replicas are not new evidence",
            "Source curation is human work and not counted as automated construction",
            "Evidence availability benefit is not by itself proof of a novel optimal decision algorithm",
            "Engineered request faults validate policy compliance, not clinical safety",
            "Stability means output change under missing inputs, not necessarily improved accuracy",
            "All reference-missing endpoints remain unscored; reported and curve-derived references are separate"]}
    if not all(r["correct"] for r in protocol["engineering_challenges"]):
        raise ValueError("Declared input-policy challenges failed")
    verify(protocol, publication=True)
    save(run / "protocol.json", protocol)
    save(run / "state.json", {"version": VERSION, "protocol": old.ref(run / "protocol.json"), "status": "prepared",
        "created_at": old.now(), "outcomes": {}, "all_predictions_locked_before_scoring": False})
    return {"status": "prepared", "drugs": 20, "logical_units": 80, "new_native_predictions": new_tasks,
            "reused_baseline_predictions": 20, "engineering_challenges": len(protocol["engineering_challenges"])}


class Monitor(old.Monitor):
    def check(self, force=False):
        super().check(force)
        # The parent checks source/code and the empty new paper_guard; publication
        # is not promoted to a permanent computational input dependency.
        if force:
            verify(self.protocol, publication=True)


def solve_task(task, run, monitor):
    key = task["key"]
    with monitor.lock:
        if key in monitor.state["outcomes"]:
            old.checked(monitor.state["outcomes"][key]); return key
    monitor.check()
    folder = run / "models" / key[:20]
    output = folder.with_suffix(".json")
    started = time.monotonic()
    if task["reuse"]:
        bundle = old.checked(task["reuse"])
        entry = deepcopy(bundle["entries"]["ours"])
        if semantic_request(entry["request"]) != semantic_request(task["request"]):
            raise old.Pause("Baseline semantic reuse mismatch")
        result = {"key": key, "case_id": task["case_id"], "entry": entry,
            "reused": True, "reused_from": task["reuse"], "new_native_calls": 0,
            "new_linear_trajectory_calls": 0, "elapsed_seconds": 0., "locked_at": old.now()}
    else:
        try:
            recovered = None
            if folder.exists():
                if (folder / "result.json").is_file():
                    recovered = old.read(folder / "result.json")
                    if recovered.get("request_sha256") != model._digest(task["request"]):
                        raise old.Pause("Incomplete execution receipt belongs to another input")
                    old.collect_refs(recovered, {})
                    recovered = {**recovered, "result_ref": old.ref(folder / "result.json")}
                else:
                    old.preserve_interrupted(folder)
            entry = recovered or model.run_ours(task["request"], task["sidecars"]["vss"],
                task["sidecars"]["half"], folder, ROOT)
            entry = deepcopy(entry)
            entry.update(request=task["request"], target_observations_used=False, condition="no_clinical_curve")
            if entry["status"] == "predicted":
                if entry["endpoints"].get("observations_used") is not False:
                    raise old.Pause("Prediction endpoint estimator used observations")
                entry["endpoints"].update(comparison_observations_used=False, endpoint_estimator_uses_observations=False)
                _native_fields(entry)
            else:
                entry["failure"] = entry.get("reason", "Prediction failed")
            infrastructure = False
            if entry.get("reason") in {"native_execution_failed", "native_source_failed"}:
                process = old.checked(entry["execution_ref"]).get("process", {})
                infrastructure = bool(process.get("stop_reason") or process.get("error") or process.get("status") == "blocked")
        except Exception as error:
            if isinstance(error, old.Pause): raise
            infrastructure = isinstance(error, (OSError, ImportError)) or type(error).__name__ == "InfrastructurePause"
            entry = {"status": "failed", "failure": type(error).__name__ + ": " + str(error),
                "reason": type(error).__name__ + ": " + str(error), "request": task["request"],
                "target_observations_used": False, "curve": None, "endpoints": None}
        if infrastructure:
            with monitor.lock:
                count = monitor.state.get("consecutive_infrastructure_errors", 0) + 1
                monitor.update(consecutive_infrastructure_errors=count)
                if count >= 3:
                    monitor.halted.set(); monitor.update(pause_reason="Three consecutive infrastructure errors")
            # Transient infrastructure errors do not become scientific failures.
            save(output, {"infrastructure_failure": True, "entry": entry, "key": key})
            raise old.Pause("Infrastructure error retained for explicit resume: " + entry.get("failure", "unknown"))
        with monitor.lock: monitor.update(consecutive_infrastructure_errors=0)
        result = {"key": key, "case_id": task["case_id"], "entry": entry, "reused": False,
            "new_native_calls": entry.get("native_solver_calls", 1),
            "new_linear_trajectory_calls": entry.get("linear_trajectory_calls", 0),
            "elapsed_seconds": time.monotonic() - started, "locked_at": old.now()}
    save(output, result)
    with monitor.lock:
        monitor.state["outcomes"][key] = old.ref(output)
        monitor.update(completed_unique=len(monitor.state["outcomes"]),
            completed_new=sum(not old.checked(r)["reused"] for r in monitor.state["outcomes"].values()))
    return key


def run_experiment(run=DEFAULT):
    run = own(run)
    with (run / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = old.read(run / "state.json"); protocol = old.checked(state["protocol"])
        if state["status"] in {"predictions_locked", "complete"}: return report(run, locked=True)
        verify(protocol, publication=True)
        (run / "models").mkdir(exist_ok=True)
        monitor = Monitor(run, protocol, state)
        thread = threading.Thread(target=monitor.watch, daemon=True); thread.start()
        monitor.update(status="running", pid=os.getpid(), started_at=state.get("started_at", old.now()))
        try:
            monitor.check(force=True)
            with ThreadPoolExecutor(max_workers=protocol["workers"]) as pool:
                futures = [pool.submit(solve_task, task, run, monitor) for task in protocol["tasks"].values()]
                for future in as_completed(futures): future.result()
            monitor.check(force=True)
            if set(state["outcomes"]) != set(protocol["tasks"]): raise old.Pause("Prediction matrix is incomplete")
            monitor.update(status="predictions_locked", all_predictions_locked_before_scoring=True,
                predictions_locked_at=old.now())
        except Exception as error:
            monitor.halted.set(); monitor.update(status="paused", pause_reason=type(error).__name__ + ": " + str(error))
            raise
        finally:
            monitor.done.set(); thread.join(timeout=20)
    return report(run)


def score_one(case, receipt, entry, store, clinical, observed_curve, truth):
    row = {k: case[k] for k in ("case_id", "compound_id", "drug", "study_id", "cluster_id", "dataset")}
    row.update(attempt_status=entry["status"], scoring_status="failed", failure=entry.get("failure"),
        curve_metrics={}, endpoints={}, reference_family=evaluation.reference_family(case))
    if entry["status"] == "predicted":
        if case["reference_kind"] == "observed_curve": row.update(score_curve_reference(case, entry, store))
        else:
            unit = {"prediction": {**receipt, "json_pointer": "/entry/curve"},
                    "request": {**receipt, "json_pointer": "/entry/request"}}
            scored, _, _ = scores._score_prediction(case, clinical, observed_curve, truth, unit, store)
            scored.pop("curve", None)
            for endpoint, value in scored["endpoints"].items():
                value["reference_kind"] = "original_reported" if value["reference"] is not None else "unavailable"
                value["native_prediction"], _, value["native_prediction_reason"] = native_value(
                    entry["endpoints"]["endpoints"][endpoint], endpoint, value["unit"], entry["request"]["human"]["weight_kg"])
            row.update(scored, scoring_status="scored" if scores.positive(scored["curve_metrics"].get("gmfe")) else "partial")
    for endpoint, unit in UNITS.items():
        if endpoint not in row["endpoints"]:
            row["endpoints"][endpoint] = {"value": None, "native_prediction": None, "reference": truth[endpoint]["value"],
                "unit": truth[endpoint].get("unit") or unit, "error": None,
                "error_metric": "absolute_error_h" if endpoint == "tmax" else "fold_error",
                "reference_kind": row["reference_family"] if truth[endpoint]["value"] is not None else "unavailable",
                "reason": row["failure"] or "Prediction unavailable", "reference_reason": truth[endpoint].get("reason", ""),
                "weak_tail": False}
        row["endpoints"][endpoint]["endpoint_label"] = evaluation.endpoint_label(endpoint, row["reference_family"])
    return row


def drift(case, nominal, masked):
    result = {"curve_geometric_fold_change": None, "endpoints": {}, "unbounded": False,
              "basis": "Fixed declared target sampling times; no observation concentrations; scalar model-only outputs"}
    if nominal["status"] != "predicted" or masked["status"] != "predicted":
        return {**result, "status": "failed_prediction", "reason": "Both predictions are needed; failure is retained"}
    times = [t for t in case["nca_policy"]["sampling_time_h"] if t > 0]
    curves = []
    for entry in (nominal, masked):
        native = entry["curve"]
        grid = [t / 60 for t in native["time_min"]]
        curves.append([interpolate(grid, native["concentration"], t) for t in times])
    changes = []
    for a, b in zip(*curves):
        if a == b: changes.append(1.)
        elif min(a, b) <= 0: result["unbounded"] = True
        else: changes.append(max(a / b, b / a))
    if changes and not result["unbounded"]:
        result["curve_geometric_fold_change"] = math.exp(math.fsum(map(math.log, changes)) / len(changes))
    for endpoint, unit in UNITS.items():
        values = [native_value(entry["endpoints"]["endpoints"][endpoint], endpoint, unit,
            entry["request"]["human"]["weight_kg"])[0] for entry in (nominal, masked)]
        value, unbounded = None, False
        if all(x is not None for x in values):
            if endpoint == "tmax": value = abs(values[1] - values[0])
            elif values[0] == values[1]: value = 1.
            elif min(values) <= 0: unbounded = True
            else: value = max(values) / min(values)
        result["endpoints"][endpoint] = {"change": value, "unbounded": unbounded,
            "metric": "absolute_change_h" if endpoint == "tmax" else "fold_change"}
    return {**result, "status": "assessed", "sampling_points": len(times)}


def summarize(rows, protocol):
    summaries, pairs, missing = [], {}, []
    cases = protocol["cases"]
    groups = {"all_20": cases, "changed_8": [c for c in cases if c["case_id"] in protocol["changed_case_ids"]],
        "original_10": [c for c in cases if c["cohort"] == "original"],
        "extension_10": [c for c in cases if c["cohort"] == "extension"]}
    for name, group in groups.items():
        ids = {c["case_id"] for c in group}; clusters = {c["case_id"]: c["cluster_id"] for c in group}
        values = {}
        for condition in ("all_ml", *SCENARIOS):
            selected = [r for r in rows if r["case_id"] in ids and r["condition"] == condition]
            valid = {r["case_id"]: r["curve_metrics"]["gmfe"] for r in selected if scores.positive(r["curve_metrics"].get("gmfe"))}
            values[condition] = valid
            epstats = {}
            for family in evaluation.REFERENCE_FAMILIES:
                group_rows = [r for r in selected if r["reference_family"] == family]
                if not group_rows: continue
                epstats[family] = {}
                for ep in UNITS:
                    items = [r["endpoints"][ep] for r in group_rows]
                    errors = [v["error"] for v in items if v["error"] is not None]
                    available = sum(v["reference"] is not None for v in items)
                    stats = ({"mae_h": math.fsum(errors) / len(errors) if errors else None, "scored": len(errors),
                              "attempted": len(items), "reference_available": available} if ep == "tmax" else
                             evaluation.fold_statistics(errors, attempted=len(items), reference_available=available))
                    stats.update(weak_scored=sum(bool(v.get("weak_tail")) and v["error"] is not None for v in items))
                    epstats[family][ep] = stats
            summaries.append({"group": name, "condition": condition, "attempted": len(selected),
                "failed": sum(r["attempt_status"] == "failed" for r in selected), "scored": len(valid),
                "curve": evaluation.fold_statistics(valid.values(), attempted=len(selected), reference_available=len(selected)),
                "curve_ci": scores.clustered_geomean(valid, {k: clusters[k] for k in valid}, resamples=2000, seed=20261004),
                "endpoints": epstats})
        pairs[name] = {"curve": scores.paired_ratio(values["available"], values["all_ml"], clusters,
            expected_ids=ids, resamples=2000, seed=20261004), "endpoints": {}}
        for family in evaluation.REFERENCE_FAMILIES:
            pairs[name]["endpoints"][family] = {}
            for endpoint in UNITS:
                if endpoint == "tmax": continue
                by_condition = {cond: {r["case_id"]: r["endpoints"][endpoint]["error"] for r in rows
                    if r["case_id"] in ids and r["condition"] == cond and r["reference_family"] == family
                    and scores.positive(r["endpoints"][endpoint]["error"])} for cond in ("available", "all_ml")}
                family_ids = {r["case_id"] for r in rows if r["case_id"] in ids and r["reference_family"] == family}
                pairs[name]["endpoints"][family][endpoint] = scores.paired_ratio(by_condition["available"],
                    by_condition["all_ml"], clusters, expected_ids=family_ids, resamples=2000, seed=20261004)
    for scenario in SCENARIOS[1:]:
        selected = [r for r in rows if r["condition"] == scenario and r["changed_from_nominal"]]
        ids = {r["case_id"] for r in selected}; clusters = {c["case_id"]: c["cluster_id"] for c in cases if c["case_id"] in ids}
        nominal = {r["case_id"]: r["curve_metrics"].get("gmfe") for r in rows if r["condition"] == "available" and r["case_id"] in ids}
        masked = {r["case_id"]: r["curve_metrics"].get("gmfe") for r in selected}
        changes = {r["case_id"]: r["prediction_drift"]["curve_geometric_fold_change"] for r in selected
                   if scores.positive(r["prediction_drift"]["curve_geometric_fold_change"])}
        missing.append({"scenario": scenario, "actually_changed_drugs": len(ids),
            "masked_over_nominal_clinical_gmfe": scores.paired_ratio(masked, nominal, clusters,
                expected_ids=ids, resamples=2000, seed=20261004),
            "prediction_drift": scores.clustered_geomean(changes, {k: clusters[k] for k in changes}, resamples=2000, seed=20261004),
            "drift_unassessable_or_unbounded": len(ids) - len(changes),
            "not_clinical_accuracy_guarantee": True})
    return summaries, pairs, missing


def report(run=DEFAULT, *, locked=False):
    run = own(run)
    if not locked:
        with (run / "run.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return report(run, locked=True)
    state = old.read(run / "state.json"); protocol = old.checked(state["protocol"])
    verify(protocol)
    if not state.get("all_predictions_locked_before_scoring") or set(state["outcomes"]) != set(protocol["tasks"]):
        raise ValueError("All unique predictions must be terminal and locked before scoring")
    loaded = {}
    for key, receipt in state["outcomes"].items():
        item = old.checked(receipt); task = protocol["tasks"][key]
        if item["key"] != key or item["case_id"] != task["case_id"] or not item.get("locked_at"):
            raise ValueError("Prediction receipt identity changed")
        entry = item["entry"]
        if entry["status"] not in {"predicted", "failed"} or entry.get("target_observations_used") is not False:
            raise ValueError("Unfinished or target-informed prediction")
        if fingerprint(entry["request"], task["sidecars"], task["core"]) != key:
            raise ValueError("Prediction source or semantic input key changed")
        if entry["status"] == "predicted": _native_fields(entry)
        old.collect_refs(entry, {}); loaded[key] = item
    store, cached_scores, rows, engineering = scores.Store(ROOT), {}, [], []
    for case in protocol["cases"]:
        cid = case["case_id"]
        clinical, observed_curve, truth = evaluation.clinical_truth(case, store)
        units = [u for u in protocol["units"] if u["case_id"] == cid]
        nominal_key = next(u["key"] for u in units if u["scenario"] == "available")
        base_key = protocol["baselines"][cid]["fingerprint"]
        for condition, key in [("all_ml", base_key), *[(u["scenario"], u["key"]) for u in units]]:
            if key not in cached_scores:
                cached_scores[key] = score_one(case, state["outcomes"][key], loaded[key]["entry"], store,
                    clinical, observed_curve, truth)
            row = deepcopy(cached_scores[key])
            row.update(condition=condition, method=condition, fingerprint=key,
                prediction_reused=loaded[key]["reused"], logical_alias=condition != "all_ml" and
                    sum(u["key"] == key for u in units) > 1,
                prediction_receipt=state["outcomes"][key], source_hash_verified=True,
                changed_from_nominal=key != nominal_key, input_changed_from_all_ml=key != base_key,
                prediction_drift=drift(case, loaded[nominal_key]["entry"], loaded[key]["entry"]))
            rows.append(row)
    for key, item in loaded.items():
        entry = item["entry"]; request = entry["request"]
        native = old.checked(entry["native_source_ref"]) if entry.get("native_source_ref") else None
        audit_ref = (native or {}).get("audit_refs", {}).get("parameters")
        audit = old.checked(audit_ref) if audit_ref else {}
        renal = next((r for r in request["evidence"] if r["parameter"] == "renal_clearance"), None)
        fu_checks = [c for c in audit.get("parameter_checks", []) if "Fraction unbound (plasma" in c["path"]]
        renal_checks = [c for c in audit.get("parameter_checks", []) if "Renal Clearance" in c["path"] and c["path"].endswith("|Plasma clearance")]
        engineering.append({"key": key, "case_id": item["case_id"], "reused": item["reused"],
            "prediction_status": entry["status"], "native_parameter_audit_passed": audit.get("accepted"),
            "fraction_unbound_readback": fu_checks, "renal_input": renal, "renal_readback": renal_checks,
            "source_hashes_verified": True, "new_native_calls": item["new_native_calls"],
            "linear_trajectory_calls": item["new_linear_trajectory_calls"], "elapsed_seconds": item["elapsed_seconds"],
            "clinical_safety_validated": False})
    summaries, pairs, missing = summarize(rows, protocol)
    store.unchanged(); verify(protocol)
    challenges_ = protocol["engineering_challenges"]
    result = {"version": VERSION, "protocol": state["protocol"], "core": protocol["core"],
        "predictions_locked_at": state["predictions_locked_at"], "all_predictions_locked_before_scoring": True,
        "logical_intervention_units": 80, "baseline_units": 20, "unique_predictions": len(loaded),
        "new_native_calls": sum(i["new_native_calls"] for i in loaded.values()), "new_training_calls": 0,
        "new_linear_trajectory_calls": sum(i["new_linear_trajectory_calls"] for i in loaded.values()),
        "curves_drug_equal": True, "threshold_comparison": "<", "bootstrap": {"resamples": 2000, "seed": 20261004, "unit": "study_cluster"},
        "units": rows, "summaries": summaries, "available_over_all_ml": pairs, "missingness": missing,
        "engineering": engineering, "engineering_challenges": {"records": challenges_, "attempted": len(challenges_),
            "correct": sum(c["correct"] for c in challenges_), "accepted": sum(c["actual_ready"] for c in challenges_),
            "valid_requests": sum(c["expected_ready"] for c in challenges_), "not_clinical_validation": True},
        "source_pins": {**protocol["source_pins"], **store.pins}, "code_pins": protocol["code_pins"],
        "limitations": protocol["limitations"], "paper_updated": False}
    save(run / "results.json", result)
    # Reuse the validated 11-output CSV writer with honest method names; its
    # local namespace is isolated and frozen helper code remains untouched.
    writer = FunctionType(evaluation.write_csv.__code__, {**evaluation.write_csv.__globals__,
        "LABELS": {name: name for name in ("all_ml", *SCENARIOS)}})
    writer(run / "results.csv", rows)
    text = ["# Evidence-selection value on the frozen clinical blueprint", "",
        "Twenty previously viewed drugs; eight reviewed independent-evidence interventions; no curve fitting or new training.",
        "The same learned-exchange core and shared twenty-drug-excluded prior serve every policy. This is an input-policy ablation, not an external-method comparison.", "",
        "| Group | Input condition | Scored / requested | Curve GMFE | Failed |", "|---|---|---:|---:|---:|"]
    for summary in summaries:
        if summary["group"] in {"all_20", "changed_8"}:
            value = summary["curve"]["gmfe"]
            text.append(f"| {summary['group']} | {summary['condition']} | {summary['scored']}/{summary['attempted']} | {value:.3f} | {summary['failed']} |")
    text += ["", "Primary evidence-value comparison uses only the eight drugs whose available inputs changed; all twenty remain reported.",
        "Curve GMFE is the geometric mean of per-drug errors. Paired study-cluster bootstrap uses 2,000 draws; Ertapenem and Ceftriaxone share one cluster. Reported clinical endpoints and observation-derived window AUC/sampled peak remain separate; missing references remain blank.",
        "Missing-clearance removes total, hepatic and renal evidence together. Stability is prediction drift, not a promise of improved accuracy. Only actual input changes count toward missingness summaries.",
        "Source extraction and literature review were manual. Source checks, selection, construction, readback, locking and scoring are automated; no manual-time saving is inferred.",
        "Engineered valid/invalid requests establish implementation compliance, not clinical safety or calibrated clinical risk.",
        f"{len(loaded)} unique predictions cover 80 masking units plus 20 baselines; {result['new_native_calls']} new native calls, {result['new_linear_trajectory_calls']} bounded linear searches; repeated aliases are not independent experiments.",
        "The frozen source roles and source-patient/body-weight assumptions are retained in protocol.source_review. Independent construction parameters are not target-patient truth.",
        "Publication assets are not computational dependencies after predictions lock; later authorised paper edits do not invalidate these results.", "",
        "[Results and paired intervals](results.json) · [All predictions, available observations and errors](results.csv)", "",
        "```bash", "runtime/app-env/bin/python -B scripts/run_blueprint_value.py status",
        "runtime/app-env/bin/python -B scripts/run_blueprint_value.py run --background",
        "runtime/app-env/bin/python -B scripts/run_blueprint_value.py report", "```", ""]
    (run / "README.md").write_text("\n".join(text))
    state.update(status="complete", completed_at=old.now(), results=old.ref(run / "results.json"),
        result_table=old.ref(run / "results.csv"), result_readme=old.ref(run / "README.md"))
    save(run / "state.json", state)
    return {"status": "complete", "primary_changed_8": pairs["changed_8"]["curve"],
            "new_native_calls": result["new_native_calls"], "unique_predictions": len(loaded)}


def status(run=DEFAULT):
    run = own(run); state = old.read(run / "state.json"); protocol = old.checked(state["protocol"])
    verify(protocol)
    for receipt in state["outcomes"].values(): old.checked(receipt)
    result = {"status": state["status"], "completed_unique": len(state["outcomes"]),
        "total_unique": len(protocol["tasks"]), "new_native_predictions": protocol["new_unique_predictions"],
        "completed_new": state.get("completed_new", 0), "pause_reason": state.get("pause_reason"),
        "monitor": state.get("monitor"), "pid": state.get("pid")}
    if state.get("results"):
        summary = old.checked(state["results"])
        result["primary_changed_8"] = summary["available_over_all_ml"]["changed_8"]["curve"]
    return result


def background(run):
    run = own(run)
    with (run / "run.log").open("ab", buffering=0) as log:
        process = subprocess.Popen([sys.executable, "-B", str(Path(__file__)), "run", "--run-dir", str(run)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
    save(run / "pid.json", {"pid": process.pid, "created_at": old.now(), "project": str(ROOT), "run": str(run)})
    return {"status": "started", "pid": process.pid, "log": str(run / "run.log")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status", "report"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare": output = prepare(args.run_dir, args.workers)
    elif args.command == "run": output = background(args.run_dir) if args.background else run_experiment(args.run_dir)
    else: output = globals()[args.command](args.run_dir)
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__": main()
