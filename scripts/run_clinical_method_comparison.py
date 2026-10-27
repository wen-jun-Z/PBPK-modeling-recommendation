#!/usr/bin/env python3
"""Bounded multi-drug Ours/RF comparison; separate prediction and clinical scoring.

Input config contains construction metadata and source paths, never evaluation
concentrations or PK labels. The clinical-reference files are hashed during
preparation and decoded only after all prediction attempts have been locked.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from run_complete_reference_pilot import ROOT, WHEEL, read, now, ref, verify
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

BASE = ROOT / "experiments/blueprint_design/clinical_complete_reference_20260928/expansion"
FORMULAS = ("PKSimStandard", "RodgersRowland")
METHODS = ("ours", "rf")


def prepare(config_path, output):
    if output.exists():
        raise FileExistsError("Use a new run directory; previous experiments are immutable")
    config_path = inside(config_path, ROOT, exists=True)
    config = read(config_path)
    if not config.get("cases"):
        raise ValueError("No curated clinical cases")
    workers = config.get("workers", 3)
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("Project native worker count must be an integer from 1 through 8")
    cases, pins, identities = [], {}, set()

    def pin(path):
        path = inside(path, ROOT, exists=True)
        pins[str(path.relative_to(ROOT))] = sha256(path)

    for path in [config_path, Path(__file__), ROOT / "scripts/run_complete_reference_pilot.py", WHEEL,
                 *sorted((ROOT / "src/pbpk_kb/blueprints").glob("*.py")),
                 *sorted((ROOT / "src/pbpk_kb/model_runtime/R").glob("*.R"))]:
        pin(path)
    human = ROOT / "artifacts/pbpk-model-assets/human_base/standard_small_molecule.json"
    pin(human)
    for source in read(human)["resources"]:
        verify({source["path"]: source["sha256"]}); pin(source["path"])
    for item in config["cases"]:
        case = read(inside(item, ROOT, exists=True)) if isinstance(item, str) else deepcopy(item)
        if isinstance(item, str):
            pin(item)
        for key in ("case_id", "compound_id", "drug", "study_id", "physchem", "development", "reference"):
            if not isinstance(case.get(key), str) or not case[key]:
                raise ValueError("Missing case metadata: " + key)
        if case["case_id"] in identities:
            raise ValueError("Duplicate clinical case")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", case["case_id"]):
            raise ValueError("Clinical case_id must be a safe single directory name")
        identities.add(case["case_id"])
        for key in ("physchem", "development", "reference"):
            pin(case[key])
        physical, development = read(ROOT / case["physchem"]), read(ROOT / case["development"])
        if development["compound_id"] != case["compound_id"] or development["study_id"] == case["study_id"]:
            raise ValueError("Development identity mismatch or same-study evaluation")
        if physical["identity"].get("pubchem_cid") is not None and case["compound_id"] != "pubchem:" + str(physical["identity"]["pubchem_cid"]):
            raise ValueError("Physicochemical identity mismatch")
        for document in (physical, development):
            sources = list(document.get("source_files", []))
            if isinstance(document.get("source"), dict) and document["source"].get("path"):
                sources.append(document["source"])
            for source in sources:
                if source.get("sha256"):
                    verify({source["path"]: source["sha256"]})
                pin(source["path"])
        if sha256(ROOT / case["development"]) == sha256(ROOT / case["reference"]):
            raise ValueError("Cannot reuse clinical reference as development observations")
        case["development_source"] = {**ref(case["development"], "Independent clinical development curve"),
                                       "role": "calibration_data_reconstruction"}
        case["reference_source"] = ref(case["reference"], "Clinical outcomes; post-prediction scoring only")
        cases.append(case)
    protocol = {"version": "clinical-ours-rf-multidrug-1", "prepared_at": now(), "cases": cases,
                "seed": 20260928, "train_limit": 1283, "parameter_test_limit": 100,
                "train_policy": "All eligible remaining rows after external identity/scaffold exclusions and scaffold-held-out parameter test; no hyperparameter search",
                "methods": list(METHODS), "partition_candidates": list(FORMULAS),
                "workers": workers, "fit_calls_per_formula": 18, "recover_failed_trials": False,
                "information_difference": "Ours additionally uses a separate same-drug clinical development curve; RF uses structure priors only. This does not isolate algorithm superiority.",
                "reference_policy": "Same-study clinical curve and at least 7 original-reported scalar endpoints (>=8/11 classes); at most 3 missing references; never fill by project NCA",
                "pins": pins}
    output.mkdir(parents=True)
    save(output / "summary.json", {"status": "prepared", "protocol": protocol, "fits": {}, "results": {}, "locks": {}})
    return protocol


def construct(case, learned, output, shared_sources):
    from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest, assemble_evidence_blueprint
    from pbpk_kb.blueprints.extended_prediction import IV_ENDPOINTS
    from pbpk_kb.blueprints.development_curve_blueprint import _training_document

    folder = output / "inputs" / case["case_id"]
    folder.mkdir(parents=True)
    physical, dev = read(ROOT / case["physchem"]), read(ROOT / case["development"])
    predicted = learned.predict(physical["identity"]["isomeric_smiles"])
    if predicted["training_structure_overlap"] or predicted["training_scaffold_overlap"]:
        raise ValueError("Clinical compound leaked into parameter training")
    values = predicted["predictions"]
    record = {"compound_id": case["compound_id"], "method": "random_forest",
              "predictions": {"clearance_ml_min_kg": values["human_CL_mL_min_kg"]["random_forest"],
                              "vss_l_kg": values["human_VDss_L_kg"]["random_forest"],
                              "fraction_unbound": values["human_fup"]["random_forest"]},
              "all_parameter_predictions": predicted, "training_overlap": False,
              "same_scaffold_training_overlap": False, "provenance": shared_sources,
              "no_clinical_outcomes_used": True, "predicted_vss_used_for_pbpk_construction": False}
    save(folder / "predicted_pk.json", record)
    pk_source = ref(folder / "predicted_pk.json", "Locked structure-derived RF predictions")
    pc_source = ref(case["physchem"], "Independent physicochemical inputs and declared assumptions")
    metadata = case["target_request_metadata"]
    assumptions = {"binding_partner": case.get("binding_partner", "Albumin"),
                   "elimination_scope": "hepatic_only", "organ_allocation_is_assumption": True,
                   "solubility": physical["optional_iv_construction_assumption"],
                   "target_demographics": metadata,
                   "note": "Same direct-construction assumptions for both methods; total clearance lumped into liver; RF Vss is predicted but not imposed"}
    save(folder / "construction_assumptions.json", assumptions)
    assumption_source = ref(folder / "construction_assumptions.json", "Declared engineering assumptions, not clinical labels")
    evidence = []

    def add(name, value, unit, basis, source, *, note="", method=None, context=None):
        item = {"parameter": name, "compound_id": case["compound_id"], "value": value,
                "unit": unit, "basis": basis, "source": source, "notes": note, "context": context or {}}
        if method:
            item["prediction_method"] = method
        evidence.append(item)

    identity, fields = physical["identity"], physical["selected_physicochemical_inputs"]
    add("molecular_weight", identity["molecular_weight_g_mol"], "g/mol", "derived", pc_source)
    add("molecular_formula", identity["molecular_formula"], "text", "derived", pc_source)
    chemical_method = {"name": "source-specified chemical descriptor", "version": "frozen source record",
                       "training_overlap": "unknown", "applicability": "Chemical input; not target clinical labels"}
    logp = fields["logp"]
    add("logp", logp["value"], "Log Units", logp.get("basis", "predicted"), pc_source,
        note=logp.get("note", "See frozen physicochemical source"),
        method=chemical_method if logp.get("basis", "predicted") == "predicted" else None)
    pka = fields.get("pka_types")
    if pka is None:
        pka = {"value": [{"Type": "Acid", "Pka": fields["acidic_pka"]["value"]}], "basis": "measured"}
    add("pka_types", pka["value"], "1", pka.get("basis", "measured"), pc_source,
        note=pka.get("note", "See frozen physicochemical source"),
        method=chemical_method if pka.get("basis") == "predicted" else None)
    sol = physical["optional_iv_construction_assumption"]
    add("solubility", sol["solubility_mg_ml"], "mg/ml", "assumption", assumption_source,
        note="Placeholder for already dissolved IV input, not measured solubility")
    add("solubility_ph", sol["solubility_ph"], "1", "assumption", assumption_source)
    add("binding_partner", assumptions["binding_partner"], "text", "assumption", assumption_source)
    add("elimination_scope", "hepatic_only", "text", "assumption", assumption_source,
        note="Lumped total plasma clearance, not validated organ-specific elimination")
    method = {"name": "random_forest", "version": "clinical-ours-rf-multidrug-1", "training_overlap": False,
              "applicability": {"target_structure_and_scaffold_excluded": True, "human_IV_parameter_model": True}}
    for name, key, unit in (("total_clearance", "clearance_ml_min_kg", "ml/min/kg"),
                             ("fraction_unbound", "fraction_unbound", "fraction")):
        add(name, record["predictions"][key], unit, "predicted", pk_source, method=method,
            context={"species": "human", "matrix": "plasma", "route": "intravenous"})
    save(folder / "evidence_ledger.json", {"evidence": evidence})
    request = {"entry_reason": "model_missing", "compound_id": case["compound_id"], "compound_name": case["drug"],
               "human": metadata["human"], "dosing": metadata["dosing"], "partition_method": "PKSimStandard",
               "target": {"endpoints": list(IV_ENDPOINTS), "primary_endpoint": "plasma_concentration_time",
                          "window_start_h": 0.0, "window_end_h": metadata["window_end_h"]},
               "evidence_ledger": ref(folder / "evidence_ledger.json", "Shared frozen construction evidence"),
               "evidence": evidence, "forbidden_sources": [case["reference_source"]],
               "assumptions": [*metadata.get("demographic_assumptions", []),
                               "RF does not receive development observations; Ours receives a separate clinical study.",
                               "Predicted Vss is recorded but not imposed on tissue distribution.",
                               "Total clearance is lumped into liver; organ allocation is not clinically validated.",
                               "One representative adult is compared with reported clinical group summaries."]}
    development = deepcopy(request)
    development["human"], development["dosing"] = deepcopy(dev["human"]), deepcopy(dev["dosing"])
    development["target"]["window_end_h"] = max(metadata["window_end_h"], math.ceil(dev["observations"][-1]["time_min"] / 60))
    for req, name in ((request, "target_request.json"), (development, "development_request.json")):
        EvidenceBlueprintRequest.model_validate(req)
        if assemble_evidence_blueprint(req, ROOT)["status"] != "assembled":
            raise ValueError("Incomplete evidence assembly: " + case["case_id"])
        save(folder / name, req)
    _training_document(case["development_source"], development, ROOT)
    return request, development, pk_source


def _execute(output):
    from pbpk_kb.blueprints.pk_parameter_learning import run_parameter_pilot
    from pbpk_kb.blueprints.ml_hybrid_blueprint import run_ml_direct_case
    from pbpk_kb.blueprints.development_curve_blueprint import fit_development_mapping, predict_with_development_mapping
    import joblib

    summary = read(output / "summary.json")
    if summary["status"] != "prepared":
        raise ValueError("Only an unstarted preparation may run")
    protocol, started = summary["protocol"], time.monotonic()
    verify(protocol["pins"])
    with (output / "runner.pid").open("x") as handle:
        handle.write(str(os.getpid()) + "\n")

    def progress(stage):
        summary.update(status=stage, updated_at=now(), elapsed_seconds=round(time.monotonic()-started, 1))
        save(output / "summary.json", summary)

    progress("training")
    exclude = {c["case_id"]: read(ROOT / c["physchem"])["identity"]["isomeric_smiles"] for c in protocol["cases"]}
    learned = run_parameter_pilot(WHEEL, exclude_smiles=exclude, train_limit=protocol["train_limit"],
                                  test_limit=protocol["parameter_test_limit"], seed=protocol["seed"], n_jobs=3)
    save(output / "parameter_split.json", learned.audit)
    joblib.dump(learned.models, output / "parameter_models.joblib", compress=3)
    summary.update(parameter_metrics=learned.metrics, parameter_test_predictions=learned.predictions)
    shared = {"split": ref(output / "parameter_split.json", "Shared scaffold split and all clinical exclusions"),
              "model": ref(output / "parameter_models.joblib", "Shared RF model fitted on allowed compounds")}
    constructed = {c["case_id"]: construct(c, learned, output, shared) for c in protocol["cases"]}
    input_paths = [output / "parameter_split.json", output / "parameter_models.joblib", *sorted((output / "inputs").rglob("*.json"))]
    summary["locks"]["inputs"] = {str(p.relative_to(ROOT)): sha256(p) for p in input_paths}
    progress("development_fits_and_rf_predictions")

    def fit(case, formula):
        req = deepcopy(constructed[case["case_id"]][1]); req["partition_method"] = formula
        folder = output / "fits" / case["case_id"] / formula
        try:
            result = fit_development_mapping(reference_request=req, training_source=case["development_source"],
                vss_anchor=None, output_dir=folder, project_root=ROOT,
                max_evaluations=protocol["fit_calls_per_formula"], recover_failed_trials=False)
            result["mapping_path"] = str((folder / "mapping_locked.json").relative_to(ROOT))
            return result
        except Exception as exc:
            return {"status": "failed", "eligible": False, "reason": type(exc).__name__ + ": " + str(exc)}

    def rf(case):
        target, _, source = constructed[case["case_id"]]
        try:
            return run_ml_direct_case(request=target, predicted_pk_source=source,
                output_dir=output / "predictions" / case["case_id"] / "rf", project_root=ROOT)
        except Exception as exc:
            return {"status": "failed", "reason": type(exc).__name__ + ": " + str(exc)}

    with ThreadPoolExecutor(max_workers=protocol["workers"]) as pool:
        jobs = {}
        for case in protocol["cases"]:
            for formula in FORMULAS:
                jobs[pool.submit(fit, case, formula)] = ("fits", case["case_id"], formula)
            jobs[pool.submit(rf, case)] = ("results", case["case_id"], "rf")
        for future in as_completed(jobs):
            group, cid, method = jobs[future]
            result = future.result()
            summary[group].setdefault(cid, {})[method] = result
            progress("development_fits_and_rf_predictions")
            print(json.dumps({"case": cid, "method": method, "status": result["status"]}), flush=True)
    verify(protocol["pins"]); verify(summary["locks"]["inputs"])
    summary["selection"] = {}
    summary["locks"]["mappings"] = {}
    for case in protocol["cases"]:
        cid = case["case_id"]
        fits = summary["fits"][cid]
        for result in fits.values():
            if result.get("mapping_path"):
                summary["locks"]["mappings"][result["mapping_path"]] = sha256(ROOT / result["mapping_path"])
        eligible = [(f, r) for f, r in fits.items() if r.get("eligible")]
        selected = min(eligible, key=lambda x: (x[1]["decision"]["objective"], x[0] != "PKSimStandard")) if eligible else None
        summary["selection"][cid] = {"locked_at": now(), "formula": selected[0] if selected else None,
                                    "mapping_path": selected[1]["mapping_path"] if selected else None}
    progress("predicting_ours")

    def ours(case):
        cid = case["case_id"]; selected = summary["selection"][cid]
        if selected["formula"] is None:
            return {"status": "failed", "reason": "Neither development formula yielded a feasible fit"}
        req = deepcopy(constructed[cid][0]); req["partition_method"] = selected["formula"]
        folder = output / "predictions" / cid / "ours"
        try:
            result = predict_with_development_mapping(request=req, mapping_path=ROOT / selected["mapping_path"],
                                                      output_dir=folder, project_root=ROOT)
            result.update(prediction_path=str((folder / "prediction_locked.json").relative_to(ROOT)),
                          endpoints_path=str((folder / "endpoints.json").relative_to(ROOT)),
                          partition_method=selected["formula"])
            return result
        except Exception as exc:
            return {"status": "failed", "reason": type(exc).__name__ + ": " + str(exc)}

    with ThreadPoolExecutor(max_workers=protocol["workers"]) as pool:
        jobs = {pool.submit(ours, c): c["case_id"] for c in protocol["cases"]}
        for future in as_completed(jobs):
            summary["results"][jobs[future]]["ours"] = future.result()
            progress("predicting_ours")
    summary["locks"]["predictions"] = {}
    for cid, results in summary["results"].items():
        summary["locks"]["predictions"][cid] = {}
        for method, result in results.items():
            if result.get("status") == "predicted":
                summary["locks"]["predictions"][cid][method] = ref(result["prediction_path"], "Native prediction locked before clinical scoring")
    verify(protocol["pins"]); verify(summary["locks"]["inputs"]); verify(summary["locks"]["mappings"])
    summary.update(finished_at=now(), evaluation_outcomes_read_during_fitting=False,
                   all_prediction_attempts_locked_before_scoring=True)
    # Keep failed diagnostics and each formula's selected trial, remove redundant trial assets.
    removed = []
    for cid, fits in summary["fits"].items():
        for formula, result in fits.items():
            if result.get("eligible"):
                chosen = result.get("decision", {}).get("evaluation")
                successful_trials = {row["evaluation"] for row in result.get("evaluations", [])
                                     if row.get("status") == "ok" and row.get("eligible_for_selection")}
                import shutil
                for trial in sorted((output / "fits" / cid / formula).glob("attempt_[0-9][0-9]")):
                    number = int(trial.name.split("_")[-1])
                    if number != chosen and number in successful_trials and not trial.is_symlink():
                        if not trial.resolve().is_relative_to(output.resolve()):
                            raise ValueError("Unsafe trial cleanup path")
                        removed.append(str(trial.relative_to(ROOT))); shutil.rmtree(trial)
    summary["removed_unselected_trial_directories"] = removed
    progress("predictions_locked")
    (output / "runner.pid").unlink(missing_ok=True)


def execute(output):
    """Record fatal failures and release only the PID claimed by this process."""
    pid_path = output / "runner.pid"

    def owns_pid():
        return pid_path.is_file() and pid_path.read_text().strip() == str(os.getpid())

    try:
        return _execute(output)
    except BaseException as exc:
        if owns_pid():
            summary = read(output / "summary.json")
            summary.update(status="failed", failed_at=now(),
                           failure={"type": type(exc).__name__, "reason": str(exc),
                                    "stage": summary.get("status")})
            save(output / "summary.json", summary)
        raise
    finally:
        if owns_pid():
            pid_path.unlink()


def clinical_unit_factor(key, unit, weight):
    """Convert prediction units to the original clinical unit; never transform O by mean weight."""
    normalized = unit.replace("*", ".").replace("·", ".")
    if key in {"auc", "auc_inf"}:
        allowed = {"ng.h/mL": 1., "mg.h/L": .001, "ug.h/mL": .001}
    elif key in {"cmax", "c_last", "curve"}:
        allowed = {"ng/mL": 1., "mg/L": .001, "ug/mL": .001, "ng/L": 1000.}
    elif key == "clearance":
        allowed = {"L/h/kg": 1., "L/h": weight, "mL/min/kg": 1000/60, "mL/min": weight*1000/60}
    elif key in {"vz", "vss"}:
        allowed = {"L/kg": 1., "L": weight, "mL/kg": 1000.}
    else:
        allowed = {"h": 1., "min": 60.}
    if normalized not in allowed:
        raise ValueError("Unsupported clinical unit conversion: " + key + "/" + unit)
    return allowed[normalized]


def predicted_nca(*, policy, **prediction_inputs):
    """Use the declared study integration rule on predictions only.

    The existing estimator still chooses sampling, BLQ handling and the terminal
    regression. Pletz2004 specifies linear AUC trapezoids; its matching AUMC here
    uses trapezoids of time multiplied by concentration. No observed PK value or
    concentration is accepted by the estimator or used to choose these rules.
    """
    from pbpk_kb.blueprints.clinical_prediction_nca import compute_clinical_prediction_nca

    result = compute_clinical_prediction_nca(**prediction_inputs)
    if policy.get("integration") != "linear_trapezoidal":
        return _apply_endpoint_protocol(result, policy, prediction_inputs)
    diagnostics, endpoints = result["diagnostics"], result["endpoints"]
    diagnostics.update(integration="Linear trapezoids of C for AUC and t*C for AUMC on the same retained prediction samples",
                       integration_policy="linear_trapezoidal",
                       integration_implementation="scripts/run_clinical_method_comparison.py::predicted_nca")
    times = diagnostics.get("nca_sample_time_h", [])
    values = diagnostics.get("nca_sample_concentration_ng_ml", [])
    if len(times) < 2:
        return _apply_endpoint_protocol(result, policy, prediction_inputs)
    auc = math.fsum((t1-t0)*(c0+c1)/2
                    for t0, t1, c0, c1 in zip(times, times[1:], values, values[1:]))
    aumc = math.fsum((t1-t0)*(t0*c0+t1*c1)/2
                     for t0, t1, c0, c1 in zip(times, times[1:], values, values[1:]))

    def assign(key, value):
        unit = endpoints[key]["unit"]
        if math.isfinite(value) and value > 0:
            endpoints[key] = {"status": "completed", "value": value, "unit": unit}
        else:
            endpoints[key] = {"status": "unavailable", "value": None, "unit": unit,
                              "reason": "Linear-trapezoid predicted " + key + " is not positive and finite"}

    assign("auc", auc)
    diagnostics.update(auc_last_ng_h_ml=auc, aumc_last_ng_h2_ml=aumc)
    lam = diagnostics.get("lambda_z_per_h")
    if lam is not None:
        tail_auc = values[-1]/lam
        tail_aumc = values[-1]*(times[-1]/lam + 1/lam**2)
        auc_inf, aumc_inf = auc+tail_auc, aumc+tail_aumc
        cl_l_h = diagnostics["dose_ng"]/auc_inf/1000
        mrt = aumc_inf/auc_inf-diagnostics["infusion_h"]/2
        weight = diagnostics["weight_kg"]
        for key, value in {"auc_inf": auc_inf, "clearance": cl_l_h/weight,
                           "vz": cl_l_h/lam/weight, "mrt": mrt,
                           "vss": cl_l_h*mrt/weight}.items():
            assign(key, value)
        diagnostics.update(auc_infinity_ng_h_ml=auc_inf, aumc_infinity_ng_h2_ml=aumc_inf,
                           auc_extrapolated_fraction=tail_auc/auc_inf,
                           aumc_extrapolated_fraction=tail_aumc/aumc_inf,
                           absolute_clearance_l_h=cl_l_h, uncorrected_residence_time_h=aumc_inf/auc_inf)
    # A changed AUMC can change the MRT availability; do not retain an obsolete
    # failure from the default integrator or turn a missing terminal fit into success.
    diagnostics["failures"] = list(dict.fromkeys(
        row["reason"] for row in endpoints.values() if row["status"] != "completed" and row.get("reason")))
    result["status"] = "completed" if all(row["status"] == "completed" for row in endpoints.values()) else "partial"
    return _apply_endpoint_protocol(result, policy, prediction_inputs)


def _apply_endpoint_protocol(result, policy, prediction_inputs):
    """Match explicitly reported fixed windows and residence-time definitions.

    Only frozen predictions and protocol constants are accepted. Fixed-window
    metrics retain actual model concentrations, including values below LLOQ;
    they are distinct from the last-quantifiable NCA used for terminal metrics.
    No reference endpoint, observed concentration, or error selects a window.
    """
    from pbpk_kb.blueprints.clinical_prediction_nca import _sample, _integrals

    keys = {"auc_window_end_h", "c_last_time_h", "mrt_infusion_correction"}
    if not keys.intersection(policy):
        return result  # Byte-for-byte values/diagnostics of the earlier protocol.
    endpoints, diag = result["endpoints"], result["diagnostics"]
    times, values = prediction_inputs["time_h"], prediction_inputs["concentration_ng_ml"]
    overrides = {}

    def endpoint(key, value):
        if not math.isfinite(value) or value <= 0:
            endpoints[key] = {"status": "unavailable", "value": None, "unit": endpoints[key]["unit"],
                              "reason": "Nonpositive/nonfinite protocol-matched prediction: " + key}
        else:
            endpoints[key] = {"status": "completed", "value": value, "unit": endpoints[key]["unit"]}

    def horizon(key):
        value = policy[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid fixed prediction time: " + key)
        if times[0] > 0 or value > times[-1] or value > max(policy["sampling_time_h"]):
            raise ValueError("Fixed endpoint window is outside the frozen prediction/sampling protocol")
        return float(value)

    if "auc_window_end_h" in policy:
        end = horizon("auc_window_end_h")
        grid = sorted({0., end, *(float(t) for t in policy["sampling_time_h"] if 0 < t < end)})
        sampled = [_sample(times, values, t) for t in grid]
        if policy.get("integration") == "linear_trapezoidal":
            area = math.fsum((b-a)*(x+y)/2 for a,b,x,y in zip(grid,grid[1:],sampled,sampled[1:]))
        else:
            area = _integrals(grid, sampled)[0]
        endpoint("auc", area)
        overrides["auc"] = {"window_h": [0., end], "sampling_time_h": grid,
                            "below_lloq": "Actual model values retained; no substituted clinical values",
                            "does_not_change_infinite_auc_or_terminal_fit": True}
    if "c_last_time_h" in policy:
        end = horizon("c_last_time_h")
        endpoint("c_last", _sample(times, values, end))
        overrides["c_last"] = {"time_h": end, "definition": "Concentration at fixed protocol time, not last quantifiable prediction"}
    correction = policy.get("mrt_infusion_correction", "subtract_half_infusion")
    if correction not in {"subtract_half_infusion", "none"}:
        raise ValueError("Unknown MRT infusion correction")
    if correction == "none":
        mrt, cl = diag.get("uncorrected_residence_time_h"), diag.get("absolute_clearance_l_h")
        if mrt is not None and cl is not None:
            endpoint("mrt", mrt)
            endpoint("vss", cl*mrt/prediction_inputs["weight_kg"])
        overrides["mrt_vss"] = {"infusion_time_subtracted_h": 0.,
                                "interpretation": "Original-study uncorrected AUMC/AUC and CL*MRT convention; not bolus-equivalent moments"}
    diag["endpoint_protocol_overrides"] = overrides
    diag["failures"] = list(dict.fromkeys(x["reason"] for x in endpoints.values()
                                         if x["status"] != "completed" and x.get("reason")))
    result["status"] = "completed" if all(x["status"] == "completed" for x in endpoints.values()) else "partial"
    return result


def reporting_curve_indices(times, policy, metadata):
    """Retain actual collection coverage separately from nominal NCA times."""
    window = max(max(policy["sampling_time_h"]), metadata["window_end_h"])
    if not times or times[0] > 0 or times[-1] < window:
        raise ValueError("Native prediction does not cover frozen clinical collection window")
    return [i for i, t in enumerate(times) if t <= window or (i > 0 and times[i-1] < window)]


def report(output, preview=False):
    from pbpk_kb.blueprints.clinical_prediction_nca import UNITS
    summary = read(output / "summary.json")
    if summary["status"] not in {"predictions_locked", "scoring_completed", "report_failed", "reported"}:
        raise ValueError("Complete all prediction attempts before reading clinical references")
    protocol = summary["protocol"]
    verify(protocol["pins"]); verify(summary["locks"]["inputs"]); verify(summary["locks"]["mappings"])
    document = {"schema_version": "pbpk_ours_rf_clinical_v1", "title": "PBPK predictions versus clinical references", "cases": []}
    scoring = {}
    for config in protocol["cases"]:
        cid = config["case_id"]
        candidates = read(ROOT / config["reference"])["cases"]
        case = deepcopy(next(c for c in candidates if c["case_id"] == cid))
        if (case["study_id"] != config["study_id"]
                or case["clinical_reference"]["study_id"] != config["study_id"]
                or case["clinical_reference"]["case_id"] != cid):
            raise ValueError("Clinical reference identity differs from the frozen evaluation case")
        policy = config.get("nca_policy", case.get("nca_policy"))
        case["nca_policy"] = deepcopy(policy)
        cutoff = policy.get("lloq_ng_ml")
        cutoff_basis = "reported_lloq"
        if cutoff is None:
            allowed = {
                "reported_lod": ("lod_ng_ml", "lod_used_as_prediction_cutoff", "reported_lod_proxy"),
                "cited_assay_calibration_lower_bound": ("calibration_lower_bound_ng_ml",
                    "assay_calibration_lower_bound_as_cutoff", "cited_assay_calibration_lower_bound_proxy"),
            }
            proxy = allowed.get(policy.get("prediction_cutoff_proxy"))
            if proxy is None or proxy[1] not in case.get("approximations", []):
                raise ValueError("Missing LLOQ needs an explicitly declared cutoff proxy")
            cutoff, cutoff_basis = policy[proxy[0]], proxy[2]
        metadata = config["target_request_metadata"]
        if not (math.isclose(policy["dose_ng"], metadata["dosing"]["dose_mg"]*1e6)
                and math.isclose(policy["infusion_h"], metadata["dosing"]["infusion_minutes"]/60)
                and math.isclose(policy["weight_kg"], metadata["human"]["weight_kg"])):
            raise ValueError("NCA policy differs from frozen administration")
        mw = read(ROOT / config["physchem"])["identity"]["molecular_weight_g_mol"]
        case["predictions"] = {}; scoring[cid] = {}
        for method in METHODS:
            lock = summary["locks"]["predictions"][cid].get(method)
            if not lock:
                case["predictions"][method] = {"status": "failed", "reason": summary["results"][cid][method].get("reason", "Native execution failed")}
                continue
            verify({lock["path"]: lock["sha256"]})
            native = read(ROOT / lock["path"])
            if native["unit"] != "umol/l" or native["time_unit"] != "min":
                raise ValueError("Unexpected native curve units")
            times = [x/60 for x in native["time_min"]]
            concentrations = [x*mw for x in native["concentration"]]
            result = predicted_nca(policy=policy, time_h=times, concentration_ng_ml=concentrations,
                dose_ng=policy["dose_ng"], infusion_h=policy["infusion_h"],
                nominal_times_h=policy["sampling_time_h"], lloq_ng_ml=cutoff, weight_kg=policy["weight_kg"])
            result["diagnostics"].update(reported_lloq_ng_ml=policy.get("lloq_ng_ml"),
                quantification_threshold_basis=cutoff_basis)
            endpoints = {}
            for key, unit in UNITS.items():
                reference = case["clinical_reference"]["endpoints"].get(key) or {}
                unit = reference.get("unit") or unit
                factor = clinical_unit_factor(key, unit, policy["weight_kg"])
                computed = result["endpoints"][key]
                endpoints[key] = {**computed, "unit": unit, "matrix": "plasma", "statistic": "representative_adult",
                                  "value": None if computed["value"] is None else computed["value"]*factor,
                                  "definition": reference.get("definition") or "Standard NCA predicted " + key,
                                  "estimation": "Prediction-only NCA using frozen sampling schedule and LLOQ"}
                if key == "mrt" and "uncertain_mrt_infusion_correction" in case.get("approximations", []):
                    endpoints[key].update(definition="Infusion-corrected MRT from predicted NCA",
                                          reference_definition=reference["definition"])
            prefix = reporting_curve_indices(times, policy, metadata)
            curve_unit = case["clinical_reference"]["curve"]["unit"]
            curve_factor = clinical_unit_factor("curve", curve_unit, policy["weight_kg"])
            prediction = {"schema_version": "pbpk_clinical_prediction_v1", "case_id": cid,
                          "study_id": case["study_id"], "method": method, "native_source": lock,
                          "status": result["status"], "curve": {"time_h": [times[i] for i in prefix],
                          "concentration": [concentrations[i]*curve_factor for i in prefix], "unit": curve_unit,
                          "matrix": "plasma", "statistic": "representative_adult"}, "endpoints": endpoints,
                          "nca_diagnostics": result["diagnostics"], "observed_values_used_for_endpoint_estimation": False,
                          "nca_code": ref(ROOT / "src/pbpk_kb/blueprints/clinical_prediction_nca.py", "Prediction-only NCA")}
            destination = output / "predictions" / cid / (method + "_reported.json")
            if destination.exists() and read(destination) != prediction:
                raise ValueError("Refusing to overwrite a different frozen prediction")
            save(destination, prediction)
            case["predictions"][method] = ref(destination, "Frozen predicted endpoints and curve")
            scoring[cid][method] = {"status": result["status"], "diagnostics": result["diagnostics"]}
        case.setdefault("reference_screen", {})["predictions_not_started"] = False
        document["cases"].append(case)
    save(output / "comparison_input.json", document)
    summary.update(status="scoring_completed", clinical_scoring=scoring, scoring_completed_at=now())
    save(output / "summary.json", summary)
    command = [sys.executable, "-B", str(ROOT / "scripts/render_pbpk_method_comparison.py"),
               "--input", str(output / "comparison_input.json")]
    if preview:
        command.append("--preview")
    try:
        subprocess.run(command, cwd=ROOT, timeout=180, check=True)
    except (subprocess.SubprocessError, OSError) as exc:
        summary.update(status="report_failed", report_failure={"type": type(exc).__name__, "reason": str(exc)})
        save(output / "summary.json", summary)
        raise
    summary.update(status="reported", reported_at=now())
    summary.pop("report_failure", None)
    save(output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "run")
    parser.add_argument("--config", type=Path, default=BASE / "batch_config.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--status", action="store_true")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(); output = inside(args.output, ROOT)
    if args.prepare:
        result = prepare(args.config, output)
        print(json.dumps({"status": "prepared", "cases": len(result["cases"])}))
    elif args.run:
        execute(output)
    elif args.report:
        report(output, args.preview)
    else:
        summary = read(output / "summary.json")
        print(json.dumps({k: summary.get(k) for k in ("status", "elapsed_seconds", "updated_at", "selection")}, indent=2))


if __name__ == "__main__":
    main()
