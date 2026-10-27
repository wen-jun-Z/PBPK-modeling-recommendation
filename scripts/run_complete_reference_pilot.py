#!/usr/bin/env python3
"""Prepare/run one study-separated Treprostinil comparison: Ours versus RF.

Clinical evaluation documents are hashed but never decoded by this runner.
Both predictions are frozen before a separate reference-scoring/report step.
Existing learners, development calibration and native execution are reused.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import time

from pbpk_kb.blueprints.source_workflow import inside, save, sha256

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/blueprint_design/clinical_complete_reference_20260928"
WHEEL = ROOT / "artifacts/pk_predictors/pksmart/pksmart-3.0.1-py3-none-any.whl"
FORMULAS = ("PKSimStandard", "RodgersRowland")
COMPOUND_ID = "pubchem:6918140"
SEED = 20260928
FIT_CALLS = 18
ACTIVE_OUTPUT = None


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def ref(path, location):
    path = inside(path, ROOT, exists=True)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "location": location}


def verify(pins):
    for name, digest in pins.items():
        if sha256(inside(name, ROOT, exists=True)) != digest:
            raise ValueError("Frozen input changed: " + name)


def prepare(output, args):
    if output.exists():
        raise FileExistsError("Existing run is immutable; use --run-prepared only for an unstarted preparation")
    physchem = inside(args.physchem, ROOT, exists=True)
    development = inside(args.development, ROOT, exists=True)
    reference = inside(args.reference, ROOT, exists=True)
    target_curve = inside(args.target_curve, ROOT, exists=True) if args.target_curve else None
    # Only independent physicochemical inputs and allowed development data are decoded.
    physical, dev = read(physchem), read(development)
    if physical["identity"]["pubchem_cid"] != 6918140 or dev.get("compound_id") != COMPOUND_ID:
        raise ValueError("Development and physicochemical drug identities must match Treprostinil")
    if (dev.get("role") != "calibration_data_reconstruction" or dev.get("matrix") != "plasma"
            or dev.get("time_unit") != "min" or dev.get("unit") != "umol/l"):
        raise ValueError("Development data must declare plasma, min, umol/l and calibration role")
    from pbpk_kb.blueprints.evidence_evaluation import observed_rows
    observed_rows(dev)
    if development == reference or sha256(development) == sha256(reference):
        raise ValueError("Development and evaluation sources must be distinct")
    if target_curve and (development == target_curve or sha256(development) == sha256(target_curve)):
        raise ValueError("Evaluation curve cannot be the development curve")
    if dev.get("study_id") != "P01:07":
        raise ValueError("This frozen pilot permits the independent P01:07 development study only")
    sources = [Path(__file__).resolve(), WHEEL, physchem, development, reference]
    human_definition = ROOT / "artifacts/pbpk-model-assets/human_base/standard_small_molecule.json"
    sources.append(human_definition)
    for record in read(human_definition)["resources"]:
        path = inside(record["path"], ROOT, exists=True)
        if sha256(path) != record["sha256"]:
            raise ValueError("Registered human physiology resource changed")
        sources.append(path)
    if target_curve:
        sources.append(target_curve)
    sources += sorted((ROOT / "src/pbpk_kb/blueprints").glob("*.py"))
    sources += sorted((ROOT / "src/pbpk_kb/model_runtime/R").glob("*.R"))
    for document in (physical, dev):
        # Optional original-source pins supplied by the curators remain auditable.
        original_sources = list(document.get("source_files", []))
        if isinstance(document.get("source"), dict) and document["source"].get("path"):
            original_sources.append(document["source"])
        for record in original_sources:
            path = inside(record["path"], ROOT, exists=True)
            if record.get("sha256") and sha256(path) != record["sha256"]:
                raise ValueError("Curated original-source hash mismatch")
            sources.append(path)
    evaluation_pdf = BASE / "sources/treprostinil_pmda2023.pdf"
    if evaluation_pdf.exists():
        sources.append(evaluation_pdf)  # Hash only; no evaluation content is decoded.
    pins = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    protocol = {
        "schema_version": "complete-reference-pilot-1", "prepared_at": now(), "seed": SEED,
        "methods": ["rf_direct", "ours_blueprint"], "compound_id": COMPOUND_ID,
        "drug": "treprostinil", "development_study_id": "P01:07",
        "evaluation_study_id": "LRX-TRIUMPH BA.001", "train_limit": 400, "parameter_test_limit": 100,
        "physchem_source": ref(physchem, "Independent structure and physicochemical records"),
        "development_source": {**ref(development, "Independent P01:07 plasma observations"),
                               "role": "calibration_data_reconstruction"},
        "evaluation_reference": ref(reference, "Evaluation scalar references; scoring only"),
        "evaluation_curve": ref(target_curve, "Evaluation curve; scoring only") if target_curve else None,
        "development_point_count": len(dev["observations"]),
        "partition_candidates": list(FORMULAS), "fit_calls_per_formula": FIT_CALLS,
        "selection": "Minimum eligible development objective; PKSimStandard wins exact ties",
        "vss_anchor": None, "failure_recovery": False,
        "clinical_target_values_used_for_fitting_or_selection": False,
        "clinical_outcomes_decoded_by_runner": False,
        "same_rf_clearance_and_fu_priors_for_both_methods": True,
        "comparison_information": "Ours additionally receives a separate same-drug development curve; RF is structure-only. This comparison does not isolate pure algorithm effects.",
        "resources": {"parallel_native_workers": 3, "formula_fit_workers": 2,
                      "native_cpu_threads": 2, "native_timeout_seconds": 180,
                      "max_output_bytes_per_native_call": 64 * 1024**2,
                      "max_native_calls": 2 * FIT_CALLS + 2,
                      "parameter_forest_workers": 3, "LLM_calls": 0,
                      "solver": "CPU CVODES; no GPU implementation"},
        "target_request_metadata": {
            "human": {"age_years": 30.0, "weight_kg": 69.5, "height_cm": 175.0,
                      "sex": "MALE", "population": "European_ICRP_2002", "seed": 17},
            "dosing": {"route": "intravenous", "dose_mg": 15.0 * 69.5 * 60.0 / 1e6,
                       "start_h": 0.0, "infusion_minutes": 60.0},
            "window_end_h": 9.0,
            "demographic_assumptions": ["Age 30, male sex, height 175 cm and default adult population are representative assumptions, not study measurements.",
                                        "Weight 69.5 kg is the declared study-group mean; one adult cannot reproduce intersubject variation."],
        },
        "pins": pins,
    }
    output.mkdir(parents=True)
    save(output / "summary.json", {"status": "prepared", "updated_at": now(), "protocol": protocol,
         "progress": {"stage": "prepared", "fit_jobs_completed": 0, "fit_jobs_total": 2,
                      "prediction_jobs_completed": 0, "prediction_jobs_total": 2},
         "fits": {}, "results": {}, "locks": {}})
    return protocol


def construct_requests(output, protocol, learned):
    from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest, assemble_evidence_blueprint
    from pbpk_kb.blueprints.extended_prediction import IV_ENDPOINTS
    from pbpk_kb.blueprints.development_curve_blueprint import _training_document
    import joblib

    folder = output / "inputs"
    folder.mkdir()
    physical = read(ROOT / protocol["physchem_source"]["path"])
    dev = read(ROOT / protocol["development_source"]["path"])
    predicted = learned.predict(physical["identity"]["isomeric_smiles"])
    if predicted["training_structure_overlap"] or predicted["training_scaffold_overlap"]:
        raise ValueError("Clinical drug/scaffold leaked into parameter training")
    split = folder / "parameter_split.json"
    save(split, learned.audit)
    models = output / "parameter_models.joblib"
    joblib.dump(learned.models, models, compress=3)
    v = predicted["predictions"]
    prediction = {"compound_id": COMPOUND_ID, "method": "random_forest",
        "predictions": {"clearance_ml_min_kg": v["human_CL_mL_min_kg"]["random_forest"],
                        "vss_l_kg": v["human_VDss_L_kg"]["random_forest"],
                        "fraction_unbound": v["human_fup"]["random_forest"]},
        "all_parameter_predictions": predicted, "training_overlap": False,
        "same_scaffold_training_overlap": False,
        "provenance": {"split": ref(split, "Frozen scaffold split and external drug exclusions"),
                       "model": ref(models, "Fixed RF parameters trained only on permitted compounds")},
        "training_medians_log10": learned.medians_log10,
        "label_sources": "Published PKSmart human PK training subset with per-endpoint missing-label masks",
        "no_clinical_outcomes_used": True, "predicted_vss_used_for_pbpk_construction": False}
    prediction_path = folder / "predicted_pk.json"
    save(prediction_path, prediction)
    predicted_source = ref(prediction_path, "Locked structure-derived PK predictions; not measured anchors")
    assumptions_path = folder / "construction_assumptions.json"
    save(assumptions_path, {
        "binding_partner": "Albumin", "binding_partner_basis": "Common structural convention for this acidic compound; not target-study binding data",
        "elimination_scope": "hepatic_only", "elimination_scope_basis": "Lumped systemic clearance assigned to liver; organ fractions are not verified",
        "solubility": physical["optional_iv_construction_assumption"],
        "representative_target": protocol["target_request_metadata"],
        "same_assumptions_for_both_methods": True})
    assumption_source = ref(assumptions_path, "Declared assumptions; none are clinical reference labels")
    evidence = []

    def add(parameter, value, unit, basis, source, notes="", context=None, method=None):
        row = {"parameter": parameter, "compound_id": COMPOUND_ID, "value": value,
               "unit": unit, "basis": basis, "source": source,
               "context": context or {}, "notes": notes}
        if method:
            row["prediction_method"] = method
        evidence.append(row)

    pc_source = protocol["physchem_source"]
    identity = physical["identity"]
    fields = physical["selected_physicochemical_inputs"]
    add("molecular_weight", identity["molecular_weight_g_mol"], "g/mol", "derived", pc_source)
    add("molecular_formula", identity["molecular_formula"], "text", "derived", pc_source)
    add("logp", fields["logp"]["value"], "Log Units", "predicted", pc_source,
        "Computed neutral-species XLogP3-AA, not measured logD", method={
            "name": "PubChem XLogP3-AA", "version": "PubChem retrieval 2026-09-28",
            "training_overlap": "unknown", "applicability": "Chemical descriptor; not clinical labels"})
    add("pka_types", [{"Type": "Acid", "Pka": fields["acidic_pka"]["value"]}], "1", "measured", pc_source,
        "Reported mixed-solvent titration with 20% ethanol; pure-water pKa not independently verified")
    sol = physical["optional_iv_construction_assumption"]
    add("solubility", sol["solubility_mg_ml"], "mg/ml", "assumption", assumption_source,
        "Engineering placeholder for already dissolved IV administration; not experimental solubility")
    add("solubility_ph", sol["solubility_ph"], "1", "assumption", assumption_source)
    add("binding_partner", "Albumin", "text", "assumption", assumption_source)
    add("elimination_scope", "hepatic_only", "text", "assumption", assumption_source,
        "Lumped total plasma clearance; not validated hepatic/renal allocation")
    prediction_method = {"name": "random_forest", "version": "complete-reference-pilot-1",
        "training_overlap": False, "applicability": {"target_structure_and_scaffold_excluded": True,
        "human_IV_parameter_model": True, "prospective_clinical_validation": False}}
    for parameter, key, unit in (("total_clearance", "clearance_ml_min_kg", "ml/min/kg"),
                                 ("fraction_unbound", "fraction_unbound", "fraction")):
        add(parameter, prediction["predictions"][key], unit, "predicted", predicted_source,
            "Same locked RF prediction in both methods; no clinical target labels",
            context={"species": "human", "matrix": "plasma", "route": "intravenous"}, method=prediction_method)
    ledger_path = folder / "evidence_ledger.json"
    save(ledger_path, {"evidence": evidence})
    metadata = protocol["target_request_metadata"]
    forbidden = [protocol["evaluation_reference"]]
    if protocol.get("evaluation_curve"):
        forbidden.append(protocol["evaluation_curve"])
    request = {"entry_reason": "model_missing", "compound_id": COMPOUND_ID,
        "compound_name": "Treprostinil", "human": deepcopy(metadata["human"]),
        "dosing": deepcopy(metadata["dosing"]), "partition_method": "PKSimStandard",
        "target": {"endpoints": list(IV_ENDPOINTS), "primary_endpoint": "plasma_concentration_time",
                   "window_start_h": 0.0, "window_end_h": metadata["window_end_h"]},
        "evidence_ledger": ref(ledger_path, "Immutable shared drug evidence"), "evidence": evidence,
        "forbidden_sources": forbidden,
        "assumptions": [*metadata["demographic_assumptions"],
            "All target-compound structures and scaffold groups are excluded from parameter training and parameter test.",
            "RF is structure-only; Ours additionally uses an independent P01:07 clinical development curve.",
            "Two partition formulas use identical development data; selection never uses BA.001 clinical outcomes.",
            "Vss is recorded as an RF prediction but is not an observed anchor and does not constrain either method.",
            "Solubility is a construction placeholder for already dissolved IV delivery; protein partner and organ allocation are assumptions.",
            "The fixed 9 h output window is declared before scoring; reported AUClast/Clast references require their own evaluation window.",
        ]}
    development_request = deepcopy(request)
    development_request["human"] = deepcopy(dev["human"])
    development_request["dosing"] = deepcopy(dev["dosing"])
    development_request["target"]["window_end_h"] = max(9.0, math.ceil(dev["observations"][-1]["time_min"] / 60))
    development_request["assumptions"] = [
        *dev.get("population_source", {}).get("representative_simulation_assumptions", []),
        "Development age and weight use the P01:07 group means; one representative adult approximates its mixed-sex cohort.",
        *request["assumptions"][len(metadata["demographic_assumptions"]):],
    ]
    for req, name in ((request, "target_request.json"), (development_request, "development_request.json")):
        EvidenceBlueprintRequest.model_validate(req)
        assembled = assemble_evidence_blueprint(req, ROOT)
        if assembled["status"] != "assembled":
            raise ValueError("Evidence assembly is incomplete: " + name)
        save(folder / name, req)
    _training_document(protocol["development_source"], development_request, ROOT)
    pins = {str(p.relative_to(ROOT)): sha256(p) for p in [models, *sorted(folder.glob("*.json"))]}
    return request, development_request, predicted_source, pins


def trim_unselected_trials(output, fits):
    """Remove only this run's reproducible unselected trials; retain decisions/failures."""
    removed = []
    for formula, result in fits.items():
        if not result.get("mapping_path") or not result.get("eligible"):
            continue  # Unexpected failures retain their logs for diagnosis.
        selected = result.get("decision", {}).get("evaluation") if result.get("decision") else None
        folder = output / "fits" / formula
        for trial in sorted(folder.glob("attempt_[0-9][0-9]")):
            if selected == int(trial.name.split("_")[-1]):
                continue
            if trial.is_symlink() or not trial.resolve().is_relative_to(output.resolve()):
                raise ValueError("Unsafe trial cleanup path")
            removed.append(str(trial.relative_to(ROOT)))
            shutil.rmtree(trial)
    return removed


def execute(output, protocol, *, keep_trials=False):
    global ACTIVE_OUTPUT
    summary_path = output / "summary.json"
    summary = read(summary_path)
    if summary["status"] != "prepared":
        raise FileExistsError("This run has already started; a new run must use a new output directory")
    verify(protocol["pins"])
    # Exclusive creation also prevents two simultaneous --run-prepared invocations.
    with (output / "runner.pid").open("x") as handle:
        handle.write(str(os.getpid()) + "\n")
    ACTIVE_OUTPUT = output
    started = time.monotonic()
    summary.update(status="running", started_at=now(), pid=os.getpid())

    def progress(stage, **extra):
        summary.update(updated_at=now(), elapsed_seconds=round(time.monotonic() - started, 1))
        summary["progress"] = {"stage": stage, "fit_jobs_completed": len(summary["fits"]),
            "fit_jobs_total": 2, "prediction_jobs_completed": len(summary["results"]),
            "prediction_jobs_total": 2, **extra}
        save(summary_path, summary)

    from pbpk_kb.blueprints.pk_parameter_learning import run_parameter_pilot
    from pbpk_kb.blueprints.ml_hybrid_blueprint import run_ml_direct_case
    from pbpk_kb.blueprints.development_curve_blueprint import fit_development_mapping, predict_with_development_mapping
    physical = read(ROOT / protocol["physchem_source"]["path"])
    progress("train_parameter_models")
    learned = run_parameter_pilot(WHEEL,
        exclude_smiles={"treprostinil": physical["identity"]["isomeric_smiles"]},
        train_limit=400, test_limit=100, seed=SEED, n_jobs=3)
    summary.update(parameter_split=learned.audit, parameter_metrics=learned.metrics,
                   parameter_test_predictions=learned.predictions)
    target, development, predicted_source, input_pins = construct_requests(output, protocol, learned)
    summary["locks"]["construction_inputs"] = input_pins
    progress("development_fits_and_rf_prediction")

    def fit(formula):
        req = deepcopy(development)
        req["partition_method"] = formula
        folder = output / "fits" / formula
        try:
            result = fit_development_mapping(reference_request=req, training_source=protocol["development_source"],
                vss_anchor=None, output_dir=folder, project_root=ROOT, max_evaluations=FIT_CALLS,
                recover_failed_trials=False)
            result["mapping_path"] = str((folder / "mapping_locked.json").relative_to(ROOT))
            return result
        except Exception as exc:
            return {"status": "failed", "eligible": False, "reason": type(exc).__name__ + ": " + str(exc)}

    def predict_rf():
        try:
            return run_ml_direct_case(request=target, predicted_pk_source=predicted_source,
                output_dir=output / "cases/rf_direct", project_root=ROOT)
        except Exception as exc:
            return {"status": "failed", "reason": type(exc).__name__ + ": " + str(exc)}

    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(fit, formula): ("fit", formula) for formula in FORMULAS}
        jobs[pool.submit(predict_rf)] = ("prediction", "rf_direct")
        for future in as_completed(jobs):
            kind, name = jobs[future]
            result = future.result()
            summary["fits" if kind == "fit" else "results"][name] = result
            progress("development_fits_and_rf_prediction", latest={"job": name, "status": result["status"]})
            print(json.dumps({"job": name, "status": result["status"], "eligible": result.get("eligible")}), flush=True)
    verify(protocol["pins"])
    verify(input_pins)
    eligible = [(formula, result) for formula, result in summary["fits"].items() if result.get("eligible")]
    mapping_pins = {r["mapping_path"]: sha256(ROOT / r["mapping_path"])
                    for r in summary["fits"].values() if r.get("mapping_path")}
    summary["locks"]["mappings"] = mapping_pins
    selected = min(eligible, key=lambda row: (row[1]["decision"]["objective"], row[0] != "PKSimStandard")) if eligible else None
    summary["selection"] = {"locked_at": now(), "formula": selected[0] if selected else None,
        "mapping_path": selected[1]["mapping_path"] if selected else None,
        "basis": "Independent development curve objective only"}
    progress("predict_target_with_frozen_mapping")
    if selected:
        formula, mapping = selected
        req = deepcopy(target)
        req["partition_method"] = formula
        folder = output / "cases/ours_blueprint"
        try:
            result = predict_with_development_mapping(request=req, mapping_path=ROOT / mapping["mapping_path"],
                output_dir=folder, project_root=ROOT)
            result.update(prediction_path=str((folder / "prediction_locked.json").relative_to(ROOT)),
                          endpoints_path=str((folder / "endpoints.json").relative_to(ROOT)),
                          mapping_path=mapping["mapping_path"], partition_method=formula)
        except Exception as exc:
            result = {"status": "failed", "reason": type(exc).__name__ + ": " + str(exc)}
    else:
        result = {"status": "mapping_unavailable", "reason": "Neither formula has an eligible development fit; no fallback"}
    summary["results"]["ours_blueprint"] = result
    locks = {}
    for method, result in summary["results"].items():
        if result.get("status") == "predicted":
            locks[method] = {"prediction": ref(result["prediction_path"], "Target prediction locked before reference scoring"),
                             "endpoints": ref(result["endpoints_path"], "Native predicted endpoints"), "locked_at": now()}
    summary["locks"]["predictions"] = locks
    verify(protocol["pins"])
    verify(input_pins)
    verify(mapping_pins)
    summary.update(status="predictions_ready" if len(locks) == 2 else "predictions_ready_with_failures",
        finished_at=now(), clinical_reference_scoring="pending_separate_scoring_step",
        successful_native_predictions=len(locks), all_predictions_locked_before_scoring=True,
        target_observations_or_scalar_references_read=False,
        limitations=["One study pair for one drug cannot establish generalization or clinical usefulness.",
            "Ours has additional same-drug development observations, whereas RF does not.",
            "Formula selection and calibration use the same development study.",
            "Representative adult demographics, lumped liver clearance and IV solubility are explicit assumptions.",
            "Feasible budget-limited development optimization does not imply numerical convergence."])
    progress(summary["status"])
    if not keep_trials:
        summary["cleanup"] = {"removed_unselected_trial_directories": trim_unselected_trials(output, summary["fits"]),
                              "retained": "Full mapping decisions/trial metrics, selected development trial, both final predictions and provenance"}
        progress(summary["status"])
    return 0 if len(locks) == 2 else 1



def report(output, *, preview=False):
    """Score immutable predictions with the predeclared clinical sampling policy.

    Clinical scalar references remain paper-reported. No simulation, training,
    parameter selection or fitting occurs in this post-prediction step.
    """
    from pbpk_kb.blueprints.clinical_prediction_nca import compute_clinical_prediction_nca
    import subprocess
    import sys
    summary = read(output / "summary.json")
    if summary["status"] not in {"predictions_ready", "predictions_ready_with_failures"}:
        raise ValueError("Finish and lock both prediction attempts before reporting")
    protocol = summary["protocol"]
    verify(protocol["pins"])
    for group in ("construction_inputs", "mappings"):
        verify(summary["locks"].get(group, {}))
    reference_path = ROOT / protocol["evaluation_reference"]["path"]
    document = read(reference_path)
    if len(document["cases"]) != 1:
        raise ValueError("This small pilot expects exactly one clinical case")
    case = document["cases"][0]
    policy = case["nca_policy"]
    target = read(output / "inputs/target_request.json")
    mw = next(e["value"] for e in target["evidence"] if e["parameter"] == "molecular_weight")
    if not (math.isclose(policy["dose_ng"], target["dosing"]["dose_mg"] * 1e6)
            and math.isclose(policy["infusion_h"], target["dosing"]["infusion_minutes"] / 60)
            and math.isclose(policy["weight_kg"], target["human"]["weight_kg"])):
        raise ValueError("Clinical sampling policy and frozen dosing do not agree")
    case["predictions"] = {}
    nca_audit = {}
    for method, native_method in (("ours", "ours_blueprint"), ("rf", "rf_direct")):
        locked = summary["locks"]["predictions"].get(native_method)
        if not locked:
            nca_audit[method] = {"status": "native_prediction_unavailable", "reason": summary["results"][native_method]}
            continue
        verify({item["path"]: item["sha256"] for item in locked.values() if isinstance(item, dict) and "path" in item})
        native = read(ROOT / locked["prediction"]["path"])
        if native.get("unit") != "umol/l" or native.get("time_unit") != "min":
            raise ValueError("Unexpected native curve units")
        times = [v / 60 for v in native["time_min"]]
        concentrations = [v * mw for v in native["concentration"]]
        result = compute_clinical_prediction_nca(time_h=times, concentration_ng_ml=concentrations,
            dose_ng=policy["dose_ng"], infusion_h=policy["infusion_h"],
            nominal_times_h=policy["sampling_time_h"], lloq_ng_ml=policy["lloq_ng_ml"],
            weight_kg=policy["weight_kg"])
        nca_audit[method] = result
        endpoints = {}
        for key, reference in case["clinical_reference"]["endpoints"].items():
            computed = result["endpoints"][key]
            if computed.get("unit") != reference["unit"]:
                raise ValueError("Endpoint unit differs: " + key)
            endpoints[key] = {"value": computed.get("value"), "unit": reference["unit"],
                "definition": reference["definition"], "statistic": "representative_adult", "matrix": "plasma",
                "status": computed.get("status"), "estimation": "Frozen nominal sampling / clinical LLOQ / standard NCA on predicted concentrations; not a reference reconstruction"}
        # Retain only the clinically used prefix here; the full immutable native
        # trajectory remains the reproducible source, never overwritten.
        end = max(policy["sampling_time_h"])
        prefix = [i for i,t in enumerate(times) if t <= end]
        payload = {"schema_version": "pbpk_clinical_prediction_v1", "case_id": case["case_id"],
            "study_id": case["study_id"], "method": method,
            "native_source": locked["prediction"], "status": result["status"],
            "curve": {"time_h": [times[i] for i in prefix],
                      "concentration": [concentrations[i] for i in prefix],
                      "unit": "ng/mL", "matrix": "plasma", "statistic": "representative_adult"},
            "endpoints": endpoints, "nca_diagnostics": result.get("diagnostics", {}),
            "observed_values_used_for_endpoint_estimation": False,
            "nca_code": ref(ROOT / "src/pbpk_kb/blueprints/clinical_prediction_nca.py", "Prediction-only NCA implementation")}
        path = output / ("prediction_" + method + ".json")
        if path.exists() and read(path) != payload:
            raise ValueError("Existing reported prediction differs; do not overwrite")
        save(path, payload)
        case["predictions"][method] = ref(path, "Native prediction and sampling-aligned NCA locked before error reporting")
    case["reference_screen"]["predictions_not_started"] = False
    comparison_path = output / "comparison_input.json"
    save(comparison_path, document)
    summary["clinical_reference_scoring"] = {"status": "endpoint_estimation_completed",
        "reference_file": protocol["evaluation_reference"], "nca": nca_audit,
        "all_10_references_originally_reported": True,
        "approximations": case["approximations"], "methods_information": protocol["comparison_information"]}
    save(output / "summary.json", summary)
    command = [sys.executable, "-B", str(ROOT / "scripts/render_pbpk_method_comparison.py"), "--input", str(comparison_path)]
    if preview:
        command.append("--preview")
    completed = subprocess.run(command, cwd=ROOT, timeout=120, text=True, capture_output=True)
    print(completed.stdout[-6000:])
    if completed.returncode:
        raise RuntimeError("Report renderer did not complete: " + completed.stderr[-2000:])
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "run")
    parser.add_argument("--physchem", type=Path, default=BASE / "sources/treprostinil_physchem.json")
    parser.add_argument("--development", type=Path, default=BASE / "treprostinil_development.json")
    parser.add_argument("--reference", type=Path, default=BASE / "clinical_reference.json")
    parser.add_argument("--target-curve", type=Path, help="Optional separately stored target observations, hashed but never decoded")
    parser.add_argument("--keep-trials", action="store_true", help="Keep reproducible unselected native trials for debugging")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run-prepared", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--report", action="store_true")
    parser.add_argument("--preview", action="store_true", help="One temporary report preview; no per-page images")
    args = parser.parse_args()
    output = inside(args.output, ROOT)
    if args.report:
        return report(output, preview=args.preview)
    if args.status:
        document = read(output / "summary.json")
        status = {key: document.get(key) for key in ("status", "updated_at", "elapsed_seconds", "progress", "selection")}
        status["native_fit_progress"] = {}
        for formula in FORMULAS:
            path = output / "fits" / formula / "progress.json"
            if path.exists():
                progress = read(path)
                status["native_fit_progress"][formula] = {key: progress.get(key) for key in
                    ("status", "native_evaluations", "maximum_evaluations", "eligible")}
        print(json.dumps(status, indent=2))
        return 0
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "2"
    if args.run_prepared:
        protocol = read(output / "summary.json")["protocol"]
    else:
        protocol = prepare(output, args)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "output": str(output)}))
        return 0
    return execute(output, protocol, keep_trials=args.keep_trials)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if ACTIVE_OUTPUT is not None:
            summary_path = ACTIVE_OUTPUT / "summary.json"
            summary = read(summary_path)
            summary.update(status="failed", updated_at=now(), failure_reason=type(error).__name__ + ": " + str(error))
            summary["progress"]["stage"] = "failed"
            save(summary_path, summary)
        raise
