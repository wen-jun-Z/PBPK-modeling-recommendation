#!/usr/bin/env python3
"""Extend locked recommendation trajectories using existing NCA mathematics.

Prediction only: no reference concentrations, fitting, inference, or simulation.
Native model quantities and clinical-sampling NCA retain different definitions.
All main and supplementary trajectories retain their separate analysis strata.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from export_recommendation_curve_cases import Inputs, OUT, FUP, PINS, one_ref, require, close
from pbpk_kb.blueprints.endpoint_prediction import predict_requested_endpoints
from pbpk_kb.blueprints.extended_prediction import summarize_extended_prediction
from pbpk_kb.blueprints.clinical_prediction_nca import _integrals, _sample, _tail_fit

DEST = ROOT / "results/data/model_recommendation_extended_predictions.json"
ENDPOINTS = {"Cmax": "ng/mL", "Tmax": "h", "AUC_window": "ng.h/mL", "AUC_last": "ng.h/mL",
             "AUC_infinity": "ng.h/mL", "half_life": "h", "CL": "L/h", "Vz": "L", "MRT": "h", "Vss": "L", "C_last": "ng/mL"}
POLICY = {"terminal_selection": "All post-peak/post-infusion quantified suffixes >=3 points; best adjusted R2; longest within 0.0001",
          "minimum_adjusted_r2": .98, "minimum_span_half_lives": 1., "maximum_auc_tail_fraction": .2,
          "maximum_aumc_tail_fraction": .2, "steady_state_relative_cycle_tolerance": .01,
          "observations_used": False, "native_oral_absorption_completion_not_invented": True,
          "dose_time_boundary": "For single-dose oral/infusion only, a missing dose-time NCA boundary may use verified model C(0)=0; never a measured sample or lambda-z fit point",
          "nca_computation_reference": "https://onlinehelp.certara.com/phoenix/8.3/topics/ncacomprules.htm"}


def walk(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield prefix + "." + key, key, item
            if key not in {"source_independence", "sealed_references", "source_files"}:
                yield from walk(item, prefix + "." + key)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from walk(item, prefix + f"[{i}]")


def assay_cutoff(scenario):
    """Use only explicitly recorded quantification limits, never guessed LOD."""
    found = []
    for path, key, value in walk(scenario):
        if key.lower() in {"lloq_ng_ml", "individual_assay_lloq_ng_ml", "assay_lloq_ng_ml"} and isinstance(value, (int, float)) and value > 0:
            found.append((float(value), path))
        elif key.lower() in {"lloq", "quantification_limit"} and isinstance(value, dict):
            factor = {"ng/mL": 1., "ug/mL": 1000., "mg/L": 1000.}.get(value.get("unit"))
            if factor and isinstance(value.get("value"), (int, float)) and value["value"] > 0:
                found.append((factor * value["value"], path))
    values = {v for v, _ in found}
    if len(values) > 1:
        raise ValueError("Conflicting frozen LLOQ values")
    return (next(iter(values)), [path for _, path in found]) if values else (None, [])


def empty(reason):
    return {"value": None, "eligibility": False, "status": "not_estimable", "reason": reason}


def completed(value, definition, **extra):
    if not math.isfinite(value):
        raise ValueError("Nonfinite derived prediction")
    return {"value": float(value), "eligibility": True, "status": "computed", "reason": None,
            "definition": definition, **extra}


def source_body_weight(plan):
    """Only the actual locked model individual; never target/cohort weight."""
    source = plan.get("source_individual")
    if isinstance(source, dict):
        weight = source.get("OriginData", {}).get("Weight", {})
        value, unit = weight.get("Value"), weight.get("Unit")
        location = "source_individual.OriginData.Weight"
    elif isinstance(source, list) and len(source) == 1:
        weight = source[0].get("origin_raw_base_units", {})
        value, unit = weight.get("weight"), weight.get("weightUnit")
        location = "source_individual[0].origin_raw_base_units.weight"
    else:
        return None, None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None, None
    return (value, location) if unit == "kg" and math.isfinite(value) and value > 0 else (None, None)


def normalized_definitions(endpoints, body_weight_kg, dose_mg):
    """Additional units/definitions, preserving the absolute predictions."""
    if body_weight_kg is None:
        return []
    derived = []
    for key in ("CL", "Vz", "Vss", "Cmax", "AUC_window", "AUC_last", "AUC_infinity", "C_last"):
        item = dict(endpoints[key])
        weight_normalized = key in {"CL", "Vz", "Vss"}
        factor = 1 / body_weight_kg if weight_normalized else body_weight_kg / dose_mg
        item["normalization_basis"] = "locked model individual body weight; no target-cohort covariate substitution"
        item["body_weight_kg"] = body_weight_kg
        item["body_weight_normalized"] = weight_normalized
        item["dose_normalization"] = None if weight_normalized else "dose_per_body_weight"
        item["unit"] = ("L/h/kg" if key == "CL" else "L/kg") if weight_normalized else ("kg*ng*h/(mL*mg)" if key.startswith("AUC") else "kg*ng/(mL*mg)")
        if item["value"] is not None:
            item["value"] *= factor
        item["native_prediction"] = dict(item["native_prediction"])
        if item["native_prediction"]["value"] is not None:
            item["native_prediction"]["value"] *= factor
        derived.append(item)
    return derived


def clinical_nca(times, values, schedule, *, dose_mg, infusion_h, route, lloq):
    """Matched single-dose NCA; oral terminal values are apparent, not elimination."""
    names = ("AUC_last", "AUC_infinity", "half_life", "CL", "Vz", "MRT", "Vss")
    out = {key: empty("No declared LLOQ and qualifying clinical terminal samples") for key in names}
    diagnostics = {"policy": POLICY, "route": route, "lloq_ng_ml": lloq, "observed_values_used": False}
    if lloq is None or not schedule or times[0] > 0:
        return out, diagnostics
    if schedule[0] > 0:
        if (route == "oral" or infusion_h > 0) and _sample(times, values, 0.) == 0:
            schedule = [0., *schedule]
            diagnostics["added_dose_time_boundary"] = {"time_h": 0., "concentration_ng_ml": 0.,
                "observed_sample": False, "used_for_terminal_regression": False, "reason": "single-dose oral/infusion model predose verified zero"}
        else:
            return out, diagnostics
    sampled = [_sample(times, values, t) for t in schedule]
    quantified = [i for i, c in enumerate(sampled) if schedule[i] > 0 and c >= lloq]
    if not quantified:
        return out, diagnostics
    retained = [0, *quantified]
    t, c = [schedule[i] for i in retained], [sampled[i] for i in retained]
    auc, aumc = _integrals(t, c)
    out["AUC_last"] = completed(auc, "Predicted last quantifiable nominal sample, linear-up/log-down NCA; not individual observed tlast", window_h=[0., t[-1]],
                                comparison_qualification="prediction_cutoff_approximation_requires_explicit_reference_match")
    peak = max(quantified, key=lambda i: sampled[i])
    tail = [i for i in quantified if schedule[i] > max(schedule[peak], infusion_h)]
    fits = []
    for n in range(3, len(tail) + 1):
        ix = tail[-n:]
        fit = _tail_fit([schedule[i] for i in ix], [sampled[i] for i in ix])
        if fit:
            fits.append(fit)
    if not fits:
        diagnostics["failure"] = "Fewer than three negative-slope quantified post-peak tail samples"
        return out, diagnostics
    maximum = max(r["adjusted_r_squared"] for r in fits)
    fit = max((r for r in fits if r["adjusted_r_squared"] >= maximum - .0001), key=lambda r: r["n"])
    lam = fit["lambda_z_per_h"]
    half = math.log(2) / lam
    span = (fit["end_h"] - fit["start_h"]) / half
    tail_auc, tail_aumc = c[-1] / lam, c[-1] * (t[-1] / lam + 1 / lam**2)
    infinity, moment = auc + tail_auc, aumc + tail_aumc
    diagnostics.update(selected_tail=fit, terminal_span_in_half_lives=span,
                       auc_extrapolated_fraction=tail_auc / infinity, aumc_extrapolated_fraction=tail_aumc / moment,
                       nca_sample_time_h=t, tlast_h=t[-1], terminal_quantity="apparent_terminal_half_life" if route == "oral" else "terminal_half_life")
    if fit["adjusted_r_squared"] < POLICY["minimum_adjusted_r2"] or span < 1:
        diagnostics["failure"] = "Terminal fit fails fixed adjusted-R2 or one-half-life-span screen"
        for key in names[1:]: out[key] = empty(diagnostics["failure"])
        return out, diagnostics
    out["half_life"] = completed(half, "ln(2)/lambda_z from fixed clinical-sampling BestFit rule; oral value is apparent terminal half-life")
    if tail_auc / infinity > .2:
        for key in ("AUC_infinity", "CL", "Vz", "MRT", "Vss"): out[key] = empty("AUC extrapolation exceeds 20%")
        return out, diagnostics
    cl = dose_mg * 1000 / infinity
    out["AUC_infinity"] = completed(infinity, "AUC to predicted last quantifiable sample plus Clast/lambda_z", window_h=[0., None])
    out["CL"] = completed(cl, "Dose/AUCinf; CL/F for oral, systemic CL for intravenous", window_h=[0., None])
    out["Vz"] = completed(cl / lam, "CL/lambda_z (IV) or (CL/F)/lambda_z (oral)", window_h=[0., None])
    if tail_aumc / moment > .2:
        out["MRT"], out["Vss"] = empty("AUMC extrapolation exceeds 20%"), empty("AUMC extrapolation exceeds 20%")
    else:
        mrt = moment / infinity - (infusion_h / 2 if route == "intravenous" else 0)
        if mrt > 0:
            out["MRT"] = completed(mrt, "AUMCinf/AUCinf; IV subtracts half infusion, oral includes absorption", window_h=[0., None])
            out["Vss"] = completed(cl * mrt, "CL times infusion-corrected MRT; intravenous only", window_h=[0., None]) if route == "intravenous" else empty("Oral plasma NCA does not identify Vss without absorption/bioavailability information")
    return out, diagnostics


def interval_clearance(times, values, contract):
    """Optional steady-periodic CL/F: complete three-cycle consistency required."""
    administration = np.asarray(contract["administration_times_min"], float) / 60
    origin = contract["observation_origin_min"] / 60
    if len(administration) < 4 or not np.isclose(administration, origin, rtol=0, atol=1e-9).any():
        return empty("A verified repeated administration interval is absent"), {}
    gaps = np.diff(administration)
    tau = float(gaps[0])
    if tau <= 0 or not np.allclose(gaps, tau, rtol=0, atol=1e-8) or origin < 2 * tau or times[0] > -2 * tau or times[-1] < tau:
        return empty("Three complete equal dosing intervals are not covered"), {}
    # Compare corresponding half-open cycles: the final boundary may contain the
    # next administration in earlier cycles but not after the last planned dose.
    phases = np.linspace(0, tau, 101, endpoint=False)
    profiles = [np.interp(phases + shift, times, values) for shift in (-2 * tau, -tau, 0)]
    scale = max(float(np.max(profiles[-1])), 1e-30)
    change = max(float(np.max(np.abs(a - b))) / scale for a, b in zip(profiles, profiles[1:]))
    diagnostics = {"interval_h": tau, "three_cycle_max_concentration_relative_change": change,
                   "tolerance": .01, "steady_state_evidence": "model periodicity only, not an assumption that clinical subjects reached steady state"}
    if change > .01:
        return empty("Consecutive model intervals fail predeclared 1% periodicity check"), diagnostics
    grid = [0., *[t for t in times if 0 < t < tau], tau]
    concentrations = [_sample(times, values, t) for t in grid]
    auc = float(np.trapezoid(concentrations, grid))
    if auc <= 0:
        return empty("Nonpositive interval exposure"), diagnostics
    return completed(contract["dose_mg"] * 1000 / auc, "Dose/AUCtau after same-dose schedule and three-cycle model periodicity verification",
                     window_h=[0., tau], steady_state=True), diagnostics


def derive(prediction, plan, scenario, clinical_protocol=None):
    times = list(map(float, prediction["time_h"]))
    values = list(map(float, prediction["concentration_ng_ml"]))
    contract = plan["contract"]
    route = {"Oral": "oral", "Intravenous": "intravenous"}.get(contract["route"])
    require(route is not None, "Unsupported administration route")
    schedule = sorted(set(float(t) / 60 for t in contract["sampling_times_min"] if t >= 0))
    require(len(schedule) >= 2 and times[0] <= schedule[0] and times[-1] >= schedule[-1], "Clinical window is not covered")
    native_curve = prediction["native_curve"]
    require(native_curve["time_unit"] == "min" and native_curve["unit"] == "umol/l", "Native curve units differ")
    mw = float(prediction["molecular_weight_g_per_mol"])
    require(mw > 0 and len(times) == len(values) == len(native_curve["time_min"]) == len(native_curve["concentration"]), "Unaligned native curve")
    require(np.isfinite(times).all() and np.isfinite(values).all() and min(values) >= 0 and np.all(np.diff(times) > 0), "Invalid native trajectory")
    origin = float(prediction["time_origin_h_after_first_dose"])
    require(np.allclose(np.asarray(native_curve["time_min"]) / 60 - origin, times, rtol=1e-10, atol=1e-9), "Native time conversion differs")
    require(np.allclose(np.asarray(native_curve["concentration"]) * mw, values, rtol=1e-10, atol=1e-9), "Native concentration conversion differs")
    # Rebase the already observed-origin curve only for the finite-window API;
    # full negative-time dosing history is retained in the exported trajectory.
    start, end = max(0., schedule[0]), schedule[-1]
    subset_t = [start, *[t for t in times if start < t < end], end]
    subset_c = [_sample(times, values, t) / mw for t in subset_t]
    curve = {"time_min": [t * 60 for t in subset_t], "concentration": subset_c, "time_unit": "min", "unit": "umol/l"}
    direct = predict_requested_endpoints(curve, {"endpoints": ["cmax", "tmax", "auc", "c_last"], "primary_endpoint": "auc",
        "window_start_h": start, "window_end_h": end}, mw)["endpoints"]
    native = {"Cmax": completed(direct["cmax"]["value_mg_per_l"] * 1000, "native grid maximum in requested window"),
              "Tmax": completed(direct["tmax"]["value_h"], "earliest native grid maximum; time after observation origin"),
              "AUC_window": completed(direct["auc"]["value_mg_h_per_l"] * 1000, "linear trapezoidal native AUC in exact finite window"),
              "C_last": completed(direct["c_last"]["value_mg_per_l"] * 1000, "native concentration at exact requested end")}
    out = {key: empty("No qualified terminal or last-quantifiable prediction") for key in ENDPOINTS}
    protocol_schedule = (clinical_protocol or {}).get("nominal_times_h")
    clinical_sampling_qualified = bool(protocol_schedule) or (prediction.get("sampling_grid_registration") or {}).get("reported_clinical_sampling_grid_known") is True
    matched_schedule = sorted(set(float(t) for t in protocol_schedule if t >= 0)) if protocol_schedule else schedule
    require(matched_schedule and times[0] <= matched_schedule[0] and times[-1] >= matched_schedule[-1], "Original-study sampling exceeds prediction horizon")
    sampled = [_sample(times, values, t) for t in matched_schedule]
    peak = int(np.argmax(sampled))
    out.update(Cmax=completed(sampled[peak], "maximum at frozen nominal clinical sampling times"),
               Tmax=completed(matched_schedule[peak], "earliest maximum at frozen nominal clinical sampling times; score by absolute hours"),
               AUC_window=dict(native["AUC_window"]), C_last=dict(native["C_last"]))
    if not clinical_sampling_qualified:
        for key in ("Cmax", "Tmax"):
            out[key] = dict(native[key])
    dosing_times = contract["administration_times_min"]
    single = len(dosing_times) == 1
    cutoff, cutoff_paths = assay_cutoff(scenario)
    if clinical_protocol and clinical_protocol.get("lloq_ng_ml") is not None:
        declared = float(clinical_protocol["lloq_ng_ml"])
        if cutoff is not None:
            close(cutoff, declared, "Existing and newly source-verified LLOQ differ")
        cutoff = declared
        cutoff_paths.append("source_verified_clinical_protocol.lloq_ng_ml")
    target_steady_state = bool((clinical_protocol or {}).get("steady_state", False))
    diagnostics = {"lloq_metadata_locations": cutoff_paths, "single_dose": single,
                   "clinical_sampling_qualified": clinical_sampling_qualified,
                   "target_regimen_steady_state": target_steady_state,
                   "steady_state_definition": "The source study target regimen; model periodicity is independently checked only for dose/AUCtau clearance"}
    if single:
        if clinical_sampling_qualified:
            nca, diag = clinical_nca(times, values, matched_schedule, dose_mg=contract["dose_mg"],
                infusion_h=float(contract.get("infusion_minutes") or 0) / 60, route=route, lloq=cutoff)
        else:
            nca = {key: empty("Source clinical sampling grid is unreported; numerical execution mesh is not clinical sampling")
                   for key in ("AUC_last", "AUC_infinity", "half_life", "CL", "Vz", "MRT", "Vss")}
            diag = {"failure": "Source clinical sampling grid is unreported", "observed_values_used": False}
        out.update(nca); diagnostics["clinical_nca"] = diag
        if route == "intravenous":
            request = {"dosing": {"route": route, "single_dose": True, "dose_mg": contract["dose_mg"], "start_h": 0.,
                                  "infusion_minutes": contract.get("infusion_minutes") or 0},
                       "evidence": [{"parameter": "molecular_weight", "value": mw}],
                       "target": {"window_start_h": start, "window_end_h": end, "primary_endpoint": "plasma_concentration_time"}}
            terminal = summarize_extended_prediction(native_curve, request)
            mapping = {"half_life": ("terminal_half_life", "value_h", 1), "AUC_infinity": ("auc_inf", "value_mg_h_per_l", 1000),
                       "CL": ("clearance", "value_l_per_h", 1), "Vz": ("vz", "value_l", 1), "MRT": ("mrt", "value_h", 1), "Vss": ("vss", "value_l", 1)}
            for name, (key, field, factor) in mapping.items():
                record = terminal["endpoints"][key]
                native[name] = completed(record[field] * factor, record.get("definition", "quality-gated native terminal estimate")) if record["status"] == "completed" else empty(record.get("reason", "native tail unavailable"))
            diagnostics["native_terminal"] = terminal["diagnostics"]
    else:
        for name in ("AUC_last", "AUC_infinity", "half_life", "CL", "Vz", "MRT", "Vss"):
            out[name] = empty("Repeated dosing; single-dose infinity/terminal NCA is not applicable")
        out["CL"], diagnostics["steady_state_clearance"] = interval_clearance(times, values, contract)
    for key in ENDPOINTS:
        native.setdefault(key, empty("Native oral terminal elimination is not qualified without declared absorption completion" if route == "oral" else "Native endpoint unavailable"))
        out[key].update(endpoint_key=key, unit=ENDPOINTS[key], native_prediction=native[key],
            prediction_kind="native" if not clinical_sampling_qualified and key in {"Cmax", "Tmax"} else
                ("protocol_matched" if key in {"Cmax", "Tmax", "AUC_last", "AUC_infinity", "half_life", "CL", "Vz", "MRT", "Vss"} else "native_window"),
            body_weight_normalized=False, error_metric="absolute_error_h" if key == "Tmax" else "fold_error")
        out[key].setdefault("window_h", [start, end])
        out[key].setdefault("steady_state", target_steady_state)
        out[key]["matrix"] = prediction["matrix"]
        out[key]["analyte"] = prediction["analyte"]
        out[key]["binding"] = prediction["binding"]
        out[key]["dose_normalization"] = None
        out[key]["quantity"] = ({"CL": "CL/F", "Vz": "Vz/F", "MRT": "MRT_including_absorption", "half_life": "apparent_terminal_half_life"}.get(key, key)
                                if route == "oral" else {"half_life": "terminal_half_life"}.get(key, key))
    return out, diagnostics, {"time_h": times, "concentration_ng_ml": values, "dose_mg": contract["dose_mg"],
        "route": route, "time_origin_h_after_first_dose": origin, "molecular_weight_g_per_mol": mw,
        "sampling_time_h": matched_schedule, "lloq_ng_ml": cutoff, "steady_state": target_steady_state,
        "clinical_sampling_qualified": clinical_sampling_qualified,
        "contract": contract, "compatibility": prediction["compatibility"],
        "matrix": prediction["matrix"], "analyte": prediction["analyte"], "binding": prediction["binding"]}


def build():
    inputs = Inputs()
    documents = {p: inputs.read({"path": p, "sha256": h}) for p, h in PINS.items()}
    report = documents[str(FUP / "primary/results.json")]
    accounting = documents[str(FUP / "final_accounting.json")]
    protocol = inputs.read({"path": str(OUT / "protocol.json"), "sha256": report["locks"]["protocol_sha256"]})
    native_lock = inputs.read({"path": str(OUT / "native/predictions_lock.json"), "sha256": report["locks"]["predictions_lock_sha256"]})
    require(accounting["completed"] and documents[str(FUP / "audit.json")]["accepted"], "Incomplete main execution")
    require(native_lock["protocol"]["sha256"] == report["locks"]["protocol_sha256"] and not native_lock["target_observations_read"], "Native protocol barrier differs")
    def lines(path):
        return [json.loads(line) for line in inputs.check(one_ref(protocol["inputs"], path)).read_text().splitlines()]
    scenarios = {x["scenario_id"]: x for x in lines(OUT / "inputs/scenarios.jsonl")}
    candidates = {x["candidate_id"]: x for x in lines(OUT / "rankings/inputs/candidates.jsonl")}
    clinical_protocols = {}
    source_file = ROOT / "results/data/model_recommendation_extended_references.json"
    if source_file.exists():
        # Explicit allowlist: endpoint reference values are never model/NCA inputs.
        source_document = inputs.read({"path": str(source_file.relative_to(ROOT)), "sha256": hashlib.sha256(source_file.read_bytes()).hexdigest()})
        for clinical in source_document["clinical_protocols"]:
            for key in ("lloq_source", "sampling_source"):
                ref = clinical.get(key)
                if ref:
                    inputs.check({"path": ref["path"], "sha256": ref["sha256"]})
            clinical_protocols[clinical["scenario_id"]] = clinical
        del source_document
    jobs = {}
    for job in native_lock["jobs"]:
        refs = {Path(ref["path"]).name: ref for ref in job["artifacts"]}
        if "prediction.json" in refs:
            jobs[(job["scenario_id"], job["candidate_id"])] = {"prediction": refs["prediction.json"], "plan": refs["execution_plan.json"], "prediction_lock": refs["prediction_lock.json"]}
    repairs = inputs.read(report["post_scoring_followup"]["execution_completion"])
    require(repairs["completed"] and len(repairs["jobs"]) == 16, "Main repair completion differs")
    for job in repairs["jobs"]:
        if job.get("prediction"):
            require(job["numeric_replay_verified"], "Repair lacks independent numerical replay")
            jobs[(job["scenario_id"], job["candidate_id"])] = {key: job[key] for key in ("prediction", "plan", "prediction_lock")}
    expected = {(r["scenario_id"], r["candidate_id"]) for r in accounting["all_original_pairs"] if r["executed_in_main"]}
    require(set(jobs) == expected and len(jobs) == 136, "Main prediction accounting incomplete")
    pending = [(sid, cid, "main", refs, None) for (sid, cid), refs in sorted(jobs.items())]
    for row in accounting["all_original_pairs"]:
        for supplement in row["successful_supplements"]:
            refs = supplement["artifacts"]
            if "prediction" in refs:
                inputs.check(supplement["receipt"])
                pending.append((row["scenario_id"], row["candidate_id"], supplement["analysis_stratum"],
                                {key: refs[key] for key in ("prediction", "plan", "prediction_lock")}, supplement.get("variant")))
    predictions, trajectories = [], []
    for parent_sid, cid, stratum, refs, variant in pending:
        prediction, plan, lock = [inputs.read(refs[key]) for key in ("prediction", "plan", "prediction_lock")]
        sid = prediction["scenario_id"]
        require((plan["scenario_id"], plan["candidate_id"]) == (sid, cid) == (lock["scenario_id"], lock["candidate_id"]), "Artifact identity mismatch")
        require(prediction["candidate_id"] == cid and (sid == parent_sid if stratum == "main" else sid.startswith(parent_sid)), "Parent scenario mismatch")
        require(lock["execution_verified"] is True and lock["sealed_observations_accessed"] is False, "Prediction was not outcome-free and verified")
        for key in ("prediction", "plan"):
            require(lock["files"].get(refs[key]["path"]) == refs[key]["sha256"], "Global/local prediction lock mismatch")
        require(plan["source_sha256"] == lock["source_sha256"], "Model source checksum mismatch")
        inputs.check({"path": plan["source_path"], "sha256": plan["source_sha256"]})
        require(prediction["matrix"] in {"plasma", "whole_blood"} and prediction["binding"] == "total", "Unqualified biological observation output")
        require(prediction["compatibility"]["drug_identity_match"] and prediction["compatibility"]["route_match"], "Identity/route mismatch")
        scenario = scenarios[parent_sid]
        if stratum == "main":
            require(plan["scenario_sha256"] == hashlib.sha256(json.dumps(scenario, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "Scenario digest mismatch")
            require(plan["source_sha256"] == candidates[cid]["provenance"]["source_sha256"], "Catalogue source mismatch")
        require(plan["contract"]["dose_mg"] > 0 and plan["compatibility"]["dose_match_planned"], "Dose binding unverified")
        clinical = clinical_protocols.get(parent_sid) if stratum == "main" else None
        endpoints, diagnostics, trajectory = derive(prediction, plan, scenario, clinical)
        identity = {"scenario_id": sid, "parent_scenario_id": parent_sid, "candidate_id": cid,
                    "compound": scenario["compound"], "analysis_stratum": stratum, "variant": variant,
                    "source_refs": refs, "trajectory_index": len(trajectories)}
        weight, weight_location = source_body_weight(plan)
        trajectory.update(body_weight_kg=weight, body_weight_source_field=weight_location,
                          body_weight_source=refs["plan"] if weight is not None else None)
        trajectories.append({**identity, **trajectory, "diagnostics": diagnostics})
        predictions.extend({**identity, **item} for item in endpoints.values())
        predictions.extend({**identity, **item} for item in normalized_definitions(endpoints, weight, trajectory["dose_mg"]))
        if len(trajectories) % 25 == 0:
            print(f"Derived {len(trajectories)}/{len(pending)} locked trajectories", flush=True)
    inputs.closing_check()
    counts = Counter((r["analysis_stratum"], r["endpoint_key"], r["status"]) for r in predictions)
    return {"schema": "recommendation-endpoint-extension-predictions-v1", "state": "complete", "policy": POLICY,
        "trajectories": trajectories, "predictions": predictions,
        "counts": {"main_trajectories": sum(t["analysis_stratum"] == "main" for t in trajectories),
                   "supplement_trajectories": sum(t["analysis_stratum"] != "main" for t in trajectories),
                   "prediction_rows": len(predictions), "base_prediction_rows": len(trajectories) * len(ENDPOINTS),
                   "normalized_definition_rows": len(predictions) - len(trajectories) * len(ENDPOINTS),
                   "original_pairs": len(accounting["all_original_pairs"])},
        "endpoint_status_counts": [{"analysis_stratum": s, "endpoint_key": e, "status": k, "count": n} for (s,e,k),n in sorted(counts.items())],
        "unexecuted_main_pairs": [{"scenario_id": r["scenario_id"], "candidate_id": r["candidate_id"], "status": r["main_native_status"]} for r in accounting["all_original_pairs"] if not r["executed_in_main"]],
        "original_artifacts_unchanged": True, "models_or_parameters_refitted": False, "new_simulations": 0,
        "sealed_observation_files_opened": 0, "source_files": inputs.refs,
        "producer": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}


def publish():
    report = build()
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    result = publish()
    print(json.dumps(result["counts"], indent=2))
