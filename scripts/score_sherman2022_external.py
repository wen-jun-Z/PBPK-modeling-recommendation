#!/usr/bin/env python3
"""Score locked external predictions; never fit parameters or expose rankings.

The sealed output separates sampled Cmax, fixed-window AUC proxies, and any
curated partial concentration curves. Two doses and repeated seeds remain one
external clinical study and one already-developed compound.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from curate_sherman2022_external_data import ROOT, local, sha, validate_lock, write_json


def geometric(values, weights):
    if any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Cannot form geometric mean from nonpositive endpoints")
    return float(np.exp(np.dot(weights, np.log(values))))


def errors(predicted: float, observed: float) -> dict:
    if not (math.isfinite(predicted) and math.isfinite(observed) and predicted > 0 and observed > 0):
        raise ValueError("Invalid endpoint value")
    ratio = predicted / observed
    return {"predicted": predicted, "observed": observed, "prediction_observation_ratio": ratio,
            "absolute_fold_error": max(ratio, 1/ratio),
            "signed_relative_error": ratio-1, "absolute_relative_error": abs(ratio-1),
            "absolute_log_error": abs(math.log(ratio))}


def evaluate(case: dict, design: dict, protocol: dict, observations: dict) -> list[dict]:
    nominal = np.array(design["prediction_contract"]["nominal_sampling_times_h"], dtype=float)
    weights_map = protocol["population_approximation"]["sex_weights"]
    sexes = list(weights_map)
    weights = np.array([weights_map[s] for s in sexes], dtype=float)
    if not np.isclose(weights.sum(), 1) or np.any(weights <= 0):
        raise ValueError("Invalid prespecified population weights")
    threshold = protocol["endpoint_policy"]["lloq_ng_per_ml"] / protocol["endpoint_policy"]["molecular_weight_g_per_mol"]
    endpoints = {e["endpoint"]: e for e in observations["endpoint_rows"]
                 if e["case_id"] == case["case_id"] and e["canonical_unit_representation"]}
    if set(endpoints) != {"cmax", "auc_last"}:
        raise ValueError("Missing or duplicate canonical endpoints")
    points = [p for p in observations["partial_curve_points"]
              if p["case_id"] == case["case_id"] and p["eligible_for_scoring"]]
    model_results = {}
    for key in {m["model_key"] for m in case["models"]}:
        representatives = {m["sex"]: m for m in case["models"] if m["model_key"] == key}
        if set(representatives) != set(sexes):
            raise ValueError("Locked model lacks a sex representative")
        reps, sampled_cmax, dense_cmax, auc48, auc_lloq, tlast = [], [], [], [], [], []
        for sex in sexes:
            m = representatives[sex]
            if m["time_unit"] != "min" or m["unit"] not in {"µmol/l", "umol/L", "µmol/L"}:
                raise ValueError("Prediction units require explicit conversion")
            t = np.array(m["time_min"], dtype=float) / 60
            c = np.array(m["concentration"], dtype=float)
            if len(t) != len(c) or len(t) < 3 or not np.all(np.isfinite(c)) or not np.all(np.isfinite(t)):
                raise ValueError("Invalid locked time series")
            if np.any(np.diff(t) <= 0) or t[0] != 0 or not np.isclose(t[-1], 48) or np.min(c) < -1e-10:
                raise ValueError("Time coverage or concentration invariant failed")
            c = np.maximum(c, 0)
            sampled = np.interp(nominal, t, c)
            measured = nominal[(sampled >= threshold) & (nominal > 0)]
            if not len(measured):
                raise ValueError("No nominal sample exceeds prespecified LLOQ")
            last = float(measured[-1])
            mask = t < last
            integral = float(np.trapezoid(np.r_[c[mask], np.interp(last, t, c)], np.r_[t[mask], last]))
            sampled_cmax.append(float(np.max(sampled[1:])))
            dense_cmax.append(float(np.max(c)))
            auc48.append(float(np.trapezoid(c, t)))
            auc_lloq.append(integral)
            tlast.append(last)
            reps.append((t, c))
        obs_cmax = endpoints["cmax"]["value"] / 1000
        obs_auc = endpoints["auc_last"]["value"] / 1000
        data = {
            "primary_cmax_nominal": {**errors(geometric(sampled_cmax, weights), obs_cmax), "unit": "umol/L",
                                     "population_statistic": "sex-weighted geometric mean of representative sampled maxima",
                                     "sample_size_claim": "one aggregate endpoint in this dose group; representative physiologies approximate unknown group distributions"},
            "secondary_cmax_dense": {**errors(geometric(dense_cmax, weights), obs_cmax), "unit": "umol/L",
                                      "primary": False, "caveat": "dense-grid maximum differs from sampled observed Cmax"},
            "auxiliary_auc48_vs_published_auc_last": {**errors(geometric(auc48, weights), obs_auc), "unit": "h*umol/L",
                "primary": False, "same_observation_window": False,
                "caveat": "fixed 0-48h prediction versus published individual AUClast; not an exact AUC validation"},
            "auxiliary_predicted_lloq_auc_vs_published_auc_last": {**errors(geometric(auc_lloq, weights), obs_auc), "unit": "h*umol/L",
                "primary": False, "same_observation_window": False,
                "representative_predicted_last_nominal_quantifiable_times_h": dict(zip(sexes, tlast)),
                "caveat": "prediction-defined LLOQ/nominal-tlast sensitivity; observed individual tlast and BLQ rules unavailable"},
            "partial_curve": {"available": bool(points), "point_count": len(points),
                              "complete_concentration_curve": False,
                              "extraction_type": "digitized group-mean figure, with explicit unresolved points"},
        }
        if points:
            times = np.array([p["time_h"] for p in points], dtype=float)
            observed = np.array([p["concentration"] for p in points], dtype=float)
            uncertainty = np.array([p["absolute_digitization_error_bound"] for p in points], dtype=float)
            predicted = np.sum([w * np.interp(times, t, c) for w, (t, c) in zip(weights, reps)], axis=0)
            if np.any(observed <= 0) or np.any(predicted <= 0):
                raise ValueError("Partial-curve log metric requires positive points")
            logs = np.log(predicted / observed)
            data["partial_curve"].update({
                "statistic": "sex-weighted arithmetic mean concentration",
                "times_h": times.tolist(), "predicted_umol_per_l": predicted.tolist(),
                "observed_digitized_umol_per_l": observed.tolist(),
                "absolute_digitization_error_bound_umol_per_l": uncertainty.tolist(),
                "geometric_mean_absolute_fold_error": float(np.exp(np.mean(abs(logs)))),
                "rmse_log_concentration": float(np.sqrt(np.mean(logs**2))),
                "rmse_umol_per_l": float(np.sqrt(np.mean((predicted-observed)**2))),
                "within_digitization_interval_fraction": float(np.mean(abs(predicted-observed) <= uncertainty)),
                "sd_se_weighting_used": False,
                "uncertainty_caveat": "digitization coordinate bounds are not clinical variability or confidence intervals",
            })
        model_results[key] = data
    result = []
    for choice in case["selections"]:
        result.append({"study_id": design["study_id"], "case_id": case["case_id"],
                       "method": choice["method"], "seed": choice["seed"],
                       "candidate_id": choice["candidate_id"], "model_key": choice["model_key"],
                       "status": "scored", **model_results[choice["model_key"]]})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", required=True)
    parser.add_argument("--prediction-lock", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--curation-receipt", required=True)
    parser.add_argument("--output", required=True)
    a = parser.parse_args()
    design_path, lock_path, obs_path, receipt_path, out = map(local, [a.design, a.prediction_lock, a.observations, a.curation_receipt, a.output])
    design, lock = validate_lock(design_path, lock_path)
    obs = json.loads(obs_path.read_text())
    curator = json.loads(receipt_path.read_text())
    if curator.get("status") != "curation_complete" or not obs.get("endpoint_transcription_visual_qc_complete"):
        raise ValueError("Curated observations have not completed source/QC verification")
    listed = [f for f in curator["files"] if local(f["path"]) == obs_path]
    if len(listed) != 1 or listed[0]["sha256"] != sha(obs_path) or obs["prediction_lock_sha256"] != sha(lock_path):
        raise ValueError("Observation provenance does not bind this locked prediction set")
    protocol_path = local(lock["protocol_path"])
    if sha(protocol_path) != lock["protocol_sha256"]:
        raise ValueError("Prediction protocol changed after locking")
    protocol = json.loads(protocol_path.read_text())
    rows = []
    for record in lock["predictions"]:
        case = json.loads(local(record["path"]).read_text())
        if case["case_id"] != record["case_id"] or case["protocol_sha256"] != sha(protocol_path):
            raise ValueError("Prediction case/protocol identity mismatch")
        rows.extend(evaluate(case, design, protocol, obs))
    expected = len(design["cases"]) * lock["method_seed_count"]
    if len(rows) != expected or len({(x["method"],x["seed"],x["case_id"]) for x in rows}) != expected:
        raise ValueError("Missing or duplicate method/seed/dose evaluations")
    methods = {}
    for method in sorted({x["method"] for x in rows}):
        members = [r for r in rows if r["method"] == method]
        per_seed = []
        for seed in sorted({r["seed"] for r in members}):
            sr = [r for r in members if r["seed"] == seed]
            metric = float(np.mean([r["primary_cmax_nominal"]["absolute_log_error"] for r in sr]))
            per_seed.append({"seed": seed, "mean_absolute_log_cmax_error_over_two_doses": metric,
                             "geometric_mean_absolute_cmax_fold_error_over_two_doses": math.exp(metric)})
        methods[method] = {"seed_summaries": per_seed,
                           "seed_median_absolute_log_cmax_error": float(np.median([s["mean_absolute_log_cmax_error_over_two_doses"] for s in per_seed])),
                           "inference_unit": "one independent study, one existing compound",
                           "independent_studies_n": 1, "independent_compounds_n": 1,
                           "new_compounds_n": 0, "dose_groups_n": 2,
                           "confidence_interval": None, "significance_test": None,
                           "reason_no_population_inference": "Seeds and dose groups do not create independent clinical studies."}
    if out.exists():
        raise ValueError("Never overwrite a sealed score release")
    os.umask(0o077)
    out.mkdir(parents=True, mode=0o700)
    report = {"schema_version": "pbpk-independent-study-scores-v1", "study_id": design["study_id"],
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "design_sha256": sha(design_path), "prediction_lock_sha256": sha(lock_path),
              "observations_sha256": sha(obs_path), "curation_receipt_sha256": sha(receipt_path),
              "score_script_sha256": sha(Path(__file__)),
              "no_fitting_or_model_selection_performed": True,
              "result_rows": rows, "method_summaries": methods,
              "primary_dimension": "nominal-sampling aggregate Cmax accuracy on one new study",
              "auxiliary_dimensions": ["partial mid/late concentration curve accuracy", "AUC window/LLOQ sensitivity with explicit non-equivalence"],
              "global_weighted_composite": None,
              "generalization_claim": "new-study external evaluation of existing caffeine models; not new-compound validation"}
    write_json(out / "scores.json", report)
    receipt = {"status": "scoring_complete_sealed", "study_id": design["study_id"],
               "case_ids": [x["case_id"] for x in design["cases"]],
               "method_seed_count": lock["method_seed_count"], "dose_count": 2,
               "successful_evaluation_rows": len(rows), "failed_evaluation_rows": 0,
               "independent_studies": 1, "independent_compounds": 1, "new_compounds": 0,
               "observed_values_and_rankings_released": False,
               "sealed_scores": str((out / "scores.json").relative_to(ROOT)),
               "scores_sha256": sha(out / "scores.json"),
               "prediction_lock_sha256": sha(lock_path), "observations_sha256": sha(obs_path),
               "script_sha256": sha(Path(__file__))}
    write_json(out.parent / "scoring_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
