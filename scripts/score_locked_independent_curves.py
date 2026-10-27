"""Score source-disjoint curves only after verifying immutable prediction locks.

Scientific outputs are sealed; console/receipt contain execution counts only.
The existing float64 curve scorer provides the natural-log RMSE definition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def project_path(value: str) -> Path:
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    path.relative_to(ROOT)
    return path


def verified(path: str, expected_sha: str) -> Path:
    result = project_path(path)
    if hashlib.sha256(result.read_bytes()).hexdigest() != expected_sha:
        raise ValueError(f"Hash mismatch: {path}")
    return result


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_locked_inputs(lock_path: Path) -> tuple[dict, list[dict]]:
    lock = read_json(lock_path)
    if lock.get("schema_version") != "pbpk-external-predictions-lock-v1" or lock.get("model_selection_frozen") is not True:
        raise ValueError("Predictions require the frozen external lock schema")
    freeze = lock.get("config_freeze", {})
    frozen_config = read_json(verified(freeze["path"], freeze["sha256"]))
    records = lock.get("predictions")
    if not isinstance(records, list) or not records:
        raise ValueError("No locked prediction cases")
    frozen_cases = frozen_config.get("cases")
    if frozen_cases is None and isinstance(frozen_config.get("protocols"), dict):
        # Equivalent already-running predictor contract: protocols were hashed
        # before native execution, and each binds a design hash from frozen inputs.
        frozen_inputs = frozen_config.get("frozen_input_sha256", {})
        for source_path, source_sha in frozen_inputs.items():
            verified(source_path, source_sha)
        frozen_cases = []
        for frozen_protocol in frozen_config["protocols"].values():
            content = read_json(verified(frozen_protocol["path"], frozen_protocol["sha256"]))
            design_paths = [path for path, sha in frozen_inputs.items() if sha == content.get("design_sha256")]
            if len(design_paths) != 1:
                raise ValueError("Frozen protocol has no unique hashed design input")
            frozen_cases.append({"case_id": content["case_id"], "design_path": design_paths[0],
                                 "design_sha256": content["design_sha256"],
                                 "protocol_path": frozen_protocol["path"], "protocol_sha256": frozen_protocol["sha256"]})
    if not isinstance(frozen_cases, list) or not frozen_cases:
        raise ValueError("No preregistered cases in config freeze")
    frozen_by_id = {c["case_id"]: c for c in frozen_cases}
    if len(frozen_by_id) != len(frozen_cases) or set(frozen_by_id) != {r["case_id"] for r in records}:
        raise ValueError("Final prediction case set differs from pre-run freeze")
    checked, seen = [], set()
    for item in records:
        case_id = item["case_id"]
        if case_id in seen:
            raise ValueError("Duplicate case in lock")
        seen.add(case_id)
        frozen_case = frozen_by_id[case_id]
        for field in ("design_path", "design_sha256", "protocol_path", "protocol_sha256"):
            if item[field] != frozen_case[field]:
                raise ValueError("Final protocol/design differs from pre-run freeze")
        design_path = verified(item["design_path"], item["design_sha256"])
        protocol_path = verified(item["protocol_path"], item["protocol_sha256"])
        prediction_path = verified(item["path"], item["sha256"])
        design, protocol, prediction = map(read_json, (design_path, protocol_path, prediction_path))
        if any(x.get("case_id") != case_id for x in (design, protocol, prediction)):
            raise ValueError("Case identities differ")
        if protocol.get("design_sha256") != item["design_sha256"]:
            raise ValueError("Protocol is not tied to the locked design")
        if design.get("schema_version") != "pbpk-independent-curve-design-v1":
            raise ValueError("Unsupported independent curve design")
        if protocol.get("prediction_aggregation") != "sex_weighted_reference_concentration":
            raise ValueError("Unregistered population aggregation")
        molecular_weight = protocol.get("molecular_weight_g_mol")
        if not finite_number(molecular_weight) or molecular_weight <= 0:
            raise ValueError("Explicit positive molecular weight required")
        weights = {str(k).lower(): v for k, v in protocol["sex_weights"].items()}
        if set(weights) != {"male", "female"} or any(not finite_number(v) or v <= 0 for v in weights.values()) or not math.isclose(sum(weights.values()), 1, abs_tol=1e-12):
            raise ValueError("Invalid population weights")
        population = design["population"]
        for sex in weights:
            if not math.isclose(weights[sex], population[sex + "_n"] / population["n"], abs_tol=1e-12):
                raise ValueError("Weights differ from preregistered observed group demographics")
        expected_times = [t * 60 for t in design["sampling"]["time_h"]]
        models = prediction.get("models", [])
        if len(models) != 2 or {str(m.get("sex")).lower() for m in models} != set(weights):
            raise ValueError("Expected exactly one male and one female source-reference curve")
        weighted = [0.0] * len(expected_times)
        for model in models:
            if model.get("model_key") != protocol.get("model_key") or not protocol.get("model_key"):
                raise ValueError("Locked source model selection differs")
            if str(model.get("unit", "")).replace("µ", "u").replace("μ", "u") != "umol/l":
                raise ValueError("Explicit micromolar prediction unit required")
            times, values = model.get("time_min"), model.get("concentration")
            if not isinstance(times, list) or not isinstance(values, list) or len(times) != len(expected_times) or len(values) != len(times):
                raise ValueError("Prediction shape differs from original nominal schedule")
            if any(not finite_number(t) or not math.isclose(t, target, rel_tol=0, abs_tol=1e-8) for t, target in zip(times, expected_times)):
                raise ValueError("Prediction grid shifted away from original nominal times")
            if any(not finite_number(v) for v in values):
                raise ValueError("Nonfinite prediction")
            sex = str(model["sex"]).lower()
            weighted = [old + weights[sex] * v for old, v in zip(weighted, values)]
        # Deliberately do not read/hash any sealed observation until ALL designs,
        # protocols, prediction hashes, and sampling grids above have passed.
        checked.append({"case_id": case_id, "design": design, "protocol": protocol, "time_min": expected_times,
                        "concentration_umol_l": weighted, "weights": weights})
    return lock, checked


def score_subject(record: dict, times: list[float], predicted: list[float], molecular_weight: float, lloq: float) -> dict:
    from pbpk_kb.blueprints.design_scoring import score_calibration_curves

    points, observations = [], []
    if len(record["observations"]) != len(times):
        raise ValueError("Sealed observation grid length differs")
    for index, (point, time, prediction) in enumerate(zip(record["observations"], times, predicted)):
        if not math.isclose(point["nominal_time_h"] * 60, time, abs_tol=1e-8):
            raise ValueError("Sealed sampling time mismatch")
        status, value = point["observation_status"], point["concentration_ng_ml"]
        reason = None
        if status == "left_censored_below_LLOQ":
            reason = "source_BQL_left_censored_not_exact_log_observation"
        elif status == "source_missing_unspecified":
            reason = "source_missing_token_not_imputed"
        elif status != "quantified":
            raise ValueError("Unknown observation status")
        elif not finite_number(value) or value < 0:
            raise ValueError("Invalid quantified observation")
        elif value == 0:
            reason = "zero_observation_log_undefined"
        elif value < lloq:
            reason = "below_source_LLOQ_not_quantifiable"
        item = {"source_index": index, "nominal_time_min": time, "source_status": status,
                "source_token": point["source_token"], "included": reason is None,
                "exclusion_reason": reason, "predicted_concentration_umol_l": prediction}
        points.append(item)
        if reason is None:
            observations.append({"time_min": time, "concentration_umol_per_l": value / molecular_weight})
    result = {"subject_id": record["subject_id"], "sequence": record["sequence"], "period": record["period"],
              "treatment": record["treatment"], "points": points, "positive_quantifiable_observations": len(observations),
              "status": "failed", "log_rmse": None, "log_mse": None, "error": None}
    if not observations:
        result["error"] = "no_positive_quantifiable_observations"
        return result
    scored = score_calibration_curves(runs=[{"vector_index": 0, "time_unit": "min", "unit": "umol/l",
                                            "time_min": times, "concentration": predicted}],
                                      observations=observations, device="cpu")["rows"][0]
    if not scored["valid"]:
        result["error"] = scored["error"]
        return result
    result.update(status="evaluated", log_rmse=scored["log_rmse"], log_mse=scored["log_mse"])
    return result


def score_locked(lock_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Scoring output must be a new directory")
    lock, checked = validate_locked_inputs(lock_path)
    # First access to sealed outcome bytes occurs beyond this barrier.
    results = []
    for case in checked:
        design, protocol = case["design"], case["protocol"]
        ref = design["sealed_observation_reference"]
        sealed = read_json(verified(ref["path"], ref["sha256"]))
        if sealed.get("case_id") != case["case_id"] or sealed.get("publication_cluster_id") != design["publication_cluster_id"]:
            raise ValueError("Sealed case/source identity mismatch")
        if sealed.get("unit") != "ng/ml" or sealed.get("assay_LLOQ_ng_ml") != design["curve_evaluation_contract"]["assay_LLOQ_ng_ml"]:
            raise ValueError("Sealed measurement contract mismatch")
        subjects = [r for r in sealed["records"] if r["treatment"] == "R"]
        planned_links = {x["subject_id"]: x for x in design["crossover"]["all_subject_links"]}
        if len(subjects) != len(planned_links) or {r["subject_id"] for r in subjects} != set(planned_links):
            raise ValueError("Sealed reference subject set differs from locked design")
        for subject in subjects:
            link = planned_links[subject["subject_id"]]
            if subject["period"] != link["periods"]["R"] or subject["sequence"] != link["sequence"]:
                raise ValueError("Crossover pairing changed")
        scored = [score_subject(r, case["time_min"], case["concentration_umol_l"], protocol["molecular_weight_g_mol"], sealed["assay_LLOQ_ng_ml"]) for r in subjects]
        complete = all(s["status"] == "evaluated" for s in scored)
        results.append({"case_id": case["case_id"], "publication_cluster_id": design["publication_cluster_id"],
                        "cohort_id": design["cohort_id"], "compound": design["compound"], "model_key": protocol["model_key"],
                        "status": "evaluated" if complete else "failed_retained_in_denominator",
                        "subjects": scored, "dose_log_rmse": sum(s["log_rmse"] for s in scored) / len(scored) if complete else None,
                        "aggregation": "arithmetic mean of all reference subject natural-log RMSEs; no success-only substitution",
                        "prediction_is_group_sex_weighted_reference_not_personalized": True})
    groups = {}
    for case in results:
        groups.setdefault((case["compound"], case["publication_cluster_id"]), []).append(case)
    studies = []
    for (compound, study), cases in sorted(groups.items()):
        complete = all(c["dose_log_rmse"] is not None for c in cases)
        studies.append({"compound": compound, "publication_cluster_id": study, "dose_cohorts": len(cases),
                        "study_log_rmse": sum(c["dose_log_rmse"] for c in cases) / len(cases) if complete else None,
                        "status": "evaluated" if complete else "failed_retained_in_denominator",
                        "aggregation": "equal-weight arithmetic mean across preregistered dose-cohort log RMSEs",
                        "drug_bootstrap_confidence_interval": None, "CI_reason": "one independent drug/study; 48 subjects are not 48 independent drugs"})
    output.mkdir(parents=True)
    sealed_dir = output / "sealed"
    sealed_dir.mkdir(mode=0o700)
    score_path = sealed_dir / "curve_scores.json"
    score_path.write_text(json.dumps({"schema_version": "pbpk-independent-curve-scores-v1", "cases": results, "studies": studies,
                                      "prediction_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
                                      "scorer_library": "pbpk_kb.blueprints.design_scoring.score_calibration_curves",
                                      "prediction_clipping": False, "observation_imputation": False,
                                      "gpu_used": False, "scores_sealed_from_method_development": True}, indent=2) + "\n")
    score_path.chmod(0o600)
    receipt = {"status": "scored_and_sealed", "cases_expected": len(checked), "cases_evaluated": sum(c["status"] == "evaluated" for c in results),
               "cases_failed_retained": sum(c["status"] != "evaluated" for c in results), "reference_subjects_expected": sum(len(c["subjects"]) for c in results),
               "independent_drugs": len(set(c["compound"] for c in results)), "independent_studies": len(studies),
               "sealed_scores_path": str(score_path.relative_to(ROOT)), "sealed_scores_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
               "lock_path": str(lock_path.relative_to(ROOT)), "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
               "metrics_exposed_to_method_development": False, "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    print(json.dumps(score_locked(project_path(args.lock), project_path(args.output)), indent=2))


if __name__ == "__main__":
    main()
