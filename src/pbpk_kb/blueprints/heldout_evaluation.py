"""Evaluate a preselected prediction without searching on external outcomes.

This module reads no files and runs no solver. The caller must persist the
prediction lock before revealing observations; matching a digest establishes
content integrity, not the chronology or scientific independence of a study.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any

from .design_scoring import score_calibration_curves


TIERS = frozenset({"main_holdout", "bottom_up_external", "reconstruction", "development"})
_UNITS = {"umol/l", "µmol/l", "μmol/l"}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def lock_predictions(predicted_curves: dict, fallback_curves: dict | None = None) -> str:
    """Hash both primary and optional common-fallback curves before evaluation."""
    if not isinstance(predicted_curves, dict) or (fallback_curves is not None and not isinstance(fallback_curves, dict)):
        raise ValueError("Predictions and fallbacks must map scenario IDs to one curve or null")
    return hashlib.sha256(_canonical({"predicted_curves": predicted_curves,
                                     "fallback_curves": fallback_curves or {}}).encode()).hexdigest()


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty identifier")
    return value


def _role(scenario: dict) -> str | None:
    first, second = scenario.get("study_role"), scenario.get("role")
    if first is not None and second is not None and first != second:
        raise ValueError("Conflicting study_role and role")
    return first if first is not None else second


def _main_provenance(scenario: dict) -> None:
    if _role(scenario) != "test":
        raise ValueError("main_holdout requires a final test study, never calibration or selection")
    ref = scenario.get("study_reference") or scenario.get("source_audit", {}).get("study_reference")
    if (not isinstance(ref, dict) or ref.get("role") != "test"
            or ref.get("study_id") != scenario["study_group_id"]):
        raise ValueError("main_holdout requires a matching test study_reference")
    _identifier(ref.get("source_id"), "study_reference.source_id")
    if not isinstance(ref.get("source_checksum"), str) or not re.fullmatch(r"[a-fA-F0-9]{64}", ref["source_checksum"]):
        raise ValueError("main_holdout requires the original source checksum")
    groups, cohorts = scenario.get("development_study_group_ids"), scenario.get("development_cohort_ids")
    if not isinstance(groups, list) or not isinstance(cohorts, list):
        raise ValueError("main_holdout requires explicit development study and cohort identifiers")
    if any(not isinstance(x, str) or not x.strip() for x in groups + cohorts):
        raise ValueError("Invalid development study/cohort identifier")
    if len(set(groups)) < 2 or not cohorts:
        raise ValueError("main_holdout requires separate calibration/selection studies with cohort records")
    if scenario["study_group_id"] in groups:
        raise ValueError("Final study overlaps a development study")
    final_cohorts = ref.get("cohort_ids")
    if not isinstance(final_cohorts, list) or not final_cohorts or any(not isinstance(x, str) or not x.strip() for x in final_cohorts):
        raise ValueError("main_holdout requires explicit final-study cohort identities")
    if set(final_cohorts) & set(cohorts):
        raise ValueError("Final cohort overlaps a development cohort")
    observation_records = scenario.get("observations")
    for point in observation_records if isinstance(observation_records, list) else []:
        if not isinstance(point, dict):
            continue
        provenance = point.get("provenance", [])
        references = [point] + (provenance if isinstance(provenance, list) else [provenance])
        for reference in references:
            if isinstance(reference, dict) and any(reference.get(key) not in {None, "test"}
                                                  for key in ("study_role", "role", "split")):
                raise ValueError("Observation provenance contains a non-test study role")


def _observations(scenario: dict) -> tuple[list[dict] | None, str | None]:
    values = scenario.get("observations")
    if not isinstance(values, list) or not values:
        return None, "observations_missing"
    previous = -1.0
    for point in values:
        if not isinstance(point, dict):
            return None, "observation_not_an_object"
        time, value = point.get("time_min"), point.get("concentration_umol_per_l")
        if (isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time)
                or time < 0 or time <= previous):
            return None, "observation_times_must_be_finite_and_strictly_increasing"
        previous = time
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            return None, "unquantifiable_observation_requires_separate_censoring_protocol"
        if point.get("unit", "umol/l") not in _UNITS:
            return None, "unsupported_observation_concentration_unit"
    if scenario["tier"] == "main_holdout" and len(values) < 5:
        return None, "main_curve_requires_five_quantifiable_points"
    return values, None


def _curve_error(curve: Any, required_times: list[float]) -> str | None:
    """Engineering checks depend on required times, never observed concentrations."""
    if curve is None:
        return "primary_prediction_missing"
    if not isinstance(curve, dict):
        return "prediction_must_be_one_native_curve"
    if curve.get("time_unit") != "min" or curve.get("unit") not in _UNITS:
        return "unsupported_native_output_unit"
    times, values = curve.get("time_min"), curve.get("concentration")
    if not isinstance(times, list) or not isinstance(values, list) or len(times) < 2 or len(times) != len(values):
        return "invalid_native_curve_shape"
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in times + values):
        return "nonfinite_native_curve"
    if times[0] < 0 or any(b <= a for a, b in zip(times, times[1:])):
        return "invalid_native_time_grid"
    for time in required_times:
        if time < times[0] or time > times[-1]:
            return "prediction_would_require_extrapolation"
        upper = min(len(times) - 1, max(1, bisect_right(times, time)))
        lower = upper - 1
        fraction = (time - times[lower]) / (times[upper] - times[lower])
        prediction = (1 - fraction) * values[lower] + fraction * values[upper]
        if not math.isfinite(prediction) or prediction <= 0:
            return "nonpositive_or_nonfinite_prediction_at_required_time"
    return None


def _endpoint(observed: float, predicted: float, **metadata) -> dict:
    difference = math.log(predicted) - math.log(observed)
    return {"observed": observed, "predicted": predicted, "log_ratio": difference,
            "absolute_log_error": abs(difference), **metadata}


def _endpoints(scenario: dict, observations: list[dict], predictions: list[float]) -> dict:
    result = {"observed_window_auc": None, "sampled_cmax": None}
    times = [row["time_min"] for row in observations]
    values = [row["concentration_umol_per_l"] for row in observations]
    if scenario.get("auc_evaluable") is True and len(times) >= 2:
        def integrate(data):
            return sum((right - left) * (a + b) / 2
                       for left, right, a, b in zip(times, times[1:], data, data[1:]))
        observed, predicted = integrate(values), integrate(predictions)
        if all(math.isfinite(x) and x > 0 for x in (observed, predicted)):
            result["observed_window_auc"] = _endpoint(observed, predicted, unit="umol*min/l",
                    start_min=times[0], end_min=times[-1], rule="linear_trapezoid_at_observed_times",
                    extrapolated=False)
    if scenario.get("cmax_evaluable") is True and len(times) >= 2:
        result["sampled_cmax"] = _endpoint(max(values), max(predictions), unit="umol/l",
                sampling_times_min=times, continuous_peak_estimated=False)
    return result


def _mean(values):
    return sum(values) / len(values) if values else None


def _hierarchical(rows: list[dict], metric: str, *, logarithmic: bool = False) -> tuple[float | None, int]:
    """Equal curves within study, equal studies within drug, then equal drugs."""
    by_drug = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = row.get(metric)
        if value is not None:
            by_drug[row["compound_id"]][row["study_group_id"]].append(math.log(value) if logarithmic else value)
    value = _mean([_mean([_mean(curves) for curves in studies.values()]) for studies in by_drug.values()])
    return (math.exp(value) if logarithmic and value is not None else value), len(by_drug)


def _summarize(rows: list[dict]) -> dict:
    unique = [row for row in rows if row["duplicate_of"] is None]
    valid = [row for row in unique if row["log_rmse"] is not None]
    complete = len(valid) == len(unique) and bool(unique)
    error, _ = _hierarchical(valid, "log_rmse")
    gmfe, _ = _hierarchical(valid, "gmfe", logarithmic=True)
    native = [row for row in valid if not row["fallback_used"]]
    compounds = sorted({row["compound_id"] for row in unique})
    by_compound = []
    for compound in compounds:
        group = [row for row in unique if row["compound_id"] == compound]
        available = [row for row in group if row["log_rmse"] is not None]
        value, _ = _hierarchical(available, "log_rmse")
        by_compound.append({"compound_id": compound, "study_group_count": len({r["study_group_id"] for r in group}),
                "curve_count": len(group), "scored_curve_count": len(available),
                "complete": len(available) == len(group),
                "mean_curve_log_rmse": value if len(available) == len(group) else None,
                "available_curve_log_rmse": value})
    endpoints = {}
    for name in ("observed_window_auc", "sampled_cmax"):
        qualified = [dict(row, endpoint_absolute_log_error=row["endpoints"][name]["absolute_log_error"])
                     for row in valid if row["endpoints"][name] is not None]
        value, drug_count = _hierarchical(qualified, "endpoint_absolute_log_error")
        endpoints[name] = {"requested_curve_count": sum(row["endpoint_eligibility"][name] for row in unique),
                           "evaluable_curve_count": len(qualified), "evaluable_compound_count": drug_count,
                           "mean_absolute_log_error": value}
    return {"scenario_count": len(rows), "duplicate_scenario_count": len(rows) - len(unique),
            "unique_curve_count": len(unique), "compound_count": len(compounds),
            "unique_global_study_groups": len({r["study_group_id"] for r in unique}),
            "drug_study_memberships": len({(r["compound_id"], r["study_group_id"]) for r in unique}),
            "scored_curve_count": len(valid), "unscored_curve_count": len(unique) - len(valid),
            "native_success_count": len(native), "fallback_count": sum(r["fallback_used"] for r in unique),
            "analysis_complete": complete, "mean_curve_log_rmse": error if complete else None,
            "gmfe": gmfe if complete else None, "available_curve_log_rmse": error,
            "available_curve_gmfe": gmfe, "native_success_only_log_rmse": _hierarchical(native, "log_rmse")[0],
            "aggregation": "curve_RMSE_then_equal_studies_then_equal_drugs; GMFE_geometric_at_same_levels",
            "by_compound": by_compound, "endpoints": endpoints}


def evaluate_locked_predictions(
    scenarios: list[dict], predicted_curves: dict[str, dict | None], locked_selection: dict,
    *, fallback_curves: dict[str, dict | None] | None = None, device: str = "cpu",
) -> dict:
    """Evaluate one locked method/seed selection across explicit study scenarios.

    A scenario supplies scenario_id, compound_id, study_group_id, tier and
    observations. curve_id defaults to scenario_id. Only explicitly enabled
    auc_evaluable/cmax_evaluable endpoints are reported. Exact duplicate curves
    are counted once; conflicting values/predictions for one curve are rejected.
    Missing/invalid predictions keep their denominator, using only an optional
    prelocked engineering fallback. A large final error never triggers fallback.
    """
    if not isinstance(locked_selection, dict):
        raise ValueError("Provide one explicitly locked selection")
    if set(locked_selection) & {"candidates", "candidate_scores", "alternatives"}:
        raise ValueError("Final evaluation cannot select among candidate models")
    method = _identifier(locked_selection.get("method"), "method")
    seed = locked_selection.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("A locked integer seed is required")
    expected = locked_selection.get("predictions_sha256")
    actual = lock_predictions(predicted_curves, fallback_curves)
    if not isinstance(expected, str) or actual != expected:
        raise ValueError("Predictions or fallback changed after the prediction lock")
    status = locked_selection.get("status", "completed")
    if status not in {"completed", "success", "failed", "blocked", "budget_exhausted"}:
        raise ValueError("Unknown locked submission status")
    if status in {"completed", "success", "budget_exhausted"} and any(predicted_curves.values()):
        _identifier(locked_selection.get("candidate_id"), "candidate_id")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Provide explicit evaluation scenarios")
    if any(not isinstance(scenario, dict) for scenario in scenarios):
        raise ValueError("Every scenario must be a dictionary")
    ids = [_identifier(s.get("scenario_id"), "scenario_id") for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate scenario_id would duplicate a submitted result")
    if (set(predicted_curves) | set(fallback_curves or {})) - set(ids):
        raise ValueError("Predictions contain unregistered scenarios")
    final_groups = {scenario.get("study_group_id") for scenario in scenarios if scenario.get("tier") == "main_holdout"}
    development_groups = {scenario.get("study_group_id") for scenario in scenarios
                          if _role(scenario) in {"calibration", "selection"}
                          or scenario.get("tier") in {"development", "reconstruction"}}
    if final_groups & development_groups:
        raise ValueError("A main-holdout study also appears among development/reconstruction scenarios")
    rows, seen = [], {}
    for source in scenarios:
        scenario = deepcopy(source)
        for name in ("compound_id", "study_group_id"):
            _identifier(scenario.get(name), name)
        tier = scenario.get("tier")
        if tier not in TIERS:
            raise ValueError("Use an explicit registered evaluation tier")
        role = _role(scenario)
        if tier == "main_holdout":
            _main_provenance(scenario)
        elif tier == "bottom_up_external" and role not in {None, "external", "test"}:
            raise ValueError("A development study cannot be labeled bottom_up_external")
        scenario_id = scenario["scenario_id"]
        curve_id = _identifier(scenario.get("curve_id", scenario_id), "curve_id")
        if any(type(scenario.get(flag, False)) is not bool for flag in ("auc_evaluable", "cmax_evaluable")):
            raise ValueError("Endpoint eligibility flags must be explicit booleans")
        row = {"scenario_id": scenario_id, "compound_id": scenario["compound_id"],
               "study_group_id": scenario["study_group_id"], "curve_id": curve_id,
               "tier": tier, "study_role": role, "duplicate_of": None,
               "status": "unscored", "native_prediction_error": None, "fallback_used": False,
               "endpoint_eligibility": {"observed_window_auc": scenario.get("auc_evaluable", False),
                                        "sampled_cmax": scenario.get("cmax_evaluable", False)},
               "log_rmse": None, "gmfe": None, "endpoints": {"observed_window_auc": None, "sampled_cmax": None}}
        # Preserve the qualification audit for later cohort aggregation. A tier
        # label alone must never promote a historical reconstruction to a test.
        row["observation_sha256"] = hashlib.sha256(_canonical(scenario.get("observations")).encode()).hexdigest()
        row["main_holdout_provenance_verified"] = tier == "main_holdout"
        if tier == "main_holdout":
            row["study_reference"] = deepcopy(scenario.get("study_reference") or scenario["source_audit"]["study_reference"])
            row["development_study_group_ids"] = list(scenario["development_study_group_ids"])
            row["development_cohort_ids"] = list(scenario["development_cohort_ids"])
        curve, fallback = predicted_curves.get(scenario_id), (fallback_curves or {}).get(scenario_id)
        key = (tier, scenario["compound_id"], scenario["study_group_id"], curve_id)
        signature = _canonical({"observations": scenario.get("observations"), "curve": curve,
                                "fallback": fallback, "auc": scenario.get("auc_evaluable", False),
                                "cmax": scenario.get("cmax_evaluable", False)})
        if key in seen:
            previous_signature, previous = seen[key]
            if signature != previous_signature:
                raise ValueError("Conflicting observations or primary predictions for a repeated curve")
            duplicate = deepcopy(previous)
            duplicate.update(scenario_id=scenario_id, duplicate_of=previous["scenario_id"])
            rows.append(duplicate)
            continue
        observations, error = _observations(scenario)
        if error:
            row.update(status="observations_unscorable", error=error)
        else:
            times = [point["time_min"] for point in observations]
            error = "locked_submission_failed" if status in {"failed", "blocked"} or locked_selection.get("engineering_valid") is False else _curve_error(curve, times)
            row["native_prediction_error"] = error
            if error:
                fallback_error = _curve_error(fallback, times)
                if fallback_error is None:
                    curve = fallback
                    row["fallback_used"] = True
                else:
                    row.update(status="prediction_unscorable", error=error, fallback_error=fallback_error)
                    curve = None
            if curve is not None:
                normalized = dict(curve, vector_index=0)
                score = score_calibration_curves(runs=[normalized], observations=observations, device=device)["rows"][0]
                if score["valid"]:
                    row.update(status="scored_with_fallback" if row["fallback_used"] else "scored",
                               log_rmse=score["log_rmse"], gmfe=score["gmfe"],
                               endpoints=_endpoints(scenario, observations, score["predictions_umol_per_l"]))
                else:
                    # Metric overflow is not permission to change the model after seeing outcomes.
                    row.update(status="scoring_failed", error=score["error"])
        row["scoring_device_used"] = device if row["status"] in {"scored", "scored_with_fallback", "scoring_failed"} else None
        rows.append(row)
        seen[key] = (signature, row)
    tiers = {tier: _summarize([row for row in rows if row["tier"] == tier])
             for tier in sorted({row["tier"] for row in rows})}
    return {"method": method, "seed": seed, "candidate_id": locked_selection.get("candidate_id"),
            "predictions_sha256": actual, "prediction_integrity_verified": True,
            "pretest_lock_timing_verified": False, "candidate_selection_performed": False,
            "scientific_acceptance_inferred": False, "device": device, "rows": rows, "tiers": tiers}
