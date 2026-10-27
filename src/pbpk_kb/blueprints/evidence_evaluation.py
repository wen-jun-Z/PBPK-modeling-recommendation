"""Study-separated descriptive evaluation; no fit or model selection."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .endpoint_prediction import predict_requested_endpoints


def observed_rows(document):
    rows = document.get("observations")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("Observation document must contain at least two observations")
    times = [r["time_min"] for r in rows]
    values = [r["concentration_umol_per_l"] for r in rows]
    if (any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in times + values)
            or any(t < 0 for t in times) or any(v <= 0 for v in values)
            or any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError("Observations require increasing nonnegative times and positive concentrations")
    return times, values


def score_case(curve, observations, request):
    times, observed = observed_rows(observations)
    pt, pv = curve.get("time_min", []), curve.get("concentration", [])
    if (curve.get("time_unit") != "min" or curve.get("unit") != "umol/l"
            or len(pt) < 2 or len(pt) != len(pv)
            or any(not math.isfinite(x) for x in pt + pv)
            or any(x < 0 for x in pt + pv) or any(b <= a for a, b in zip(pt, pt[1:]))
            or times[0] < pt[0] or times[-1] > pt[-1]):
        raise ValueError("Prediction must cover observations without extrapolation")
    predicted = np.interp(times, pt, pv).tolist()
    if any(v <= 0 for v in predicted):
        raise ValueError("Nonpositive prediction at an observation time")
    errors = [math.log(p / o) for p, o in zip(predicted, observed)]
    fold = lambda a, b: max(a / b, b / a)
    auc = lambda values: math.fsum((b-a)/60 * (x+y)/2 for a,b,x,y in zip(times,times[1:],values,values[1:]))
    mw = next(e["value"] for e in request["evidence"] if e["parameter"] == "molecular_weight")
    context = {k: request["dosing"].get(k, 0.) for k in ("route", "dose_mg", "start_h", "infusion_minutes")}
    context["single_dose"] = True
    target = {"endpoints": ["tmax", "c_last", "terminal_half_life"], "primary_endpoint": "terminal_half_life",
              "window_start_h": times[0]/60, "window_end_h": times[-1]/60}
    if request["target"].get("terminal_fit"):
        target["terminal_fit"] = request["target"]["terminal_fit"]
    endpoints = []
    for values in (observed, predicted):
        endpoints.append(predict_requested_endpoints(
            {"time_min": times, "concentration": values, "time_unit": "min", "unit": "umol/l"}, target, mw, context=context))
    observed_endpoints, predicted_endpoints = [e["endpoints"] for e in endpoints]
    half_o, half_p = observed_endpoints["terminal_half_life"], predicted_endpoints["terminal_half_life"]
    half = {"status": "not_estimable", "observed": half_o, "predicted": half_p,
            "definition": "same predeclared terminal window and observed sample times; derived from historical curves"}
    if all(x.get("status", "completed") == "completed" for x in (half_o, half_p)):
        half.update(status="completed", fold_error=fold(half_o["value_h"], half_p["value_h"]))
    to, tp = times[observed.index(max(observed))]/60, times[predicted.index(max(predicted))]/60
    return {"point_count": len(times), "gmfe": math.exp(math.fsum(abs(e) for e in errors)/len(errors)),
            "log_rmse": math.sqrt(math.fsum(e*e for e in errors)/len(errors)),
            "observed_window_h": [times[0]/60, times[-1]/60],
            "auc": {"observed_umol_h_l": auc(observed), "predicted_umol_h_l": auc(predicted),
                    "fold_error": fold(auc(observed), auc(predicted)), "definition": "linear trapezoids at identical observation times"},
            "sampled_cmax": {"observed_umol_l": max(observed), "predicted_umol_l": max(predicted),
                             "fold_error": fold(max(observed), max(predicted)), "definition": "maximum at sampled times; not continuous early Cmax"},
            "sampled_tmax": {"observed_h": to, "predicted_h": tp, "absolute_error_h": abs(tp-to),
                             "definition": "earliest maximum at observed times only; absolute error, not ratio"},
            "c_last": {"observed_umol_l": observed[-1], "predicted_umol_l": predicted[-1],
                       "fold_error": fold(observed[-1], predicted[-1]), "definition": "concentration at last observed time; not steady-state trough"},
            "terminal_half_life": half,
            "time_min": times, "observed_umol_l": observed, "predicted_umol_l": predicted,
            "auc_inf_clearance_volume_mrt_validation": "not_claimed_without_qualified_independent_endpoint_references"}


def aggregate_cases(results):
    """Keep every attempt in coverage; report calibration/comparison separately.

    Repeated dose arms are averaged within study, then studies within drug.
    Drugs receive equal weight in the descriptive macro summary. No confidence
    intervals or inferential claim are manufactured from this small cohort.
    """
    reports = {}
    metric_paths = {"gmfe": ("gmfe",), "auc_fold": ("auc", "fold_error"),
                    "sampled_cmax_fold": ("sampled_cmax", "fold_error"),
                    "c_last_fold": ("c_last", "fold_error"),
                    "terminal_half_life_fold": ("terminal_half_life", "fold_error")}
    for role in sorted({r["role"] for r in results}):
        rows = [r for r in results if r["role"] == role]
        report = {"attempted_cases": len(rows), "scored_cases": sum(r.get("scoring_status") == "completed" for r in rows),
                  "compound_count": len({r["compound_id"] for r in rows}), "metrics": {}}
        for metric, path in metric_paths.items():
            groups = defaultdict(list)
            for row in rows:
                value = row.get("metrics", {})
                for part in path:
                    value = value.get(part) if isinstance(value, dict) else None
                if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
                    groups[(row["compound_id"], row["study_id"])].append(math.log(value))
            drug_studies = defaultdict(list)
            for (drug, _), values in groups.items():
                drug_studies[drug].append(math.fsum(values)/len(values))
            by_drug = {drug: math.exp(math.fsum(values)/len(values)) for drug, values in drug_studies.items()}
            report["metrics"][metric] = {"by_compound": by_drug, "eligible_compounds": len(by_drug),
                "eligible_studies": len(groups), "eligible_cases": sum(len(v) for v in groups.values()),
                "macro_geometric_mean": math.exp(math.fsum(math.log(v) for v in by_drug.values())/len(by_drug)) if by_drug else None}
        reports[role] = report
    return reports
