"""Paired, post-lock evaluation of the four purpose-refinement ablations.

Consumes the existing hundred-evaluation ``records`` schema. Prediction/source
locks are the caller's responsibility; this layer checks complete terminal case
coverage before reading any frozen reference and recomputes all error metrics.
It never changes the original scorer, predictions, references or tolerances.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from itertools import combinations
import math
from pathlib import Path

import numpy as np

from .hundred_evaluation import (
    ENDPOINT_UNITS, POLICY_STATUSES, SCOPE, _checked_evaluation, _reference_values,
)

VERSION = "purpose-refinement-evaluation-1"
METHODS = ("baseline", "parameter_only", "capacity_only", "combined")
ENDPOINTS = ("clearance", "vss", "terminal_half_life", "mrt")
RISK_TOLERANCE = 1e-8
CODE_FILES = ["src/pbpk_kb/blueprint_policy/purpose_refinement_evaluation.py",
              "src/pbpk_kb/blueprint_policy/hundred_evaluation.py"]


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _ci(values):
    return [float(x) for x in np.quantile(values, [.025, .975])]


def _exp(value):
    return math.exp(value) if value < 709.78 else None


def _metrics(rows, repeats, seed):
    if not rows:
        return {"n": 0, "gmfe": None, "gmfe_ci95": None,
                "mean_absolute_ln_error": None, "mean_absolute_ln_error_ci95": None, "within_twofold_n": 0,
                "within_twofold_rate": None, "within_twofold_ci95": None}
    error = np.array([r["absolute_ln_error"] for r in rows])
    good = np.array([r["within_twofold"] for r in rows], dtype=float)
    draws = np.random.default_rng(seed).integers(len(rows), size=(repeats, len(rows)))
    return {"n": len(rows), "gmfe": _exp(float(error.mean())),
            "gmfe_ci95": [_exp(x) for x in _ci(error[draws].mean(axis=1))],
            "mean_absolute_ln_error": float(error.mean()),
            "mean_absolute_ln_error_ci95": _ci(error[draws].mean(axis=1)), "within_twofold_n": int(good.sum()),
            "within_twofold_rate": float(good.mean()),
            "within_twofold_ci95": _ci(good[draws].mean(axis=1))}


def tolerant_risk_order(case_scores, tolerance=RISK_TOLERANCE):
    """Rank scores, grouping near ties against each group's lowest score.

    ``case_scores`` maps distinct case IDs to nonnegative finite scores or None.
    The tolerance is relative, with an absolute floor of ``tolerance``. Groups
    are anchored, not chained; this avoids a nontransitive pairwise comparator.
    SHA256(case_id) orders each group and the final unknown-score group.
    """
    if not _finite(tolerance) or tolerance <= 0:
        raise ValueError("Risk tolerance must be positive and finite")
    if any(not isinstance(cid, str) or not cid for cid in case_scores):
        raise ValueError("Risk scores need nonempty case identifiers")
    if any(v is not None and (not _finite(v) or v < 0) for v in case_scores.values()):
        raise ValueError("Risk scores must be nonnegative finite numbers or null")
    tie = lambda cid: sha256(cid.encode()).hexdigest()
    known = sorted((v, cid) for cid, v in case_scores.items() if v is not None)
    groups, group, anchor = [], [], None
    for score, cid in known:
        if anchor is None or abs(score - anchor) > tolerance * max(1., abs(score), abs(anchor)):
            if group:
                groups.append(sorted(group, key=tie))
            group, anchor = [], score
        group.append(cid)
    if group:
        groups.append(sorted(group, key=tie))
    groups.append(sorted((cid for cid, v in case_scores.items() if v is None), key=tie))
    return [cid for group in groups for cid in group]


def _barrier(protocol, results):
    if not isinstance(results, dict) or set(results) != set(METHODS):
        raise ValueError("Exactly baseline, parameter_only, capacity_only and combined are required")
    cases = protocol.get("cases")
    expected = protocol.get("expected_compounds", 100)
    if (isinstance(expected, bool) or not isinstance(expected, int) or expected < 1
            or not isinstance(cases, list) or len(cases) != expected):
        raise ValueError("Complete registered cases are required")
    indexed = {}
    for case in cases:
        cid = case.get("case_id")
        if not isinstance(cid, str) or not cid or cid in indexed or case.get("evaluation_scope") != SCOPE:
            raise ValueError("Invalid or duplicate registered case")
        indexed[cid] = case
    for key in ("compound_id", "identity_smiles"):
        values = [c.get(key) for c in cases]
        if any(not isinstance(v, str) or not v for v in values) or len(set(values)) != expected:
            raise ValueError("Cases need distinct compound and chemical identities")
    parsed = {}
    for method in METHODS:
        result = results[method]
        if (not isinstance(result, dict) or result.get("all_predictions_locked_before_scoring") is not True
                or not isinstance(result.get("records"), list)):
            raise ValueError("Each method requires post-lock canonical records")
        rows, statuses = {}, {}
        for row in result["records"]:
            cid, endpoint = row.get("case_id"), row.get("endpoint")
            if cid not in indexed or endpoint not in ENDPOINT_UNITS or (cid, endpoint) in rows:
                raise ValueError("Unknown or duplicate case/endpoint record")
            case = indexed[cid]
            if any(row.get(k) != case.get(k) for k in ("compound_id", "identity_smiles", "evaluation_scope")):
                raise ValueError("Record identity differs from the frozen cohort")
            status = row.get("construction_status")
            if status not in {"predicted", "failed", "blocked"} or statuses.get(cid, status) != status:
                raise ValueError("Every method needs a consistent terminal status for every case")
            statuses[cid] = status
            rows[cid, endpoint] = row
        if set(statuses) != set(indexed):
            raise ValueError("A method omitted a registered case; failures must remain explicit")
        parsed[method] = (rows, statuses)
    return indexed, parsed


def _normalize(case, endpoint, row, status, reference):
    value, reason = None, "missing_endpoint_record" if status == "predicted" else "construction_" + status
    policy, sensitivity, input_score, unbounded = "unassessed", None, None, False
    if row is not None:
        if row.get("reference_value") != reference:
            raise ValueError("Record reference differs from the original frozen evaluation")
        if row.get("reference_artifact", case["evaluation"]) != case["evaluation"]:
            raise ValueError("Record reference source differs from the original evaluation")
        if str(row.get("unit", "")).strip().lower().replace(" ", "") != ENDPOINT_UNITS[endpoint]:
            raise ValueError("Noncanonical endpoint unit")
        raw = row.get("prediction_value")
        if raw is not None and not _finite(raw):
            raise ValueError("Nonfinite prediction values must be represented as null failures")
        if status != "predicted":
            if raw is not None:
                raise ValueError("A failed or blocked case cannot supply an endpoint prediction")
            reason = "construction_" + status
        elif raw is not None and raw > 0:
            value, reason = float(raw), None
        else:
            reason = row.get("prediction_missing_reason") or "endpoint_not_estimable"
        policy = row.get("policy_status", "unassessed")
        if policy not in POLICY_STATUSES:
            raise ValueError("Unknown policy status")
        sensitivity, input_score = row.get("sensitivity_score"), row.get("input_only_score")
        unbounded = row.get("sensitivity_unbounded", False)
        if not isinstance(unbounded, bool):
            raise ValueError("Sensitivity unbounded flag must be boolean")
        for score in (sensitivity, input_score):
            if score is not None and (not _finite(score) or score < 0):
                raise ValueError("Invalid risk score")
    error = abs(math.log(value) - math.log(reference)) if value is not None and reference is not None else None
    return {"case_id": case["case_id"], "drug_name": case.get("drug_name"), "endpoint": endpoint,
            "construction_status": status, "prediction_value": value, "reference_value": reference,
            "prediction_missing_reason": reason, "scorable": error is not None,
            "absolute_ln_error": error, "within_twofold": error <= math.log(2.) + 1e-12 if error is not None else None,
            "policy_status": policy, "sensitivity_score": sensitivity, "input_only_score": input_score,
            "sensitivity_unbounded": unbounded}


def _paired(left, right, total, repeats, seed):
    ids = sorted(set(left) & set(right))
    out = {"common_scorable_n": len(ids), "registered_n": total,
           "common_scorable_case_ids": ids, "left_only_scorable_n": len(set(left) - set(right)),
           "right_only_scorable_n": len(set(right) - set(left)),
           "neither_scorable_n": total - len(set(left) | set(right)),
           "left_common_scorable": _metrics([left[i] for i in ids], repeats, seed),
           "right_common_scorable": _metrics([right[i] for i in ids], repeats, seed),
           "gmfe_ratio_right_over_left": None, "gmfe_ratio_ci95": None,
           "mean_absolute_ln_error_difference_right_minus_left": None,
           "mean_absolute_ln_error_difference_ci95": None,
           "within_twofold_rate_difference_right_minus_left": None,
           "within_twofold_rate_difference_ci95": None}
    if not ids:
        return out
    delta = np.array([right[i]["absolute_ln_error"] - left[i]["absolute_ln_error"] for i in ids])
    good = np.array([int(right[i]["within_twofold"]) - int(left[i]["within_twofold"]) for i in ids])
    draws = np.random.default_rng(seed).integers(len(ids), size=(repeats, len(ids)))
    error_ci = _ci(delta[draws].mean(axis=1))
    out.update(gmfe_ratio_right_over_left=_exp(float(delta.mean())),
               gmfe_ratio_ci95=[_exp(x) for x in error_ci],
               mean_absolute_ln_error_difference_right_minus_left=float(delta.mean()),
               mean_absolute_ln_error_difference_ci95=error_ci,
               within_twofold_rate_difference_right_minus_left=float(good.mean()),
               within_twofold_rate_difference_ci95=_ci(good[draws].mean(axis=1)))
    return out


def _risk(rows, repeats, seed):
    scored = {r["case_id"]: r for r in rows if r["scorable"]}
    scores = {"endpoint_sensitivity": {}, "input_only": {}}
    for cid, row in scored.items():
        scores["endpoint_sensitivity"][cid] = (row["sensitivity_score"] if not row["sensitivity_unbounded"]
            and row["policy_status"] in {"sensitive", "within_declared_limits"} else None)
        scores["input_only"][cid] = row["input_only_score"]
    usable = {key: sum(v is not None for v in values.values()) for key, values in scores.items()}
    result = {"common_scorable_pool_n": len(scored), "registered_n": len(rows),
              "usable_scores_n": usable, "relative_tie_tolerance": RISK_TOLERANCE,
              "absolute_tie_floor": RISK_TOLERANCE, "tie_breaker": "SHA256(case_id)",
              "unknown_scores_rank_last": True, "points": [],
              "status": "assessed" if all(usable.values()) else "not_assessed_no_comparable_risk_scores"}
    if not all(usable.values()):
        return result
    ids = sorted(scored)
    pos = {cid: i for i, cid in enumerate(ids)}
    err = np.array([scored[cid]["absolute_ln_error"] for cid in ids])
    bad = np.array([not scored[cid]["within_twofold"] for cid in ids], dtype=float)
    samples = np.random.default_rng(seed).integers(len(ids), size=(repeats, len(ids)))
    orders, resampled = {}, {}
    for method, values in scores.items():
        order = np.array([pos[cid] for cid in tolerant_risk_order(values)])
        ranks = np.empty(len(ids), dtype=int)
        ranks[order] = np.arange(len(ids))
        orders[method] = order
        resampled[method] = np.take_along_axis(samples, np.argsort(ranks[samples], axis=1, kind="stable"), axis=1)
    for coverage in (.2, .4, .6, .8, 1.):
        n = math.ceil(coverage * len(ids))
        point = {"requested_coverage": coverage, "selected_n": n,
                 "coverage_registered": n / len(rows), "coverage_scorable": n / len(ids), "rankings": {}}
        draws = {}
        for method in scores:
            selected = orders[method][:n]
            draws[method] = err[resampled[method][:, :n]].mean(axis=1)
            point["rankings"][method] = {
                "selected_case_ids": [ids[i] for i in selected], "mean_absolute_ln_error": float(err[selected].mean()),
                "mean_absolute_ln_error_ci95": _ci(draws[method]),
                "over_twofold_n": int(bad[selected].sum()), "over_twofold_rate": float(bad[selected].mean())}
        point["difference_sensitivity_minus_input_only"] = {
            "mean_absolute_ln_error": point["rankings"]["endpoint_sensitivity"]["mean_absolute_ln_error"]
            - point["rankings"]["input_only"]["mean_absolute_ln_error"],
            "mean_absolute_ln_error_ci95": _ci(draws["endpoint_sensitivity"] - draws["input_only"])}
        result["points"].append(point)
    return result


def score_refinement(protocol, results_by_method, project_root, *, bootstrap_repeats=2000, seed=20260930):
    """Score four complete post-lock results using only protocol evaluation refs.

    Each result is ``{'all_predictions_locked_before_scoring': True,
    'records': [...]}``; records use the original hundred evaluation schema.
    Missing endpoint rows become explicit unavailable outcomes; missing cases or
    unfinished methods are rejected before any evaluation file is opened.
    """
    if isinstance(bootstrap_repeats, bool) or not isinstance(bootstrap_repeats, int) or bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    cases, parsed = _barrier(protocol, results_by_method)
    root = Path(project_root).resolve()
    refs = {cid: _reference_values(_checked_evaluation(case["evaluation"], root), case)
            for cid, case in cases.items()}
    records = {method: [_normalize(cases[cid], ep, parsed[method][0].get((cid, ep)), parsed[method][1][cid], refs[cid].get(ep))
                        for cid in sorted(cases) for ep in ENDPOINTS] for method in METHODS}
    endpoints = {}
    for ep in ENDPOINTS:
        rows = {method: [r for r in records[method] if r["endpoint"] == ep] for method in METHODS}
        scored = {method: {r["case_id"]: r for r in rr if r["scorable"]} for method, rr in rows.items()}
        common = sorted(set.intersection(*(set(v) for v in scored.values())))
        methods = {}
        for method in METHODS:
            rr = rows[method]
            reference_n = sum(r["reference_value"] is not None for r in rr)
            prediction_n = sum(r["prediction_value"] is not None for r in rr)
            methods[method] = {"registered_n": len(cases), "reference_available_n": reference_n,
                "missing_reference_n": len(cases) - reference_n,
                "prediction_available_n": prediction_n,
                "prediction_available_coverage_registered": prediction_n / len(cases),
                "scorable_coverage_registered": len(scored[method]) / len(cases),
                "scorable_coverage_reference": len(scored[method]) / reference_n if reference_n else None,
                "reference_available_prediction_missing_n": sum(r["reference_value"] is not None and r["prediction_value"] is None for r in rr),
                "construction_status_counts": dict(Counter(r["construction_status"] for r in rr)),
                "prediction_missing_reason_counts": dict(Counter(r["prediction_missing_reason"] for r in rr if r["prediction_value"] is None)),
                "policy_status_counts": dict(Counter(r["policy_status"] for r in rr)),
                "own_scorable": _metrics(list(scored[method].values()), bootstrap_repeats, seed),
                "all_methods_common_scorable": _metrics([scored[method][cid] for cid in common], bootstrap_repeats, seed),
                "risk_coverage": _risk(rr, bootstrap_repeats, seed)}
        endpoints[ep] = {"registered_n": len(cases), "all_methods_common_scorable_n": len(common),
            "all_methods_common_scorable_case_ids": common, "methods": methods,
            "paired_comparisons": {right + "_vs_" + left: {
                "left_method": left, "right_method": right,
                **_paired(scored[left], scored[right], len(cases), bootstrap_repeats, seed)}
                for left, right in combinations(METHODS, 2)}}
    return {"version": VERSION, "evaluation_scope": SCOPE, "total_compounds": len(cases),
            "methods": list(METHODS), "clinical_curve_validation_n": 0,
            "references": {cid: case["evaluation"] for cid, case in cases.items()},
            "bootstrap": {"unit": "compound", "repeats": bootstrap_repeats, "seed": seed,
                          "confidence_level": .95, "paired": True, "study_clustered": False},
            "endpoint_metrics": endpoints, "records_by_method": records,
            "limitations": ["Original aggregate PK references only; no matched clinical curves.",
                "Pairwise metrics require both predictions; all failure and availability counts remain explicit.",
                "Different own-scorable subsets are descriptive; use common-case or paired comparisons.",
                "Risk scores are supplied, never fabricated; absent comparisons remain unassessed.",
                "Descriptive bootstrap intervals do not establish prospective or clinical validity."]}
