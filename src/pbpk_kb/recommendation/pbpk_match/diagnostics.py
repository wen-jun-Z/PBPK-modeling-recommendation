"""Array-only diagnostics for automatic PBPK document-support judgments.

These summaries never load labels, select a training split, or certify scientific
gold. A paired comparison requires a matching candidate/grade digest per query.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np

METRICS = ("ndcg@10", "recall@10", "mrr")
BOOTSTRAP_SEED = 20260927


def _ratio(a, b):
    return float(a / b) if b else None


def _grade_array(grades):
    values = np.asarray(grades)
    if values.ndim != 2 or not values.shape[1] or not np.isin(values, [-1, 0, 1, 2, 3]).all():
        raise ValueError("grades must be [Q,C], containing -1 or 0..3")
    return values.astype(int)


def _unique_records(records, key):
    ids = [record.get(key) for record in records]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"Records need unique nonempty {key} values")
    return ids


def bind_label_pools(report, grades, query_records, candidate_records):
    """Return a report copy binding every candidate's grade, including unknown.

    Hashing all candidates prevents comparing a reduced or differently judged
    catalog as though it were the same pool. Counts alone cannot establish this.
    No score, source identity feature or training signal is added to model input.
    """
    grades = _grade_array(grades)
    qids = _unique_records(query_records, "query_id")
    cids = _unique_records(candidate_records, "candidate_id")
    if grades.shape != (len(qids), len(cids)):
        raise ValueError("Grade dimensions differ from query/candidate records")
    rows = report.get("per_query", [])
    if set(_unique_records(rows, "query_id")) != set(qids):
        raise ValueError("Report query IDs differ from the supplied labels")
    lookup = {qid: index for index, qid in enumerate(qids)}
    compounds = {record["query_id"]: record.get("compound_id") for record in query_records}
    result = dict(report)
    bound = []
    for row in rows:
        i = lookup[row["query_id"]]
        if row.get("compound_id") != compounds[row["query_id"]]:
            raise ValueError("Report compound identity differs from supplied query")
        if row.get("known_grades") != int((grades[i] >= 0).sum()) or row.get("candidate_count") != len(cids):
            raise ValueError("Report grade coverage differs from supplied labels")
        pool = sorted(zip(cids, grades[i].tolist()))
        digest = hashlib.sha256(json.dumps(pool, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        bound.append({**row, "candidate_grade_pool_sha256": digest})
    result.update(per_query=bound, label_pool_binding="all candidate IDs and grades including unknown; sorted by candidate ID")
    return result


def compare_reports_paired(report_a, report_b, seed=BOOTSTRAP_SEED, replicates=2000):
    """A-minus-B query-macro differences, resampling whole compound clusters.

    The eligible population is the common judged-only positive-query subset.
    This is conditional subset inference, never full-catalog accuracy. One
    compound can yield a point difference but cannot yield a cluster interval.
    """
    if not isinstance(replicates, (int, np.integer)) or isinstance(replicates, bool) or replicates < 2:
        raise ValueError("replicates must be an integer >= 2")
    a = {row["query_id"]: row for row in report_a.get("per_query", [])}
    b = {row["query_id"]: row for row in report_b.get("per_query", [])}
    _unique_records(report_a.get("per_query", []), "query_id")
    _unique_records(report_b.get("per_query", []), "query_id")
    common = sorted(set(a) & set(b))
    eligible = [qid for qid in common if all(
        row.get("judged_only") is not None and all(
            row["judged_only"].get(name) is not None and np.isfinite(row["judged_only"][name]) for name in METRICS)
        for row in (a[qid], b[qid]))]
    result = {
        "state": "unavailable", "scientific_gold": False, "direction": "report_a_minus_report_b",
        "pool": "intersection of query IDs with defined judged-only positive-query metrics in both reports",
        "aggregation": "query macro; whole compound clusters resampled with replacement",
        "total_queries_a": len(a), "total_queries_b": len(b), "common_queries": len(common),
        "eligible_queries": len(eligible), "eligible_compounds": 0, "seed": int(seed), "replicates": int(replicates),
        "metrics": {name: {"point_difference": None, "ci95": None} for name in METRICS},
        "reason": "no_common_eligible_queries", "label_pools_verified": False,
        "limitation": "Conditional automatic-label agreement; clusters are drugs, not independently verified studies. This is not a test-set claim or clinical accuracy.",
    }
    if not eligible:
        return result
    missing, different, compound_mismatch = [], [], []
    for qid in eligible:
        left, right = a[qid], b[qid]
        if not left.get("candidate_grade_pool_sha256") or not right.get("candidate_grade_pool_sha256"):
            missing.append(qid)
        elif left["candidate_grade_pool_sha256"] != right["candidate_grade_pool_sha256"]:
            different.append(qid)
        if not left.get("compound_id") or left.get("compound_id") != right.get("compound_id"):
            compound_mismatch.append(qid)
    if missing or different or compound_mismatch:
        result.update(reason="unverified_or_different_label_pools_or_compounds",
                      missing_pool_binding_queries=missing, different_pool_queries=different,
                      compound_mismatch_queries=compound_mismatch)
        return result
    groups = defaultdict(list)
    for qid in eligible:
        groups[a[qid]["compound_id"]].append(qid)
    compounds = sorted(groups)
    counts = np.array([len(groups[compound]) for compound in compounds])
    sums = np.array([[sum(a[qid]["judged_only"][name] - b[qid]["judged_only"][name]
                              for qid in groups[compound]) for name in METRICS] for compound in compounds], dtype=float)
    points = sums.sum(axis=0) / counts.sum()
    draws = None
    if len(compounds) >= 2:
        indices = np.random.default_rng(seed).integers(len(compounds), size=(replicates, len(compounds)))
        draws = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)[:, None]
    for column, name in enumerate(METRICS):
        result["metrics"][name] = {
            "point_difference": float(points[column]),
            "compound_macro_point_difference": float(np.mean(sums[:, column] / counts)),
            "ci95": np.quantile(draws[:, column], [.025, .975]).tolist() if draws is not None else None,
        }
    result.update(state="evaluated" if draws is not None else "point_only",
                  reason=None if draws is not None else "fewer_than_two_eligible_compounds",
                  eligible_compounds=len(compounds), eligible_compound_ids=compounds,
                  label_pools_verified=True, few_compound_clusters=len(compounds) < 5,
                  bootstrap_interpretation="Percentile interval on the paired eligible-query difference; no independent-pair resampling")
    return result


def risk_coverage_from_predictions(scores, probabilities, app, coverage_levels=(.25, .5, .75, 1)):
    """Selective risk on known Y/N Top-1 outcomes, exposing unknown/invalid mass.

    Exact-score Top-1 ties receive uniform candidate probability, not an ID-based
    winner. Select query sets by mean P(Y) across that tie. Confidence ties at
    each coverage boundary are all included, so actual coverage can exceed the
    requested target. Risk conditions only on known Y/N candidate probability
    mass and reports the excluded U and invalid mass separately.
    """
    scores = np.asarray(scores, dtype=float)
    app = np.asarray(app)
    probabilities = np.asarray(probabilities, dtype=float)
    if scores.ndim != 2 or not scores.shape[1] or not np.isfinite(scores).all():
        raise ValueError("scores must be finite [Q,C]")
    if app.shape != scores.shape or not np.isin(app, [-1, 0, 1, 2]).all():
        raise ValueError("app must be [Q,C] with invalid=-1,Y=0,N=1,U=2")
    if (probabilities.shape != (*scores.shape, 3) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(-1), 1, rtol=0, atol=1e-7)):
        raise ValueError("probabilities must be normalized finite [Q,C,3] in Y,N,U order")
    levels = tuple(float(level) for level in coverage_levels)
    if not levels or any(not np.isfinite(level) or not 0 < level <= 1 for level in levels):
        raise ValueError("coverage_levels must be in (0,1]")
    result = {"scientific_gold": False, "query_count": len(scores), "curve": [],
              "confidence": "mean P(Y) over all exact-score Top-1 candidates",
              "tie_policy": "uniform Top-1 candidate expectation; include every query tied at the confidence boundary",
              "risk_denominator": "selected known Y/N expected Top-1 mass, not all selected queries",
              "coverage_denominator": "all queries, including unknown and invalid labels",
              "interpretation": "Automatic documented-applicability risk conditional on known outcomes; unknown/invalid outcomes are neither errors nor successes"}
    if not len(scores):
        return {**result, "state": "unavailable", "reason": "no_queries"}
    top = scores == scores.max(axis=1, keepdims=True)
    weights = top / top.sum(axis=1, keepdims=True)
    confidence = np.sum(probabilities[..., 0] * weights, axis=1)
    known = np.sum(np.isin(app, [0, 1]) * weights, axis=1)
    error = np.sum((app == 1) * weights, axis=1)
    unknown = np.sum((app == 2) * weights, axis=1)
    invalid = np.sum((app == -1) * weights, axis=1)
    fully_known = np.all(~top | np.isin(app, [0, 1]), axis=1)
    ordered = np.sort(confidence)[::-1]
    for level in levels:
        threshold = float(ordered[min(int(np.ceil(len(scores) * level)), len(scores)) - 1])
        selected = confidence >= threshold
        count = int(selected.sum())
        known_mass, wrong_mass = float(known[selected].sum()), float(error[selected].sum())
        result["curve"].append({
            "target_coverage": level, "probability_threshold": threshold,
            "selected_queries": count, "coverage": count / len(scores),
            "risk_known_only": _ratio(wrong_mass, known_mass),
            "known_expected_top1_mass": known_mass, "error_expected_top1_mass": wrong_mass,
            "known_expected_top1_fraction": float(known[selected].mean()),
            "unknown_expected_top1_fraction": float(unknown[selected].mean()),
            "invalid_expected_top1_fraction": float(invalid[selected].mean()),
            "unknown_or_invalid_expected_top1_fraction": float((unknown[selected] + invalid[selected]).mean()),
            "selected_queries_with_unknown_top1_fraction": float(np.mean(unknown[selected] > 0)),
            "selected_queries_with_invalid_top1_fraction": float(np.mean(invalid[selected] > 0)),
            "selected_queries_with_fully_known_top1": int(fully_known[selected].sum()),
            "reason": None if known_mass else "no_known_Y_N_top1_mass_in_selected_queries",
        })
    return {**result, "state": "evaluated", "reason": None}


def summarize_grade_support(grades, query_records=None):
    """Describe label support without implying absent grades have been learned."""
    values = _grade_array(grades)
    counts = {str(grade): int((values == grade).sum()) for grade in range(4)}
    present = [grade for grade in range(4) if counts[str(grade)]]
    result = {"scientific_gold": False, "total_queries": len(values), "candidate_count": values.shape[1],
              "total_pairs": int(values.size), "known_pairs": sum(counts.values()),
              "unknown_pairs": int((values < 0).sum()), "known_fraction": _ratio(sum(counts.values()), values.size),
              "grade_counts": counts, "present_grades": present,
              "absent_grades": [grade for grade in range(4) if grade not in present],
              "queries_with_known_positive": int(np.any(values >= 2, axis=1).sum()),
              "queries_with_unequal_known_grades": sum(len(set(row[row >= 0])) >= 2 for row in values),
              "grade2_observed": 2 in present, "grade2_learning_claim_supported": False,
              "grade2_reason": "Grade 2 examples are absent; partial applicability or adaptation ranking cannot be claimed learned" if 2 not in present else
              "Grade 2 examples exist; their presence alone does not demonstrate learned generalization",
              "independent_pair_samples": False}
    if query_records is not None:
        if len(query_records) != len(values) or any(not row.get("compound_id") for row in query_records):
            raise ValueError("query_records must match grade rows and supply compound_id")
        compounds = sorted({row["compound_id"] for row in query_records})
        result["compound_count"] = len(compounds)
        result["compounds_with_known_positive"] = len({row["compound_id"] for row, grades_row in zip(query_records, values) if np.any(grades_row >= 2)})
        result["by_compound"] = {
            compound: {str(grade): int(np.sum(values[[i for i, row in enumerate(query_records) if row["compound_id"] == compound]] == grade))
                       for grade in range(4)} for compound in compounds}
    return result
