"""Evaluate a shared catalog-identity policy without rewriting ranking scores.

Grades remain source-judgment targets. Eligibility comes only from the supplied
request and catalog metadata. All ranking denominators/ideal rankings retain
the *pre-policy* known pool, so filtering a known positive incurs a penalty.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .evaluation import _labels, _matrix


def _ratio(a, b):
    return float(a / b) if b else None


def retained_pool_rank_metrics(scores, grades, eligible):
    """Known-label list metrics with a fixed, pre-filter positive/ideal pool.

    Unknown candidates are excluded only for this explicitly conditional metric.
    Ties in the eligible known list use uniform permutations, matching the frozen
    evaluator. The pre-filter pool must have a grade >= 2 to define its metrics.
    """
    scores, grades, eligible = np.asarray(scores, float), np.asarray(grades), np.asarray(eligible)
    if (scores.ndim != 1 or grades.shape != scores.shape or eligible.shape != scores.shape
            or eligible.dtype.kind != "b" or not np.isfinite(scores).all()
            or not np.isin(grades, [-1, 0, 1, 2, 3]).all()):
        raise ValueError("Expected finite score, valid grade and boolean eligibility vectors of equal shape")
    original = grades[grades >= 0].astype(int)
    positive_count = int((original >= 2).sum())
    if not positive_count:
        return None
    selected = eligible & (grades >= 0)
    s, g = scores[selected], grades[selected].astype(int)
    order = np.argsort(-s, kind="stable")
    s, g = s[order], g[order]
    gains, positive = np.zeros(len(g)), np.zeros(len(g))
    reciprocal, found_positive, top_grade = 0.0, False, None
    if len(g):
        starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
        ends = np.r_[starts[1:], len(g)]
        top_grade = float(np.mean(g[:ends[0]]))
        for start, end in zip(starts, ends):
            block = g[start:end]
            gains[start:end] = np.mean(2.0 ** block - 1)
            count = int((block >= 2).sum())
            positive[start:end] = count / len(block)
            if not found_positive and count:
                found_positive, prefix = True, 1.0
                for j in range(len(block) - count + 1):
                    reciprocal += prefix * count / (len(block) - j) / (int(start) + j + 1)
                    prefix *= (len(block) - count - j) / (len(block) - j)
    cutoff = min(10, len(g))
    original_cutoff = min(10, len(original))
    ideal = 2.0 ** np.sort(original)[::-1] - 1
    ideal_dcg = float(np.sum(ideal[:original_cutoff] / np.log2(np.arange(original_cutoff) + 2)))
    dcg = float(np.sum(gains[:cutoff] / np.log2(np.arange(cutoff) + 2)))
    return {"ndcg@10": dcg / ideal_dcg,
            "recall@10": float(positive[:cutoff].sum() / positive_count),
            "mrr": float(reciprocal), "top1_expected_grade": top_grade,
            "pre_policy_known_candidates": len(original), "pre_policy_known_positives": positive_count,
            "pre_policy_idcg@10": ideal_dcg, "policy_known_candidates": len(g),
            "policy_known_positives": int((g >= 2).sum())}


def evaluate_identity_policy(scores, grades, app, query_records, candidate_records, *, policy_function=None):
    """Apply the same outcome-free policy to each method's untouched raw scores.

    This report has a separate policy dimension. It must not replace the original
    280-candidate ranking report or attribute eligibility-rule benefits to a neural
    architecture. No files, services or scientific labels are loaded here.
    """
    scores = _matrix(scores, "scores")
    grades = _labels(grades, "grades", scores.shape, [-1, 0, 1, 2, 3])
    app = _labels(app, "app", scores.shape, [-1, 0, 1, 2])
    if len(query_records) != scores.shape[0] or len(candidate_records) != scores.shape[1]:
        raise ValueError("Query/candidate record counts must match score axes")
    if policy_function is None:
        from .policy import catalog_identity_policy
        policy_function = catalog_identity_policy
    per_query, masks = [], []
    candidate_ids = [row.get("candidate_id", str(i)) for i, row in enumerate(candidate_records)]
    for i, query in enumerate(query_records):
        decision = policy_function(query, candidate_records)
        mask_value = decision.get("eligible_mask", decision.get("eligible"))
        mask = np.asarray(mask_value)
        if mask.shape != (scores.shape[1],) or mask.dtype.kind != "b":
            raise ValueError("Policy must return one boolean eligible_mask value per candidate")
        if "eligible" in decision and "eligible_mask" in decision:
            if not np.array_equal(decision["eligible"], decision["eligible_mask"]):
                raise ValueError("Conflicting eligibility aliases")
        if "abstain" in decision and bool(decision["abstain"]) != (not mask.any()):
            raise ValueError("Policy abstention and eligibility mask disagree")
        masks.append(mask)
        original_known = grades[i] >= 0
        positives = grades[i] >= 2
        eligible_indices = np.flatnonzero(mask)
        top = np.zeros(scores.shape[1], dtype=bool)
        if len(eligible_indices):
            top = mask & (scores[i] == scores[i, eligible_indices].max())
        filtered_positive = positives & ~mask
        per_query.append({
            "query_id": query.get("query_id", str(i)), "compound_id": query.get("compound_id"),
            "policy_version": decision.get("policy_version", "caller_supplied_policy"),
            "policy_scope": decision.get("scope"), "policy_abstain": not bool(mask.any()),
            "policy_abstention_reason": decision.get("abstention_reason"),
            "eligible_candidates": int(mask.sum()), "eligible_original_catalog_indices": eligible_indices.tolist(),
            "pre_policy_known_candidates": int(original_known.sum()), "pre_policy_known_positives": int(positives.sum()),
            "eligible_known_candidates": int((original_known & mask).sum()),
            "eligible_unknown_candidates": int((~original_known & mask).sum()),
            "filtered_known_positive_count": int(filtered_positive.sum()),
            "filtered_known_positive_candidate_ids": [candidate_ids[j] for j in np.flatnonzero(filtered_positive)],
            "metrics": retained_pool_rank_metrics(scores[i], grades[i], mask),
            "policy_top1_tie_size": int(top.sum()),
            "policy_top1_unknown_grade_fraction": float((grades[i, top] < 0).mean()) if top.any() else None,
            "policy_top1_unknown_applicability_fraction": float(np.isin(app[i, top], [-1, 2]).mean()) if top.any() else None,
            "policy_top1_invalid_applicability_fraction": float((app[i, top] == -1).mean()) if top.any() else None,
        })
    measurable = [row for row in per_query if row["metrics"] is not None]
    covered = [row for row in per_query if not row["policy_abstain"]]
    aggregate = {name: float(np.mean([row["metrics"][name] for row in measurable])) if measurable else None
                 for name in ("ndcg@10", "recall@10", "mrr")}
    grade_unknown_mass = sum(row["policy_top1_unknown_grade_fraction"] for row in covered)
    app_unknown_mass = sum(row["policy_top1_unknown_applicability_fraction"] for row in covered)
    mask_binding = json.dumps({"query_ids": [row["query_id"] for row in per_query],
                               "candidate_ids": candidate_ids, "masks": [mask.tolist() for mask in masks]},
                              ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    filtered_positives = sum(row["filtered_known_positive_count"] for row in per_query)
    original_positives = int((grades >= 2).sum())
    return {
        "policy_dimension": "shared_catalog_identity_eligibility",
        "policy_versions": sorted({row["policy_version"] for row in per_query}),
        "scientific_gold": False, "policy_is_scientific_applicability_label": False,
        "raw_scores_modified": False, "raw_scores_sha256": hashlib.sha256(scores.astype('<f8').tobytes()).hexdigest(),
        "policy_mask_sha256": hashlib.sha256(mask_binding.encode()).hexdigest(),
        "interpretation": "Shared metadata eligibility policy, evaluated separately from raw full-catalog ranking; rule gains are not neural architectural gains",
        "ranking_population": "Eligible known-label recommendation list; unknowns removed only for this explicitly conditional list metric",
        "denominator_policy": "Recall positives and NDCG ideal ordering remain the full pre-policy known pool; filtering positives is penalized",
        "tie_policy": "Uniform permutation expectation within exact-score ties; no candidate-ID winner",
        "metrics": aggregate, "eligible_ranking_queries": len(measurable),
        "queries_without_pre_policy_known_positive": len(per_query) - len(measurable),
        "coverage": {
            "total_queries": len(per_query), "candidate_count": scores.shape[1], "total_pairs": int(scores.size),
            "known_grade_pairs": int((grades >= 0).sum()), "known_grade_fraction": _ratio(int((grades >= 0).sum()), scores.size),
            "queries_with_nonempty_eligible_catalog": len(covered), "nonempty_candidate_coverage": _ratio(len(covered), len(per_query)),
            "zero_eligible_queries": len(per_query) - len(covered),
            "zero_eligible_with_known_positive_queries": sum(row["policy_abstain"] and row["pre_policy_known_positives"] > 0 for row in per_query),
            "filtered_known_positive_count": filtered_positives,
            "pre_policy_known_positive_count": original_positives,
            "filtered_known_positive_fraction": _ratio(filtered_positives, original_positives),
            "eligible_known_candidate_count": sum(row["eligible_known_candidates"] for row in per_query),
            "eligible_unknown_candidate_count": sum(row["eligible_unknown_candidates"] for row in per_query),
            "top1_unknown_grade_rate_among_nonempty_queries": _ratio(grade_unknown_mass, len(covered)),
            "top1_unknown_grade_mass_over_all_queries": _ratio(grade_unknown_mass, len(per_query)),
            "top1_unknown_applicability_rate_among_nonempty_queries": _ratio(app_unknown_mass, len(covered)),
            "top1_queries_containing_unknown_grade": sum(row["policy_top1_unknown_grade_fraction"] > 0 for row in covered),
        },
        "per_query": per_query,
    }
