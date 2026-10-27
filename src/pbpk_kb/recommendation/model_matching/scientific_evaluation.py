"""Evaluate independently judged PBPK recommendations without inventing negatives.

Input is one JSON object with ``queries``, ``judgments`` and ``predictions`` lists.
Judgments cover every query/candidate pair, including explicit unjudged rows.
Predictions cover the full candidate pool for every method/seed and query.
Passing these consistency checks does not certify scientific ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


VERSION = "pbpk-scientific-evaluation-1.1.0"
LABEL_SOURCES = {"independent_execution", "blinded_expert", "unjudged"}


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _ids(value: Any, field: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    values = [_identifier(item, field) for item in value]
    if len(set(values)) != len(values):
        raise ValueError(f"{field} contains duplicates")
    if nonempty and not values:
        raise ValueError(f"{field} must not be empty")
    return values


def _rows(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    rows = payload.get(key)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{key} must be a list of objects")
    return rows


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _candidate_compounds(row: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Normalize single-drug and DDI membership without inferring missing data."""
    single = row.get("candidate_compound_id")
    if single is not None:
        _identifier(single, "candidate_compound_id")
    multiple = row.get("candidate_compound_ids")
    if multiple is None:
        return (single,) if single is not None else None
    compounds = tuple(sorted(_ids(multiple, "candidate_compound_ids", nonempty=True)))
    if single is not None and single not in compounds:
        raise ValueError("candidate_compound_id must belong to candidate_compound_ids")
    return compounds


def _audit(payload: Mapping[str, Any]) -> tuple[dict, dict, dict, dict]:
    if not isinstance(payload, dict):
        raise ValueError("Input must be a JSON object")
    queries: dict[str, Mapping[str, Any]] = {}
    input_evidence: set[str] = set()
    development_studies: set[str] = set()
    for row in _rows(payload, "queries"):
        query_id = _identifier(row.get("query_id"), "query_id")
        if query_id in queries:
            raise ValueError(f"Duplicate query: {query_id}")
        for field in ("compound_id", "study_id", "task_type"):
            _identifier(row.get(field), field)
        _ids(row.get("candidate_ids"), "candidate_ids", nonempty=True)
        input_evidence.update(_ids(row.get("input_evidence_ids"), "input_evidence_ids"))
        development_studies.update(
            _ids(row.get("development_study_ids"), "development_study_ids")
        )
        queries[query_id] = row
    if not queries:
        raise ValueError("At least one query is required")

    expected = {
        (query_id, candidate_id)
        for query_id, query in queries.items()
        for candidate_id in query["candidate_ids"]
    }
    judgments: dict[tuple[str, str], Mapping[str, Any]] = {}
    gold_evidence: set[str] = set()
    all_judgment_evidence: set[str] = set()
    gold_studies: set[str] = set()
    candidate_metadata: dict[tuple[str, str], str | tuple[str, ...]] = {}
    for row in _rows(payload, "judgments"):
        pair = (
            _identifier(row.get("query_id"), "judgment.query_id"),
            _identifier(row.get("candidate_id"), "judgment.candidate_id"),
        )
        if pair not in expected or pair in judgments:
            raise ValueError(f"Extra or duplicate judgment: {pair}")
        if "grade" not in row or "applicable" not in row:
            raise ValueError(f"Judgment needs explicit grade and applicable (null if unknown): {pair}")
        grade, applicable = row["grade"], row["applicable"]
        if grade is not None and (type(grade) is not int or grade not in range(4)):
            raise ValueError(f"grade must be null or an integer from 0 to 3: {pair}")
        if applicable is not None and type(applicable) is not bool:
            raise ValueError(f"applicable must be null or boolean: {pair}")
        source = row.get("label_source")
        if not isinstance(source, str) or source not in LABEL_SOURCES:
            raise ValueError(f"Independent label_source required; rule/metadata labels forbidden: {pair}")
        evidence = _ids(row.get("evidence_ids"), "judgment.evidence_ids")
        all_judgment_evidence.update(evidence)
        if source == "unjudged":
            if grade is not None or applicable is not None:
                raise ValueError(f"An unjudged row cannot contain known labels: {pair}")
        else:
            if grade is None and applicable is None:
                raise ValueError(f"A judged row needs at least one known label: {pair}")
            if not evidence:
                raise ValueError(f"Known judgments require independent evidence_ids: {pair}")
            gold_evidence.update(evidence)
            gold_studies.add(_identifier(row.get("study_id"), "judgment.study_id"))
        compounds = _candidate_compounds(row)
        if row.get("family_id") is not None:
            _identifier(row["family_id"], "family_id")
        for field, value in (("candidate_compound_ids", compounds), ("family_id", row.get("family_id"))):
            if value is not None:
                metadata_key = (pair[1], field)
                if metadata_key in candidate_metadata and candidate_metadata[metadata_key] != value:
                    raise ValueError(f"Inconsistent candidate metadata across queries: {metadata_key}")
                candidate_metadata[metadata_key] = value
        judgments[pair] = row
    if set(judgments) != expected:
        raise ValueError("Missing judgments; represent unknown pairs explicitly as unjudged")
    overlap = input_evidence & all_judgment_evidence
    if overlap:
        raise ValueError(f"Input evidence overlaps judgment evidence globally: {sorted(overlap)[:5]}")
    overlap = development_studies & gold_studies
    if overlap:
        raise ValueError(f"Development studies overlap judgment studies globally: {sorted(overlap)[:5]}")

    predictions: dict[tuple[str, str], dict[tuple[str, str], Mapping[str, Any]]] = defaultdict(dict)
    for row in _rows(payload, "predictions"):
        method = _identifier(row.get("method"), "prediction.method")
        seed = row.get("seed")
        if not ((type(seed) is int) or (isinstance(seed, str) and seed.strip())):
            raise ValueError("prediction.seed must be an integer or nonempty string")
        group = (method, str(seed))
        pair = (
            _identifier(row.get("query_id"), "prediction.query_id"),
            _identifier(row.get("candidate_id"), "prediction.candidate_id"),
        )
        if pair not in expected or pair in predictions[group]:
            raise ValueError(f"Extra or duplicate prediction: {group}, {pair}")
        _finite(row.get("score"), "prediction.score")
        probability = row.get("probability")
        if probability is not None and not 0 <= _finite(probability, "probability") <= 1:
            raise ValueError("probability must be in [0, 1]")
        if row.get("predicted_applicable") is not None and type(row["predicted_applicable"]) is not bool:
            raise ValueError("predicted_applicable must be null or boolean")
        predictions[group][pair] = row
    if not predictions:
        raise ValueError("At least one method/seed prediction group is required")
    for group, rows in predictions.items():
        if set(rows) != expected:
            raise ValueError(f"Incomplete full candidate predictions for method/seed {group}")
    audit = {
        "query_count": len(queries),
        "compound_count": len({row["compound_id"] for row in queries.values()}),
        "judgment_pair_count": len(judgments),
        "label_sources": dict(Counter(row["label_source"] for row in judgments.values())),
        "grade_known_count": sum(row["grade"] is not None for row in judgments.values()),
        "applicability_known_count": sum(row["applicable"] is not None for row in judgments.values()),
        "independent_judgment_evidence_count": len(gold_evidence),
        "independent_judgment_study_count": len(gold_studies),
        "queries_without_input_evidence_ids": sum(not row["input_evidence_ids"] for row in queries.values()),
        "method_seed_count": len(predictions),
        "input_gold_evidence_overlap_count": 0,
        "development_gold_study_overlap_count": 0,
        "consistency_checks_passed": True,
        "provenance_content_independently_verified": False,
    }
    return queries, judgments, predictions, audit


def _dcg(grades: Sequence[int], k: int) -> float:
    return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades[:k]))


def _query_ranking(rows: Sequence[dict], ks: Sequence[int], threshold: int) -> dict[str, float | None]:
    ordered = sorted(rows, key=lambda row: (-row["score"], row["candidate_id"]))
    grades = [row["grade"] for row in ordered]
    ideal = sorted(grades, reverse=True)
    positives = sum(grade >= threshold for grade in grades)
    result: dict[str, float | None] = {}
    for k in ks:
        result[f"ndcg@{k}"] = _dcg(grades, k) / _dcg(ideal, k)
        result[f"recall@{k}"] = sum(grade >= threshold for grade in grades[:k]) / positives
        result[f"precision@{k}"] = sum(grade >= threshold for grade in grades[:k]) / min(k, len(grades))
        result[f"hit_rate@{k}"] = float(any(grade >= threshold for grade in grades[:k]))
    result["mrr"] = next(1 / rank for rank, grade in enumerate(grades, 1) if grade >= threshold)
    concordance = []
    for left in range(len(ordered)):
        for right in range(left + 1, len(ordered)):
            a, b = ordered[left], ordered[right]
            if a["grade"] == b["grade"]:
                continue
            concordance.append(
                0.5 if a["score"] == b["score"] else float((a["score"] > b["score"]) == (a["grade"] > b["grade"]))
            )
    result["pairwise_accuracy"] = mean(concordance) if concordance else None
    return result


def _mean_metrics(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[str, float | None]:
    return {
        key: mean(values) if (values := [row[key] for row in rows if row.get(key) is not None]) else None
        for key in keys
    }


def _ranking(by_query: Mapping[str, Sequence[dict]], ks: Sequence[int], threshold: int) -> dict:
    keys = [f"{name}@{k}" for k in ks for name in ("ndcg", "recall", "precision", "hit_rate")]
    keys += ["mrr", "pairwise_accuracy"]
    scores = []
    incomplete = no_positive = 0
    for rows in by_query.values():
        if any(row["grade"] is None for row in rows):
            incomplete += 1
        elif not any(row["grade"] >= threshold for row in rows):
            no_positive += 1
        else:
            scores.append(_query_ranking(rows, ks, threshold))
    return {
        "metrics": _mean_metrics(scores, keys),
        "query_count": len(by_query),
        "eligible_query_count": len(scores),
        "coverage": len(scores) / len(by_query) if by_query else None,
        "omitted_query_count": incomplete + no_positive,
        "unjudged_query_count": incomplete,
        "no_positive_query_count": no_positive,
        "pairwise_query_count": sum(row["pairwise_accuracy"] is not None for row in scores),
    }


def _decisions(rows: Sequence[dict]) -> dict:
    available = [row for row in rows if row["applicable"] is not None and row["predicted_applicable"] is not None]
    confusion = [[0, 0], [0, 0]]
    for row in available:
        confusion[int(row["applicable"])][int(row["predicted_applicable"])] += 1
    f1 = {}
    for label in (0, 1):
        tp = confusion[label][label]
        fp = confusion[1 - label][label]
        fn = confusion[label][1 - label]
        if 2 * tp + fp + fn:
            f1[str(label)] = 2 * tp / (2 * tp + fp + fn)
    accepted_known = confusion[0][1] + confusion[1][1]
    known_positive = confusion[1][0] + confusion[1][1]
    known_negative = confusion[0][0] + confusion[0][1]
    supplied_decisions = sum(row["predicted_applicable"] is not None for row in rows)
    recommended = sum(row["predicted_applicable"] is True for row in rows)
    return {
        "pair_count": len(rows),
        "support": len(available),
        "known_gold_count": sum(row["applicable"] is not None for row in rows),
        "missing_prediction_with_known_gold_count": sum(row["applicable"] is not None and row["predicted_applicable"] is None for row in rows),
        "unknown_gold_count": sum(row["applicable"] is None for row in rows),
        "confusion_matrix": confusion,
        "confusion_axes": "rows=gold, columns=prediction; order=[inapplicable, applicable]",
        "macro_f1": mean(f1.values()) if f1 else None,
        "f1_by_present_class": f1,
        "accuracy": sum(confusion[i][i] for i in (0, 1)) / len(available) if available else None,
        "unsafe_recommendation_rate": confusion[0][1] / accepted_known if accepted_known else None,
        "unsafe_recommendation_count": confusion[0][1],
        "recommended_known_gold_count": accepted_known,
        "recommended_unknown_gold_count": sum(row["predicted_applicable"] is True and row["applicable"] is None for row in rows),
        "supplied_decision_count": supplied_decisions,
        "decision_coverage": supplied_decisions / len(rows) if rows else None,
        "recommendation_coverage": recommended / len(rows) if rows else None,
        "recommendation_coverage_among_supplied_decisions": recommended / supplied_decisions if supplied_decisions else None,
        "false_rejection_count": confusion[1][0],
        "false_rejection_rate": confusion[1][0] / known_positive if known_positive else None,
        "false_rejection_denominator": known_positive,
        "false_acceptance_rate": confusion[0][1] / known_negative if known_negative else None,
        "false_acceptance_denominator": known_negative,
    }


def _selective_risk(rows: Sequence[dict]) -> dict:
    """Binary decision risk, preserving unknowns and averaging confidence ties.

    Discrete AURC is mean cumulative error at every retained sample count.
    Within tied confidence groups the cumulative error is its expectation over
    random orderings, so gold labels and input order cannot break the tie.
    Coverage is relative to fully evaluable decisions, with its denominator
    and coverage of all supplied rows separately reported.
    """
    available = [row for row in rows if row["applicable"] is not None
                 and row["predicted_applicable"] is not None and row["probability"] is not None]
    groups: dict[float, list[dict]] = defaultdict(list)
    for row in available:
        confidence = row["probability"] if row["predicted_applicable"] else 1 - row["probability"]
        groups[confidence].append(row)
    accepted = errors = 0
    area = 0.0
    curve = [{"confidence_threshold": None, "retained_count": 0,
              "coverage": 0.0, "error_count": 0, "risk": None}]
    for confidence, group in sorted(groups.items(), reverse=True):
        size = len(group)
        group_errors = sum(row["predicted_applicable"] != row["applicable"] for row in group)
        for step in range(1, size + 1):
            area += (errors + step * group_errors / size) / (accepted + step) / len(available)
        accepted += size
        errors += group_errors
        curve.append({
            "confidence_threshold": confidence, "retained_count": accepted,
            "coverage": accepted / len(available), "error_count": errors,
            "risk": errors / accepted,
        })
    return {
        "row_count": len(rows), "support": len(available),
        "evaluable_coverage": len(available) / len(rows) if rows else None,
        "unknown_gold_count": sum(row["applicable"] is None for row in rows),
        "missing_decision_with_known_gold_count": sum(row["applicable"] is not None and row["predicted_applicable"] is None for row in rows),
        "missing_probability_with_known_gold_and_decision_count": sum(row["applicable"] is not None and row["predicted_applicable"] is not None and row["probability"] is None for row in rows),
        "aurc": area if available else None,
        "full_coverage_risk": errors / len(available) if available else None,
        "confidence": "probability if predicted_applicable=True else 1-probability; no score conversion",
        "coverage_denominator": "Rows with independently known applicability, explicit decision and probability",
        "integration": "Mean cumulative 0/1 error over retained counts 1..N; expectation over random permutations within confidence ties",
        "curve_policy": "Points at complete confidence blocks; no arbitrary ordering of tied rows",
        "curve": curve if available else [],
    }


def _recommendation_decisions(by_query: Mapping[str, Sequence[dict]]) -> dict:
    """Evaluate an explicit and reproducible decision policy, not infer labels.

    A ranking alone makes no recommendation/refusal claim. Among explicitly
    accepted candidates choose the highest rank; only all-False candidate
    decisions establish refusal. Missing decisions remain unknown.
    """
    counts = Counter()
    selected = []
    for rows in by_query.values():
        accepted = [row for row in rows if row["predicted_applicable"] is True]
        if accepted:
            action = "recommend"
            selected.append(sorted(accepted, key=lambda row: (-row["score"], row["candidate_id"]))[0])
        elif all(row["predicted_applicable"] is False for row in rows):
            action = "reject"
        else:
            action = "unknown"
        counts[action] += 1
        if any(row["applicable"] is True for row in rows):
            availability = "available"
        elif all(row["applicable"] is False for row in rows):
            availability = "unavailable"
        else:
            availability = "unknown"
        counts[f"gold_{availability}"] += 1
        counts[f"gold_{availability}_action_{action}"] += 1
    known_selected = [row for row in selected if row["applicable"] is not None]
    unsafe = sum(row["applicable"] is False for row in known_selected)
    positive_decisions = counts["gold_available_action_recommend"] + counts["gold_available_action_reject"]
    negative_decisions = counts["gold_unavailable_action_recommend"] + counts["gold_unavailable_action_reject"]
    total = len(by_query)
    return {
        "policy": "Recommend highest-scoring candidate with predicted_applicable=True; reject only when all candidates are False; otherwise action unknown",
        "query_count": total,
        "recommendation_query_count": counts["recommend"],
        "rejection_query_count": counts["reject"],
        "unknown_action_query_count": counts["unknown"],
        "decision_coverage": (counts["recommend"] + counts["reject"]) / total if total else None,
        "recommendation_coverage": counts["recommend"] / total if total else None,
        "rejection_coverage": counts["reject"] / total if total else None,
        "known_available_query_count": counts["gold_available"],
        "fully_judged_no_applicable_query_count": counts["gold_unavailable"],
        "unknown_availability_query_count": counts["gold_unknown"],
        "false_rejection_count": counts["gold_available_action_reject"],
        "false_rejection_denominator": positive_decisions,
        "false_rejection_rate": counts["gold_available_action_reject"] / positive_decisions if positive_decisions else None,
        "missing_action_with_known_available_query_count": counts["gold_available_action_unknown"],
        "correct_rejection_count": counts["gold_unavailable_action_reject"],
        "correct_rejection_denominator": negative_decisions,
        "correct_rejection_rate": counts["gold_unavailable_action_reject"] / negative_decisions if negative_decisions else None,
        "missing_action_with_no_applicable_query_count": counts["gold_unavailable_action_unknown"],
        "recommended_selected_gold_known_count": len(known_selected),
        "recommended_selected_gold_unknown_count": len(selected) - len(known_selected),
        "unsafe_recommendation_count": unsafe,
        "unsafe_recommendation_rate": unsafe / len(known_selected) if known_selected else None,
        "selected_recommendation_risk_coverage": _selective_risk(selected),
    }


def _calibration(rows: Sequence[dict], bins: int) -> dict:
    available = [row for row in rows if row["applicable"] is not None and row["probability"] is not None]
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in available:
        groups[min(int(row["probability"] * bins), bins - 1)].append(row)
    statistics = []
    for index in range(bins):
        group = groups[index]
        statistics.append({
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "upper_inclusive": index == bins - 1,
            "count": len(group),
            "mean_probability": mean(row["probability"] for row in group) if group else None,
            "observed_applicable_fraction": mean(row["applicable"] for row in group) if group else None,
        })
    return {
        "support": len(available),
        "missing_probability_with_known_gold_count": sum(row["applicable"] is not None and row["probability"] is None for row in rows),
        "brier_score": mean((row["probability"] - row["applicable"]) ** 2 for row in available) if available else None,
        "ece": sum(row["count"] / len(available) * abs(row["mean_probability"] - row["observed_applicable_fraction"]) for row in statistics if row["count"]) if available else None,
        "binning": "equal_width; left_closed, right_open except final bin includes 1",
        "bins": statistics,
    }


def _cohort(queries: Mapping[str, Mapping], joined: Mapping[str, list[dict]], ks: Sequence[int], threshold: int, bins: int) -> dict:
    rows = [row for group in joined.values() for row in group]
    same_drug = {}
    unavailable = insufficient_families = no_grade_difference = duplicate_families = 0
    for query_id, group in joined.items():
        if any(row["candidate_compound_ids"] is None for row in group):
            unavailable += 1
            continue
        same = [row for row in group if queries[query_id]["compound_id"] in row["candidate_compound_ids"]]
        if any(row["family_id"] is None for row in same):
            unavailable += 1
        elif len({row["family_id"] for row in same}) < len(same):
            duplicate_families += 1
        elif len({row["family_id"] for row in same}) < 2:
            insufficient_families += 1
        elif any(row["grade"] is None for row in same):
            same_drug[query_id] = same  # Included in unknown/coverage accounting, never scored.
        elif len({row["grade"] for row in same}) < 2:
            no_grade_difference += 1
        else:
            same_drug[query_id] = same
    same_result = _ranking(same_drug, ks, threshold)
    same_result.update({
        "total_query_count": len(joined),
        "metadata_unavailable_query_count": unavailable,
        "duplicate_family_versions_query_count": duplicate_families,
        "fewer_than_two_independent_families_query_count": insufficient_families,
        "no_grade_difference_query_count": no_grade_difference,
        "coverage_of_all_queries": same_result["eligible_query_count"] / len(joined) if joined else None,
    })
    top = [sorted(group, key=lambda row: (-row["score"], row["candidate_id"]))[0] for group in joined.values()]
    top_known = [row for row in top if row["applicable"] is not None]
    return {
        "ranking": _ranking(joined, ks, threshold),
        "within_drug_ranking": same_result,
        "applicability": _decisions(rows),
        "calibration": _calibration(rows, bins),
        "selective_classification": _selective_risk(rows),
        "recommendation_decisions": _recommendation_decisions(joined),
        "top1_applicability": {
            "known_gold_query_count": len(top_known),
            "unknown_gold_query_count": len(top) - len(top_known),
            "inapplicable_rate": mean(not row["applicable"] for row in top_known) if top_known else None,
        },
    }


def evaluate(payload: Mapping[str, Any], *, ks: Sequence[int] = (1, 3, 5, 10), positive_grade_threshold: int = 2, calibration_bins: int = 10) -> dict:
    """Audit complete pools and return per-method/seed, compound and task results.

    Grades and applicability always come from ``judgments``; similarly named
    fields in prediction objects are ignored. Unknown judgments stay unknown.
    """
    if not ks or any(type(k) is not int or k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("ks must contain distinct positive integers")
    if type(positive_grade_threshold) is not int or positive_grade_threshold not in (1, 2, 3):
        raise ValueError("positive_grade_threshold must be 1, 2 or 3")
    if type(calibration_bins) is not int or calibration_bins < 1:
        raise ValueError("calibration_bins must be a positive integer")
    queries, judgments, predictions, audit = _audit(payload)
    results = []
    for (method, seed), prediction_rows in sorted(predictions.items()):
        joined: dict[str, list[dict]] = defaultdict(list)
        for (query_id, candidate_id), prediction in prediction_rows.items():
            gold = judgments[query_id, candidate_id]
            joined[query_id].append({
                "candidate_id": candidate_id,
                "grade": gold["grade"],
                "applicable": gold["applicable"],
                "score": float(prediction["score"]),
                "probability": prediction.get("probability"),
                "predicted_applicable": prediction.get("predicted_applicable"),
                "candidate_compound_ids": _candidate_compounds(gold),
                "family_id": gold.get("family_id"),
            })
        overall = _cohort(queries, joined, ks, positive_grade_threshold, calibration_bins)
        subgroups = {}
        for field in ("compound_id", "task_type"):
            subgroups[field] = {
                value: _cohort(queries, {qid: rows for qid, rows in joined.items() if queries[qid][field] == value}, ks, positive_grade_threshold, calibration_bins)
                for value in sorted({query[field] for query in queries.values()})
            }
        compound_results = list(subgroups["compound_id"].values())
        compound_metrics = [row["ranking"]["metrics"] for row in compound_results]
        per_query = []
        for query_id, rows in sorted(joined.items()):
            ranking = _ranking({query_id: rows}, ks, positive_grade_threshold)
            omission_reason = (
                "unjudged_candidates" if ranking["unjudged_query_count"]
                else "no_positive_candidate" if ranking["no_positive_query_count"]
                else None
            )
            per_query.append({
                "query_id": query_id,
                "compound_id": queries[query_id]["compound_id"],
                "study_id": queries[query_id]["study_id"],
                "task_type": queries[query_id]["task_type"],
                "candidate_count": len(rows),
                "grade_known_count": sum(row["grade"] is not None for row in rows),
                "grade_complete": all(row["grade"] is not None for row in rows),
                "positive_candidate_count": sum(row["grade"] is not None and row["grade"] >= positive_grade_threshold for row in rows),
                "ranking_eligible": bool(ranking["eligible_query_count"]),
                "ranking_omission_reason": omission_reason,
                "ranking_metrics": ranking["metrics"],
            })
        results.append({
            "method": method,
            "seed": seed,
            "overall": overall,
            "by_compound": subgroups["compound_id"],
            "by_task_type": subgroups["task_type"],
            "per_query": per_query,
            "compound_macro": {
                "ranking_metrics": _mean_metrics(compound_metrics, list(overall["ranking"]["metrics"])),
                "ranking_compound_count": sum(row["ranking"]["eligible_query_count"] > 0 for row in compound_results),
                "applicability_macro_f1": _mean_metrics([row["applicability"] for row in compound_results], ["macro_f1"])["macro_f1"],
                "applicability_compound_count": sum(row["applicability"]["support"] > 0 for row in compound_results),
            },
        })
    return {
        "version": VERSION,
        "scientific_claim_eligible": False,
        "audit": audit,
        "protocol": {
            "positive_grade_threshold": positive_grade_threshold,
            "k": list(ks),
            "gain": "2**grade - 1",
            "precision_denominator": "min(k, candidate_pool_size)",
            "ranking_eligibility": "Every candidate grade known and at least one grade >= positive_grade_threshold",
            "within_drug_protocol": "Compound membership includes DDI candidates. At least two independent families with differing grades are required. Predefine one representative per family in the candidate pool; duplicate same-drug family versions make this metric NA. Never select a representative using prediction scores or gold labels.",
            "overall_ranking_scope": "The supplied version-level pool; may contain multiple versions per family. This is distinct from independent-family selection quality.",
            "tie_handling": "Ranking: score descending, candidate_id ascending; unequal-grade pair score ties: 0.5",
            "macro_f1_classes": "Union of observed gold and predicted classes; absent classes omitted",
            "probability_target": "P(applicable=True), independent of ordinal relevance grade",
            "selective_risk": "Explicit predicted_applicable decisions ranked by their predicted-class probability; tie-averaged discrete AURC on known-gold rows only, with evaluable coverage disclosed",
            "query_decision_policy": "Highest-scoring explicitly applicable candidate is recommended. All candidates explicitly False means refusal. Otherwise action is unknown. No decision or probability is inferred from ranking scores.",
            "seed_handling": "Separate results; repeated seeds are not additional independent compounds or queries",
            "unknown_policy": "Never convert missing judgments or predictions to negatives",
        },
        "limitations": [
            "Identifier disjointness cannot prove evidence content independence or label validity; external provenance review remains required.",
            "Blinded expert judgments do not establish successful execution or pharmacokinetic predictive validity.",
            "No clinical endpoint accuracy, clustered confidence intervals or significance tests are computed by this module.",
            "Candidate family and compound assignments are supplied metadata and require an independent audit.",
            "Metrics computed on judged subsets can be selective; coverage and unknown counts must accompany scores.",
        ],
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--positive-grade-threshold", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--calibration-bins", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        result = evaluate(payload, ks=args.k, positive_grade_threshold=args.positive_grade_threshold, calibration_bins=args.calibration_bins)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
