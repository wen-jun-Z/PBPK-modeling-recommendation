"""Document-support evaluation, with unknown labels and tied scores preserved.

No scientific labels are loaded here. Callers must supply their permitted split.
Grades are -1/0/1/2/3; applicability is invalid=-1, Y=0, N=1, U=2.
Metrics describe agreement with the supplied labels, not PBPK execution accuracy.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _matrix(value, name, shape=None):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or not array.shape[1] or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite [query,candidate] matrix")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape differs from scores")
    return array


def _labels(value, name, shape, domain):
    array = _matrix(value, name, shape)
    if not np.isin(array, domain).all():
        raise ValueError(f"{name} contains an unsupported label")
    return array.astype(int)


def _probabilities(value, shape):
    array = np.asarray(value, dtype=float)
    if (array.shape != (*shape, 3) or not np.isfinite(array).all()
            or np.any(array < 0) or np.any(array > 1)
            or not np.allclose(array.sum(axis=-1), 1, atol=1e-7, rtol=0)):
        raise ValueError("probabilities must be normalized finite [Q,C,3] in Y,N,U order")
    return array


def tie_aware_rank_metrics(scores, grades):
    """Exact expectation over uniform permutations within exact-score ties.

    Same DCG/Recall/MRR definition as the frozen frontier evaluator. Unknowns
    invalidate this pool; callers may explicitly construct a judged-only pool.
    """
    scores, grades = np.asarray(scores, float), np.asarray(grades)
    if scores.ndim != 1 or grades.shape != scores.shape:
        raise ValueError("Expected equally sized score and grade vectors")
    if not np.isfinite(scores).all() or not np.isin(grades, [-1, 0, 1, 2, 3]).all():
        raise ValueError("Invalid rank input")
    if not len(grades) or np.any(grades < 0) or not np.any(grades >= 2):
        return None
    order = np.argsort(-scores, kind="stable")
    s, g = scores[order], grades[order].astype(int)
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    ends = np.r_[starts[1:], len(g)]
    gain, positive = np.zeros(len(g)), np.zeros(len(g))
    reciprocal = None
    for start, end in zip(starts, ends):
        block = g[start:end]
        gain[start:end] = np.mean(2.0 ** block - 1)
        count = int((block >= 2).sum())
        positive[start:end] = count / len(block)
        if reciprocal is None and count:
            prefix, reciprocal = 1.0, 0.0
            for j in range(len(block) - count + 1):
                reciprocal += prefix * count / (len(block) - j) / (int(start) + j + 1)
                prefix *= (len(block) - count - j) / (len(block) - j)
    stop = min(10, len(g))
    discount = 1 / np.log2(np.arange(len(g)) + 2)
    ideal = 2.0 ** np.sort(g)[::-1] - 1
    return {
        "ndcg@10": float(np.sum(gain[:stop] * discount[:stop]) / np.sum(ideal[:stop] * discount[:stop])),
        "recall@10": float(np.sum(positive[:stop]) / np.sum(g >= 2)),
        "mrr": float(reciprocal),
        "top1_expected_grade": float(np.mean(g[:ends[0]])),
        "top1_tie_size": int(ends[0]),
    }


def _aggregate(rows, field):
    eligible = [row[field] for row in rows if row[field] is not None]
    names = ("ndcg@10", "recall@10", "mrr", "top1_expected_grade")
    return {
        "eligible_queries": len(eligible), "total_queries": len(rows),
        "eligible_query_fraction": _ratio(len(eligible), len(rows)),
        "metrics": {key: float(np.mean([row[key] for row in eligible])) if eligible else None for key in names},
    }


def _pair_preference(scores, grades, query, candidates):
    compounds = set(query.get("compounds", []))
    indices = [i for i, row in enumerate(candidates) if compounds & set(row.get("compounds", []))]
    count, correct, identical = 0, 0.0, 0
    for offset, left in enumerate(indices):
        for right in indices[offset + 1:]:
            a, b = candidates[left], candidates[right]
            if not a.get("family_id") or not b.get("family_id") or a["family_id"] == b["family_id"]:
                continue
            material_a = a.get("source_packet_sha256") or a.get("material_key")
            material_b = b.get("source_packet_sha256") or b.get("material_key")
            if material_a and material_a == material_b:
                identical += 1
                continue
            if grades[left] < 0 or grades[right] < 0 or grades[left] == grades[right]:
                continue
            count += 1
            correct += .5 if scores[left] == scores[right] else float(
                (scores[left] > scores[right]) == (grades[left] > grades[right]))
    return {"pairs": count, "correct_credit": correct, "accuracy": _ratio(correct, count),
            "source_indistinguishable_pairs_excluded": identical}


def _applicability_metrics(app, probabilities):
    valid, binary = app >= 0, (app == 0) | (app == 1)
    support = [int((app == i).sum()) for i in range(3)]
    result = {
        "class_order": ["Y", "N", "U"], "valid_pairs": int(valid.sum()),
        "invalid_or_unprocessed_pairs": int((~valid).sum()),
        "valid_fraction": _ratio(int(valid.sum()), app.size), "class_support": dict(zip(("Y", "N", "U"), support)),
        "binary_known_pairs": int(binary.sum()), "binary_known_fraction": _ratio(int(binary.sum()), app.size),
        "macro_f1": None, "per_class_f1": {key: None for key in ("Y", "N", "U")},
        "brier": None, "ece": None, "calibration_bins": [],
        "binary_probability_definition": "Unconditional P(Y) on documented Y/N pairs; U labels excluded, not recoded as N.",
        "reason": "probabilities_not_supplied", "binary_reason": "probabilities_not_supplied",
    }
    if probabilities is None:
        return result
    prediction = probabilities.argmax(axis=-1)
    confusion = np.zeros((3, 3), dtype=int)
    np.add.at(confusion, (app[valid], prediction[valid]), 1)
    result["confusion_true_rows_predicted_columns"] = confusion.tolist()
    for i, name in enumerate(("Y", "N", "U")):
        denominator = confusion[i].sum() + confusion[:, i].sum()
        # Missing target classes have no measurable recall or per-class F1.
        if support[i]:
            result["per_class_f1"][name] = _ratio(2 * confusion[i, i], denominator)
    if all(support):
        result["macro_f1"] = float(np.mean(list(result["per_class_f1"].values())))
        result["reason"] = None
    else:
        result["reason"] = "three_class_macro_f1_requires_observed_Y_N_U"
    if not support[0] or not support[1]:
        result["binary_reason"] = "binary_metrics_require_observed_Y_and_N"
        return result
    target, probability = (app[binary] == 0).astype(float), probabilities[..., 0][binary]
    result["brier"] = float(np.mean((probability - target) ** 2))
    result["ece"], result["binary_reason"] = 0.0, None
    buckets = np.minimum((probability * 10).astype(int), 9)
    for i in range(10):
        chosen = buckets == i
        if not chosen.any():
            continue
        confidence, fraction = float(probability[chosen].mean()), float(target[chosen].mean())
        result["ece"] += float(chosen.mean()) * abs(confidence - fraction)
        result["calibration_bins"].append({"bin": i, "count": int(chosen.sum()),
                                           "mean_probability": confidence, "observed_Y_fraction": fraction})
    return result


def evaluate_ranking(scores, grades, app, query_records: Sequence[dict],
                     candidate_records: Sequence[dict], probabilities=None):
    """Evaluate a caller-selected split. This function never loads/chooses labels."""
    scores = _matrix(scores, "scores")
    grades = _labels(grades, "grades", scores.shape, [-1, 0, 1, 2, 3])
    app = _labels(app, "app", scores.shape, [-1, 0, 1, 2])
    if len(query_records) != scores.shape[0] or len(candidate_records) != scores.shape[1]:
        raise ValueError("Record counts must match score axes")
    probabilities = None if probabilities is None else _probabilities(probabilities, scores.shape)
    per_query = []
    for i, query in enumerate(query_records):
        known = grades[i] >= 0
        top = scores[i] == scores[i].max()
        per_query.append({
            "query_id": query.get("query_id", str(i)), "compound_id": query.get("compound_id"),
            "candidate_count": scores.shape[1], "known_grades": int(known.sum()),
            "known_grade_fraction": float(known.mean()), "known_positive_grades": int((grades[i] >= 2).sum()),
            "judged_only": tie_aware_rank_metrics(scores[i, known], grades[i, known]),
            "strict_full_pool": tie_aware_rank_metrics(scores[i], grades[i]),
            "full_catalog_top1_tie_size": int(top.sum()),
            "full_catalog_top1_unknown_grade_fraction": float((~known[top]).mean()),
            "full_catalog_top1_unknown_applicability_fraction": float(np.isin(app[i, top], [-1, 2]).mean()),
            "same_drug_different_family_preference": _pair_preference(scores[i], grades[i], query, candidate_records),
        })
    preferences = [row["same_drug_different_family_preference"] for row in per_query]
    pairs = sum(row["pairs"] for row in preferences)
    eligible = [row["accuracy"] for row in preferences if row["pairs"]]
    unknown = [row["full_catalog_top1_unknown_grade_fraction"] for row in per_query]
    app_unknown = [row["full_catalog_top1_unknown_applicability_fraction"] for row in per_query]
    return {
        "evaluation_target": "Agreement with supplied documented-support labels, not predictive PK accuracy",
        "independent_pk_prediction_accuracy_measured": False,
        "aggregation": "Query macro; candidate pairs and repeated versions are not independent samples",
        "judged_only_interpretation": "Unknown candidates removed in this named subset; its Recall/NDCG is not full-catalog performance",
        "tie_policy": "Uniform permutations within exact equal-score blocks",
        "coverage": {
            "queries": len(per_query), "candidates": scores.shape[1], "total_pairs": int(scores.size),
            "known_grade_pairs": int((grades >= 0).sum()), "known_grade_fraction": _ratio(int((grades >= 0).sum()), scores.size),
            "unknown_grade_pairs": int((grades < 0).sum()),
            "queries_without_known_positive": sum(row["known_positive_grades"] == 0 for row in per_query),
            "full_catalog_top1_unknown_rate": float(np.mean(unknown)) if unknown else None,
            "full_catalog_top1_contains_unknown_query_fraction": float(np.mean(np.array(unknown) > 0)) if unknown else None,
            "full_catalog_top1_unknown_applicability_rate": float(np.mean(app_unknown)) if app_unknown else None,
            "full_catalog_top1_tied_queries": sum(row["full_catalog_top1_tie_size"] > 1 for row in per_query),
        },
        "judged_only": _aggregate(per_query, "judged_only"),
        "strict_full_pool": _aggregate(per_query, "strict_full_pool"),
        "same_drug_different_family_preference": {
            "pairs": pairs, "eligible_queries": len(eligible), "pair_micro_accuracy": _ratio(sum(row["correct_credit"] for row in preferences), pairs),
            "query_macro_accuracy": float(np.mean(eligible)) if eligible else None,
            "source_indistinguishable_pairs_excluded": sum(row["source_indistinguishable_pairs_excluded"] for row in preferences),
            "independence_verified": False, "family_interpretation": "Different registered family IDs do not establish scientific independence",
        },
        "applicability": _applicability_metrics(app, probabilities), "per_query": per_query,
    }


def compare_stability(original_scores, variant_scores, variant="reordered", changed_query_mask=None):
    """Tie-set stability; missing evidence is sensitivity, not an invariance target."""
    left = _matrix(original_scores, "original_scores")
    right = _matrix(variant_scores, "variant_scores", left.shape)
    if variant not in {"reordered", "missing_evidence"}:
        raise ValueError("variant must be reordered or missing_evidence")
    changed = np.ones(len(left), dtype=bool) if changed_query_mask is None else np.asarray(changed_query_mask)
    if changed.shape != (len(left),) or changed.dtype.kind != "b":
        raise ValueError("changed_query_mask must be a boolean query vector")
    left, right = left[changed], right[changed]
    result = {"variant": variant, "changed_input_queries": len(left), "total_queries": len(changed),
              "top1_set_jaccard": None, "inclusive_top10_set_jaccard": None,
              "unique_top1_pair_queries": 0, "unique_top1_consistency": None,
              "interpretation": "Sentence-order invariance, not recommendation correctness" if variant == "reordered" else
              "Sensitivity to removed evidence; neither change nor invariance establishes correctness"}
    if not len(left):
        return result
    top_left, top_right = left == left.max(axis=1, keepdims=True), right == right.max(axis=1, keepdims=True)
    k = min(10, left.shape[1])
    ten_left = left >= np.sort(left, axis=1)[:, -k, None]
    ten_right = right >= np.sort(right, axis=1)[:, -k, None]
    def jaccard(a, b):
        return float(np.mean((a & b).sum(axis=1) / (a | b).sum(axis=1)))
    unique = (top_left.sum(axis=1) == 1) & (top_right.sum(axis=1) == 1)
    result.update(top1_set_jaccard=jaccard(top_left, top_right), inclusive_top10_set_jaccard=jaccard(ten_left, ten_right),
                  unique_top1_pair_queries=int(unique.sum()),
                  unique_top1_consistency=float(np.mean(left[unique].argmax(axis=1) == right[unique].argmax(axis=1))) if unique.any() else None)
    return result


def _softmax(logits, temperature=1.0):
    values = np.asarray(logits, dtype=float) / temperature
    values = values - values.max(axis=-1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=-1, keepdims=True)


def fit_probability_calibrator(logits, app, query_records: Sequence[dict], *,
                               allowed_compound_ids: Iterable[str], excluded_compound_ids: Iterable[str],
                               model_fit_compound_ids: Iterable[str]):
    """Fit one temperature using an inner calibration subset of outer training.

    The model generating these logits must not have fitted on the calibration
    compounds. Provide that model's true fit IDs. Outer validation/test IDs must
    be excluded. This function rejects overlap but cannot audit false provenance.
    Missing Y/N/U classes yield unavailable, never an invented calibration.
    """
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 3 or logits.shape[-1] != 3 or not logits.shape[1] or not np.isfinite(logits).all():
        raise ValueError("logits must be finite [Q,C,3]")
    labels = _labels(app, "app", logits.shape[:2], [-1, 0, 1, 2])
    if len(query_records) != len(logits) or any(not q.get("compound_id") for q in query_records):
        raise ValueError("Every calibration query requires a compound_id")
    allowed, excluded, model_fit = set(allowed_compound_ids), set(excluded_compound_ids), set(model_fit_compound_ids)
    compounds = {q["compound_id"] for q in query_records}
    if allowed & excluded or model_fit & excluded or not model_fit <= allowed:
        raise ValueError("Model-fit/development IDs overlap excluded compounds or escape the allowed outer-training pool")
    if not compounds <= allowed or compounds & excluded or compounds & model_fit:
        raise ValueError("Calibration compounds must be held out from model fitting and outer validation/test")
    support = [int((labels == i).sum()) for i in range(3)]
    result = {"state": "unavailable", "method": "scalar_temperature_fixed_log_grid_0.1_to_10_161_points",
              "temperature": None, "fit_compound_ids": sorted(compounds),
              "model_fit_compound_ids": sorted(model_fit), "excluded_compound_ids": sorted(excluded),
              "fit_pairs": sum(support), "class_support": dict(zip(("Y", "N", "U"), support)),
              "reason": "calibration_requires_observed_Y_N_U", "weighting": "equal compound, then equal query, then equal valid pair"}
    if not all(support):
        return result
    valid = labels >= 0
    weights = np.zeros(labels.shape, dtype=float)
    for compound in compounds:
        indices = [i for i, q in enumerate(query_records) if q["compound_id"] == compound and valid[i].any()]
        for i in indices:
            weights[i, valid[i]] = 1 / (len(indices) * int(valid[i].sum()))
    weights /= weights.sum()
    x, y, w = logits[valid], labels[valid], weights[valid]
    losses = []
    temperatures = np.exp(np.linspace(np.log(.1), np.log(10.), 161))
    for temperature in temperatures:
        z = x / temperature
        z -= z.max(axis=1, keepdims=True)
        nll = np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(y)), y]
        losses.append(float(np.sum(w * nll)))
    best = int(np.argmin(losses))
    result.update(state="fitted", temperature=float(temperatures[best]), reason=None, inner_calibration_nll=losses[best])
    return result


def apply_probability_calibrator(logits, calibrator):
    """Apply an already fitted temperature; never silently replace a failed fit."""
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 3 or logits.shape[-1] != 3 or not np.isfinite(logits).all():
        raise ValueError("logits must be finite [Q,C,3]")
    temperature = calibrator.get("temperature")
    if calibrator.get("state") != "fitted" or not isinstance(temperature, (int, float)) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Probability calibrator is unavailable or invalid")
    return _softmax(logits, temperature)
