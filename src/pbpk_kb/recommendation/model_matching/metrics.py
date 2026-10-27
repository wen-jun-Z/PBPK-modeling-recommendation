from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


METRICS_VERSION = "pbpk-model-matching-metrics-1.5.1"
METADATA_METRICS_VERSION = "pbpk-metadata-ranking-metrics-1.0.0"
METADATA_RELEVANCE_SEMANTICS = "metadata_compatibility_not_validated_use"

MINIMUM_GRADE_BY_USE_TIER = {
    "direct_reuse": 3.0,
    "controlled_adaptation": 2.0,
    "mechanistic_reference": 1.0,
    "reject": 0.0,
}


def _predicted_use_tier(row: Mapping[str, Any]) -> str:
    value = row.get("predicted_use_tier")
    if value in MINIMUM_GRADE_BY_USE_TIER:
        return str(value)
    if bool(row.get("reuse_decision")):
        return "direct_reuse"
    if bool(row.get("adaptation_decision")):
        return "controlled_adaptation"
    if bool(row.get("reject_decision")):
        return "reject"
    return "mechanistic_reference"


def _unsafe_use_claim(row: Mapping[str, Any]) -> bool:
    tier = _predicted_use_tier(row)
    if float(row["relevance_grade"]) < MINIMUM_GRADE_BY_USE_TIER[tier]:
        return True
    return tier == "direct_reuse" and not bool(
        row.get("hard_gate_passed", True)
    )


def _decision_class_from_grade(grade: float) -> int:
    return 0 if grade == 3.0 else 1 if grade == 2.0 else 2 if grade == 1.0 else 3


def _decision_quality(
    rows: Sequence[Mapping[str, Any]], *, prediction_key: str
) -> dict[str, Any] | None:
    available = [
        row
        for row in rows
        if row.get(prediction_key) is not None
        and row.get("relevance_grade") is not None
    ]
    if not available:
        return None
    class_count = 4
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for row in available:
        target = _decision_class_from_grade(float(row["relevance_grade"]))
        prediction = int(row[prediction_key])
        if prediction < 0 or prediction >= class_count:
            raise ValueError(
                f"Invalid decision class {prediction} in {prediction_key}."
            )
        confusion[target][prediction] += 1
    present_classes = sorted(
        {
            index
            for index in range(class_count)
            if sum(confusion[index]) > 0
            or sum(row[index] for row in confusion) > 0
        }
    )
    f1_by_class: dict[str, float] = {}
    for decision_class in present_classes:
        true_positive = confusion[decision_class][decision_class]
        false_positive = sum(
            confusion[row][decision_class]
            for row in range(class_count)
            if row != decision_class
        )
        false_negative = sum(
            confusion[decision_class][column]
            for column in range(class_count)
            if column != decision_class
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1_by_class[str(decision_class)] = (
            2 * true_positive / denominator if denominator else 0.0
        )
    correct = sum(confusion[index][index] for index in range(class_count))
    return {
        "macro_f1": mean(f1_by_class.values()) if f1_by_class else 0.0,
        "accuracy": correct / len(available),
        "f1_by_class": f1_by_class,
        "confusion_matrix": confusion,
        "present_classes": present_classes,
        "support": len(available),
    }


def _dcg(relevances: Sequence[float], k: int) -> float:
    return sum(
        (2.0**relevance - 1.0) / math.log2(rank + 2.0)
        for rank, relevance in enumerate(relevances[:k])
    )


def _average_precision(relevances: Sequence[float]) -> float:
    binary = [value >= 2.0 for value in relevances]
    positives = sum(binary)
    if positives == 0:
        return 0.0
    return sum(
        sum(binary[: rank + 1]) / (rank + 1)
        for rank, value in enumerate(binary)
        if value
    ) / positives


def _ece(probabilities: Sequence[float], labels: Sequence[float], bins: int = 10) -> float:
    if bins <= 0 or len(probabilities) != len(labels):
        raise ValueError("ECE requires positive bins and aligned probabilities/labels")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("ECE probabilities must be finite values in [0, 1]")
    if any(value not in (0, 1) for value in labels):
        raise ValueError("ECE targets must be binary")
    if not probabilities:
        return 0.0
    chunks: list[list[int]] = [[] for _ in range(bins)]
    for index, probability in enumerate(probabilities):
        chunks[min(bins - 1, int(probability * bins))].append(index)
    error = 0.0
    for chunk in chunks:
        if not chunk:
            continue
        confidence = mean(probabilities[index] for index in chunk)
        accuracy = mean(labels[index] for index in chunk)
        error += len(chunk) / len(probabilities) * abs(confidence - accuracy)
    return error


def _pairwise_accuracy(relevances: Sequence[float], scores: Sequence[float]) -> float:
    correct = 0.0
    total = 0
    for left in range(len(relevances)):
        for right in range(left + 1, len(relevances)):
            if relevances[left] == relevances[right]:
                continue
            total += 1
            expected = relevances[left] > relevances[right]
            observed = scores[left] > scores[right]
            correct += 0.5 if scores[left] == scores[right] else 1.0 if expected == observed else 0.0
    return correct / total if total else 0.0


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + end - 1) / 2.0 + 1.0
        for index in order[cursor:end]:
            ranks[index] = average_rank
        cursor = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = mean(left_ranks)
    right_mean = mean(right_ranks)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_ranks, right_ranks, strict=True)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left_ranks)
        * sum((y - right_mean) ** 2 for y in right_ranks)
    )
    return numerator / denominator if denominator else None


def _selective_risk(top_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = []
    for row in top_rows:
        if row.get("calibrated_probability") is not None:
            confidence = float(row["calibrated_probability"])
            source = "calibrated_probability"
        elif row.get("predicted_uncertainty") is not None:
            confidence = 1.0 - float(row["predicted_uncertainty"])
            source = "one_minus_predicted_uncertainty"
        else:
            continue
        unsafe = _unsafe_use_claim(row)
        values.append((confidence, float(unsafe), source))
    if not values:
        return {
            "aurc": None,
            "risk_at_50pct_coverage": None,
            "risk_at_80pct_coverage": None,
            "risk_at_100pct_coverage": None,
            "confidence_source": None,
        }
    values.sort(key=lambda value: -value[0])
    cumulative_risk = []
    failures = 0.0
    for rank, (_, unsafe, _) in enumerate(values, 1):
        failures += unsafe
        cumulative_risk.append(failures / rank)

    def at_coverage(coverage: float) -> float:
        index = max(0, min(len(values) - 1, math.ceil(len(values) * coverage) - 1))
        return cumulative_risk[index]

    return {
        "aurc": mean(cumulative_risk),
        "risk_at_50pct_coverage": at_coverage(0.5),
        "risk_at_80pct_coverage": at_coverage(0.8),
        "risk_at_100pct_coverage": cumulative_risk[-1],
        "confidence_source": values[0][2],
    }


def _query_cohort(maximum_grade: float) -> str:
    if maximum_grade >= 3.0:
        return "direct_reuse_available"
    if maximum_grade >= 2.0:
        return "controlled_adaptation_available"
    if maximum_grade >= 1.0:
        return "mechanistic_reference_only"
    return "no_supported_candidate"


def _mean_query_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        - {"max_relevance_grade"}
    )
    return {
        "query_count": len(rows),
        **{
            key: mean(float(row[key]) for row in rows)
            for key in numeric_keys
            if all(row.get(key) is not None for row in rows)
        },
    }


def evaluate_rankings(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    all_rows = list(rows)
    for row in all_rows:
        grouped[str(row["query_id"])].append(row)

    cutoffs = (1, 3, 5, 10)
    per_query: list[dict[str, float | str]] = []
    probabilities: list[float] = []
    binary_labels: list[float] = []
    hard_violations = 0
    false_reuse = 0
    reuse_decisions = 0
    adaptation_decisions = 0
    false_adaptation = 0
    false_rejections = 0
    unsafe_top1 = 0
    reuse_adapt_true_positive = 0
    reuse_adapt_false_positive = 0
    reuse_adapt_false_negative = 0
    reuse_adapt_true_negative = 0
    top_rows: list[Mapping[str, Any]] = []
    adaptation_errors: list[float] = []
    adaptation_predictions: list[float] = []
    adaptation_labels: list[float] = []
    operation_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    for query_id, query_rows in sorted(grouped.items()):
        ranked = sorted(
            query_rows,
            key=lambda row: (-float(row["score"]), str(row["candidate_id"])),
        )
        relevance = [float(row["relevance_grade"]) for row in ranked]
        scores = [float(row["score"]) for row in ranked]
        ideal = sorted(relevance, reverse=True)
        maximum_grade = max(relevance, default=0.0)
        cohort = _query_cohort(maximum_grade)
        metrics: dict[str, float | str] = {
            "query_id": query_id,
            "cohort": cohort,
            "max_relevance_grade": maximum_grade,
        }
        first_positive = next(
            (index for index, value in enumerate(relevance, 1) if value >= 2.0),
            None,
        )
        metrics["mrr"] = 1.0 / first_positive if first_positive else 0.0
        metrics["map"] = _average_precision(relevance)
        metrics["pairwise_accuracy"] = _pairwise_accuracy(relevance, scores)
        positive_total = sum(value >= 2.0 for value in relevance)
        reference_total = sum(value >= 1.0 for value in relevance)
        reuse_total = sum(value >= 3.0 for value in relevance)
        for k in cutoffs:
            selected = relevance[:k]
            ideal_dcg = _dcg(ideal, k)
            hits = sum(value >= 2.0 for value in selected)
            metrics[f"ndcg@{k}"] = _dcg(selected, k) / ideal_dcg if ideal_dcg else 0.0
            metrics[f"precision@{k}"] = hits / min(k, len(relevance)) if relevance else 0.0
            metrics[f"recall@{k}"] = hits / positive_total if positive_total else 0.0
            metrics[f"hitrate@{k}"] = float(hits > 0)
            reference_hits = sum(value >= 1.0 for value in selected)
            reuse_hits = sum(value >= 3.0 for value in selected)
            metrics[f"reference_precision@{k}"] = (
                reference_hits / min(k, len(relevance)) if relevance else 0.0
            )
            metrics[f"reference_recall@{k}"] = (
                reference_hits / reference_total if reference_total else 0.0
            )
            metrics[f"reference_hitrate@{k}"] = float(reference_hits > 0)
            metrics[f"reuse_precision@{k}"] = (
                reuse_hits / min(k, len(relevance)) if relevance else 0.0
            )
            metrics[f"reuse_recall@{k}"] = (
                reuse_hits / reuse_total if reuse_total else 0.0
            )
            metrics[f"reuse_hitrate@{k}"] = float(reuse_hits > 0)
        top = ranked[0]
        predicted_tier = _predicted_use_tier(top)
        reuse_adapt_available = maximum_grade >= 2.0
        reuse_adapt_claimed = predicted_tier in {
            "direct_reuse",
            "controlled_adaptation",
        }
        metrics["reuse_adapt_available"] = float(reuse_adapt_available)
        metrics["reuse_adapt_claimed"] = float(reuse_adapt_claimed)
        metrics["correct_reuse_adapt_abstention"] = float(
            not reuse_adapt_available and not reuse_adapt_claimed
        )
        if reuse_adapt_available and reuse_adapt_claimed:
            reuse_adapt_true_positive += 1
        elif not reuse_adapt_available and reuse_adapt_claimed:
            reuse_adapt_false_positive += 1
        elif reuse_adapt_available:
            reuse_adapt_false_negative += 1
        else:
            reuse_adapt_true_negative += 1
        top_rows.append(top)
        if _unsafe_use_claim(top):
            unsafe_top1 += 1
        if bool(top.get("reuse_decision")):
            reuse_decisions += 1
            if float(top["relevance_grade"]) < 3.0:
                false_reuse += 1
            if not bool(top.get("hard_gate_passed", True)):
                hard_violations += 1
        if bool(top.get("adaptation_decision")):
            adaptation_decisions += 1
            if float(top["relevance_grade"]) < 2.0:
                false_adaptation += 1
        if (
            _predicted_use_tier(top) == "reject"
            and max(relevance, default=0.0) >= 2.0
        ):
            false_rejections += 1
        per_query.append(metrics)
        for row in ranked:
            if row.get("predicted_adaptation_cost") is not None and row.get(
                "adaptation_cost_label"
            ) is not None:
                adaptation_errors.append(
                    abs(
                        float(row["predicted_adaptation_cost"])
                        - float(row["adaptation_cost_label"])
                    )
                )
                adaptation_predictions.append(
                    float(row["predicted_adaptation_cost"])
                )
                adaptation_labels.append(float(row["adaptation_cost_label"]))
            operation_labels = row.get("adaptation_operation_labels")
            if operation_labels is not None:
                expected = {str(value) for value in operation_labels}
                predicted = {
                    str(value)
                    for value in row.get("predicted_adaptation_operations") or []
                }
                for operation in expected | predicted:
                    if operation in expected and operation in predicted:
                        operation_counts[operation]["tp"] += 1
                    elif operation in predicted:
                        operation_counts[operation]["fp"] += 1
                    else:
                        operation_counts[operation]["fn"] += 1
            if row.get("calibrated_probability") is None:
                continue
            probabilities.append(float(row["calibrated_probability"]))
            binary_labels.append(float(float(row["relevance_grade"]) >= 2.0))

    aggregate = (
        {
            key: value
            for key, value in _mean_query_metrics(per_query).items()
            if key != "query_count"
        }
        if per_query
        else {}
    )
    operation_f1: dict[str, float] = {}
    total_tp = total_fp = total_fn = 0
    for operation, counts in sorted(operation_counts.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        denominator = 2 * tp + fp + fn
        operation_f1[operation] = 2 * tp / denominator if denominator else 0.0
        total_tp += tp
        total_fp += fp
        total_fn += fn
    micro_denominator = 2 * total_tp + total_fp + total_fn
    aggregate.update(
        {
            "query_count": len(per_query),
            "pair_count": len(all_rows),
            "reuse_decision_count": reuse_decisions,
            "false_reuse_rate": false_reuse / reuse_decisions if reuse_decisions else 0.0,
            "hard_constraint_violation_rate": (
                hard_violations / reuse_decisions if reuse_decisions else 0.0
            ),
            "adaptation_decision_count": adaptation_decisions,
            "false_adaptation_rate": (
                false_adaptation / adaptation_decisions
                if adaptation_decisions
                else 0.0
            ),
            "false_rejection_rate": (
                false_rejections / len(per_query) if per_query else 0.0
            ),
            "unsafe_top1_rate": unsafe_top1 / len(per_query) if per_query else 0.0,
            "adaptation_cost_mae": mean(adaptation_errors) if adaptation_errors else None,
            "adaptation_cost_spearman": _spearman(
                adaptation_predictions, adaptation_labels
            ),
            "adaptation_operation_micro_f1": (
                2 * total_tp / micro_denominator
                if micro_denominator
                else None
            ),
            "adaptation_operation_macro_f1": (
                mean(operation_f1.values()) if operation_f1 else None
            ),
            "adaptation_operation_f1_by_type": operation_f1,
        }
    )
    raw_decision_quality = _decision_quality(
        all_rows, prediction_key="raw_decision_class"
    )
    gated_decision_quality = _decision_quality(
        all_rows, prediction_key="decision_class"
    )
    for prefix, quality in (
        ("raw_decision", raw_decision_quality),
        ("gated_decision", gated_decision_quality),
    ):
        aggregate[f"{prefix}_macro_f1"] = (
            quality["macro_f1"] if quality is not None else None
        )
        aggregate[f"{prefix}_accuracy"] = (
            quality["accuracy"] if quality is not None else None
        )
        aggregate[f"{prefix}_f1_by_class"] = (
            quality["f1_by_class"] if quality is not None else {}
        )
        aggregate[f"{prefix}_confusion_matrix"] = (
            quality["confusion_matrix"] if quality is not None else None
        )
        aggregate[f"{prefix}_present_classes"] = (
            quality["present_classes"] if quality is not None else []
        )
    if probabilities:
        aggregate["brier_score"] = mean(
            (probability - label) ** 2
            for probability, label in zip(probabilities, binary_labels, strict=True)
        )
        aggregate["ece_equal_mass_10"] = _ece(probabilities, binary_labels)
        aggregate["negative_log_likelihood"] = mean(
            -(
                label * math.log(min(1.0 - 1e-7, max(1e-7, probability)))
                + (1.0 - label)
                * math.log(min(1.0 - 1e-7, max(1e-7, 1.0 - probability)))
            )
            for probability, label in zip(probabilities, binary_labels, strict=True)
        )
    else:
        aggregate["brier_score"] = None
        aggregate["ece_equal_mass_10"] = None
        aggregate["negative_log_likelihood"] = None
    aggregate.update(_selective_risk(top_rows))
    reuse_adapt_precision_denominator = (
        reuse_adapt_true_positive + reuse_adapt_false_positive
    )
    reuse_adapt_recall_denominator = (
        reuse_adapt_true_positive + reuse_adapt_false_negative
    )
    reuse_adapt_precision = (
        reuse_adapt_true_positive / reuse_adapt_precision_denominator
        if reuse_adapt_precision_denominator
        else 0.0
    )
    reuse_adapt_recall = (
        reuse_adapt_true_positive / reuse_adapt_recall_denominator
        if reuse_adapt_recall_denominator
        else 0.0
    )
    aggregate.update(
        {
            "reuse_adapt_availability_precision": reuse_adapt_precision,
            "reuse_adapt_availability_recall": reuse_adapt_recall,
            "reuse_adapt_availability_f1": (
                2.0
                * reuse_adapt_precision
                * reuse_adapt_recall
                / (reuse_adapt_precision + reuse_adapt_recall)
                if reuse_adapt_precision + reuse_adapt_recall
                else 0.0
            ),
            "reuse_adapt_abstention_specificity": (
                reuse_adapt_true_negative
                / (reuse_adapt_true_negative + reuse_adapt_false_positive)
                if reuse_adapt_true_negative + reuse_adapt_false_positive
                else 0.0
            ),
            "reuse_adapt_availability_confusion": {
                "true_positive": reuse_adapt_true_positive,
                "false_positive": reuse_adapt_false_positive,
                "false_negative": reuse_adapt_false_negative,
                "true_negative": reuse_adapt_true_negative,
            },
        }
    )
    cohort_aggregate = {
        cohort: _mean_query_metrics(
            [row for row in per_query if row["cohort"] == cohort]
        )
        for cohort in (
            "direct_reuse_available",
            "controlled_adaptation_available",
            "mechanistic_reference_only",
            "no_supported_candidate",
        )
        if any(row["cohort"] == cohort for row in per_query)
    }
    positive_present = [
        row for row in per_query if float(row["max_relevance_grade"]) >= 2.0
    ]
    return {
        "metrics_version": METRICS_VERSION,
        "aggregate": aggregate,
        "cohort_aggregate": cohort_aggregate,
        "positive_candidate_query_aggregate": _mean_query_metrics(
            positive_present
        ),
        "per_query": per_query,
    }


def evaluate_metadata_rankings(
    rows: Iterable[Mapping[str, Any]], *, positive_threshold: float = 2.0
) -> dict[str, Any]:
    """Evaluate documented metadata compatibility, never scientific usability.

    Unknown labels are excluded from a query's *condensed* judged ranking. Its
    top-k judgment coverage is also reported against the original ranking, so
    filtering cannot silently present a partially judged pool as fully labeled.
    Queries without a judged positive have undefined NDCG/MRR/AP/recall; these
    values remain null and each aggregate metric reports its own query support.
    The labels can be weak supervision: fitting them does not independently
    establish model quality, successful execution, reuse, or adaptation.
    """
    if not math.isfinite(positive_threshold) or positive_threshold <= 0:
        raise ValueError("Metadata positive_threshold must be finite and positive.")
    grouped: dict[str, list[tuple[Mapping[str, Any], float | None]]] = defaultdict(list)
    pair_ids: set[tuple[str, str]] = set()
    for row in rows:
        query_id, candidate_id = str(row["query_id"]), str(row["candidate_id"])
        key = (query_id, candidate_id)
        if key in pair_ids:
            raise ValueError(f"Duplicate metadata ranking pair: {key}")
        pair_ids.add(key)
        if not math.isfinite(float(row["score"])):
            raise ValueError(f"Non-finite metadata ranking score: {key}")
        value = next(
            (row[field] for field in (
                "metadata_relevance", "metadata_relevance_grade", "relevance_grade"
            ) if field in row),
            None,
        )
        known = (
            row.get("label_known", True) is not False
            and row.get("compatibility_label_known", True) is not False
            and value is not None
        )
        label = float(value) if known else None
        if label is not None and (not math.isfinite(label) or not 0 <= label <= 3):
            raise ValueError(f"Metadata relevance must be a finite grade in [0, 3]: {key}")
        grouped[query_id].append((row, label))

    cutoffs = (1, 3, 5, 10)
    per_query: list[dict[str, Any]] = []
    metric_names = {"mrr", "map", "pairwise_accuracy"}
    metric_names.update(
        f"{metric}@{k}"
        for k in cutoffs
        for metric in ("ndcg", "precision", "recall", "hitrate", "judged_fraction")
    )
    for query_id, values in sorted(grouped.items()):
        ranked = sorted(values, key=lambda item: (-float(item[0]["score"]), str(item[0]["candidate_id"])))
        judged = [(row, label) for row, label in ranked if label is not None]
        relevance = [float(label) for _, label in judged]
        scores = [float(row["score"]) for row, _ in judged]
        ideal = sorted(relevance, reverse=True)
        positives = [grade >= positive_threshold for grade in relevance]
        positive_total = sum(positives)
        first_positive = next((index for index, positive in enumerate(positives, 1) if positive), None)
        # Four-grade labels need only a few counters, avoiding a quadratic
        # candidate-pair loop on every validation pass of the full catalogue.
        counts = Counter(relevance)
        pairwise_count = (len(judged) ** 2 - sum(count ** 2 for count in counts.values())) // 2
        pairwise_correct = 0.0
        previously_ranked: Counter[float] = Counter()
        cursor = 0
        while cursor < len(judged):
            end = cursor + 1
            while end < len(judged) and scores[end] == scores[cursor]:
                end += 1
            tied = Counter(relevance[cursor:end])
            for grade, count in tied.items():
                pairwise_correct += count * sum(previous_count for previous_grade, previous_count in previously_ranked.items() if previous_grade > grade)
            tied_comparable = ((end - cursor) ** 2 - sum(count ** 2 for count in tied.values())) // 2
            pairwise_correct += 0.5 * tied_comparable
            previously_ranked.update(tied)
            cursor = end
        query_metrics: dict[str, Any] = {
            "query_id": query_id,
            "candidate_count": len(ranked),
            "judged_candidate_count": len(judged),
            "unknown_candidate_count": len(ranked) - len(judged),
            "positive_candidate_count": positive_total,
            "comparable_pair_count": pairwise_count,
            "mrr": 1.0 / first_positive if first_positive else None,
            "map": (
                sum(sum(positives[:index]) / index for index, positive in enumerate(positives, 1) if positive)
                / positive_total if positive_total else None
            ),
            "pairwise_accuracy": pairwise_correct / pairwise_count if pairwise_count else None,
        }
        for k in cutoffs:
            selected = relevance[:k]
            hits = sum(grade >= positive_threshold for grade in selected)
            ideal_dcg = _dcg(ideal, k)
            query_metrics.update({
                f"ndcg@{k}": _dcg(selected, k) / ideal_dcg if ideal_dcg else None,
                f"precision@{k}": hits / len(selected) if selected else None,
                f"recall@{k}": hits / positive_total if positive_total else None,
                f"hitrate@{k}": float(hits > 0) if positive_total else None,
                f"judged_fraction@{k}": sum(label is not None for _, label in ranked[:k]) / min(k, len(ranked)),
            })
        per_query.append(query_metrics)

    metric_query_support = {
        metric: sum(row[metric] is not None for row in per_query)
        for metric in sorted(metric_names)
    }
    aggregate = {
        metric: mean(float(row[metric]) for row in per_query if row[metric] is not None)
        if support else None
        for metric, support in metric_query_support.items()
    }
    aggregate.update({
        "query_count": len(per_query),
        "pair_count": len(pair_ids),
        "judged_pair_count": sum(row["judged_candidate_count"] for row in per_query),
        "unknown_pair_count": sum(row["unknown_candidate_count"] for row in per_query),
        "positive_candidate_query_count": sum(row["positive_candidate_count"] > 0 for row in per_query),
        "unjudged_query_count": sum(row["judged_candidate_count"] == 0 for row in per_query),
        "metric_query_support": metric_query_support,
    })
    return {
        "metrics_version": METADATA_METRICS_VERSION,
        "metrics_scope": METADATA_RELEVANCE_SEMANTICS,
        "relevance_label_semantics": METADATA_RELEVANCE_SEMANTICS,
        "scientific_claim_eligible": False,
        "positive_threshold": positive_threshold,
        "label_interpretation": {
            "0": "no_documented_relevance_not_validated_unsuitability",
            "1": "documented_cross_compound_mechanism_overlap",
            "2": "same_compound_partial_or_unspecified_scope",
            "3": "same_compound_documented_requirement_coverage",
        },
        "unknown_label_policy": "exclude_from_condensed_judged_ranking_and_report_top_k_judgment_coverage",
        "aggregation_policy": "macro_mean_over_defined_query_metrics_with_explicit_support",
        "interpretation": "Weak-supervision fit and grouped generalization of documented compatibility; no independently validated model-use or scientific-quality claim.",
        "aggregate": aggregate,
        "per_query": per_query,
    }
