from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


MODEL_VERSION = "wesin/pubmedbert-relation-extraction@3616405c9fbb882ed2a0b9cf95da3f6405e3110a"
RUNNER_VERSION = "pbpk-supervised-re-runner-1.0.0"
POSITIVE_LABELS = {
    "activates",
    "inhibits",
    "converts",
    "causes",
    "treats",
    "associated_with",
    "interacts_with",
    "located_in",
}
ACCEPTANCE_THRESHOLDS = {
    "positive_recall": 0.75,
    "hard_negative_specificity": 0.80,
    "balanced_accuracy": 0.80,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supervised PubMedBERT relation classification.")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--mentions", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--positive-threshold", type=float, default=0.50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = run_supervised_relation(
        evidence_path=args.evidence,
        mentions_path=args.mentions,
        model_path=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        positive_threshold=args.positive_threshold,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def run_supervised_relation(
    *,
    evidence_path: Path,
    mentions_path: Path,
    model_path: Path,
    output_dir: Path,
    batch_size: int,
    positive_threshold: float,
    device: str,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    evidence = {
        str(row["evidence_id"]): row
        for row in _iter_jsonl(evidence_path)
        if row.get("evidence_type") == "ddi"
    }
    mentions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mention in _iter_jsonl(mentions_path):
        evidence_id = str(mention.get("evidence_id") or "")
        if evidence_id in evidence:
            mentions[evidence_id].append(mention)
    pairs = [
        pair
        for evidence_id, row in evidence.items()
        for pair in build_supervised_pairs(row, mentions.get(evidence_id, []))
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    predictions = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        inputs = tokenizer(
            [item["marked_text"] for item in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(model(**inputs).logits, dim=-1).cpu()
        for pair, row_probs in zip(batch, probabilities):
            class_id = int(torch.argmax(row_probs).item())
            label = str(model.config.id2label[class_id])
            confidence = float(row_probs[class_id].item())
            positive_probability = float(
                sum(
                    row_probs[index].item()
                    for index, class_label in model.config.id2label.items()
                    if str(class_label) in POSITIVE_LABELS
                )
            )
            predictions.append(
                {
                    **pair,
                    "predicted_label": label,
                    "predicted_confidence": round(confidence, 8),
                    "positive_probability": round(positive_probability, 8),
                    "positive_signal": label in POSITIVE_LABELS and positive_probability >= positive_threshold,
                    "model_version": MODEL_VERSION,
                    "runner_version": RUNNER_VERSION,
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(output_dir / "supervised_relation_signals.jsonl", predictions)
    benchmark = evaluate_locked_silver(predictions)
    _atomic_json(output_dir / "supervised_relation_benchmark.json", benchmark)
    summary = {
        "accepted": benchmark["accepted"],
        "input_evidence_path": str(evidence_path),
        "input_evidence_sha256": _sha256(evidence_path),
        "input_mentions_path": str(mentions_path),
        "input_mentions_sha256": _sha256(mentions_path),
        "model_path": str(model_path),
        "model_version": MODEL_VERSION,
        "runner_version": RUNNER_VERSION,
        "pair_count": len(predictions),
        "positive_signal_count": sum(bool(row["positive_signal"]) for row in predictions),
        "predicted_label_counts": dict(sorted(Counter(row["predicted_label"] for row in predictions).items())),
        "benchmark_metrics": benchmark["metrics"],
        "benchmark_checks": benchmark["checks"],
    }
    _atomic_json(output_dir / "supervised_relation_summary.json", summary)
    return summary


def build_supervised_pairs(
    row: dict[str, Any],
    mentions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    text = str(row["source_text"])
    target_node_id = f"Compound:{row['compound_id']}"
    by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mention in mentions:
        node_id = str(mention.get("linked_node_id") or "")
        if node_id.startswith("Compound:"):
            by_node[node_id].append(mention)
    output = []
    node_ids = sorted(by_node)
    for left_index, left_id in enumerate(node_ids):
        for right_id in node_ids[left_index + 1 :]:
            left, right = _closest_nonoverlapping_mentions(by_node[left_id], by_node[right_id])
            if left is None or right is None:
                continue
            marked = mark_entity_pair(text, left, right)
            is_target_pair = target_node_id in {left_id, right_id}
            output.append(
                {
                    "pair_id": _stable_pair_id(str(row["evidence_id"]), left_id, right_id),
                    "evidence_id": str(row["evidence_id"]),
                    "source_text_hash": str(row["source_text_hash"]),
                    "entity1_node_id": left_id,
                    "entity1_text": str(left["text"]),
                    "entity1_start": int(left["start_char"]),
                    "entity1_end": int(left["end_char"]),
                    "entity2_node_id": right_id,
                    "entity2_text": str(right["text"]),
                    "entity2_start": int(right["start_char"]),
                    "entity2_end": int(right["end_char"]),
                    "marked_text": marked,
                    "target_node_id": target_node_id,
                    "target_role": str(row.get("compound_role") or ""),
                    "silver_label": "positive" if is_target_pair and len(node_ids) == 2 else (
                        "hard_negative" if not is_target_pair else "unlabeled"
                    ),
                }
            )
    return output


def mark_entity_pair(
    text: str,
    entity1: dict[str, Any],
    entity2: dict[str, Any],
) -> str:
    spans = [
        (int(entity1["start_char"]), int(entity1["end_char"]), "[E1]", "[/E1]"),
        (int(entity2["start_char"]), int(entity2["end_char"]), "[E2]", "[/E2]"),
    ]
    if any(start < 0 or end > len(text) or end <= start for start, end, _, _ in spans):
        raise ValueError("Entity marker span is outside source_text.")
    spans.sort(key=lambda item: item[0], reverse=True)
    if spans[0][0] < spans[1][1]:
        raise ValueError("Entity marker spans overlap.")
    marked = text
    for start, end, opening, closing in spans:
        marked = marked[:start] + opening + marked[start:end] + closing + marked[end:]
    return marked


def evaluate_locked_silver(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [row for row in predictions if row["silver_label"] in {"positive", "hard_negative"}]
    positives = [row for row in labeled if row["silver_label"] == "positive"]
    negatives = [row for row in labeled if row["silver_label"] == "hard_negative"]
    true_positive = sum(bool(row["positive_signal"]) for row in positives)
    true_negative = sum(not bool(row["positive_signal"]) for row in negatives)
    positive_recall = true_positive / len(positives) if positives else 0.0
    specificity = true_negative / len(negatives) if negatives else 0.0
    balanced_accuracy = (positive_recall + specificity) / 2
    metrics = {
        "positive_count": len(positives),
        "hard_negative_count": len(negatives),
        "true_positive_count": true_positive,
        "true_negative_count": true_negative,
        "positive_recall": round(positive_recall, 6),
        "hard_negative_specificity": round(specificity, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
    }
    checks = {
        key: metrics[key] >= threshold for key, threshold in ACCEPTANCE_THRESHOLDS.items()
    }
    return {
        "accepted": bool(labeled) and all(checks.values()),
        "benchmark_type": "locked_pbpk_silver_unique_target_pair_and_non_target_pair",
        "limitations": [
            "Positive labels are derived from auto-validated DDI evidence with one uniquely linked counterpart.",
            "Hard negatives are non-target compound pairs co-mentioned in the same DDI evidence.",
            "This benchmark is silver and must not be reported as gold relation-extraction accuracy.",
        ],
        "thresholds": ACCEPTANCE_THRESHOLDS,
        "metrics": metrics,
        "checks": checks,
    }


def _closest_nonoverlapping_mentions(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    choices = []
    for first in left:
        for second in right:
            first_start, first_end = int(first["start_char"]), int(first["end_char"])
            second_start, second_end = int(second["start_char"]), int(second["end_char"])
            if min(first_end, second_end) > max(first_start, second_start):
                continue
            choices.append((abs(first_start - second_start), first_start, second_start, first, second))
    if not choices:
        return None, None
    _, _, _, first, second = min(choices, key=lambda item: item[:3])
    return first, second


def _stable_pair_id(evidence_id: str, entity1: str, entity2: str) -> str:
    digest = hashlib.sha256(f"{evidence_id}\x1f{entity1}\x1f{entity2}".encode()).hexdigest()
    return f"supervised-re-pair:{digest[:24]}"


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            yield value


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
