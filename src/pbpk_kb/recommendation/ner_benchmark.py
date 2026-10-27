from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.recommendation.entity_extraction import (
    LABEL_TO_NODE_TYPE,
    GlinerNERBackend,
)


DEFAULT_MODELS = (
    "Ihor/gliner-biomed-small-v1.0",
    "Ihor/gliner-biomed-base-v1.0",
    "Ihor/gliner-biomed-large-v1.0",
)
ACCEPTANCE_THRESHOLDS = {
    "expected_count": 100,
    "exact_recall": 0.80,
    "overlap_type_recall": 0.85,
    "compound_exact_recall": 0.90,
    "pk_parameter_exact_recall": 0.65,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GLiNER-biomed models on a locked PBPK silver set.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--per-type", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    rows = list(_iter_jsonl(args.input))
    examples = build_silver_examples(rows, per_type=args.per_type)
    report = {
        "benchmark_type": "locked_structured-field silver benchmark",
        "limitations": [
            "Only exact structured fields present verbatim in source_text are labeled.",
            "Unlabeled model predictions are not counted as false positives.",
            "This benchmark selects a model for PBPK field recall; it is not a gold NER accuracy claim.",
        ],
        "input_path": str(args.input),
        "input_sha256": _sha256(args.input),
        "example_count": len(examples),
        "expected_mention_count": sum(len(example["expected"]) for example in examples),
        "models": [],
    }
    for model_name in args.models:
        backend = GlinerNERBackend(
            model_name,
            threshold=args.threshold,
            device=args.device,
        )
        started = time.perf_counter()
        metrics = evaluate_model(backend, examples)
        metrics["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        metrics["model_name"] = model_name
        report["models"].append(metrics)
        del backend
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
    selected = max(
        report["models"],
        key=lambda item: (item["selection_score"], item["exact_recall"], -item["elapsed_seconds"]),
    )
    report["selected_model"] = selected["model_name"]
    report["acceptance_thresholds"] = ACCEPTANCE_THRESHOLDS
    report["acceptance_checks"] = evaluate_acceptance(selected)
    report["accepted"] = all(report["acceptance_checks"].values())
    report["threshold_registration"] = (
        "post_hoc hardening before the formal Gate 1 run; fixed for all subsequent runs"
    )
    _atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def build_silver_examples(
    rows: list[dict[str, Any]],
    *,
    per_type: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("evidence_type") or "unknown")].append(row)
    selected = []
    for evidence_type in sorted(grouped):
        ordered = sorted(
            grouped[evidence_type],
            key=lambda row: hashlib.sha256(str(row.get("evidence_id")).encode()).hexdigest(),
        )
        selected.extend(ordered[:per_type])

    examples = []
    for row in selected:
        text = str(row.get("source_text") or "")
        expected = []
        for entity_type, value in (
            ("Compound", row.get("compound_name") or row.get("attributed_compound_name")),
            ("PKParameter", row.get("parameter")),
            ("Unit", row.get("canonical_unit") or row.get("unit")),
            ("AdministrationRoute", row.get("route")),
        ):
            mention = str(value or "").strip()
            if not mention:
                continue
            for start, end in _find_spans(text, mention):
                expected.append(
                    {
                        "start": start,
                        "end": end,
                        "entity_type": entity_type,
                        "text": text[start:end],
                    }
                )
        if expected:
            examples.append(
                {
                    "evidence_id": str(row["evidence_id"]),
                    "evidence_type": row.get("evidence_type"),
                    "text": text,
                    "expected": expected,
                }
            )
    return examples


def evaluate_model(backend: GlinerNERBackend, examples: list[dict[str, Any]]) -> dict[str, Any]:
    expected_total = 0
    exact = 0
    overlap_type = 0
    type_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"expected": 0, "exact": 0, "overlap": 0})
    for example in examples:
        predictions = []
        for item in backend.predict(example["text"]):
            entity_type = LABEL_TO_NODE_TYPE.get(str(item.get("label") or ""))
            if not entity_type:
                continue
            predictions.append(
                {
                    "start": int(item.get("start", -1)),
                    "end": int(item.get("end", -1)),
                    "entity_type": entity_type,
                }
            )
        for expected in example["expected"]:
            expected_total += 1
            type_counts[expected["entity_type"]]["expected"] += 1
            exact_match = any(
                prediction["start"] == expected["start"]
                and prediction["end"] == expected["end"]
                and prediction["entity_type"] == expected["entity_type"]
                for prediction in predictions
            )
            overlap_match = any(
                prediction["entity_type"] == expected["entity_type"]
                and min(prediction["end"], expected["end"]) > max(prediction["start"], expected["start"])
                for prediction in predictions
            )
            if exact_match:
                exact += 1
                type_counts[expected["entity_type"]]["exact"] += 1
            if overlap_match:
                overlap_type += 1
                type_counts[expected["entity_type"]]["overlap"] += 1
    exact_recall = exact / expected_total if expected_total else 0.0
    overlap_recall = overlap_type / expected_total if expected_total else 0.0
    return {
        "expected_count": expected_total,
        "exact_match_count": exact,
        "overlap_type_match_count": overlap_type,
        "exact_recall": round(exact_recall, 6),
        "overlap_type_recall": round(overlap_recall, 6),
        "selection_score": round((0.7 * exact_recall) + (0.3 * overlap_recall), 6),
        "per_type": {
            key: {
                **value,
                "exact_recall": round(value["exact"] / value["expected"], 6) if value["expected"] else 0.0,
                "overlap_recall": round(value["overlap"] / value["expected"], 6) if value["expected"] else 0.0,
            }
            for key, value in sorted(type_counts.items())
        },
    }


def evaluate_acceptance(metrics: dict[str, Any]) -> dict[str, bool]:
    per_type = metrics.get("per_type") or {}
    compound = per_type.get("Compound") or {}
    pk_parameter = per_type.get("PKParameter") or {}
    return {
        "expected_count": int(metrics.get("expected_count") or 0)
        >= ACCEPTANCE_THRESHOLDS["expected_count"],
        "exact_recall": float(metrics.get("exact_recall") or 0.0)
        >= ACCEPTANCE_THRESHOLDS["exact_recall"],
        "overlap_type_recall": float(metrics.get("overlap_type_recall") or 0.0)
        >= ACCEPTANCE_THRESHOLDS["overlap_type_recall"],
        "compound_exact_recall": float(compound.get("exact_recall") or 0.0)
        >= ACCEPTANCE_THRESHOLDS["compound_exact_recall"],
        "pk_parameter_exact_recall": float(pk_parameter.get("exact_recall") or 0.0)
        >= ACCEPTANCE_THRESHOLDS["pk_parameter_exact_recall"],
    }


def _find_spans(text: str, mention: str) -> Iterable[tuple[int, int]]:
    import re

    pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(mention) + r"(?![A-Za-z0-9])", re.IGNORECASE)
    for match in pattern.finditer(text):
        yield match.start(), match.end()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


if __name__ == "__main__":
    main()
