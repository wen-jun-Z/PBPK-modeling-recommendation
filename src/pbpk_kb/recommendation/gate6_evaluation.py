from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.recommendation.graph_prompt import (
    GraphPromptCorpus,
    validate_recommendation,
)


GATE6_VERSION = "pbpk-recommendation-gate6-1.0.0"
SPLITS = ("train", "validation", "test")
TARGET_FIELDS = (
    "component_ids",
    "evidence_ids",
    "public_model_ids",
    "gap_codes",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def set_metrics(
    predicted_rows: Iterable[set[str]],
    expected_rows: Iterable[set[str]],
) -> dict[str, float | int]:
    pairs = list(zip(predicted_rows, expected_rows, strict=True))
    true_positive = sum(len(predicted & expected) for predicted, expected in pairs)
    false_positive = sum(len(predicted - expected) for predicted, expected in pairs)
    false_negative = sum(len(expected - predicted) for predicted, expected in pairs)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    nonempty = [
        (predicted, expected)
        for predicted, expected in pairs
        if expected
    ]
    empty = [
        (predicted, expected)
        for predicted, expected in pairs
        if not expected
    ]
    macro_nonempty = []
    for predicted, expected in nonempty:
        tp = len(predicted & expected)
        row_precision = tp / max(1, len(predicted))
        row_recall = tp / len(expected)
        macro_nonempty.append(
            2 * row_precision * row_recall / (row_precision + row_recall)
            if row_precision + row_recall
            else 0.0
        )
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_set_accuracy": sum(
            predicted == expected for predicted, expected in pairs
        )
        / max(1, len(pairs)),
        "nonempty_target_count": len(nonempty),
        "nonempty_macro_f1": sum(macro_nonempty) / max(1, len(macro_nonempty)),
        "empty_target_count": len(empty),
        "empty_target_specificity": sum(
            not predicted for predicted, _ in empty
        )
        / max(1, len(empty)),
    }


def evaluate_variant_rows(
    *,
    prompt_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    example_ids: set[str],
) -> dict[str, Any]:
    prompt_by_id = {
        str(row["example_id"]): row
        for row in prompt_rows
        if str(row["example_id"]) in example_ids
    }
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        if str(row["example_id"]) in prompt_by_id:
            by_variant[str(row["variant"])].append(row)
    reports: dict[str, Any] = {}
    for variant in ("no_kg", "text_graph", "soft_graph", "hybrid"):
        rows = sorted(
            by_variant.get(variant, []),
            key=lambda row: str(row["example_id"]),
        )
        if len(rows) != len(prompt_by_id):
            reports[variant] = {
                "status": "incomplete_outputs",
                "expected_count": len(prompt_by_id),
                "actual_count": len(rows),
            }
            continue
        report: dict[str, Any] = {
            "status": "evaluated",
            "example_count": len(rows),
            "schema_valid_rate": sum(bool(row["schema_valid"]) for row in rows)
            / max(1, len(rows)),
            "mean_latency_seconds": sum(
                float((row.get("generation") or {}).get("elapsed_seconds") or 0.0)
                for row in rows
            )
            / max(1, len(rows)),
            "mean_input_tokens": sum(
                int((row.get("generation") or {}).get("input_token_count") or 0)
                for row in rows
            )
            / max(1, len(rows)),
            "mean_output_tokens": sum(
                int((row.get("generation") or {}).get("output_token_count") or 0)
                for row in rows
            )
            / max(1, len(rows)),
        }
        for field in TARGET_FIELDS:
            predicted = []
            expected = []
            for row in rows:
                selection = row.get("selection") if row.get("schema_valid") else None
                predicted.append(
                    set(map(str, (selection or {}).get(field) or ()))
                )
                expected.append(
                    set(
                        map(
                            str,
                            prompt_by_id[str(row["example_id"])]["target"].get(
                                field
                            )
                            or (),
                        )
                    )
                )
            report[field] = set_metrics(predicted, expected)
        reports[variant] = report
    return reports


def build_split_manifest(
    *,
    axis: str,
    silver_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def resources(row: dict[str, Any]) -> list[str]:
        if axis == "compound":
            return [str(row["compound_id"])]
        if axis == "source_document":
            return sorted(map(str, row.get("source_document_ids") or ()))
        if axis == "public_model":
            return sorted(map(str, row.get("public_model_checksums") or ()))
        raise ValueError(f"Unsupported split axis: {axis}")

    owners: dict[str, set[str]] = defaultdict(set)
    examples_by_split: dict[str, list[str]] = {split: [] for split in SPLITS}
    compounds_by_split: dict[str, set[str]] = {
        split: set() for split in SPLITS
    }
    ineligible: list[dict[str, str]] = []
    for row in silver_rows:
        split = str(row["split"])
        row_resources = resources(row)
        if not row_resources:
            ineligible.append(
                {
                    "example_id": str(row["example_id"]),
                    "reason": f"missing_{axis}_resource",
                }
            )
            continue
        examples_by_split[split].append(str(row["example_id"]))
        compounds_by_split[split].add(str(row["compound_id"]))
        for resource in row_resources:
            owners[resource].add(split)
    leakage = {
        resource: sorted(splits)
        for resource, splits in owners.items()
        if len(splits) > 1
    }
    test_ids = set(examples_by_split["test"])
    target_field = (
        "public_model_ids" if axis == "public_model" else "evidence_ids"
    )
    test_positive_count = sum(
        bool((row.get("target") or {}).get(target_field))
        for row in silver_rows
        if str(row["example_id"]) in test_ids
    )
    eligible = (
        not leakage
        and len(compounds_by_split["test"]) >= 2
        and test_positive_count >= 1
    )
    return {
        "axis": axis,
        "status": "eligible" if eligible else "insufficient_silver",
        "split_method": (
            "locked Gate 5 ownership assignment; no resource may cross splits"
        ),
        "example_ids": {
            split: sorted(values)
            for split, values in examples_by_split.items()
        },
        "example_counts": {
            split: len(values)
            for split, values in examples_by_split.items()
        },
        "compound_counts": {
            split: len(values)
            for split, values in compounds_by_split.items()
        },
        "resource_count": len(owners),
        "resource_cross_split_count": len(leakage),
        "resource_cross_split_examples": dict(
            list(sorted(leakage.items()))[:100]
        ),
        "test_positive_target_count": test_positive_count,
        "ineligible_examples": ineligible,
        "eligibility_rule": (
            "zero leakage, at least two independent test compounds, "
            "and at least one non-empty test target"
        ),
    }


def build_time_manifest(
    silver_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ineligible = [
        {
            "example_id": str(row["example_id"]),
            "compound_id": str(row["compound_id"]),
            "source_document_ids": list(row.get("source_document_ids") or ()),
            "reason": "no_verified_publication_or_release_date_in_locked_dataset",
        }
        for row in silver_rows
    ]
    return (
        {
            "axis": "time",
            "status": "time_split_ineligible",
            "eligible_example_count": 0,
            "ineligible_example_count": len(ineligible),
            "download_date_used": False,
            "reason": (
                "The locked KG does not carry a verified publication, label "
                "release, or public-model version date for these examples."
            ),
            "example_ids": {split: [] for split in SPLITS},
            "resource_cross_split_count": 0,
        },
        ineligible,
    )


def retrieval_report(prompt_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in prompt_rows if row["split"] == "test"]
    recalls = []
    hits = []
    full_coverage = []
    for row in rows:
        retrieved = set(map(str, row.get("retrieved_subgraph_ids") or ()))
        positive = set(map(str, row.get("positive_subgraph_ids") or ()))
        recalls.append(len(retrieved & positive) / max(1, len(positive)))
        hits.append(bool(retrieved & positive))
        full_coverage.append(positive <= retrieved)
    return {
        "benchmark_level": "silver",
        "example_count": len(rows),
        "subgraph_hit_at_8": sum(hits) / max(1, len(hits)),
        "subgraph_recall_at_8": sum(recalls) / max(1, len(recalls)),
        "complete_positive_subgraph_coverage_rate": sum(full_coverage)
        / max(1, len(full_coverage)),
        "mean_retrieved_subgraphs": sum(
            len(row.get("retrieved_subgraph_ids") or ()) for row in rows
        )
        / max(1, len(rows)),
        "notes": [
            "Positive subgraphs were locked before online retrieval.",
            "Path Recall is not promoted to an acceptance metric because the "
            "silver path list contains all supporting KG paths rather than a "
            "single expected minimal path.",
        ],
    }


def _number_is_verbatim(value: float, text: str) -> bool:
    for match in re.finditer(
        r"(?<![A-Za-z0-9])[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?",
        text,
    ):
        try:
            if math.isclose(
                float(match.group(0)),
                value,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                return True
        except ValueError:
            continue
    return False


def safety_report(
    *,
    corpus: GraphPromptCorpus,
    gate5_dir: Path,
    test_example_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    emitted = 0
    citation_count = 0
    provenance_failures = 0
    unsupported_numeric = 0
    numeric_count = 0
    quarantined_leakage = 0
    ontology_invalid = 0
    for example_id in sorted(test_example_ids):
        run_dir = gate5_dir / "recommendation_runs" / example_id
        validation_path = run_dir / "recommendation_validation.json"
        final_path = run_dir / "recommendation_final.json"
        if not validation_path.exists() or not final_path.exists():
            rows.append(
                {
                    "example_id": example_id,
                    "status": "missing_run_artifacts",
                    "errors": ["missing_run_artifacts"],
                }
            )
            continue
        stored_validation = json.loads(validation_path.read_text(encoding="utf-8"))
        recommendation = json.loads(final_path.read_text(encoding="utf-8"))
        if recommendation.get("status") != "recommendation_validated":
            rows.append(
                {
                    "example_id": example_id,
                    "status": str(stored_validation.get("status")),
                    "errors": list(stored_validation.get("errors") or ()),
                }
            )
            continue
        emitted += 1
        request = json.loads(
            (run_dir / "recommendation_request.json").read_text(
                encoding="utf-8"
            )
        )
        allowed_subgraph_ids = {
            str(row["subgraph_id"])
            for row in recommendation.get("retrieved_subgraphs") or ()
        }
        independent = validate_recommendation(
            corpus=corpus,
            request=corpus_request(request),
            recommendation=recommendation,
            allowed_subgraph_ids=allowed_subgraph_ids,
        )
        errors = list(independent.get("errors") or ())
        if errors:
            ontology_invalid += int(
                any(
                    error.startswith(
                        (
                            "public_model_not_reusable",
                            "parameter_compound_mismatch",
                            "retrieved_subgraph_outside_runtime_selection",
                        )
                    )
                    for error in errors
                )
            )
        for citation in recommendation.get("citations") or ():
            citation_count += 1
            evidence_id = str(citation.get("evidence_id") or "")
            try:
                reference = corpus.evidence_record(evidence_id)
            except KeyError:
                provenance_failures += 1
                continue
            if reference.get("quality_status") != "auto_validated":
                quarantined_leakage += 1
            if any(
                not citation.get(field)
                or citation.get(field) != reference.get(field)
                for field in (
                    "source_id",
                    "source_location",
                    "supporting_text",
                    "source_text_hash",
                )
            ):
                provenance_failures += 1
        for candidate in recommendation.get("parameter_candidates") or ():
            values = [
                float(candidate[key])
                for key in ("value", "value_low", "value_high")
                if candidate.get(key) is not None
            ]
            if not values:
                continue
            numeric_count += 1
            if not all(
                _number_is_verbatim(
                    value,
                    str(candidate.get("supporting_text") or ""),
                )
                for value in values
            ):
                unsupported_numeric += 1
        rows.append(
            {
                "example_id": example_id,
                "status": independent["status"],
                "errors": errors,
                "citation_provenance_completeness": independent.get(
                    "citation_provenance_completeness"
                ),
                "unsupported_numeric_parameter_rate": independent.get(
                    "unsupported_numeric_parameter_rate"
                ),
            }
        )
    report = {
        "test_recommendation_count": len(test_example_ids),
        "validated_recommendation_count": emitted,
        "validated_rate": emitted / max(1, len(test_example_ids)),
        "citation_count": citation_count,
        "citation_provenance_completeness": (
            (citation_count - provenance_failures) / max(1, citation_count)
        ),
        "provenance_failure_count": provenance_failures,
        "numeric_parameter_count": numeric_count,
        "unsupported_numeric_parameter_count": unsupported_numeric,
        "unsupported_numeric_parameter_rate": unsupported_numeric
        / max(1, numeric_count),
        "quarantined_evidence_leakage": quarantined_leakage,
        "ontology_invalid_recommendation": ontology_invalid,
        "all_formal_outputs_validated": emitted == len(test_example_ids),
    }
    report["accepted"] = (
        report["all_formal_outputs_validated"]
        and report["citation_provenance_completeness"] == 1.0
        and report["unsupported_numeric_parameter_rate"] == 0.0
        and report["quarantined_evidence_leakage"] == 0
        and report["ontology_invalid_recommendation"] == 0
    )
    return report, rows


def corpus_request(value: dict[str, Any]) -> Any:
    from pbpk_kb.recommendation.schemas import ModelingRequest

    return ModelingRequest.model_validate(value)


def run_gate6(
    *,
    dataset_dir: Path,
    gate5_dir: Path,
    graph_dir: Path,
    subgraph_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gate5_manifest_path = gate5_dir / "gate5_manifest.json"
    gate5_comparison_path = gate5_dir / "gate5_comparison.json"
    dataset_manifest_path = dataset_dir / "dataset_manifest.json"
    prompt_path = dataset_dir / "prompt_examples.jsonl"
    silver_path = dataset_dir / "recommendation_silver.jsonl"
    raw_path = gate5_dir / "raw_variant_outputs.jsonl"
    required = (
        gate5_manifest_path,
        gate5_comparison_path,
        dataset_manifest_path,
        prompt_path,
        silver_path,
        raw_path,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Gate 6 inputs are missing: {missing}")
    gate5_manifest = json.loads(gate5_manifest_path.read_text(encoding="utf-8"))
    if gate5_manifest.get("accepted") is not True:
        raise RuntimeError("Gate 6 requires an accepted Gate 5 manifest.")
    prompt_rows = load_jsonl(prompt_path)
    silver_rows = load_jsonl(silver_path)
    raw_rows = load_jsonl(raw_path)
    prompt_ids = {str(row["example_id"]) for row in prompt_rows}
    silver_ids = {str(row["example_id"]) for row in silver_rows}
    if prompt_ids != silver_ids:
        raise RuntimeError("Prompt and silver example IDs differ.")

    split_manifests = {
        axis: build_split_manifest(axis=axis, silver_rows=silver_rows)
        for axis in ("compound", "source_document", "public_model")
    }
    time_manifest, time_ineligible = build_time_manifest(silver_rows)
    split_manifests["time"] = time_manifest
    for axis, manifest in split_manifests.items():
        atomic_json(output_dir / f"{axis}_split_manifest.json", manifest)
    atomic_jsonl(output_dir / "time_split_ineligible.jsonl", time_ineligible)

    test_ids = {
        str(row["example_id"])
        for row in prompt_rows
        if row["split"] == "test"
    }
    variant_reports = evaluate_variant_rows(
        prompt_rows=prompt_rows,
        raw_rows=raw_rows,
        example_ids=test_ids,
    )
    retrieval = retrieval_report(prompt_rows)
    recommendation = {
        "benchmark_level": "silver",
        "test_example_count": len(test_ids),
        "variants": variant_reports,
        "formal_variant": "hybrid",
        "gold_claim": False,
    }
    ablation = {
        "benchmark_level": "silver",
        "same_test_requests": True,
        "variants": variant_reports,
        "hybrid_minus_no_kg_component_f1": (
            variant_reports["hybrid"]["component_ids"]["f1"]
            - variant_reports["no_kg"]["component_ids"]["f1"]
        ),
        "hybrid_minus_text_graph_component_f1": (
            variant_reports["hybrid"]["component_ids"]["f1"]
            - variant_reports["text_graph"]["component_ids"]["f1"]
        ),
    }

    corpus = GraphPromptCorpus(
        graph_dir=graph_dir,
        subgraph_dir=subgraph_dir,
    )
    safety, safety_rows = safety_report(
        corpus=corpus,
        gate5_dir=gate5_dir,
        test_example_ids=test_ids,
    )
    atomic_json(output_dir / "retrieval_evaluation.json", retrieval)
    atomic_json(output_dir / "recommendation_evaluation.json", recommendation)
    atomic_json(output_dir / "safety_evaluation.json", safety)
    atomic_jsonl(output_dir / "safety_evaluation_rows.jsonl", safety_rows)
    atomic_json(output_dir / "ablation_evaluation.json", ablation)

    generalization: dict[str, Any] = {
        "benchmark_level": "silver",
        "gold_claim": False,
        "axes": {},
    }
    for axis, split_manifest in split_manifests.items():
        axis_report: dict[str, Any] = {
            "status": split_manifest["status"],
            "resource_cross_split_count": split_manifest.get(
                "resource_cross_split_count", 0
            ),
        }
        if split_manifest["status"] == "eligible":
            axis_ids = set(split_manifest["example_ids"]["test"])
            axis_report["metrics"] = evaluate_variant_rows(
                prompt_rows=prompt_rows,
                raw_rows=raw_rows,
                example_ids=axis_ids,
            )["hybrid"]
        else:
            axis_report["metrics"] = None
        generalization["axes"][axis] = axis_report
    generalization["generalization_claim_eligible"] = all(
        row["status"] == "eligible"
        for row in generalization["axes"].values()
    )
    atomic_json(
        output_dir / "generalization_evaluation.json",
        generalization,
    )

    comparison = json.loads(
        gate5_comparison_path.read_text(encoding="utf-8")
    )
    gate5_checks = comparison.get("acceptance_checks") or {}
    report_paths = (
        "retrieval_evaluation.json",
        "recommendation_evaluation.json",
        "safety_evaluation.json",
        "ablation_evaluation.json",
        "generalization_evaluation.json",
    )
    engineering_acceptance = (
        all(bool(value) for value in gate5_checks.values())
        and safety["accepted"]
        and split_manifests["compound"]["status"] == "eligible"
        and split_manifests["source_document"]["status"] == "eligible"
        and split_manifests["public_model"]["status"] == "eligible"
        and all(
            int(manifest.get("resource_cross_split_count") or 0) == 0
            for manifest in split_manifests.values()
        )
    )
    manifest = {
        "accepted": engineering_acceptance,
        "engineering_acceptance": engineering_acceptance,
        "generalization_claim_eligible": generalization[
            "generalization_claim_eligible"
        ],
        "benchmark_level": "silver",
        "gold_claim": False,
        "clinical_validity_claim": False,
        "gate6_version": GATE6_VERSION,
        "generated_at": utc_now(),
        "input_hashes": {
            path.name: sha256_file(path) for path in required
        },
        "split_status": {
            axis: manifest["status"]
            for axis, manifest in split_manifests.items()
        },
        "gate5_acceptance_checks": gate5_checks,
        "safety_acceptance": safety["accepted"],
        "artifacts": {
            name: {"sha256": sha256_file(output_dir / name)}
            for name in report_paths
        },
        "limitations": [
            (
                "No expert gold labels were used; metrics are KG-derived "
                "silver benchmark metrics."
            ),
            (
                "Time generalization is not claim-eligible until verified "
                "publication or release dates are represented in the KG."
            ),
        ],
    }
    atomic_json(output_dir / "gate6_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run locked Gate 6 PBPK recommendation evaluation."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--gate5-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--subgraph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_gate6(
        dataset_dir=args.dataset_dir,
        gate5_dir=args.gate5_dir,
        graph_dir=args.graph_dir,
        subgraph_dir=args.subgraph_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
