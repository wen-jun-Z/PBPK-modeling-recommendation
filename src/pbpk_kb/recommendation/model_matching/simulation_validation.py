from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


SIMULATION_EVALUATION_VERSION = "pbpk-recommendation-simulation-evaluation-1.0.0"


def _geometric(values: Iterable[float]) -> float | None:
    selected = [value for value in values if value > 0]
    return math.exp(mean(math.log(value) for value in selected)) if selected else None


def evaluate_simulation_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    for row in values:
        for field in ("observed", "predicted"):
            if row.get(field) is None or not math.isfinite(float(row[field])):
                raise ValueError(f"Simulation {field} must be present and finite")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in values:
        grouped[str(row.get("endpoint") or "unknown")].append(row)

    def metrics(selected: list[Mapping[str, Any]]) -> dict[str, Any]:
        valid = [
            row
            for row in selected
            if float(row.get("observed") or 0) > 0
            and float(row.get("predicted") or 0) > 0
        ]
        ratios = [float(row["predicted"]) / float(row["observed"]) for row in valid]
        fold_errors = [max(ratio, 1.0 / ratio) for ratio in ratios]
        return {
            "row_count": len(selected),
            "positive_pair_count": len(valid),
            "geometric_mean_fold_error": _geometric(fold_errors),
            "geometric_bias": _geometric(ratios),
            "fraction_within_two_fold": (
                sum(error <= 2.0 for error in fold_errors) / len(fold_errors)
                if fold_errors
                else None
            ),
            "mean_absolute_log_error": (
                mean(abs(math.log(ratio)) for ratio in ratios) if ratios else None
            ),
        }

    hard_failures = [
        row
        for row in values
        if row.get("compile_success") is False
        or row.get("execution_success") is False
        or row.get("mass_balance_passed") is False
        or row.get("dimensional_check_passed") is False
        or row.get("reproducibility_passed") is False
    ]
    return {
        "simulation_evaluation_version": SIMULATION_EVALUATION_VERSION,
        "aggregate": metrics(values),
        "by_endpoint": {
            endpoint: metrics(selected)
            for endpoint, selected in sorted(grouped.items())
        },
        "compile_success_rate": mean(
            float(bool(row.get("compile_success"))) for row in values
        ) if values else None,
        "execution_success_rate": mean(
            float(bool(row.get("execution_success"))) for row in values
        ) if values else None,
        "scientific_gate_failure_count": len(hard_failures),
        "unknown_check_counts": {
            key: sum(row.get(key) is None for row in values)
            for key in ("compile_success", "execution_success", "mass_balance_passed", "dimensional_check_passed", "reproducibility_passed")
        },
        "unknown_checks_are_passed": False,
        "scientific_gate_failure_run_ids": sorted(
            {str(row.get("run_id")) for row in hard_failures if row.get("run_id")}
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_validated_artifacts(
    *,
    audit_path: Path,
    dataset_metrics_path: Path,
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_checksum = str(audit.get("verification_dataset_metrics_sha256") or "")
    actual_checksum = _sha256(dataset_metrics_path)
    if not expected_checksum or actual_checksum != expected_checksum:
        raise ValueError("Validation dataset metrics checksum does not match the audit.")
    datasets = [
        json.loads(line)
        for line in dataset_metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    engineering = audit.get("engineering_checks") or {}
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        observed = dataset.get("positive_observed_mol_per_litre") or []
        predicted = dataset.get("positive_predicted_mol_per_litre") or []
        if len(observed) != len(predicted):
            raise ValueError(
                f"Observed/predicted length mismatch for {dataset.get('dataset_name')}."
            )
        for index, (observed_value, predicted_value) in enumerate(
            zip(observed, predicted, strict=True)
        ):
            rows.append(
                {
                    "run_id": str(audit.get("protocol_id") or "unknown"),
                    "dataset_name": dataset.get("dataset_name"),
                    "observation_index": index,
                    "endpoint": "plasma_concentration_time",
                    "observed": observed_value,
                    "predicted": predicted_value,
                    "compile_success": engineering.get("all_artifacts_verified"),
                    "execution_success": engineering.get("all_simulations_passed"),
                    "mass_balance_passed": None,
                    "dimensional_check_passed": dataset.get("dataset_guard_passed"),
                    "reproducibility_passed": engineering.get(
                        "csv_repeatability_passed"
                    ),
                    "source_location": dataset.get("source_location"),
                    "reference": dataset.get("reference"),
                    "prediction_csv": dataset.get("prediction_csv"),
                }
            )
    result = evaluate_simulation_rows(rows)
    result.update(
        {
            "evaluation_scope": "single_audited_protocol_supplied_by_caller",
            "recommendation_quality_claim_eligible": False,
            "claim_blocker": "SINGLE_PROTOCOL_NOT_MULTI_MODEL_RECOMMENDATION_VALIDATION",
            "validation_grade": audit.get("validation_grade"),
            "ranking_label": audit.get("ranking_label"),
            "dataset_count": len(datasets),
            "audit_path": str(audit_path),
            "audit_sha256": _sha256(audit_path),
            "dataset_metrics_path": str(dataset_metrics_path),
            "dataset_metrics_sha256": actual_checksum,
            "unavailable_checks": ["mass_balance_passed"],
        }
    )
    return result


def _atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate real PBPK simulation outputs for recommendation validation."
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--dataset-metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.audit is not None or args.dataset_metrics is not None:
        if args.audit is None or args.dataset_metrics is None:
            parser.error("--audit and --dataset-metrics must be supplied together.")
        result = evaluate_validated_artifacts(
            audit_path=args.audit,
            dataset_metrics_path=args.dataset_metrics,
        )
    else:
        if args.input is None:
            parser.error("--input or both validated-artifact arguments are required.")
        rows = [
            json.loads(line)
            for line in args.input.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        result = evaluate_simulation_rows(rows)
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
