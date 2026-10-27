from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping


STATISTICS_VERSION = "pbpk-model-matching-statistics-1.3.0"

PROPOSED_METHODS = ("cg_hhgm",)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _atomic_json(path: Path, value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
        default=str,
    ).encode("utf-8") + b"\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def bootstrap_interval(
    values: list[float],
    *,
    seed: int = 20260821,
    repetitions: int = 5000,
) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "lower_95": None, "upper_95": None, "n": 0}
    randomizer = random.Random(seed)
    estimates = sorted(
        mean(randomizer.choices(values, k=len(values)))
        for _ in range(repetitions)
    )
    lower = estimates[int(0.025 * (repetitions - 1))]
    upper = estimates[int(0.975 * (repetitions - 1))]
    return {
        "mean": mean(values),
        "standard_deviation": pstdev(values),
        "lower_95": lower,
        "upper_95": upper,
        "n": len(values),
        "bootstrap_repetitions": repetitions,
    }


def paired_permutation_pvalue(
    differences: list[float],
    *,
    seed: int = 20260821,
    repetitions: int = 10000,
) -> float | None:
    if not differences:
        return None
    observed = abs(mean(differences))
    randomizer = random.Random(seed)
    exceedances = 0
    for _ in range(repetitions):
        permuted = mean(
            value if randomizer.random() < 0.5 else -value
            for value in differences
        )
        exceedances += abs(permuted) >= observed
    return (exceedances + 1) / (repetitions + 1)


def holm_adjust(pvalues: Mapping[str, float | None]) -> dict[str, float | None]:
    available = sorted(
        ((name, value) for name, value in pvalues.items() if value is not None),
        key=lambda item: item[1],
    )
    adjusted: dict[str, float | None] = {name: None for name in pvalues}
    running = 0.0
    count = len(available)
    for index, (name, value) in enumerate(available):
        running = max(running, min(1.0, value * (count - index)))
        adjusted[name] = running
    return adjusted


def compare_metric_files(
    metric_files: Iterable[Path],
    *,
    primary_metric: str,
    reference_method: str,
) -> dict[str, Any]:
    methods: dict[str, dict[str, Any]] = {}
    for path in metric_files:
        method = path.stem.removeprefix("metrics_")
        payload = _read_json(path)
        per_query = {
            str(row["query_id"]): float(row[primary_metric])
            for row in payload.get("per_query") or []
        }
        methods[method] = {
            "path": str(path),
            "values": per_query,
            "interval": bootstrap_interval(list(per_query.values())),
        }
    if reference_method not in methods:
        raise ValueError(f"Reference method {reference_method!r} is not available.")
    reference = methods[reference_method]["values"]
    raw_pvalues: dict[str, float | None] = {}
    comparisons: dict[str, Any] = {}
    for method, payload in methods.items():
        if method == reference_method:
            continue
        shared = sorted(set(reference) & set(payload["values"]))
        differences = [payload["values"][key] - reference[key] for key in shared]
        raw_pvalues[method] = paired_permutation_pvalue(differences)
        comparisons[method] = {
            "reference": reference_method,
            "paired_query_count": len(shared),
            "mean_difference": mean(differences) if differences else None,
            "difference_95_ci": bootstrap_interval(differences),
            "permutation_pvalue": raw_pvalues[method],
        }
    adjusted = holm_adjust(raw_pvalues)
    for method, value in adjusted.items():
        comparisons[method]["holm_adjusted_pvalue"] = value
    return {
        "statistics_version": STATISTICS_VERSION,
        "primary_metric": primary_metric,
        "reference_method": reference_method,
        "methods": {
            method: {
                "metric_file": payload["path"],
                "interval": payload["interval"],
            }
            for method, payload in methods.items()
        },
        "paired_comparisons": comparisons,
    }


def _metric_values(path: Path, primary_metric: str) -> dict[str, float]:
    payload = _read_json(path)
    return {
        str(row["query_id"]): float(row[primary_metric])
        for row in payload.get("per_query") or []
    }


def _aggregate_safety(paths: list[Path]) -> dict[str, float | None]:
    keys = (
        "false_reuse_rate",
        "false_adaptation_rate",
        "false_rejection_rate",
        "hard_constraint_violation_rate",
        "unsafe_top1_rate",
        "aurc",
    )
    aggregates = [(_read_json(path).get("aggregate") or {}) for path in paths]
    return {
        key: (
            mean(float(value[key]) for value in aggregates if value.get(key) is not None)
            if any(value.get(key) is not None for value in aggregates)
            else None
        )
        for key in keys
    }


def _perturbation_intervals(paths: list[Path]) -> dict[str, Any] | None:
    if not paths:
        return None
    by_pair: dict[str, dict[str, list[float]]] = {}
    metrics = {
        "top1_stable": lambda value: float(bool(value)),
        "topk_jaccard": float,
        "kendall_tau": float,
        "reuse_decision_stable": lambda value: float(bool(value)),
    }
    for path in paths:
        payload = _read_json(path)
        for row in payload.get("details") or []:
            pair_id = f"{row['base_query_id']}->{row['perturbed_query_id']}"
            target = by_pair.setdefault(
                pair_id, {metric: [] for metric in metrics}
            )
            for metric, converter in metrics.items():
                value = row.get(metric)
                if value is not None:
                    target[metric].append(converter(value))
    per_pair = {
        pair_id: {
            metric: mean(values) if values else None
            for metric, values in values_by_metric.items()
        }
        for pair_id, values_by_metric in sorted(by_pair.items())
    }
    return {
        "seed_file_count": len(paths),
        "pair_count": len(per_pair),
        "metrics": {
            metric: bootstrap_interval(
                [
                    float(row[metric])
                    for row in per_pair.values()
                    if row.get(metric) is not None
                ]
            )
            for metric in metrics
        },
        "per_pair_seed_mean": per_pair,
        "metric_files": [str(path) for path in paths],
    }


def compare_research_suite(
    *,
    benchmark_directory: Path,
    suite_directory: Path,
    primary_metric: str = "ndcg@10",
) -> dict[str, Any]:
    methods: dict[str, dict[str, Any]] = {}
    for path in sorted(benchmark_directory.glob("metrics_*_test.json")):
        method = path.stem.removeprefix("metrics_").removesuffix("_test")
        methods[method] = {
            "role": "deterministic_baseline",
            "metric_files": [str(path)],
            "values": _metric_values(path, primary_metric),
            "safety": _aggregate_safety([path]),
            "perturbation": _perturbation_intervals(
                [
                    benchmark_directory
                    / f"perturbation_metrics_{method}_test.json"
                ]
                if (
                    benchmark_directory
                    / f"perturbation_metrics_{method}_test.json"
                ).is_file()
                else []
            ),
        }

    neural_root = suite_directory / "neural"
    if neural_root.is_dir():
        for variant_directory in sorted(
            path for path in neural_root.iterdir() if path.is_dir()
        ):
            paths = sorted(
                path
                for path in variant_directory.glob("seed_*_metrics.json")
                if "_perturbation_" not in path.name
            )
            if not paths:
                continue
            per_seed = [_metric_values(path, primary_metric) for path in paths]
            shared = sorted(set.intersection(*(set(values) for values in per_seed)))
            name = variant_directory.name
            methods[name] = {
                "role": (
                    "proposed_method"
                    if name in PROPOSED_METHODS
                    else "ablation"
                    if name.startswith("cg_hhgm_")
                    else "learned_baseline"
                ),
                "metric_files": [str(path) for path in paths],
                "seed_count": len(paths),
                "values": {
                    query_id: mean(values[query_id] for values in per_seed)
                    for query_id in shared
                },
                "safety": _aggregate_safety(paths),
                "perturbation": _perturbation_intervals(
                    sorted(
                        variant_directory.glob(
                            "seed_*_perturbation_metrics.json"
                        )
                    )
                ),
            }

    auxiliary_paths = [
        ("lambdamart", suite_directory / "lambdamart" / "metrics_lambdamart.json")
    ]
    auxiliary_paths.extend(
        (
            path.stem.removeprefix("metrics_").removesuffix("_test"),
            path,
        )
        for path in sorted(suite_directory.glob("molecular*/metrics_*_test.json"))
    )
    for method, path in auxiliary_paths:
        if path.is_file():
            values = _metric_values(path, primary_metric)
            existing = methods.get(method)
            if existing is not None:
                if existing["values"] != values:
                    raise ValueError(
                        f"Conflicting auxiliary metrics for {method}: "
                        f"{existing['metric_files']} and {path}"
                    )
                existing["metric_files"].append(str(path))
                continue
            methods[method] = {
                "role": "auxiliary_baseline",
                "metric_files": [str(path)],
                "values": values,
                "safety": _aggregate_safety([path]),
                "perturbation": None,
            }

    baseline_names = [
        name
        for name, payload in methods.items()
        if payload["role"] not in {"proposed_method", "ablation"}
    ]
    if not baseline_names:
        raise ValueError("No completed baseline metrics were found.")
    reference_method = max(
        baseline_names,
        key=lambda name: mean(methods[name]["values"].values()),
    )
    reference = methods[reference_method]["values"]
    raw_pvalues: dict[str, float | None] = {}
    comparisons: dict[str, Any] = {}
    for method, payload in methods.items():
        if method == reference_method:
            continue
        shared = sorted(set(reference) & set(payload["values"]))
        differences = [
            payload["values"][query_id] - reference[query_id]
            for query_id in shared
        ]
        raw_pvalues[method] = paired_permutation_pvalue(differences)
        standard_deviation = pstdev(differences) if len(differences) > 1 else 0.0
        comparisons[method] = {
            "reference": reference_method,
            "paired_query_count": len(shared),
            "mean_difference": mean(differences) if differences else None,
            "difference_95_ci": bootstrap_interval(differences),
            "paired_effect_size_dz": (
                mean(differences) / standard_deviation
                if differences and standard_deviation
                else None
            ),
            "permutation_pvalue": raw_pvalues[method],
        }
    adjusted = holm_adjust(raw_pvalues)
    for method, value in adjusted.items():
        comparisons[method]["holm_adjusted_pvalue"] = value

    return {
        "statistics_version": STATISTICS_VERSION,
        "evaluation_scope": "isolated_test_split",
        "primary_metric": primary_metric,
        "strongest_non_proposed_baseline": reference_method,
        "methods": {
            method: {
                "role": payload["role"],
                "seed_count": payload.get("seed_count", 1),
                "query_count": len(payload["values"]),
                "interval": bootstrap_interval(list(payload["values"].values())),
                "safety": payload["safety"],
                "perturbation": payload.get("perturbation"),
                "metric_files": payload["metric_files"],
            }
            for method, payload in methods.items()
        },
        "paired_comparisons": comparisons,
        "proposed_method": (
            "cg_hhgm"
        ),
        "proposed_method_comparison": comparisons.get(
            "cg_hhgm"
        ),
        "proposed_method_comparisons": {
            method: comparisons.get(method)
            for method in PROPOSED_METHODS
            if method in methods
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute bootstrap intervals and paired model-matcher tests."
    )
    parser.add_argument("--metrics", action="append", type=Path)
    parser.add_argument("--primary-metric", default="ndcg@10")
    parser.add_argument("--reference")
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.benchmark is not None or args.suite is not None:
        if args.benchmark is None or args.suite is None:
            parser.error("--benchmark and --suite must be supplied together.")
        result = compare_research_suite(
            benchmark_directory=args.benchmark,
            suite_directory=args.suite,
            primary_metric=args.primary_metric,
        )
    else:
        if not args.metrics or not args.reference:
            parser.error("--metrics and --reference are required for file comparison.")
        result = compare_metric_files(
            args.metrics,
            primary_metric=args.primary_metric,
            reference_method=args.reference,
        )
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
