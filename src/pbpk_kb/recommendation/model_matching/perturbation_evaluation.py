from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.licensed_snapshot_reuse import (
    FrozenObservedSeries,
    SnapshotWindowsManifest,
    _auc,
    _interpolate,
    _prediction_series_from_csvs,
)
from pbpk_kb.recommendation.model_matching.adaptation_planner import (
    _atomic_json,
    _atomic_jsonl,
    _read_jsonl,
)


PERTURBATION_EVALUATOR_VERSION = "pbpk-scenario-perturbation-evaluator-1.0.0"


class PerturbationEvaluationError(RuntimeError):
    pass


def _curve_metrics(
    *,
    base_times: list[float],
    base_values: list[float],
    perturbed_times: list[float],
    perturbed_values: list[float],
    dose_ratio: float,
) -> dict[str, float | int | None]:
    common_times = [
        value
        for value in base_times
        if perturbed_times[0] <= value <= perturbed_times[-1]
    ]
    if len(common_times) < 2:
        raise PerturbationEvaluationError("Base and perturbed outputs have no usable overlap.")
    base_by_time = dict(zip(base_times, base_values, strict=True))
    base = [base_by_time[value] for value in common_times]
    perturbed = [
        _interpolate(
            perturbed_times,
            perturbed_values,
            value,
            logarithmic=False,
        )
        for value in common_times
    ]
    if any(not math.isfinite(value) for value in base + perturbed):
        raise PerturbationEvaluationError("Curve contains a non-finite value.")
    scale = max(abs(value) for value in base)
    rmse = math.sqrt(mean((right - left) ** 2 for left, right in zip(base, perturbed, strict=True)))
    normalized_rmse = rmse / scale if scale > 0 else None
    base_normalized = base
    perturbed_normalized = [value / dose_ratio for value in perturbed]
    normalized_scale = max(abs(value) for value in base_normalized)
    dose_normalized_rmse = math.sqrt(
        mean(
            (right - left) ** 2
            for left, right in zip(base_normalized, perturbed_normalized, strict=True)
        )
    )
    base_auc = _auc(common_times, base)
    perturbed_auc = _auc(common_times, perturbed)
    base_cmax = max(base)
    perturbed_cmax = max(perturbed)
    return {
        "common_time_point_count": len(common_times),
        "base_auc": base_auc,
        "perturbed_auc": perturbed_auc,
        "auc_ratio": perturbed_auc / base_auc if base_auc != 0 else None,
        "base_cmax": base_cmax,
        "perturbed_cmax": perturbed_cmax,
        "cmax_ratio": perturbed_cmax / base_cmax if base_cmax != 0 else None,
        "curve_rmse": rmse,
        "curve_nrmse": normalized_rmse,
        "dose_normalized_curve_nrmse": (
            dose_normalized_rmse / normalized_scale if normalized_scale > 0 else None
        ),
    }


def _probe(output_probe: Mapping[str, Any], output_path: str) -> FrozenObservedSeries:
    return FrozenObservedSeries(
        dataset_name=str(output_probe["dataset_name"]),
        dataset_index=0,
        column_index=0,
        source_location=str(output_probe["source_location"]),
        reference=output_probe.get("reference"),
        output_path=output_path,
        scaling=str(output_probe["scaling"]),
        time_values=[0.0],
        time_unit="h",
        observed_values=[1.0],
        observed_unit="mol/l",
        molecular_weight_g_per_mol=float(
            output_probe["molecular_weight_g_per_mol"]
        ),
    )


def evaluate_simulation_perturbations(
    *,
    plan_directory: Path,
    audit_directory: Path,
    windows_results_root: Path,
    output_directory: Path,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    plan_manifest = json.loads(
        (plan_directory / "manifest.json").read_text(encoding="utf-8")
    )
    audit_summary = json.loads(
        (audit_directory / "summary.json").read_text(encoding="utf-8")
    )
    plan_path = plan_directory / "perturbation_plans.jsonl"
    audit_path = audit_directory / "audit_index.jsonl"
    if sha256_file(plan_path) != str(
        plan_manifest["output_checksums"]["perturbation_plans.jsonl"]
    ):
        raise PerturbationEvaluationError("Perturbation plan checksum drifted.")
    if sha256_file(audit_path) != str(
        audit_summary["output_checksums"]["audit_index.jsonl"]
    ):
        raise PerturbationEvaluationError("Perturbation audit checksum drifted.")
    audits = {
        str(row["protocol_key"]): row for row in _read_jsonl(audit_path)
    }
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for plan in _read_jsonl(plan_path):
        key = str(plan["protocol_key"])
        try:
            audit = audits[key]
            if not audit.get("engineering_accepted"):
                raise PerturbationEvaluationError(
                    "Perturbation protocol did not pass engineering audit."
                )
            result_root = windows_results_root / key
            manifest_path = result_root / "windows_run_manifest.json"
            manifest = SnapshotWindowsManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8-sig")
            )
            run = next(value for value in manifest.runs if value.run_id == "run_1")
            by_sequence = {value.sequence: value for value in run.simulations}
            base_result = by_sequence[1]
            for perturbation in plan["perturbations"]:
                perturbed_result = by_sequence[int(perturbation["simulation_sequence"])]
                for output_probe in plan["output_probes"]:
                    base_observed = _probe(
                        output_probe, str(output_probe["base_output_path"])
                    )
                    perturbed_path = (
                        f"{perturbation['simulation_name']}|"
                        f"{output_probe['output_suffix']}"
                    )
                    perturbed_observed = _probe(output_probe, perturbed_path)
                    base_times, base_values, base_unit, base_csv = (
                        _prediction_series_from_csvs(
                            result_root=result_root,
                            artifacts=base_result.artifacts,
                            simulation_name=str(plan["base_simulation_name"]),
                            observed=base_observed,
                        )
                    )
                    perturbed_times, perturbed_values, perturbed_unit, perturbed_csv = (
                        _prediction_series_from_csvs(
                            result_root=result_root,
                            artifacts=perturbed_result.artifacts,
                            simulation_name=str(perturbation["simulation_name"]),
                            observed=perturbed_observed,
                        )
                    )
                    rows.append(
                        {
                            "perturbation_group_id": plan["perturbation_group_id"],
                            "protocol_key": key,
                            "task_ids": plan["task_ids"],
                            "active_moiety": plan["active_moiety"],
                            "dataset_name": output_probe["dataset_name"],
                            "perturbation_type": perturbation["perturbation_type"],
                            "domain_status": perturbation["domain_status"],
                            "old_value": perturbation["old_value"],
                            "new_value": perturbation["new_value"],
                            "unit": perturbation["unit"],
                            "dose_ratio": perturbation["dose_ratio"],
                            "base_prediction_unit": base_unit,
                            "perturbed_prediction_unit": perturbed_unit,
                            "base_prediction_csv": base_csv,
                            "perturbed_prediction_csv": perturbed_csv,
                            **_curve_metrics(
                                base_times=base_times,
                                base_values=base_values,
                                perturbed_times=perturbed_times,
                                perturbed_values=perturbed_values,
                                dose_ratio=float(perturbation["dose_ratio"]),
                            ),
                            "engineering_accepted": True,
                            "scientific_grade_generated": False,
                        }
                    )
        except Exception as exc:
            failures.append(
                {
                    "perturbation_group_id": plan["perturbation_group_id"],
                    "protocol_key": key,
                    "failure_type": type(exc).__name__,
                    "detail": str(exc),
                    "scientific_grade_generated": False,
                }
            )
    outputs = {
        "simulation_continuity.jsonl": _atomic_jsonl(
            output_directory / "simulation_continuity.jsonl", rows
        ),
        "failures.jsonl": _atomic_jsonl(
            output_directory / "failures.jsonl", failures
        ),
    }
    manifest = {
        "evaluator_version": PERTURBATION_EVALUATOR_VERSION,
        "evaluated_curve_count": len(rows),
        "failed_group_count": len(failures),
        "complete": not failures,
        "scientific_grade_generated": False,
        "output_checksums": outputs,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(output_directory / "manifest.json", manifest)
    return manifest


def _rank_correlation(
    base: Mapping[str, float], perturbed: Mapping[str, float]
) -> float | None:
    candidates = sorted(set(base) & set(perturbed))
    concordant = discordant = 0
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            base_delta = base[left] - base[right]
            perturbed_delta = perturbed[left] - perturbed[right]
            if base_delta == 0 or perturbed_delta == 0:
                continue
            if base_delta * perturbed_delta > 0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else None


def evaluate_ranking_continuity(
    *,
    base_rows: Iterable[Mapping[str, Any]],
    perturbed_rows: Iterable[Mapping[str, Any]],
    task_links: Mapping[str, str],
    k: int = 10,
) -> dict[str, Any]:
    base_by_query: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    perturbed_by_query: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in base_rows:
        base_by_query[str(row["query_id"])].append(row)
    for row in perturbed_rows:
        perturbed_by_query[str(row["query_id"])].append(row)
    details = []
    for perturbed_query, base_query in sorted(task_links.items()):
        base_ranked = sorted(
            base_by_query[base_query],
            key=lambda row: (-float(row["score"]), str(row["candidate_id"])),
        )
        perturbed_ranked = sorted(
            perturbed_by_query[perturbed_query],
            key=lambda row: (-float(row["score"]), str(row["candidate_id"])),
        )
        if not base_ranked or not perturbed_ranked:
            raise PerturbationEvaluationError(
                f"Missing ranking rows for {base_query} -> {perturbed_query}."
            )
        base_scores = {
            str(row["candidate_id"]): float(row["score"]) for row in base_ranked
        }
        perturbed_scores = {
            str(row["candidate_id"]): float(row["score"])
            for row in perturbed_ranked
        }
        left = {str(row["candidate_id"]) for row in base_ranked[:k]}
        right = {str(row["candidate_id"]) for row in perturbed_ranked[:k]}
        base_top = base_ranked[0]
        perturbed_top = perturbed_ranked[0]
        details.append(
            {
                "base_query_id": base_query,
                "perturbed_query_id": perturbed_query,
                "top1_stable": str(base_top["candidate_id"])
                == str(perturbed_top["candidate_id"]),
                "topk_jaccard": len(left & right) / len(left | right),
                "kendall_tau": _rank_correlation(base_scores, perturbed_scores),
                "reuse_decision_stable": base_top.get("reuse_decision")
                == perturbed_top.get("reuse_decision"),
            }
        )
    return {
        "pair_count": len(details),
        "top1_stability_rate": mean(row["top1_stable"] for row in details)
        if details
        else None,
        "mean_topk_jaccard": mean(row["topk_jaccard"] for row in details)
        if details
        else None,
        "mean_kendall_tau": mean(
            row["kendall_tau"] for row in details if row["kendall_tau"] is not None
        )
        if any(row["kendall_tau"] is not None for row in details)
        else None,
        "reuse_layer_stability_rate": mean(
            row["reuse_decision_stable"] for row in details
        )
        if details
        else None,
        "details": details,
        "scientific_grade_generated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure simulation continuity for executed perturbation protocols."
    )
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--audits", type=Path, required=True)
    parser.add_argument("--windows-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_simulation_perturbations(
        plan_directory=args.plans,
        audit_directory=args.audits,
        windows_results_root=args.windows_results,
        output_directory=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
