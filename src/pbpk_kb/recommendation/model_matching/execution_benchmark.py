from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Iterator, Mapping


BENCHMARK_VERSION = "pbpk-execution-derived-recommendation-benchmark-1.0.0"


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, default=str)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = _canonical_bytes(row)
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield row


def _stable_id(namespace: str, *parts: Any) -> str:
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8"))
    for part in parts:
        digest.update(b"\x1f")
        digest.update(_canonical_bytes(part))
    return f"{namespace}:{digest.hexdigest()}"


def _normalized(values: object) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    return {
        " ".join(str(value).casefold().split())
        for value in values
        if str(value).strip()
    }


def _split(group_key: str) -> str:
    bucket = int(hashlib.sha256(group_key.casefold().encode("utf-8")).hexdigest()[:8], 16) % 10
    return "train" if bucket < 6 else "validation" if bucket < 8 else "test"


_ROUTE_PATTERN = re.compile(
    r"(?<![a-z])(?P<route>po|oral|iv|i\.v\.?|intravenous)(?![a-z])",
    re.IGNORECASE,
)
_DOSE_PATTERN = re.compile(
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)\s*"
    r"(?P<unit>mg|g|ug|µg|μg)(?P<basis>\s*/\s*kg)?\b",
    re.IGNORECASE,
)


def parse_simulation_requirement(name: str) -> dict[str, Any]:
    lowered = name.casefold()
    route_match = _ROUTE_PATTERN.search(name)
    dose_match = _DOSE_PATTERN.search(name)
    route = None
    dose_value = None
    dose_unit = None
    if route_match:
        raw_route = route_match.group("route").casefold()
        route = "oral" if raw_route in {"po", "oral"} else "intravenous"
    if dose_match:
        dose_value = float(dose_match.group("value"))
        dose_unit = dose_match.group("unit").replace("μ", "µ")
        if dose_match.group("basis"):
            dose_unit += "/kg"
    frequency = None
    for token, canonical in (
        ("bid", "twice_daily"),
        ("tid", "three_times_daily"),
        ("qd", "once_daily"),
        ("single", "single_dose"),
    ):
        if re.search(rf"\b{token}\b", lowered):
            frequency = canonical
            break
    if frequency is None:
        frequency = "single_dose"
    sex = "female" if "female" in lowered else "male" if "male" in lowered else None
    formulation = None
    for token, canonical in (
        ("caps", "capsule"),
        ("capsule", "capsule"),
        ("tab", "tablet"),
        ("tablet", "tablet"),
        ("sol", "solution"),
        ("solution", "solution"),
        ("bolus", "bolus"),
        ("infusion", "infusion"),
    ):
        if re.search(rf"\b{token}\b", lowered):
            formulation = canonical
            break
    return {
        "route": route,
        "dose_value": dose_value,
        "dose_unit": dose_unit,
        "frequency": frequency,
        "sex": sex,
        "formulation": formulation,
    }


def _load_registration(registration: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = registration / "provenance" / "snapshot_reuse_protocol.json"
    model_versions_path = registration / "model_versions.jsonl.gz"
    if not protocol_path.is_file() or not model_versions_path.is_file():
        raise FileNotFoundError(f"Registration is incomplete: {registration}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    models = list(_iter_jsonl(model_versions_path))
    if len(models) != 1:
        raise ValueError(f"Expected one model version in {registration}; found {len(models)}")
    return protocol, models[0]


def _candidate_record(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": model["model_version_id"],
        "model_family": model.get("model_family"),
        "active_moieties": sorted(_normalized(model.get("compounds"))),
        "species": sorted(_normalized(model.get("species"))),
        "routes": sorted(_normalized(model.get("routes"))),
        "enzymes": sorted(_normalized(model.get("enzymes"))),
        "transporters": sorted(_normalized(model.get("transporters"))),
        "process_types": sorted(_normalized(model.get("process_types"))),
        "populations": sorted(_normalized(model.get("populations"))),
        "formulations": sorted(_normalized(model.get("formulations"))),
        "compartments": sorted(_normalized(model.get("compartments"))),
        "output_names": sorted(_normalized(model.get("output_names"))),
        "software": model.get("software"),
        "software_version": model.get("software_version"),
        "license": model.get("license"),
        "static_parse_status": model.get("static_parse_status"),
        "execution_probe_status": model.get("execution_probe_status"),
        "reusable_candidate": bool(model.get("reusable_candidate")),
        "repository_commit": model.get("repository_commit"),
        "project_sha256": model.get("project_sha256"),
        "simulation_count": int(model.get("simulation_count") or 0),
        "parameter_record_count": int(model.get("parameter_record_count") or 0),
        "dataset_reference_count": int(model.get("dataset_reference_count") or 0),
        "validation_metric_count": int(model.get("validation_metric_count") or 0),
    }


def build_execution_benchmark(
    *,
    registrations: list[Path],
    output_directory: Path,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    tasks: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    candidates: dict[str, dict[str, Any]] = {}
    registration_inputs: list[dict[str, Any]] = []
    for registration in sorted(path.resolve() for path in registrations):
        protocol, model = _load_registration(registration)
        candidate = _candidate_record(model)
        candidate_id = str(candidate["candidate_id"])
        candidates[candidate_id] = candidate
        compounds = sorted(_normalized(model.get("compounds")))
        if not compounds:
            raise ValueError(f"Model {candidate_id} has no active moiety.")
        if len(compounds) != 1:
            raise ValueError(
                f"Model {candidate_id} has {len(compounds)} active moieties; "
                "DDI/multi-compound task generation requires an explicit role map."
            )
        compound = compounds[0]
        group_id = _stable_id("validation-group", compound, protocol.get("source_snapshot_semantic_sha256"))
        for simulation in protocol.get("simulations") or []:
            if simulation.get("role") != "model_verification":
                continue
            series = simulation.get("evaluation_series")
            if not isinstance(series, dict):
                raise ValueError("Every verification simulation requires an evaluation series.")
            scenario = parse_simulation_requirement(str(simulation.get("name") or ""))
            task_id = _stable_id(
                "execution-task",
                BENCHMARK_VERSION,
                compound,
                group_id,
                int(simulation["sequence"]),
                scenario,
            )
            tasks.append(
                {
                    "benchmark_version": BENCHMARK_VERSION,
                    "benchmark_level": "execution_derived",
                    "task_id": task_id,
                    "validation_group_id": group_id,
                    "task_type": "plasma_concentration_time_prediction",
                    "active_moieties": [compound],
                    "species": "human",
                    "scenario": scenario,
                    "required_outputs": ["plasma_concentration_time", "auc0_last", "cmax"],
                    "split": _split(compound),
                    "hidden_truth_available": True,
                }
            )
            truth.append(
                {
                    "task_id": task_id,
                    "validation_group_id": group_id,
                    "source_candidate_id": candidate_id,
                    "source_simulation_sequence": int(simulation["sequence"]),
                    "dataset_name": series["dataset_name"],
                    "source_location": series["source_location"],
                    "reference": series.get("reference"),
                    "output_path": series["output_path"],
                    "time_values": series["time_values"],
                    "time_unit": series["time_unit"],
                    "observed_values": series["observed_values"],
                    "observed_unit": series["observed_unit"],
                    "molecular_weight_g_per_mol": series["molecular_weight_g_per_mol"],
                    "criteria": protocol["criteria"],
                }
            )
        protocol_path = registration / "provenance" / "snapshot_reuse_protocol.json"
        registration_inputs.append(
            {
                "registration": str(registration),
                "summary_sha256": _sha256_file(registration / "summary.json"),
                "protocol_sha256": _sha256_file(protocol_path),
                "model_versions_sha256": _sha256_file(registration / "model_versions.jsonl.gz"),
            }
        )
    tasks.sort(key=lambda row: str(row["task_id"]))
    truth.sort(key=lambda row: str(row["task_id"]))
    candidate_rows = sorted(candidates.values(), key=lambda row: str(row["candidate_id"]))
    leakage_audit = _assign_compound_and_family_disjoint_splits(
        tasks=tasks,
        candidates=candidate_rows,
    )
    if not leakage_audit["passed"]:
        raise ValueError("Compound/model-family leakage audit failed.")
    task_ids = {str(row["task_id"]) for row in tasks}
    if len(task_ids) != len(tasks) or task_ids != {str(row["task_id"]) for row in truth}:
        raise ValueError("Task/truth IDs are not unique and identical.")
    forbidden_tokens = {
        str(row["source_candidate_id"]).casefold()
        for row in truth
    } | {
        str(row["dataset_name"]).casefold()
        for row in truth
    }
    for task in tasks:
        serialized = json.dumps(task, ensure_ascii=True, sort_keys=True).casefold()
        if any(token and token in serialized for token in forbidden_tokens):
            raise ValueError(f"Answer leakage detected in task {task['task_id']}")
    outputs = {
        "benchmark_tasks.jsonl": _atomic_jsonl(output_directory / "benchmark_tasks.jsonl", tasks),
        "benchmark_truth.jsonl": _atomic_jsonl(output_directory / "benchmark_truth.jsonl", truth),
        "candidates.jsonl": _atomic_jsonl(output_directory / "candidates.jsonl", candidate_rows),
        "leakage_audit.json": _atomic_json(
            output_directory / "leakage_audit.json", leakage_audit
        ),
    }
    os.chmod(output_directory / "benchmark_truth.jsonl", 0o600)
    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_level": "execution_derived",
        "scientific_claim_eligible": False,
        "task_count": len(tasks),
        "validation_group_count": len({row["validation_group_id"] for row in tasks}),
        "compound_count": len({tuple(row["active_moieties"]) for row in tasks}),
        "candidate_count": len(candidate_rows),
        "answer_leakage_check_passed": True,
        "truth_file_mode": "0600",
        "registrations": registration_inputs,
        "output_checksums": outputs,
        "created_at": datetime.now(UTC).isoformat(),
        "claim_blockers": [
            "EXECUTION_RESULTS_NOT_YET_ATTACHED",
            "INDEPENDENT_COMPOUND_TEST_SPLIT_NOT_YET_VERIFIED",
        ],
    }
    _atomic_json(output_directory / "manifest.json", manifest)
    return manifest


def _to_hours(values: list[float], unit: str) -> list[float]:
    factor = {
        "h": 1.0,
        "hr": 1.0,
        "hour": 1.0,
        "hours": 1.0,
        "min": 1.0 / 60.0,
        "minute": 1.0 / 60.0,
        "minutes": 1.0 / 60.0,
        "s": 1.0 / 3600.0,
    }.get(unit.casefold())
    if factor is None:
        raise ValueError(f"Unsupported time unit: {unit}")
    return [float(value) * factor for value in values]


def _to_mol_per_litre(values: list[float], unit: str, molecular_weight: float) -> list[float]:
    normalized = unit.casefold().replace("μ", "µ").replace(" ", "")
    if normalized in {"mol/l", "m"}:
        factor = 1.0
    elif normalized in {"mmol/l", "mm"}:
        factor = 1e-3
    elif normalized in {"µmol/l", "umol/l", "µm"}:
        factor = 1e-6
    elif normalized in {"nmol/l", "nm"}:
        factor = 1e-9
    elif normalized in {"mg/l", "µg/ml", "ug/ml"}:
        factor = 1e-3 / molecular_weight
    elif normalized in {"ng/ml", "µg/l", "ug/l"}:
        factor = 1e-6 / molecular_weight
    else:
        raise ValueError(f"Unsupported concentration unit: {unit}")
    return [float(value) * factor for value in values]


def _interpolate_log_linear(
    source_time: list[float], source_values: list[float], target_time: list[float]
) -> list[float]:
    if len(source_time) != len(source_values) or len(source_time) < 2:
        raise ValueError("Prediction time and value arrays must be aligned and contain at least two points.")
    ordered = sorted(zip(source_time, source_values, strict=True))
    if len({item[0] for item in ordered}) != len(ordered):
        raise ValueError("Prediction times must be unique.")
    output = []
    for target in target_time:
        if target < ordered[0][0] or target > ordered[-1][0]:
            raise ValueError("Prediction time range does not cover every observation.")
        right = next(index for index, item in enumerate(ordered) if item[0] >= target)
        if ordered[right][0] == target:
            output.append(float(ordered[right][1]))
            continue
        left = right - 1
        x0, y0 = ordered[left]
        x1, y1 = ordered[right]
        fraction = (target - x0) / (x1 - x0)
        if y0 > 0 and y1 > 0:
            output.append(math.exp(math.log(y0) + fraction * (math.log(y1) - math.log(y0))))
        else:
            output.append(y0 + fraction * (y1 - y0))
    return output


def _geometric(values: Iterable[float]) -> float | None:
    selected = [float(value) for value in values if float(value) > 0]
    return math.exp(mean(math.log(value) for value in selected)) if selected else None


def _auc(time_values: list[float], concentration_values: list[float]) -> float:
    return sum(
        (time_values[index] - time_values[index - 1])
        * (concentration_values[index] + concentration_values[index - 1])
        / 2.0
        for index in range(1, len(time_values))
    )


def evaluate_execution_result(
    result: Mapping[str, Any], truth: Mapping[str, Any]
) -> dict[str, Any]:
    engineering_failures = [
        field
        for field in (
            "compile_success",
            "execution_success",
            "dimensional_check_passed",
            "reproducibility_passed",
        )
        if result.get(field) is not True
    ]
    if result.get("mass_balance_passed") is False:
        engineering_failures.append("mass_balance_passed")
    base = {
        "task_id": truth["task_id"],
        "candidate_id": result["candidate_id"],
        "validation_group_id": truth["validation_group_id"],
        "engineering_passed": not engineering_failures,
        "engineering_failures": engineering_failures,
        "mass_balance_status": (
            "passed" if result.get("mass_balance_passed") is True
            else "failed" if result.get("mass_balance_passed") is False
            else "not_available"
        ),
    }
    if engineering_failures:
        return {**base, "scientific_passed": False, "reason_code": "ENGINEERING_EXECUTION_FAILED"}
    molecular_weight = float(truth["molecular_weight_g_per_mol"])
    observed_time = _to_hours(list(truth["time_values"]), str(truth["time_unit"]))
    observed = _to_mol_per_litre(
        list(truth["observed_values"]), str(truth["observed_unit"]), molecular_weight
    )
    predicted_time = _to_hours(
        [float(value) for value in result["predicted_time_values"]],
        str(result["predicted_time_unit"]),
    )
    predicted = _to_mol_per_litre(
        [float(value) for value in result["predicted_values"]],
        str(result["predicted_unit"]),
        molecular_weight,
    )
    aligned = _interpolate_log_linear(predicted_time, predicted, observed_time)
    pairs = [
        (observed_value, predicted_value)
        for observed_value, predicted_value in zip(observed, aligned, strict=True)
        if observed_value > 0 and predicted_value > 0
    ]
    if not pairs:
        return {**base, "scientific_passed": False, "reason_code": "NO_POSITIVE_ALIGNED_PAIRS"}
    ratios = [predicted_value / observed_value for observed_value, predicted_value in pairs]
    fold_errors = [max(ratio, 1.0 / ratio) for ratio in ratios]
    gmfe = _geometric(fold_errors)
    bias = _geometric(ratios)
    two_fold = sum(value <= 2.0 for value in fold_errors) / len(fold_errors)
    auc_observed = _auc(observed_time, observed)
    auc_predicted = _auc(observed_time, aligned)
    auc_ratio = auc_predicted / auc_observed if auc_observed > 0 else None
    cmax_observed = max(observed)
    cmax_predicted = max(aligned)
    cmax_ratio = cmax_predicted / cmax_observed if cmax_observed > 0 else None
    criteria = truth["criteria"]
    dataset_passed = (
        gmfe is not None
        and gmfe <= float(criteria["dataset_gmfe_max"])
        and bias is not None
        and float(criteria["dataset_bias_min"]) <= bias <= float(criteria["dataset_bias_max"])
    )
    endpoint_passed = all(
        value is not None
        and float(criteria["endpoint_ratio_min"]) <= value <= float(criteria["endpoint_ratio_max"])
        for value in (auc_ratio, cmax_ratio)
    )
    return {
        **base,
        "scientific_passed": bool(dataset_passed and endpoint_passed),
        "reason_code": "PASSED" if dataset_passed and endpoint_passed else "SCIENTIFIC_CRITERIA_FAILED",
        "positive_pair_count": len(pairs),
        "gmfe": gmfe,
        "geometric_bias": bias,
        "two_fold_fraction": two_fold,
        "auc0_last_ratio": auc_ratio,
        "cmax_ratio": cmax_ratio,
        "observed_mol_per_litre": [item[0] for item in pairs],
        "predicted_mol_per_litre": [item[1] for item in pairs],
        "adaptation_operations": result.get("adaptation_operations") or [],
        "model_structure_changed": bool(result.get("model_structure_changed")),
        "compound_parameters_changed": bool(result.get("compound_parameters_changed")),
        "execution_manifest_sha256": result.get("execution_manifest_sha256"),
    }


def build_automatic_labels(
    *,
    benchmark_directory: Path,
    execution_results_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    tasks = {str(row["task_id"]): row for row in _iter_jsonl(benchmark_directory / "benchmark_tasks.jsonl")}
    truth = {str(row["task_id"]): row for row in _iter_jsonl(benchmark_directory / "benchmark_truth.jsonl")}
    candidates = {str(row["candidate_id"]): row for row in _iter_jsonl(benchmark_directory / "candidates.jsonl")}
    expected_manifest = json.loads((benchmark_directory / "manifest.json").read_text(encoding="utf-8"))
    if _sha256_file(benchmark_directory / "benchmark_truth.jsonl") != expected_manifest["output_checksums"]["benchmark_truth.jsonl"]:
        raise ValueError("Benchmark truth checksum drifted.")
    raw_results = list(_iter_jsonl(execution_results_path))
    seen_pairs: set[tuple[str, str]] = set()
    scenario_metrics: list[dict[str, Any]] = []
    result_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in raw_results:
        task_id = str(row.get("task_id") or "")
        candidate_id = str(row.get("candidate_id") or "")
        pair = (task_id, candidate_id)
        if task_id not in tasks or candidate_id not in candidates:
            raise ValueError(f"Unknown execution result pair: {pair}")
        if pair in seen_pairs:
            raise ValueError(f"Duplicate execution result pair: {pair}")
        seen_pairs.add(pair)
        result_by_pair[pair] = row
        scenario_metrics.append(evaluate_execution_result(row, truth[task_id]))
    metrics_by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scenario_metrics:
        metrics_by_group[(str(row["validation_group_id"]), str(row["candidate_id"]))].append(row)
    group_labels: list[dict[str, Any]] = []
    for (group_id, candidate_id), rows in sorted(metrics_by_group.items()):
        group_tasks = [task for task in tasks.values() if task["validation_group_id"] == group_id]
        query_compounds = set(
            group_tasks[0].get("required_model_compounds")
            or group_tasks[0]["active_moieties"]
        )
        candidate = candidates[candidate_id]
        same_active_moiety = query_compounds.issubset(
            set(candidate.get("active_moieties") or [])
        )
        complete = len(rows) == len(group_tasks)
        positive_pairs = [
            (observed, predicted)
            for row in rows
            for observed, predicted in zip(
                row.get("observed_mol_per_litre") or [],
                row.get("predicted_mol_per_litre") or [],
                strict=True,
            )
        ]
        ratios = [predicted / observed for observed, predicted in positive_pairs if observed > 0 and predicted > 0]
        fold_errors = [max(ratio, 1.0 / ratio) for ratio in ratios]
        pooled_gmfe = _geometric(fold_errors)
        pooled_bias = _geometric(ratios)
        pooled_two_fold = (
            sum(value <= 2.0 for value in fold_errors) / len(fold_errors)
            if fold_errors else None
        )
        criteria = truth[str(group_tasks[0]["task_id"])]["criteria"]
        scenario_pass_fraction = sum(bool(row.get("scientific_passed")) for row in rows) / len(group_tasks)
        pooled_passed = (
            complete
            and pooled_gmfe is not None
            and pooled_gmfe <= float(criteria["pooled_gmfe_max"])
            and pooled_bias is not None
            and float(criteria["pooled_bias_min"]) <= pooled_bias <= float(criteria["pooled_bias_max"])
            and pooled_two_fold is not None
            and pooled_two_fold >= float(criteria["pooled_two_fold_fraction_min"])
            and scenario_pass_fraction >= float(criteria["endpoint_scenario_fraction_min"])
            and all(bool(row.get("engineering_passed")) for row in rows)
        )
        result_rows = [result_by_pair[(str(task["task_id"]), candidate_id)] for task in group_tasks if (str(task["task_id"]), candidate_id) in result_by_pair]
        adaptations = [operation for result in result_rows for operation in (result.get("adaptation_operations") or [])]
        unchanged = not adaptations and not any(
            bool(result.get("model_structure_changed")) or bool(result.get("compound_parameters_changed"))
            for result in result_rows
        )
        mechanism_overlap = bool(
            set(candidate.get("enzymes") or [])
            | set(candidate.get("transporters") or [])
            | set(candidate.get("process_types") or [])
        )
        if same_active_moiety and pooled_passed and unchanged:
            grade, decision = 3, "direct_reuse"
        elif same_active_moiety and pooled_passed and adaptations:
            grade, decision = 2, "controlled_adaptation"
        elif mechanism_overlap and any(bool(row.get("engineering_passed")) for row in rows):
            grade, decision = 1, "mechanistic_reference"
        else:
            grade, decision = 0, "reject"
        group_labels.append(
            {
                "validation_group_id": group_id,
                "candidate_id": candidate_id,
                "active_moiety_match": same_active_moiety,
                "execution_coverage": len(rows) / len(group_tasks),
                "scenario_pass_fraction": scenario_pass_fraction,
                "pooled_positive_pair_count": len(ratios),
                "pooled_gmfe": pooled_gmfe,
                "pooled_geometric_bias": pooled_bias,
                "pooled_two_fold_fraction": pooled_two_fold,
                "pooled_scientific_passed": pooled_passed,
                "adaptation_operation_count": len(adaptations),
                "relevance_grade": grade,
                "decision": decision,
            }
        )
    label_by_group = {
        (str(row["validation_group_id"]), str(row["candidate_id"])): row
        for row in group_labels
    }
    pair_labels = []
    for row in scenario_metrics:
        group = label_by_group[(str(row["validation_group_id"]), str(row["candidate_id"]))]
        grade = int(group["relevance_grade"])
        if not row.get("scientific_passed") and grade >= 2:
            grade = 1
        pair_labels.append(
            {
                "task_id": row["task_id"],
                "candidate_id": row["candidate_id"],
                "relevance_grade": grade,
                "decision": group["decision"] if grade == group["relevance_grade"] else "mechanistic_reference",
                "scenario_scientific_passed": bool(row.get("scientific_passed")),
                "validation_group_id": row["validation_group_id"],
            }
        )
    outputs = {
        "scenario_metrics.jsonl": _atomic_jsonl(output_directory / "scenario_metrics.jsonl", scenario_metrics),
        "candidate_group_labels.jsonl": _atomic_jsonl(output_directory / "candidate_group_labels.jsonl", group_labels),
        "labels.jsonl": _atomic_jsonl(output_directory / "labels.jsonl", pair_labels),
    }
    fully_covered_groups = sum(float(row["execution_coverage"]) == 1.0 for row in group_labels)
    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_level": "execution_derived",
        "execution_results_path": str(execution_results_path.resolve()),
        "execution_results_sha256": _sha256_file(execution_results_path),
        "scenario_result_count": len(scenario_metrics),
        "candidate_group_count": len(group_labels),
        "fully_covered_candidate_group_count": fully_covered_groups,
        "label_counts": {
            str(grade): sum(int(row["relevance_grade"]) == grade for row in pair_labels)
            for grade in range(4)
        },
        "scientific_claim_eligible": (
            bool(group_labels)
            and fully_covered_groups == len(group_labels)
            and len({tuple(task["active_moieties"]) for task in tasks.values()}) >= 3
            and len({task["split"] for task in tasks.values()}) == 3
        ),
        "output_checksums": outputs,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(output_directory / "manifest.json", manifest)
    return manifest


def import_locked_validation_results(
    *,
    benchmark_directory: Path,
    registration: Path,
    output_path: Path,
) -> dict[str, Any]:
    protocol, model = _load_registration(registration.resolve())
    candidate_id = str(model["model_version_id"])
    truth_rows = list(_iter_jsonl(benchmark_directory / "benchmark_truth.jsonl"))
    relevant_truth = [
        row for row in truth_rows if str(row["source_candidate_id"]) == candidate_id
    ]
    metrics_path = registration / "provenance" / "verification_dataset_metrics.jsonl"
    audit_path = registration / "provenance" / "snapshot_reuse_audit.json"
    metrics = {
        str(row["dataset_name"]): row for row in _iter_jsonl(metrics_path)
    }
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    engineering = audit.get("engineering_checks") or {}
    results = []
    for truth in relevant_truth:
        metric = metrics.get(str(truth["dataset_name"]))
        if metric is None:
            raise ValueError(f"Locked metric is missing for {truth['dataset_name']}")
        predicted = metric.get("positive_predicted_mol_per_litre") or []
        observed = metric.get("positive_observed_mol_per_litre") or []
        if len(predicted) != len(observed) or len(predicted) != len(truth["time_values"]):
            raise ValueError(
                f"Locked prediction alignment differs for {truth['dataset_name']}"
            )
        results.append(
            {
                "task_id": truth["task_id"],
                "candidate_id": candidate_id,
                "result_origin": "locked_historical_validation_import",
                "compile_success": bool(engineering.get("all_artifacts_verified")),
                "execution_success": bool(engineering.get("all_simulations_passed")),
                "dimensional_check_passed": bool(metric.get("dataset_guard_passed")),
                "reproducibility_passed": bool(engineering.get("csv_repeatability_passed")),
                "mass_balance_passed": None,
                "predicted_time_values": truth["time_values"],
                "predicted_time_unit": truth["time_unit"],
                "predicted_values": predicted,
                "predicted_unit": "mol/l",
                "adaptation_operations": [],
                "model_structure_changed": False,
                "compound_parameters_changed": False,
                "execution_manifest_sha256": audit.get("windows_manifest_sha256"),
                "source_audit_sha256": _sha256_file(audit_path),
                "source_metrics_sha256": _sha256_file(metrics_path),
            }
        )
    checksum = _atomic_jsonl(output_path, results)
    return {
        "result_origin": "locked_historical_validation_import",
        "candidate_id": candidate_id,
        "result_count": len(results),
        "output_path": str(output_path.resolve()),
        "output_sha256": checksum,
        "new_execution_performed": False,
        "scientific_claim_eligible": False,
        "purpose": "Validate task alignment and automatic-label implementation only.",
        "protocol_sha256": _sha256_file(
            registration / "provenance" / "snapshot_reuse_protocol.json"
        ),
    }


AUDIT_DERIVED_BUILD_VERSION = "pbpk-audit-derived-execution-benchmark-1.2.0"

_TRANSPORTER_TOKENS = (
    "oatp",
    "oat",
    "oct",
    "mate",
    "p-gp",
    "pgp",
    "bcrp",
    "mrp",
    "ntcp",
    "asbt",
    "pmat",
    "pept",
    "ent",
    "cnt",
    "lat",
)
_ENZYME_TOKENS = (
    "cyp",
    "ugt",
    "fmo",
    "mao",
    "ces",
    "aldh",
    "adh",
    "sult",
    "nat",
    "aadac",
    "dpd",
    "xanthine oxidase",
    "aldehyde oxidase",
)


def _identity_token(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _primary_compound(model_name: str, compounds: list[str]) -> str:
    if not compounds:
        raise ValueError(f"Model {model_name!r} has no compounds.")
    model_token = _identity_token(
        re.sub(r"(?:[-_ ]model)$", "", model_name, flags=re.IGNORECASE)
    )
    exact = [value for value in compounds if _identity_token(value) == model_token]
    if len(exact) == 1:
        return exact[0]
    if len(compounds) == 1:
        return compounds[0]
    contained = [
        value
        for value in compounds
        if model_token
        and (
            model_token in _identity_token(value)
            or _identity_token(value) in model_token
        )
    ]
    if len(contained) == 1:
        return contained[0]
    raise ValueError(
        f"Primary compound is ambiguous for {model_name!r}: {compounds!r}."
    )


def _all_protocol_series(simulation: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    first = simulation.get("evaluation_series")
    if isinstance(first, dict):
        output.append(first)
    output.extend(
        item
        for item in simulation.get("additional_evaluation_series") or []
        if isinstance(item, dict)
    )
    return output


_STRUCTURED_ROUTE = {
    "Oral": "oral",
    "Intravenous": "intravenous",
    "IntravenousBolus": "intravenous",
}
_STRUCTURED_FREQUENCY = {
    "Single": "single_dose",
    "DI_24": "once_daily",
    "DI_12_12": "twice_daily",
    "DI_8_8_8": "three_times_daily",
    "DI_6_6_12": "three_times_daily_variable_interval",
}


def _named_snapshot_row(rows: object, name: str, kind: str) -> dict[str, Any]:
    matches = [
        row
        for row in rows or []
        if isinstance(row, dict) and str(row.get("Name") or "") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {kind} named {name!r}; found {len(matches)}.")
    return matches[0]


def _structured_simulation_requirement(
    *, snapshot: Mapping[str, Any], simulation_name: str, primary_compound: str
) -> dict[str, Any]:
    simulation = _named_snapshot_row(
        snapshot.get("Simulations"), simulation_name, "simulation"
    )
    compound_rows = [
        row for row in simulation.get("Compounds") or [] if isinstance(row, dict)
    ]
    primary_rows = [
        row
        for row in compound_rows
        if _identity_token(str(row.get("Name") or ""))
        == _identity_token(primary_compound)
    ]
    if len(primary_rows) != 1:
        raise ValueError(
            f"Simulation {simulation_name!r} does not contain one primary "
            f"compound row for {primary_compound!r}."
        )
    regimen = _structured_compound_regimen(
        snapshot=snapshot,
        simulation=simulation,
        compound_row=primary_rows[0],
    )
    regimen["co_compounds"] = sorted(
        {
            str(row.get("Name") or "")
            for row in compound_rows
            if _identity_token(str(row.get("Name") or ""))
            != _identity_token(primary_compound)
        },
        key=str.casefold,
    )
    regimen["scenario_source"] = "structured_pksim_snapshot"
    return regimen


def _structured_compound_regimen(
    *,
    snapshot: Mapping[str, Any],
    simulation: Mapping[str, Any],
    compound_row: Mapping[str, Any],
) -> dict[str, Any]:
    compound_name = str(compound_row.get("Name") or "").strip()
    if not compound_name:
        raise ValueError("Simulation compound has no name.")
    protocol_reference = compound_row.get("Protocol") or {}
    protocol_name = str(protocol_reference.get("Name") or "")
    if not protocol_name:
        raise ValueError(
            f"Simulation compound {compound_name!r} has no dosing protocol."
        )
    protocol = _named_snapshot_row(
        snapshot.get("Protocols"), protocol_name, "protocol"
    )

    def parameter_index(rows: object) -> dict[str, Mapping[str, Any]]:
        return {
            str(row.get("Name") or ""): row
            for row in rows or []
            if isinstance(row, Mapping)
        }

    def event_from_parameters(
        *,
        application_type: str,
        parameters: object,
        schema_name: str | None,
        schema_start_value: float | None = None,
        schema_start_unit: str | None = None,
        repetition_count: int | None = None,
        repetition_interval_value: float | None = None,
        repetition_interval_unit: str | None = None,
    ) -> dict[str, Any]:
        route_value = _STRUCTURED_ROUTE.get(application_type)
        if route_value is None:
            raise ValueError(
                f"Unsupported application type: {application_type!r}."
            )
        indexed = parameter_index(parameters)
        dose = indexed.get("InputDose")
        if dose is None:
            raise ValueError(
                f"Protocol {protocol_name!r} contains a dosing event without InputDose."
            )
        dose_value = float(dose["Value"])
        dose_unit = str(dose.get("Unit") or "")
        if not math.isfinite(dose_value) or dose_value <= 0 or not dose_unit:
            raise ValueError(
                f"Invalid structured dose: {dose_value!r} {dose_unit!r}."
            )
        item_start = indexed.get("Start time")
        item_start_value = (
            float(item_start["Value"])
            if item_start is not None and item_start.get("Value") is not None
            else 0.0
        )
        item_start_unit = (
            str(item_start.get("Unit") or "") or schema_start_unit
            if item_start is not None
            else schema_start_unit
        )
        start_value = (
            item_start_value + schema_start_value
            if schema_start_value is not None
            and (not item_start_unit or item_start_unit == schema_start_unit)
            else item_start_value
        )
        infusion = indexed.get("Infusion time")
        return {
            "schema_name": schema_name,
            "application_type": application_type,
            "route": route_value,
            "dose_value": dose_value,
            "dose_unit": dose_unit,
            "start_time_value": start_value,
            "start_time_unit": item_start_unit,
            "infusion_duration_value": (
                float(infusion["Value"])
                if infusion is not None and infusion.get("Value") is not None
                else None
            ),
            "infusion_duration_unit": (
                str(infusion.get("Unit") or "") or None
                if infusion is not None
                else None
            ),
            "repetition_count": repetition_count,
            "repetition_interval_value": repetition_interval_value,
            "repetition_interval_unit": repetition_interval_unit,
        }

    events: list[dict[str, Any]] = []
    application_type = str(protocol.get("ApplicationType") or "")
    if application_type:
        events.append(
            event_from_parameters(
                application_type=application_type,
                parameters=protocol.get("Parameters"),
                schema_name=None,
            )
        )
    else:
        for schema in protocol.get("Schemas") or []:
            if not isinstance(schema, Mapping):
                continue
            schema_parameters = parameter_index(schema.get("Parameters"))
            schema_start = schema_parameters.get("Start time")
            repetitions = schema_parameters.get("NumberOfRepetitions")
            interval_parameter = schema_parameters.get("TimeBetweenRepetitions")
            for item in schema.get("SchemaItems") or []:
                if not isinstance(item, Mapping):
                    continue
                events.append(
                    event_from_parameters(
                        application_type=str(item.get("ApplicationType") or ""),
                        parameters=item.get("Parameters"),
                        schema_name=str(schema.get("Name") or "") or None,
                        schema_start_value=(
                            float(schema_start["Value"])
                            if schema_start is not None
                            and schema_start.get("Value") is not None
                            else None
                        ),
                        schema_start_unit=(
                            str(schema_start.get("Unit") or "") or None
                            if schema_start is not None
                            else None
                        ),
                        repetition_count=(
                            int(repetitions["Value"])
                            if repetitions is not None
                            and repetitions.get("Value") is not None
                            else None
                        ),
                        repetition_interval_value=(
                            float(interval_parameter["Value"])
                            if interval_parameter is not None
                            and interval_parameter.get("Value") is not None
                            else None
                        ),
                        repetition_interval_unit=(
                            str(interval_parameter.get("Unit") or "") or None
                            if interval_parameter is not None
                            else None
                        ),
                    )
                )
    if not events:
        raise ValueError(f"Protocol {protocol_name!r} has no structured dosing event.")

    normalized_units = {
        str(event["dose_unit"])
        .replace("μ", "u")
        .replace("µ", "u")
        .casefold(): str(event["dose_unit"])
        for event in events
    }
    mass_factors = {"g": 1000.0, "mg": 1.0, "ug": 0.001}
    normalized_keys = set(normalized_units)
    per_kg = all(unit.endswith("/kg") for unit in normalized_keys)
    bases = {unit.removesuffix("/kg") for unit in normalized_keys}
    if bases and bases.issubset(mass_factors) and (
        per_kg or all(not unit.endswith("/kg") for unit in normalized_keys)
    ):
        dose_value = sum(
            float(event["dose_value"])
            * mass_factors[
                str(event["dose_unit"])
                .replace("μ", "u")
                .replace("µ", "u")
                .casefold()
                .removesuffix("/kg")
            ]
            for event in events
        )
        dose_unit = "mg/kg" if per_kg else "mg"
    elif len(normalized_keys) == 1:
        dose_value = sum(float(event["dose_value"]) for event in events)
        dose_unit = str(events[0]["dose_unit"])
    else:
        raise ValueError(
            f"Protocol {protocol_name!r} mixes incompatible dose units: "
            f"{sorted(normalized_units.values())}."
        )
    routes = sorted({str(event["route"]) for event in events})
    application_types = sorted(
        {str(event["application_type"]) for event in events}
    )
    route = routes[0] if len(routes) == 1 else "mixed"
    application_type = (
        application_types[0] if len(application_types) == 1 else "Mixed"
    )
    interval = str(protocol.get("DosingInterval") or "")
    frequency = _STRUCTURED_FREQUENCY.get(interval, "custom_schedule")
    if len(events) > 1:
        frequency += "_multi_event"
    formulations = sorted(
        {
            str(row.get("Name") or "").strip()
            for row in protocol_reference.get("Formulations") or []
            if isinstance(row, dict) and str(row.get("Name") or "").strip()
        },
        key=str.casefold,
    )
    individual_name = str(simulation.get("Individual") or "")
    population_name = str(simulation.get("Population") or "")
    if individual_name:
        individual = _named_snapshot_row(
            snapshot.get("Individuals"), individual_name, "individual"
        )
        origin = individual.get("OriginData") or {}
    elif population_name:
        population = _named_snapshot_row(
            snapshot.get("Populations"), population_name, "population"
        )
        population_individual = (
            (population.get("Settings") or {}).get("Individual") or {}
        )
        origin = population_individual.get("OriginData") or {}
    else:
        origin = {}
    parameters = {
        str(row.get("Name") or ""): row
        for row in protocol.get("Parameters") or []
        if isinstance(row, dict)
    }

    def optional_parameter(name: str) -> tuple[float | None, str | None]:
        row = parameters.get(name)
        if row is None or row.get("Value") is None:
            return None, None
        return float(row["Value"]), str(row.get("Unit") or "") or None

    start_value, start_unit = optional_parameter("Start time")
    end_value, end_unit = optional_parameter("End time")
    infusion_value, infusion_unit = optional_parameter("Infusion time")
    if start_value is None:
        event_starts = [
            event
            for event in events
            if event.get("start_time_value") is not None
        ]
        start_units = {event.get("start_time_unit") for event in event_starts}
        if event_starts and len(start_units) == 1:
            start_value = min(
                float(event["start_time_value"]) for event in event_starts
            )
            start_unit = event_starts[0].get("start_time_unit")
    if infusion_value is None:
        infusions = [
            event
            for event in events
            if event.get("infusion_duration_value") is not None
        ]
        infusion_signatures = {
            (
                float(event["infusion_duration_value"]),
                event.get("infusion_duration_unit"),
            )
            for event in infusions
        }
        if len(infusion_signatures) == 1:
            infusion_value, infusion_unit = next(iter(infusion_signatures))
    return {
        "compound": compound_name,
        "route": route,
        "application_type": application_type,
        "dose_value": dose_value,
        "dose_unit": dose_unit,
        "dose_basis": "per_kg" if dose_unit.casefold().endswith("/kg") else "absolute",
        "frequency": frequency,
        "dosing_interval": interval,
        "schedule_kind": "simple" if len(events) == 1 else "multi_event",
        "dosing_events": events,
        "formulation": formulations[0] if len(formulations) == 1 else None,
        "formulations": formulations,
        "species": str(origin.get("Species") or "") or None,
        "population": str(origin.get("Population") or "") or None,
        "population_name": population_name or None,
        "sex": str(origin.get("Gender") or "").casefold() or None,
        "start_time_value": start_value,
        "start_time_unit": start_unit,
        "end_time_value": end_value,
        "end_time_unit": end_unit,
        "infusion_duration_value": infusion_value,
        "infusion_duration_unit": infusion_unit,
    }


def _series_compound(
    *, compounds: list[str], series: Mapping[str, Any]
) -> str:
    path_segments = {
        _identity_token(segment)
        for segment in str(series.get("output_path") or "").split("|")
        if segment.strip()
    }
    matches = [
        compound
        for compound in compounds
        if _identity_token(compound) in path_segments
    ]
    if len(matches) != 1:
        raise ValueError(
            "Observed output must identify exactly one declared compound: "
            f"path={series.get('output_path')!r}, matches={matches!r}."
        )
    return matches[0]


def _structured_series_requirement(
    *,
    snapshot: Mapping[str, Any],
    simulation_name: str,
    observed_compound: str,
    declared_compounds: list[str],
) -> dict[str, Any]:
    simulation = _named_snapshot_row(
        snapshot.get("Simulations"), simulation_name, "simulation"
    )
    declared_tokens = {_identity_token(value) for value in declared_compounds}
    regimens = []
    for compound_row in simulation.get("Compounds") or []:
        if not isinstance(compound_row, Mapping):
            continue
        compound_name = str(compound_row.get("Name") or "").strip()
        protocol_reference = compound_row.get("Protocol") or {}
        if (
            _identity_token(compound_name) not in declared_tokens
            or not str(protocol_reference.get("Name") or "").strip()
        ):
            continue
        regimens.append(
            _structured_compound_regimen(
                snapshot=snapshot,
                simulation=simulation,
                compound_row=compound_row,
            )
        )
    regimens.sort(key=lambda row: str(row["compound"]).casefold())
    if not regimens:
        raise ValueError(
            f"Simulation {simulation_name!r} has no structured dosing regimen."
        )
    observed_regimens = [
        row
        for row in regimens
        if _identity_token(str(row["compound"]))
        == _identity_token(observed_compound)
    ]
    if len(observed_regimens) == 1:
        primary = observed_regimens[0]
        primary_role = "observed_compound_regimen"
    elif len(regimens) == 1:
        primary = regimens[0]
        primary_role = "sole_administered_compound_regimen"
    else:
        primary = regimens[0]
        primary_role = "multi_regimen_compatibility_projection"
    administered = [str(row["compound"]) for row in regimens]
    return {
        **primary,
        "observed_compound": observed_compound,
        "administered_compounds": administered,
        "co_compounds": [
            value
            for value in administered
            if _identity_token(value) != _identity_token(observed_compound)
        ],
        "regimens": regimens,
        "primary_regimen_role": primary_role,
        "scenario_source": "structured_pksim_snapshot",
    }


def _output_quantity_path(simulation_name: str, output_path: str) -> str:
    prefix = simulation_name.strip().casefold() + "|"
    normalized = output_path.strip().casefold()
    return normalized[len(prefix) :] if normalized.startswith(prefix) else normalized


def _series_signature(
    *,
    observed_compound: str,
    simulation_name: str,
    series: Mapping[str, Any],
) -> str:
    return _stable_id(
        "observed-series",
        observed_compound.casefold(),
        " ".join(str(series.get("dataset_name") or "").casefold().split()),
        str(series.get("reference") or "").strip().casefold(),
        _output_quantity_path(
            simulation_name, str(series.get("output_path") or "")
        ),
        [float(value) for value in series.get("time_values") or []],
        str(series.get("time_unit") or "").casefold(),
        [float(value) for value in series.get("observed_values") or []],
        str(series.get("observed_unit") or "").casefold(),
        float(series.get("molecular_weight_g_per_mol") or 0),
    )


def _candidate_features_from_snapshot(
    *,
    snapshot: Mapping[str, Any],
    compounds: list[str],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    compound_rows = [
        row for row in snapshot.get("Compounds") or [] if isinstance(row, dict)
    ]
    selected_rows = [
        row
        for row in compound_rows
        if _identity_token(str(row.get("Name") or ""))
        in {_identity_token(value) for value in compounds}
    ]
    if len(selected_rows) != len(compounds):
        raise ValueError(
            f"Expected {len(compounds)} declared compound rows; "
            f"found {len(selected_rows)}."
        )
    enzymes: set[str] = set()
    transporters: set[str] = set()
    process_types: set[str] = {"distribution"}
    for compound_row in selected_rows:
        for process in compound_row.get("Processes") or []:
            if not isinstance(process, dict):
                continue
            molecule = str(process.get("MoleculeName") or "").strip()
            lowered = molecule.casefold()
            if molecule and any(token in lowered for token in _TRANSPORTER_TOKENS):
                transporters.add(molecule)
                process_types.add("transport")
            elif molecule and any(token in lowered for token in _ENZYME_TOKENS):
                enzymes.add(molecule)
                process_types.add("metabolism")
            systemic = str(process.get("SystemicProcessType") or "").casefold()
            name = str(process.get("Name") or "").casefold()
            if "gfr" in systemic or "renal" in systemic or "glomerular" in name:
                process_types.add("renal excretion")
    validated_scenarios = []
    for item in protocol.get("simulations") or []:
        if item.get("role") != "model_verification":
            continue
        simulation = _named_snapshot_row(
            snapshot.get("Simulations"),
            str(item.get("name") or ""),
            "simulation",
        )
        for compound_row in simulation.get("Compounds") or []:
            if not isinstance(compound_row, Mapping):
                continue
            name = str(compound_row.get("Name") or "")
            protocol_reference = compound_row.get("Protocol") or {}
            if (
                _identity_token(name)
                not in {_identity_token(value) for value in compounds}
                or not str(protocol_reference.get("Name") or "").strip()
            ):
                continue
            regimen = _structured_compound_regimen(
                snapshot=snapshot,
                simulation=simulation,
                compound_row=compound_row,
            )
            regimen["scenario_source"] = "structured_pksim_snapshot"
            validated_scenarios.append(regimen)
    routes = {row["route"] for row in validated_scenarios}
    if "oral" in routes:
        process_types.add("absorption")
    formulations = {
        str(item.get("Name") or "").strip()
        for item in snapshot.get("Formulations") or []
        if isinstance(item, dict) and str(item.get("Name") or "").strip()
    }
    outputs = {
        str(value).strip()
        for simulation in snapshot.get("Simulations") or []
        if isinstance(simulation, dict)
        for value in simulation.get("OutputSelections") or []
        if str(value).strip()
    }
    return {
        "species": ["human"],
        "routes": sorted(routes),
        "enzymes": sorted(enzymes, key=str.casefold),
        "transporters": sorted(transporters, key=str.casefold),
        "process_types": sorted(process_types),
        "populations": ["human"],
        "formulations": sorted(formulations, key=str.casefold),
        "compartments": [],
        "output_names": sorted(outputs, key=str.casefold),
        "validated_scenarios": sorted(
            validated_scenarios,
            key=lambda row: (
                str(row["compound"]),
                str(row["route"]),
                str(row["frequency"]),
                str(row["dose_unit"]),
                float(row["dose_value"]),
            ),
        ),
    }


def _read_audit_index(audit_directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = audit_directory / "summary.json"
    index_path = audit_directory / "audit_index.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    expected = (summary.get("output_checksums") or {}).get("audit_index.jsonl")
    if not expected or _sha256_file(index_path) != expected:
        raise ValueError("Execution audit index checksum drifted.")
    rows = list(_iter_jsonl(index_path))
    if any(row.get("audit_status") == "auditor_failed" for row in rows):
        raise ValueError(
            "Auditor failures cannot be converted into scientific candidate labels."
        )
    return summary, rows


def _read_linux_repeat_index(
    linux_repeat_directory: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary_path = linux_repeat_directory / "summary.json"
    protocol_path = linux_repeat_directory / "protocol_results.jsonl"
    simulation_path = linux_repeat_directory / "simulation_results.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    checksums = summary.get("output_checksums") or {}
    expected_protocol = checksums.get("protocol_results.jsonl")
    expected_simulation = checksums.get("simulation_results.jsonl")
    if not expected_protocol or _sha256_file(protocol_path) != expected_protocol:
        raise ValueError("Linux OSPSuite-R protocol index checksum drifted.")
    if not expected_simulation or _sha256_file(simulation_path) != expected_simulation:
        raise ValueError("Linux OSPSuite-R simulation index checksum drifted.")
    rows = list(_iter_jsonl(protocol_path))
    indexed = {str(row["protocol_key"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Linux OSPSuite-R protocol index contains duplicates.")
    simulations = list(_iter_jsonl(simulation_path))
    simulation_ids = {
        (str(row["protocol_key"]), int(row["sequence"])) for row in simulations
    }
    if len(simulation_ids) != len(simulations):
        raise ValueError("Linux OSPSuite-R simulation index contains duplicates.")
    allowed_statuses = {"passed", "executed_negative", "runner_failed"}
    if any(row.get("status") not in allowed_statuses for row in simulations):
        raise ValueError("Linux OSPSuite-R simulation index has non-terminal status.")
    if {str(row["protocol_key"]) for row in simulations} != set(indexed):
        raise ValueError("Linux OSPSuite-R protocol and simulation inventories differ.")
    counts = summary.get("counts") or {}
    expected_counts = {
        "protocol_count": len(rows),
        "simulation_count": len(simulations),
        "simulation_passed_count": sum(
            row["status"] == "passed" for row in simulations
        ),
        "executed_negative_count": sum(
            row["status"] == "executed_negative" for row in simulations
        ),
        "runner_failed_count": sum(
            row["status"] == "runner_failed" for row in simulations
        ),
    }
    if any(int(counts.get(name, -1)) != value for name, value in expected_counts.items()):
        raise ValueError("Linux OSPSuite-R summary counts do not match terminal indexes.")
    for protocol_key, protocol in indexed.items():
        protocol_simulations = [
            row
            for row in simulations
            if str(row["protocol_key"]) == protocol_key
        ]
        protocol_counts = {
            "simulation_count": len(protocol_simulations),
            "passed_count": sum(
                row["status"] == "passed" for row in protocol_simulations
            ),
            "executed_negative_count": sum(
                row["status"] == "executed_negative"
                for row in protocol_simulations
            ),
            "runner_failed_count": sum(
                row["status"] == "runner_failed" for row in protocol_simulations
            ),
        }
        if any(
            int(protocol.get(name, -1)) != value
            for name, value in protocol_counts.items()
        ):
            raise ValueError(
                f"Linux OSPSuite-R protocol counts drifted for {protocol_key}."
            )
    return summary, indexed


def _assign_compound_and_family_disjoint_splits(
    *,
    tasks: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(values: Iterable[str]) -> None:
        normalized = sorted(
            {
                " ".join(str(value).casefold().split())
                for value in values
                if str(value).strip()
            }
        )
        if not normalized:
            return
        root = find(normalized[0])
        for value in normalized[1:]:
            other = find(value)
            if root != other:
                parent[other] = root

    for candidate in candidates:
        union(candidate.get("active_moieties") or [])
    for task in tasks:
        union(
            task.get("required_model_compounds")
            or task.get("active_moieties")
            or []
        )

    components: dict[str, set[str]] = defaultdict(set)
    for compound in sorted(parent):
        components[find(compound)].add(compound)
    component_keys = {
        root: "|".join(sorted(compounds))
        for root, compounds in components.items()
    }
    initial_split_by_root = {
        root: _split(component_keys[root]) for root in components
    }
    split_by_root = dict(initial_split_by_root)
    reusable_roots = {
        find(
            sorted(
                " ".join(str(value).casefold().split())
                for value in candidate.get("active_moieties") or []
            )[0]
        )
        for candidate in candidates
        if candidate.get("reusable_candidate")
        and candidate.get("active_moieties")
    }
    component_task_counts: Counter[str] = Counter()
    for task in tasks:
        normalized = sorted(
            " ".join(str(value).casefold().split())
            for value in task.get("required_model_compounds")
            or task.get("active_moieties")
            or []
        )
        if normalized:
            component_task_counts[find(normalized[0])] += 1
    total_tasks = sum(component_task_counts.values())
    target_task_counts = {
        "train": total_tasks * 0.6,
        "validation": total_tasks * 0.2,
        "test": total_tasks * 0.2,
    }
    stratification_adjustments: list[dict[str, Any]] = []

    def reusable_counts() -> Counter[str]:
        return Counter(split_by_root[root] for root in reusable_roots)

    def task_counts() -> Counter[str]:
        values: Counter[str] = Counter()
        for root, count in component_task_counts.items():
            values[split_by_root[root]] += count
        return values

    for target_split in ("train", "validation", "test"):
        if reusable_counts()[target_split] > 0:
            continue
        donor_roots = [
            root
            for root in reusable_roots
            if reusable_counts()[split_by_root[root]] > 1
        ]
        if not donor_roots:
            continue

        def move_cost(root: str) -> tuple[float, str]:
            before = task_counts()
            source_split = split_by_root[root]
            count = component_task_counts[root]
            after = Counter(before)
            after[source_split] -= count
            after[target_split] += count
            deviation = sum(
                abs(after[split] - target_task_counts[split])
                for split in ("train", "validation", "test")
            )
            return deviation, component_keys[root]

        selected_root = min(donor_roots, key=move_cost)
        source_split = split_by_root[selected_root]
        split_by_root[selected_root] = target_split
        stratification_adjustments.append(
            {
                "component": component_keys[selected_root],
                "from_split": source_split,
                "to_split": target_split,
                "reason": "ensure_reusable_candidate_component_coverage",
                "task_count": component_task_counts[selected_root],
            }
        )

    component_split = {
        compound: split_by_root[root]
        for root, compounds in components.items()
        for compound in compounds
    }

    task_rows = []
    for task in tasks:
        compounds = [
            " ".join(str(value).casefold().split())
            for value in task.get("required_model_compounds")
            or task.get("active_moieties")
            or []
        ]
        splits = {component_split[value] for value in compounds}
        if len(splits) != 1:
            raise ValueError(
                f"Task compounds cross split components: {task['task_id']}."
            )
        task["split"] = next(iter(splits))
        task_rows.append(
            {
                "task_id": task["task_id"],
                "required_model_compounds": compounds,
                "split": task["split"],
            }
        )

    candidate_rows = []
    for candidate in candidates:
        compounds = [
            " ".join(str(value).casefold().split())
            for value in candidate.get("active_moieties") or []
        ]
        splits = {component_split[value] for value in compounds}
        if len(splits) != 1:
            raise ValueError(
                f"Candidate model family crosses splits: {candidate['candidate_id']}."
            )
        candidate_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "model_family": candidate.get("model_family"),
                "model_compounds": compounds,
                "reusable_candidate": bool(
                    candidate.get("reusable_candidate")
                ),
                "split": next(iter(splits)),
            }
        )

    compound_splits: dict[str, set[str]] = defaultdict(set)
    for row in task_rows:
        for compound in row["required_model_compounds"]:
            compound_splits[compound].add(str(row["split"]))
    family_splits: dict[str, set[str]] = defaultdict(set)
    for row in candidate_rows:
        family_splits[str(row["model_family"])].add(str(row["split"]))
    return {
        "audit_version": "compound-model-family-disjoint-split-audit-1.1.0",
        "assignment_policy": (
            "stable_hash_then_deterministic_reusable_component_stratification"
        ),
        "passed": (
            all(len(values) == 1 for values in compound_splits.values())
            and all(len(values) == 1 for values in family_splits.values())
        ),
        "component_count": len(components),
        "components": [
            {
                "compounds": sorted(compounds),
                "initial_split": initial_split_by_root[find(next(iter(compounds)))],
                "split": component_split[next(iter(compounds))],
            }
            for compounds in sorted(
                components.values(), key=lambda values: sorted(values)
            )
        ],
        "task_assignments": task_rows,
        "candidate_assignments": candidate_rows,
        "stratification_adjustments": stratification_adjustments,
        "reusable_candidate_split_counts": dict(
            sorted(reusable_counts().items())
        ),
        "compound_split_violations": {
            key: sorted(values)
            for key, values in compound_splits.items()
            if len(values) != 1
        },
        "model_family_split_violations": {
            key: sorted(values)
            for key, values in family_splits.items()
            if len(values) != 1
        },
    }


def build_execution_benchmark_from_audits(
    *,
    batch_directory: Path,
    audit_directory: Path,
    output_directory: Path,
    linux_repeat_directory: Path | None = None,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    batch_directory = batch_directory.resolve()
    audit_directory = audit_directory.resolve()
    jobs_path = batch_directory / "jobs.jsonl"
    batch_manifest = json.loads(
        (batch_directory / "batch_manifest.json").read_text(encoding="utf-8-sig")
    )
    if _sha256_file(jobs_path) != batch_manifest["jobs_sha256"]:
        raise ValueError("Windows batch jobs checksum drifted.")
    jobs = {str(row["protocol_key"]): row for row in _iter_jsonl(jobs_path)}
    audit_summary, audit_rows = _read_audit_index(audit_directory)
    if set(jobs) != {str(row["protocol_key"]) for row in audit_rows}:
        raise ValueError("Execution audit and Windows batch protocol inventories differ.")
    linux_summary: dict[str, Any] | None = None
    linux_rows: dict[str, dict[str, Any]] = {}
    if linux_repeat_directory is not None:
        linux_repeat_directory = linux_repeat_directory.resolve()
        linux_summary, linux_rows = _read_linux_repeat_index(
            linux_repeat_directory
        )
        if str(linux_summary.get("jobs_sha256")) != _sha256_file(jobs_path):
            raise ValueError("Linux repeat and Windows batch jobs differ.")
        if set(linux_rows) != set(jobs):
            raise ValueError(
                "Linux repeat and Windows batch protocol inventories differ."
            )

    contexts_by_semantic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for audit_row in audit_rows:
        key = str(audit_row["protocol_key"])
        job = jobs[key]
        protocol_path = batch_directory / str(job["protocol_relative_path"])
        snapshot_path = batch_directory / str(job["snapshot_relative_path"])
        if _sha256_file(protocol_path) != str(job["protocol_sha256"]):
            raise ValueError(f"Protocol checksum drifted for {key}.")
        if _sha256_file(snapshot_path) != str(job["snapshot_sha256"]):
            raise ValueError(f"Snapshot checksum drifted for {key}.")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        audit_path = audit_directory / str(audit_row["audit_relative_path"])
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if _sha256_file(audit_path) != str(audit_row["audit_report_sha256"]):
            raise ValueError(f"Audit report checksum drifted for {key}.")
        metrics_path = audit_path.parent / "verification_dataset_metrics.jsonl"
        metrics = list(_iter_jsonl(metrics_path))
        metric_index = {
            (
                int(row["simulation_sequence"]),
                str(row["dataset_name"]),
                str(row["source_location"]),
            ): row
            for row in metrics
        }
        compounds = [str(value) for value in job.get("compounds") or []]
        features = _candidate_features_from_snapshot(
            snapshot=snapshot,
            compounds=compounds,
            protocol=protocol,
        )
        series_map: dict[str, list[dict[str, Any] | None]] = defaultdict(list)
        series_rows: list[dict[str, Any]] = []
        for simulation in protocol.get("simulations") or []:
            if simulation.get("role") != "model_verification":
                continue
            for series in _all_protocol_series(simulation):
                observed_compound = _series_compound(
                    compounds=compounds,
                    series=series,
                )
                signature = _series_signature(
                    observed_compound=observed_compound,
                    simulation_name=str(simulation.get("name") or ""),
                    series=series,
                )
                metric = metric_index.get(
                    (
                        int(simulation["sequence"]),
                        str(series["dataset_name"]),
                        str(series["source_location"]),
                    )
                )
                series_map[signature].append(metric)
                series_rows.append(
                    {
                        "signature": signature,
                        "simulation": simulation,
                        "series": series,
                        "metric": metric,
                        "observed_compound": observed_compound,
                    }
                )
        semantic = str(protocol["source_snapshot_semantic_sha256"])
        contexts_by_semantic[semantic].append(
            {
                "key": key,
                "job": job,
                "protocol": protocol,
                "audit": audit,
                "audit_row": audit_row,
                "snapshot": snapshot,
                "compounds": compounds,
                "features": features,
                "series_map": series_map,
                "series_rows": series_rows,
            }
        )

    candidate_rows: list[dict[str, Any]] = []
    candidate_contexts: dict[str, dict[str, Any]] = {}
    for semantic, aliases in sorted(contexts_by_semantic.items()):
        aliases.sort(key=lambda item: str(item["job"].get("version_id") or ""))
        compound_values = {
            json.dumps(
                sorted(
                    (str(value).casefold() for value in item["compounds"]),
                )
            )
            for item in aliases
        }
        feature_values = {
            json.dumps(item["features"], sort_keys=True) for item in aliases
        }
        if len(compound_values) != 1 or len(feature_values) != 1:
            raise ValueError(
                f"Semantic snapshot aliases disagree for {semantic}."
            )
        model_compounds = json.loads(next(iter(compound_values)))
        candidate_id = f"model-version-semantic:{semantic}"
        windows_engineering = all(
            bool(item["audit"].get("engineering_accepted")) for item in aliases
        )
        windows_scientific = all(
            bool(item["audit"].get("scientific_accepted")) for item in aliases
        )
        linux_repeat_passed = bool(linux_rows) and all(
            linux_rows[item["key"]].get("status") == "passed"
            for item in aliases
        )
        linux_repeat_runner_failed = bool(linux_rows) and any(
            linux_rows[item["key"]].get("status") == "runner_failed"
            for item in aliases
        )
        linux_repeat_executed_negative = bool(linux_rows) and any(
            linux_rows[item["key"]].get("status") == "executed_negative"
            for item in aliases
        )
        # PK-Sim on Windows is the canonical execution platform. The Linux
        # OSPSuite-R repeat is an independent portability claim and must not
        # overwrite a valid Windows scientific label.
        engineering = windows_engineering
        scientific = windows_scientific
        cross_platform_engineering = windows_engineering and (
            linux_repeat_passed if linux_summary is not None else False
        )
        cross_platform_scientific = windows_scientific and (
            linux_repeat_passed if linux_summary is not None else False
        )
        run_grade = min(int(item["audit"].get("run_label_grade") or 0) for item in aliases)
        merged_series_map: dict[str, list[dict[str, Any] | None]] = defaultdict(list)
        for alias in aliases:
            for signature, values in alias["series_map"].items():
                merged_series_map[signature].extend(values)
        candidate = {
            "candidate_id": candidate_id,
            "model_family": str(aliases[0]["protocol"]["source"]["model_name"]),
            "active_moieties": model_compounds,
            "model_compounds": model_compounds,
            "co_compounds": model_compounds[1:],
            **aliases[0]["features"],
            "software": "PK-Sim",
            "software_version": str(
                aliases[0]["protocol"]["runtime"]["pksim_version"]
            ),
            "license": {
                "status": "allowed",
                "spdx_id": str(
                    aliases[0]["protocol"]["source"]["license_spdx_id"]
                ),
            },
            "static_parse_status": "parsed",
            "execution_probe_status": "passed" if engineering else "failed",
            "reusable_candidate": scientific,
            "validation_grade": (
                "external_study_validated" if scientific
                else "execution_verified" if engineering
                else "execution_failed"
            ),
            "run_label_grade": run_grade,
            "repository_commit": aliases[0]["job"].get("repository_commit"),
            "project_sha256": aliases[0]["audit"].get("roundtrip", {}).get(
                "roundtrip_semantic_sha256"
            ),
            "source_snapshot_semantic_sha256": semantic,
            "version_aliases": [
                {
                    "protocol_key": item["key"],
                    "version_id": item["job"].get("version_id"),
                    "repository_commit": item["job"].get("repository_commit"),
                    "audit_report_sha256": item["audit_row"].get(
                        "audit_report_sha256"
                    ),
                }
                for item in aliases
            ],
            "semantic_alias_count": len(aliases),
            "simulation_count": max(
                int(item["protocol"]["counts"]["simulations"]) for item in aliases
            ),
            "parameter_record_count": 0,
            "dataset_reference_count": len(merged_series_map),
            "validation_metric_count": sum(
                value is not None
                for values in merged_series_map.values()
                for value in values
            ),
            "engineering_accepted": engineering,
            "windows_engineering_accepted": windows_engineering,
            "scientific_accepted": scientific,
            "windows_scientific_accepted": windows_scientific,
            "linux_ospsuite_repeat_passed": linux_repeat_passed,
            "linux_ospsuite_repeat_runner_failed": linux_repeat_runner_failed,
            "linux_ospsuite_repeat_executed_negative": (
                linux_repeat_executed_negative
            ),
            "cross_platform_engineering_accepted": cross_platform_engineering,
            "cross_platform_scientific_accepted": cross_platform_scientific,
        }
        candidate_rows.append(candidate)
        candidate_contexts[candidate_id] = {
            "candidate": candidate,
            "aliases": aliases,
            "series_map": merged_series_map,
        }

    tasks_by_key: dict[str, dict[str, Any]] = {}
    truth_by_key: dict[str, dict[str, Any]] = {}
    for candidate_id, context in candidate_contexts.items():
        source_candidate = context["candidate"]
        for alias in context["aliases"]:
            protocol = alias["protocol"]
            for item in alias["series_rows"]:
                simulation = item["simulation"]
                series = item["series"]
                observed_compound = str(item["observed_compound"])
                signature = str(item["signature"])
                scenario = _structured_series_requirement(
                    snapshot=alias["snapshot"],
                    simulation_name=str(simulation.get("name") or ""),
                    observed_compound=observed_compound,
                    declared_compounds=alias["compounds"],
                )
                active_moieties = sorted(
                    {
                        str(value).casefold()
                        for value in scenario["administered_compounds"]
                    }
                )
                required_model_compounds = sorted(
                    set(active_moieties) | {observed_compound.casefold()}
                )
                request_identity = {
                    "active_moieties": active_moieties,
                    "observed_analytes": [observed_compound.casefold()],
                    "required_model_compounds": required_model_compounds,
                }
                task_key = _stable_id(
                    "execution-task-source",
                    request_identity,
                    signature,
                    scenario,
                )
                group_id = _stable_id(
                    "compound-validation-program", request_identity
                )
                task_id = _stable_id(
                    "execution-task",
                    AUDIT_DERIVED_BUILD_VERSION,
                    request_identity,
                    signature,
                    scenario,
                )
                if task_key not in tasks_by_key:
                    tasks_by_key[task_key] = {
                        "benchmark_version": BENCHMARK_VERSION,
                        "benchmark_build_version": AUDIT_DERIVED_BUILD_VERSION,
                        "benchmark_level": "execution_derived",
                        "task_id": task_id,
                        "validation_group_id": group_id,
                        "observed_series_signature": signature,
                        "task_type": "plasma_concentration_time_prediction",
                        **request_identity,
                        "species": "human",
                        "scenario": scenario,
                        "enzymes": source_candidate.get("enzymes") or [],
                        "transporters": source_candidate.get("transporters") or [],
                        "process_types": source_candidate.get("process_types") or [],
                        "required_outputs": [
                            "plasma_concentration_time",
                            "auc0_last",
                            "cmax",
                        ],
                        "split": None,
                        "hidden_truth_available": True,
                    }
                    truth_by_key[task_key] = {
                        "task_id": task_id,
                        "validation_group_id": group_id,
                        "observed_series_signature": signature,
                        "observed_compound": observed_compound.casefold(),
                        "administered_compounds": active_moieties,
                        "source_candidate_ids": [candidate_id],
                        "source_protocol_keys": [alias["key"]],
                        "source_simulation_sequence": int(simulation["sequence"]),
                        "dataset_name": series["dataset_name"],
                        "source_location": series["source_location"],
                        "reference": series.get("reference"),
                        "output_path": series["output_path"],
                        "time_values": series["time_values"],
                        "time_unit": series["time_unit"],
                        "observed_values": series["observed_values"],
                        "observed_unit": series["observed_unit"],
                        "molecular_weight_g_per_mol": series[
                            "molecular_weight_g_per_mol"
                        ],
                        "criteria": protocol["criteria"],
                    }
                else:
                    truth = truth_by_key[task_key]
                    if truth["criteria"] != protocol["criteria"]:
                        raise ValueError(
                            f"Duplicate hidden task has different criteria: {task_id}."
                        )
                    if candidate_id not in truth["source_candidate_ids"]:
                        truth["source_candidate_ids"].append(candidate_id)
                    if alias["key"] not in truth["source_protocol_keys"]:
                        truth["source_protocol_keys"].append(alias["key"])

    tasks = list(tasks_by_key.values())
    candidate_rows.sort(key=lambda row: str(row["candidate_id"]))
    leakage_audit = _assign_compound_and_family_disjoint_splits(
        tasks=tasks,
        candidates=candidate_rows,
    )
    if not leakage_audit["passed"]:
        raise ValueError("Compound/model-family leakage audit failed.")
    tasks.sort(key=lambda row: str(row["task_id"]))
    truth_rows = sorted(truth_by_key.values(), key=lambda row: str(row["task_id"]))
    source_by_task = {
        str(row["task_id"]): row for row in truth_rows
    }
    candidate_pairs: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        truth = source_by_task[task_id]
        signature = str(task["observed_series_signature"])
        task_mechanisms = set(task.get("enzymes") or []) | set(
            task.get("transporters") or []
        ) | set(task.get("process_types") or [])
        for candidate in candidate_rows:
            candidate_id = str(candidate["candidate_id"])
            context = candidate_contexts[candidate_id]
            required_compounds = set(task["required_model_compounds"])
            candidate_compounds = set(candidate["active_moieties"])
            same_active = bool(required_compounds) and required_compounds.issubset(
                candidate_compounds
            )
            exact_metrics = [
                value
                for value in context["series_map"].get(signature, [])
                if value is not None
            ]
            candidate_mechanisms = set(candidate.get("enzymes") or []) | set(
                candidate.get("transporters") or []
            ) | set(candidate.get("process_types") or [])
            mechanism_overlap = sorted(task_mechanisms & candidate_mechanisms)
            reasons: list[str] = []
            adaptation_required = False
            if not bool(candidate["engineering_accepted"]):
                grade = 0
                decision = "reject"
                origin = "hard_gate"
                reasons.append("CANDIDATE_ENGINEERING_AUDIT_FAILED")
            elif same_active and exact_metrics:
                exact_passed = bool(candidate["scientific_accepted"]) and all(
                    bool(metric.get("dataset_guard_passed"))
                    and bool(metric.get("endpoint_passed"))
                    for metric in exact_metrics
                )
                if exact_passed:
                    grade = 3
                    decision = "direct_reuse"
                else:
                    grade = 1
                    decision = "mechanistic_reference"
                    reasons.append("EXACT_SCENARIO_SCIENTIFIC_VALIDATION_FAILED")
                origin = "exact_executed_scenario"
            elif same_active:
                grade = 1
                decision = "mechanistic_reference"
                origin = "mechanistic_reference"
                adaptation_required = True
                reasons.append("CONTROLLED_ADAPTATION_EXECUTION_REQUIRED")
            elif mechanism_overlap:
                grade = 1
                decision = "mechanistic_reference"
                origin = "mechanistic_reference"
                reasons.append("ACTIVE_MOIETY_MISMATCH_NUMERIC_REUSE_FORBIDDEN")
            else:
                grade = 0
                decision = "reject"
                origin = "hard_gate"
                reasons.extend(
                    [
                        "ACTIVE_MOIETY_MISMATCH",
                        "NO_TRACEABLE_MECHANISM_OVERLAP",
                    ]
                )
            candidate_pairs.append(
                {
                    "task_id": task_id,
                    "validation_group_id": task["validation_group_id"],
                    "candidate_id": candidate_id,
                    "relevance_grade": grade,
                    "decision": decision,
                    "label_origin": origin,
                    "same_active_moiety": same_active,
                    "required_model_compounds": sorted(required_compounds),
                    "candidate_model_compounds": sorted(candidate_compounds),
                    "exact_executed_scenario": bool(exact_metrics),
                    "adaptation_required": adaptation_required,
                    "adaptation_operation_count": 0,
                    "mechanism_overlap": mechanism_overlap,
                    "reason_codes": reasons,
                    "source_audit_report_sha256s": [
                        alias["audit_row"]["audit_report_sha256"]
                        for alias in context["aliases"]
                    ],
                }
            )

    forbidden = {
        str(candidate["candidate_id"]).casefold() for candidate in candidate_rows
    }
    for task in tasks:
        serialized = json.dumps(task, ensure_ascii=True, sort_keys=True).casefold()
        if any(token in serialized for token in forbidden):
            raise ValueError(f"Source candidate identity leaked into task {task['task_id']}.")

    output_checksums = {
        "benchmark_tasks.jsonl": _atomic_jsonl(
            output_directory / "benchmark_tasks.jsonl", tasks
        ),
        "benchmark_truth.jsonl": _atomic_jsonl(
            output_directory / "benchmark_truth.jsonl", truth_rows
        ),
        "candidates.jsonl": _atomic_jsonl(
            output_directory / "candidates.jsonl", candidate_rows
        ),
        "candidate_pairs.jsonl": _atomic_jsonl(
            output_directory / "candidate_pairs.jsonl", candidate_pairs
        ),
        "leakage_audit.json": _atomic_json(
            output_directory / "leakage_audit.json", leakage_audit
        ),
    }
    os.chmod(output_directory / "benchmark_truth.jsonl", 0o600)
    split_counts = {
        split: sum(task["split"] == split for task in tasks)
        for split in ("train", "validation", "test")
    }
    same_compound_counts = {
        str(task["task_id"]): sum(
            set(task["required_model_compounds"]).issubset(
                set(candidate["active_moieties"])
            )
            for candidate in candidate_rows
        )
        for task in tasks
    }
    non_degenerate_compounds = {
        "+".join(task["required_model_compounds"])
        for task in tasks
        if same_compound_counts[str(task["task_id"])] >= 2
    }
    independent_compounds = sorted(
        {
            str(compound)
            for task in tasks
            for compound in task["required_model_compounds"]
        }
    )
    compound_count = len(independent_compounds)
    pairs_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_pairs:
        pairs_by_task[str(row["task_id"])].append(row)
    abstention_task_count = sum(
        not any(int(row["relevance_grade"]) >= 2 for row in rows)
        for rows in pairs_by_task.values()
    )
    abstention_fraction = (
        abstention_task_count / len(tasks) if tasks else 0.0
    )
    hard_negative_complete_count = sum(
        any(int(row["relevance_grade"]) == 0 for row in rows)
        for rows in pairs_by_task.values()
    )
    soft_negative_complete_count = sum(
        any(int(row["relevance_grade"]) == 1 for row in rows)
        for rows in pairs_by_task.values()
    )
    non_degenerate_task_count = sum(
        same_compound_counts[str(task["task_id"])] >= 2 for task in tasks
    )
    decision_grade_counts = {
        str(grade): sum(
            int(row["relevance_grade"]) == grade for row in candidate_pairs
        )
        for grade in range(4)
    }
    task_split_by_id = {
        str(task["task_id"]): str(task["split"]) for task in tasks
    }
    grade_3_split_counts = {
        split: sum(
            int(row["relevance_grade"]) == 3
            and task_split_by_id[str(row["task_id"])] == split
            for row in candidate_pairs
        )
        for split in ("train", "validation", "test")
    }
    blockers = []
    cross_platform_blockers = []
    if len(tasks) < 100:
        blockers.append("TASK_COUNT_LT_100")
    if compound_count < 30:
        blockers.append("INDEPENDENT_COMPOUND_COUNT_LT_30")
    if non_degenerate_task_count != len(tasks):
        blockers.append(
            "NON_DEGENERATE_SAME_COMPOUND_QUERY_COVERAGE_INCOMPLETE"
        )
    if hard_negative_complete_count != len(tasks):
        blockers.append("HARD_NEGATIVE_QUERY_COVERAGE_INCOMPLETE")
    if soft_negative_complete_count != len(tasks):
        blockers.append("DIFFICULT_SOFT_NEGATIVE_QUERY_COVERAGE_INCOMPLETE")
    if abstention_fraction < 0.20:
        blockers.append("ABSTENTION_TASK_FRACTION_LT_0_20")
    if any(row.get("adaptation_required") for row in candidate_pairs):
        blockers.append("CONTROLLED_ADAPTATION_EXECUTION_REQUIRED")
    if any(value == 0 for value in split_counts.values()):
        blockers.append("TRAIN_VALIDATION_TEST_SPLITS_NOT_ALL_PRESENT")
    if not any(row["relevance_grade"] == 3 for row in candidate_pairs):
        blockers.append("GRADE_3_EXECUTION_LABEL_COUNT_ZERO")
    if any(value == 0 for value in grade_3_split_counts.values()):
        blockers.append("GRADE_3_SPLIT_COVERAGE_INCOMPLETE")
    if linux_summary is None:
        cross_platform_blockers.append("LINUX_OSPSUITE_R_REPEAT_REQUIRED")
    else:
        linux_counts = linux_summary.get("counts") or {}
        if int(linux_counts.get("runner_failed_count") or 0) > 0:
            cross_platform_blockers.append(
                "LINUX_OSPSUITE_R_RUNNER_FAILURES_PRESENT"
            )
        if int(linux_counts.get("executed_negative_count") or 0) > 0:
            cross_platform_blockers.append(
                "LINUX_OSPSUITE_R_CROSS_PLATFORM_MISMATCH_PRESENT"
            )
    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_build_version": AUDIT_DERIVED_BUILD_VERSION,
        "benchmark_level": "execution_derived",
        "scientific_claim_eligible": not blockers,
        "scientific_claim_blockers": blockers,
        "cross_platform_claim_eligible": not cross_platform_blockers,
        "cross_platform_claim_blockers": cross_platform_blockers,
        "task_count": len(tasks),
        "candidate_count": len(candidate_rows),
        "candidate_pair_count": len(candidate_pairs),
        "compound_count": compound_count,
        "independent_compounds": independent_compounds,
        "semantic_alias_count": sum(
            int(candidate["semantic_alias_count"]) for candidate in candidate_rows
        ),
        "linux_ospsuite_repeat": (
            {
                "directory": str(linux_repeat_directory),
                "summary_sha256": _sha256_file(
                    linux_repeat_directory / "summary.json"
                ),
                "complete": bool(linux_summary.get("complete")),
                "terminal_inventory_complete": True,
                "execution_complete": bool(
                    linux_summary.get(
                        "execution_complete", linux_summary.get("complete")
                    )
                ),
                "cross_platform_all_passed": bool(
                    linux_summary.get("cross_platform_all_passed")
                ),
                "counts": linux_summary.get("counts") or {},
            }
            if linux_summary is not None and linux_repeat_directory is not None
            else None
        ),
        "non_degenerate_same_compound_groups": sorted(non_degenerate_compounds),
        "non_degenerate_same_compound_task_count": non_degenerate_task_count,
        "hard_negative_complete_task_count": hard_negative_complete_count,
        "soft_negative_complete_task_count": soft_negative_complete_count,
        "abstention_task_count": abstention_task_count,
        "abstention_task_fraction": abstention_fraction,
        "decision_grade_counts": decision_grade_counts,
        "grade_3_split_counts": grade_3_split_counts,
        "split_task_counts": split_counts,
        "answer_leakage_check_passed": True,
        "truth_file_mode": "0600",
        "batch_manifest_sha256": _sha256_file(
            batch_directory / "batch_manifest.json"
        ),
        "audit_summary_sha256": _sha256_file(audit_directory / "summary.json"),
        "audit_complete": bool(audit_summary.get("complete")),
        "output_checksums": output_checksums,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(output_directory / "manifest.json", manifest)
    return manifest


def build_automatic_labels_from_audits(
    *,
    benchmark_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    benchmark_manifest = json.loads(
        (benchmark_directory / "manifest.json").read_text(encoding="utf-8")
    )
    pair_path = benchmark_directory / "candidate_pairs.jsonl"
    if _sha256_file(pair_path) != benchmark_manifest["output_checksums"][
        "candidate_pairs.jsonl"
    ]:
        raise ValueError("Candidate pair checksum drifted.")
    pairs = list(_iter_jsonl(pair_path))
    tasks = {
        str(row["task_id"]): row
        for row in _iter_jsonl(benchmark_directory / "benchmark_tasks.jsonl")
    }
    candidates = list(_iter_jsonl(benchmark_directory / "candidates.jsonl"))
    expected_pair_count = len(tasks) * len(candidates)
    identities = {
        (str(row["task_id"]), str(row["candidate_id"])) for row in pairs
    }
    if len(pairs) != expected_pair_count or len(identities) != expected_pair_count:
        raise ValueError("Audit-derived candidate labels do not cover every pair once.")
    labels = [
        {
            "task_id": row["task_id"],
            "candidate_id": row["candidate_id"],
            "validation_group_id": row["validation_group_id"],
            "relevance_grade": int(row["relevance_grade"]),
            "decision": row["decision"],
            "label_origin": row["label_origin"],
            "scenario_scientific_passed": bool(
                row["relevance_grade"] in {2, 3}
            ),
            "reason_codes": row.get("reason_codes") or [],
            "adaptation_plan_id": row.get("adaptation_plan_id"),
            "adaptation_attempt_status": row.get(
                "adaptation_attempt_status"
            ),
            "adaptation_executed": bool(row.get("adaptation_executed")),
            "adaptation_scientific_passed": bool(
                row.get("adaptation_scientific_passed")
            ),
            "adaptation_operation_multilabel": row.get(
                "adaptation_operation_multilabel"
            )
            or [],
            "adaptation_operation_count": int(
                row.get("adaptation_operation_count") or 0
            ),
            "adaptation_weighted_cost": row.get(
                "adaptation_weighted_cost"
            ),
            "adaptation_runtime_seconds": row.get(
                "adaptation_runtime_seconds"
            ),
            "adaptation_failure_class": row.get(
                "adaptation_failure_class"
            ),
        }
        for row in pairs
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[(str(row["validation_group_id"]), str(row["candidate_id"]))].append(row)
    group_labels = []
    for (group_id, candidate_id), rows in sorted(grouped.items()):
        exact = [row for row in rows if row.get("exact_executed_scenario")]
        grades = [int(row["relevance_grade"]) for row in rows]
        if grades and all(grade == 3 for grade in grades):
            group_grade, decision = 3, "direct_reuse"
        elif any(grade == 2 for grade in grades):
            group_grade, decision = 2, "controlled_adaptation"
        elif any(grade == 1 for grade in grades):
            group_grade, decision = 1, "mechanistic_reference"
        else:
            group_grade, decision = 0, "reject"
        group_labels.append(
            {
                "validation_group_id": group_id,
                "candidate_id": candidate_id,
                "execution_coverage": len(exact) / len(rows) if rows else 0.0,
                "scenario_pass_fraction": (
                    sum(grade >= 2 for grade in grades) / len(grades)
                    if grades
                    else 0.0
                ),
                "adaptation_operation_count": sum(
                    int(row.get("adaptation_operation_count") or 0) for row in rows
                ),
                "adaptation_weighted_cost": sum(
                    float(row.get("adaptation_weighted_cost") or 0.0)
                    for row in rows
                    if row.get("adaptation_executed")
                ),
                "adaptation_executed_count": sum(
                    bool(row.get("adaptation_executed")) for row in rows
                ),
                "adaptation_scientific_passed_count": sum(
                    bool(row.get("adaptation_scientific_passed"))
                    for row in rows
                ),
                "relevance_grade": group_grade,
                "decision": decision,
                "label_origin_counts": {
                    origin: sum(row["label_origin"] == origin for row in rows)
                    for origin in sorted({str(row["label_origin"]) for row in rows})
                },
            }
        )
    outputs = {
        "candidate_group_labels.jsonl": _atomic_jsonl(
            output_directory / "candidate_group_labels.jsonl", group_labels
        ),
        "labels.jsonl": _atomic_jsonl(output_directory / "labels.jsonl", labels),
    }
    manifest = {
        "benchmark_version": benchmark_manifest["benchmark_version"],
        "benchmark_build_version": benchmark_manifest.get(
            "benchmark_build_version"
        ),
        "benchmark_level": "execution_derived",
        "candidate_group_count": len(group_labels),
        "scenario_result_count": len(labels),
        "label_counts": {
            str(grade): sum(row["relevance_grade"] == grade for row in labels)
            for grade in range(4)
        },
        "label_origin_counts": {
            origin: sum(row["label_origin"] == origin for row in labels)
            for origin in sorted({str(row["label_origin"]) for row in labels})
        },
        "adaptation_supervision_available": any(
            row["adaptation_executed"] for row in labels
        ),
        "adaptation_executed_count": sum(
            row["adaptation_executed"] for row in labels
        ),
        "adaptation_scientific_passed_count": sum(
            row["adaptation_scientific_passed"] for row in labels
        ),
        "fully_covered_candidate_group_count": sum(
            float(row["execution_coverage"]) == 1.0 for row in group_labels
        ),
        "scientific_claim_eligible": bool(
            benchmark_manifest.get("scientific_claim_eligible")
        ),
        "scientific_claim_blockers": benchmark_manifest.get(
            "scientific_claim_blockers"
        )
        or [],
        "source_candidate_pairs_sha256": _sha256_file(pair_path),
        "output_checksums": outputs,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(output_directory / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and label execution-derived PBPK recommendation benchmarks.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--registration", action="append", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    label = subparsers.add_parser("label")
    label.add_argument("--benchmark", type=Path, required=True)
    label.add_argument("--execution-results", type=Path, required=True)
    label.add_argument("--output", type=Path, required=True)
    import_locked = subparsers.add_parser("import-locked-validation")
    import_locked.add_argument("--benchmark", type=Path, required=True)
    import_locked.add_argument("--registration", type=Path, required=True)
    import_locked.add_argument("--output", type=Path, required=True)
    build_audited = subparsers.add_parser("build-from-audits")
    build_audited.add_argument("--batch", type=Path, required=True)
    build_audited.add_argument("--audits", type=Path, required=True)
    build_audited.add_argument("--linux-repeat", type=Path)
    build_audited.add_argument("--output", type=Path, required=True)
    label_audited = subparsers.add_parser("label-from-audits")
    label_audited.add_argument("--benchmark", type=Path, required=True)
    label_audited.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_execution_benchmark(registrations=args.registration, output_directory=args.output)
    elif args.command == "label":
        result = build_automatic_labels(
            benchmark_directory=args.benchmark,
            execution_results_path=args.execution_results,
            output_directory=args.output,
        )
    elif args.command == "import-locked-validation":
        result = import_locked_validation_results(
            benchmark_directory=args.benchmark,
            registration=args.registration,
            output_path=args.output,
        )
    elif args.command == "build-from-audits":
        result = build_execution_benchmark_from_audits(
            batch_directory=args.batch,
            audit_directory=args.audits,
            output_directory=args.output,
            linux_repeat_directory=args.linux_repeat,
        )
    else:
        result = build_automatic_labels_from_audits(
            benchmark_directory=args.benchmark,
            output_directory=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
