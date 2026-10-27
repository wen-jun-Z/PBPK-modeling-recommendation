from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Iterable, Mapping

from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.licensed_snapshot_reuse import (
    LicensedSnapshotReuseProtocol,
    snapshot_semantic_sha256,
)
from pbpk_kb.recommendation.model_matching.adaptation_planner import (
    AdaptationPlannerError,
    _atomic_json,
    _atomic_jsonl,
    _convert_dose,
    _find_named,
    _identity_token,
    _load_locked_json,
    _parameter,
    _protocol_descriptor,
    _quantity_suffix,
    _read_jsonl,
    _rewrite_protocol_paths,
    _stable_hex,
)


PERTURBATION_PLANNER_VERSION = "pbpk-scenario-perturbation-planner-1.3.0"
PERTURBATION_PROTOCOL_SCHEMA_VERSION = "1.0.0"


class PerturbationError(RuntimeError):
    pass


_AUDITED_CONTINUOUS_VARIABLES = (
    "dose",
    "age",
    "body_weight",
    "organ_function",
    "administration_time",
)


def _canonical_decimal_float(value: float, significant_digits: int = 12) -> float:
    if not math.isfinite(value):
        raise PerturbationError("Perturbation values must be finite.")
    decimal_value = Decimal(str(value))
    if decimal_value.is_zero():
        return 0.0
    quantum = Decimal(1).scaleb(decimal_value.adjusted() - significant_digits + 1)
    return float(decimal_value.quantize(quantum, rounding=ROUND_HALF_EVEN))


def _variable_eligibility(
    *,
    snapshot: Mapping[str, Any],
    simulation: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    perturbation_group_id: str,
    active_moiety: str,
    perturbations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    protocols = list(snapshot.get("Protocols") or [])
    protocol_index = next(
        index
        for index, row in enumerate(protocols)
        if row is descriptor["protocol"]
    )
    dose = _parameter(descriptor["protocol"], "InputDose")
    if dose is None:
        raise PerturbationError("Dose eligibility audit lost the InputDose parameter.")
    dose_index = list(descriptor["protocol"].get("Parameters") or []).index(dose)
    rows = [
        {
            "perturbation_group_id": perturbation_group_id,
            "active_moiety": active_moiety,
            "variable": "dose",
            "status": "planned",
            "structured_path": f"/Protocols/{protocol_index}/Parameters/{dose_index}/Value",
            "unit": descriptor["dose_unit"],
            "generated_perturbation_count": len(perturbations),
            "reason_code": None,
        }
    ]
    individual_name = str(simulation.get("Individual") or "").strip()
    individual = _find_named(
        snapshot.get("Individuals"), individual_name, kind="individual"
    )
    individuals = list(snapshot.get("Individuals") or [])
    individual_index = individuals.index(individual)
    origin = individual.get("OriginData") or {}
    for variable, key in (("age", "Age"), ("body_weight", "Weight")):
        quantity = origin.get(key)
        has_quantity = (
            isinstance(quantity, dict)
            and quantity.get("Value") is not None
            and str(quantity.get("Unit") or "").strip()
        )
        rows.append(
            {
                "perturbation_group_id": perturbation_group_id,
                "active_moiety": active_moiety,
                "variable": variable,
                "status": "blocked",
                "structured_path": (
                    f"/Individuals/{individual_index}/OriginData/{key}/Value"
                    if has_quantity
                    else None
                ),
                "unit": quantity.get("Unit") if has_quantity else None,
                "generated_perturbation_count": 0,
                "reason_code": (
                    "PUBLIC_VALIDATION_BOUNDARY_UNAVAILABLE"
                    if has_quantity
                    else "UNIQUE_STRUCTURED_PARAMETER_PATH_UNAVAILABLE"
                ),
            }
        )
    start_time = _parameter(descriptor["protocol"], "Start time")
    start_time_index = (
        list(descriptor["protocol"].get("Parameters") or []).index(start_time)
        if start_time is not None
        else None
    )
    rows.extend(
        (
            {
                "perturbation_group_id": perturbation_group_id,
                "active_moiety": active_moiety,
                "variable": "organ_function",
                "status": "blocked",
                "structured_path": None,
                "unit": None,
                "generated_perturbation_count": 0,
                "reason_code": "UNIQUE_STRUCTURED_PARAMETER_PATH_UNAVAILABLE",
            },
            {
                "perturbation_group_id": perturbation_group_id,
                "active_moiety": active_moiety,
                "variable": "administration_time",
                "status": "blocked",
                "structured_path": (
                    f"/Protocols/{protocol_index}/Parameters/{start_time_index}/Value"
                    if start_time_index is not None
                    else None
                ),
                "unit": start_time.get("Unit") if start_time is not None else None,
                "generated_perturbation_count": 0,
                "reason_code": (
                    "PUBLIC_VALIDATION_BOUNDARY_UNAVAILABLE"
                    if start_time is not None
                    else "UNIQUE_STRUCTURED_PARAMETER_PATH_UNAVAILABLE"
                ),
            },
        )
    )
    return rows


def _protocol_counts(snapshot: Mapping[str, Any], execution_count: int) -> dict[str, int]:
    return {
        "compounds": len(snapshot.get("Compounds") or []),
        "expression_profiles": len(snapshot.get("ExpressionProfiles") or []),
        "formulations": len(snapshot.get("Formulations") or []),
        "individuals": len(snapshot.get("Individuals") or []),
        "protocols": len(snapshot.get("Protocols") or []),
        "simulations": 1 + execution_count,
        "observed_data": len(snapshot.get("ObservedData") or []),
        "simulation_classifications": 0,
        "model_building": 0,
        "model_verification": 1,
        "execution_only": execution_count,
    }


def _absolute_delta_in_unit(unit: str) -> float:
    normalized = unit.strip().casefold().replace("μ", "µ")
    per_kg = normalized.endswith("/kg")
    target = "mg/kg" if per_kg else "mg"
    delta = 0.001 if per_kg else 0.1
    return _convert_dose(delta, target, unit)


def _compatible_doses(
    *,
    snapshot: Mapping[str, Any],
    protocol: Mapping[str, Any],
    primary: str,
    base: Mapping[str, Any],
) -> list[float]:
    frozen_names = {
        str(row["name"])
        for row in protocol.get("simulations") or []
        if row.get("role") == "model_verification"
    }
    doses: list[float] = []
    for simulation in snapshot.get("Simulations") or []:
        if not isinstance(simulation, dict) or str(simulation.get("Name")) not in frozen_names:
            continue
        try:
            descriptor = _protocol_descriptor(snapshot, simulation, primary)
        except (AdaptationPlannerError, KeyError, TypeError, ValueError):
            continue
        if any(
            descriptor[field] != base[field]
            for field in (
                "application_type",
                "dose_basis",
                "dosing_interval",
                "formulations",
                "individual",
                "output_selections",
            )
        ):
            continue
        doses.append(
            _convert_dose(
                descriptor["dose_value"], descriptor["dose_unit"], base["dose_unit"]
            )
        )
    return sorted(set(doses))


def _perturbation_values(
    *, base_value: float, unit: str, compatible_doses: list[float]
) -> list[dict[str, Any]]:
    base_decimal = Decimal(str(base_value))
    candidates = [
        (
            "absolute_small_decrease",
            base_decimal - Decimal(str(_absolute_delta_in_unit(unit))),
            "local",
        ),
        ("relative_minus_1pct", base_decimal * Decimal("0.99"), "local"),
        ("relative_plus_1pct", base_decimal * Decimal("1.01"), "local"),
    ]
    distinct = [
        value
        for value in compatible_doses
        if not math.isclose(value, base_value, rel_tol=1e-12, abs_tol=1e-12)
    ]
    if distinct:
        nearest = min(distinct, key=lambda value: (abs(value - base_value), value))
        candidates.append(
            (
                "validated_interval_midpoint",
                (Decimal(str(nearest)) + base_decimal) / Decimal("2"),
                "interpolation",
            )
        )
    rows: list[dict[str, Any]] = []
    seen: list[float] = []
    for kind, decimal_value, domain in candidates:
        value = _canonical_decimal_float(float(decimal_value))
        if not math.isfinite(value) or value <= 0:
            continue
        if any(math.isclose(value, old, rel_tol=1e-10, abs_tol=1e-12) for old in seen):
            continue
        seen.append(value)
        rows.append(
            {
                "perturbation_type": kind,
                "new_value": value,
                "unit": unit,
                "domain_status": domain,
                "validated_dose_bounds": (
                    [min(compatible_doses), max(compatible_doses)]
                    if len(compatible_doses) >= 2
                    else None
                ),
            }
        )
    return rows


def _series_rows(frozen: Mapping[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    if frozen.get("evaluation_series"):
        values.append(copy.deepcopy(frozen["evaluation_series"]))
    values.extend(copy.deepcopy(frozen.get("additional_evaluation_series") or []))
    if not values:
        raise PerturbationError("Base verification simulation has no quantitative series.")
    return values


def _derive_snapshot(
    *,
    snapshot: Mapping[str, Any],
    base_simulation: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    primary: str,
    group_key: str,
    perturbations: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    derived = copy.deepcopy(snapshot)
    base_copy = copy.deepcopy(base_simulation)
    base_copy.pop("ObservedData", None)
    base_copy.pop("OutputMappings", None)
    base_copy["HasResults"] = False
    base_copy.pop("IndividualAnalyses", None)
    simulations = [base_copy]
    protocol_rows = list(derived.get("Protocols") or [])
    records: list[dict[str, Any]] = []
    for index, perturbation in enumerate(perturbations, 1):
        protocol_copy = copy.deepcopy(descriptor["protocol"])
        simulation_copy = copy.deepcopy(base_simulation)
        simulation_name = f"PERT-{group_key}-{index:02d}"
        protocol_name = f"Perturbation {group_key}-{index:02d}"
        protocol_copy["Name"] = protocol_name
        dose = _parameter(protocol_copy, "InputDose")
        if dose is None:
            raise PerturbationError("Perturbation template has no InputDose.")
        old_value = float(dose["Value"])
        dose["Value"] = float(perturbation["new_value"])
        protocol_rows.append(protocol_copy)
        protocol_index = len(protocol_rows) - 1
        parameter_index = list(protocol_copy.get("Parameters") or []).index(dose)
        simulation_copy["Name"] = simulation_name
        simulation_copy.pop("ObservedData", None)
        simulation_copy.pop("OutputMappings", None)
        simulation_copy["HasResults"] = False
        simulation_copy.pop("IndividualAnalyses", None)
        _rewrite_protocol_paths(
            simulation_copy,
            old_protocol_name=str(descriptor["protocol"]["Name"]),
            new_protocol_name=protocol_name,
        )
        compounds = [
            row for row in simulation_copy.get("Compounds") or [] if isinstance(row, dict)
        ]
        primary_rows = [
            row
            for row in compounds
            if _identity_token(row.get("Name")) == _identity_token(primary)
        ]
        if len(compounds) != 1 or len(primary_rows) != 1:
            raise PerturbationError(
                "Dose perturbation requires one primary compound and no co-medication."
            )
        primary_rows[0].setdefault("Protocol", {})["Name"] = protocol_name
        simulations.append(simulation_copy)
        records.append(
            {
                **perturbation,
                "simulation_sequence": index + 1,
                "simulation_name": simulation_name,
                "old_value": old_value,
                "json_path": (
                    f"/Protocols/{protocol_index}/Parameters/{parameter_index}/Value"
                ),
                "dose_ratio": float(perturbation["new_value"]) / old_value,
            }
        )
    derived["Protocols"] = protocol_rows
    derived["Simulations"] = simulations
    for optional_collection in (
        "SimulationClassifications",
        "ObservedData",
        "ObservedDataClassifications",
        "ParameterIdentifications",
        "ParameterIdentificationClassifications",
        "SensitivityAnalyses",
    ):
        derived.pop(optional_collection, None)
    return derived, records


def plan_scenario_perturbations(
    *,
    batch_directory: Path,
    audit_directory: Path,
    benchmark_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    batch_directory = batch_directory.resolve()
    audit_directory = audit_directory.resolve()
    benchmark_directory = benchmark_directory.resolve()
    batch_manifest = json.loads(
        (batch_directory / "batch_manifest.json").read_text(encoding="utf-8")
    )
    audit_summary = json.loads(
        (audit_directory / "summary.json").read_text(encoding="utf-8")
    )
    benchmark_manifest = json.loads(
        (benchmark_directory / "manifest.json").read_text(encoding="utf-8")
    )
    jobs_path = batch_directory / "jobs.jsonl"
    if sha256_file(jobs_path) != str(batch_manifest["jobs_sha256"]):
        raise PerturbationError("Batch jobs checksum drifted.")
    audit_path = audit_directory / "audit_index.jsonl"
    if sha256_file(audit_path) != str(
        audit_summary["output_checksums"]["audit_index.jsonl"]
    ):
        raise PerturbationError("Audit index checksum drifted.")
    truth_path = benchmark_directory / "benchmark_truth.jsonl"
    if sha256_file(truth_path) != str(
        benchmark_manifest["output_checksums"]["benchmark_truth.jsonl"]
    ):
        raise PerturbationError("Benchmark truth checksum drifted.")

    jobs = {str(row["protocol_key"]): row for row in _read_jsonl(jobs_path)}
    audits = {
        str(row["protocol_key"]): row for row in _read_jsonl(audit_path)
    }
    tasks = {
        str(row["task_id"]): row
        for row in _read_jsonl(benchmark_directory / "benchmark_tasks.jsonl")
    }
    grouped_truths: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    blocked: list[dict[str, Any]] = []
    for truth in _read_jsonl(truth_path):
        source_keys = sorted(str(value) for value in truth["source_protocol_keys"])
        if not source_keys:
            blocked.append(
                {
                    "task_id": truth["task_id"],
                    "reason_code": "SOURCE_PROTOCOL_MISSING",
                }
            )
            continue
        grouped_truths[(source_keys[0], int(truth["source_simulation_sequence"]))].append(
            truth
        )

    plans: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []
    perturbed_tasks: list[dict[str, Any]] = []
    variable_eligibility: list[dict[str, Any]] = []
    for (source_key, sequence), truths in sorted(grouped_truths.items()):
        task_ids = sorted(str(row["task_id"]) for row in truths)
        group_key = _stable_hex("scenario-perturbation", source_key, sequence, task_ids, length=24)
        base_record = {
            "perturbation_group_id": f"perturbation-group:{group_key}",
            "protocol_key": group_key,
            "source_protocol_key": source_key,
            "source_simulation_sequence": sequence,
            "task_ids": task_ids,
            "planner_version": PERTURBATION_PLANNER_VERSION,
        }
        try:
            audit = audits[source_key]
            if not audit.get("engineering_accepted"):
                raise PerturbationError("Source protocol did not pass engineering audit.")
            job = jobs[source_key]
            protocol = _load_locked_json(
                batch_directory / str(job["protocol_relative_path"]),
                str(job["protocol_sha256"]),
                label="source protocol",
            )
            snapshot = _load_locked_json(
                batch_directory / str(job["snapshot_relative_path"]),
                str(job["snapshot_sha256"]),
                label="source snapshot",
            )
            frozen = next(
                row
                for row in protocol["simulations"]
                if int(row["sequence"]) == sequence
            )
            if frozen.get("role") != "model_verification":
                raise PerturbationError("Source simulation is not a verification scenario.")
            base_simulation = _find_named(
                snapshot.get("Simulations"), str(frozen["name"]), kind="base simulation"
            )
            compounds = [
                value
                for task_id in task_ids
                for value in tasks[task_id].get("active_moieties") or []
            ]
            normalized_compounds = {_identity_token(value) for value in compounds}
            if len(normalized_compounds) != 1:
                raise PerturbationError("Grouped tasks do not resolve to one active moiety.")
            primary = str(tasks[task_ids[0]]["active_moieties"][0])
            descriptor = _protocol_descriptor(snapshot, base_simulation, primary)
            values = _perturbation_values(
                base_value=float(descriptor["dose_value"]),
                unit=str(descriptor["dose_unit"]),
                compatible_doses=_compatible_doses(
                    snapshot=snapshot,
                    protocol=protocol,
                    primary=primary,
                    base=descriptor,
                ),
            )
            if not values:
                raise PerturbationError("No valid positive dose perturbation was generated.")
            derived, perturbations = _derive_snapshot(
                snapshot=snapshot,
                base_simulation=base_simulation,
                descriptor=descriptor,
                primary=primary,
                group_key=group_key,
                perturbations=values,
            )
            variable_eligibility.extend(
                _variable_eligibility(
                    snapshot=snapshot,
                    simulation=base_simulation,
                    descriptor=descriptor,
                    perturbation_group_id=base_record["perturbation_group_id"],
                    active_moiety=primary,
                    perturbations=perturbations,
                )
            )
            plan_root = output_directory / "plans" / group_key
            snapshot_path = plan_root / "perturbed_snapshot.json"
            snapshot_sha = _atomic_json(snapshot_path, derived)
            semantic_sha = snapshot_semantic_sha256(derived)
            base_series = _series_rows(frozen)
            output_probes = []
            for series in base_series:
                suffix = _quantity_suffix(str(frozen["name"]), str(series["output_path"]))
                output_probes.append(
                    {
                        "dataset_name": series["dataset_name"],
                        "base_output_path": series["output_path"],
                        "output_suffix": suffix,
                        "molecular_weight_g_per_mol": series[
                            "molecular_weight_g_per_mol"
                        ],
                        "scaling": series["scaling"],
                        "source_location": series["source_location"],
                        "reference": series.get("reference"),
                    }
                )
            source_lock = copy.deepcopy(protocol["source"])
            source_lock["snapshot_declared_path"] = str(
                source_lock.get("snapshot_relative_path") or ""
            )
            source_lock["snapshot_relative_path"] = snapshot_path.name
            source_lock["snapshot_resolution_mode"] = "declared_relative_path"
            source_lock["snapshot_sha256"] = snapshot_sha
            protocol_payload = {
                "protocol_schema_version": PERTURBATION_PROTOCOL_SCHEMA_VERSION,
                "protocol_builder_version": PERTURBATION_PLANNER_VERSION,
                "protocol_id": "snapshot-reuse-"
                + _stable_hex("perturbation-protocol", group_key, snapshot_sha, length=32),
                "generated_at": datetime.now(UTC).isoformat(),
                "source": source_lock,
                "source_archive_path": protocol["source_archive_path"],
                "source_snapshot_semantic_sha256": semantic_sha,
                "counts": _protocol_counts(derived, len(perturbations)),
                "runtime": protocol["runtime"],
                "criteria": protocol["criteria"],
                "simulations": [
                    {
                        "sequence": 1,
                        "name": frozen["name"],
                        "role": "model_verification",
                        "individual": frozen.get("individual"),
                        "observed_dataset_names": frozen.get("observed_dataset_names") or [],
                        "evaluation_series": base_series[0],
                        "additional_evaluation_series": base_series[1:],
                    },
                    *[
                        {
                            "sequence": row["simulation_sequence"],
                            "name": row["simulation_name"],
                            "role": "execution_only",
                            "individual": frozen.get("individual"),
                            "observed_dataset_names": [],
                            "evaluation_series": None,
                            "additional_evaluation_series": [],
                        }
                        for row in perturbations
                    ],
                ],
            }
            LicensedSnapshotReuseProtocol.model_validate(protocol_payload)
            protocol_path = plan_root / "perturbation_protocol.json"
            protocol_sha = _atomic_json(protocol_path, protocol_payload)
            plan = {
                **base_record,
                "status": "planned",
                "active_moiety": primary,
                "base_simulation_name": frozen["name"],
                "base_dose_value": descriptor["dose_value"],
                "base_dose_unit": descriptor["dose_unit"],
                "base_snapshot_sha256": job["snapshot_sha256"],
                "perturbed_snapshot_relative_path": snapshot_path.relative_to(
                    output_directory
                ).as_posix(),
                "perturbed_snapshot_sha256": snapshot_sha,
                "perturbed_snapshot_semantic_sha256": semantic_sha,
                "protocol_relative_path": protocol_path.relative_to(
                    output_directory
                ).as_posix(),
                "protocol_sha256": protocol_sha,
                "output_probes": output_probes,
                "perturbations": perturbations,
                "scientific_grade_generated": False,
                "created_at": datetime.now(UTC).isoformat(),
            }
            plan_path = plan_root / "perturbation_plan.json"
            plan["perturbation_plan_sha256"] = _atomic_json(plan_path, plan)
            plans.append(plan)
            for task_id in task_ids:
                for perturbation in perturbations:
                    perturbed_task_id = _stable_hex(
                        "perturbed-ranking-task",
                        task_id,
                        perturbation["perturbation_type"],
                        perturbation["new_value"],
                        perturbation["unit"],
                        length=32,
                    )
                    scenario = copy.deepcopy(tasks[task_id].get("scenario") or {})
                    scenario["dose_value"] = perturbation["new_value"]
                    scenario["dose_unit"] = perturbation["unit"]
                    regimens = scenario.get("regimens") or []
                    if len(regimens) != 1:
                        raise PerturbationError(
                            "Dose perturbation requires exactly one structured regimen."
                        )
                    regimens[0]["dose_value"] = perturbation["new_value"]
                    regimens[0]["dose_unit"] = perturbation["unit"]
                    perturbed_tasks.append(
                        {
                            **copy.deepcopy(tasks[task_id]),
                            "task_id": perturbed_task_id,
                            "base_task_id": task_id,
                            "perturbation_group_id": plan[
                                "perturbation_group_id"
                            ],
                            "perturbation_type": perturbation[
                                "perturbation_type"
                            ],
                            "domain_status": perturbation["domain_status"],
                            "scenario": scenario,
                            "scientific_grade_generated": False,
                        }
                    )
            protocol_rows.append(
                {
                    "protocol_key": group_key,
                    "protocol_id": protocol_payload["protocol_id"],
                    "protocol_path": str(protocol_path.resolve()),
                    "protocol_sha256": protocol_sha,
                    "checkout_path": str(plan_root.resolve()),
                    "snapshot_relative_path": snapshot_path.name,
                    "full_name": f"perturbation/{primary}/{source_key}/{sequence}",
                    "version_id": job.get("version_id"),
                    "repository_commit": job["repository_commit"],
                    "model_name": job["model_name"],
                    "compounds": [primary],
                    "model_verification_count": 1,
                    "perturbation_group_id": plan["perturbation_group_id"],
                }
            )
        except (
            AdaptationPlannerError,
            PerturbationError,
            KeyError,
            StopIteration,
            TypeError,
            ValueError,
        ) as exc:
            blocked.append(
                {
                    **base_record,
                    "status": "blocked",
                    "reason_code": "PERTURBATION_NOT_EXECUTABLE",
                    "detail": str(exc),
                }
            )
            variable_eligibility.extend(
                {
                    "perturbation_group_id": base_record[
                        "perturbation_group_id"
                    ],
                    "active_moiety": None,
                    "variable": variable,
                    "status": "blocked",
                    "structured_path": None,
                    "unit": None,
                    "generated_perturbation_count": 0,
                    "reason_code": "BASE_SCENARIO_NOT_ELIGIBLE",
                    "detail": str(exc),
                }
                for variable in _AUDITED_CONTINUOUS_VARIABLES
            )

    plans.sort(key=lambda row: str(row["perturbation_group_id"]))
    blocked.sort(key=lambda row: str(row["perturbation_group_id"]))
    protocol_rows.sort(key=lambda row: str(row["protocol_key"]))
    perturbed_tasks.sort(key=lambda row: str(row["task_id"]))
    variable_eligibility.sort(
        key=lambda row: (
            str(row["perturbation_group_id"]), str(row["variable"])
        )
    )
    outputs = {
        "perturbation_plans.jsonl": _atomic_jsonl(
            output_directory / "perturbation_plans.jsonl", plans
        ),
        "blocked_perturbations.jsonl": _atomic_jsonl(
            output_directory / "blocked_perturbations.jsonl", blocked
        ),
        "protocols.jsonl": _atomic_jsonl(
            output_directory / "protocols.jsonl", protocol_rows
        ),
        "perturbed_tasks.jsonl": _atomic_jsonl(
            output_directory / "perturbed_tasks.jsonl", perturbed_tasks
        ),
        "variable_eligibility.jsonl": _atomic_jsonl(
            output_directory / "variable_eligibility.jsonl",
            variable_eligibility,
        ),
    }
    manifest = {
        "planner_version": PERTURBATION_PLANNER_VERSION,
        "batch_directory": str(batch_directory),
        "audit_directory": str(audit_directory),
        "benchmark_directory": str(benchmark_directory),
        "group_count": len(plans),
        "perturbation_count": sum(len(row["perturbations"]) for row in plans),
        "perturbed_ranking_task_count": len(perturbed_tasks),
        "blocked_group_count": len(blocked),
        "continuous_variable_audit_count": len(variable_eligibility),
        "continuous_variable_status_counts": {
            status: sum(
                row["status"] == status for row in variable_eligibility
            )
            for status in ("planned", "blocked")
        },
        "scientific_grade_generated": False,
        "output_checksums": outputs,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(output_directory / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build checksum-locked paired PK-Sim dose perturbation protocols."
    )
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--audits", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = plan_scenario_perturbations(
        batch_directory=args.batch,
        audit_directory=args.audits,
        benchmark_directory=args.benchmark,
        output_directory=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
