from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.licensed_snapshot_reuse import (
    LicensedSnapshotReuseProtocol,
    snapshot_semantic_sha256,
)


ADAPTATION_PLANNER_VERSION = "pbpk-controlled-adaptation-planner-1.5.0"
ADAPTATION_PROTOCOL_SCHEMA_VERSION = "1.0.0"

_MASS_FACTORS_TO_MG = {
    "g": 1000.0,
    "mg": 1.0,
    "ug": 0.001,
    "µg": 0.001,
    "μg": 0.001,
}
_SUPPORTED_APPLICATION_TYPES = {
    "Oral": "oral",
    "Intravenous": "intravenous_infusion",
    "IntravenousBolus": "intravenous_bolus",
}
_KNOWN_DOSING_INTERVALS = {
    "Single",
    "DI_24",
    "DI_12_12",
    "DI_8_8_8",
    "DI_6_6_12",
}
_OPERATION_WEIGHTS = {
    "dose": 1.0,
    "frequency": 2.0,
    "start_time": 0.5,
    "end_time": 0.5,
    "infusion_duration": 1.0,
}
_TIME_FACTORS_TO_HOURS = {
    "h": 1.0,
    "hr": 1.0,
    "hour": 1.0,
    "hours": 1.0,
    "min": 1.0 / 60.0,
    "minute": 1.0 / 60.0,
    "minutes": 1.0 / 60.0,
    "s": 1.0 / 3600.0,
    "sec": 1.0 / 3600.0,
    "d": 24.0,
    "day": 24.0,
    "days": 24.0,
}


class AdaptationPlannerError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def _stable_hex(namespace: str, *parts: Any, length: int = 32) -> str:
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8"))
    for part in parts:
        digest.update(b"\x1f")
        digest.update(_canonical_bytes(part))
    return digest.hexdigest()[:length]


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AdaptationPlannerError(
                    f"Expected JSON object at {path}:{line_number}."
                )
            rows.append(value)
    return rows


def _normalized(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _identity_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalized(value))


def _find_named(rows: object, name: str, *, kind: str) -> dict[str, Any]:
    matches = [
        row
        for row in (rows or [])
        if isinstance(row, dict) and str(row.get("Name") or "") == name
    ]
    if len(matches) != 1:
        raise AdaptationPlannerError(
            f"Expected one {kind} named {name!r}; found {len(matches)}."
        )
    return matches[0]


def _parameter(protocol: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    matches = [
        row
        for row in protocol.get("Parameters") or []
        if isinstance(row, dict) and str(row.get("Name") or "") == name
    ]
    if len(matches) > 1:
        raise AdaptationPlannerError(
            f"Protocol {protocol.get('Name')!r} has duplicate {name!r} parameters."
        )
    return matches[0] if matches else None


def _mass_unit(unit: object) -> tuple[str, bool]:
    normalized = str(unit or "").strip().casefold().replace("μ", "µ")
    per_kg = normalized.endswith("/kg")
    base = normalized[:-3] if per_kg else normalized
    if base not in _MASS_FACTORS_TO_MG:
        raise AdaptationPlannerError(f"Unsupported dose unit: {unit!r}.")
    return base, per_kg


def _convert_dose(value: object, from_unit: object, to_unit: object) -> float:
    source, source_per_kg = _mass_unit(from_unit)
    target, target_per_kg = _mass_unit(to_unit)
    if source_per_kg != target_per_kg:
        raise AdaptationPlannerError(
            "Absolute and body-weight-normalized doses cannot be interconverted."
        )
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise AdaptationPlannerError(f"Dose must be finite and positive: {value!r}.")
    return numeric * _MASS_FACTORS_TO_MG[source] / _MASS_FACTORS_TO_MG[target]


def _time_in_hours(value: object, unit: object) -> float:
    normalized = str(unit or "").strip().casefold()
    if normalized not in _TIME_FACTORS_TO_HOURS:
        raise AdaptationPlannerError(f"Unsupported time unit: {unit!r}.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise AdaptationPlannerError(
            f"Time must be finite and non-negative: {value!r}."
        )
    return numeric * _TIME_FACTORS_TO_HOURS[normalized]


def _extend_output_horizon(
    simulation: dict[str, Any],
    *,
    required_end_time_hours: float,
) -> dict[str, Any] | None:
    end_parameters: list[tuple[float, int, int, dict[str, Any]]] = []
    for schema_index, schema in enumerate(simulation.get("OutputSchema") or []):
        if not isinstance(schema, dict):
            continue
        for parameter_index, parameter in enumerate(schema.get("Parameters") or []):
            if (
                isinstance(parameter, dict)
                and str(parameter.get("Name") or "") == "End time"
            ):
                end_parameters.append(
                    (
                        _time_in_hours(
                            parameter.get("Value"), parameter.get("Unit")
                        ),
                        schema_index,
                        parameter_index,
                        parameter,
                    )
                )
    if not end_parameters:
        raise AdaptationPlannerError(
            "Adapted simulation has no structured OutputSchema End time."
        )
    current_hours, schema_index, parameter_index, parameter = max(
        end_parameters, key=lambda row: (row[0], row[1], row[2])
    )
    schema = simulation["OutputSchema"][schema_index]
    resolution = _parameter(schema, "Resolution")
    if resolution is None:
        raise AdaptationPlannerError(
            "Last OutputSchema interval has no structured Resolution."
        )
    resolution_unit = str(resolution.get("Unit") or "").strip().casefold()
    if not resolution_unit.startswith("pts/"):
        raise AdaptationPlannerError(
            f"Unsupported output resolution unit: {resolution.get('Unit')!r}."
        )
    resolution_time_unit = resolution_unit.split("/", 1)[1]
    if resolution_time_unit not in _TIME_FACTORS_TO_HOURS:
        raise AdaptationPlannerError(
            f"Unsupported output resolution unit: {resolution.get('Unit')!r}."
        )
    points_per_time_unit = float(resolution.get("Value"))
    if not math.isfinite(points_per_time_unit) or points_per_time_unit <= 0:
        raise AdaptationPlannerError(
            f"Output resolution must be finite and positive: "
            f"{resolution.get('Value')!r}."
        )
    grid_step_hours = (
        _TIME_FACTORS_TO_HOURS[resolution_time_unit] / points_per_time_unit
    )
    required_with_margin_hours = required_end_time_hours + grid_step_hours
    if current_hours >= required_with_margin_hours:
        return None
    unit = str(parameter.get("Unit") or "")
    factor = _TIME_FACTORS_TO_HOURS[unit.strip().casefold()]
    old_value = parameter.get("Value")
    new_value = round(required_with_margin_hours / factor, 12)
    parameter["Value"] = new_value
    return {
        "adjustment_type": "output_horizon",
        "json_path": (
            f"/Simulations/0/OutputSchema/{schema_index}/Parameters/"
            f"{parameter_index}/Value"
        ),
        "old_value": old_value,
        "new_value": new_value,
        "unit": unit,
        "grid_step_hours": grid_step_hours,
        "required_observation_end_time_hours": required_end_time_hours,
        "reason": "cover_hidden_observation_horizon_without_extrapolation",
    }


def _administered_compound_rows(
    simulation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        row
        for row in simulation.get("Compounds") or []
        if isinstance(row, dict)
        and str((row.get("Protocol") or {}).get("Name") or "").strip()
    ]


def _formulations(simulation: Mapping[str, Any], primary: str) -> tuple[str, ...]:
    compounds = _administered_compound_rows(simulation)
    primary_rows = [
        row
        for row in compounds
        if _identity_token(row.get("Name")) == _identity_token(primary)
    ]
    if len(primary_rows) != 1 or len(compounds) != 1:
        raise AdaptationPlannerError(
            "Controlled scenario adaptation requires one administered primary "
            "compound and no co-administered dosing protocol."
        )
    protocol = primary_rows[0].get("Protocol") or {}
    return tuple(
        sorted(
            {
                _normalized(row.get("Name"))
                for row in protocol.get("Formulations") or []
                if isinstance(row, dict) and _normalized(row.get("Name"))
            }
        )
    )


def _simulation_protocol_name(
    simulation: Mapping[str, Any], primary: str
) -> str:
    compounds = _administered_compound_rows(simulation)
    primary_rows = [
        row
        for row in compounds
        if _identity_token(row.get("Name")) == _identity_token(primary)
    ]
    if len(primary_rows) != 1 or len(compounds) != 1:
        raise AdaptationPlannerError(
            "Controlled adaptation requires exactly one administered primary "
            "compound in the template simulation."
        )
    protocol_name = str(
        (primary_rows[0].get("Protocol") or {}).get("Name") or ""
    ).strip()
    if not protocol_name:
        raise AdaptationPlannerError("Template simulation has no protocol reference.")
    return protocol_name


def _individual_descriptor(
    snapshot: Mapping[str, Any], simulation: Mapping[str, Any]
) -> tuple[str, str, str]:
    name = str(simulation.get("Individual") or "").strip()
    population_name = str(simulation.get("Population") or "").strip()
    if name:
        individual = _find_named(
            snapshot.get("Individuals"), name, kind="individual"
        )
        origin = individual.get("OriginData") or {}
    elif population_name:
        population = _find_named(
            snapshot.get("Populations"), population_name, kind="population"
        )
        individual = (population.get("Settings") or {}).get("Individual") or {}
        origin = individual.get("OriginData") or {}
        name = population_name
    else:
        raise AdaptationPlannerError(
            "Simulation has neither an individual nor a population reference."
        )
    descriptor = (
        _normalized(origin.get("Species")),
        _normalized(origin.get("Population")),
        _normalized(origin.get("Gender")),
    )
    if not all(descriptor):
        raise AdaptationPlannerError(
            f"Individual {name!r} lacks species, population, or gender metadata."
        )
    return descriptor


def _quantity_suffix(simulation_name: str, output_path: str) -> str:
    prefix = simulation_name.strip() + "|"
    return output_path[len(prefix) :] if output_path.startswith(prefix) else output_path


def _output_selections(simulation: Mapping[str, Any]) -> set[str]:
    return {
        _normalized(value)
        for value in simulation.get("OutputSelections") or []
        if str(value).strip()
    }


def _protocol_descriptor(
    snapshot: Mapping[str, Any], simulation: Mapping[str, Any], primary: str
) -> dict[str, Any]:
    protocol_name = _simulation_protocol_name(simulation, primary)
    protocol = _find_named(snapshot.get("Protocols"), protocol_name, kind="protocol")
    application_type = str(protocol.get("ApplicationType") or "")
    if application_type not in _SUPPORTED_APPLICATION_TYPES:
        raise AdaptationPlannerError(
            f"Unsupported or implicit application type: {application_type!r}."
        )
    interval = str(protocol.get("DosingInterval") or "")
    if interval not in _KNOWN_DOSING_INTERVALS:
        raise AdaptationPlannerError(f"Unsupported dosing interval: {interval!r}.")
    dose = _parameter(protocol, "InputDose")
    if dose is None:
        raise AdaptationPlannerError("Protocol has no InputDose parameter.")
    dose_base, dose_per_kg = _mass_unit(dose.get("Unit"))
    return {
        "protocol_name": protocol_name,
        "protocol": protocol,
        "application_type": application_type,
        "route_class": _SUPPORTED_APPLICATION_TYPES[application_type],
        "dosing_interval": interval,
        "dose_value": float(dose["Value"]),
        "dose_unit": str(dose["Unit"]),
        "dose_basis": "per_kg" if dose_per_kg else "absolute",
        "dose_mg": float(dose["Value"]) * _MASS_FACTORS_TO_MG[dose_base],
        "formulations": _formulations(simulation, primary),
        "individual": _individual_descriptor(snapshot, simulation),
        "output_selections": _output_selections(simulation),
    }


def _context_reasons(
    source: Mapping[str, Any], candidate: Mapping[str, Any], output_suffix: str
) -> list[str]:
    reasons: list[str] = []
    if source["application_type"] != candidate["application_type"]:
        reasons.append("ROUTE_OR_APPLICATION_TYPE_ADAPTATION_NOT_ALLOWED")
    if source["dose_basis"] != candidate["dose_basis"]:
        reasons.append("DOSE_BASIS_MISMATCH")
    if source["formulations"] != candidate["formulations"]:
        reasons.append("FORMULATION_STRUCTURE_MISMATCH")
    if source["individual"] != candidate["individual"]:
        reasons.append("POPULATION_OR_GENDER_STRUCTURE_MISMATCH")
    if _normalized(output_suffix) not in candidate["output_selections"]:
        reasons.append("REQUIRED_OUTPUT_NOT_AVAILABLE_IN_TEMPLATE")
    source_protocol = source["protocol"]
    candidate_protocol = candidate["protocol"]
    if source["application_type"] == "Intravenous":
        if _parameter(source_protocol, "Infusion time") is None:
            reasons.append("SOURCE_INFUSION_DURATION_MISSING")
        if _parameter(candidate_protocol, "Infusion time") is None:
            reasons.append("TEMPLATE_INFUSION_DURATION_MISSING")
    if source["dosing_interval"] != "Single" and _parameter(
        source_protocol, "End time"
    ) is None:
        reasons.append("SOURCE_REPEAT_END_TIME_MISSING")
    return sorted(set(reasons))


def _operation(
    *,
    operation_type: str,
    json_path: str,
    old_value: object,
    new_value: object,
    unit: str | None,
) -> dict[str, Any]:
    return {
        "operation_type": operation_type,
        "json_path": json_path,
        "old_value": old_value,
        "new_value": new_value,
        "unit": unit,
        "weight": _OPERATION_WEIGHTS[operation_type],
    }


def _set_parameter(
    *,
    protocol: dict[str, Any],
    name: str,
    source_protocol: Mapping[str, Any],
    protocol_index: int,
    operation_type: str,
    required: bool,
) -> dict[str, Any] | None:
    target = _parameter(protocol, name)
    source = _parameter(source_protocol, name)
    if source is None:
        if required:
            raise AdaptationPlannerError(f"Source protocol lacks {name!r}.")
        return None
    if target is None:
        protocol.setdefault("Parameters", []).append(copy.deepcopy(source))
        target = _parameter(protocol, name)
        assert target is not None
        old_value = None
    else:
        old_value = target.get("Value")
    old_unit = str(target.get("Unit") or source.get("Unit") or "") or None
    new_value = source.get("Value")
    new_unit = str(source.get("Unit") or "") or None
    target["Value"] = new_value
    if new_unit:
        target["Unit"] = new_unit
    if old_value == new_value and old_unit == new_unit:
        return None
    parameter_index = list(protocol.get("Parameters") or []).index(target)
    return _operation(
        operation_type=operation_type,
        json_path=(
            f"/Protocols/{protocol_index}/Parameters/{parameter_index}/Value"
        ),
        old_value=old_value,
        new_value=new_value,
        unit=new_unit,
    )


def _rewrite_protocol_paths(
    value: Any,
    *,
    old_protocol_name: str,
    new_protocol_name: str,
) -> int:
    rewritten = 0
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "Path" and isinstance(child, str):
                segments = child.split("|")
                if (
                    len(segments) >= 2
                    and segments[0] in {"Applications", "Events"}
                    and segments[1] == old_protocol_name
                ):
                    segments[1] = new_protocol_name
                    value[key] = "|".join(segments)
                    rewritten += 1
                    continue
            rewritten += _rewrite_protocol_paths(
                child,
                old_protocol_name=old_protocol_name,
                new_protocol_name=new_protocol_name,
            )
    elif isinstance(value, list):
        for child in value:
            rewritten += _rewrite_protocol_paths(
                child,
                old_protocol_name=old_protocol_name,
                new_protocol_name=new_protocol_name,
            )
    return rewritten


def _adapt_snapshot(
    *,
    candidate_snapshot: Mapping[str, Any],
    template_simulation: Mapping[str, Any],
    template_descriptor: Mapping[str, Any],
    source_simulation: Mapping[str, Any],
    source_descriptor: Mapping[str, Any],
    primary: str,
    adapted_name: str,
    required_observation_end_time_hours: float,
) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]
]:
    adapted_snapshot = copy.deepcopy(candidate_snapshot)
    adapted_protocol = copy.deepcopy(template_descriptor["protocol"])
    protocol_name = f"Adaptation {adapted_name}"
    adapted_protocol["Name"] = protocol_name
    adapted_snapshot.setdefault("Protocols", []).append(adapted_protocol)
    protocol_index = len(adapted_snapshot["Protocols"]) - 1
    operations: list[dict[str, Any]] = []

    target_dose = _parameter(adapted_protocol, "InputDose")
    source_dose = _parameter(source_descriptor["protocol"], "InputDose")
    assert target_dose is not None and source_dose is not None
    converted = _convert_dose(
        source_dose["Value"], source_dose["Unit"], target_dose["Unit"]
    )
    if not math.isclose(
        float(target_dose["Value"]), converted, rel_tol=1e-12, abs_tol=1e-12
    ):
        parameter_index = adapted_protocol["Parameters"].index(target_dose)
        operations.append(
            _operation(
                operation_type="dose",
                json_path=(
                    f"/Protocols/{protocol_index}/Parameters/"
                    f"{parameter_index}/Value"
                ),
                old_value=target_dose["Value"],
                new_value=converted,
                unit=str(target_dose["Unit"]),
            )
        )
        target_dose["Value"] = converted

    source_interval = str(source_descriptor["dosing_interval"])
    old_interval = str(adapted_protocol.get("DosingInterval") or "")
    if old_interval != source_interval:
        operations.append(
            _operation(
                operation_type="frequency",
                json_path=f"/Protocols/{protocol_index}/DosingInterval",
                old_value=old_interval,
                new_value=source_interval,
                unit=None,
            )
        )
        adapted_protocol["DosingInterval"] = source_interval

    for name, operation_type, required in (
        ("Start time", "start_time", True),
        ("End time", "end_time", source_interval != "Single"),
        (
            "Infusion time",
            "infusion_duration",
            source_descriptor["application_type"] == "Intravenous",
        ),
    ):
        operation = _set_parameter(
            protocol=adapted_protocol,
            name=name,
            source_protocol=source_descriptor["protocol"],
            protocol_index=protocol_index,
            operation_type=operation_type,
            required=required,
        )
        if operation is not None:
            operations.append(operation)

    adapted_simulation = copy.deepcopy(template_simulation)
    adapted_simulation["Name"] = adapted_name
    adapted_simulation["OutputSchema"] = copy.deepcopy(
        source_simulation.get("OutputSchema") or []
    )
    execution_adjustments = []
    output_horizon_adjustment = _extend_output_horizon(
        adapted_simulation,
        required_end_time_hours=required_observation_end_time_hours,
    )
    if output_horizon_adjustment is not None:
        execution_adjustments.append(output_horizon_adjustment)
    adapted_simulation.pop("ObservedData", None)
    adapted_simulation.pop("OutputMappings", None)
    adapted_simulation["HasResults"] = False
    adapted_simulation.pop("IndividualAnalyses", None)
    _rewrite_protocol_paths(
        adapted_simulation,
        old_protocol_name=str(template_descriptor["protocol"]["Name"]),
        new_protocol_name=protocol_name,
    )
    compound_rows = [
        row
        for row in adapted_simulation.get("Compounds") or []
        if isinstance(row, dict)
    ]
    administered_rows = [
        row
        for row in compound_rows
        if str((row.get("Protocol") or {}).get("Name") or "").strip()
    ]
    primary_rows = [
        row
        for row in administered_rows
        if _identity_token(row.get("Name")) == _identity_token(primary)
    ]
    if len(administered_rows) != 1 or len(primary_rows) != 1:
        raise AdaptationPlannerError(
            "Adapted simulation no longer has exactly one administered primary "
            "compound."
        )
    primary_rows[0].setdefault("Protocol", {})["Name"] = protocol_name
    adapted_snapshot["Simulations"] = [adapted_simulation]
    for optional_collection in (
        "SimulationClassifications",
        "ObservedData",
        "ObservedDataClassifications",
        "ParameterIdentifications",
        "SensitivityAnalyses",
    ):
        adapted_snapshot.pop(optional_collection, None)
    return adapted_snapshot, operations, execution_adjustments


def _protocol_counts(snapshot: Mapping[str, Any]) -> dict[str, int]:
    return {
        "compounds": len(snapshot.get("Compounds") or []),
        "expression_profiles": len(snapshot.get("ExpressionProfiles") or []),
        "formulations": len(snapshot.get("Formulations") or []),
        "individuals": len(snapshot.get("Individuals") or []),
        "protocols": len(snapshot.get("Protocols") or []),
        "simulations": 1,
        "observed_data": len(snapshot.get("ObservedData") or []),
        "simulation_classifications": len(
            snapshot.get("SimulationClassifications") or []
        ),
        "model_building": 0,
        "model_verification": 1,
        "execution_only": 0,
    }


def _adapted_observed_series(
    *, truth: Mapping[str, Any], source_simulation_name: str, adapted_name: str
) -> dict[str, Any]:
    output_suffix = _quantity_suffix(
        source_simulation_name, str(truth["output_path"])
    )
    return {
        "dataset_name": truth["dataset_name"],
        "dataset_index": 0,
        "column_index": 0,
        "source_location": truth["source_location"],
        "reference": truth.get("reference"),
        "output_path": f"{adapted_name}|{output_suffix}",
        "scaling": "Log",
        "time_values": truth["time_values"],
        "time_unit": truth["time_unit"],
        "observed_values": truth["observed_values"],
        "observed_unit": truth["observed_unit"],
        "molecular_weight_g_per_mol": truth["molecular_weight_g_per_mol"],
    }


def _load_locked_json(path: Path, expected_sha256: str, *, label: str) -> Any:
    if not path.is_file():
        raise AdaptationPlannerError(f"Missing {label}: {path}.")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise AdaptationPlannerError(
            f"{label} checksum mismatch: expected {expected_sha256}, found {actual}."
        )
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _candidate_alias(
    candidate: Mapping[str, Any], jobs: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any]:
    aliases = sorted(
        candidate.get("version_aliases") or [],
        key=lambda row: (
            str(row.get("repository_commit") or ""),
            str(row.get("protocol_key") or ""),
        ),
    )
    for alias in aliases:
        key = str(alias.get("protocol_key") or "")
        if key in jobs:
            return jobs[key]
    raise AdaptationPlannerError(
        f"Candidate {candidate.get('candidate_id')} has no batch alias."
    )


def plan_controlled_adaptations(
    *,
    batch_directory: Path,
    benchmark_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    batch_directory = batch_directory.resolve()
    benchmark_directory = benchmark_directory.resolve()
    batch_manifest = json.loads(
        (batch_directory / "batch_manifest.json").read_text(encoding="utf-8")
    )
    benchmark_manifest = json.loads(
        (benchmark_directory / "manifest.json").read_text(encoding="utf-8")
    )
    jobs_path = batch_directory / "jobs.jsonl"
    pairs_path = benchmark_directory / "candidate_pairs.jsonl"
    if sha256_file(jobs_path) != str(batch_manifest["jobs_sha256"]):
        raise AdaptationPlannerError("Batch jobs checksum drifted.")
    if sha256_file(pairs_path) != str(
        benchmark_manifest["output_checksums"]["candidate_pairs.jsonl"]
    ):
        raise AdaptationPlannerError("Benchmark candidate pairs checksum drifted.")
    jobs = {str(row["protocol_key"]): row for row in _read_jsonl(jobs_path)}
    candidates = {
        str(row["candidate_id"]): row
        for row in _read_jsonl(benchmark_directory / "candidates.jsonl")
    }
    tasks = {
        str(row["task_id"]): row
        for row in _read_jsonl(benchmark_directory / "benchmark_tasks.jsonl")
    }
    truths = {
        str(row["task_id"]): row
        for row in _read_jsonl(benchmark_directory / "benchmark_truth.jsonl")
    }
    pairs = _read_jsonl(pairs_path)
    planned: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []
    for pair in pairs:
        if not pair.get("adaptation_required") or not pair.get(
            "same_active_moiety"
        ):
            continue
        task_id = str(pair["task_id"])
        candidate_id = str(pair["candidate_id"])
        task = tasks[task_id]
        truth = truths[task_id]
        candidate = candidates[candidate_id]
        plan_key = _stable_hex(
            "controlled-adaptation", task_id, candidate_id, length=24
        )
        base_record = {
            "adaptation_plan_id": f"adaptation-plan:{plan_key}",
            "protocol_key": plan_key,
            "task_id": task_id,
            "candidate_id": candidate_id,
            "active_moieties": task["active_moieties"],
            "planner_version": ADAPTATION_PLANNER_VERSION,
        }
        try:
            candidate_job = _candidate_alias(candidate, jobs)
            candidate_protocol_path = batch_directory / str(
                candidate_job["protocol_relative_path"]
            )
            candidate_snapshot_path = batch_directory / str(
                candidate_job["snapshot_relative_path"]
            )
            candidate_protocol = _load_locked_json(
                candidate_protocol_path,
                str(candidate_job["protocol_sha256"]),
                label="candidate protocol",
            )
            candidate_snapshot = _load_locked_json(
                candidate_snapshot_path,
                str(candidate_job["snapshot_sha256"]),
                label="candidate snapshot",
            )
            source_keys = sorted(str(value) for value in truth["source_protocol_keys"])
            source_key = source_keys[0]
            source_job = jobs[source_key]
            source_protocol = _load_locked_json(
                batch_directory / str(source_job["protocol_relative_path"]),
                str(source_job["protocol_sha256"]),
                label="source protocol",
            )
            source_snapshot = _load_locked_json(
                batch_directory / str(source_job["snapshot_relative_path"]),
                str(source_job["snapshot_sha256"]),
                label="source snapshot",
            )
            source_frozen = next(
                row
                for row in source_protocol["simulations"]
                if int(row["sequence"]) == int(truth["source_simulation_sequence"])
            )
            source_simulation_name = str(source_frozen["name"])
            same_semantic_snapshot = (
                str(candidate_protocol["source_snapshot_semantic_sha256"])
                == str(source_protocol["source_snapshot_semantic_sha256"])
            )
            source_simulation = _find_named(
                source_snapshot.get("Simulations"),
                source_simulation_name,
                kind="source simulation",
            )
            primary = str(task["active_moieties"][0])
            source_descriptor = _protocol_descriptor(
                source_snapshot, source_simulation, primary
            )
            output_suffix = _quantity_suffix(
                str(source_frozen["name"]), str(truth["output_path"])
            )
            eligible: list[
                tuple[float, str, dict[str, Any], dict[str, Any]]
            ] = []
            rejection_reasons: set[str] = set()
            withheld_target_simulation = False
            frozen_candidate_names = {
                str(row["name"])
                for row in candidate_protocol.get("simulations") or []
            }
            for candidate_simulation in candidate_snapshot.get("Simulations") or []:
                if not isinstance(candidate_simulation, dict):
                    continue
                candidate_simulation_name = str(
                    candidate_simulation.get("Name") or ""
                )
                if candidate_simulation_name not in frozen_candidate_names:
                    continue
                if (
                    same_semantic_snapshot
                    and candidate_simulation_name == source_simulation_name
                ):
                    withheld_target_simulation = True
                    rejection_reasons.add("WITHHELD_TARGET_SIMULATION")
                    continue
                try:
                    descriptor = _protocol_descriptor(
                        candidate_snapshot, candidate_simulation, primary
                    )
                    reasons = _context_reasons(
                        source_descriptor, descriptor, output_suffix
                    )
                    if reasons:
                        rejection_reasons.update(reasons)
                        continue
                    source_in_candidate_unit = _convert_dose(
                        source_descriptor["dose_value"],
                        source_descriptor["dose_unit"],
                        descriptor["dose_unit"],
                    )
                    log_distance = abs(
                        math.log(
                            source_in_candidate_unit / descriptor["dose_value"]
                        )
                    )
                    interval_penalty = float(
                        source_descriptor["dosing_interval"]
                        != descriptor["dosing_interval"]
                    )
                    eligible.append(
                        (
                            interval_penalty * 100.0 + log_distance,
                            str(candidate_simulation["Name"]),
                            candidate_simulation,
                            descriptor,
                        )
                    )
                except (AdaptationPlannerError, KeyError, TypeError, ValueError) as exc:
                    rejection_reasons.add(
                        f"TEMPLATE_NOT_ADAPTABLE:{type(exc).__name__}:{exc}"
                    )
            if not eligible:
                raise AdaptationPlannerError(
                    "No scenario-compatible template simulation: "
                    + "; ".join(sorted(rejection_reasons)[:20])
                )
            eligible.sort(key=lambda row: (row[0], row[1].casefold()))
            _, template_name, template_simulation, template_descriptor = eligible[0]
            adapted_name = f"ADAPT-{plan_key}"
            required_observation_end_time_hours = max(
                _time_in_hours(value, truth["time_unit"])
                for value in truth["time_values"]
            )
            adapted_snapshot, operations, execution_adjustments = _adapt_snapshot(
                candidate_snapshot=candidate_snapshot,
                template_simulation=template_simulation,
                template_descriptor=template_descriptor,
                source_simulation=source_simulation,
                source_descriptor=source_descriptor,
                primary=primary,
                adapted_name=adapted_name,
                required_observation_end_time_hours=(
                    required_observation_end_time_hours
                ),
            )
            if not operations:
                raise AdaptationPlannerError(
                    "Selected scenario is unchanged but was not recognized as an exact "
                    "executed scenario."
                )
            plan_root = output_directory / "plans" / plan_key
            snapshot_path = plan_root / "adapted_snapshot.json"
            snapshot_sha = _atomic_json(snapshot_path, adapted_snapshot)
            semantic_sha = snapshot_semantic_sha256(adapted_snapshot)
            observed = _adapted_observed_series(
                truth=truth,
                source_simulation_name=str(source_frozen["name"]),
                adapted_name=adapted_name,
            )
            source_lock = copy.deepcopy(candidate_protocol["source"])
            source_lock["snapshot_relative_path"] = "adapted_snapshot.json"
            source_lock["snapshot_declared_path"] = str(
                candidate_protocol["source"].get("snapshot_relative_path") or ""
            )
            source_lock["snapshot_resolution_mode"] = "declared_relative_path"
            source_lock["snapshot_sha256"] = snapshot_sha
            protocol_payload = {
                "protocol_schema_version": ADAPTATION_PROTOCOL_SCHEMA_VERSION,
                "protocol_builder_version": ADAPTATION_PLANNER_VERSION,
                "protocol_id": (
                    "snapshot-reuse-"
                    + _stable_hex(
                        "adaptation-protocol",
                        task_id,
                        candidate_id,
                        snapshot_sha,
                        length=32,
                    )
                ),
                "generated_at": datetime.now(UTC).isoformat(),
                "source": source_lock,
                "source_archive_path": candidate_protocol["source_archive_path"],
                "source_snapshot_semantic_sha256": semantic_sha,
                "counts": _protocol_counts(adapted_snapshot),
                "runtime": candidate_protocol["runtime"],
                "criteria": truth["criteria"],
                "simulations": [
                    {
                        "sequence": 1,
                        "name": adapted_name,
                        "role": "model_verification",
                        "individual": template_simulation.get("Individual"),
                        "observed_dataset_names": [truth["dataset_name"]],
                        "evaluation_series": observed,
                        "additional_evaluation_series": [],
                    }
                ],
            }
            LicensedSnapshotReuseProtocol.model_validate(protocol_payload)
            protocol_path = plan_root / "adaptation_protocol.json"
            protocol_sha = _atomic_json(protocol_path, protocol_payload)
            plan = {
                **base_record,
                "status": "planned",
                "source_protocol_key": source_key,
                "source_simulation_sequence": truth["source_simulation_sequence"],
                "source_simulation_name": source_simulation_name,
                "source_truth_location": truth["source_location"],
                "candidate_protocol_key": candidate_job["protocol_key"],
                "candidate_model_family": candidate["model_family"],
                "candidate_repository_commit": candidate_job.get(
                    "repository_commit"
                ),
                "template_simulation_name": template_name,
                "template_selection_policy": (
                    "leave-one-scenario-out when source and candidate share a "
                    "semantic snapshot; same application type, dose basis, "
                    "formulation, population, gender and output; then minimum "
                    "interval penalty and log-dose distance"
                ),
                "withheld_target_simulation": withheld_target_simulation,
                "base_snapshot_sha256": candidate_job["snapshot_sha256"],
                "adapted_snapshot_relative_path": snapshot_path.relative_to(
                    output_directory
                ).as_posix(),
                "adapted_snapshot_sha256": snapshot_sha,
                "adapted_snapshot_semantic_sha256": semantic_sha,
                "protocol_relative_path": protocol_path.relative_to(
                    output_directory
                ).as_posix(),
                "protocol_sha256": protocol_sha,
                "operations": operations,
                "adaptation_operation_multilabel": sorted(
                    {str(row["operation_type"]) for row in operations}
                ),
                "adaptation_operation_count": len(operations),
                "adaptation_weighted_cost": sum(
                    float(row["weight"]) for row in operations
                ),
                "execution_adjustments": execution_adjustments,
                "execution_adjustment_count": len(execution_adjustments),
                "created_at": datetime.now(UTC).isoformat(),
            }
            plan_path = plan_root / "adaptation_plan.json"
            plan["adaptation_plan_sha256"] = _atomic_json(plan_path, plan)
            planned.append(plan)
            protocol_rows.append(
                {
                    "protocol_key": plan_key,
                    "protocol_id": protocol_payload["protocol_id"],
                    "protocol_path": str(protocol_path.resolve()),
                    "protocol_sha256": protocol_sha,
                    "checkout_path": str(plan_root.resolve()),
                    "snapshot_relative_path": snapshot_path.name,
                    "full_name": (
                        f"adaptation/{candidate['model_family']}/{task_id}"
                    ),
                    "version_id": candidate_id,
                    "repository_commit": candidate_job["repository_commit"],
                    "model_name": candidate["model_family"],
                    "compounds": task["active_moieties"],
                    "model_verification_count": 1,
                    "adaptation_plan_id": plan["adaptation_plan_id"],
                }
            )
        except (
            AdaptationPlannerError,
            KeyError,
            StopIteration,
            TypeError,
            ValueError,
        ) as exc:
            blocked.append(
                {
                    **base_record,
                    "status": "blocked",
                    "reason_code": "CONTROLLED_ADAPTATION_NOT_EXECUTABLE",
                    "detail": str(exc),
                }
            )

    planned.sort(key=lambda row: str(row["adaptation_plan_id"]))
    blocked.sort(
        key=lambda row: (str(row["task_id"]), str(row["candidate_id"]))
    )
    protocol_rows.sort(key=lambda row: str(row["protocol_key"]))
    outputs = {
        "adaptation_plans.jsonl": _atomic_jsonl(
            output_directory / "adaptation_plans.jsonl", planned
        ),
        "blocked_adaptations.jsonl": _atomic_jsonl(
            output_directory / "blocked_adaptations.jsonl", blocked
        ),
        "protocols.jsonl": _atomic_jsonl(
            output_directory / "protocols.jsonl", protocol_rows
        ),
    }
    manifest = {
        "planner_version": ADAPTATION_PLANNER_VERSION,
        "batch_directory": str(batch_directory),
        "benchmark_directory": str(benchmark_directory),
        "batch_manifest_sha256": sha256_file(
            batch_directory / "batch_manifest.json"
        ),
        "benchmark_manifest_sha256": sha256_file(
            benchmark_directory / "manifest.json"
        ),
        "eligible_pair_count": len(planned) + len(blocked),
        "planned_count": len(planned),
        "blocked_count": len(blocked),
        "operation_type_counts": {
            operation_type: sum(
                operation_type in row["adaptation_operation_multilabel"]
                for row in planned
            )
            for operation_type in sorted(_OPERATION_WEIGHTS)
        },
        "execution_adjustment_type_counts": {
            "output_horizon": sum(
                any(
                    row.get("adjustment_type") == "output_horizon"
                    for row in plan.get("execution_adjustments") or []
                )
                for plan in planned
            )
        },
        "output_checksums": outputs,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(output_directory / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build provenance-locked controlled-adaptation snapshots and Windows "
            "execution protocols for same-compound benchmark pairs."
        )
    )
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = plan_controlled_adaptations(
        batch_directory=args.batch,
        benchmark_directory=args.benchmark,
        output_directory=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
