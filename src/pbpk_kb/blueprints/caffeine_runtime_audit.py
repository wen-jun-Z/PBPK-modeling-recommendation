from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from pbpk_kb.blueprints.schemas import StrictBlueprintModel
from pbpk_kb.model_assets.execution_plan import (
    canonical_sha256,
    write_json_atomic,
)
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.osp_pkml_parser import (
    parse_pkml_simulation,
    parse_result_csv,
)


RUNTIME_AUDITOR_VERSION = "caffeine-runtime-auditor-1.0.0"
PROJECT_KEY = "cafe12a2cafe12a2cafe12a2cafe12a2"
TARGET_OUTPUT = (
    "Organism|PeripheralVenousBlood|Caffeine|"
    "Plasma (Peripheral Venous Blood)"
)
EXPECTED_SCENARIOS = {
    "S1": {"route": "Oral", "formulation": "Weibull"},
    "S2": {"route": "IntravenousBolus", "formulation": None},
}


class CaffeineRuntimeAuditError(RuntimeError):
    pass


class RuntimeArtifact(StrictBlueprintModel):
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(gt=0)


class TargetSeriesSummary(StrictBlueprintModel):
    output_name: Literal[TARGET_OUTPUT] = TARGET_OUTPUT
    unit: str = Field(min_length=1)
    row_count: int = Field(gt=0)
    time_min: float
    time_max: float
    concentration_min: float
    concentration_max: float
    all_finite: bool
    all_nonnegative: bool
    time_strictly_increasing: bool


class ScenarioRuntimeAudit(StrictBlueprintModel):
    run_id: Literal["run_1", "run_2"]
    simulation_name: Literal["S1", "S2"]
    expected_route: str
    expected_formulation: str | None = None
    summary: dict[str, Any]
    administrations: list[dict[str, Any]]
    target_series: TargetSeriesSummary
    metabolism_reaction_count: int = Field(ge=0)
    renal_transport_count: int = Field(ge=0)
    unresolved_required_participant_count: int = Field(ge=0)
    topology_semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: list[RuntimeArtifact] = Field(min_length=3)
    checks: dict[str, bool]
    accepted: bool


class CaffeineTemplateRuntimeAudit(StrictBlueprintModel):
    audit_schema_version: Literal["1.0.0"] = "1.0.0"
    auditor_version: Literal[RUNTIME_AUDITOR_VERSION] = RUNTIME_AUDITOR_VERSION
    generated_at: datetime
    runtime_root: str
    scenarios: list[ScenarioRuntimeAudit] = Field(min_length=4, max_length=4)
    repeatability_checks: dict[str, bool]
    runtime_log_checks: dict[str, bool]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


def _artifact(path: Path, root: Path) -> RuntimeArtifact:
    if not path.is_file() or path.stat().st_size <= 0:
        raise CaffeineRuntimeAuditError(f"Runtime artifact is missing: {path}")
    return RuntimeArtifact(
        relative_path=path.relative_to(root).as_posix(),
        sha256=sha256_file(path),
        byte_size=path.stat().st_size,
    )


def _target_series(path: Path) -> TargetSeriesSummary:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(
            line for line in handle if line.strip() and not line.startswith("#")
        )
        if not reader.fieldnames:
            raise CaffeineRuntimeAuditError(f"CSV has no header: {path}")
        target_header = next(
            (
                name
                for name in reader.fieldnames
                if name == TARGET_OUTPUT or name.startswith(TARGET_OUTPUT + " [")
            ),
            None,
        )
        time_header = next(
            (
                name
                for name in reader.fieldnames
                if name == "Time" or name.startswith("Time [")
            ),
            None,
        )
        if target_header is None or time_header is None:
            raise CaffeineRuntimeAuditError(
                f"CSV lacks target concentration or time: {path}"
            )
        unit = (
            target_header[len(TARGET_OUTPUT) + 2 : -1]
            if target_header.startswith(TARGET_OUTPUT + " [")
            else "unknown"
        )
        times: list[float] = []
        concentrations: list[float] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                times.append(float(row[time_header]))
                concentrations.append(float(row[target_header]))
            except (TypeError, ValueError) as exc:
                raise CaffeineRuntimeAuditError(
                    f"Invalid target series value at {path}:{row_number}"
                ) from exc
    if not times:
        raise CaffeineRuntimeAuditError(f"Target series is empty: {path}")
    all_finite = all(
        math.isfinite(value) for value in (*times, *concentrations)
    )
    finite_times = [value for value in times if math.isfinite(value)]
    finite_concentrations = [
        value for value in concentrations if math.isfinite(value)
    ]
    if not finite_times or not finite_concentrations:
        raise CaffeineRuntimeAuditError(f"Target series is non-finite: {path}")
    return TargetSeriesSummary(
        unit=unit,
        row_count=len(times),
        time_min=min(finite_times),
        time_max=max(finite_times),
        concentration_min=min(finite_concentrations),
        concentration_max=max(finite_concentrations),
        all_finite=all_finite,
        all_nonnegative=all(value >= 0 for value in finite_concentrations),
        time_strictly_increasing=all(
            right > left for left, right in zip(times, times[1:])
        ),
    )


def _scenario_audit(
    *,
    runtime_root: Path,
    run_id: Literal["run_1", "run_2"],
    simulation_name: Literal["S1", "S2"],
) -> ScenarioRuntimeAudit:
    expected = EXPECTED_SCENARIOS[simulation_name]
    scenario_root = runtime_root / run_id / simulation_name
    csv_path = scenario_root / f"{simulation_name}-Results.csv"
    xml_path = scenario_root / f"{simulation_name}.xml"
    pkml_path = scenario_root / f"{simulation_name}.pkml"
    artifacts = [
        _artifact(csv_path, runtime_root),
        _artifact(xml_path, runtime_root),
        _artifact(pkml_path, runtime_root),
    ]
    csv_summary = parse_result_csv(
        csv_path,
        artifact_relative_path=artifacts[0].relative_path,
        source_relative_path=csv_path.name,
        expected_sha256=artifacts[0].sha256,
    )
    parsed = parse_pkml_simulation(
        pkml_path,
        artifact_relative_path=artifacts[2].relative_path,
        artifact_sha256=artifacts[2].sha256,
        job_id="caffeine-cyp1a2-runtime-v2",
        project_key=PROJECT_KEY,
        simulation_id="s0001" if simulation_name == "S1" else "s0002",
        simulation_name=simulation_name,
        csv_results=[csv_summary],
    )
    metabolism = [
        process
        for process in parsed.processes
        if process.element_type == "Reaction"
        and process.name == "Caffeine-CYP1A2-MM"
    ]
    renal = [
        process
        for process in parsed.processes
        if process.element_type == "Transport"
        and process.process_type == "renal_excretion"
        and process.name == "Renal Clearances-Birkett and Miners 1991-Caffeine"
    ]
    required_processes = [*metabolism, *renal]
    unresolved_count = sum(
        len(process.unresolved_partner_identifiers)
        for process in required_processes
    )
    administrations = [
        {
            "route": item.route,
            "molecule_name": item.molecule_name,
            "formulation": item.formulation,
        }
        for item in parsed.administrations
    ]
    target_series = _target_series(csv_path)
    topology = {
        "species": parsed.summary.species,
        "compounds": parsed.summary.compounds,
        "enzymes": parsed.summary.enzymes,
        "routes": parsed.summary.routes,
        "formulations": parsed.summary.formulations,
        "administrations": administrations,
        "metabolism": sorted(
            (
                process.name,
                tuple(process.educts),
                tuple(process.products),
                tuple(process.modifiers),
                tuple(process.unresolved_partner_identifiers),
            )
            for process in metabolism
        ),
        "renal": sorted(
            (
                process.name,
                tuple(process.educts),
                tuple(process.products),
                tuple(process.modifiers),
                tuple(process.unresolved_partner_identifiers),
            )
            for process in renal
        ),
        "target_output": target_series.output_name,
        "target_unit": target_series.unit,
    }
    metabolism_valid = bool(metabolism) and all(
        process.educts == ["Caffeine"]
        and process.products == ["Caffeine-CYP1A2-MM Metabolite"]
        and process.modifiers == ["CYP1A2"]
        for process in metabolism
    )
    renal_valid = len(renal) == 1 and all(
        process.educts == ["Caffeine"]
        and process.products == ["Caffeine"]
        for process in renal
    )
    checks = {
        "pkml_parsed_without_warnings": (
            parsed.summary.parse_status == "parsed"
            and not parsed.summary.warnings
        ),
        "human_caffeine_cyp1a2_exact": (
            parsed.summary.species == ["Human"]
            and parsed.summary.compounds == ["Caffeine"]
            and parsed.summary.enzymes == ["CYP1A2"]
        ),
        "route_exact": parsed.summary.routes == [expected["route"]],
        "formulation_exact": parsed.summary.formulations
        == ([expected["formulation"]] if expected["formulation"] else []),
        "administration_exact": administrations
        == [
            {
                "route": expected["route"],
                "molecule_name": "Caffeine",
                "formulation": expected["formulation"],
            }
        ],
        "cyp1a2_metabolism_instantiated": metabolism_valid,
        "renal_clearance_instantiated": renal_valid,
        "required_process_participants_resolved": unresolved_count == 0,
        "csv_parsed_finite": (
            csv_summary.parse_status == "parsed"
            and csv_summary.non_finite_value_count == 0
            and target_series.all_finite
        ),
        "target_concentration_nonnegative_and_nonzero": (
            target_series.all_nonnegative
            and target_series.concentration_max > 0
        ),
        "time_grid_valid": (
            target_series.time_min == 0
            and target_series.time_max > 0
            and target_series.time_strictly_increasing
        ),
    }
    return ScenarioRuntimeAudit(
        run_id=run_id,
        simulation_name=simulation_name,
        expected_route=expected["route"],
        expected_formulation=expected["formulation"],
        summary={
            "species": parsed.summary.species,
            "compounds": parsed.summary.compounds,
            "enzymes": parsed.summary.enzymes,
            "routes": parsed.summary.routes,
            "formulations": parsed.summary.formulations,
            "parameter_count": parsed.summary.parameter_count,
            "component_count": parsed.summary.component_count,
            "process_count": parsed.summary.process_count,
            "warnings": parsed.summary.warnings,
        },
        administrations=administrations,
        target_series=target_series,
        metabolism_reaction_count=len(metabolism),
        renal_transport_count=len(renal),
        unresolved_required_participant_count=unresolved_count,
        topology_semantic_sha256=canonical_sha256(topology),
        artifacts=artifacts,
        checks=checks,
        accepted=all(checks.values()),
    )


def build_caffeine_runtime_audit(
    *,
    runtime_root: Path,
    output_path: Path,
) -> CaffeineTemplateRuntimeAudit:
    runtime_root = runtime_root.resolve()
    scenarios = [
        _scenario_audit(
            runtime_root=runtime_root,
            run_id=run_id,
            simulation_name=simulation_name,
        )
        for run_id in ("run_1", "run_2")
        for simulation_name in ("S1", "S2")
    ]
    by_key = {
        (item.run_id, item.simulation_name): item for item in scenarios
    }
    repeatability: dict[str, bool] = {}
    for simulation_name in EXPECTED_SCENARIOS:
        first = by_key[("run_1", simulation_name)]
        second = by_key[("run_2", simulation_name)]
        first_artifacts = {
            Path(item.relative_path).suffix: item for item in first.artifacts
        }
        second_artifacts = {
            Path(item.relative_path).suffix: item for item in second.artifacts
        }
        repeatability[f"{simulation_name}_csv_byte_identical"] = (
            first_artifacts[".csv"].sha256 == second_artifacts[".csv"].sha256
        )
        repeatability[f"{simulation_name}_simmodel_xml_byte_identical"] = (
            first_artifacts[".xml"].sha256 == second_artifacts[".xml"].sha256
        )
        repeatability[f"{simulation_name}_topology_semantic_identical"] = (
            first.topology_semantic_sha256
            == second.topology_semantic_sha256
        )
    log_checks: dict[str, bool] = {}
    for run_id in ("run_1", "run_2"):
        log_path = runtime_root / "logs" / f"export_{run_id}.log"
        text = log_path.read_text(encoding="utf-8", errors="replace").casefold()
        log_checks[f"{run_id}_log_exists_nonempty"] = (
            log_path.is_file() and log_path.stat().st_size > 0
        )
        log_checks[f"{run_id}_log_no_warning"] = "warning" not in text
        log_checks[f"{run_id}_log_no_error"] = "error" not in text
    accepted = (
        all(item.accepted for item in scenarios)
        and all(repeatability.values())
        and all(log_checks.values())
    )
    payload = {
        "audit_schema_version": "1.0.0",
        "auditor_version": RUNTIME_AUDITOR_VERSION,
        "generated_at": datetime.now(UTC),
        "runtime_root": runtime_root.as_posix(),
        "scenarios": scenarios,
        "repeatability_checks": repeatability,
        "runtime_log_checks": log_checks,
        "accepted": accepted,
    }
    draft = CaffeineTemplateRuntimeAudit(**payload, checksum="0" * 64)
    audit = draft.model_copy(
        update={
            "checksum": canonical_sha256(
                draft.model_dump(mode="json", exclude={"checksum"})
            )
        }
    )
    write_json_atomic(output_path, audit, canonical=True)
    return audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit repeatable caffeine PK-Sim runtime exports."
    )
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    audit = build_caffeine_runtime_audit(
        runtime_root=args.runtime_root,
        output_path=args.output,
    )
    print(json.dumps(audit.model_dump(mode="json"), ensure_ascii=False))
    if not audit.accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
