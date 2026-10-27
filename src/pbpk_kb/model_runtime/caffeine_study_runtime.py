from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from defusedxml import ElementTree as SafeET

from pbpk_kb.blueprints.caffeine_runtime_audit import (
    TARGET_OUTPUT,
    _target_series,
)
from pbpk_kb.blueprints.caffeine_study_snapshot import (
    CaffeineStudyScenarioManifest,
    RENAL_PROCESS_DATA_SOURCE,
)
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
from pbpk_kb.model_runtime.result_parser import compare_csv_files


AUDITOR_VERSION = "caffeine-study-runtime-auditor-1.1.0"
EXPECTED_SCENARIOS = (
    "CAL_BLANCHARD_IV",
    "CAL_BLANCHARD_PO",
    "HOLD_WAHLLANDER_IV",
    "EXT_EUDRACT_PO",
    "EXT_PMC535_PO",
    "EXT2_PMC572_PO_MALE",
    "EXT2_PMC572_PO_FEMALE",
)
RUN_IDS = ("run_1", "run_2")


class CaffeineStudyRuntimeError(RuntimeError):
    pass


class RuntimeArtifact(StrictBlueprintModel):
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(gt=0)


class StudyScenarioRuntimeAudit(StrictBlueprintModel):
    run_id: Literal["run_1", "run_2"]
    scenario_name: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    expected_route: Literal["Intravenous", "Oral"]
    observed_routes: list[str]
    observed_formulations: list[str]
    target_row_count: int = Field(gt=0)
    target_time_min: float
    target_time_max: float
    target_concentration_max: float = Field(gt=0)
    individual_age_years: float = Field(gt=0)
    individual_weight_kg: float = Field(gt=0)
    individual_height_m: float = Field(gt=0)
    individual_bmi_kg_per_m2: float = Field(gt=0)
    topology_semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: list[RuntimeArtifact] = Field(min_length=3, max_length=3)
    checks: dict[str, bool]
    accepted: bool


class CaffeineStudyRuntimeAudit(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    auditor_version: Literal[AUDITOR_VERSION] = AUDITOR_VERSION
    generated_at: datetime
    runtime_root: str = Field(min_length=1)
    scenario_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_names: list[str] = Field(min_length=7, max_length=7)
    scenarios: list[StudyScenarioRuntimeAudit] = Field(min_length=14, max_length=14)
    repeatability_checks: dict[str, bool]
    repeatability_details: list[dict[str, Any]]
    runtime_log_checks: dict[str, bool]
    runtime_log_issues: list[dict[str, str]]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


def _artifact(path: Path, root: Path) -> RuntimeArtifact:
    if not path.is_file() or path.stat().st_size <= 0:
        raise CaffeineStudyRuntimeError(f"Runtime artifact is missing: {path}")
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise CaffeineStudyRuntimeError(
            f"Runtime artifact escapes root: {path}"
        ) from exc
    return RuntimeArtifact(
        relative_path=relative,
        sha256=sha256_file(path),
        byte_size=path.stat().st_size,
    )


def _scenario_files(
    runtime_root: Path,
    run_id: str,
    scenario_name: str,
) -> tuple[Path, Path, Path]:
    root = runtime_root / run_id / scenario_name
    return (
        root / f"{scenario_name}-Results.csv",
        root / f"{scenario_name}.xml",
        root / f"{scenario_name}.pkml",
    )


def _individual_demographics(pkml_path: Path) -> dict[str, float]:
    raw: dict[str, str] = {}
    for _, element in SafeET.iterparse(pkml_path, events=("end",)):
        if element.tag.rsplit("}", 1)[-1] == "StringExtendedProperty":
            name = element.attrib.get("name")
            value = element.attrib.get("value")
            if name in {"Age", "Height", "Weight"} and value:
                if name in raw:
                    raise CaffeineStudyRuntimeError(
                        f"Duplicate individual metadata {name}: {pkml_path}"
                    )
                raw[name] = value
        element.clear()
    if set(raw) != {"Age", "Height", "Weight"}:
        raise CaffeineStudyRuntimeError(
            f"Incomplete individual metadata in PKML: {pkml_path}"
        )

    def parse_value(name: str) -> tuple[float, str]:
        match = re.fullmatch(
            r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+(.+?)\s*",
            raw[name],
        )
        if match is None:
            raise CaffeineStudyRuntimeError(
                f"Invalid {name} metadata {raw[name]!r}: {pkml_path}"
            )
        return float(match.group(1)), match.group(2)

    age, age_unit = parse_value("Age")
    height, height_unit = parse_value("Height")
    weight, weight_unit = parse_value("Weight")
    if age_unit != "year(s)" or weight_unit != "kg":
        raise CaffeineStudyRuntimeError(
            f"Unexpected age/weight units in PKML: {pkml_path}"
        )
    if height_unit == "cm":
        height /= 100.0
    elif height_unit != "m":
        raise CaffeineStudyRuntimeError(
            f"Unexpected height unit {height_unit!r}: {pkml_path}"
        )
    if not all(math.isfinite(item) and item > 0 for item in (age, height, weight)):
        raise CaffeineStudyRuntimeError(
            f"Non-positive individual metadata in PKML: {pkml_path}"
        )
    return {
        "age_years": age,
        "height_m": height,
        "weight_kg": weight,
        "bmi_kg_per_m2": weight / (height * height),
    }


def _instantiated_renal_processes(processes: list[Any]) -> list[Any]:
    return [
        process
        for process in processes
        if process.element_type == "Transport"
        and process.process_type == "renal_excretion"
        and RENAL_PROCESS_DATA_SOURCE in process.name
    ]


def _scenario_audit(
    *,
    runtime_root: Path,
    run_id: Literal["run_1", "run_2"],
    scenario: Any,
    individual: Any,
    expected_route: Literal["Intravenous", "Oral"],
) -> StudyScenarioRuntimeAudit:
    csv_path, xml_path, pkml_path = _scenario_files(
        runtime_root,
        run_id,
        scenario.name,
    )
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
        job_id="caffeine-study-runtime-v3",
        project_key=hashlib.sha256(
            f"caffeine-study-scenarios-v3:{scenario.name}".encode("utf-8")
        ).hexdigest()[:32],
        simulation_id=scenario.name,
        simulation_name=scenario.name,
        csv_results=[csv_summary],
    )
    target = _target_series(csv_path)
    cyp = [
        process
        for process in parsed.processes
        if process.element_type == "Reaction"
        and process.name == "Caffeine-CYP1A2-MM"
    ]
    renal = _instantiated_renal_processes(parsed.processes)
    required = [*cyp, *renal]
    unresolved = sum(
        len(process.unresolved_partner_identifiers) for process in required
    )
    administration_matches = [
        item
        for item in parsed.administrations
        if item.molecule_name == "Caffeine" and item.route == expected_route
    ]
    demographics = _individual_demographics(pkml_path)
    topology = {
        "species": parsed.summary.species,
        "compounds": parsed.summary.compounds,
        "enzymes": parsed.summary.enzymes,
        "routes": parsed.summary.routes,
        "formulations": parsed.summary.formulations,
        "administrations": [
            (item.route, item.molecule_name, item.formulation)
            for item in parsed.administrations
        ],
        "cyp": [
            (
                item.name,
                item.educts,
                item.products,
                item.modifiers,
                item.unresolved_partner_identifiers,
            )
            for item in cyp
        ],
        "renal": [
            (
                item.name,
                item.educts,
                item.products,
                item.modifiers,
                item.unresolved_partner_identifiers,
            )
            for item in renal
        ],
        "target_output": TARGET_OUTPUT,
        "target_unit": target.unit,
        "individual_demographics": demographics,
    }
    checks = {
        "pkml_parsed_without_warnings": (
            parsed.summary.parse_status == "parsed" and not parsed.summary.warnings
        ),
        "human_caffeine_cyp1a2_exact": (
            parsed.summary.species == ["Human"]
            and parsed.summary.compounds == ["Caffeine"]
            and parsed.summary.enzymes == ["CYP1A2"]
        ),
        "route_exact": parsed.summary.routes == [expected_route],
        "single_caffeine_administration": len(administration_matches) == 1
        and len(parsed.administrations) == 1,
        "oral_formulation_present_only_for_oral": (
            bool(parsed.summary.formulations) == (expected_route == "Oral")
        ),
        "cyp1a2_metabolism_exact": len(cyp) >= 1
        and all(
            item.educts == ["Caffeine"]
            and item.products == ["Caffeine-CYP1A2-MM Metabolite"]
            and item.modifiers == ["CYP1A2"]
            for item in cyp
        ),
        "renal_clearance_exact": len(renal) == 1
        and renal[0].educts == ["Caffeine"]
        and renal[0].products == ["Caffeine"],
        "required_process_participants_resolved": unresolved == 0,
        "target_csv_finite_nonnegative_nonzero": (
            csv_summary.parse_status == "parsed"
            and csv_summary.non_finite_value_count == 0
            and target.all_finite
            and target.all_nonnegative
            and target.concentration_max > 0
        ),
        "target_time_grid_valid": target.time_min == 0
        and target.time_max >= scenario.output_intervals[-1].end_h * 60,
        "individual_age_exact": math.isclose(
            demographics["age_years"],
            individual.age_years,
            rel_tol=0,
            abs_tol=1e-9,
        ),
        "reported_weight_exact_or_generated": (
            demographics["weight_kg"] > 0
            if individual.weight_kg is None
            else math.isclose(
                demographics["weight_kg"],
                individual.weight_kg,
                rel_tol=0,
                abs_tol=0.005,
            )
        ),
        "reported_height_exact_or_generated": (
            demographics["height_m"] > 0
            if individual.height_m is None
            else math.isclose(
                demographics["height_m"],
                individual.height_m,
                rel_tol=0,
                abs_tol=0.00005,
            )
        ),
        "contextual_bmi_not_encoded_as_reported_anthropometry": (
            individual.reported_bmi_kg_per_m2 is None
            or (
                individual.context_only_fields == ["bmi"]
                and set(individual.system_generated_fields)
                == {"weight", "height"}
            )
        ),
    }
    return StudyScenarioRuntimeAudit(
        run_id=run_id,
        scenario_name=scenario.name,
        dataset_id=scenario.dataset_id,
        split=scenario.split,
        expected_route=expected_route,
        observed_routes=parsed.summary.routes,
        observed_formulations=parsed.summary.formulations,
        target_row_count=target.row_count,
        target_time_min=target.time_min,
        target_time_max=target.time_max,
        target_concentration_max=target.concentration_max,
        individual_age_years=demographics["age_years"],
        individual_weight_kg=demographics["weight_kg"],
        individual_height_m=demographics["height_m"],
        individual_bmi_kg_per_m2=demographics["bmi_kg_per_m2"],
        topology_semantic_sha256=canonical_sha256(topology),
        artifacts=artifacts,
        checks=checks,
        accepted=all(checks.values()),
    )


def _log_issues(path: Path) -> list[str]:
    if not path.is_file() or path.stat().st_size <= 0:
        return ["LOG_MISSING_OR_EMPTY"]
    issues: list[str] = []
    pattern = re.compile(r"\b(warn(?:ing)?|error|fatal|exception)\b", re.I)
    zero_pattern = re.compile(r"\b0\s+(?:warnings?|errors?)\b", re.I)
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        if pattern.search(line) and not zero_pattern.search(line):
            issues.append(f"line {line_number}: {line.strip()}")
    return issues


def build_caffeine_study_runtime_audit(
    *,
    runtime_root: Path,
    scenario_manifest_path: Path,
    output_path: Path,
) -> CaffeineStudyRuntimeAudit:
    runtime_root = runtime_root.resolve()
    manifest = CaffeineStudyScenarioManifest.model_validate_json(
        scenario_manifest_path.read_text(encoding="utf-8")
    )
    if canonical_sha256(
        manifest.model_dump(mode="json", exclude={"checksum"})
    ) != manifest.checksum:
        raise CaffeineStudyRuntimeError("Scenario manifest checksum is invalid.")
    scenario_names = [item.name for item in manifest.simulations]
    if scenario_names != list(EXPECTED_SCENARIOS):
        raise CaffeineStudyRuntimeError("Scenario order or identity drifted.")
    protocols = {item.name: item for item in manifest.protocols}
    individuals = {item.name: item for item in manifest.individuals}
    audits: list[StudyScenarioRuntimeAudit] = []
    for run_id in RUN_IDS:
        for scenario in manifest.simulations:
            route = protocols[scenario.protocol_name].application_type
            audits.append(
                _scenario_audit(
                    runtime_root=runtime_root,
                    run_id=run_id,
                    scenario=scenario,
                    individual=individuals[scenario.individual_name],
                    expected_route=route,
                )
            )
    by_key = {(item.run_id, item.scenario_name): item for item in audits}
    repeatability_checks: dict[str, bool] = {}
    repeatability_details: list[dict[str, Any]] = []
    for scenario_name in EXPECTED_SCENARIOS:
        first = by_key[("run_1", scenario_name)]
        second = by_key[("run_2", scenario_name)]
        csv_a, _, _ = _scenario_files(runtime_root, "run_1", scenario_name)
        csv_b, _, _ = _scenario_files(runtime_root, "run_2", scenario_name)
        comparison = compare_csv_files(
            run_a_path=csv_a,
            run_b_path=csv_b,
            simulation_id=scenario_name,
            csv_ordinal=1,
            absolute_tolerance=1e-12,
            relative_tolerance=1e-12,
        )
        repeatability_checks[f"{scenario_name}_csv_repeatable"] = (
            comparison.accepted
        )
        repeatability_checks[f"{scenario_name}_topology_repeatable"] = (
            first.topology_semantic_sha256 == second.topology_semantic_sha256
        )
        repeatability_details.append(
            {
                "scenario_name": scenario_name,
                "csv_accepted": comparison.accepted,
                "maximum_absolute_difference": comparison.maximum_absolute_difference,
                "maximum_relative_difference": comparison.maximum_relative_difference,
                "mismatch_count": len(comparison.mismatches),
                "topology_semantic_identical": (
                    first.topology_semantic_sha256
                    == second.topology_semantic_sha256
                ),
            }
        )
    for run_id in RUN_IDS:
        blanchard_iv = by_key[(run_id, "CAL_BLANCHARD_IV")]
        blanchard_po = by_key[(run_id, "CAL_BLANCHARD_PO")]
        repeatability_checks[
            f"{run_id}_blanchard_crossover_demographics_match"
        ] = all(
            math.isclose(left, right, rel_tol=0, abs_tol=1e-12)
            for left, right in (
                (
                    blanchard_iv.individual_age_years,
                    blanchard_po.individual_age_years,
                ),
                (
                    blanchard_iv.individual_weight_kg,
                    blanchard_po.individual_weight_kg,
                ),
                (
                    blanchard_iv.individual_height_m,
                    blanchard_po.individual_height_m,
                ),
            )
        )
    runtime_log_checks: dict[str, bool] = {}
    runtime_log_issues: list[dict[str, str]] = []
    for run_id in RUN_IDS:
        log_path = runtime_root / "logs" / f"export_{run_id}.log"
        issues = _log_issues(log_path)
        runtime_log_checks[f"{run_id}_log_clean"] = not issues
        runtime_log_issues.extend(
            {"run_id": run_id, "issue": issue} for issue in issues
        )
    accepted = (
        all(item.accepted for item in audits)
        and all(repeatability_checks.values())
        and all(runtime_log_checks.values())
    )
    payload = {
        "generated_at": datetime.now(UTC),
        "runtime_root": runtime_root.as_posix(),
        "scenario_manifest_sha256": sha256_file(scenario_manifest_path),
        "scenario_manifest_checksum": manifest.checksum,
        "scenario_names": scenario_names,
        "scenarios": audits,
        "repeatability_checks": repeatability_checks,
        "repeatability_details": repeatability_details,
        "runtime_log_checks": runtime_log_checks,
        "runtime_log_issues": runtime_log_issues,
        "accepted": accepted,
    }
    draft = CaffeineStudyRuntimeAudit(**payload, checksum="0" * 64)
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
        description="Audit repeatable seven-scenario caffeine PK-Sim exports."
    )
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--scenario-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    audit = build_caffeine_study_runtime_audit(
        runtime_root=args.runtime_root,
        scenario_manifest_path=args.scenario_manifest,
        output_path=args.output,
    )
    print(json.dumps(audit.model_dump(mode="json"), ensure_ascii=False))
    if not audit.accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
