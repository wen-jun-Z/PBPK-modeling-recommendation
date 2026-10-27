from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field

from pbpk_kb.blueprints.caffeine_runtime_audit import (
    TARGET_OUTPUT,
    _target_series,
)
from pbpk_kb.blueprints.caffeine_study_snapshot import (
    CaffeineStudyScenarioManifest,
)
from pbpk_kb.blueprints.caffeine_v2_snapshot import (
    EXPECTED_SIMULATIONS,
    PATHWAYS,
    CaffeineV2RoundTripAudit,
    CaffeineV2ScenarioManifest,
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
from pbpk_kb.model_runtime.caffeine_study_runtime import (
    RuntimeArtifact,
    _artifact,
    _individual_demographics,
    _instantiated_renal_processes,
    _log_issues,
)
from pbpk_kb.model_runtime.result_parser import compare_csv_files


AUDITOR_VERSION = "caffeine-v2-runtime-auditor-1.0.0"
RUN_IDS = ("run_1", "run_2")
EXPECTED_REACTIONS = {
    f"Caffeine-CYP1A2-{data_source}": (
        f"Caffeine-CYP1A2-{data_source} Metabolite"
    )
    for data_source, _, _ in PATHWAYS
}


class CaffeineV2RuntimeError(RuntimeError):
    pass


class CaffeineV2ScenarioRuntimeAudit(StrictBlueprintModel):
    run_id: Literal["run_1", "run_2"]
    scenario_name: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    expected_route: Literal["Intravenous", "Oral"]
    observed_routes: list[str]
    observed_formulations: list[str]
    observed_metabolites: list[str]
    pathway_reaction_counts: dict[str, int]
    renal_transport_count: int = Field(ge=0)
    unresolved_required_participant_count: int = Field(ge=0)
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


class CaffeineV2RuntimeAudit(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    auditor_version: Literal[AUDITOR_VERSION] = AUDITOR_VERSION
    generated_at: datetime
    windows_run_root: str = Field(min_length=1)
    v2_scenario_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v2_scenario_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_scenario_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_scenario_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    roundtrip_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    roundtrip_audit_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    windows_run_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_manifest_row_count: int = Field(gt=0)
    lineage_checks: dict[str, bool]
    artifact_manifest_checks: dict[str, bool]
    scenarios: list[CaffeineV2ScenarioRuntimeAudit] = Field(
        min_length=14,
        max_length=14,
    )
    repeatability_checks: dict[str, bool]
    repeatability_details: list[dict[str, Any]]
    runtime_log_checks: dict[str, bool]
    runtime_log_issues: list[dict[str, str]]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


def _checksum_valid(model: StrictBlueprintModel) -> bool:
    return canonical_sha256(
        model.model_dump(mode="json", exclude={"checksum"})
    ) == getattr(model, "checksum")


def _verify_windows_file_manifest(
    windows_run_root: Path,
) -> tuple[dict[str, bool], int]:
    path = windows_run_root / "files.sha256.csv"
    if not path.is_file() or path.stat().st_size <= 0:
        raise CaffeineV2RuntimeError("Windows file manifest is missing.")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"RelativePath", "ByteSize", "SHA256"}
    if not rows or not required.issubset(rows[0]):
        raise CaffeineV2RuntimeError("Windows file manifest schema is invalid.")
    relative_paths: list[str] = []
    safe = True
    complete = True
    sizes_match = True
    hashes_match = True
    for row in rows:
        relative = row["RelativePath"].replace("\\", "/")
        parsed = PurePosixPath(relative)
        is_safe = (
            not parsed.is_absolute()
            and ".." not in parsed.parts
            and not (parsed.parts and ":" in parsed.parts[0])
        )
        safe = safe and is_safe
        relative_paths.append(relative)
        if not is_safe:
            continue
        artifact = windows_run_root / relative
        if not artifact.is_file():
            complete = False
            continue
        try:
            expected_size = int(row["ByteSize"])
        except (TypeError, ValueError) as exc:
            raise CaffeineV2RuntimeError(
                f"Invalid byte size for {relative}."
            ) from exc
        sizes_match = sizes_match and artifact.stat().st_size == expected_size
        hashes_match = hashes_match and (
            sha256_file(artifact) == row["SHA256"].lower()
        )
    checks = {
        "file_manifest_has_50_rows": len(rows) == 50,
        "file_manifest_paths_unique": len(relative_paths)
        == len(set(relative_paths)),
        "file_manifest_paths_safe": safe,
        "file_manifest_files_complete": complete,
        "file_manifest_byte_sizes_match": sizes_match,
        "file_manifest_sha256_match": hashes_match,
    }
    return checks, len(rows)


def _scenario_paths(
    windows_run_root: Path,
    run_id: str,
    scenario_name: str,
) -> tuple[Path, Path, Path]:
    root = windows_run_root / "exports" / run_id / scenario_name
    return (
        root / f"{scenario_name}-Results.csv",
        root / f"{scenario_name}.xml",
        root / f"{scenario_name}.pkml",
    )


def _scenario_audit(
    *,
    windows_run_root: Path,
    run_id: Literal["run_1", "run_2"],
    scenario: Any,
    individual: Any,
    expected_route: Literal["Intravenous", "Oral"],
    runtime_job_id: str = "caffeine-study-runtime-v4",
    project_key_prefix: str = "caffeine-study-scenarios-v4",
) -> CaffeineV2ScenarioRuntimeAudit:
    csv_path, xml_path, pkml_path = _scenario_paths(
        windows_run_root,
        run_id,
        scenario.name,
    )
    artifacts = [
        _artifact(csv_path, windows_run_root),
        _artifact(xml_path, windows_run_root),
        _artifact(pkml_path, windows_run_root),
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
        job_id=runtime_job_id,
        project_key=hashlib.sha256(
            f"{project_key_prefix}:{scenario.name}".encode("utf-8")
        ).hexdigest()[:32],
        simulation_id=scenario.name,
        simulation_name=scenario.name,
        csv_results=[csv_summary],
    )
    target = _target_series(csv_path)
    caffeine_cyp1a2 = [
        process
        for process in parsed.processes
        if process.element_type == "Reaction"
        and process.process_type == "metabolism"
        and process.educts == ["Caffeine"]
        and process.modifiers == ["CYP1A2"]
    ]
    pathway_counts = Counter(process.name for process in caffeine_cyp1a2)
    renal = _instantiated_renal_processes(parsed.processes)
    required = [*caffeine_cyp1a2, *renal]
    unresolved = sum(
        len(process.unresolved_partner_identifiers) for process in required
    )
    administration_matches = [
        item
        for item in parsed.administrations
        if item.molecule_name == "Caffeine" and item.route == expected_route
    ]
    demographics = _individual_demographics(pkml_path)
    pathway_participants_exact = all(
        process.name in EXPECTED_REACTIONS
        and process.products == [EXPECTED_REACTIONS[process.name]]
        and process.educts == ["Caffeine"]
        and process.modifiers == ["CYP1A2"]
        for process in caffeine_cyp1a2
    )
    topology = {
        "species": parsed.summary.species,
        "compounds": parsed.summary.compounds,
        "metabolites": parsed.summary.metabolites,
        "enzymes": parsed.summary.enzymes,
        "routes": parsed.summary.routes,
        "formulations": parsed.summary.formulations,
        "administrations": [
            (item.route, item.molecule_name, item.formulation)
            for item in parsed.administrations
        ],
        "cyp1a2_pathways": sorted(
            (
                item.name,
                item.educts,
                item.products,
                item.modifiers,
                item.unresolved_partner_identifiers,
            )
            for item in caffeine_cyp1a2
        ),
        "renal": sorted(
            (
                item.name,
                item.educts,
                item.products,
                item.modifiers,
                item.unresolved_partner_identifiers,
            )
            for item in renal
        ),
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
        "three_expected_metabolites_exact": set(parsed.summary.metabolites)
        == set(EXPECTED_REACTIONS.values()),
        "route_exact": parsed.summary.routes == [expected_route],
        "single_caffeine_administration": len(administration_matches) == 1
        and len(parsed.administrations) == 1,
        "oral_formulation_present_only_for_oral": bool(
            parsed.summary.formulations
        )
        == (expected_route == "Oral"),
        "three_pathway_names_exact": set(pathway_counts)
        == set(EXPECTED_REACTIONS),
        "each_pathway_instantiated": all(
            pathway_counts[name] > 0 for name in EXPECTED_REACTIONS
        ),
        "pathway_participants_exact": pathway_participants_exact,
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
        "reported_weight_exact_or_generated": demographics["weight_kg"] > 0
        if individual.weight_kg is None
        else math.isclose(
            demographics["weight_kg"],
            individual.weight_kg,
            rel_tol=0,
            abs_tol=0.005,
        ),
        "reported_height_exact_or_generated": demographics["height_m"] > 0
        if individual.height_m is None
        else math.isclose(
            demographics["height_m"],
            individual.height_m,
            rel_tol=0,
            abs_tol=0.00005,
        ),
    }
    return CaffeineV2ScenarioRuntimeAudit(
        run_id=run_id,
        scenario_name=scenario.name,
        dataset_id=scenario.dataset_id,
        split=scenario.split,
        expected_route=expected_route,
        observed_routes=parsed.summary.routes,
        observed_formulations=parsed.summary.formulations,
        observed_metabolites=parsed.summary.metabolites,
        pathway_reaction_counts=dict(sorted(pathway_counts.items())),
        renal_transport_count=len(renal),
        unresolved_required_participant_count=unresolved,
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


def build_caffeine_v2_runtime_audit(
    *,
    windows_run_root: Path,
    v2_scenario_manifest_path: Path,
    source_scenario_manifest_path: Path,
    roundtrip_audit_path: Path,
    output_path: Path,
) -> CaffeineV2RuntimeAudit:
    if output_path.exists():
        raise CaffeineV2RuntimeError(f"Immutable output exists: {output_path}")
    windows_run_root = windows_run_root.resolve()
    v2_manifest = CaffeineV2ScenarioManifest.model_validate_json(
        v2_scenario_manifest_path.read_text(encoding="utf-8")
    )
    source_manifest = CaffeineStudyScenarioManifest.model_validate_json(
        source_scenario_manifest_path.read_text(encoding="utf-8")
    )
    roundtrip = CaffeineV2RoundTripAudit.model_validate_json(
        roundtrip_audit_path.read_text(encoding="utf-8")
    )
    windows_manifest_path = windows_run_root / "windows_run_manifest.json"
    windows_manifest = json.loads(
        windows_manifest_path.read_text(encoding="utf-8-sig")
    )
    artifact_checks, manifest_rows = _verify_windows_file_manifest(
        windows_run_root
    )
    lineage_checks = {
        "v2_manifest_checksum_valid": _checksum_valid(v2_manifest),
        "source_manifest_checksum_valid": _checksum_valid(source_manifest),
        "roundtrip_audit_checksum_valid": _checksum_valid(roundtrip),
        "roundtrip_audit_accepted": roundtrip.accepted,
        "roundtrip_has_no_semantic_differences": not roundtrip.semantic_differences,
        "source_manifest_sha_matches_v2": sha256_file(
            source_scenario_manifest_path
        )
        == v2_manifest.source_scenario_manifest_sha256,
        "source_manifest_checksum_matches_v2": source_manifest.checksum
        == v2_manifest.source_scenario_manifest_checksum,
        "source_individuals_match_v2": [
            item.model_dump(mode="json") for item in source_manifest.individuals
        ]
        == v2_manifest.inherited_individuals,
        "source_protocols_match_v2": [
            item.model_dump(mode="json") for item in source_manifest.protocols
        ]
        == v2_manifest.inherited_protocols,
        "source_simulations_match_v2": [
            item.model_dump(mode="json") for item in source_manifest.simulations
        ]
        == v2_manifest.inherited_simulations,
        "scenario_split_map_matches_v2": {
            item.name: item.split for item in source_manifest.simulations
        }
        == v2_manifest.split_by_simulation,
        "windows_run_completed": windows_manifest.get("Status") == "completed",
        "windows_run_has_seven_scenarios": windows_manifest.get("ScenarioCount")
        == 7,
        "windows_run_has_two_runs": windows_manifest.get("RunCount") == 2,
        "windows_run_locked_v4_snapshot": windows_manifest.get(
            "SnapshotFileName"
        )
        == "caffeine_study_scenarios_v4.json",
    }
    scenario_names = [item.name for item in source_manifest.simulations]
    if scenario_names != list(EXPECTED_SIMULATIONS):
        raise CaffeineV2RuntimeError("Scenario order or identity drifted.")
    protocols = {item.name: item for item in source_manifest.protocols}
    individuals = {item.name: item for item in source_manifest.individuals}
    scenarios: list[CaffeineV2ScenarioRuntimeAudit] = []
    for run_id in RUN_IDS:
        for scenario in source_manifest.simulations:
            scenarios.append(
                _scenario_audit(
                    windows_run_root=windows_run_root,
                    run_id=run_id,
                    scenario=scenario,
                    individual=individuals[scenario.individual_name],
                    expected_route=protocols[
                        scenario.protocol_name
                    ].application_type,
                )
            )
    by_key = {(item.run_id, item.scenario_name): item for item in scenarios}
    repeatability_checks: dict[str, bool] = {}
    repeatability_details: list[dict[str, Any]] = []
    for scenario_name in EXPECTED_SIMULATIONS:
        first = by_key[("run_1", scenario_name)]
        second = by_key[("run_2", scenario_name)]
        csv_a, xml_a, _ = _scenario_paths(
            windows_run_root, "run_1", scenario_name
        )
        csv_b, xml_b, _ = _scenario_paths(
            windows_run_root, "run_2", scenario_name
        )
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
        repeatability_checks[f"{scenario_name}_simmodel_xml_byte_identical"] = (
            sha256_file(xml_a) == sha256_file(xml_b)
        )
        repeatability_checks[f"{scenario_name}_topology_repeatable"] = (
            first.topology_semantic_sha256 == second.topology_semantic_sha256
        )
        repeatability_details.append(
            {
                "scenario_name": scenario_name,
                "csv_accepted": comparison.accepted,
                "maximum_absolute_difference": (
                    comparison.maximum_absolute_difference
                ),
                "maximum_relative_difference": (
                    comparison.maximum_relative_difference
                ),
                "mismatch_count": len(comparison.mismatches),
                "simmodel_xml_byte_identical": sha256_file(xml_a)
                == sha256_file(xml_b),
                "topology_semantic_identical": (
                    first.topology_semantic_sha256
                    == second.topology_semantic_sha256
                ),
            }
        )
    runtime_log_checks: dict[str, bool] = {}
    runtime_log_issues: list[dict[str, str]] = []
    for log_name in (
        "snapshot_to_project.log",
        "project_to_snapshot.log",
        "export_run_1.log",
        "export_run_2.log",
    ):
        issues = _log_issues(windows_run_root / "logs" / log_name)
        runtime_log_checks[f"{log_name}_clean"] = not issues
        runtime_log_issues.extend(
            {"log_name": log_name, "issue": issue} for issue in issues
        )
    accepted = (
        all(lineage_checks.values())
        and all(artifact_checks.values())
        and all(item.accepted for item in scenarios)
        and all(repeatability_checks.values())
        and all(runtime_log_checks.values())
    )
    payload = {
        "generated_at": datetime.now(UTC),
        "windows_run_root": windows_run_root.as_posix(),
        "v2_scenario_manifest_sha256": sha256_file(
            v2_scenario_manifest_path
        ),
        "v2_scenario_manifest_checksum": v2_manifest.checksum,
        "source_scenario_manifest_sha256": sha256_file(
            source_scenario_manifest_path
        ),
        "source_scenario_manifest_checksum": source_manifest.checksum,
        "roundtrip_audit_sha256": sha256_file(roundtrip_audit_path),
        "roundtrip_audit_checksum": roundtrip.checksum,
        "windows_run_manifest_sha256": sha256_file(windows_manifest_path),
        "file_manifest_row_count": manifest_rows,
        "lineage_checks": lineage_checks,
        "artifact_manifest_checks": artifact_checks,
        "scenarios": scenarios,
        "repeatability_checks": repeatability_checks,
        "repeatability_details": repeatability_details,
        "runtime_log_checks": runtime_log_checks,
        "runtime_log_issues": runtime_log_issues,
        "accepted": accepted,
    }
    draft = CaffeineV2RuntimeAudit(**payload, checksum="0" * 64)
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
        description="Audit the caffeine v2 seven-scenario PK-Sim runtime."
    )
    parser.add_argument("--windows-run-root", type=Path, required=True)
    parser.add_argument("--v2-scenario-manifest", type=Path, required=True)
    parser.add_argument("--source-scenario-manifest", type=Path, required=True)
    parser.add_argument("--roundtrip-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    audit = build_caffeine_v2_runtime_audit(
        windows_run_root=args.windows_run_root,
        v2_scenario_manifest_path=args.v2_scenario_manifest,
        source_scenario_manifest_path=args.source_scenario_manifest,
        roundtrip_audit_path=args.roundtrip_audit,
        output_path=args.output,
    )
    print(json.dumps(audit.model_dump(mode="json"), ensure_ascii=False))
    if not audit.accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
