from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from pbpk_kb.blueprints.caffeine_study_snapshot import (
    StudyIndividualSpec,
    StudyProtocolSpec,
    StudySimulationSpec,
)
from pbpk_kb.blueprints.caffeine_v2_snapshot import (
    EXPECTED_SIMULATIONS,
    CaffeineV2ScenarioManifest,
)
from pbpk_kb.blueprints.caffeine_v3_snapshot import (
    CaffeineV3RoundTripAudit,
    CaffeineV3ScenarioManifest,
)
from pbpk_kb.blueprints.schemas import StrictBlueprintModel
from pbpk_kb.model_assets.execution_plan import (
    canonical_sha256,
    write_json_atomic,
)
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_runtime.caffeine_study_runtime import _log_issues
from pbpk_kb.model_runtime.caffeine_v2_runtime import (
    RUN_IDS,
    CaffeineV2ScenarioRuntimeAudit,
    _scenario_audit,
    _scenario_paths,
    _verify_windows_file_manifest,
)
from pbpk_kb.model_runtime.result_parser import compare_csv_files


AUDITOR_VERSION = "caffeine-v3-runtime-auditor-1.0.0"


class CaffeineV3RuntimeError(RuntimeError):
    pass


class CaffeineV3RuntimeAudit(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    auditor_version: Literal[AUDITOR_VERSION] = AUDITOR_VERSION
    generated_at: datetime
    windows_run_root: str = Field(min_length=1)
    v3_scenario_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_scenario_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
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


def _inherited_study_specs(
    manifest: CaffeineV2ScenarioManifest,
) -> tuple[
    list[StudyIndividualSpec],
    list[StudyProtocolSpec],
    list[StudySimulationSpec],
]:
    """Restore the typed study contract embedded by the immediate v4 parent."""
    return (
        [
            StudyIndividualSpec.model_validate(item)
            for item in manifest.inherited_individuals
        ],
        [
            StudyProtocolSpec.model_validate(item)
            for item in manifest.inherited_protocols
        ],
        [
            StudySimulationSpec.model_validate(item)
            for item in manifest.inherited_simulations
        ],
    )


def build_caffeine_v3_runtime_audit(
    *,
    windows_run_root: Path,
    v3_scenario_manifest_path: Path,
    source_scenario_manifest_path: Path,
    roundtrip_audit_path: Path,
    output_path: Path,
) -> CaffeineV3RuntimeAudit:
    if output_path.exists():
        raise CaffeineV3RuntimeError(f"Immutable output exists: {output_path}")
    windows_run_root = windows_run_root.resolve()
    v3_manifest = CaffeineV3ScenarioManifest.model_validate_json(
        v3_scenario_manifest_path.read_text(encoding="utf-8")
    )
    source_manifest = CaffeineV2ScenarioManifest.model_validate_json(
        source_scenario_manifest_path.read_text(encoding="utf-8")
    )
    individuals_list, protocols_list, simulations_list = (
        _inherited_study_specs(source_manifest)
    )
    roundtrip = CaffeineV3RoundTripAudit.model_validate_json(
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
        "v3_manifest_checksum_valid": _checksum_valid(v3_manifest),
        "source_manifest_checksum_valid": _checksum_valid(source_manifest),
        "roundtrip_audit_checksum_valid": _checksum_valid(roundtrip),
        "roundtrip_audit_accepted": roundtrip.accepted,
        "roundtrip_has_no_semantic_differences": not roundtrip.semantic_differences,
        "source_manifest_sha_matches_v3": sha256_file(
            source_scenario_manifest_path
        )
        == v3_manifest.source_scenario_manifest_sha256,
        "source_manifest_checksum_matches_v3": source_manifest.checksum
        == v3_manifest.source_scenario_manifest_checksum,
        "source_individuals_match_v3": source_manifest.inherited_individuals
        == v3_manifest.inherited_individuals,
        "source_protocols_match_v3": source_manifest.inherited_protocols
        == v3_manifest.inherited_protocols,
        "source_simulations_match_v3": source_manifest.inherited_simulations
        == v3_manifest.inherited_simulations,
        "scenario_split_map_matches_v3": source_manifest.split_by_simulation
        == v3_manifest.split_by_simulation,
        "windows_run_completed": windows_manifest.get("Status") == "completed",
        "windows_run_has_seven_scenarios": windows_manifest.get("ScenarioCount")
        == 7,
        "windows_run_has_two_runs": windows_manifest.get("RunCount") == 2,
        "windows_run_locked_v5_snapshot": windows_manifest.get(
            "SnapshotFileName"
        )
        == "caffeine_study_scenarios_v5.json",
    }
    scenario_names = [item.name for item in simulations_list]
    if scenario_names != list(EXPECTED_SIMULATIONS):
        raise CaffeineV3RuntimeError("Scenario order or identity drifted.")
    protocols = {item.name: item for item in protocols_list}
    individuals = {item.name: item for item in individuals_list}
    scenarios: list[CaffeineV2ScenarioRuntimeAudit] = []
    for run_id in RUN_IDS:
        for scenario in simulations_list:
            scenarios.append(
                _scenario_audit(
                    windows_run_root=windows_run_root,
                    run_id=run_id,
                    scenario=scenario,
                    individual=individuals[scenario.individual_name],
                    expected_route=protocols[
                        scenario.protocol_name
                    ].application_type,
                    runtime_job_id="caffeine-study-runtime-v5",
                    project_key_prefix="caffeine-study-scenarios-v5",
                )
            )

    by_key = {(item.run_id, item.scenario_name): item for item in scenarios}
    repeatability_checks: dict[str, bool] = {}
    repeatability_details: list[dict[str, Any]] = []
    for scenario_name in EXPECTED_SIMULATIONS:
        first = by_key[("run_1", scenario_name)]
        second = by_key[("run_2", scenario_name)]
        csv_a, xml_a, _ = _scenario_paths(
            windows_run_root,
            "run_1",
            scenario_name,
        )
        csv_b, xml_b, _ = _scenario_paths(
            windows_run_root,
            "run_2",
            scenario_name,
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
        "v3_scenario_manifest_sha256": sha256_file(
            v3_scenario_manifest_path
        ),
        "v3_scenario_manifest_checksum": v3_manifest.checksum,
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
    draft = CaffeineV3RuntimeAudit(**payload, checksum="0" * 64)
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
        description="Audit the caffeine v3 seven-scenario PK-Sim runtime."
    )
    parser.add_argument("--windows-run-root", type=Path, required=True)
    parser.add_argument("--v3-scenario-manifest", type=Path, required=True)
    parser.add_argument("--source-scenario-manifest", type=Path, required=True)
    parser.add_argument("--roundtrip-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    audit = build_caffeine_v3_runtime_audit(
        windows_run_root=args.windows_run_root,
        v3_scenario_manifest_path=args.v3_scenario_manifest,
        source_scenario_manifest_path=args.source_scenario_manifest,
        roundtrip_audit_path=args.roundtrip_audit,
        output_path=args.output,
    )
    print(json.dumps(audit.model_dump(mode="json"), ensure_ascii=False))
    if not audit.accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
