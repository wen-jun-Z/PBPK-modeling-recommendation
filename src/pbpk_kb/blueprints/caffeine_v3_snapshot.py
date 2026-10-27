from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from pbpk_kb.blueprints.caffeine_study_snapshot import (
    _normalize_snapshot_numeric_serialization,
)
from pbpk_kb.blueprints.caffeine_v2_snapshot import (
    EXPECTED_SIMULATIONS,
    PATHWAYS,
    PERMEABILITY_GROUP,
    RENAL_PROCESS_DATA_SOURCE,
    RENAL_PROCESS_NAME,
    V2ProcessSpec,
    _normalize_inherited_individual_elisions,
    _one,
)
from pbpk_kb.blueprints.caffeine_v3_parameters import (
    CaffeineV3ParameterManifest,
    MODEL_VERSION,
)
from pbpk_kb.blueprints.pksim_snapshot_builder import (
    _diff_paths,
    _semantic_projection,
)
from pbpk_kb.blueprints.schemas import StrictBlueprintModel
from pbpk_kb.model_assets.execution_plan import (
    canonical_sha256,
    write_json_atomic,
)
from pbpk_kb.model_assets.license_policy import sha256_file


SNAPSHOT_BUILDER_VERSION = "caffeine-v3-snapshot-builder-1.0.0"
SCENARIO_VERSION = "caffeine_study_scenarios_v5"
SOURCE_SCENARIO_VERSION = "caffeine_study_scenarios_v4"
SOURCE_PERMEABILITY_ALTERNATIVE = "PBPK-KB-v2-estimated"
PERMEABILITY_ALTERNATIVE = "PBPK-KB-v3-set-identified"


class CaffeineV3SnapshotError(RuntimeError):
    pass


class CaffeineV3ScenarioManifest(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    builder_version: Literal[SNAPSHOT_BUILDER_VERSION] = SNAPSHOT_BUILDER_VERSION
    scenario_version: Literal[SCENARIO_VERSION] = SCENARIO_VERSION
    model_version: Literal[MODEL_VERSION] = MODEL_VERSION
    generated_at: datetime
    source_scenario_version: Literal[SOURCE_SCENARIO_VERSION] = (
        SOURCE_SCENARIO_VERSION
    )
    source_scenario_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_scenario_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameter_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameter_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_version: Literal["caffeine_dataset_v3"] = "caffeine_dataset_v3"
    dataset_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    transformed_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transformed_snapshot_semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pathway_processes: list[V2ProcessSpec] = Field(min_length=3, max_length=3)
    permeability_alternative: Literal[PERMEABILITY_ALTERNATIVE] = (
        PERMEABILITY_ALTERNATIVE
    )
    permeability_interpretation: Literal[
        "search_initial_value_not_point_estimate"
    ] = "search_initial_value_not_point_estimate"
    simulation_names: list[str] = Field(min_length=7, max_length=7)
    split_by_simulation: dict[str, str]
    inherited_individuals: list[dict[str, Any]] = Field(min_length=7, max_length=7)
    inherited_protocols: list[dict[str, Any]] = Field(min_length=6, max_length=6)
    inherited_simulations: list[dict[str, Any]] = Field(min_length=7, max_length=7)
    allowed_changed_paths: list[str] = Field(min_length=3)
    changed_paths: list[str] = Field(min_length=3)
    checks: dict[str, bool]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


class CaffeineV3RoundTripAudit(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    auditor_version: Literal["caffeine-v3-roundtrip-auditor-1.0.0"] = (
        "caffeine-v3-roundtrip-auditor-1.0.0"
    )
    generated_at: datetime
    scenario_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    transformed_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    roundtrip_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transformed_semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    roundtrip_semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_differences: list[str]
    accepted_normalizations: list[str]
    checks: dict[str, bool]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


def _verified_manifest(
    path: Path,
    *,
    version_key: str,
    version: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checksum = payload.get("checksum")
    if payload.get(version_key) != version or payload.get("accepted") is not True:
        raise CaffeineV3SnapshotError(f"Unaccepted or drifted manifest: {path}")
    if canonical_sha256(
        {key: value for key, value in payload.items() if key != "checksum"}
    ) != checksum:
        raise CaffeineV3SnapshotError(f"Manifest checksum is invalid: {path}")
    return payload


def _parameter_map(manifest: CaffeineV3ParameterManifest) -> dict[str, Any]:
    return {item.parameter_id: item for item in manifest.parameters}


def _transform_v4_snapshot(
    source: dict[str, Any],
    manifest: CaffeineV3ParameterManifest,
) -> dict[str, Any]:
    output = copy.deepcopy(source)
    compound = _one(output.get("Compounds", []), "Name", "Caffeine")
    alternatives = compound.get("IntestinalPermeability", [])
    if len(alternatives) != 1 or alternatives[0].get("Name") != (
        SOURCE_PERMEABILITY_ALTERNATIVE
    ):
        raise CaffeineV3SnapshotError("Source v4 permeability alternative drifted.")
    permeability = _parameter_map(manifest)[
        "specific_intestinal_permeability_transcellular"
    ]
    compound["IntestinalPermeability"] = [
        {
            "Name": PERMEABILITY_ALTERNATIVE,
            "IsDefault": False,
            "Parameters": [
                {
                    "Name": "Specific intestinal permeability (transcellular)",
                    "Value": permeability.value,
                    "Unit": permeability.unit,
                }
            ],
        }
    ]
    compound["Description"] = (
        "PBPK-KB caffeine parent pathway model v3; total CYP1A2 clearance "
        "point-estimated from IV data; intestinal permeability set-identified "
        "from absorption constraints. The stored value is a search initial "
        "value, not an observed or point-estimated permeability."
    )
    simulations = output.get("Simulations", [])
    if tuple(item.get("Name") for item in simulations) != EXPECTED_SIMULATIONS:
        raise CaffeineV3SnapshotError("Source simulation identity/order drifted.")
    for simulation in simulations:
        selected = _one(simulation.get("Compounds", []), "Name", "Caffeine")
        if selected.get("Alternatives") != [
            {
                "AlternativeName": SOURCE_PERMEABILITY_ALTERNATIVE,
                "GroupName": PERMEABILITY_GROUP,
            }
        ]:
            raise CaffeineV3SnapshotError(
                f"Source alternative drifted for {simulation.get('Name')}."
            )
        selected["Alternatives"] = [
            {
                "AlternativeName": PERMEABILITY_ALTERNATIVE,
                "GroupName": PERMEABILITY_GROUP,
            }
        ]
    return output


def _path_allowed(path: str) -> bool:
    if path.startswith("/Compounds/0/Description"):
        return True
    if path.startswith("/Compounds/0/IntestinalPermeability"):
        return True
    return path.startswith("/Simulations/") and (
        "/Compounds/0/Alternatives" in path
    )


def _structural_checks(
    *,
    source: dict[str, Any],
    transformed: dict[str, Any],
    manifest: CaffeineV3ParameterManifest,
) -> dict[str, bool]:
    source_compound = _one(source.get("Compounds", []), "Name", "Caffeine")
    compound = _one(transformed.get("Compounds", []), "Name", "Caffeine")
    permeability = compound.get("IntestinalPermeability", [])
    permeability_value = _one(
        permeability[0].get("Parameters", []),
        "Name",
        "Specific intestinal permeability (transcellular)",
    ).get("Value") if len(permeability) == 1 else math.nan
    expected_value = _parameter_map(manifest)[
        "specific_intestinal_permeability_transcellular"
    ].value
    simulation_alternatives = []
    for simulation in transformed.get("Simulations", []):
        selected = _one(simulation.get("Compounds", []), "Name", "Caffeine")
        simulation_alternatives.append(selected.get("Alternatives"))
    return {
        "one_caffeine_compound": len(transformed.get("Compounds", [])) == 1,
        "processes_byte_equivalent_to_audited_v4": (
            source_compound.get("Processes") == compound.get("Processes")
        ),
        "calculation_methods_byte_equivalent_to_audited_v4": (
            source_compound.get("CalculationMethods")
            == compound.get("CalculationMethods")
        ),
        "one_v3_set_identified_alternative": len(permeability) == 1
        and permeability[0].get("Name") == PERMEABILITY_ALTERNATIVE
        and math.isclose(
            float(permeability_value),
            expected_value,
            rel_tol=0,
            abs_tol=1e-15,
        ),
        "seven_simulations_exact": tuple(
            item.get("Name") for item in transformed.get("Simulations", [])
        )
        == EXPECTED_SIMULATIONS,
        "all_simulations_select_v3_alternative": len(simulation_alternatives) == 7
        and all(
            item
            == [
                {
                    "AlternativeName": PERMEABILITY_ALTERNATIVE,
                    "GroupName": PERMEABILITY_GROUP,
                }
            ]
            for item in simulation_alternatives
        ),
        "individuals_byte_equivalent": source.get("Individuals")
        == transformed.get("Individuals"),
        "protocols_byte_equivalent": source.get("Protocols")
        == transformed.get("Protocols"),
        "formulations_byte_equivalent": source.get("Formulations")
        == transformed.get("Formulations"),
        "expression_profiles_byte_equivalent": source.get("ExpressionProfiles")
        == transformed.get("ExpressionProfiles"),
    }


def build_caffeine_v3_snapshot_package(
    *,
    source_snapshot_path: Path,
    source_scenario_manifest_path: Path,
    parameter_manifest_path: Path,
    output_dir: Path,
) -> CaffeineV3ScenarioManifest:
    if output_dir.exists():
        raise CaffeineV3SnapshotError(f"Immutable output exists: {output_dir}")
    source_manifest = _verified_manifest(
        source_scenario_manifest_path,
        version_key="scenario_version",
        version=SOURCE_SCENARIO_VERSION,
    )
    if sha256_file(source_snapshot_path) != source_manifest.get(
        "transformed_snapshot_sha256"
    ):
        raise CaffeineV3SnapshotError("Source v4 snapshot checksum drifted.")
    parameter_payload = _verified_manifest(
        parameter_manifest_path,
        version_key="manifest_version",
        version="caffeine_parameter_manifest_v3",
    )
    parameter = CaffeineV3ParameterManifest.model_validate(parameter_payload)
    if parameter.dataset_manifest_checksum != source_manifest.get(
        "dataset_manifest_checksum"
    ):
        raise CaffeineV3SnapshotError("Dataset lineage drifted between v4 and v5.")

    source = json.loads(source_snapshot_path.read_text(encoding="utf-8-sig"))
    transformed = _transform_v4_snapshot(source, parameter)
    changed_paths = sorted(_diff_paths(source, transformed))
    checks = _structural_checks(
        source=source,
        transformed=transformed,
        manifest=parameter,
    )
    checks.update(
        {
            "source_scenario_manifest_accepted": source_manifest["accepted"]
            is True,
            "parameter_manifest_accepted": parameter.accepted,
            "changes_restricted_to_v3_semantics": bool(changed_paths)
            and all(_path_allowed(path) for path in changed_paths),
        }
    )

    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        raise CaffeineV3SnapshotError(f"Stale output exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        snapshot_path = temporary / "study_snapshot.json"
        write_json_atomic(snapshot_path, transformed, canonical=False)
        projected, normalization_notes = _semantic_projection(transformed)
        checks["semantic_projection_has_no_ignored_scientific_fields"] = not any(
            "ignored" in note.lower() for note in normalization_notes
        )
        payload = {
            "schema_version": "1.0.0",
            "builder_version": SNAPSHOT_BUILDER_VERSION,
            "scenario_version": SCENARIO_VERSION,
            "model_version": MODEL_VERSION,
            "generated_at": datetime.now(UTC),
            "source_scenario_version": SOURCE_SCENARIO_VERSION,
            "source_scenario_manifest_sha256": sha256_file(
                source_scenario_manifest_path
            ),
            "source_scenario_manifest_checksum": source_manifest["checksum"],
            "source_snapshot_sha256": sha256_file(source_snapshot_path),
            "parameter_manifest_sha256": sha256_file(parameter_manifest_path),
            "parameter_manifest_checksum": parameter.checksum,
            "dataset_version": "caffeine_dataset_v3",
            "dataset_manifest_checksum": parameter.dataset_manifest_checksum,
            "transformed_snapshot_sha256": sha256_file(snapshot_path),
            "transformed_snapshot_semantic_sha256": canonical_sha256(projected),
            "pathway_processes": source_manifest["pathway_processes"],
            "permeability_alternative": PERMEABILITY_ALTERNATIVE,
            "permeability_interpretation": (
                "search_initial_value_not_point_estimate"
            ),
            "simulation_names": list(EXPECTED_SIMULATIONS),
            "split_by_simulation": source_manifest["split_by_simulation"],
            "inherited_individuals": source_manifest["inherited_individuals"],
            "inherited_protocols": source_manifest["inherited_protocols"],
            "inherited_simulations": source_manifest["inherited_simulations"],
            "allowed_changed_paths": [
                "/Compounds/0/Description",
                "/Compounds/0/IntestinalPermeability",
                "/Simulations/*/Compounds/0/Alternatives",
            ],
            "changed_paths": changed_paths,
            "checks": checks,
            "accepted": all(checks.values()),
        }
        draft = CaffeineV3ScenarioManifest(**payload, checksum="0" * 64)
        manifest = draft.model_copy(
            update={
                "checksum": canonical_sha256(
                    draft.model_dump(mode="json", exclude={"checksum"})
                )
            }
        )
        if not manifest.accepted:
            failed = [name for name, passed in checks.items() if not passed]
            raise CaffeineV3SnapshotError(
                "Caffeine v3 snapshot checks failed: " + ", ".join(failed)
            )
        write_json_atomic(
            temporary / "study_scenario_manifest.json", manifest, canonical=False
        )
        write_json_atomic(
            temporary / "transformation_audit.json",
            {
                "source_snapshot_sha256": sha256_file(source_snapshot_path),
                "transformed_snapshot_sha256": sha256_file(snapshot_path),
                "changed_paths": changed_paths,
                "allowed_changed_paths": manifest.allowed_changed_paths,
                "checks": manifest.checks,
                "accepted": manifest.accepted,
            },
            canonical=False,
        )
        files = [
            {
                "relative_path": path.name,
                "sha256": sha256_file(path),
                "byte_size": path.stat().st_size,
            }
            for path in sorted(temporary.iterdir(), key=lambda item: item.name)
            if path.is_file()
        ]
        write_json_atomic(
            temporary / "files.sha256.json", {"files": files}, canonical=True
        )
        temporary.replace(output_dir)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def audit_caffeine_v3_roundtrip(
    *,
    transformed_snapshot_path: Path,
    roundtrip_snapshot_path: Path,
    scenario_manifest_path: Path,
) -> CaffeineV3RoundTripAudit:
    manifest = CaffeineV3ScenarioManifest.model_validate_json(
        scenario_manifest_path.read_text(encoding="utf-8")
    )
    if canonical_sha256(
        manifest.model_dump(mode="json", exclude={"checksum"})
    ) != manifest.checksum:
        raise CaffeineV3SnapshotError("Scenario manifest checksum is invalid.")
    transformed = json.loads(
        transformed_snapshot_path.read_text(encoding="utf-8-sig")
    )
    roundtrip = json.loads(
        roundtrip_snapshot_path.read_text(encoding="utf-8-sig")
    )
    transformed, roundtrip, individual_notes = (
        _normalize_inherited_individual_elisions(
            transformed,
            roundtrip,
            manifest.inherited_individuals,
        )
    )
    left, left_notes = _semantic_projection(transformed)
    right, right_notes = _semantic_projection(roundtrip)
    numeric_notes: list[str] = []
    left = _normalize_snapshot_numeric_serialization(
        left,
        right,
        notes=numeric_notes,
    )
    differences = sorted(_diff_paths(left, right))
    compound = _one(roundtrip.get("Compounds", []), "Name", "Caffeine")
    process_sources = {
        item.get("DataSource")
        for item in compound.get("Processes", [])
        if item.get("InternalName") == "MetabolizationIntrinsic_FirstOrder"
    }
    simulation_checks: list[bool] = []
    expected_process_names = {
        item.process_name for item in manifest.pathway_processes
    }
    for simulation in roundtrip.get("Simulations", []):
        selected = _one(simulation.get("Compounds", []), "Name", "Caffeine")
        simulation_checks.append(
            {
                item.get("Name")
                for item in selected.get("Processes", [])
                if item.get("MoleculeName") == "CYP1A2"
            }
            == expected_process_names
            and any(
                item.get("Name") == RENAL_PROCESS_NAME
                and item.get("SystemicProcessType") == "Renal"
                for item in selected.get("Processes", [])
            )
            and selected.get("Alternatives")
            == [
                {
                    "AlternativeName": PERMEABILITY_ALTERNATIVE,
                    "GroupName": PERMEABILITY_GROUP,
                }
            ]
        )
    checks = {
        "complete_semantic_snapshot_equal": not differences,
        "transformed_snapshot_matches_manifest": sha256_file(
            transformed_snapshot_path
        )
        == manifest.transformed_snapshot_sha256,
        "seven_simulations_preserved": tuple(
            item.get("Name") for item in roundtrip.get("Simulations", [])
        )
        == EXPECTED_SIMULATIONS,
        "three_pathways_preserved": process_sources
        == {item[0] for item in PATHWAYS},
        "renal_process_preserved": any(
            item.get("InternalName") == "KidneyClearance"
            and item.get("DataSource") == RENAL_PROCESS_DATA_SOURCE
            for item in compound.get("Processes", [])
        ),
        "set_identified_permeability_preserved": len(
            compound.get("IntestinalPermeability", [])
        )
        == 1
        and compound["IntestinalPermeability"][0].get("Name")
        == PERMEABILITY_ALTERNATIVE,
        "all_simulation_selections_preserved": len(simulation_checks) == 7
        and all(simulation_checks),
    }
    payload = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(UTC),
        "scenario_manifest_checksum": manifest.checksum,
        "transformed_snapshot_sha256": sha256_file(transformed_snapshot_path),
        "roundtrip_snapshot_sha256": sha256_file(roundtrip_snapshot_path),
        "transformed_semantic_sha256": canonical_sha256(left),
        "roundtrip_semantic_sha256": canonical_sha256(right),
        "semantic_differences": differences,
        "accepted_normalizations": sorted(
            set(individual_notes)
            | set(left_notes)
            | set(right_notes)
            | set(numeric_notes)
        ),
        "checks": checks,
        "accepted": all(checks.values()),
    }
    draft = CaffeineV3RoundTripAudit(**payload, checksum="0" * 64)
    return draft.model_copy(
        update={
            "checksum": canonical_sha256(
                draft.model_dump(mode="json", exclude={"checksum"})
            )
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or audit the caffeine v3 set-identified snapshot."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source-snapshot", type=Path, required=True)
    build.add_argument("--source-scenario-manifest", type=Path, required=True)
    build.add_argument("--parameter-manifest", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--transformed-snapshot", type=Path, required=True)
    audit.add_argument("--roundtrip-snapshot", type=Path, required=True)
    audit.add_argument("--scenario-manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        result = build_caffeine_v3_snapshot_package(
            source_snapshot_path=args.source_snapshot,
            source_scenario_manifest_path=args.source_scenario_manifest,
            parameter_manifest_path=args.parameter_manifest,
            output_dir=args.output_dir,
        )
    else:
        result = audit_caffeine_v3_roundtrip(
            transformed_snapshot_path=args.transformed_snapshot,
            roundtrip_snapshot_path=args.roundtrip_snapshot,
            scenario_manifest_path=args.scenario_manifest,
        )
        write_json_atomic(args.output, result, canonical=False)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
