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
from pbpk_kb.blueprints.caffeine_v2_parameters import (
    CaffeineV2ParameterManifest,
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


SNAPSHOT_BUILDER_VERSION = "caffeine-v2-snapshot-builder-1.0.0"
SCENARIO_VERSION = "caffeine_study_scenarios_v4"
SOURCE_SCENARIO_VERSION = "caffeine_study_scenarios_v3"
PERMEABILITY_ALTERNATIVE = "PBPK-KB-v2-estimated"
PERMEABILITY_GROUP = "COMPOUND_INTESTINAL_PERMEABILITY"
RENAL_PROCESS_DATA_SOURCE = "Birkett 1991 Table 1 rederived PBPK-KB-v1"
RENAL_PROCESS_NAME = f"Renal Clearances-{RENAL_PROCESS_DATA_SOURCE}"
EXPECTED_SIMULATIONS = (
    "CAL_BLANCHARD_IV",
    "CAL_BLANCHARD_PO",
    "HOLD_WAHLLANDER_IV",
    "EXT_EUDRACT_PO",
    "EXT_PMC535_PO",
    "EXT2_PMC572_PO_MALE",
    "EXT2_PMC572_PO_FEMALE",
)
PATHWAYS = (
    ("N3-Paraxanthine", "n3_paraxanthine", 0.839),
    ("N1-Theobromine", "n1_theobromine", 0.121),
    ("N7-Theophylline", "n7_theophylline", 0.040),
)


class CaffeineV2SnapshotError(RuntimeError):
    pass


class V2ProcessSpec(StrictBlueprintModel):
    data_source: str = Field(min_length=1)
    pathway_id: str = Field(min_length=1)
    process_name: str = Field(min_length=1)
    fraction: float = Field(gt=0, lt=1)
    initial_specific_clearance: float = Field(gt=0)
    unit: Literal["1/min"] = "1/min"
    source_artifact_id: Literal["lelo_1986_partial_clearance_pubmed_xml"] = (
        "lelo_1986_partial_clearance_pubmed_xml"
    )


class CaffeineV2ScenarioManifest(StrictBlueprintModel):
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
    simulation_names: list[str] = Field(min_length=7, max_length=7)
    split_by_simulation: dict[str, str]
    inherited_individuals: list[dict[str, Any]] = Field(min_length=7, max_length=7)
    inherited_protocols: list[dict[str, Any]] = Field(min_length=6, max_length=6)
    inherited_simulations: list[dict[str, Any]] = Field(min_length=7, max_length=7)
    allowed_changed_paths: list[str] = Field(min_length=4)
    changed_paths: list[str] = Field(min_length=4)
    checks: dict[str, bool]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


class CaffeineV2RoundTripAudit(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    builder_version: Literal[SNAPSHOT_BUILDER_VERSION] = SNAPSHOT_BUILDER_VERSION
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


def _manifest_payload(path: Path, *, version_key: str, version: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checksum = payload.get("checksum")
    if payload.get(version_key) != version or payload.get("accepted") is not True:
        raise CaffeineV2SnapshotError(f"Unaccepted or drifted manifest: {path}")
    if canonical_sha256(
        {key: value for key, value in payload.items() if key != "checksum"}
    ) != checksum:
        raise CaffeineV2SnapshotError(f"Manifest checksum is invalid: {path}")
    return payload


def _normalize_inherited_individual_elisions(
    transformed: dict[str, Any],
    roundtrip: dict[str, Any],
    inherited_individuals: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    left = copy.deepcopy(transformed)
    right = copy.deepcopy(roundtrip)
    notes: list[str] = []
    left_individuals = {
        item.get("Name"): item for item in left.get("Individuals", [])
    }
    right_individuals = {
        item.get("Name"): item for item in right.get("Individuals", [])
    }
    for spec in inherited_individuals:
        name = spec["name"]
        left_origin = left_individuals.get(name, {}).get("OriginData", {})
        right_origin = right_individuals.get(name, {}).get("OriginData", {})
        for key, expected_value, expected_unit in (
            ("Weight", spec.get("weight_kg"), "kg"),
            ("Height", spec.get("height_m"), "m"),
        ):
            if expected_value is None or key in right_origin:
                continue
            candidate = left_origin.get(key)
            if candidate != {"Value": expected_value, "Unit": expected_unit}:
                continue
            left_origin.pop(key)
            notes.append(
                "population_mean_elision: "
                f"{name}.OriginData.{key}={expected_value} {expected_unit}"
            )
    return left, right, notes


def _one(values: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    matches = [item for item in values if item.get(key) == value]
    if len(matches) != 1:
        raise CaffeineV2SnapshotError(
            f"Expected one object with {key}={value!r}; found {len(matches)}."
        )
    return matches[0]


def _parameter_map(manifest: CaffeineV2ParameterManifest) -> dict[str, Any]:
    return {item.parameter_id: item for item in manifest.parameters}


def _pathway_processes(
    manifest: CaffeineV2ParameterManifest,
) -> tuple[list[dict[str, Any]], list[V2ProcessSpec]]:
    parameters = _parameter_map(manifest)
    total = parameters["total_cyp1a2_specific_clearance"]
    source_fractions = {
        "n3_paraxanthine": parameters["caffeine_fraction_n3_paraxanthine"].value,
        "n1_theobromine": parameters["caffeine_fraction_n1_theobromine"].value,
        "n7_theophylline": parameters["caffeine_fraction_n7_theophylline"].value,
    }
    processes: list[dict[str, Any]] = []
    specs: list[V2ProcessSpec] = []
    for data_source, pathway_id, expected_fraction in PATHWAYS:
        fraction = source_fractions[pathway_id]
        if not math.isclose(fraction, expected_fraction, rel_tol=0, abs_tol=1e-12):
            raise CaffeineV2SnapshotError(f"Pathway fraction drifted: {pathway_id}")
        process_name = f"CYP1A2-{data_source}"
        initial = total.value * fraction
        processes.append(
            {
                "DataSource": data_source,
                "InternalName": "MetabolizationIntrinsic_FirstOrder",
                "Molecule": "CYP1A2",
                "Parameters": [
                    {
                        "Name": "Intrinsic clearance",
                        "Unit": "l/min",
                        "Value": 0.0,
                    },
                    {
                        "Name": "Specific clearance",
                        "Unit": "1/min",
                        "Value": initial,
                    },
                ],
                "Species": "Human",
            }
        )
        specs.append(
            V2ProcessSpec(
                data_source=data_source,
                pathway_id=pathway_id,
                process_name=process_name,
                fraction=fraction,
                initial_specific_clearance=initial,
            )
        )
    return processes, specs


def _transformed_snapshot(
    source: dict[str, Any],
    manifest: CaffeineV2ParameterManifest,
) -> tuple[dict[str, Any], list[V2ProcessSpec]]:
    output = copy.deepcopy(source)
    compound = _one(output.get("Compounds", []), "Name", "Caffeine")
    source_processes = compound.get("Processes", [])
    renal = [
        item
        for item in source_processes
        if item.get("InternalName") == "KidneyClearance"
        and item.get("DataSource") == RENAL_PROCESS_DATA_SOURCE
    ]
    source_mm = [
        item
        for item in source_processes
        if item.get("InternalName") == "MetabolizationLiverMicrosomes_MM"
        and item.get("Molecule") == "CYP1A2"
    ]
    if len(renal) != 1 or len(source_mm) != 1 or len(source_processes) != 2:
        raise CaffeineV2SnapshotError("Source v3 process topology drifted.")
    processes, specs = _pathway_processes(manifest)
    compound["Processes"] = processes + [renal[0]]

    parameters = _parameter_map(manifest)
    permeability = parameters[
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
        "PBPK-KB caffeine parent pathway model v2; N3/N1/N7 fractions fixed "
        "from PMID 3756066; total clearance and intestinal permeability estimated."
    )

    expected_process_selections = [
        {"MoleculeName": "CYP1A2", "Name": item.process_name}
        for item in specs
    ] + [{"Name": RENAL_PROCESS_NAME, "SystemicProcessType": "Renal"}]
    simulations = output.get("Simulations", [])
    if tuple(item.get("Name") for item in simulations) != EXPECTED_SIMULATIONS:
        raise CaffeineV2SnapshotError("Source simulation identity/order drifted.")
    for simulation in simulations:
        simulation_compound = _one(
            simulation.get("Compounds", []), "Name", "Caffeine"
        )
        simulation_compound["Processes"] = copy.deepcopy(
            expected_process_selections
        )
        simulation_compound["Alternatives"] = [
            {
                "AlternativeName": PERMEABILITY_ALTERNATIVE,
                "GroupName": PERMEABILITY_GROUP,
            }
        ]
    return output, specs


def _path_allowed(path: str) -> bool:
    allowed_fragments = (
        "/Compounds/0/Description",
        "/Compounds/0/Processes",
        "/Compounds/0/IntestinalPermeability",
        "/Simulations/",
    )
    if path.startswith(allowed_fragments[:3]):
        return True
    return path.startswith("/Simulations/") and (
        "/Compounds/0/Processes" in path
        or "/Compounds/0/Alternatives" in path
    )


def _structural_checks(
    snapshot: dict[str, Any],
    manifest: CaffeineV2ParameterManifest,
) -> dict[str, bool]:
    compound = _one(snapshot.get("Compounds", []), "Name", "Caffeine")
    processes = compound.get("Processes", [])
    expected_data_sources = {item[0] for item in PATHWAYS}
    pathway_processes = [
        item
        for item in processes
        if item.get("InternalName") == "MetabolizationIntrinsic_FirstOrder"
        and item.get("Molecule") == "CYP1A2"
    ]
    permeability = compound.get("IntestinalPermeability", [])
    lipophilicity = _one(compound.get("Lipophilicity", []), "Name", "PubChem-XLogP-Initial-v1")
    lipophilicity_value = _one(
        lipophilicity.get("Parameters", []), "Name", "Lipophilicity"
    ).get("Value")
    expected_process_names = {f"CYP1A2-{item}" for item in expected_data_sources}
    simulation_checks: list[bool] = []
    for simulation in snapshot.get("Simulations", []):
        selected = _one(simulation.get("Compounds", []), "Name", "Caffeine")
        selected_processes = selected.get("Processes", [])
        selected_names = {
            item.get("Name")
            for item in selected_processes
            if item.get("MoleculeName") == "CYP1A2"
        }
        renal_count = sum(
            item.get("Name") == RENAL_PROCESS_NAME
            and item.get("SystemicProcessType") == "Renal"
            for item in selected_processes
        )
        simulation_checks.append(
            selected_names == expected_process_names
            and renal_count == 1
            and selected.get("Alternatives")
            == [
                {
                    "AlternativeName": PERMEABILITY_ALTERNATIVE,
                    "GroupName": PERMEABILITY_GROUP,
                }
            ]
        )
    total = _parameter_map(manifest)["total_cyp1a2_specific_clearance"].value
    actual_sum = sum(
        _one(item.get("Parameters", []), "Name", "Specific clearance")["Value"]
        for item in pathway_processes
    )
    return {
        "one_caffeine_compound": len(snapshot.get("Compounds", [])) == 1,
        "three_first_order_cyp1a2_pathways": len(pathway_processes) == 3
        and {item.get("DataSource") for item in pathway_processes}
        == expected_data_sources,
        "pathway_initial_clearances_sum_to_total": math.isclose(
            actual_sum, total, rel_tol=0, abs_tol=1e-15
        ),
        "one_renal_process_preserved": sum(
            item.get("InternalName") == "KidneyClearance"
            and item.get("DataSource") == RENAL_PROCESS_DATA_SOURCE
            for item in processes
        )
        == 1,
        "one_direct_permeability_alternative": len(permeability) == 1
        and permeability[0].get("Name") == PERMEABILITY_ALTERNATIVE,
        "lipophilicity_fixed_minus_point_one": math.isclose(
            float(lipophilicity_value), -0.1, rel_tol=0, abs_tol=1e-15
        ),
        "seven_simulations_exact": tuple(
            item.get("Name") for item in snapshot.get("Simulations", [])
        )
        == EXPECTED_SIMULATIONS,
        "all_simulations_select_three_pathways_renal_permeability": (
            len(simulation_checks) == 7 and all(simulation_checks)
        ),
    }


def build_caffeine_v2_snapshot_package(
    *,
    source_snapshot_path: Path,
    source_scenario_manifest_path: Path,
    parameter_manifest_path: Path,
    output_dir: Path,
) -> CaffeineV2ScenarioManifest:
    if output_dir.exists():
        raise CaffeineV2SnapshotError(f"Immutable output exists: {output_dir}")
    source_manifest = _manifest_payload(
        source_scenario_manifest_path,
        version_key="scenario_version",
        version=SOURCE_SCENARIO_VERSION,
    )
    if sha256_file(source_snapshot_path) != source_manifest["transformed_snapshot_file_sha256"]:
        raise CaffeineV2SnapshotError("Source v3 snapshot checksum drifted.")
    parameter_payload = _manifest_payload(
        parameter_manifest_path,
        version_key="manifest_version",
        version="caffeine_parameter_manifest_v2",
    )
    parameter_manifest = CaffeineV2ParameterManifest.model_validate(
        parameter_payload
    )
    if (
        parameter_manifest.dataset_manifest_checksum
        != source_manifest["dataset_manifest_checksum"]
    ):
        raise CaffeineV2SnapshotError("Dataset lineage drifted between v3 and v2.")
    source = json.loads(source_snapshot_path.read_text(encoding="utf-8-sig"))
    transformed, specs = _transformed_snapshot(source, parameter_manifest)
    changed_paths = sorted(_diff_paths(source, transformed))
    checks = _structural_checks(transformed, parameter_manifest)
    checks.update(
        {
            "source_scenario_manifest_accepted": source_manifest["accepted"] is True,
            "parameter_manifest_accepted": parameter_manifest.accepted,
            "changes_restricted_to_registered_structure": bool(changed_paths)
            and all(_path_allowed(path) for path in changed_paths),
            "individuals_byte_equivalent": source.get("Individuals")
            == transformed.get("Individuals"),
            "protocols_byte_equivalent": source.get("Protocols")
            == transformed.get("Protocols"),
            "formulations_byte_equivalent": source.get("Formulations")
            == transformed.get("Formulations"),
        }
    )

    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        raise CaffeineV2SnapshotError(f"Stale output exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        snapshot_path = temporary / "study_snapshot.json"
        write_json_atomic(snapshot_path, transformed, canonical=False)
        projected, normalization_notes = _semantic_projection(transformed)
        split_by_simulation = {
            item["name"]: item["split"]
            for item in source_manifest["simulations"]
        }
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
            "parameter_manifest_checksum": parameter_manifest.checksum,
            "dataset_version": "caffeine_dataset_v3",
            "dataset_manifest_checksum": parameter_manifest.dataset_manifest_checksum,
            "transformed_snapshot_sha256": sha256_file(snapshot_path),
            "transformed_snapshot_semantic_sha256": canonical_sha256(projected),
            "pathway_processes": specs,
            "permeability_alternative": PERMEABILITY_ALTERNATIVE,
            "simulation_names": list(EXPECTED_SIMULATIONS),
            "split_by_simulation": split_by_simulation,
            "inherited_individuals": source_manifest["individuals"],
            "inherited_protocols": source_manifest["protocols"],
            "inherited_simulations": source_manifest["simulations"],
            "allowed_changed_paths": [
                "/Compounds/0/Description",
                "/Compounds/0/Processes",
                "/Compounds/0/IntestinalPermeability",
                "/Simulations/*/Compounds/0/Processes",
                "/Simulations/*/Compounds/0/Alternatives",
            ],
            "changed_paths": changed_paths,
            "checks": {
                **checks,
                "semantic_projection_has_no_ignored_scientific_fields": not any(
                    "ignored" in note.lower() for note in normalization_notes
                ),
            },
            "accepted": all(checks.values())
            and not any("ignored" in note.lower() for note in normalization_notes),
        }
        draft = CaffeineV2ScenarioManifest(**payload, checksum="0" * 64)
        manifest = draft.model_copy(
            update={
                "checksum": canonical_sha256(
                    draft.model_dump(mode="json", exclude={"checksum"})
                )
            }
        )
        if not manifest.accepted:
            failed = [name for name, passed in manifest.checks.items() if not passed]
            raise CaffeineV2SnapshotError(
                "Caffeine v2 snapshot checks failed: " + ", ".join(failed)
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
            shutil.rmtree(temporary)
        raise


def audit_caffeine_v2_roundtrip(
    *,
    transformed_snapshot_path: Path,
    roundtrip_snapshot_path: Path,
    scenario_manifest_path: Path,
) -> CaffeineV2RoundTripAudit:
    manifest = CaffeineV2ScenarioManifest.model_validate_json(
        scenario_manifest_path.read_text(encoding="utf-8")
    )
    if canonical_sha256(
        manifest.model_dump(mode="json", exclude={"checksum"})
    ) != manifest.checksum:
        raise CaffeineV2SnapshotError("Scenario manifest checksum is invalid.")
    transformed = json.loads(
        transformed_snapshot_path.read_text(encoding="utf-8-sig")
    )
    roundtrip = json.loads(roundtrip_snapshot_path.read_text(encoding="utf-8-sig"))
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
        left, right, notes=numeric_notes
    )
    differences = sorted(_diff_paths(left, right))
    compound = _one(roundtrip.get("Compounds", []), "Name", "Caffeine")
    process_sources = {
        item.get("DataSource")
        for item in compound.get("Processes", [])
        if item.get("InternalName") == "MetabolizationIntrinsic_FirstOrder"
    }
    simulation_ok = []
    expected_process_names = {item.process_name for item in manifest.pathway_processes}
    for simulation in roundtrip.get("Simulations", []):
        selected = _one(simulation.get("Compounds", []), "Name", "Caffeine")
        simulation_ok.append(
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
        "direct_permeability_preserved": compound.get("IntestinalPermeability")
        is not None
        and len(compound["IntestinalPermeability"]) == 1
        and compound["IntestinalPermeability"][0].get("Name")
        == PERMEABILITY_ALTERNATIVE,
        "all_simulation_process_selections_preserved": len(simulation_ok) == 7
        and all(simulation_ok),
    }
    payload = {
        "schema_version": "1.0.0",
        "builder_version": SNAPSHOT_BUILDER_VERSION,
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
    draft = CaffeineV2RoundTripAudit(**payload, checksum="0" * 64)
    return draft.model_copy(
        update={
            "checksum": canonical_sha256(
                draft.model_dump(mode="json", exclude={"checksum"})
            )
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or audit the caffeine v2 pathway snapshot."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source-snapshot", type=Path, required=True)
    build.add_argument("--source-scenario-manifest", type=Path, required=True)
    build.add_argument("--parameter-manifest", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--transformed-snapshot", type=Path, required=True)
    audit.add_argument("--roundtrip-snapshot", type=Path, required=True)
    audit.add_argument("--scenario-manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        result = build_caffeine_v2_snapshot_package(
            source_snapshot_path=args.source_snapshot,
            source_scenario_manifest_path=args.source_scenario_manifest,
            parameter_manifest_path=args.parameter_manifest,
            output_dir=args.output_dir,
        )
    else:
        result = audit_caffeine_v2_roundtrip(
            transformed_snapshot_path=args.transformed_snapshot,
            roundtrip_snapshot_path=args.roundtrip_snapshot,
            scenario_manifest_path=args.scenario_manifest,
        )
        write_json_atomic(args.output, result, canonical=False)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
