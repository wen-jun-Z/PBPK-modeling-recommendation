from __future__ import annotations

import argparse
import copy
import json
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from pbpk_kb.blueprints.schemas import StrictBlueprintModel
from pbpk_kb.model_assets.execution_plan import (
    canonical_sha256,
    write_json_atomic,
)
from pbpk_kb.model_assets.license_policy import sha256_file


SNAPSHOT_BUILDER_VERSION = "pksim-snapshot-builder-1.1.0"
PROFILE_NAME_SEPARATOR = "|"

_UNORDERED_SNAPSHOT_COLLECTIONS = {
    "Alternatives",
    "CalculationMethods",
    "Compounds",
    "ExpressionProfiles",
    "Formulations",
    "Individuals",
    "OutputSelections",
    "Parameters",
    "PkaTypes",
    "Processes",
    "Protocols",
    "Simulations",
}
_DERIVED_KIDNEY_CLEARANCE_PARAMETERS = {
    "Blood flow rate (kidney)",
    "Body weight",
    "Volume (kidney)",
}


class PKSimSnapshotBuildError(RuntimeError):
    pass


class LockedSourceArtifact(StrictBlueprintModel):
    role: Literal[
        "source_project",
        "source_snapshot",
        "pksim_database",
        "pksim_source_code",
        "license",
    ]
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SQLQueryEvidence(StrictBlueprintModel):
    query_id: str = Field(min_length=1)
    sql: str = Field(min_length=1)
    parameters: list[str]
    rows: list[dict[str, Any]] = Field(min_length=1)


class PKSimSystemParameterManifest(StrictBlueprintModel):
    manifest_schema_version: Literal["1.1.0"] = "1.1.0"
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    pksim_version: str = Field(min_length=1)
    generated_at: datetime
    source_artifacts: list[LockedSourceArtifact] = Field(min_length=5)
    queries: list[SQLQueryEvidence] = Field(min_length=4)
    cyp1a2_reference_concentration_umol_per_l: float = Field(gt=0)
    protein_default_intestine_half_life_minutes: float = Field(gt=0)
    protein_default_intestine_half_life_hours: float = Field(gt=0)
    protein_default_intestine_half_life_rule: str = Field(min_length=1)
    cyp1a2_liver_half_life_minutes: float = Field(gt=0)
    cyp1a2_liver_half_life_hours: float = Field(gt=0)
    liver_relative_expression: Literal[1.0] = 1.0
    liver_relative_expression_rule: str = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


class RemovedSimulation(StrictBlueprintModel):
    name: str = Field(min_length=1)
    reason_code: str = Field(min_length=1)


class SnapshotPathMigration(StrictBlueprintModel):
    simulation_name: str = Field(min_length=1)
    original_path: str = Field(min_length=1)
    canonical_path: str = Field(min_length=1)
    reason_code: str = Field(min_length=1)

    @model_validator(mode="after")
    def paths_must_differ(self) -> "SnapshotPathMigration":
        if self.original_path == self.canonical_path:
            raise ValueError("Snapshot path migration must change the path.")
        return self


class CaffeineSnapshotTransformationSpec(StrictBlueprintModel):
    spec_schema_version: Literal["1.0.0"] = "1.0.0"
    builder_version: Literal[SNAPSHOT_BUILDER_VERSION] = (
        SNAPSHOT_BUILDER_VERSION
    )
    transformation_id: str = Field(min_length=1)
    created_at: datetime
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    pksim_version: str = Field(min_length=1)
    source_project_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    system_parameter_manifest_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    compound_name: Literal["Caffeine"] = "Caffeine"
    expression_profile_type: Literal["Enzyme"] = "Enzyme"
    expression_profile_species: Literal["Human"] = "Human"
    expression_profile_molecule: Literal["CYP1A2"] = "CYP1A2"
    expression_profile_category: str = Field(min_length=1)
    expression_profile_localization: Literal["Intracellular"] = (
        "Intracellular"
    )
    expression_profile_ontogeny: Literal["CYP1A2"] = "CYP1A2"
    target_individual_names: list[str] = Field(min_length=1)
    target_simulation_names: list[str] = Field(min_length=1)
    expected_simulation_protocols: dict[str, str] = Field(min_length=1)
    removed_simulations: list[RemovedSimulation] = Field(default_factory=list)
    simulation_parameter_path_migrations: list[SnapshotPathMigration] = Field(
        default_factory=list
    )
    process_name: Literal["CYP1A2-MM"] = "CYP1A2-MM"
    process_internal_name: Literal["MetabolizationLiverMicrosomes_MM"] = (
        "MetabolizationLiverMicrosomes_MM"
    )
    process_data_source: Literal["MM"] = "MM"
    output_selections: list[str] = Field(min_length=1)

    @field_validator(
        "target_individual_names",
        "target_simulation_names",
        "output_selections",
    )
    @classmethod
    def unique_values(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Snapshot transformation lists must be unique.")
        return value

    @model_validator(mode="after")
    def simulation_sets_do_not_overlap(
        self,
    ) -> "CaffeineSnapshotTransformationSpec":
        removed = [item.name for item in self.removed_simulations]
        if len(removed) != len(set(removed)):
            raise ValueError("Removed simulation names must be unique.")
        if set(removed) & set(self.target_simulation_names):
            raise ValueError(
                "A simulation cannot be both targeted and removed."
            )
        if set(self.expected_simulation_protocols) != set(
            self.target_simulation_names
        ):
            raise ValueError(
                "Expected protocol keys must exactly match target simulations."
            )
        migration_keys = [
            (item.simulation_name, item.original_path)
            for item in self.simulation_parameter_path_migrations
        ]
        if len(migration_keys) != len(set(migration_keys)):
            raise ValueError("Snapshot path migrations must be unique.")
        invalid_migrations = {
            item.simulation_name
            for item in self.simulation_parameter_path_migrations
            if item.simulation_name not in self.target_simulation_names
        }
        if invalid_migrations:
            raise ValueError(
                "Snapshot path migrations reference non-target simulations."
            )
        return self


class SnapshotTransformationAudit(StrictBlueprintModel):
    audit_schema_version: Literal["1.0.0"] = "1.0.0"
    builder_version: Literal[SNAPSHOT_BUILDER_VERSION] = (
        SNAPSHOT_BUILDER_VERSION
    )
    transformation_id: str
    generated_at: datetime
    source_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transformed_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    changed_paths: list[str]
    allowed_changed_paths: list[str]
    checks: dict[str, bool]
    counts: dict[str, int]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


class SnapshotRoundTripAudit(StrictBlueprintModel):
    audit_schema_version: Literal["1.1.0"] = "1.1.0"
    builder_version: Literal[SNAPSHOT_BUILDER_VERSION] = (
        SNAPSHOT_BUILDER_VERSION
    )
    transformation_id: str
    generated_at: datetime
    transformed_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    roundtrip_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transformed_semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    roundtrip_semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checks: dict[str, bool]
    accepted_normalizations: list[str]
    semantic_differences: list[str]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


def _quoted_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _query_rows(
    connection: sqlite3.Connection,
    *,
    query_id: str,
    sql: str,
    parameters: tuple[str, ...],
) -> SQLQueryEvidence:
    cursor = connection.execute(sql, parameters)
    columns = [item[0] for item in cursor.description]
    rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    if not rows:
        raise PKSimSnapshotBuildError(
            f"PK-Sim database query returned no rows: {query_id}"
        )
    return SQLQueryEvidence(
        query_id=query_id,
        sql=sql,
        parameters=list(parameters),
        rows=rows,
    )


def build_pksim_system_parameter_manifest(
    *,
    source_project_path: Path,
    source_snapshot_path: Path,
    pksim_database_path: Path,
    protein_factory_source_path: Path,
    license_path: Path,
    source_root: Path,
    source_commit: str,
    pksim_version: str,
) -> PKSimSystemParameterManifest:
    paths = {
        "source_project": source_project_path.resolve(),
        "source_snapshot": source_snapshot_path.resolve(),
        "pksim_database": pksim_database_path.resolve(),
        "pksim_source_code": protein_factory_source_path.resolve(),
        "license": license_path.resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise PKSimSnapshotBuildError(
                f"Locked {role} artifact is missing: {path}"
            )
    source_root = source_root.resolve()
    source_artifacts: list[LockedSourceArtifact] = []
    for role, path in paths.items():
        try:
            relative_path = path.relative_to(source_root).as_posix()
        except ValueError:
            relative_path = path.name
        source_artifacts.append(
            LockedSourceArtifact(
                role=role,
                relative_path=relative_path,
                sha256=sha256_file(path),
            )
        )

    connection = sqlite3.connect(
        f"file:{pksim_database_path.resolve()}?mode=ro",
        uri=True,
    )
    try:
        queries = [
            _query_rows(
                connection,
                query_id="cyp1a2_molecule_parameters",
                sql=(
                    "SELECT molecule, parameter_name, distribution_type, "
                    "mean_value, standard_deviation, min_value, "
                    "min_isallowed, max_value, max_isallowed "
                    "FROM tab_molecule_parameters WHERE molecule = ? "
                    "ORDER BY parameter_name"
                ),
                parameters=("CYP1A2",),
            ),
            _query_rows(
                connection,
                query_id="cyp1a2_parameter_definitions",
                sql=(
                    "SELECT parameter_name, display_name, description, "
                    "dimension, default_unit FROM tab_parameters "
                    "WHERE parameter_name IN (?, ?, ?) ORDER BY parameter_name"
                ),
                parameters=(
                    "Reference concentration",
                    "t1/2 (intestine)",
                    "t1/2 (liver)",
                ),
            ),
            _query_rows(
                connection,
                query_id="protein_intestine_half_life_default",
                sql=(
                    "SELECT calculation_method, formula_rate, formula AS rate, "
                    "dimension "
                    "FROM tab_calculation_method_rates "
                    "WHERE calculation_method = ? AND formula_rate = ?"
                ),
                parameters=("CompoundCommon", "PARAM_HalfLife_Intestine"),
            ),
            _query_rows(
                connection,
                query_id="cyp1a2_ontogeny",
                sql=(
                    "SELECT molecule, species, group_name, "
                    "postmenstrual_age, ontogeny_factor, deviation "
                    "FROM tab_ontogenies WHERE molecule = ? "
                    "AND species = ? ORDER BY group_name, postmenstrual_age"
                ),
                parameters=("CYP1A2", "Human"),
            ),
            _query_rows(
                connection,
                query_id="protein_relative_expression_default",
                sql=(
                    "SELECT container_type, container_name, parameter_name, "
                    "calculation_method, formula_rate, min_value, "
                    "min_isallowed, max_value, max_isallowed "
                    "FROM tab_container_parameter_rates "
                    "WHERE container_type = ? AND container_name = ? "
                    "AND parameter_name = ?"
                ),
                parameters=("GENERAL", "PROTEIN", "Relative expression"),
            ),
        ]
    finally:
        connection.close()

    molecule_rows = {
        item["parameter_name"]: item
        for item in queries[0].rows
    }
    required = {"Reference concentration", "t1/2 (liver)"}
    if not required.issubset(molecule_rows):
        raise PKSimSnapshotBuildError(
            "PK-Sim database lacks required CYP1A2 system parameters."
        )
    reference = float(molecule_rows["Reference concentration"]["mean_value"])
    intestine_default_rows = next(
        item.rows
        for item in queries
        if item.query_id == "protein_intestine_half_life_default"
    )
    if len(intestine_default_rows) != 1:
        raise PKSimSnapshotBuildError(
            "PK-Sim database does not define one protein intestine half-life default."
        )
    intestine_half_life_minutes = float(intestine_default_rows[0]["rate"])
    liver_half_life_minutes = float(
        molecule_rows["t1/2 (liver)"]["mean_value"]
    )
    if not all(
        math.isfinite(value) and value > 0
        for value in (
            reference,
            intestine_half_life_minutes,
            liver_half_life_minutes,
        )
    ):
        raise PKSimSnapshotBuildError(
            "PK-Sim CYP1A2 system parameters are not finite positive values."
        )
    payload = {
        "manifest_schema_version": "1.1.0",
        "source_commit": source_commit,
        "pksim_version": pksim_version,
        "generated_at": datetime.now(UTC),
        "source_artifacts": source_artifacts,
        "queries": queries,
        "cyp1a2_reference_concentration_umol_per_l": reference,
        "protein_default_intestine_half_life_minutes": (
            intestine_half_life_minutes
        ),
        "protein_default_intestine_half_life_hours": (
            intestine_half_life_minutes / 60.0
        ),
        "protein_default_intestine_half_life_rule": (
            "PK-Sim CompoundCommon system default for protein intestine "
            "turnover; not a CYP1A2-specific observed parameter."
        ),
        "cyp1a2_liver_half_life_minutes": liver_half_life_minutes,
        "cyp1a2_liver_half_life_hours": liver_half_life_minutes / 60.0,
        "liver_relative_expression": 1.0,
        "liver_relative_expression_rule": (
            "PK-Sim relative expression is normalized to 1.0 in the "
            "reference liver tissue; this is a model normalization rule, "
            "not an observed drug parameter."
        ),
    }
    draft = PKSimSystemParameterManifest(
        **payload,
        checksum="0" * 64,
    )
    return draft.model_copy(
        update={
            "checksum": canonical_sha256(
                draft.model_dump(mode="json", exclude={"checksum"})
            )
        }
    )


def _profile_name(spec: CaffeineSnapshotTransformationSpec) -> str:
    return PROFILE_NAME_SEPARATOR.join(
        (
            spec.expression_profile_molecule,
            spec.expression_profile_species,
            spec.expression_profile_category,
        )
    )


def _cyp1a2_profile(
    *,
    spec: CaffeineSnapshotTransformationSpec,
    manifest: PKSimSystemParameterManifest,
) -> dict[str, Any]:
    database_location = next(
        item.relative_path
        for item in manifest.source_artifacts
        if item.role == "pksim_database"
    )
    return {
        "Type": spec.expression_profile_type,
        "Species": spec.expression_profile_species,
        "Molecule": spec.expression_profile_molecule,
        "Category": spec.expression_profile_category,
        "Parameters": [
            {
                "Path": "CYP1A2|Reference concentration",
                "Value": manifest.cyp1a2_reference_concentration_umol_per_l,
                "Unit": "\u00b5mol/l",
                "ValueOrigin": {
                    "Source": "Database",
                    "Description": (
                        f"{database_location}:tab_molecule_parameters/"
                        "CYP1A2/Reference concentration"
                    ),
                },
            },
            {
                "Path": "CYP1A2|t1/2 (intestine)",
                "Value": manifest.protein_default_intestine_half_life_hours,
                "Unit": "h",
                "ValueOrigin": {
                    "Source": "Database",
                    "Description": (
                        f"{database_location}:"
                        "tab_calculation_method_rates/CompoundCommon/"
                        "PARAM_HalfLife_Intestine; system default converted "
                        "min to h; not CYP1A2-specific observed evidence. "
                        + manifest.protein_default_intestine_half_life_rule
                    ),
                },
            },
            {
                "Path": "CYP1A2|t1/2 (liver)",
                "Value": manifest.cyp1a2_liver_half_life_hours,
                "Unit": "h",
                "ValueOrigin": {
                    "Source": "Database",
                    "Description": (
                        f"{database_location}:tab_molecule_parameters/"
                        "CYP1A2/t1/2 (liver); converted min to h"
                    ),
                },
            },
            {
                "Path": (
                    "Organism|Liver|Pericentral|Intracellular|CYP1A2|"
                    "Relative expression"
                ),
                "Value": manifest.liver_relative_expression,
                "ValueOrigin": {
                    "Source": "Other",
                    "Method": "Assumption",
                    "Description": manifest.liver_relative_expression_rule,
                },
            },
            {
                "Path": (
                    "Organism|Liver|Periportal|Intracellular|CYP1A2|"
                    "Relative expression"
                ),
                "Value": manifest.liver_relative_expression,
                "ValueOrigin": {
                    "Source": "Other",
                    "Method": "Assumption",
                    "Description": manifest.liver_relative_expression_rule,
                },
            },
        ],
        "Localization": spec.expression_profile_localization,
        "Ontogeny": {"Name": spec.expression_profile_ontogeny},
    }


def _find_unique(
    values: list[dict[str, Any]],
    *,
    key: str,
    expected: str,
    kind: str,
) -> dict[str, Any]:
    matches = [item for item in values if item.get(key) == expected]
    if len(matches) != 1:
        raise PKSimSnapshotBuildError(
            f"Expected exactly one {kind} {expected!r}; found {len(matches)}."
        )
    return matches[0]


def _diff_paths(before: Any, after: Any, path: str = "") -> set[str]:
    if type(before) is not type(after):
        return {path or "/"}
    if isinstance(before, dict):
        paths: set[str] = set()
        for key in sorted(set(before) | set(after)):
            child = f"{path}/{key}"
            if key not in before or key not in after:
                paths.add(child)
            else:
                paths.update(_diff_paths(before[key], after[key], child))
        return paths
    if isinstance(before, list):
        if len(before) != len(after):
            return {path or "/"}
        paths: set[str] = set()
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            paths.update(_diff_paths(left, right, f"{path}/{index}"))
        return paths
    return set() if before == after else {path or "/"}


def transform_caffeine_snapshot(
    *,
    source_snapshot: dict[str, Any],
    spec: CaffeineSnapshotTransformationSpec,
    manifest: PKSimSystemParameterManifest,
) -> tuple[dict[str, Any], SnapshotTransformationAudit]:
    if canonical_sha256(manifest.model_dump(mode="json", exclude={"checksum"})) != (
        manifest.checksum
    ):
        raise PKSimSnapshotBuildError(
            "PK-Sim system parameter manifest checksum is invalid."
        )
    source = copy.deepcopy(source_snapshot)
    transformed = copy.deepcopy(source_snapshot)
    compounds = transformed.get("Compounds")
    individuals = transformed.get("Individuals")
    simulations = transformed.get("Simulations")
    expression_profiles = transformed.setdefault("ExpressionProfiles", [])
    if not all(
        isinstance(value, list)
        for value in (compounds, individuals, simulations, expression_profiles)
    ):
        raise PKSimSnapshotBuildError(
            "PK-Sim snapshot lacks required project collections."
        )

    compound = _find_unique(
        compounds,
        key="Name",
        expected=spec.compound_name,
        kind="compound",
    )
    process_matches = [
        item
        for item in compound.get("Processes", [])
        if item.get("InternalName") == spec.process_internal_name
        and item.get("DataSource") == spec.process_data_source
        and item.get("Molecule") == spec.expression_profile_molecule
    ]
    if len(process_matches) != 1:
        raise PKSimSnapshotBuildError(
            "The source snapshot does not contain the locked caffeine "
            "CYP1A2 process definition."
        )
    if any(
        item.get("Molecule") == spec.expression_profile_molecule
        and item.get("Species") == spec.expression_profile_species
        and item.get("Category") == spec.expression_profile_category
        for item in expression_profiles
    ):
        raise PKSimSnapshotBuildError(
            "The target CYP1A2 expression profile already exists."
        )
    expression_profiles.append(_cyp1a2_profile(spec=spec, manifest=manifest))

    profile_name = _profile_name(spec)
    for individual_name in spec.target_individual_names:
        individual = _find_unique(
            individuals,
            key="Name",
            expected=individual_name,
            kind="individual",
        )
        profiles = individual.setdefault("ExpressionProfiles", [])
        if profile_name in profiles:
            raise PKSimSnapshotBuildError(
                f"Individual already uses profile {profile_name!r}."
            )
        profiles.append(profile_name)

    source_simulation_names = [item.get("Name") for item in simulations]
    removed_names = {item.name for item in spec.removed_simulations}
    missing_removed = removed_names - set(source_simulation_names)
    if missing_removed:
        raise PKSimSnapshotBuildError(
            "Declared removed simulations are absent: "
            + ", ".join(sorted(missing_removed))
        )
    transformed["Simulations"] = [
        item for item in simulations if item.get("Name") not in removed_names
    ]
    retained_simulations = transformed["Simulations"]
    retained_names = {item.get("Name") for item in retained_simulations}
    if retained_names != set(spec.target_simulation_names):
        raise PKSimSnapshotBuildError(
            "Retained simulations do not exactly match the target set."
        )

    applied_migrations: list[tuple[str, str, str]] = []
    for migration in spec.simulation_parameter_path_migrations:
        simulation = _find_unique(
            retained_simulations,
            key="Name",
            expected=migration.simulation_name,
            kind="simulation",
        )
        parameters = simulation.get("Parameters", [])
        if not isinstance(parameters, list):
            raise PKSimSnapshotBuildError(
                f"Simulation {migration.simulation_name!r} parameters are invalid."
            )
        matches = [
            item
            for item in parameters
            if item.get("Path") == migration.original_path
        ]
        if len(matches) != 1:
            raise PKSimSnapshotBuildError(
                "Expected exactly one source parameter path for migration: "
                f"{migration.simulation_name}:{migration.original_path}"
            )
        if any(
            item.get("Path") == migration.canonical_path
            for item in parameters
        ):
            raise PKSimSnapshotBuildError(
                "Canonical simulation parameter path already exists: "
                f"{migration.simulation_name}:{migration.canonical_path}"
            )
        matches[0]["Path"] = migration.canonical_path
        applied_migrations.append(
            (
                migration.simulation_name,
                migration.original_path,
                migration.canonical_path,
            )
        )

    process_selection = {
        "Name": spec.process_name,
        "MoleculeName": spec.expression_profile_molecule,
    }
    for simulation_name in spec.target_simulation_names:
        simulation = _find_unique(
            retained_simulations,
            key="Name",
            expected=simulation_name,
            kind="simulation",
        )
        if simulation.get("Individual") not in spec.target_individual_names:
            raise PKSimSnapshotBuildError(
                f"Simulation {simulation_name!r} does not use a target individual."
            )
        simulation_compound = _find_unique(
            simulation.get("Compounds", []),
            key="Name",
            expected=spec.compound_name,
            kind="simulation compound",
        )
        selections = simulation_compound.setdefault("Processes", [])
        if any(
            item.get("Name") == spec.process_name
            or item.get("MoleculeName") == spec.expression_profile_molecule
            for item in selections
        ):
            raise PKSimSnapshotBuildError(
                f"Simulation {simulation_name!r} already selects CYP1A2."
            )
        selections.append(copy.deepcopy(process_selection))
        outputs = simulation.setdefault("OutputSelections", [])
        for output in spec.output_selections:
            if output not in outputs:
                outputs.append(output)

    changed_paths = sorted(_diff_paths(source, transformed))
    allowed_changed_paths = [
        "/ExpressionProfiles",
        "/Simulations",
    ]
    for index, individual in enumerate(source.get("Individuals", [])):
        if individual.get("Name") in spec.target_individual_names:
            allowed_changed_paths.append(f"/Individuals/{index}/ExpressionProfiles")
    unexpected_paths = set(changed_paths) - set(allowed_changed_paths)
    checks = {
        "source_compound_unique": True,
        "source_cyp1a2_process_unique": True,
        "expression_profile_added": any(
            item.get("Molecule") == spec.expression_profile_molecule
            and item.get("Category") == spec.expression_profile_category
            for item in transformed["ExpressionProfiles"]
        ),
        "target_individuals_linked": all(
            profile_name in item.get("ExpressionProfiles", [])
            for item in individuals
            if item.get("Name") in spec.target_individual_names
        ),
        "target_simulations_exact": retained_names
        == set(spec.target_simulation_names),
        "parameter_path_migrations_exact": len(applied_migrations)
        == len(spec.simulation_parameter_path_migrations),
        "changes_whitelisted": not unexpected_paths,
    }
    accepted = all(checks.values())
    if not accepted:
        raise PKSimSnapshotBuildError(
            "Snapshot transformation failed deterministic checks."
        )
    transformed_sha256 = canonical_sha256(transformed)
    payload = {
        "audit_schema_version": "1.0.0",
        "builder_version": SNAPSHOT_BUILDER_VERSION,
        "transformation_id": spec.transformation_id,
        "generated_at": datetime.now(UTC),
        "source_snapshot_sha256": canonical_sha256(source),
        "transformed_snapshot_sha256": transformed_sha256,
        "changed_paths": changed_paths,
        "allowed_changed_paths": sorted(set(allowed_changed_paths)),
        "checks": checks,
        "counts": {
            "source_simulations": len(simulations),
            "retained_simulations": len(retained_simulations),
            "removed_simulations": len(removed_names),
            "target_individuals": len(spec.target_individual_names),
            "parameter_path_migrations": len(applied_migrations),
        },
        "accepted": accepted,
    }
    draft_audit = SnapshotTransformationAudit(
        **payload,
        checksum="0" * 64,
    )
    audit = draft_audit.model_copy(
        update={
            "checksum": canonical_sha256(
                draft_audit.model_dump(mode="json", exclude={"checksum"})
            )
        }
    )
    return transformed, audit


def _canonicalize_snapshot_value(
    value: Any,
    *,
    parent_key: str | None = None,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _canonicalize_snapshot_value(item, parent_key=key)
            for key, item in sorted(value.items())
        }
    if isinstance(value, list):
        normalized = [
            _canonicalize_snapshot_value(item) for item in value
        ]
        if parent_key in _UNORDERED_SNAPSHOT_COLLECTIONS:
            normalized.sort(
                key=lambda item: json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        return normalized
    return value


def _semantic_projection(
    snapshot: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    projected = copy.deepcopy(snapshot)
    accepted_normalizations: list[str] = []
    for compound in projected.get("Compounds", []):
        for process in compound.get("Processes", []):
            if process.get("InternalName") != "KidneyClearance":
                continue
            parameters = process.get("Parameters", [])
            dropped = sorted(
                {
                    item.get("Name")
                    for item in parameters
                    if item.get("Name")
                    in _DERIVED_KIDNEY_CLEARANCE_PARAMETERS
                }
            )
            if dropped:
                process["Parameters"] = [
                    item
                    for item in parameters
                    if item.get("Name")
                    not in _DERIVED_KIDNEY_CLEARANCE_PARAMETERS
                ]
                accepted_normalizations.append(
                    "PK-Sim regenerates derived KidneyClearance system "
                    "parameters; excluded from scientific input comparison: "
                    + ", ".join(dropped)
                )
    return (
        _canonicalize_snapshot_value(projected),
        sorted(set(accepted_normalizations)),
    )


def audit_snapshot_roundtrip(
    *,
    transformed_snapshot_path: Path,
    roundtrip_snapshot_path: Path,
    spec: CaffeineSnapshotTransformationSpec,
) -> SnapshotRoundTripAudit:
    transformed = json.loads(
        transformed_snapshot_path.read_text(encoding="utf-8-sig")
    )
    roundtrip = json.loads(
        roundtrip_snapshot_path.read_text(encoding="utf-8-sig")
    )
    left, left_normalizations = _semantic_projection(transformed)
    right, right_normalizations = _semantic_projection(roundtrip)
    accepted_normalizations = sorted(
        set(left_normalizations) | set(right_normalizations)
    )
    semantic_differences = sorted(_diff_paths(left, right))
    roundtrip_simulations = {
        item.get("Name"): item
        for item in roundtrip.get("Simulations", [])
        if isinstance(item, dict) and item.get("Name")
    }

    def simulation_compound(simulation_name: str) -> dict[str, Any] | None:
        simulation = roundtrip_simulations.get(simulation_name)
        if not simulation:
            return None
        matches = [
            item
            for item in simulation.get("Compounds", [])
            if item.get("Name") == spec.compound_name
        ]
        return matches[0] if len(matches) == 1 else None

    simulation_compounds = {
        name: simulation_compound(name)
        for name in spec.target_simulation_names
    }
    checks = {
        "complete_scientific_snapshot_equal": not semantic_differences,
        "target_simulations_preserved": {
            item.get("Name") for item in roundtrip.get("Simulations", [])
        }
        == set(spec.target_simulation_names),
        "cyp1a2_profile_preserved": any(
            item.get("Molecule") == spec.expression_profile_molecule
            and item.get("Category") == spec.expression_profile_category
            for item in roundtrip.get("ExpressionProfiles", [])
        ),
        "cyp1a2_process_selected_in_all_targets": all(
            compound is not None
            and any(
                process.get("Name") == spec.process_name
                and process.get("MoleculeName")
                == spec.expression_profile_molecule
                for process in compound.get("Processes", [])
            )
            for compound in simulation_compounds.values()
        ),
        "renal_clearance_selected_in_all_targets": all(
            compound is not None
            and any(
                process.get("SystemicProcessType") == "Renal"
                for process in compound.get("Processes", [])
            )
            for compound in simulation_compounds.values()
        ),
        "target_protocols_preserved": all(
            simulation_compounds[name] is not None
            and (simulation_compounds[name].get("Protocol") or {}).get("Name")
            == protocol_name
            for name, protocol_name in spec.expected_simulation_protocols.items()
        ),
    }
    accepted = all(checks.values())
    payload = {
        "audit_schema_version": "1.1.0",
        "builder_version": SNAPSHOT_BUILDER_VERSION,
        "transformation_id": spec.transformation_id,
        "generated_at": datetime.now(UTC),
        "transformed_snapshot_sha256": sha256_file(
            transformed_snapshot_path
        ),
        "roundtrip_snapshot_sha256": sha256_file(roundtrip_snapshot_path),
        "transformed_semantic_sha256": canonical_sha256(left),
        "roundtrip_semantic_sha256": canonical_sha256(right),
        "checks": checks,
        "accepted_normalizations": accepted_normalizations,
        "semantic_differences": semantic_differences,
        "accepted": accepted,
    }
    draft = SnapshotRoundTripAudit(
        **payload,
        checksum="0" * 64,
    )
    return draft.model_copy(
        update={
            "checksum": canonical_sha256(
                draft.model_dump(mode="json", exclude={"checksum"})
            )
        }
    )


def build_caffeine_snapshot_package(
    *,
    source_project_path: Path,
    source_snapshot_path: Path,
    pksim_database_path: Path,
    protein_factory_source_path: Path,
    license_path: Path,
    source_root: Path,
    source_commit: str,
    pksim_version: str,
    output_directory: Path,
) -> Path:
    if output_directory.exists():
        raise PKSimSnapshotBuildError(
            f"Immutable output directory exists: {output_directory}"
        )
    if sha256_file(source_snapshot_path) == "0" * 64:
        raise PKSimSnapshotBuildError("Invalid source snapshot checksum.")
    manifest = build_pksim_system_parameter_manifest(
        source_project_path=source_project_path,
        source_snapshot_path=source_snapshot_path,
        pksim_database_path=pksim_database_path,
        protein_factory_source_path=protein_factory_source_path,
        license_path=license_path,
        source_root=source_root,
        source_commit=source_commit,
        pksim_version=pksim_version,
    )
    spec_payload = {
        "spec_schema_version": "1.0.0",
        "builder_version": SNAPSHOT_BUILDER_VERSION,
        "transformation_id": "caffeine-cyp1a2-adult-base-pk-v1",
        "created_at": datetime.now(UTC),
        "source_commit": source_commit,
        "pksim_version": pksim_version,
        "source_project_sha256": sha256_file(source_project_path),
        "source_snapshot_sha256": sha256_file(source_snapshot_path),
        "system_parameter_manifest_sha256": manifest.checksum,
        "expression_profile_category": "PBPK-KB-Caffeine-Adult-v1",
        "target_individual_names": ["Ind"],
        "target_simulation_names": ["S1", "S2"],
        "expected_simulation_protocols": {
            "S1": "Oral",
            "S2": "IV",
        },
        "removed_simulations": [
            {
                "name": "S3",
                "reason_code": (
                    "OUT_OF_SCOPE_PEDIATRIC_CYP3A4_TEST_FIXTURE"
                ),
            }
        ],
        "simulation_parameter_path_migrations": [
            {
                "simulation_name": "S1",
                "original_path": (
                    "Applications|Oral|Weibull|Application_1|"
                    "ProtocolSchemaItem|Volume of water/body weight"
                ),
                "canonical_path": (
                    "Events|Oral|Weibull|Application_1|"
                    "ProtocolSchemaItem|Volume of water/body weight"
                ),
                "reason_code": (
                    "PKSIM_12_3_SNAPSHOT_IMPORT_CANONICAL_EVENT_PATH"
                ),
            }
        ],
        "output_selections": [
            (
                "Organism|PeripheralVenousBlood|Caffeine|"
                "Plasma (Peripheral Venous Blood)"
            )
        ],
    }
    spec = CaffeineSnapshotTransformationSpec(**spec_payload)
    source_snapshot = json.loads(
        source_snapshot_path.read_text(encoding="utf-8-sig")
    )
    if sha256_file(source_snapshot_path) != spec.source_snapshot_sha256:
        raise PKSimSnapshotBuildError("Source snapshot checksum drift.")
    transformed, audit = transform_caffeine_snapshot(
        source_snapshot=source_snapshot,
        spec=spec,
        manifest=manifest,
    )
    temporary = output_directory.with_name(output_directory.name + ".tmp")
    if temporary.exists():
        raise PKSimSnapshotBuildError(
            f"Stale temporary output exists: {temporary}"
        )
    temporary.mkdir(parents=True)
    write_json_atomic(
        temporary / "pksim_system_parameter_manifest.json",
        manifest,
        canonical=True,
    )
    write_json_atomic(
        temporary / "snapshot_transformation_spec.json",
        spec,
        canonical=True,
    )
    write_json_atomic(
        temporary / "transformed_snapshot.json",
        transformed,
        canonical=False,
    )
    write_json_atomic(
        temporary / "transformation_audit.json",
        audit,
        canonical=True,
    )
    files = []
    for path in sorted(temporary.iterdir(), key=lambda item: item.name):
        if path.is_file():
            files.append(
                {
                    "file": path.name,
                    "sha256": sha256_file(path),
                    "byte_size": path.stat().st_size,
                }
            )
    write_json_atomic(
        temporary / "files.sha256.json",
        {"files": files},
        canonical=True,
    )
    temporary.replace(output_directory)
    return output_directory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a provenance-locked caffeine PK-Sim snapshot."
    )
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--pksim-database", type=Path, required=True)
    parser.add_argument("--protein-factory-source", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--pksim-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    output = build_caffeine_snapshot_package(
        source_project_path=args.source_project,
        source_snapshot_path=args.source_snapshot,
        pksim_database_path=args.pksim_database,
        protein_factory_source_path=args.protein_factory_source,
        license_path=args.license,
        source_root=args.source_root,
        source_commit=args.source_commit,
        pksim_version=args.pksim_version,
        output_directory=args.output,
    )
    print(output)


if __name__ == "__main__":
    main()
