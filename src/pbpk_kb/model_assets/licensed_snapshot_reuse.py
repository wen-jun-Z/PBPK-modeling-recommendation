from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from pbpk_kb.model_assets.execution_plan import (
    canonical_sha256,
    write_json_atomic,
)
from pbpk_kb.model_assets.license_policy import (
    resolve_repository_license,
    sha256_file,
)
from pbpk_kb.model_assets.models import LicenseStatus, StrictModel
from pbpk_kb.model_runtime.result_parser import compare_csv_files


PROTOCOL_BUILDER_VERSION = "licensed-snapshot-reuse-builder-1.2.0"
AUDITOR_VERSION = "licensed-snapshot-reuse-auditor-1.4.0"
UNIT_SUFFIX = re.compile(r"^(?P<name>.*?)\s*\[(?P<unit>[^]]+)]\s*$")
ROUNDTRIP_SCHEMA_MIGRATIONS = {(79, 80)}
ROUNDTRIP_OBSERVED_RELATIVE_TOLERANCE = 2e-7
ROUNDTRIP_OBSERVED_ABSOLUTE_TOLERANCE = 1e-10


class SnapshotReuseError(RuntimeError):
    pass


class SourceLock(StrictModel):
    source_id: str
    model_id: str
    model_name: str
    repository_url: str
    release_tag: str
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    license_spdx_id: str
    license_relative_path: str
    license_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_relative_path: str
    snapshot_declared_path: str | None = None
    snapshot_resolution_mode: Literal["declared_relative_path", "unique_basename_fallback"] = (
        "declared_relative_path"
    )
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_schema_version: int = Field(ge=1)
    evaluation_plan_relative_path: str
    evaluation_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_archive_name: str
    source_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExpectedCounts(StrictModel):
    compounds: int = Field(ge=1)
    expression_profiles: int = Field(ge=0)
    formulations: int = Field(ge=0)
    individuals: int = Field(ge=0)
    protocols: int = Field(ge=0)
    simulations: int = Field(ge=1)
    observed_data: int = Field(ge=0)
    simulation_classifications: int = Field(ge=0)
    model_building: int = Field(ge=0)
    model_verification: int = Field(ge=1)
    execution_only: int = Field(ge=0)

    @model_validator(mode="after")
    def partition_covers_simulations(self) -> ExpectedCounts:
        partition_total = (
            self.model_building
            + self.model_verification
            + self.execution_only
        )
        if partition_total != self.simulations:
            raise ValueError(
                "Simulation role counts must sum to the simulation count."
            )
        return self


class PartitionConfig(StrictModel):
    all_plots_path: Literal["Plots.AllPlots"]
    development_section_id: int | None = None
    verification_section_id: int | None = None
    development_section_ids: list[int] = Field(default_factory=list)
    verification_section_ids: list[int] = Field(default_factory=list)
    development_section_references: list[str] = Field(default_factory=list)
    verification_section_references: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def selectors_are_disjoint(self) -> PartitionConfig:
        development = {
            value.casefold() for value in self.development_section_references
        }
        verification = {
            value.casefold() for value in self.verification_section_references
        }
        if development & verification:
            raise ValueError("Development and verification references overlap.")
        development_ids = set(self.development_section_ids)
        verification_ids = set(self.verification_section_ids)
        if self.development_section_id is not None:
            development_ids.add(self.development_section_id)
        if self.verification_section_id is not None:
            verification_ids.add(self.verification_section_id)
        if development_ids & verification_ids:
            raise ValueError("Development and verification section IDs overlap.")
        if not verification_ids and not verification:
            raise ValueError("At least one verification section selector is required.")
        return self


class SnapshotRuntimeLock(StrictModel):
    pksim_version: str
    pksim_portable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cli_relative_path: str
    command_timeout_seconds: int = Field(gt=0)
    required_export_extensions: list[Literal[".pkml", ".xml", ".csv"]]


class ScientificCriteria(StrictModel):
    absolute_tolerance: float = Field(ge=0)
    relative_tolerance: float = Field(ge=0)
    pooled_gmfe_max: float = Field(ge=1)
    pooled_bias_min: float = Field(gt=0)
    pooled_bias_max: float = Field(gt=0)
    pooled_two_fold_fraction_min: float = Field(ge=0, le=1)
    dataset_gmfe_max: float = Field(ge=1)
    dataset_bias_min: float = Field(gt=0)
    dataset_bias_max: float = Field(gt=0)
    endpoint_ratio_min: float = Field(gt=0)
    endpoint_ratio_max: float = Field(gt=0)
    endpoint_scenario_fraction_min: float = Field(ge=0, le=1)
    positive_pair_count_min: int = Field(default=2, ge=2)
    positive_pair_fraction_min: float = Field(default=0.8, gt=0, le=1)


class SnapshotReuseConfig(StrictModel):
    schema_version: Literal["1.0.0"]
    source: SourceLock
    expected_counts: ExpectedCounts
    partition: PartitionConfig
    runtime: SnapshotRuntimeLock
    criteria: ScientificCriteria


class FrozenObservedSeries(StrictModel):
    dataset_name: str
    dataset_index: int = Field(ge=0)
    column_index: int = Field(ge=0)
    source_location: str
    reference: str | None = None
    output_path: str
    scaling: Literal["Log", "Linear"]
    time_values: list[float]
    time_unit: str
    observed_values: list[float]
    observed_unit: str
    molecular_weight_g_per_mol: float = Field(gt=0)

    @model_validator(mode="after")
    def aligned_observations(self) -> FrozenObservedSeries:
        if len(self.time_values) != len(self.observed_values):
            raise ValueError("Observed time and value arrays must have equal length.")
        if not self.time_values:
            raise ValueError("Observed series cannot be empty.")
        return self


class FrozenSimulation(StrictModel):
    sequence: int = Field(ge=1)
    name: str
    role: Literal["model_building", "model_verification", "execution_only"]
    individual: str | None = None
    observed_dataset_names: list[str]
    evaluation_series: FrozenObservedSeries | None = None
    additional_evaluation_series: list[FrozenObservedSeries] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def verification_has_series(self) -> FrozenSimulation:
        if self.role != "model_verification" and (
            self.evaluation_series is not None or self.additional_evaluation_series
        ):
            raise ValueError(
                "Only verification simulations may carry evaluation series."
            )
        return self

    def all_evaluation_series(self) -> list[FrozenObservedSeries]:
        return (
            ([self.evaluation_series] if self.evaluation_series is not None else [])
            + self.additional_evaluation_series
        )


class LicensedSnapshotReuseProtocol(StrictModel):
    protocol_schema_version: Literal["1.0.0"] = "1.0.0"
    protocol_builder_version: str
    protocol_id: str = Field(pattern=r"^snapshot-reuse-[0-9a-f]{32}$")
    generated_at: datetime
    source: SourceLock
    source_archive_path: str
    source_snapshot_semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    counts: ExpectedCounts
    runtime: SnapshotRuntimeLock
    criteria: ScientificCriteria
    simulations: list[FrozenSimulation]

    @model_validator(mode="after")
    def has_independent_quantitative_validation(
        self,
    ) -> LicensedSnapshotReuseProtocol:
        series_count = sum(
            len(simulation.all_evaluation_series())
            for simulation in self.simulations
            if simulation.role == "model_verification"
        )
        if series_count == 0:
            raise ValueError(
                "Protocol requires at least one independent quantitative validation series."
            )
        return self


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise SnapshotReuseError(
            result.stderr.strip() or result.stdout.strip() or "git command failed"
        )
    return result.stdout.strip()


def _normalized_repository_url(value: str) -> str:
    return value.strip().removesuffix("/").removesuffix(".git").casefold()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotReuseError(f"Unable to parse JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SnapshotReuseError(f"Expected a JSON object in {path}.")
    return value


def load_snapshot_reuse_config(path: Path) -> SnapshotReuseConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SnapshotReuseError(f"Unable to read snapshot reuse config: {exc}") from exc
    return SnapshotReuseConfig.model_validate(payload)


def _named_list(value: Any, *, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise SnapshotReuseError(f"Snapshot field {field} must be a list of objects.")
    names = [item.get("Name") for item in value]
    if any(not isinstance(name, str) or not name for name in names):
        raise SnapshotReuseError(f"Every {field} item must have a non-empty Name.")
    if len(names) != len(set(names)):
        raise SnapshotReuseError(f"Snapshot field {field} contains duplicate names.")
    return value


def _semantic_normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _semantic_normalize(item)
            for key, item in sorted(value.items())
            if key not in {"HasResults", "IndividualAnalyses"}
        }
    if isinstance(value, list):
        normalized = [_semantic_normalize(item) for item in value]
        if normalized and all(isinstance(item, dict) for item in normalized):
            for identity_key in ("Name", "Path", "Id", "InternalName"):
                identities = [item.get(identity_key) for item in normalized]
                if all(isinstance(item, str) for item in identities) and len(
                    set(identities)
                ) == len(identities):
                    return sorted(normalized, key=lambda item: item[identity_key])
        return normalized
    return value


def snapshot_semantic_sha256(snapshot: dict[str, Any]) -> str:
    return canonical_sha256(_semantic_normalize(snapshot))


def _roundtrip_normalize(snapshot: dict[str, Any]) -> tuple[int, Any]:
    value = copy.deepcopy(snapshot)
    version = value.pop("Version", None)
    if not isinstance(version, int):
        raise SnapshotReuseError("Snapshot Version must be an integer.")
    for dataset in value.get("ObservedData", []):
        for property_value in dataset.get("ExtendedProperties", []):
            if property_value.get("Value") == "":
                property_value.pop("Value")
        for column in dataset.get("Columns", []):
            dimension = str(column.get("Dimension") or "").casefold()
            unit = column.get("Unit")
            normalized_unit = (
                str(unit or "")
                .strip()
                .casefold()
                .replace("µ", "u")
                .replace("μ", "u")
                .replace(" ", "")
                .replace("liter", "l")
                .replace("litre", "l")
            )
            factor: float | None = None
            canonical_unit: str | None = None
            if dimension == "mass":
                mass_factors_to_mg = {
                    "kg": 1e6, "g": 1e3, "mg": 1.0,
                    "ug": 1e-3, "ng": 1e-6, "pg": 1e-9,
                }
                factor = mass_factors_to_mg.get(normalized_unit)
                if factor is not None:
                    canonical_unit = "mg"
            elif dimension == "concentration (mass)":
                mass_factors_to_mg_per_litre = {
                    "g/l": 1e3,
                    "mg/l": 1.0,
                    "ug/l": 1e-3,
                    "ng/l": 1e-6,
                    "pg/l": 1e-9,
                    "g/ml": 1e6,
                    "mg/ml": 1e3,
                    "ug/ml": 1.0,
                    "ng/ml": 1e-3,
                    "pg/ml": 1e-6,
                }
                factor = mass_factors_to_mg_per_litre.get(normalized_unit)
                if factor is not None:
                    canonical_unit = "mg/l"
            elif dimension == "concentration (molar)":
                molar_factors_to_mol_per_litre = {
                    "mol/l": 1.0,
                    "mmol/l": 1e-3,
                    "umol/l": 1e-6,
                    "nmol/l": 1e-9,
                    "pmol/l": 1e-12,
                }
                factor = molar_factors_to_mol_per_litre.get(normalized_unit)
                if factor is not None:
                    canonical_unit = "mol/l"
            elif dimension == "fraction":
                if unit in {None, ""}:
                    factor = 1.0
                    canonical_unit = None
                elif normalized_unit == "%":
                    factor = 0.01
                    canonical_unit = None
            if factor is not None:
                values = column.get("Values")
                if not isinstance(values, list) or not all(
                    isinstance(item, (int, float)) and not isinstance(item, bool)
                    for item in values
                ):
                    raise SnapshotReuseError(
                        "Round-trip observed-data values must be numeric."
                    )
                column["Values"] = [float(item) * factor for item in values]
                column["Unit"] = canonical_unit
    for simulation in value.get("Simulations", []):
        for parameter in simulation.get("Parameters", []):
            path = parameter.get("Path")
            if isinstance(path, str) and path.startswith("Events|"):
                parameter["Path"] = "Applications|" + path.removeprefix("Events|")
    # Only chart axes have this display-only default. A model parameter or
    # observed-data dimension with the same spelling remains fully compared.
    charts = [
        analysis.get("Chart", {})
        for identification in value.get("ParameterIdentifications", [])
        for analysis in identification.get("Analyses", [])
    ] + [
        comparison.get("IndividualComparison", {})
        for comparison in value.get("SimulationComparisons", [])
    ]
    for chart in charts:
        for axis in chart.get("Axes", []):
            if axis.get("Dimension") == "Dimensionless" and axis.get("Unit") in {None, ""}:
                axis.pop("Dimension")
    return version, _semantic_normalize(value)


def _normalize_residual_chart_curve_elision(
    input_snapshot: dict[str, Any], roundtrip_snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    """Accept only omitted cached residual *display* curves, never fit mappings.

    Empty X marks a computed residual curve; numerical observations, parameter
    identification configuration, and simulation OutputMappings are compared
    unchanged elsewhere. Changes to a retained curve are not normalized away.
    """
    right_identifications = {
        item.get("Name"): item
        for item in roundtrip_snapshot.get("ParameterIdentifications", [])
    }
    records = []
    for left in input_snapshot.get("ParameterIdentifications", []):
        right = right_identifications.get(left.get("Name"), {})
        right_analyses = {
            (item.get("Name"), item.get("Type")): item
            for item in right.get("Analyses", [])
        }
        for analysis in left.get("Analyses", []):
            if analysis.get("Type") != "ParameterIdentificationResidualVsTimeChart":
                continue
            target = right_analyses.get((analysis.get("Name"), analysis.get("Type")))
            if target is None or not isinstance(target.get("Chart"), dict):
                continue
            chart = analysis.get("Chart", {})
            curves = chart.get("Curves")
            if "Curves" in target["Chart"] or not isinstance(curves, list) or not curves:
                continue
            display_options = {"Color", "LegendIndex", "LineStyle", "Symbol", "VisibleInLegend"}
            if not all(
                isinstance(curve, dict)
                and set(curve) <= {"Name", "X", "Y", "CurveOptions"}
                and curve.get("X") == ""
                and isinstance(curve.get("Y"), str) and bool(curve["Y"])
                and isinstance(curve.get("CurveOptions"), dict)
                and set(curve["CurveOptions"]) <= display_options
                for curve in curves
            ):
                continue
            records.append({"parameter_identification": left.get("Name"),
                            "analysis": analysis.get("Name"),
                            "analysis_type": analysis["Type"],
                            "omitted_display_curve_count": len(curves),
                            "basis": "cached_residual_display_curves_with_empty_x_only"})
            chart.pop("Curves")
    return records


def _roundtrip_value_differences(
    left: Any,
    right: Any,
    *,
    path: tuple[str | int, ...] = (),
) -> tuple[int, list[dict[str, Any]]]:
    difference_count = 0
    examples: list[dict[str, Any]] = []

    def display_path(parts: tuple[str | int, ...]) -> str:
        rendered = "$"
        for part in parts:
            rendered += f"[{part}]" if isinstance(part, int) else f".{part}"
        return rendered

    def record(kind: str, value_a: Any, value_b: Any) -> None:
        nonlocal difference_count
        difference_count += 1
        if len(examples) < 50:
            examples.append(
                {
                    "path": display_path(current_path),
                    "kind": kind,
                    "input": repr(value_a)[:240],
                    "roundtrip": repr(value_b)[:240],
                }
            )

    def compare(value_a: Any, value_b: Any, parts: tuple[str | int, ...]) -> None:
        nonlocal current_path
        current_path = parts
        observed_value = (
            "ObservedData" in parts
            and "Columns" in parts
            and "Values" in parts
            and isinstance(value_a, (int, float))
            and not isinstance(value_a, bool)
            and isinstance(value_b, (int, float))
            and not isinstance(value_b, bool)
        )
        if observed_value:
            if not math.isclose(
                float(value_a),
                float(value_b),
                rel_tol=ROUNDTRIP_OBSERVED_RELATIVE_TOLERANCE,
                abs_tol=ROUNDTRIP_OBSERVED_ABSOLUTE_TOLERANCE,
            ):
                record("observed_numeric_difference", value_a, value_b)
            return
        if type(value_a) is not type(value_b):
            record("type_difference", type(value_a).__name__, type(value_b).__name__)
            return
        if isinstance(value_a, dict):
            for key in sorted(set(value_a) | set(value_b)):
                current_path = (*parts, key)
                if key not in value_a:
                    record("added_key", None, value_b[key])
                elif key not in value_b:
                    record("removed_key", value_a[key], None)
                else:
                    compare(value_a[key], value_b[key], (*parts, key))
            return
        if isinstance(value_a, list):
            if len(value_a) != len(value_b):
                record("list_length_difference", len(value_a), len(value_b))
            for index, (item_a, item_b) in enumerate(zip(value_a, value_b)):
                compare(item_a, item_b, (*parts, index))
            return
        if value_a != value_b:
            record("value_difference", value_a, value_b)

    current_path = path
    compare(left, right, path)
    return difference_count, examples


def compare_roundtrip_snapshots(
    *,
    input_snapshot: dict[str, Any],
    roundtrip_snapshot: dict[str, Any],
    expected_input_version: int,
) -> dict[str, Any]:
    input_version, normalized_input = _roundtrip_normalize(input_snapshot)
    roundtrip_version, normalized_roundtrip = _roundtrip_normalize(
        roundtrip_snapshot
    )
    display_elisions = _normalize_residual_chart_curve_elision(
        normalized_input, normalized_roundtrip
    )
    schema_version_accepted = (
        input_version == expected_input_version
        and (
            roundtrip_version == input_version
            or (input_version, roundtrip_version) in ROUNDTRIP_SCHEMA_MIGRATIONS
        )
    )
    difference_count, difference_examples = _roundtrip_value_differences(
        normalized_input,
        normalized_roundtrip,
    )
    return {
        "accepted": schema_version_accepted and difference_count == 0,
        "input_schema_version": input_version,
        "roundtrip_schema_version": roundtrip_version,
        "schema_version_accepted": schema_version_accepted,
        "normalized_difference_count": difference_count,
        "difference_examples": difference_examples,
        "display_only_elisions": display_elisions,
        "observed_value_relative_tolerance": (
            ROUNDTRIP_OBSERVED_RELATIVE_TOLERANCE
        ),
        "observed_value_absolute_tolerance": (
            ROUNDTRIP_OBSERVED_ABSOLUTE_TOLERANCE
        ),
        "allowed_migrations": [
            "snapshot_schema_79_to_80",
            "simulation_parameter_path_Applications_to_Events",
            "observed_mass_concentration_unit_equivalence",
            "observed_mass_unit_equivalence",
            "observed_molar_concentration_unit_equivalence",
            "observed_fraction_to_percent",
            "empty_extended_property_value_omission",
            "chart_axis_dimensionless_default_omission",
            "cached_residual_display_curve_elision_with_empty_x",
        ],
    }


def _extended_property(dataset: dict[str, Any], name: str) -> Any:
    for item in dataset.get("ExtendedProperties", []):
        if item.get("Name") == name:
            return item.get("Value")
    return None


def _observation_columns(
    dataset: dict[str, Any],
) -> list[tuple[int, dict[str, Any]]]:
    candidates = []
    for column_index, column in enumerate(dataset.get("Columns", [])):
        data_info = column.get("DataInfo") or {}
        dimension = str(column.get("Dimension") or "").casefold()
        if data_info.get("Origin") == "Observation" and dimension.startswith(
            "concentration"
        ):
            candidates.append((column_index, column))
    return candidates


def _observation_column(dataset: dict[str, Any]) -> dict[str, Any]:
    candidates = _observation_columns(dataset)
    if len(candidates) != 1:
        raise SnapshotReuseError(
            f"Expected one concentration observation column for {dataset.get('Name')!r}; "
            f"found {len(candidates)}."
        )
    return candidates[0][1]


def _frozen_observation(
    *,
    snapshot_relative_path: str,
    dataset: dict[str, Any],
    dataset_index: int,
    column: dict[str, Any],
    column_index: int,
    output_path: str,
    scaling: str,
) -> FrozenObservedSeries:
    grid = dataset.get("BaseGrid") or {}
    time_values = grid.get("Values")
    observed_values = column.get("Values")
    if not isinstance(time_values, list) or not isinstance(observed_values, list):
        raise SnapshotReuseError("Observed series values must be arrays.")
    molecular_weight = (column.get("DataInfo") or {}).get("MolWeight")
    if not isinstance(molecular_weight, (int, float)) or molecular_weight <= 0:
        raise SnapshotReuseError(
            f"Observed series {dataset.get('Name')!r} has no molecular weight."
        )
    reference = _extended_property(dataset, "Reference")
    return FrozenObservedSeries(
        dataset_name=dataset["Name"],
        dataset_index=dataset_index,
        column_index=column_index,
        source_location=(
            f"{snapshot_relative_path}#/ObservedData/{dataset_index}/Columns/{column_index}"
        ),
        reference=str(reference) if reference is not None else None,
        output_path=output_path,
        scaling="Log" if str(scaling).casefold() == "log" else "Linear",
        time_values=[float(item) for item in time_values],
        time_unit=str(grid.get("Unit") or ""),
        observed_values=[float(item) for item in observed_values],
        observed_unit=str(column.get("Unit") or ""),
        molecular_weight_g_per_mol=float(molecular_weight),
    )


def _verification_series(
    *,
    source: SourceLock,
    simulation: dict[str, Any],
    observed_by_name: dict[str, tuple[int, dict[str, Any]]],
) -> list[FrozenObservedSeries]:
    candidates: list[
        tuple[dict[str, Any], int, dict[str, Any], int, dict[str, Any]]
    ] = []
    seen: set[tuple[str, int, int]] = set()
    for mapping in simulation.get("OutputMappings", []):
        dataset_name = mapping.get("ObservedData")
        if dataset_name not in observed_by_name:
            raise SnapshotReuseError(
                f"Simulation {simulation['Name']!r} maps an unknown dataset "
                f"{dataset_name!r}."
            )
        dataset_index, dataset = observed_by_name[dataset_name]
        output_path = str(mapping.get("Path") or "")
        for column_index, column in _observation_columns(dataset):
            key = (output_path, dataset_index, column_index)
            if key not in seen:
                candidates.append(
                    (mapping, dataset_index, dataset, column_index, column)
                )
                seen.add(key)
    return [
        _frozen_observation(
            snapshot_relative_path=source.snapshot_relative_path,
            dataset=dataset,
            dataset_index=dataset_index,
            column=column,
            column_index=column_index,
            output_path=str(mapping.get("Path") or ""),
            scaling=str(mapping.get("Scaling") or "Linear"),
        )
        for mapping, dataset_index, dataset, column_index, column in candidates
    ]


def _assert_source_integrity(
    *,
    config: SnapshotReuseConfig,
    repository_root: Path,
    source_archive: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bool]]:
    source = config.source
    repository_root = repository_root.resolve()
    if not (repository_root / ".git").exists():
        raise SnapshotReuseError(f"Not a Git repository: {repository_root}")
    if _git(repository_root, "rev-parse", "--is-inside-work-tree") != "true":
        raise SnapshotReuseError(f"Not a Git work tree: {repository_root}")
    actual_commit = _git(repository_root, "rev-parse", "HEAD")
    tag_commit = _git(repository_root, "rev-list", "-n", "1", source.release_tag)
    remote_url = _git(repository_root, "remote", "get-url", "origin")
    checks = {
        "repository_commit_locked": actual_commit == source.commit_sha,
        "release_tag_locked": tag_commit == source.commit_sha,
        "repository_clean": not bool(_git(repository_root, "status", "--porcelain")),
        "repository_url_locked": (
            _normalized_repository_url(remote_url)
            == _normalized_repository_url(source.repository_url)
        ),
        "source_archive_name_locked": (
            source_archive.name == source.source_archive_name
        ),
    }
    locked_files = {
        "license": (
            repository_root / source.license_relative_path,
            source.license_sha256,
        ),
        "snapshot": (
            repository_root / source.snapshot_relative_path,
            source.snapshot_sha256,
        ),
        "evaluation_plan": (
            repository_root / source.evaluation_plan_relative_path,
            source.evaluation_plan_sha256,
        ),
        "source_archive": (source_archive.resolve(), source.source_archive_sha256),
    }
    for name, (path, expected_checksum) in locked_files.items():
        checks[f"{name}_exists"] = path.is_file()
        checks[f"{name}_checksum_locked"] = (
            path.is_file() and sha256_file(path) == expected_checksum
        )
    license_decision = resolve_repository_license(repository_root)
    checks["license_allowed"] = (
        license_decision.status == LicenseStatus.ALLOWED
        and license_decision.spdx_id == source.license_spdx_id
        and license_decision.source_location == source.license_relative_path
        and license_decision.source_checksum == source.license_sha256
    )
    if not all(checks.values()):
        failed = sorted(name for name, accepted in checks.items() if not accepted)
        raise SnapshotReuseError(
            "Source integrity audit failed: " + ", ".join(failed)
        )
    snapshot = _read_json(repository_root / source.snapshot_relative_path)
    evaluation_plan = _read_json(
        repository_root / source.evaluation_plan_relative_path
    )
    return snapshot, evaluation_plan, checks


def build_snapshot_reuse_protocol(
    *,
    config_path: Path,
    repository_root: Path,
    source_archive: Path,
    output_directory: Path,
) -> LicensedSnapshotReuseProtocol:
    config = load_snapshot_reuse_config(config_path)
    snapshot, evaluation_plan, source_checks = _assert_source_integrity(
        config=config,
        repository_root=repository_root,
        source_archive=source_archive,
    )
    source = config.source
    fields = {
        "compounds": "Compounds",
        "expression_profiles": "ExpressionProfiles",
        "formulations": "Formulations",
        "individuals": "Individuals",
        "protocols": "Protocols",
        "simulations": "Simulations",
        "observed_data": "ObservedData",
        "simulation_classifications": "SimulationClassifications",
    }
    actual_counts = {
        name: len(snapshot.get(snapshot_field, []))
        for name, snapshot_field in fields.items()
    }
    expected_base_counts = {
        name: getattr(config.expected_counts, name) for name in fields
    }
    if actual_counts != expected_base_counts:
        raise SnapshotReuseError(
            f"Snapshot count mismatch: expected {expected_base_counts}, found "
            f"{actual_counts}."
        )
    if snapshot.get("Version") != source.snapshot_schema_version:
        raise SnapshotReuseError("Snapshot schema version does not match the lock.")

    simulations = _named_list(snapshot.get("Simulations"), field="Simulations")
    observed_data = _named_list(snapshot.get("ObservedData"), field="ObservedData")
    simulation_by_name = {item["Name"]: item for item in simulations}
    observed_by_name = {
        item["Name"]: (index, item) for index, item in enumerate(observed_data)
    }
    all_plots = (evaluation_plan.get("Plots") or {}).get("AllPlots")
    if not isinstance(all_plots, list):
        raise SnapshotReuseError("Evaluation plan has no Plots.AllPlots array.")
    plot_names = [item.get("Simulation") for item in all_plots]
    if any(name not in simulation_by_name for name in plot_names):
        raise SnapshotReuseError("Evaluation plan references unknown simulations.")
    if len(plot_names) != len(set(plot_names)):
        raise SnapshotReuseError("Evaluation plan classifies a simulation more than once.")
    development_references = {
        value.casefold()
        for value in config.partition.development_section_references
    }
    verification_references = {
        value.casefold()
        for value in config.partition.verification_section_references
    }
    development_ids = set(config.partition.development_section_ids)
    verification_ids = set(config.partition.verification_section_ids)
    if config.partition.development_section_id is not None:
        development_ids.add(config.partition.development_section_id)
    if config.partition.verification_section_id is not None:
        verification_ids.add(config.partition.verification_section_id)

    def plot_partition(item: dict[str, Any]) -> str | None:
        section_id = item.get("SectionId")
        section_reference = str(item.get("SectionReference") or "").casefold()
        if (
            section_id in development_ids
        ) or section_reference in development_references:
            return "development"
        if (
            section_id in verification_ids
        ) or section_reference in verification_references:
            return "verification"
        return None

    development = {
        item["Simulation"]
        for item in all_plots
        if plot_partition(item) == "development"
    }
    verification = {
        item["Simulation"]
        for item in all_plots
        if plot_partition(item) == "verification"
    }
    unsupported_section_names = {
        item["Simulation"]
        for item in all_plots
        if plot_partition(item) is None
    }
    if unsupported_section_names:
        raise SnapshotReuseError(
            "Evaluation plan contains simulations under an unsupported section: "
            + ", ".join(sorted(unsupported_section_names))
        )
    if development & verification:
        raise SnapshotReuseError("Development and verification sets overlap.")
    execution_only = set(simulation_by_name) - development - verification
    role_counts = {
        "model_building": len(development),
        "model_verification": len(verification),
        "execution_only": len(execution_only),
    }
    expected_role_counts = {
        name: getattr(config.expected_counts, name) for name in role_counts
    }
    if role_counts != expected_role_counts:
        raise SnapshotReuseError(
            f"Evaluation role count mismatch: expected {expected_role_counts}, "
            f"found {role_counts}."
        )

    frozen_simulations: list[FrozenSimulation] = []
    for sequence, simulation in enumerate(simulations, start=1):
        name = simulation["Name"]
        if name in development:
            role = "model_building"
        elif name in verification:
            role = "model_verification"
        else:
            role = "execution_only"
        evaluation_series = (
            _verification_series(
                source=source,
                simulation=simulation,
                observed_by_name=observed_by_name,
            )
            if role == "model_verification"
            else []
        )
        frozen_simulations.append(
            FrozenSimulation(
                sequence=sequence,
                name=name,
                role=role,
                individual=simulation.get("Individual"),
                observed_dataset_names=list(simulation.get("ObservedData") or []),
                evaluation_series=(evaluation_series[0] if evaluation_series else None),
                additional_evaluation_series=evaluation_series[1:],
            )
        )

    semantic_sha256 = snapshot_semantic_sha256(snapshot)
    identity = {
        "source": source,
        "source_snapshot_semantic_sha256": semantic_sha256,
        "counts": config.expected_counts,
        "runtime": config.runtime,
        "criteria": config.criteria,
        "simulations": frozen_simulations,
    }
    protocol = LicensedSnapshotReuseProtocol(
        protocol_builder_version=PROTOCOL_BUILDER_VERSION,
        protocol_id=f"snapshot-reuse-{canonical_sha256(identity)[:32]}",
        generated_at=datetime.now(UTC),
        source=source,
        source_archive_path=str(source_archive.resolve()),
        source_snapshot_semantic_sha256=semantic_sha256,
        counts=config.expected_counts,
        runtime=config.runtime,
        criteria=config.criteria,
        simulations=frozen_simulations,
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    protocol_path = output_directory / "snapshot_reuse_protocol.json"
    protocol_sha256 = write_json_atomic(protocol_path, protocol, canonical=True)
    write_json_atomic(
        output_directory / "source_integrity_audit.json",
        {
            "accepted": True,
            "auditor_version": PROTOCOL_BUILDER_VERSION,
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol_sha256,
            "checks": source_checks,
            "actual_counts": {**actual_counts, **role_counts},
            "source_snapshot_semantic_sha256": semantic_sha256,
        },
        canonical=True,
    )
    return protocol


class ExecutionArtifact(StrictModel):
    relative_path: str
    simulation_relative_path: str
    byte_size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    extension: str


class SimulationExecution(StrictModel):
    sequence: int = Field(ge=1)
    name: str
    status: Literal["passed", "failed"]
    return_code: int
    duration_seconds: float = Field(ge=0)
    command: list[str]
    stdout_relative_path: str
    stderr_relative_path: str
    artifacts: list[ExecutionArtifact]
    reason_codes: list[str]


class SnapshotExecutionRun(StrictModel):
    run_id: Literal["run_1", "run_2"]
    status: Literal["passed", "failed"]
    simulations: list[SimulationExecution]


class SnapshotWindowsManifest(StrictModel):
    manifest_schema_version: Literal["1.1.0"]
    protocol_id: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_text_encoding: Literal["utf-8"]
    worker_script_relative_path: str
    worker_script_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["passed", "failed"]
    cli_version_output: str
    cli_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_snapshot_relative_path: str
    input_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reconstructed_project_relative_path: str
    reconstructed_project_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    roundtrip_snapshot_relative_path: str
    roundtrip_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runs: list[SnapshotExecutionRun]
    completed_at_utc: datetime


def _safe_artifact_path(root: Path, relative_path: str) -> Path:
    windows_path = PureWindowsPath(relative_path)
    normalized = PurePosixPath(relative_path.replace("\\", "/"))
    if (
        windows_path.drive
        or windows_path.is_absolute()
        or normalized.is_absolute()
        or ".." in normalized.parts
    ):
        raise SnapshotReuseError(
            f"Execution artifact path is not relative: {relative_path!r}."
        )
    path = root.joinpath(*normalized.parts).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise SnapshotReuseError(
            f"Execution artifact escapes result root: {relative_path!r}."
        ) from exc
    if not path.is_file():
        raise SnapshotReuseError(f"Execution artifact is missing: {relative_path!r}.")
    return path


def _verify_artifact(root: Path, artifact: ExecutionArtifact) -> Path:
    path = _safe_artifact_path(root, artifact.relative_path)
    if path.stat().st_size != artifact.byte_size:
        raise SnapshotReuseError(
            f"Execution artifact size mismatch: {artifact.relative_path}."
        )
    if sha256_file(path) != artifact.sha256:
        raise SnapshotReuseError(
            f"Execution artifact checksum mismatch: {artifact.relative_path}."
        )
    if path.suffix.casefold() != artifact.extension.casefold():
        raise SnapshotReuseError(
            f"Execution artifact extension mismatch: {artifact.relative_path}."
        )
    return path


def _split_header(header: str) -> tuple[str, str | None]:
    match = UNIT_SUFFIX.match(header.strip())
    if match is None:
        return header.strip(), None
    return match.group("name").strip(), match.group("unit").strip()


def _read_data_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.reader(
            line
            for line in handle
            if line.strip() and not line.startswith("#")
        )
        headers = next(rows, None)
    if headers is None:
        raise SnapshotReuseError(f"CSV is empty: {path}")
    if not headers:
        raise SnapshotReuseError(f"CSV has no columns: {path}")
    return headers


def _normalized_unit(unit: str) -> str:
    return (
        unit.strip()
        .casefold()
        .replace("\u00b5", "u")
        .replace("\u03bc", "u")
        .replace(" ", "")
        .replace("liter", "l")
        .replace("litre", "l")
    )


def _time_in_hours(value: float, unit: str) -> float:
    normalized = _normalized_unit(unit)
    factors = {
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
        "day(s)": 24.0,
    }
    if normalized not in factors:
        raise SnapshotReuseError(f"Unsupported time unit: {unit!r}.")
    return value * factors[normalized]


def _concentration_in_mol_per_litre(
    value: float,
    unit: str,
    molecular_weight_g_per_mol: float,
) -> float:
    normalized = _normalized_unit(unit)
    molar_factors = {
        "mol/l": 1.0,
        "mmol/l": 1e-3,
        "umol/l": 1e-6,
        "nmol/l": 1e-9,
        "pmol/l": 1e-12,
    }
    if normalized in molar_factors:
        return value * molar_factors[normalized]
    grams_per_litre = {
        "kg/l": 1e3,
        "g/l": 1.0,
        "mg/l": 1e-3,
        "ug/l": 1e-6,
        "ng/l": 1e-9,
        "pg/l": 1e-12,
        "g/ml": 1e3,
        "mg/ml": 1.0,
        "ug/ml": 1e-3,
        "ng/ml": 1e-6,
        "pg/ml": 1e-9,
    }
    if normalized not in grams_per_litre:
        raise SnapshotReuseError(f"Unsupported concentration unit: {unit!r}.")
    return value * grams_per_litre[normalized] / molecular_weight_g_per_mol


def _prediction_path_candidates(
    simulation_name: str,
    output_path: str,
) -> set[str]:
    values = {output_path.strip().casefold()}
    prefix = simulation_name.strip().casefold() + "|"
    if output_path.strip().casefold().startswith(prefix):
        values.add(output_path.strip()[len(simulation_name) + 1 :].casefold())
    return values


def _prediction_series_from_csvs(
    *,
    result_root: Path,
    artifacts: list[ExecutionArtifact],
    simulation_name: str,
    observed: FrozenObservedSeries,
) -> tuple[list[float], list[float], str, str]:
    targets = _prediction_path_candidates(simulation_name, observed.output_path)
    matches: list[tuple[Path, int, int, str, str]] = []
    for artifact in artifacts:
        if artifact.extension.casefold() != ".csv":
            continue
        path = _verify_artifact(result_root, artifact)
        headers = _read_data_csv_header(path)
        parsed_headers = [_split_header(header) for header in headers]
        time_indexes = [
            index
            for index, (name, _) in enumerate(parsed_headers)
            if name.casefold() == "time"
        ]
        if len(time_indexes) != 1:
            continue
        for output_index, (name, unit) in enumerate(parsed_headers):
            normalized_name = name.casefold()
            if normalized_name in targets or any(
                normalized_name.endswith("|" + target) for target in targets
            ):
                if unit is None or parsed_headers[time_indexes[0]][1] is None:
                    raise SnapshotReuseError(
                        f"Prediction CSV has an output or time column without units: {path}"
                    )
                matches.append(
                    (
                        path,
                        time_indexes[0],
                        output_index,
                        unit,
                        parsed_headers[time_indexes[0]][1],
                    )
                )
    if len(matches) != 1:
        raise SnapshotReuseError(
            f"Expected one prediction column for {simulation_name!r}; found "
            f"{len(matches)}."
        )
    path, time_index, output_index, output_unit, time_unit = matches[0]
    points: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.reader(
            line
            for line in handle
            if line.strip() and not line.startswith("#")
        )
        headers = next(rows, None)
        if headers is None:
            raise SnapshotReuseError(f"CSV is empty: {path}")
        for row in rows:
            if len(row) != len(headers):
                raise SnapshotReuseError(
                    f"CSV row width is inconsistent: {path}"
                )
            try:
                time = _time_in_hours(float(row[time_index]), time_unit)
                value = _concentration_in_mol_per_litre(
                    float(row[output_index]),
                    output_unit,
                    observed.molecular_weight_g_per_mol,
                )
            except ValueError as exc:
                raise SnapshotReuseError(
                    f"Non-numeric prediction value in {path}."
                ) from exc
            if not math.isfinite(time) or not math.isfinite(value):
                raise SnapshotReuseError(f"Non-finite prediction value in {path}.")
            points.append((time, value))
    if not points:
        raise SnapshotReuseError(f"CSV has no data rows: {path}")
    points.sort()
    if len(points) < 2 or len({time for time, _ in points}) != len(points):
        raise SnapshotReuseError(
            f"Prediction time grid must contain unique points: {path}."
        )
    return (
        [time for time, _ in points],
        [value for _, value in points],
        output_unit,
        str(path),
    )


def _interpolate(
    times: list[float],
    values: list[float],
    target: float,
    *,
    logarithmic: bool,
) -> float:
    if target < times[0] or target > times[-1]:
        raise SnapshotReuseError(
            f"Observed time {target} h is outside the prediction range."
        )
    for index, time in enumerate(times):
        if math.isclose(target, time, rel_tol=0, abs_tol=1e-12):
            return values[index]
        if time > target:
            left = index - 1
            fraction = (target - times[left]) / (time - times[left])
            left_value = values[left]
            right_value = values[index]
            if logarithmic and left_value > 0 and right_value > 0:
                return math.exp(
                    math.log(left_value)
                    + fraction * (math.log(right_value) - math.log(left_value))
                )
            return left_value + fraction * (right_value - left_value)
    return values[-1]


def _geometric_metrics(observed: list[float], predicted: list[float]) -> dict[str, float]:
    if len(observed) != len(predicted) or not observed:
        raise SnapshotReuseError("Geometric metrics require aligned non-empty arrays.")
    log_ratios = [
        math.log(predicted_value / observed_value)
        for observed_value, predicted_value in zip(observed, predicted, strict=True)
        if observed_value > 0 and predicted_value > 0
    ]
    if len(log_ratios) != len(observed):
        raise SnapshotReuseError("Geometric metrics require positive concentrations.")
    return {
        "gmfe": math.exp(sum(abs(value) for value in log_ratios) / len(log_ratios)),
        "geometric_bias": math.exp(sum(log_ratios) / len(log_ratios)),
        "log_rmse": math.sqrt(
            sum(value * value for value in log_ratios) / len(log_ratios)
        ),
        "two_fold_fraction": sum(
            1 for value in log_ratios if abs(value) <= math.log(2.0)
        )
        / len(log_ratios),
    }


def _auc(times: list[float], values: list[float]) -> float:
    return sum(
        (times[index] - times[index - 1])
        * (values[index] + values[index - 1])
        / 2.0
        for index in range(1, len(times))
    )


def _dataset_metrics(
    *,
    result_root: Path,
    artifacts: list[ExecutionArtifact],
    simulation: FrozenSimulation,
    observed: FrozenObservedSeries,
    criteria: ScientificCriteria,
) -> dict[str, Any]:
    predicted_times, predicted_values, predicted_unit, prediction_path = (
        _prediction_series_from_csvs(
            result_root=result_root,
            artifacts=artifacts,
            simulation_name=simulation.name,
            observed=observed,
        )
    )
    observed_times = [
        _time_in_hours(value, observed.time_unit) for value in observed.time_values
    ]
    if len(set(observed_times)) != len(observed_times):
        raise SnapshotReuseError(
            f"Observed time grid is not unique for {observed.dataset_name!r}."
        )
    observed_molar = [
        _concentration_in_mol_per_litre(
            value,
            observed.observed_unit,
            observed.molecular_weight_g_per_mol,
        )
        for value in observed.observed_values
    ]
    predicted_at_observed = [
        _interpolate(
            predicted_times,
            predicted_values,
            time,
            logarithmic=observed.scaling == "Log",
        )
        for time in observed_times
    ]
    positive_observation_pairs = [
        (time, observed_value, predicted_value)
        for time, observed_value, predicted_value in zip(
            observed_times,
            observed_molar,
            predicted_at_observed,
            strict=True,
        )
        if observed_value > 0
    ]
    positive_pairs = [
        item for item in positive_observation_pairs if item[2] > 0
    ]
    positive_pair_fraction = (
        len(positive_pairs) / len(positive_observation_pairs)
        if positive_observation_pairs
        else 0.0
    )
    if (
        len(positive_pairs) < criteria.positive_pair_count_min
        or positive_pair_fraction < criteria.positive_pair_fraction_min
    ):
        raise SnapshotReuseError(
            f"Verification series {observed.dataset_name!r} retained "
            f"{len(positive_pairs)}/{len(positive_observation_pairs)} positive "
            "observation-prediction pairs; required at least "
            f"{criteria.positive_pair_count_min} pairs and fraction "
            f">={criteria.positive_pair_fraction_min}."
        )
    metrics = _geometric_metrics(
        [item[1] for item in positive_pairs],
        [item[2] for item in positive_pairs],
    )
    dataset_guard_passed = (
        metrics["gmfe"] <= criteria.dataset_gmfe_max
        and criteria.dataset_bias_min
        <= metrics["geometric_bias"]
        <= criteria.dataset_bias_max
    )
    endpoint_evaluable = len(observed_times) >= 2
    auc_ratio = None
    cmax_ratio = None
    endpoint_passed = False
    if endpoint_evaluable:
        ordered = sorted(
            zip(observed_times, observed_molar, predicted_at_observed, strict=True)
        )
        endpoint_times = [item[0] for item in ordered]
        endpoint_observed = [item[1] for item in ordered]
        endpoint_predicted = [item[2] for item in ordered]
        observed_auc = _auc(endpoint_times, endpoint_observed)
        predicted_auc = _auc(endpoint_times, endpoint_predicted)
        observed_cmax = max(endpoint_observed)
        predicted_cmax = max(endpoint_predicted)
        if observed_auc > 0 and observed_cmax > 0:
            auc_ratio = predicted_auc / observed_auc
            cmax_ratio = predicted_cmax / observed_cmax
            endpoint_passed = (
                criteria.endpoint_ratio_min
                <= auc_ratio
                <= criteria.endpoint_ratio_max
                and criteria.endpoint_ratio_min
                <= cmax_ratio
                <= criteria.endpoint_ratio_max
            )
        else:
            endpoint_evaluable = False
    return {
        "simulation_sequence": simulation.sequence,
        "simulation_name": simulation.name,
        "dataset_name": observed.dataset_name,
        "source_location": observed.source_location,
        "reference": observed.reference,
        "prediction_csv": prediction_path,
        "prediction_unit": predicted_unit,
        "observation_count": len(observed_times),
        "positive_pair_count": len(positive_pairs),
        "positive_pair_fraction": positive_pair_fraction,
        "excluded_nonpositive_observation_count": (
            len(observed_times) - len(positive_observation_pairs)
        ),
        "excluded_nonpositive_prediction_count": (
            len(positive_observation_pairs) - len(positive_pairs)
        ),
        **metrics,
        "dataset_guard_passed": dataset_guard_passed,
        "endpoint_evaluable": endpoint_evaluable,
        "auc0_last_ratio": auc_ratio,
        "cmax_ratio": cmax_ratio,
        "endpoint_passed": endpoint_passed,
        "positive_observed_mol_per_litre": [item[1] for item in positive_pairs],
        "positive_predicted_mol_per_litre": [item[2] for item in positive_pairs],
    }


def _run_by_id(manifest: SnapshotWindowsManifest) -> dict[str, SnapshotExecutionRun]:
    runs = {run.run_id: run for run in manifest.runs}
    if set(runs) != {"run_1", "run_2"} or len(manifest.runs) != 2:
        raise SnapshotReuseError("Windows manifest must contain run_1 and run_2 once.")
    return runs


def _simulations_by_sequence(
    run: SnapshotExecutionRun,
) -> dict[int, SimulationExecution]:
    by_sequence = {item.sequence: item for item in run.simulations}
    if len(by_sequence) != len(run.simulations):
        raise SnapshotReuseError(f"{run.run_id} contains duplicate simulation sequences.")
    return by_sequence


def _artifact_key(artifact: ExecutionArtifact) -> tuple[str, str]:
    return (artifact.extension.casefold(), artifact.simulation_relative_path.casefold())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(
            row,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    temporary.replace(path)
    return sha256_file(path)


def audit_snapshot_reuse_execution(
    *,
    protocol_path: Path,
    windows_result_root: Path,
    output_directory: Path,
) -> dict[str, Any]:
    protocol = LicensedSnapshotReuseProtocol.model_validate_json(
        protocol_path.read_text(encoding="utf-8")
    )
    protocol_sha256 = sha256_file(protocol_path)
    windows_result_root = windows_result_root.resolve()
    manifest_path = windows_result_root / "windows_run_manifest.json"
    manifest = SnapshotWindowsManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8-sig")
    )
    manifest_sha256 = sha256_file(manifest_path)
    runs = _run_by_id(manifest)
    expected_by_sequence = {item.sequence: item for item in protocol.simulations}
    if len(expected_by_sequence) != len(protocol.simulations):
        raise SnapshotReuseError("Protocol contains duplicate simulation sequences.")

    input_snapshot_path = _safe_artifact_path(
        windows_result_root, manifest.input_snapshot_relative_path
    )
    roundtrip_snapshot_path = _safe_artifact_path(
        windows_result_root, manifest.roundtrip_snapshot_relative_path
    )
    reconstructed_project_path = _safe_artifact_path(
        windows_result_root, manifest.reconstructed_project_relative_path
    )
    worker_script_path = _safe_artifact_path(
        windows_result_root, manifest.worker_script_relative_path
    )
    input_snapshot = _read_json(input_snapshot_path)
    roundtrip_snapshot = _read_json(roundtrip_snapshot_path)
    input_semantic_sha256 = snapshot_semantic_sha256(input_snapshot)
    roundtrip_semantic_sha256 = snapshot_semantic_sha256(roundtrip_snapshot)
    roundtrip_comparison = compare_roundtrip_snapshots(
        input_snapshot=input_snapshot,
        roundtrip_snapshot=roundtrip_snapshot,
        expected_input_version=protocol.source.snapshot_schema_version,
    )
    engineering_checks: dict[str, bool] = {
        "protocol_identity_matches": (
            manifest.protocol_id == protocol.protocol_id
            and manifest.protocol_sha256 == protocol_sha256
        ),
        "worker_script_checksum_locked": (
            sha256_file(worker_script_path) == manifest.worker_script_sha256
        ),
        "windows_manifest_passed": manifest.status == "passed",
        "runtime_version_locked": (
            f"PKSim.CLI {protocol.runtime.pksim_version}"
            in manifest.cli_version_output
        ),
        "runtime_archive_checksum_locked": (
            manifest.runtime_archive_sha256
            == protocol.runtime.pksim_portable_sha256
        ),
        "input_snapshot_checksum_locked": (
            manifest.input_snapshot_sha256 == protocol.source.snapshot_sha256
            and sha256_file(input_snapshot_path) == protocol.source.snapshot_sha256
        ),
        "project_checksum_locked": (
            sha256_file(reconstructed_project_path)
            == manifest.reconstructed_project_sha256
        ),
        "roundtrip_checksum_locked": (
            sha256_file(roundtrip_snapshot_path)
            == manifest.roundtrip_snapshot_sha256
        ),
        "input_snapshot_semantics_locked": (
            input_semantic_sha256 == protocol.source_snapshot_semantic_sha256
        ),
        "roundtrip_semantics_match": (
            roundtrip_comparison["accepted"]
        ),
        "two_runs_present": set(runs) == {"run_1", "run_2"},
        "simulation_identity_complete": True,
        "all_simulations_passed": True,
        "required_export_coverage_complete": True,
        "all_artifacts_verified": True,
        "csv_semantic_inventory_matches": True,
        "csv_repeatability_passed": True,
    }
    verified_artifact_count = 0
    csv_comparisons: list[dict[str, Any]] = []
    run_simulations = {
        run_id: _simulations_by_sequence(run) for run_id, run in runs.items()
    }
    for run_id, run in runs.items():
        actual = run_simulations[run_id]
        if set(actual) != set(expected_by_sequence):
            engineering_checks["simulation_identity_complete"] = False
        for sequence in sorted(set(actual) & set(expected_by_sequence)):
            expected = expected_by_sequence[sequence]
            result = actual[sequence]
            if result.name != expected.name:
                engineering_checks["simulation_identity_complete"] = False
            if result.status != "passed" or result.return_code != 0:
                engineering_checks["all_simulations_passed"] = False
            extensions = {item.extension.casefold() for item in result.artifacts}
            if not set(protocol.runtime.required_export_extensions) <= extensions:
                engineering_checks["required_export_coverage_complete"] = False
            for artifact in result.artifacts:
                try:
                    _verify_artifact(windows_result_root, artifact)
                    verified_artifact_count += 1
                except SnapshotReuseError:
                    engineering_checks["all_artifacts_verified"] = False
    for sequence, expected in expected_by_sequence.items():
        result_a = run_simulations["run_1"].get(sequence)
        result_b = run_simulations["run_2"].get(sequence)
        if result_a is None or result_b is None:
            engineering_checks["csv_semantic_inventory_matches"] = False
            engineering_checks["csv_repeatability_passed"] = False
            continue
        csv_a = {
            _artifact_key(item): item
            for item in result_a.artifacts
            if item.extension.casefold() == ".csv"
        }
        csv_b = {
            _artifact_key(item): item
            for item in result_b.artifacts
            if item.extension.casefold() == ".csv"
        }
        if set(csv_a) != set(csv_b) or len(csv_a) == 0:
            engineering_checks["csv_semantic_inventory_matches"] = False
            engineering_checks["csv_repeatability_passed"] = False
        for key in sorted(set(csv_a) & set(csv_b)):
            path_a = _verify_artifact(windows_result_root, csv_a[key])
            path_b = _verify_artifact(windows_result_root, csv_b[key])
            comparison = compare_csv_files(
                run_a_path=path_a,
                run_b_path=path_b,
                simulation_id=expected.name,
                csv_ordinal=1,
                absolute_tolerance=protocol.criteria.absolute_tolerance,
                relative_tolerance=protocol.criteria.relative_tolerance,
            )
            if not comparison.accepted:
                engineering_checks["csv_repeatability_passed"] = False
            csv_comparisons.append(
                {
                    "simulation_sequence": sequence,
                    "simulation_name": expected.name,
                    "artifact_key": list(key),
                    "accepted": comparison.accepted,
                    "row_count": comparison.row_count,
                    "metadata_row_count": comparison.metadata_row_count,
                    "column_count": comparison.column_count,
                    "numeric_cell_count": comparison.numeric_cell_count,
                    "non_numeric_cell_count": comparison.non_numeric_cell_count,
                    "time_axis_cell_count": comparison.time_axis_cell_count,
                    "maximum_absolute_difference": (
                        comparison.maximum_absolute_difference
                    ),
                    "maximum_relative_difference": (
                        comparison.maximum_relative_difference
                    ),
                    "mismatch_count": len(comparison.mismatches),
                    "mismatch_examples": comparison.mismatches[:10],
                }
            )

    engineering_accepted = all(engineering_checks.values())
    dataset_reports: list[dict[str, Any]] = []
    dataset_errors: list[dict[str, Any]] = []
    verification_simulations = [
        item for item in protocol.simulations if item.role == "model_verification"
    ]
    quantitative_verification_series_count = sum(
        len(simulation.all_evaluation_series())
        for simulation in verification_simulations
    )
    if engineering_accepted:
        for simulation in verification_simulations:
            execution = run_simulations["run_1"][simulation.sequence]
            for observed in simulation.all_evaluation_series():
                try:
                    dataset_reports.append(
                        _dataset_metrics(
                            result_root=windows_result_root,
                            artifacts=execution.artifacts,
                            simulation=simulation,
                            observed=observed,
                            criteria=protocol.criteria,
                        )
                    )
                except SnapshotReuseError as exc:
                    dataset_errors.append(
                        {
                            "simulation_sequence": simulation.sequence,
                            "simulation_name": simulation.name,
                            "dataset_name": observed.dataset_name,
                            "source_location": observed.source_location,
                            "reason_code": "VERIFICATION_DATASET_EVALUATION_FAILED",
                            "detail": str(exc),
                        }
                    )

    pooled_observed = [
        value
        for report in dataset_reports
        for value in report["positive_observed_mol_per_litre"]
    ]
    pooled_predicted = [
        value
        for report in dataset_reports
        for value in report["positive_predicted_mol_per_litre"]
    ]
    pooled_metrics = (
        _geometric_metrics(pooled_observed, pooled_predicted)
        if pooled_observed
        else None
    )
    evaluable_endpoints = [
        report for report in dataset_reports if report["endpoint_evaluable"]
    ]
    endpoint_pass_fraction = (
        sum(1 for report in evaluable_endpoints if report["endpoint_passed"])
        / len(evaluable_endpoints)
        if evaluable_endpoints
        else 0.0
    )
    scientific_checks = {
        "verification_dataset_coverage_complete": (
            quantitative_verification_series_count > 0
            and len(dataset_reports) == quantitative_verification_series_count
            and not dataset_errors
        ),
        "pooled_gmfe_passed": (
            pooled_metrics is not None
            and pooled_metrics["gmfe"] <= protocol.criteria.pooled_gmfe_max
        ),
        "pooled_bias_passed": (
            pooled_metrics is not None
            and protocol.criteria.pooled_bias_min
            <= pooled_metrics["geometric_bias"]
            <= protocol.criteria.pooled_bias_max
        ),
        "pooled_two_fold_fraction_passed": (
            pooled_metrics is not None
            and pooled_metrics["two_fold_fraction"]
            >= protocol.criteria.pooled_two_fold_fraction_min
        ),
        "all_dataset_guards_passed": (
            quantitative_verification_series_count > 0
            and len(dataset_reports) == quantitative_verification_series_count
            and all(report["dataset_guard_passed"] for report in dataset_reports)
        ),
        "endpoint_coverage_complete": (
            len(evaluable_endpoints) == quantitative_verification_series_count
        ),
        "endpoint_scenario_fraction_passed": (
            len(evaluable_endpoints) == quantitative_verification_series_count
            and endpoint_pass_fraction
            >= protocol.criteria.endpoint_scenario_fraction_min
        ),
    }
    scientific_accepted = engineering_accepted and all(scientific_checks.values())
    if scientific_accepted:
        validation_grade = "external_study_validated"
        run_label_grade = 3
        ranking_label = "reuse_positive"
        reusable_candidate = True
    elif engineering_accepted:
        validation_grade = "execution_verified"
        run_label_grade = 1
        ranking_label = "abstain_validation_failed"
        reusable_candidate = False
    else:
        validation_grade = "execution_failed"
        run_label_grade = 0
        ranking_label = "abstain_execution_failed"
        reusable_candidate = False

    output_directory.mkdir(parents=True, exist_ok=False)
    datasets_path = output_directory / "verification_dataset_metrics.jsonl"
    datasets_sha256 = _write_jsonl(datasets_path, dataset_reports)
    comparisons_path = output_directory / "repeatability_comparisons.jsonl"
    comparisons_sha256 = _write_jsonl(comparisons_path, csv_comparisons)
    report = {
        "audit_schema_version": "1.0.0",
        "auditor_version": AUDITOR_VERSION,
        "audited_at": datetime.now(UTC).isoformat(),
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol_sha256,
        "windows_manifest_path": str(manifest_path),
        "windows_manifest_sha256": manifest_sha256,
        "engineering_accepted": engineering_accepted,
        "scientific_accepted": scientific_accepted,
        "engineering_checks": engineering_checks,
        "scientific_checks": scientific_checks,
        "counts": {
            "expected_simulations": protocol.counts.simulations,
            "verification_simulations": protocol.counts.model_verification,
            "quantitative_verification_series": quantitative_verification_series_count,
            "verification_simulations_without_quantitative_series": sum(
                not simulation.all_evaluation_series()
                for simulation in verification_simulations
            ),
            "verified_artifacts": verified_artifact_count,
            "csv_comparisons": len(csv_comparisons),
            "evaluated_datasets": len(dataset_reports),
            "dataset_errors": len(dataset_errors),
            "pooled_positive_observation_points": len(pooled_observed),
            "evaluable_endpoint_scenarios": len(evaluable_endpoints),
        },
        "roundtrip": {
            "input_semantic_sha256": input_semantic_sha256,
            "roundtrip_semantic_sha256": roundtrip_semantic_sha256,
            **roundtrip_comparison,
        },
        "pooled_metrics": pooled_metrics,
        "endpoint_scenario_pass_fraction": endpoint_pass_fraction,
        "dataset_errors": dataset_errors,
        "verification_dataset_metrics_path": str(datasets_path),
        "verification_dataset_metrics_sha256": datasets_sha256,
        "repeatability_comparisons_path": str(comparisons_path),
        "repeatability_comparisons_sha256": comparisons_sha256,
        "validation_grade": validation_grade,
        "run_label_grade": run_label_grade,
        "ranking_label": ranking_label,
        "reusable_candidate": reusable_candidate,
    }
    report_path = output_directory / "snapshot_reuse_audit.json"
    report_sha256 = write_json_atomic(report_path, report, canonical=True)
    report["report_path"] = str(report_path)
    report["report_sha256"] = report_sha256
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and audit checksum-locked licensed PBPK snapshot reuse runs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--repository", type=Path, required=True)
    prepare.add_argument("--source-archive", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--protocol", type=Path, required=True)
    audit.add_argument("--windows-result-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        protocol = build_snapshot_reuse_protocol(
            config_path=args.config,
            repository_root=args.repository,
            source_archive=args.source_archive,
            output_directory=args.output,
        )
        print(json.dumps(protocol.model_dump(mode="json"), indent=2, sort_keys=True))
        return
    report = audit_snapshot_reuse_execution(
        protocol_path=args.protocol,
        windows_result_root=args.windows_result_root,
        output_directory=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["scientific_accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
