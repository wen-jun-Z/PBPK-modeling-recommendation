from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from pbpk_kb.blueprints.caffeine_dataset import (
    CaffeineDatasetManifest,
    ObservationPoint,
    PKMetric,
)
from pbpk_kb.blueprints.caffeine_parameters import CaffeineParameterManifest
from pbpk_kb.blueprints.caffeine_study_snapshot import (
    CaffeineStudyScenarioManifest,
)
from pbpk_kb.blueprints.caffeine_v2_parameters import (
    CALIBRATION_DATASETS,
    CaffeineV2ParameterManifest,
    CaffeineV2ParameterRecord,
)
from pbpk_kb.blueprints.caffeine_v2_snapshot import (
    EXPECTED_SIMULATIONS,
    CaffeineV2ScenarioManifest,
)
from pbpk_kb.blueprints.schemas import StrictBlueprintModel
from pbpk_kb.model_assets.execution_plan import (
    canonical_sha256,
    write_json_atomic,
)
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_runtime.caffeine_calibration import (
    RuntimeInputArtifact,
    RuntimeMetric,
    RuntimeObservation,
    RuntimeScenario,
    _load_jsonl,
    _observations,
    _relative,
    _resolve_pkml,
    concentration_metrics,
)
from pbpk_kb.model_runtime.caffeine_v2_runtime import CaffeineV2RuntimeAudit


PROTOCOL_VERSION = "caffeine-fit-validation-protocol-v2-1.0.0"
EXPECTED_OSPSUITE_VERSION = "12.4.3"
TARGET_OUTPUT = (
    "Organism|PeripheralVenousBlood|Caffeine|"
    "Plasma (Peripheral Venous Blood)"
)
VALIDATION_SCENARIOS = tuple(
    name
    for name in EXPECTED_SIMULATIONS
    if name not in {"CAL_BLANCHARD_IV", "CAL_BLANCHARD_PO"}
)
PROFILE_F_CRITICAL_95_DF1_1_DF2_26 = 4.2252012731274853


class CaffeineV2CalibrationError(RuntimeError):
    pass


class CaffeineV2CalibrationThresholds(StrictBlueprintModel):
    minimum_converged_starts: Literal[5] = 5
    top_run_count: Literal[5] = 5
    top_objective_relative_tolerance: Literal[0.0001] = 0.0001
    top_parameter_relative_tolerance: Literal[0.001] = 0.001
    boundary_margin_fraction: Literal[0.01] = 0.01
    hessian_condition_number_max: Literal[100000000.0] = 100000000.0
    repeatability_relative_tolerance: Literal[1e-10] = 1e-10
    profile_grid_points: Literal[21] = 21
    profile_confidence_level: Literal[0.95] = 0.95
    gmfe_max: Literal[2.0] = 2.0
    geometric_bias_min: Literal[0.5] = 0.5
    geometric_bias_max: Literal[2.0] = 2.0
    minimum_two_fold_fraction: Literal[0.8] = 0.8
    zero_baseline_max_umol_per_l: Literal[1e-9] = 1e-9
    endpoint_fold_error_min: Literal[0.5] = 0.5
    endpoint_fold_error_max: Literal[2.0] = 2.0
    tmax_absolute_error_h_max: Literal[1.0] = 1.0
    external_coverage_min: Literal[0.8] = 0.8


class CaffeineV2UncertaintyProtocol(StrictBlueprintModel):
    sample_count: Literal[512] = 512
    seed: Literal[20260801] = 20260801
    batch_size: Literal[32] = 32
    quantiles: list[float] = Field(default_factory=lambda: [0.05, 0.5, 0.95])
    pmc572_auc_end_h: Literal[24.0] = 24.0
    pmc572_terminal_times_h: list[float] = Field(
        default_factory=lambda: [8.0, 10.0, 12.0, 16.0, 24.0]
    )
    caffeine_molecular_weight_g_per_mol: Literal[194.19] = 194.19
    fraction_unbound_mean: float = Field(gt=0, lt=1)
    fraction_unbound_sd: float = Field(gt=0)
    renal_clearance_bootstrap_values: list[float] = Field(
        min_length=6,
        max_length=6,
    )


class CaffeineV2LockedValidationPackage(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    protocol_version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    generated_at: datetime
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    v2_scenario_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v2_scenario_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_scenario_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_scenario_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenarios: list[RuntimeScenario] = Field(min_length=5, max_length=5)
    secondary_metrics: list[RuntimeMetric] = Field(min_length=5, max_length=5)
    checks: dict[str, bool]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validation_only(self) -> "CaffeineV2LockedValidationPackage":
        if tuple(item.name for item in self.scenarios) != VALIDATION_SCENARIOS:
            raise ValueError("Locked validation scenario order drifted.")
        if any(item.split == "calibration" for item in self.scenarios):
            raise ValueError("Calibration scenarios cannot enter validation package.")
        return self


class CaffeineV2CalibrationRunConfig(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    protocol_version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    generated_at: datetime
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameter_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameter_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    v2_scenario_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v2_scenario_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_scenario_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_scenario_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_audit_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_parameter_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_parameter_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    locked_validation_package_path: str = Field(min_length=1)
    locked_validation_package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    locked_validation_package_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    ospsuite_r_version: Literal[EXPECTED_OSPSUITE_VERSION] = (
        EXPECTED_OSPSUITE_VERSION
    )
    target_output: Literal[TARGET_OUTPUT] = TARGET_OUTPUT
    runtime_root: str = Field(min_length=1)
    output_directory: str = Field(min_length=1)
    parameters: list[CaffeineV2ParameterRecord] = Field(
        min_length=11,
        max_length=11,
    )
    calibration_scenarios: list[RuntimeScenario] = Field(
        min_length=2,
        max_length=2,
    )
    start_grid: list[list[float]] = Field(min_length=25, max_length=25)
    thresholds: CaffeineV2CalibrationThresholds = Field(
        default_factory=CaffeineV2CalibrationThresholds
    )
    uncertainty: CaffeineV2UncertaintyProtocol
    checks: dict[str, bool]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def calibration_only(self) -> "CaffeineV2CalibrationRunConfig":
        if tuple(item.name for item in self.calibration_scenarios) != (
            "CAL_BLANCHARD_IV",
            "CAL_BLANCHARD_PO",
        ):
            raise ValueError("Calibration config must contain only Blanchard IV/PO.")
        if any(item.split != "calibration" for item in self.calibration_scenarios):
            raise ValueError("Non-calibration split entered calibration config.")
        return self


class CaffeineV2ScientificAudit(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    auditor_version: Literal["caffeine-v2-scientific-auditor-1.0.0"] = (
        "caffeine-v2-scientific-auditor-1.0.0"
    )
    generated_at: datetime
    phase: Literal["calibration", "validation"]
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_checks: dict[str, bool]
    protocol_checks: dict[str, bool]
    metric_checks: dict[str, bool]
    scientific_status: str = Field(min_length=1)
    scientific_gate_passed: bool
    registration_allowed: bool
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


def _checksum_valid(model: StrictBlueprintModel) -> bool:
    return canonical_sha256(
        model.model_dump(mode="json", exclude={"checksum"})
    ) == getattr(model, "checksum")


def _secondary_metrics(rows: list[PKMetric]) -> list[RuntimeMetric]:
    return [
        RuntimeMetric(
            metric_id=item.metric_id,
            dataset_id=item.dataset_id,
            endpoint=item.endpoint,
            observed_value=item.value,
            unit=item.unit,
            source_artifact_id=item.provenance.artifact_id,
            source_location=item.provenance.source_location,
        )
        for item in sorted(rows, key=lambda value: value.endpoint)
    ]


def _runtime_scenario(
    *,
    scenario: Any,
    observations: list[ObservationPoint],
    runtime_root: Path,
) -> RuntimeScenario:
    pkml = _resolve_pkml(runtime_root, scenario.name)
    return RuntimeScenario(
        name=scenario.name,
        dataset_id=scenario.dataset_id,
        split=scenario.split,
        aggregation_weight=scenario.aggregation_weight,
        pkml=RuntimeInputArtifact(
            relative_path=_relative(pkml, runtime_root),
            sha256=sha256_file(pkml),
            byte_size=pkml.stat().st_size,
        ),
        observations=_observations(observations, scenario.dataset_id),
    )


def _uncertainty_protocol(
    parent: CaffeineParameterManifest,
) -> CaffeineV2UncertaintyProtocol:
    records = {item.parameter_id: item for item in parent.parameters}
    fraction_unbound = records["caffeine_fraction_unbound_plasma"]
    renal = records["caffeine_renal_plasma_clearance"]
    renal_values = [
        item.normalized_clearance_ml_per_min_per_kg
        for item in renal.derivation_inputs
    ]
    if fraction_unbound.uncertainty_type != "sd":
        raise CaffeineV2CalibrationError(
            "Fraction-unbound uncertainty is not a source SD."
        )
    if fraction_unbound.uncertainty is None:
        raise CaffeineV2CalibrationError("Fraction-unbound SD is missing.")
    return CaffeineV2UncertaintyProtocol(
        fraction_unbound_mean=fraction_unbound.value,
        fraction_unbound_sd=fraction_unbound.uncertainty,
        renal_clearance_bootstrap_values=renal_values,
    )


def prepare_caffeine_v2_calibration(
    *,
    dataset_directory: Path,
    parameter_directory: Path,
    source_parameter_directory: Path,
    v2_scenario_directory: Path,
    source_scenario_directory: Path,
    windows_run_root: Path,
    runtime_audit_path: Path,
    input_directory: Path,
    output_directory: Path,
) -> CaffeineV2CalibrationRunConfig:
    if input_directory.exists():
        raise CaffeineV2CalibrationError(
            f"Immutable input directory exists: {input_directory}"
        )
    if output_directory.exists():
        raise CaffeineV2CalibrationError(
            f"Immutable output directory exists: {output_directory}"
        )
    dataset_path = dataset_directory / "dataset_split_manifest.json"
    observations_path = dataset_directory / "observations.jsonl"
    metrics_path = dataset_directory / "pk_metrics.jsonl"
    parameter_path = parameter_directory / "caffeine_parameter_manifest.json"
    parent_parameter_path = (
        source_parameter_directory / "caffeine_parameter_manifest_v1.json"
    )
    v2_scenario_path = v2_scenario_directory / "study_scenario_manifest.json"
    source_scenario_path = (
        source_scenario_directory / "study_scenario_manifest.json"
    )
    dataset = CaffeineDatasetManifest.model_validate_json(
        dataset_path.read_text(encoding="utf-8")
    )
    parameter = CaffeineV2ParameterManifest.model_validate_json(
        parameter_path.read_text(encoding="utf-8")
    )
    parent_parameter = CaffeineParameterManifest.model_validate_json(
        parent_parameter_path.read_text(encoding="utf-8")
    )
    v2_scenario = CaffeineV2ScenarioManifest.model_validate_json(
        v2_scenario_path.read_text(encoding="utf-8")
    )
    source_scenario = CaffeineStudyScenarioManifest.model_validate_json(
        source_scenario_path.read_text(encoding="utf-8")
    )
    runtime_audit = CaffeineV2RuntimeAudit.model_validate_json(
        runtime_audit_path.read_text(encoding="utf-8")
    )
    observations = _load_jsonl(observations_path, ObservationPoint)
    metrics = _load_jsonl(metrics_path, PKMetric)
    windows_run_root = windows_run_root.resolve()
    runtime_root = (windows_run_root / "exports").resolve()
    source_by_name = {item.name: item for item in source_scenario.simulations}
    ordered = [source_by_name[name] for name in EXPECTED_SIMULATIONS]
    runtime_scenarios = [
        _runtime_scenario(
            scenario=item,
            observations=observations,
            runtime_root=runtime_root,
        )
        for item in ordered
    ]
    calibration_scenarios = [
        item for item in runtime_scenarios if item.split == "calibration"
    ]
    validation_scenarios = [
        item for item in runtime_scenarios if item.split != "calibration"
    ]
    secondary_metrics = _secondary_metrics(metrics)
    package_checks = {
        "five_validation_scenarios_exact": tuple(
            item.name for item in validation_scenarios
        )
        == VALIDATION_SCENARIOS,
        "twenty_two_validation_observations_exact": sum(
            len(item.observations) for item in validation_scenarios
        )
        == 22,
        "five_secondary_metrics_exact": len(secondary_metrics) == 5,
        "no_calibration_split": all(
            item.split != "calibration" for item in validation_scenarios
        ),
    }
    package_payload = {
        "generated_at": datetime.now(UTC),
        "dataset_manifest_sha256": sha256_file(dataset_path),
        "dataset_manifest_checksum": dataset.checksum,
        "v2_scenario_manifest_sha256": sha256_file(v2_scenario_path),
        "v2_scenario_manifest_checksum": v2_scenario.checksum,
        "source_scenario_manifest_sha256": sha256_file(source_scenario_path),
        "source_scenario_manifest_checksum": source_scenario.checksum,
        "scenarios": validation_scenarios,
        "secondary_metrics": secondary_metrics,
        "checks": package_checks,
        "accepted": all(package_checks.values()),
    }
    package_draft = CaffeineV2LockedValidationPackage(
        **package_payload,
        checksum="0" * 64,
    )
    package = package_draft.model_copy(
        update={
            "checksum": canonical_sha256(
                package_draft.model_dump(mode="json", exclude={"checksum"})
            )
        }
    )
    if not package.accepted:
        failed = [key for key, value in package_checks.items() if not value]
        raise CaffeineV2CalibrationError(
            "Validation package checks failed: " + ", ".join(failed)
        )
    temporary = input_directory.with_name(input_directory.name + ".tmp")
    if temporary.exists():
        raise CaffeineV2CalibrationError(f"Stale input exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        package_path = temporary / "locked_validation_package.json"
        write_json_atomic(package_path, package, canonical=True)
        final_package_path = input_directory / package_path.name
        estimated = [item for item in parameter.parameters if item.calibrate]
        start_grid = [
            [
                item.total_cyp1a2_specific_clearance,
                item.specific_intestinal_permeability_transcellular,
            ]
            for item in parameter.calibration_starts
        ]
        checks = {
            "dataset_manifest_accepted": dataset.accepted
            and all(dataset.checks.values()),
            "dataset_manifest_checksum_valid": _checksum_valid(dataset),
            "parameter_manifest_accepted": parameter.accepted
            and all(parameter.checks.values()),
            "parameter_manifest_checksum_valid": _checksum_valid(parameter),
            "parent_parameter_manifest_checksum_valid": _checksum_valid(
                parent_parameter
            ),
            "parent_parameter_checksum_matches_v2": (
                parent_parameter.checksum
                == parameter.parent_parameter_manifest_checksum
            ),
            "v2_scenario_manifest_accepted": v2_scenario.accepted
            and all(v2_scenario.checks.values()),
            "v2_scenario_manifest_checksum_valid": _checksum_valid(v2_scenario),
            "source_scenario_manifest_accepted": source_scenario.accepted
            and all(source_scenario.checks.values()),
            "source_scenario_manifest_checksum_valid": _checksum_valid(
                source_scenario
            ),
            "source_scenario_checksum_matches_v2": (
                source_scenario.checksum
                == v2_scenario.source_scenario_manifest_checksum
            ),
            "runtime_audit_accepted": runtime_audit.accepted,
            "runtime_audit_checksum_valid": _checksum_valid(runtime_audit),
            "runtime_audit_root_matches": Path(
                runtime_audit.windows_run_root
            ).resolve()
            == windows_run_root,
            "two_calibration_scenarios_only": tuple(
                item.name for item in calibration_scenarios
            )
            == ("CAL_BLANCHARD_IV", "CAL_BLANCHARD_PO"),
            "twenty_eight_calibration_observations_exact": sum(
                len(item.observations) for item in calibration_scenarios
            )
            == 28,
            "only_two_external_parameters_estimated": [
                item.parameter_id for item in estimated
            ]
            == [
                "total_cyp1a2_specific_clearance",
                "specific_intestinal_permeability_transcellular",
            ],
            "twenty_five_start_grid_exact": len(start_grid) == 25
            and len({tuple(item) for item in start_grid}) == 25,
            "validation_package_accepted": package.accepted,
            "all_pkml_nonempty": all(
                item.pkml.byte_size > 0 for item in runtime_scenarios
            ),
        }
        config_payload = {
            "generated_at": datetime.now(UTC),
            "dataset_manifest_sha256": sha256_file(dataset_path),
            "dataset_manifest_checksum": dataset.checksum,
            "parameter_manifest_sha256": sha256_file(parameter_path),
            "parameter_manifest_checksum": parameter.checksum,
            "v2_scenario_manifest_sha256": sha256_file(v2_scenario_path),
            "v2_scenario_manifest_checksum": v2_scenario.checksum,
            "source_scenario_manifest_sha256": sha256_file(
                source_scenario_path
            ),
            "source_scenario_manifest_checksum": source_scenario.checksum,
            "runtime_audit_sha256": sha256_file(runtime_audit_path),
            "runtime_audit_checksum": runtime_audit.checksum,
            "parent_parameter_manifest_sha256": sha256_file(
                parent_parameter_path
            ),
            "parent_parameter_manifest_checksum": parent_parameter.checksum,
            "locked_validation_package_path": final_package_path.as_posix(),
            "locked_validation_package_sha256": sha256_file(package_path),
            "locked_validation_package_checksum": package.checksum,
            "runtime_root": runtime_root.as_posix(),
            "output_directory": output_directory.resolve().as_posix(),
            "parameters": parameter.parameters,
            "calibration_scenarios": calibration_scenarios,
            "start_grid": start_grid,
            "uncertainty": _uncertainty_protocol(parent_parameter),
            "checks": checks,
            "accepted": all(checks.values()),
        }
        config_draft = CaffeineV2CalibrationRunConfig(
            **config_payload,
            checksum="0" * 64,
        )
        config = config_draft.model_copy(
            update={
                "checksum": canonical_sha256(
                    config_draft.model_dump(
                        mode="json",
                        exclude={"checksum"},
                    )
                )
            }
        )
        if not config.accepted:
            failed = [key for key, value in checks.items() if not value]
            raise CaffeineV2CalibrationError(
                "Calibration input checks failed: " + ", ".join(failed)
            )
        write_json_atomic(
            temporary / "calibration_config.json",
            config,
            canonical=True,
        )
        write_json_atomic(
            temporary / "input_manifest.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "calibration_config_sha256": sha256_file(
                    temporary / "calibration_config.json"
                ),
                "locked_validation_package_sha256": sha256_file(package_path),
                "validation_package_loaded": False,
                "accepted": True,
            },
            canonical=True,
        )
        temporary.replace(input_directory)
        return config
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def execute_r_phase(
    *,
    config_path: Path,
    rscript_path: Path,
    script_path: Path,
    phase: Literal["preflight", "calibrate", "validate"],
    phase_output_directory: Path | None = None,
) -> None:
    if phase == "validate" and script_path.name == "caffeine_v2_fit_validate.R":
        script_path = script_path.with_name("caffeine_v2_validate.R")
    command = [
        str(rscript_path.resolve()),
        str(script_path.resolve()),
        str(config_path.resolve()),
        phase,
    ]
    if phase_output_directory is not None:
        command.append(str(phase_output_directory.resolve()))
    subprocess.run(command, check=True)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_bool(value: Any) -> bool:
    return str(value).strip().casefold() == "true"


def _close(left: Any, right: Any, *, tolerance: float = 1e-11) -> bool:
    try:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=tolerance,
            abs_tol=tolerance,
        )
    except (TypeError, ValueError):
        return False


def _optional_close(
    reported: Any,
    reproduced: float | None,
    *,
    tolerance: float = 1e-11,
) -> bool:
    if reproduced is None:
        return reported is None
    return _close(reported, reproduced, tolerance=tolerance)


def _artifact_inventory_checks(
    *,
    root: Path,
    inventory: Any,
    excluded_names: set[str],
) -> dict[str, bool]:
    rows = inventory if isinstance(inventory, list) else []
    expected_paths = [str(item.get("relative_path", "")) for item in rows]
    actual_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in excluded_names
    )
    checks = {
        "artifact_inventory_nonempty": bool(rows),
        "artifact_inventory_paths_unique": (
            len(expected_paths) == len(set(expected_paths))
        ),
        "artifact_inventory_exact_file_set": sorted(expected_paths)
        == actual_paths,
    }
    integrity = True
    for item in rows:
        relative = str(item.get("relative_path", ""))
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            integrity = False
            continue
        if (
            not candidate.is_file()
            or candidate.stat().st_size != item.get("byte_size")
            or sha256_file(candidate) != item.get("sha256")
        ):
            integrity = False
    checks["artifact_inventory_sha256_and_size_verified"] = integrity
    return checks


def _metric_reproduction_checks(
    *,
    point_path: Path,
    metric_path: Path,
    thresholds: CaffeineV2CalibrationThresholds,
) -> dict[str, bool]:
    if not point_path.is_file() or not metric_path.is_file():
        return {"metric_inputs_present": False}
    point_rows = _read_csv_rows(point_path)
    metric_rows = _read_csv_rows(metric_path)
    reported = {row["dataset_id"]: row for row in metric_rows}
    dataset_ids = sorted({row["dataset_id"] for row in point_rows})
    checks: dict[str, bool] = {
        "metric_inputs_present": True,
        "one_metric_row_per_dataset": set(reported) == set(dataset_ids)
        and len(reported) == len(metric_rows),
    }
    for dataset_id in dataset_ids:
        rows = [row for row in point_rows if row["dataset_id"] == dataset_id]
        try:
            recalculated = concentration_metrics(
                rows,
                zero_baseline_max=thresholds.zero_baseline_max_umol_per_l,
            )
        except Exception:
            checks[f"{dataset_id}_metrics_recalculated"] = False
            continue
        row = reported.get(dataset_id)
        checks[f"{dataset_id}_metrics_recalculated"] = row is not None
        if row is None:
            continue
        for field in (
            "gmfe",
            "geometric_bias",
            "log_rmse",
            "two_fold_fraction",
        ):
            checks[f"{dataset_id}_{field}_reproduced"] = _close(
                row.get(field),
                recalculated[field],
            )
        checks[f"{dataset_id}_included_count_reproduced"] = (
            int(row["included_count"]) == recalculated["included_count"]
        )
        if "positive_count" in row:
            checks[f"{dataset_id}_positive_count_reproduced"] = (
                int(row["positive_count"]) == recalculated["positive_count"]
            )
        checks[f"{dataset_id}_zero_baseline_reproduced"] = _csv_bool(
            row["zero_baseline_pass"]
        ) is recalculated["zero_baseline_pass"]
        expected_accepted = (
            float(recalculated["gmfe"]) <= thresholds.gmfe_max
            and thresholds.geometric_bias_min
            <= float(recalculated["geometric_bias"])
            <= thresholds.geometric_bias_max
            and float(recalculated["two_fold_fraction"])
            >= thresholds.minimum_two_fold_fraction
            and bool(recalculated["zero_baseline_pass"])
        )
        checks[f"{dataset_id}_acceptance_reproduced"] = _csv_bool(
            row["accepted"]
        ) is expected_accepted
        checks[f"{dataset_id}_source_provenance_present"] = all(
            row_item.get("source_artifact_id", "").strip()
            and row_item.get("source_location", "").strip()
            for row_item in rows
        )
    return checks


def _profile_crossing(
    rows: list[dict[str, str]],
    *,
    best_z: float,
    side: Literal["left", "right"],
    threshold: float,
) -> float | None:
    selected = [
        row
        for row in rows
        if (
            float(row["fixed_z"]) <= best_z + 1e-12
            if side == "left"
            else float(row["fixed_z"]) >= best_z - 1e-12
        )
    ]
    selected.sort(key=lambda row: float(row["fixed_z"]), reverse=side == "left")
    for first, second in zip(selected, selected[1:]):
        first_z = float(first["fixed_z"])
        second_z = float(second["fixed_z"])
        first_objective = float(first["objective"])
        second_objective = float(second["objective"])
        first_delta = first_objective - threshold
        second_delta = second_objective - threshold
        if first_delta == 0:
            return first_z
        if first_delta * second_delta <= 0:
            if second_objective == first_objective:
                return (first_z + second_z) / 2
            return first_z + (threshold - first_objective) * (
                second_z - first_z
            ) / (second_objective - first_objective)
    return None


def _profile_reproduction_checks(
    *,
    profile_path: Path,
    summary_path: Path,
    best_objective: float,
) -> dict[str, bool]:
    if not profile_path.is_file() or not summary_path.is_file():
        return {"profile_inputs_present": False}
    rows = _read_csv_rows(profile_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    parameter_summaries = summary.get("parameters", [])
    expected_threshold = best_objective * (
        1 + PROFILE_F_CRITICAL_95_DF1_1_DF2_26 / 26
    )
    checks = {
        "profile_inputs_present": True,
        "profile_threshold_reproduced": _close(
            summary.get("threshold"), expected_threshold
        ),
        "all_profile_row_thresholds_reproduced": all(
            _close(row.get("threshold"), expected_threshold) for row in rows
        ),
        "two_profile_parameters_exact": len(parameter_summaries) == 2,
        "forty_two_profile_rows_exact": len(rows) == 42,
    }
    for parameter in parameter_summaries:
        parameter_id = str(parameter.get("parameter_id"))
        selected = [row for row in rows if row.get("parameter_id") == parameter_id]
        best_z = float(parameter["best_z"])
        left = _profile_crossing(
            selected,
            best_z=best_z,
            side="left",
            threshold=expected_threshold,
        )
        right = _profile_crossing(
            selected,
            best_z=best_z,
            side="right",
            threshold=expected_threshold,
        )
        checks[f"{parameter_id}_21_grid_points"] = len(selected) == 21
        checks[f"{parameter_id}_convergence_codes_audited"] = all(
            row.get("convergence_code", "").lstrip("-").isdigit()
            for row in selected
        )
        checks[f"{parameter_id}_left_crossing_reproduced"] = (
            _optional_close(parameter.get("left_crossing_z"), left)
        )
        checks[f"{parameter_id}_right_crossing_reproduced"] = (
            _optional_close(parameter.get("right_crossing_z"), right)
        )
        expected_accepted = (
            left is not None
            and right is not None
            and 0 < left < best_z < right < 1
            and all(math.isfinite(float(row["objective"])) for row in selected)
            and all(int(row["convergence_code"]) == 0 for row in selected)
        )
        checks[f"{parameter_id}_acceptance_reproduced"] = (
            parameter.get("accepted") is expected_accepted
        )
    return checks


def audit_caffeine_v2_run(
    *,
    config_path: Path,
    run_root: Path,
    output_path: Path,
) -> CaffeineV2ScientificAudit:
    if output_path.exists():
        raise CaffeineV2CalibrationError(
            f"Immutable audit output exists: {output_path}"
        )
    config = CaffeineV2CalibrationRunConfig.model_validate_json(
        config_path.read_text(encoding="utf-8")
    )
    run_root = run_root.resolve()
    calibration_root = run_root / "01_calibration"
    calibration_manifest_path = calibration_root / "calibration_manifest.json"
    if not calibration_manifest_path.is_file():
        raise CaffeineV2CalibrationError("Calibration manifest is missing.")
    calibration_manifest = json.loads(
        calibration_manifest_path.read_text(encoding="utf-8")
    )
    calibration = calibration_manifest.get("calibration", {})
    calibration_checks = calibration.get("checks", {})
    artifact_checks = {
        "config_checksum_reproducible": _checksum_valid(config),
        "configured_run_root_matches": Path(config.output_directory).resolve()
        == run_root,
        "calibration_config_sha256_matches": (
            calibration_manifest.get("config_sha256")
            == sha256_file(config_path)
        ),
    }
    artifact_checks.update(
        {
            f"calibration_{key}": value
            for key, value in _artifact_inventory_checks(
                root=calibration_root,
                inventory=calibration_manifest.get("artifacts"),
                excluded_names={"calibration_manifest.json"},
            ).items()
        }
    )
    start_path = calibration_root / "calibration_multi_start_results.csv"
    evaluation_path = calibration_root / "calibration_evaluations.csv"
    start_rows = _read_csv_rows(start_path) if start_path.is_file() else []
    evaluation_rows = (
        _read_csv_rows(evaluation_path) if evaluation_path.is_file() else []
    )
    calibration_accepted = calibration.get("accepted") is True
    calibration_status = str(
        calibration_manifest.get("scientific_status", "run_incomplete")
    )
    expected_calibration_status = (
        "calibration_accepted_validation_locked"
        if calibration_accepted
        else {
            "parameter_unidentifiable",
            "calibration_failed",
            "calibration_runtime_failed",
        }
    )
    status_consistent = (
        calibration_status == expected_calibration_status
        if isinstance(expected_calibration_status, str)
        else calibration_status in expected_calibration_status
    )
    protocol_checks = {
        "calibration_phase_declared": calibration_manifest.get("phase")
        == "calibrate",
        "two_calibration_scenarios_only": calibration_manifest.get(
            "loaded_scenarios"
        )
        == ["CAL_BLANCHARD_IV", "CAL_BLANCHARD_PO"],
        "validation_package_remained_locked": (
            calibration_manifest.get("validation_package_loaded") is False
            and calibration_manifest.get("validation_scenarios_loaded") is False
            and calibration_manifest.get("validation") is None
        ),
        "calibration_never_registers": calibration_manifest.get(
            "registration_allowed"
        )
        is False,
        "twenty_five_starts_declared": calibration.get("start_count") == 25,
        "twenty_five_starts_recorded": len(start_rows) == 25,
        "evaluation_log_nonempty": bool(evaluation_rows),
        "manifest_successful_evaluation_count_reproduced": calibration.get(
            "successful_evaluation_count"
        )
        == sum(
            row.get("status") == "ok" and row.get("phase") == "optimization"
            for row in evaluation_rows
        ),
        "calibration_status_consistent": status_consistent,
        "calibration_check_set_present": bool(calibration_checks),
        "calibration_acceptance_matches_checks": calibration_accepted
        is (bool(calibration_checks) and all(calibration_checks.values())),
    }
    metric_checks = _metric_reproduction_checks(
        point_path=calibration_root / "calibration_point_predictions.csv",
        metric_path=calibration_root / "calibration_metrics.csv",
        thresholds=config.thresholds,
    )
    if calibration_manifest.get("run_completed") is True:
        profile_rows = _read_csv_rows(
            calibration_root / "calibration_profile_likelihood.csv"
        )
        profile_summary = json.loads(
            (calibration_root / "calibration_profile_summary.json").read_text(
                encoding="utf-8"
            )
        )
        metric_checks.update(
            _profile_reproduction_checks(
                profile_path=calibration_root
                / "calibration_profile_likelihood.csv",
                summary_path=calibration_root
                / "calibration_profile_summary.json",
                best_objective=float(calibration["best_objective"]),
            )
        )
        best_rows = _read_csv_rows(calibration_root / "best_fit_parameters.csv")
        protocol_checks.update(
            {
                "calibration_run_completed": True,
                "two_best_fit_parameters_exact": [
                    row.get("parameter_id") for row in best_rows
                ]
                == [
                    "total_cyp1a2_specific_clearance",
                    "specific_intestinal_permeability_transcellular",
                ],
                "two_calibration_metric_rows_exact": len(
                    _read_csv_rows(calibration_root / "calibration_metrics.csv")
                )
                == 2,
                "forty_two_profile_rows_exact": len(profile_rows) == 42,
                "two_profile_summaries_exact": len(
                    profile_summary.get("parameters", [])
                )
                == 2,
                "profile_acceptance_matches_manifest": bool(
                    calibration_checks.get(
                        "both_profile_likelihood_intervals_finite"
                    )
                )
                is all(
                    item.get("accepted") is True
                    for item in profile_summary.get("parameters", [])
                ),
                "covariance_present_only_after_accepted_calibration": (
                    calibration_root
                    / "calibration_covariance_original_units.csv"
                ).is_file()
                is calibration_accepted,
            }
        )
    else:
        protocol_checks["runtime_failure_status_declared"] = (
            calibration_status == "calibration_runtime_failed"
        )

    final_manifest_path = run_root / "run_manifest.json"
    phase: Literal["calibration", "validation"] = "calibration"
    result_manifest_path = calibration_manifest_path
    scientific_status = calibration_status
    scientific_gate_passed = calibration_accepted
    registration_allowed = False
    if final_manifest_path.is_file():
        phase = "validation"
        result_manifest_path = final_manifest_path
        final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
        validation_root = run_root / "02_validation_uncertainty"
        validation_manifest_path = validation_root / "validation_manifest.json"
        validation_manifest = json.loads(
            validation_manifest_path.read_text(encoding="utf-8")
        )
        artifact_checks.update(
            {
                f"final_{key}": value
                for key, value in _artifact_inventory_checks(
                    root=run_root,
                    inventory=final_manifest.get("artifacts"),
                    excluded_names={"run_manifest.json"},
                ).items()
            }
        )
        artifact_checks.update(
            {
                f"validation_{key}": value
                for key, value in _artifact_inventory_checks(
                    root=validation_root,
                    inventory=validation_manifest.get("artifacts"),
                    excluded_names={"validation_manifest.json"},
                ).items()
            }
        )
        validation_points = _read_csv_rows(
            validation_root / "point_predictions_and_residuals.csv"
        )
        uncertainty_samples = _read_csv_rows(
            validation_root / "uncertainty_parameter_samples.csv"
        )
        endpoint_samples = _read_csv_rows(
            validation_root / "uncertainty_endpoint_samples.csv"
        )
        prcc_rows = _read_csv_rows(validation_root / "uncertainty_prcc.csv")
        coverage_rows = _read_csv_rows(
            validation_root / "uncertainty_observation_coverage.csv"
        )
        execution_progress = json.loads(
            (
                validation_root / "validation_execution_progress.json"
            ).read_text(encoding="utf-8")
        )
        external_coverage_rows = [
            row
            for row in coverage_rows
            if row.get("split") == "external_validation"
            and _csv_bool(row.get("included"))
        ]
        reproduced_coverage = (
            sum(_csv_bool(row.get("covered_90_percent")) for row in external_coverage_rows)
            / len(external_coverage_rows)
            if external_coverage_rows
            else math.nan
        )
        metric_checks.update(
            {
                f"validation_{key}": value
                for key, value in _metric_reproduction_checks(
                    point_path=validation_root
                    / "point_predictions_and_residuals.csv",
                    metric_path=validation_root / "validation_metrics.csv",
                    thresholds=config.thresholds,
                ).items()
            }
        )
        validation_uncertainty = validation_manifest.get("uncertainty", {})
        final_uncertainty = final_manifest.get("uncertainty", {})
        scientific_status = str(
            final_manifest.get("scientific_status", "run_incomplete")
        )
        registration_allowed = final_manifest.get("registration_allowed") is True
        scientific_gate_passed = scientific_status == "external_study_validated"
        protocol_checks.update(
            {
                "validation_loaded_only_after_accepted_calibration": (
                    calibration_accepted
                    and final_manifest.get("validation_package_loaded") is True
                    and validation_manifest.get("validation_package_loaded") is True
                ),
                "all_seven_scenarios_declared": final_manifest.get(
                    "scenario_count"
                )
                == 7,
                "fifty_point_predictions_exact": len(validation_points) == 50,
                "five_validation_metric_rows_exact": len(
                    _read_csv_rows(validation_root / "validation_metrics.csv")
                )
                == 5,
                "five_secondary_endpoint_rows_exact": len(
                    _read_csv_rows(
                        validation_root / "secondary_validation_metrics.csv"
                    )
                )
                == 5,
                "five_hundred_twelve_lhs_samples_exact": len(
                    uncertainty_samples
                )
                == config.uncertainty.sample_count,
                "seven_times_512_endpoint_rows_exact": len(endpoint_samples)
                == 7 * config.uncertainty.sample_count,
                "eighty_four_prcc_rows_exact": len(prcc_rows) == 7 * 3 * 4,
                "seven_scenario_checkpoints_exact": len(
                    list((validation_root / "checkpoints").glob("*.rds"))
                )
                == 7,
                "resumable_execution_completed_exact_scenarios": (
                    execution_progress.get("status") == "completed"
                    and execution_progress.get("config_sha256")
                    == sha256_file(config_path)
                    and execution_progress.get("completed_scenarios")
                    == [
                        "CAL_BLANCHARD_IV",
                        "CAL_BLANCHARD_PO",
                        "HOLD_WAHLLANDER_IV",
                        "EXT_EUDRACT_PO",
                        "EXT_PMC535_PO",
                        "EXT2_PMC572_PO_MALE",
                        "EXT2_PMC572_PO_FEMALE",
                    ]
                ),
                "all_prcc_values_finite": all(
                    row.get("status") == "ok"
                    and math.isfinite(float(row["prcc"]))
                    for row in prcc_rows
                ),
                "uncertainty_success_count_consistent": (
                    validation_uncertainty.get("successful_sample_count")
                    == config.uncertainty.sample_count
                    and final_uncertainty.get("successful_sample_count")
                    == config.uncertainty.sample_count
                ),
                "external_coverage_reproduced": _close(
                    validation_uncertainty.get(
                        "external_observation_coverage_90"
                    ),
                    reproduced_coverage,
                ),
                "validation_status_and_registration_consistent": (
                    registration_allowed is scientific_gate_passed
                    and scientific_status
                    in {"external_study_validated", "abstain_validation_failed"}
                ),
            }
        )

    accepted = (
        all(artifact_checks.values())
        and all(protocol_checks.values())
        and all(metric_checks.values())
    )
    payload = {
        "generated_at": datetime.now(UTC),
        "phase": phase,
        "config_sha256": sha256_file(config_path),
        "result_manifest_sha256": sha256_file(result_manifest_path),
        "artifact_checks": artifact_checks,
        "protocol_checks": protocol_checks,
        "metric_checks": metric_checks,
        "scientific_status": scientific_status,
        "scientific_gate_passed": scientific_gate_passed,
        "registration_allowed": registration_allowed,
        "accepted": accepted,
    }
    draft = CaffeineV2ScientificAudit(**payload, checksum="0" * 64)
    audit = draft.model_copy(
        update={
            "checksum": canonical_sha256(
                draft.model_dump(mode="json", exclude={"checksum"})
            )
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, audit, canonical=True)
    return audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or execute the isolated caffeine v2 fit."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--dataset-directory", type=Path, required=True)
    prepare.add_argument("--parameter-directory", type=Path, required=True)
    prepare.add_argument("--source-parameter-directory", type=Path, required=True)
    prepare.add_argument("--v2-scenario-directory", type=Path, required=True)
    prepare.add_argument("--source-scenario-directory", type=Path, required=True)
    prepare.add_argument("--windows-run-root", type=Path, required=True)
    prepare.add_argument("--runtime-audit", type=Path, required=True)
    prepare.add_argument("--input-directory", type=Path, required=True)
    prepare.add_argument("--output-directory", type=Path, required=True)
    execute = commands.add_parser("execute")
    execute.add_argument("--config", type=Path, required=True)
    execute.add_argument("--rscript", type=Path, required=True)
    execute.add_argument("--phase", choices=("preflight", "calibrate", "validate"), required=True)
    execute.add_argument("--phase-output-directory", type=Path)
    execute.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).with_name("R") / "caffeine_v2_fit_validate.R",
    )
    audit = commands.add_parser("audit")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--run-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        config = prepare_caffeine_v2_calibration(
            dataset_directory=args.dataset_directory,
            parameter_directory=args.parameter_directory,
            source_parameter_directory=args.source_parameter_directory,
            v2_scenario_directory=args.v2_scenario_directory,
            source_scenario_directory=args.source_scenario_directory,
            windows_run_root=args.windows_run_root,
            runtime_audit_path=args.runtime_audit,
            input_directory=args.input_directory,
            output_directory=args.output_directory,
        )
        print(json.dumps(config.model_dump(mode="json"), ensure_ascii=False))
        return
    if args.command == "audit":
        audit = audit_caffeine_v2_run(
            config_path=args.config,
            run_root=args.run_root,
            output_path=args.output,
        )
        print(json.dumps(audit.model_dump(mode="json"), ensure_ascii=False))
        if not audit.accepted:
            raise SystemExit(1)
        return
    execute_r_phase(
        config_path=args.config,
        rscript_path=args.rscript,
        script_path=args.script,
        phase=args.phase,
        phase_output_directory=args.phase_output_directory,
    )


if __name__ == "__main__":
    main()
