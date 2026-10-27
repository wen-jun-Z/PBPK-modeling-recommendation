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
    CaffeineV2ParameterManifest,
)
from pbpk_kb.blueprints.caffeine_v2_snapshot import EXPECTED_SIMULATIONS
from pbpk_kb.blueprints.caffeine_v3_parameters import (
    AbsorptionSetProtocol,
    CaffeineV3ParameterManifest,
    CaffeineV3ParameterRecord,
    ClearanceStart,
    PermeabilityCandidate,
)
from pbpk_kb.blueprints.caffeine_v3_snapshot import CaffeineV3ScenarioManifest
from pbpk_kb.blueprints.caffeine_v3_sources import CaffeineV3SourceManifest
from pbpk_kb.blueprints.schemas import StrictBlueprintModel
from pbpk_kb.model_assets.execution_plan import (
    canonical_sha256,
    write_json_atomic,
)
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_runtime.caffeine_calibration import (
    RuntimeInputArtifact,
    RuntimeMetric,
    RuntimeScenario,
    _load_jsonl,
    _observations,
    _relative,
    _resolve_pkml,
)
from pbpk_kb.model_runtime.caffeine_v2_calibration import (
    _artifact_inventory_checks,
    _metric_reproduction_checks,
)
from pbpk_kb.model_runtime.caffeine_v3_runtime import CaffeineV3RuntimeAudit


PROTOCOL_VERSION = "caffeine-fit-validation-protocol-v3-1.0.0"
EXPECTED_OSPSUITE_VERSION = "12.4.3"
TARGET_OUTPUT = (
    "Organism|PeripheralVenousBlood|Caffeine|"
    "Plasma (Peripheral Venous Blood)"
)
CALIBRATION_SCENARIOS = ("CAL_BLANCHARD_IV", "CAL_BLANCHARD_PO")
VALIDATION_SCENARIOS = tuple(
    name for name in EXPECTED_SIMULATIONS if name not in CALIBRATION_SCENARIOS
)


class CaffeineV3IdentificationError(RuntimeError):
    pass


class CaffeineV3IdentificationThresholds(StrictBlueprintModel):
    minimum_converged_starts: Literal[5] = 5
    top_run_count: Literal[5] = 5
    top_objective_relative_tolerance: Literal[0.0001] = 0.0001
    top_parameter_normalized_range_max: Literal[0.001] = 0.001
    boundary_margin_fraction: Literal[0.01] = 0.01
    repeatability_relative_tolerance: Literal[1e-10] = 1e-10
    profile_grid_points: Literal[31] = 31
    profile_confidence_level: Literal[0.95] = 0.95
    disposition_observation_count: Literal[17] = 17
    disposition_parameter_count: Literal[1] = 1
    gmfe_max: Literal[2.0] = 2.0
    geometric_bias_min: Literal[0.5] = 0.5
    geometric_bias_max: Literal[2.0] = 2.0
    minimum_two_fold_fraction: Literal[0.8] = 0.8
    zero_baseline_max_umol_per_l: Literal[1e-9] = 1e-9
    endpoint_fold_error_min: Literal[0.5] = 0.5
    endpoint_fold_error_max: Literal[2.0] = 2.0
    tmax_absolute_error_h_max: Literal[1.0] = 1.0
    external_coverage_min: Literal[0.8] = 0.8
    prediction_set_fold_span_max: Literal[1.10] = 1.10


class CaffeineV3UncertaintyProtocol(StrictBlueprintModel):
    sample_count: Literal[512] = 512
    seed: Literal[20260803] = 20260803
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
    clearance_sampling: Literal[
        "truncated_normal_from_iv_hessian"
    ] = "truncated_normal_from_iv_hessian"
    permeability_sampling: Literal[
        "stratified_log_uniform_over_admissible_numerical_set"
    ] = "stratified_log_uniform_over_admissible_numerical_set"
    permeability_upper_bound_semantics: Literal[
        "right_censored_at_numerical_guard"
    ] = "right_censored_at_numerical_guard"


class CaffeineV3LockedValidationPackage(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    protocol_version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    generated_at: datetime
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_scenario_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_scenario_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_scenario_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_scenario_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenarios: list[RuntimeScenario] = Field(min_length=5, max_length=5)
    secondary_metrics: list[RuntimeMetric] = Field(min_length=5, max_length=5)
    checks: dict[str, bool]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validation_only(self) -> "CaffeineV3LockedValidationPackage":
        if tuple(item.name for item in self.scenarios) != VALIDATION_SCENARIOS:
            raise ValueError("Locked validation scenario order drifted.")
        if any(item.split == "calibration" for item in self.scenarios):
            raise ValueError("Calibration data entered the validation package.")
        return self


class CaffeineV3IdentificationRunConfig(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    protocol_version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    generated_at: datetime
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameter_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameter_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_scenario_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_scenario_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_scenario_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_scenario_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_audit_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    uncertainty_parameter_manifest_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    uncertainty_parameter_manifest_checksum: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    locked_validation_package_path: str = Field(min_length=1)
    locked_validation_package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    locked_validation_package_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    ospsuite_r_version: Literal[EXPECTED_OSPSUITE_VERSION] = (
        EXPECTED_OSPSUITE_VERSION
    )
    target_output: Literal[TARGET_OUTPUT] = TARGET_OUTPUT
    runtime_root: str = Field(min_length=1)
    output_directory: str = Field(min_length=1)
    parameters: list[CaffeineV3ParameterRecord] = Field(
        min_length=11,
        max_length=11,
    )
    calibration_scenarios: list[RuntimeScenario] = Field(
        min_length=2,
        max_length=2,
    )
    clearance_starts: list[ClearanceStart] = Field(min_length=9, max_length=9)
    permeability_candidates: list[PermeabilityCandidate] = Field(
        min_length=71,
        max_length=71,
    )
    absorption_set_protocol: AbsorptionSetProtocol
    thresholds: CaffeineV3IdentificationThresholds = Field(
        default_factory=CaffeineV3IdentificationThresholds
    )
    uncertainty: CaffeineV3UncertaintyProtocol
    checks: dict[str, bool]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def split_and_identification_contract(
        self,
    ) -> "CaffeineV3IdentificationRunConfig":
        if tuple(item.name for item in self.calibration_scenarios) != (
            CALIBRATION_SCENARIOS
        ):
            raise ValueError("Identification config scenario order drifted.")
        if any(item.split != "calibration" for item in self.calibration_scenarios):
            raise ValueError("Non-calibration data entered identification config.")
        point = [
            item.parameter_id
            for item in self.parameters
            if item.identification_mode == "point_estimated"
        ]
        set_identified = [
            item.parameter_id
            for item in self.parameters
            if item.identification_mode == "set_identified"
        ]
        if point != ["total_cyp1a2_specific_clearance"]:
            raise ValueError("Clearance must be the only point estimate.")
        if set_identified != ["specific_intestinal_permeability_transcellular"]:
            raise ValueError("Permeability must be the only set estimate.")
        return self


class CaffeineV3ScientificAudit(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    auditor_version: Literal["caffeine-v3-scientific-auditor-1.0.0"] = (
        "caffeine-v3-scientific-auditor-1.0.0"
    )
    generated_at: datetime
    phase: Literal["identification", "validation"]
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_checks: dict[str, bool]
    protocol_checks: dict[str, bool]
    metric_checks: dict[str, bool]
    scientific_status: str = Field(min_length=1)
    parameter_identifiability: bool
    prediction_identifiability: bool
    registration_allowed: bool
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


def _checksum_valid(model: StrictBlueprintModel) -> bool:
    return canonical_sha256(
        model.model_dump(mode="json", exclude={"checksum"})
    ) == getattr(model, "checksum")


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


def _uncertainty_protocol(
    parent: CaffeineParameterManifest,
) -> CaffeineV3UncertaintyProtocol:
    records = {item.parameter_id: item for item in parent.parameters}
    fraction_unbound = records["caffeine_fraction_unbound_plasma"]
    renal = records["caffeine_renal_plasma_clearance"]
    if fraction_unbound.uncertainty_type != "sd" or (
        fraction_unbound.uncertainty is None
    ):
        raise CaffeineV3IdentificationError(
            "Fraction-unbound source SD is missing."
        )
    renal_values = [
        item.normalized_clearance_ml_per_min_per_kg
        for item in renal.derivation_inputs
    ]
    return CaffeineV3UncertaintyProtocol(
        fraction_unbound_mean=fraction_unbound.value,
        fraction_unbound_sd=fraction_unbound.uncertainty,
        renal_clearance_bootstrap_values=renal_values,
    )


def prepare_caffeine_v3_identification(
    *,
    source_manifest_path: Path,
    dataset_directory: Path,
    parameter_directory: Path,
    parent_parameter_directory: Path,
    v3_scenario_directory: Path,
    source_scenario_directory: Path,
    windows_run_root: Path,
    runtime_audit_path: Path,
    input_directory: Path,
    output_directory: Path,
) -> CaffeineV3IdentificationRunConfig:
    if input_directory.exists():
        raise CaffeineV3IdentificationError(
            f"Immutable input directory exists: {input_directory}"
        )
    if output_directory.exists():
        raise CaffeineV3IdentificationError(
            f"Immutable output directory exists: {output_directory}"
        )
    dataset_path = dataset_directory / "dataset_split_manifest.json"
    observations_path = dataset_directory / "observations.jsonl"
    metrics_path = dataset_directory / "pk_metrics.jsonl"
    parameter_path = parameter_directory / "caffeine_parameter_manifest.json"
    parent_parameter_path = (
        parent_parameter_directory / "caffeine_parameter_manifest.json"
    )
    v3_scenario_path = v3_scenario_directory / "study_scenario_manifest.json"
    source_scenario_path = (
        source_scenario_directory / "study_scenario_manifest.json"
    )

    source = CaffeineV3SourceManifest.model_validate_json(
        source_manifest_path.read_text(encoding="utf-8")
    )
    source_parent_path = (
        source_manifest_path.parent / "caffeine_parameter_manifest_v2.json"
    )
    source_parent = CaffeineV2ParameterManifest.model_validate_json(
        source_parent_path.read_text(encoding="utf-8")
    )
    dataset = CaffeineDatasetManifest.model_validate_json(
        dataset_path.read_text(encoding="utf-8")
    )
    parameter = CaffeineV3ParameterManifest.model_validate_json(
        parameter_path.read_text(encoding="utf-8")
    )
    parent_parameter = CaffeineParameterManifest.model_validate_json(
        parent_parameter_path.read_text(encoding="utf-8")
    )
    v3_scenario = CaffeineV3ScenarioManifest.model_validate_json(
        v3_scenario_path.read_text(encoding="utf-8")
    )
    source_scenario = CaffeineStudyScenarioManifest.model_validate_json(
        source_scenario_path.read_text(encoding="utf-8")
    )
    runtime_audit = CaffeineV3RuntimeAudit.model_validate_json(
        runtime_audit_path.read_text(encoding="utf-8")
    )
    observations = _load_jsonl(observations_path, ObservationPoint)
    metrics = _load_jsonl(metrics_path, PKMetric)

    windows_run_root = windows_run_root.resolve()
    runtime_root = (windows_run_root / "exports").resolve()
    source_by_name = {item.name: item for item in source_scenario.simulations}
    runtime_scenarios = [
        _runtime_scenario(
            scenario=source_by_name[name],
            observations=observations,
            runtime_root=runtime_root,
        )
        for name in EXPECTED_SIMULATIONS
    ]
    calibration_scenarios = [
        item for item in runtime_scenarios if item.name in CALIBRATION_SCENARIOS
    ]
    validation_scenarios = [
        item for item in runtime_scenarios if item.name in VALIDATION_SCENARIOS
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
        "v3_scenario_manifest_sha256": sha256_file(v3_scenario_path),
        "v3_scenario_manifest_checksum": v3_scenario.checksum,
        "source_scenario_manifest_sha256": sha256_file(source_scenario_path),
        "source_scenario_manifest_checksum": source_scenario.checksum,
        "scenarios": validation_scenarios,
        "secondary_metrics": secondary_metrics,
        "checks": package_checks,
        "accepted": all(package_checks.values()),
    }
    package_draft = CaffeineV3LockedValidationPackage(
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
        failed = [name for name, passed in package_checks.items() if not passed]
        raise CaffeineV3IdentificationError(
            "Validation package checks failed: " + ", ".join(failed)
        )

    temporary = input_directory.with_name(input_directory.name + ".tmp")
    if temporary.exists():
        raise CaffeineV3IdentificationError(f"Stale input exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        package_path = temporary / "locked_validation_package.json"
        write_json_atomic(package_path, package, canonical=True)
        final_package_path = input_directory / package_path.name
        parameter_by_id = {
            item.parameter_id: item for item in parameter.parameters
        }
        iv = next(
            item for item in calibration_scenarios if item.name == "CAL_BLANCHARD_IV"
        )
        po = next(
            item for item in calibration_scenarios if item.name == "CAL_BLANCHARD_PO"
        )
        checks = {
            "source_manifest_accepted": source.accepted
            and all(source.checks.values()),
            "source_manifest_checksum_valid": _checksum_valid(source),
            "dataset_manifest_accepted": dataset.accepted
            and all(dataset.checks.values()),
            "dataset_manifest_checksum_valid": _checksum_valid(dataset),
            "parameter_manifest_accepted": parameter.accepted
            and all(parameter.checks.values()),
            "parameter_manifest_checksum_valid": _checksum_valid(parameter),
            "parent_parameter_manifest_checksum_valid": _checksum_valid(
                parent_parameter
            ),
            "parameter_source_lineage_matches": (
                parameter.source_manifest_checksum == source.checksum
                and parameter.source_manifest_sha256
                == sha256_file(source_manifest_path)
            ),
            "parameter_parent_lineage_matches": (
                parameter.parent_parameter_manifest_checksum
                == source.parent_parameter_manifest_checksum
            ),
            "source_parent_manifest_checksum_valid": _checksum_valid(
                source_parent
            ),
            "source_parent_artifact_lineage_matches": (
                source.parent_parameter_manifest_sha256
                == sha256_file(source_parent_path)
                and source.parent_parameter_manifest_checksum
                == source_parent.checksum
            ),
            "source_parent_to_uncertainty_manifest_lineage_matches": (
                source_parent.parent_parameter_manifest_checksum
                == parent_parameter.checksum
            ),
            "parameter_dataset_lineage_matches": (
                parameter.dataset_manifest_checksum == dataset.checksum
            ),
            "v3_scenario_manifest_accepted": v3_scenario.accepted
            and all(v3_scenario.checks.values()),
            "v3_scenario_manifest_checksum_valid": _checksum_valid(v3_scenario),
            "scenario_parameter_lineage_matches": (
                v3_scenario.parameter_manifest_checksum == parameter.checksum
                and v3_scenario.parameter_manifest_sha256
                == sha256_file(parameter_path)
            ),
            "source_scenario_manifest_accepted": source_scenario.accepted
            and all(source_scenario.checks.values()),
            "source_scenario_manifest_checksum_valid": _checksum_valid(
                source_scenario
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
            == CALIBRATION_SCENARIOS,
            "seventeen_iv_observations_exact": len(iv.observations) == 17,
            "eleven_po_observations_exact": len(po.observations) == 11,
            "only_clearance_point_estimated": (
                parameter_by_id[
                    "total_cyp1a2_specific_clearance"
                ].identification_mode
                == "point_estimated"
            ),
            "only_permeability_set_identified": (
                parameter_by_id[
                    "specific_intestinal_permeability_transcellular"
                ].identification_mode
                == "set_identified"
            ),
            "nine_clearance_starts_exact": len(parameter.clearance_starts) == 9,
            "seventy_one_permeability_candidates_exact": len(
                parameter.permeability_candidates
            )
            == 71,
            "validation_package_accepted": package.accepted,
            "all_runtime_pkml_nonempty": all(
                item.pkml.byte_size > 0 for item in runtime_scenarios
            ),
        }
        config_payload = {
            "generated_at": datetime.now(UTC),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "source_manifest_checksum": source.checksum,
            "dataset_manifest_sha256": sha256_file(dataset_path),
            "dataset_manifest_checksum": dataset.checksum,
            "parameter_manifest_sha256": sha256_file(parameter_path),
            "parameter_manifest_checksum": parameter.checksum,
            "v3_scenario_manifest_sha256": sha256_file(v3_scenario_path),
            "v3_scenario_manifest_checksum": v3_scenario.checksum,
            "source_scenario_manifest_sha256": sha256_file(
                source_scenario_path
            ),
            "source_scenario_manifest_checksum": source_scenario.checksum,
            "runtime_audit_sha256": sha256_file(runtime_audit_path),
            "runtime_audit_checksum": runtime_audit.checksum,
            "uncertainty_parameter_manifest_sha256": sha256_file(
                parent_parameter_path
            ),
            "uncertainty_parameter_manifest_checksum": (
                parent_parameter.checksum
            ),
            "locked_validation_package_path": final_package_path.as_posix(),
            "locked_validation_package_sha256": sha256_file(package_path),
            "locked_validation_package_checksum": package.checksum,
            "runtime_root": runtime_root.as_posix(),
            "output_directory": output_directory.resolve().as_posix(),
            "parameters": parameter.parameters,
            "calibration_scenarios": calibration_scenarios,
            "clearance_starts": parameter.clearance_starts,
            "permeability_candidates": parameter.permeability_candidates,
            "absorption_set_protocol": parameter.absorption_set_protocol,
            "uncertainty": _uncertainty_protocol(parent_parameter),
            "checks": checks,
            "accepted": all(checks.values()),
        }
        config_draft = CaffeineV3IdentificationRunConfig(
            **config_payload,
            checksum="0" * 64,
        )
        config = config_draft.model_copy(
            update={
                "checksum": canonical_sha256(
                    config_draft.model_dump(mode="json", exclude={"checksum"})
                )
            }
        )
        if not config.accepted:
            failed = [name for name, passed in checks.items() if not passed]
            raise CaffeineV3IdentificationError(
                "Identification input checks failed: " + ", ".join(failed)
            )
        config_path = temporary / "identification_config.json"
        write_json_atomic(config_path, config, canonical=True)
        write_json_atomic(
            temporary / "input_manifest.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "identification_config_sha256": sha256_file(config_path),
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
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def execute_r_phase(
    *,
    config_path: Path,
    rscript_path: Path,
    script_directory: Path,
    phase: Literal["preflight", "identify", "validate"],
    phase_output_directory: Path | None = None,
) -> None:
    script_name = (
        "caffeine_v3_validate.R"
        if phase == "validate"
        else "caffeine_v3_identify.R"
    )
    command = [
        str(rscript_path.resolve()),
        str((script_directory / script_name).resolve()),
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


def _close(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(left_value) and math.isfinite(right_value) and math.isclose(
        left_value,
        right_value,
        rel_tol=tolerance,
        abs_tol=tolerance,
    )


def _same_number(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    if math.isinf(left_value) or math.isinf(right_value):
        return left_value == right_value
    return _close(left_value, right_value, tolerance)


def _optional_number_equal(
    left: Any,
    right: Any,
    tolerance: float = 1e-10,
) -> bool:
    left_text = "" if left is None else str(left).strip()
    right_text = "" if right is None else str(right).strip()
    if not left_text or not right_text:
        return left_text == right_text
    return _same_number(left_text, right_text, tolerance)


def _set_reproduction_checks(
    *,
    candidate_path: Path,
    comparison_path: Path,
    set_manifest_path: Path,
    config: CaffeineV3IdentificationRunConfig,
) -> dict[str, bool]:
    if not all(path.is_file() for path in (candidate_path, comparison_path, set_manifest_path)):
        return {"absorption_set_inputs_present": False}
    rows = _read_csv_rows(candidate_path)
    comparisons = _read_csv_rows(comparison_path)
    manifest = json.loads(set_manifest_path.read_text(encoding="utf-8"))
    first_run = [row for row in rows if row.get("repeat_id") == "1"]
    second_run = [row for row in rows if row.get("repeat_id") == "2"]
    protocol = config.absorption_set_protocol
    absolute = {
        int(row["candidate_id"]): (
            float(row["f_model"]) >= protocol.bioavailability_lower
            and float(row["f_model"]) <= protocol.bioavailability_upper
            and float(row["tmax_po_min"]) >= protocol.tmax_lower_min
            and float(row["tmax_po_min"]) <= protocol.tmax_upper_min
            and row.get("status") == "ok"
        )
        for row in first_run
    }
    pair_pass = {
        int(row["lower_candidate_id"]): _csv_bool(row.get("accepted"))
        for row in comparisons
        if row.get("repeat_id") == "1"
    }
    possible_starts = [
        candidate_id
        for candidate_id in range(1, 62)
        if all(absolute.get(index, False) for index in range(candidate_id, 72))
        and all(pair_pass.get(index, False) for index in range(candidate_id, 62))
    ]
    expected = list(range(min(possible_starts), 72)) if possible_starts else []
    reported = [int(value) for value in manifest.get("admissible_candidate_ids", [])]
    return {
        "absorption_set_inputs_present": True,
        "two_times_seventy_one_candidates_exact": len(rows) == 142,
        "two_times_sixty_one_comparisons_exact": len(comparisons) == 122,
        "candidate_identity_repeatable": [
            row.get("candidate_id") for row in first_run
        ]
        == [row.get("candidate_id") for row in second_run],
        "candidate_metrics_repeatable": all(
            _optional_number_equal(left.get(field), right.get(field))
            for left, right in zip(first_run, second_run)
            for field in ("auc_iv_0_24", "auc_po_0_24", "f_model", "tmax_po_min")
        )
        and all(
            left.get("status") == right.get("status")
            and left.get("error") == right.get("error")
            for left, right in zip(first_run, second_run)
        ),
        "admissible_set_reproduced": reported == expected,
        "admissible_set_contiguous": not reported
        or reported == list(range(reported[0], reported[-1] + 1)),
        "admissible_set_upper_guard_outcome_reproduced": (
            not expected or expected[-1] == 71
        ),
        "manifest_acceptance_reproduced": manifest.get("accepted")
        is bool(expected),
    }


def _set_validation_reproduction_checks(
    *,
    validation_root: Path,
    config: CaffeineV3IdentificationRunConfig,
    admissible_count: int,
) -> dict[str, bool]:
    point_path = validation_root / "set_point_predictions_and_residuals.csv"
    metric_path = validation_root / "set_validation_metrics.csv"
    secondary_path = validation_root / "set_secondary_validation_metrics.csv"
    endpoint_path = validation_root / "set_endpoint_predictions.csv"
    stability_path = validation_root / "set_endpoint_stability.csv"
    required = (
        point_path,
        metric_path,
        secondary_path,
        endpoint_path,
        stability_path,
    )
    if not all(path.is_file() for path in required):
        return {"set_validation_reproduction_inputs_present": False}
    points = _read_csv_rows(point_path)
    metrics = _read_csv_rows(metric_path)
    secondary = _read_csv_rows(secondary_path)
    endpoints = _read_csv_rows(endpoint_path)
    stability = _read_csv_rows(stability_path)
    thresholds = config.thresholds

    metric_reproduced = True
    for metric in metrics:
        selected = [
            row
            for row in points
            if row.get("repeat_id") == metric.get("repeat_id")
            and row.get("candidate_id") == metric.get("candidate_id")
            and row.get("dataset_id") == metric.get("dataset_id")
            and _csv_bool(row.get("included"))
        ]
        positive = [
            row for row in selected if float(row["observed_umol_per_l"]) > 0
        ]
        zero = [
            row for row in selected if float(row["observed_umol_per_l"]) == 0
        ]
        if not selected or not positive:
            numeric_fields = (
                "gmfe",
                "geometric_bias",
                "log_rmse",
                "two_fold_fraction",
            )
            metric_reproduced = metric_reproduced and all(
                (
                    not str(metric.get(field, "")).strip()
                    for field in numeric_fields
                )
            ) and not _csv_bool(metric.get("accepted"))
            continue
        predictions = [float(row["predicted_umol_per_l"]) for row in selected]
        ratios = [
            float(row["predicted_umol_per_l"])
            / float(row["observed_umol_per_l"])
            for row in positive
        ]
        if any(value <= 0 or not math.isfinite(value) for value in ratios):
            metric_reproduced = False
            continue
        logs = [math.log(value) for value in ratios]
        gmfe = math.exp(sum(abs(value) for value in logs) / len(logs))
        bias = math.exp(sum(logs) / len(logs))
        log_rmse = math.sqrt(sum(value**2 for value in logs) / len(logs))
        two_fold = sum(0.5 <= value <= 2 for value in ratios) / len(ratios)
        zero_pass = all(
            float(row["predicted_umol_per_l"])
            <= thresholds.zero_baseline_max_umol_per_l
            for row in zero
        )
        finite_nonnegative = all(
            math.isfinite(value) and value >= 0 for value in predictions
        )
        expected_accepted = (
            finite_nonnegative
            and zero_pass
            and gmfe <= thresholds.gmfe_max
            and thresholds.geometric_bias_min
            <= bias
            <= thresholds.geometric_bias_max
            and two_fold >= thresholds.minimum_two_fold_fraction
        )
        metric_reproduced = metric_reproduced and all(
            (
                int(metric["included_count"]) == len(selected),
                int(metric["positive_count"]) == len(positive),
                _close(metric.get("gmfe"), gmfe),
                _close(metric.get("geometric_bias"), bias),
                _close(metric.get("log_rmse"), log_rmse),
                _close(metric.get("two_fold_fraction"), two_fold),
                _csv_bool(metric.get("zero_baseline_pass")) is zero_pass,
                _csv_bool(metric.get("finite_nonnegative"))
                is finite_nonnegative,
                _csv_bool(metric.get("accepted")) is expected_accepted,
            )
        )

    secondary_reproduced = True
    for row in secondary:
        if not str(row.get("predicted_value", "")).strip():
            secondary_reproduced = secondary_reproduced and all(
                (
                    not str(row.get("fold_error", "")).strip(),
                    not str(row.get("absolute_error", "")).strip(),
                    not _csv_bool(row.get("accepted")),
                    bool(row.get("metric_id")),
                    bool(row.get("source_artifact_id")),
                    bool(row.get("source_location")),
                )
            )
            continue
        predicted = float(row["predicted_value"])
        observed = float(row["observed_value"])
        fold_error = predicted / observed
        absolute_error = abs(predicted - observed)
        expected_accepted = (
            absolute_error <= thresholds.tmax_absolute_error_h_max
            if row.get("endpoint") == "tmax"
            else thresholds.endpoint_fold_error_min
            <= fold_error
            <= thresholds.endpoint_fold_error_max
        )
        secondary_reproduced = secondary_reproduced and all(
            (
                _close(row.get("fold_error"), fold_error),
                _close(row.get("absolute_error"), absolute_error),
                _csv_bool(row.get("accepted")) is expected_accepted,
                bool(row.get("metric_id")),
                bool(row.get("source_artifact_id")),
                bool(row.get("source_location")),
            )
        )

    stability_reproduced = True
    first_endpoints = [row for row in endpoints if row.get("repeat_id") == "1"]
    for row in stability:
        selected = [
            item
            for item in first_endpoints
            if item.get("scenario_name") == row.get("scenario_name")
            and item.get("endpoint") == row.get("endpoint")
        ]
        values = [float(item["predicted_value"]) for item in selected]
        if not values:
            stability_reproduced = False
            continue
        minimum = min(values)
        maximum = max(values)
        span = maximum / minimum if minimum > 0 else math.inf
        expected_accepted = (
            len(values) == admissible_count
            and all(math.isfinite(value) for value in values)
            and span <= thresholds.prediction_set_fold_span_max
        )
        stability_reproduced = stability_reproduced and all(
            (
                int(row["candidate_count"]) == len(values),
                _close(row.get("minimum"), minimum),
                _close(row.get("maximum"), maximum),
                _same_number(row.get("max_min_fold_span"), span),
                _csv_bool(row.get("accepted")) is expected_accepted,
            )
        )

    return {
        "set_validation_reproduction_inputs_present": True,
        "set_point_provenance_complete": bool(points)
        and all(
            row.get("source_artifact_id") and row.get("source_location")
            for row in points
        ),
        "all_set_concentration_metrics_reproduced": metric_reproduced,
        "all_set_secondary_metrics_reproduced": secondary_reproduced,
        "all_set_endpoint_spans_reproduced": stability_reproduced,
        "set_metric_row_count_exact": len(metrics) == admissible_count * 5 * 2,
        "set_secondary_row_count_exact": len(secondary)
        == admissible_count * 5 * 2,
    }


def audit_caffeine_v3_run(
    *,
    config_path: Path,
    run_root: Path,
    output_path: Path,
) -> CaffeineV3ScientificAudit:
    if output_path.exists():
        raise CaffeineV3IdentificationError(
            f"Immutable audit output exists: {output_path}"
        )
    config = CaffeineV3IdentificationRunConfig.model_validate_json(
        config_path.read_text(encoding="utf-8")
    )
    run_root = run_root.resolve()
    identification_root = run_root / "01_identification"
    identification_manifest_path = (
        identification_root / "identification_manifest.json"
    )
    if not identification_manifest_path.is_file():
        raise CaffeineV3IdentificationError("Identification manifest is missing.")
    identification_manifest = json.loads(
        identification_manifest_path.read_text(encoding="utf-8")
    )
    identification = identification_manifest.get("identification", {})
    clearance = identification.get("clearance", {})
    absorption = identification.get("absorption_set", {})
    scientific_status = str(
        identification_manifest.get("scientific_status", "run_incomplete")
    )
    artifact_checks = {
        "config_checksum_reproducible": _checksum_valid(config),
        "configured_run_root_matches": Path(config.output_directory).resolve()
        == run_root,
        "identification_config_sha256_matches": identification_manifest.get(
            "config_sha256"
        )
        == sha256_file(config_path),
    }
    artifact_checks.update(
        {
            f"identification_{key}": value
            for key, value in _artifact_inventory_checks(
                root=identification_root,
                inventory=identification_manifest.get("artifacts"),
                excluded_names={"identification_manifest.json"},
            ).items()
        }
    )
    start_path = identification_root / "clearance_multi_start_results.csv"
    profile_path = identification_root / "clearance_profile_likelihood.csv"
    start_rows = _read_csv_rows(start_path) if start_path.is_file() else []
    profile_rows = _read_csv_rows(profile_path) if profile_path.is_file() else []
    identification_completed = identification_manifest.get("run_completed") is True
    protocol_checks = {
        "identification_phase_declared": identification_manifest.get("phase")
        == "identify",
        "two_calibration_scenarios_only": identification_manifest.get(
            "loaded_scenarios"
        )
        == list(CALIBRATION_SCENARIOS),
        "validation_package_remained_locked": (
            identification_manifest.get("validation_package_loaded") is False
            and identification_manifest.get("validation_scenarios_loaded") is False
            and identification_manifest.get("validation") is None
        ),
        "identification_never_registers": identification_manifest.get(
            "registration_allowed"
        )
        is False,
        "nine_clearance_starts_recorded": len(start_rows) == 9,
        "clearance_profile_presence_matches_completion": (
            len(profile_rows) == 31
            if identification_completed
            else not profile_path.exists()
        ),
        "clearance_acceptance_matches_checks": clearance.get("accepted")
        is (
            bool(clearance.get("checks"))
            and all(clearance.get("checks", {}).values())
        ),
        "absorption_acceptance_matches_checks": absorption.get("accepted")
        is (
            bool(absorption.get("checks"))
            and all(absorption.get("checks", {}).values())
        ),
        "identification_acceptance_is_conjunction": identification.get(
            "accepted"
        )
        is (clearance.get("accepted") is True and absorption.get("accepted") is True),
    }
    clearance_point_path = identification_root / "clearance_point_predictions.csv"
    clearance_metric_path = identification_root / "clearance_metrics.csv"
    if clearance_point_path.is_file() and clearance_metric_path.is_file():
        metric_checks = _metric_reproduction_checks(
            point_path=clearance_point_path,
            metric_path=clearance_metric_path,
            thresholds=config.thresholds,
        )
    else:
        metric_checks = {
            "clearance_metric_absence_matches_runtime_failure": (
                scientific_status == "calibration_runtime_failed"
                and not clearance_point_path.exists()
                and not clearance_metric_path.exists()
            )
        }
    set_paths = (
        identification_root / "absorption_candidates.csv",
        identification_root / "absorption_plateau_comparisons.csv",
        identification_root / "absorption_set_manifest.json",
    )
    if all(path.is_file() for path in set_paths):
        metric_checks.update(
            _set_reproduction_checks(
                candidate_path=set_paths[0],
                comparison_path=set_paths[1],
                set_manifest_path=set_paths[2],
                config=config,
            )
        )
    else:
        metric_checks["absorption_outputs_absent_only_after_clearance_failure"] = (
            clearance.get("accepted") is False
            and not any(path.exists() for path in set_paths)
        )

    phase: Literal["identification", "validation"] = "identification"
    result_manifest_path = identification_manifest_path
    parameter_identifiability = False
    prediction_identifiability = False
    registration_allowed = False
    final_manifest_path = run_root / "run_manifest.json"
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
        set_manifest = json.loads(
            (identification_root / "absorption_set_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        admissible_count = len(set_manifest.get("admissible_candidate_ids", []))
        validation_metrics = _read_csv_rows(
            validation_root / "set_validation_metrics.csv"
        )
        secondary_metrics = _read_csv_rows(
            validation_root / "set_secondary_validation_metrics.csv"
        )
        stability_rows = _read_csv_rows(
            validation_root / "set_endpoint_stability.csv"
        )
        status_rows = _read_csv_rows(
            validation_root / "set_candidate_execution_status.csv"
        )
        repeatability = json.loads(
            (validation_root / "deterministic_repeatability.json").read_text(
                encoding="utf-8"
            )
        )
        uncertainty_sample_path = (
            validation_root / "uncertainty_parameter_samples.csv"
        )
        uncertainty_endpoint_path = (
            validation_root / "uncertainty_endpoint_samples.csv"
        )
        has_uncertainty_outputs = (
            uncertainty_sample_path.is_file()
            and uncertainty_endpoint_path.is_file()
        )
        uncertainty_samples = (
            _read_csv_rows(uncertainty_sample_path)
            if has_uncertainty_outputs
            else []
        )
        uncertainty_endpoints = (
            _read_csv_rows(uncertainty_endpoint_path)
            if has_uncertainty_outputs
            else []
        )
        if has_uncertainty_outputs:
            coverage_rows = _read_csv_rows(
                validation_root / "uncertainty_observation_coverage.csv"
            )
            prcc_rows = _read_csv_rows(
                validation_root / "uncertainty_prcc.csv"
            )
            progress = json.loads(
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
                sum(
                    _csv_bool(row.get("covered_90_percent"))
                    for row in external_coverage_rows
                )
                / len(external_coverage_rows)
                if external_coverage_rows
                else math.nan
            )
            allowed_ids = [
                int(value)
                for value in set_manifest.get("admissible_candidate_ids", [])
            ]
            permeability_values = {
                item.candidate_id: item.value_dm_per_min
                for item in config.permeability_candidates
            }
            allowed_permeability = [
                permeability_values[value] for value in allowed_ids
            ]
            clearance_parameter = next(
                item
                for item in config.parameters
                if item.parameter_id == "total_cyp1a2_specific_clearance"
            )
            if (
                clearance_parameter.lower_bound is None
                or clearance_parameter.upper_bound is None
            ):
                raise CaffeineV3IdentificationError(
                    "Clearance bounds disappeared during scientific audit."
                )
            sample_numbers_valid = all(
                math.isfinite(float(row[field]))
                for row in uncertainty_samples
                for field in (
                    "total_cyp1a2_specific_clearance",
                    "specific_intestinal_permeability_transcellular",
                    "caffeine_fraction_unbound_plasma",
                    "caffeine_renal_plasma_clearance",
                )
            )
            sample_ranges_valid = sample_numbers_valid and all(
                clearance_parameter.lower_bound
                <= float(row["total_cyp1a2_specific_clearance"])
                <= clearance_parameter.upper_bound
                and min(allowed_permeability)
                <= float(
                    row["specific_intestinal_permeability_transcellular"]
                )
                <= max(allowed_permeability)
                and 0 < float(row["caffeine_fraction_unbound_plasma"]) < 1
                and float(row["caffeine_renal_plasma_clearance"]) > 0
                for row in uncertainty_samples
            )
            independent_uncertainty_accepted = (
                len(uncertainty_samples) == config.uncertainty.sample_count
                and len(uncertainty_endpoints)
                == len(EXPECTED_SIMULATIONS) * config.uncertainty.sample_count
                and sample_ranges_valid
                and math.isfinite(reproduced_coverage)
                and reproduced_coverage >= config.thresholds.external_coverage_min
                and len(prcc_rows) == len(EXPECTED_SIMULATIONS) * 3 * 4
                and all(
                    row.get("status") == "ok"
                    and math.isfinite(float(row["prcc"]))
                    for row in prcc_rows
                )
                and len(list((validation_root / "checkpoints").glob("*.rds")))
                == len(EXPECTED_SIMULATIONS)
                and progress.get("status") == "completed"
                and progress.get("completed_scenarios")
                == list(EXPECTED_SIMULATIONS)
            )
        else:
            reproduced_coverage = math.nan
            sample_ranges_valid = False
            independent_uncertainty_accepted = False
        scientific_status = str(
            final_manifest.get("scientific_status", "run_incomplete")
        )
        parameter_identifiability = final_manifest.get(
            "parameter_identifiability"
        ) is True
        prediction_identifiability = final_manifest.get(
            "prediction_identifiability"
        ) is True
        registration_allowed = final_manifest.get("registration_allowed") is True
        deterministic_reproduced = (
            len(status_rows) == admissible_count * 2
            and all(row.get("status") == "ok" for row in status_rows)
            and len(validation_metrics) == admissible_count * 5 * 2
            and all(
                _csv_bool(row.get("accepted")) for row in validation_metrics
            )
            and len(secondary_metrics) == admissible_count * 5 * 2
            and all(
                _csv_bool(row.get("accepted")) for row in secondary_metrics
            )
            and bool(stability_rows)
            and all(_csv_bool(row.get("accepted")) for row in stability_rows)
            and repeatability.get("accepted") is True
        )
        reported_deterministic = validation_manifest.get(
            "deterministic", {}
        ).get("accepted")
        reported_uncertainty = validation_manifest.get("uncertainty", {}).get(
            "accepted"
        )
        uncertainty_was_required = deterministic_reproduced
        uncertainty_shape_valid = (
            len(uncertainty_samples) == config.uncertainty.sample_count
            and len(uncertainty_endpoints)
            == len(EXPECTED_SIMULATIONS) * config.uncertainty.sample_count
        )
        expected_prediction_identifiability = (
            deterministic_reproduced and independent_uncertainty_accepted
        )
        protocol_checks.update(
            {
                "validation_loaded_only_after_identification": (
                    identification.get("accepted") is True
                    and validation_manifest.get("validation_package_loaded") is True
                ),
                "set_validation_row_count_exact": len(validation_metrics)
                == admissible_count * 5 * 2,
                "set_secondary_row_count_exact": len(secondary_metrics)
                == admissible_count * 5 * 2,
                "deterministic_acceptance_reproduced": (
                    reported_deterministic is deterministic_reproduced
                ),
                "uncertainty_presence_matches_deterministic_gate": (
                    has_uncertainty_outputs is uncertainty_was_required
                ),
                "uncertainty_shapes_match_when_executed": (
                    uncertainty_shape_valid if has_uncertainty_outputs else True
                ),
                "uncertainty_acceptance_independently_reproduced": (
                    reported_uncertainty is independent_uncertainty_accepted
                ),
                "uncertainty_sample_ranges_valid_when_executed": (
                    sample_ranges_valid if has_uncertainty_outputs else True
                ),
                "uncertainty_external_coverage_reproduced": (
                    _close(
                        validation_manifest.get("uncertainty", {}).get(
                            "external_observation_coverage_90"
                        ),
                        reproduced_coverage,
                    )
                    if has_uncertainty_outputs
                    else True
                ),
                "parameter_identifiability_remains_false": not parameter_identifiability,
                "prediction_identifiability_reproduced": (
                    prediction_identifiability
                    is expected_prediction_identifiability
                ),
                "registration_matches_prediction_identifiability": (
                    registration_allowed is prediction_identifiability
                ),
                "status_and_registration_consistent": (
                    (
                        scientific_status
                        == "externally_validated_with_set_identified_permeability"
                        and registration_allowed
                    )
                    or (
                        scientific_status
                        in {
                            "quarantined_prediction_not_robust",
                            "quarantined_uncertainty_failed",
                        }
                        and not registration_allowed
                    )
                ),
            }
        )
        metric_checks.update(
            {
                "set_validation_gate_reproduced": (
                    all(
                        _csv_bool(row.get("accepted"))
                        for row in validation_metrics
                    )
                    is bool(
                        validation_manifest.get("deterministic", {})
                        .get("checks", {})
                        .get(
                            "all_five_validation_curves_pass_for_every_candidate"
                        )
                    )
                ),
                "set_secondary_gate_reproduced": (
                    all(
                        _csv_bool(row.get("accepted"))
                        for row in secondary_metrics
                    )
                    is bool(
                        validation_manifest.get("deterministic", {})
                        .get("checks", {})
                        .get(
                            "all_five_secondary_endpoints_pass_for_every_candidate"
                        )
                    )
                ),
                "endpoint_stability_gate_reproduced": (
                    all(
                        _csv_bool(row.get("accepted")) for row in stability_rows
                    )
                    is bool(
                        validation_manifest.get("deterministic", {})
                        .get("checks", {})
                        .get("all_endpoint_fold_spans_at_most_1_10")
                    )
                ),
            }
        )
        metric_checks.update(
            _set_validation_reproduction_checks(
                validation_root=validation_root,
                config=config,
                admissible_count=admissible_count,
            )
        )

    accepted = (
        bool(artifact_checks)
        and all(artifact_checks.values())
        and bool(protocol_checks)
        and all(protocol_checks.values())
        and bool(metric_checks)
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
        "parameter_identifiability": parameter_identifiability,
        "prediction_identifiability": prediction_identifiability,
        "registration_allowed": registration_allowed,
        "accepted": accepted,
    }
    draft = CaffeineV3ScientificAudit(**payload, checksum="0" * 64)
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
        description="Prepare, execute, or audit caffeine v3 set identification."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-manifest", type=Path, required=True)
    prepare.add_argument("--dataset-directory", type=Path, required=True)
    prepare.add_argument("--parameter-directory", type=Path, required=True)
    prepare.add_argument("--parent-parameter-directory", type=Path, required=True)
    prepare.add_argument("--v3-scenario-directory", type=Path, required=True)
    prepare.add_argument("--source-scenario-directory", type=Path, required=True)
    prepare.add_argument("--windows-run-root", type=Path, required=True)
    prepare.add_argument("--runtime-audit", type=Path, required=True)
    prepare.add_argument("--input-directory", type=Path, required=True)
    prepare.add_argument("--output-directory", type=Path, required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--config", type=Path, required=True)
    execute.add_argument("--rscript", type=Path, required=True)
    execute.add_argument("--script-directory", type=Path, required=True)
    execute.add_argument(
        "--phase",
        choices=("preflight", "identify", "validate"),
        required=True,
    )
    execute.add_argument("--phase-output-directory", type=Path)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--run-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare_caffeine_v3_identification(
            source_manifest_path=args.source_manifest,
            dataset_directory=args.dataset_directory,
            parameter_directory=args.parameter_directory,
            parent_parameter_directory=args.parent_parameter_directory,
            v3_scenario_directory=args.v3_scenario_directory,
            source_scenario_directory=args.source_scenario_directory,
            windows_run_root=args.windows_run_root,
            runtime_audit_path=args.runtime_audit,
            input_directory=args.input_directory,
            output_directory=args.output_directory,
        )
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False))
    elif args.command == "execute":
        execute_r_phase(
            config_path=args.config,
            rscript_path=args.rscript,
            script_directory=args.script_directory,
            phase=args.phase,
            phase_output_directory=args.phase_output_directory,
        )
    else:
        result = audit_caffeine_v3_run(
            config_path=args.config,
            run_root=args.run_root,
            output_path=args.output,
        )
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False))
        if not result.accepted:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
