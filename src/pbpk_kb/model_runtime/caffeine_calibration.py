from __future__ import annotations

import argparse
import csv
import json
import math
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
from pbpk_kb.blueprints.caffeine_parameters import (
    CaffeineParameterManifest,
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


PROTOCOL_VERSION = "caffeine-fit-validation-protocol-1.1.0"
TARGET_OUTPUT = (
    "Organism|PeripheralVenousBlood|Caffeine|"
    "Plasma (Peripheral Venous Blood)"
)
EXPECTED_OSPSUITE_VERSION = "12.4.3"
EXPECTED_SCENARIOS = (
    "CAL_BLANCHARD_IV",
    "CAL_BLANCHARD_PO",
    "HOLD_WAHLLANDER_IV",
    "EXT_EUDRACT_PO",
    "EXT_PMC535_PO",
    "EXT2_PMC572_PO_MALE",
    "EXT2_PMC572_PO_FEMALE",
)
CALIBRATION_DATASETS = (
    "blanchard_1983_iv_5mgkg",
    "blanchard_1983_oral_5mgkg",
)

RUNTIME_PARAMETER_PATHS = {
    "caffeine_molecular_weight": "Caffeine|Molecular weight",
    "caffeine_pka_base": "Caffeine|pKa value 0",
    "caffeine_water_solubility_25c": (
        "Caffeine|Solubility at reference pH"
    ),
    "caffeine_fraction_unbound_plasma": (
        "Caffeine|Fraction unbound (plasma, reference value)"
    ),
    "caffeine_renal_plasma_clearance": (
        "Neighborhoods|Kidney_pls_Kidney_ur|Caffeine|"
        f"Renal Clearances-{RENAL_PROCESS_DATA_SOURCE}-Caffeine|"
        "Plasma clearance"
    ),
    "caffeine_cyp1a2_km_n3_demethylation": "Caffeine-CYP1A2-MM|Km",
    "caffeine_cyp1a2_vmax_effective_n3": (
        "Caffeine-CYP1A2-MM|In vitro Vmax for liver microsomes"
    ),
    "caffeine_effective_lipophilicity": "Caffeine|Lipophilicity",
}


class CaffeineCalibrationError(RuntimeError):
    pass


class RuntimeInputArtifact(StrictBlueprintModel):
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(gt=0)


class RuntimeParameter(StrictBlueprintModel):
    parameter_id: str = Field(min_length=1)
    runtime_path: str = Field(min_length=1)
    origin: Literal["observed", "derived", "estimated"]
    value: float
    unit: str = Field(min_length=1)
    uncertainty_type: Literal["sd"] | None = None
    uncertainty: float | None = Field(default=None, ge=0)
    reported_lower: float | None = None
    reported_upper: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    calibrate: bool
    bootstrap_values: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def parameter_contract(self) -> "RuntimeParameter":
        if not math.isfinite(self.value):
            raise ValueError("Runtime parameter values must be finite.")
        if self.calibrate:
            if self.origin != "estimated":
                raise ValueError("Only estimated parameters may be calibrated.")
            if self.lower_bound is None or self.upper_bound is None:
                raise ValueError("Calibrated parameters require bounds.")
        if self.parameter_id == "caffeine_renal_plasma_clearance":
            if len(self.bootstrap_values) != 6:
                raise ValueError("Renal clearance requires six source subjects.")
        elif self.bootstrap_values:
            raise ValueError("Only renal clearance has bootstrap values.")
        return self


class RuntimeObservation(StrictBlueprintModel):
    observation_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    split: Literal[
        "calibration",
        "internal_holdout",
        "external_validation",
    ]
    time_min: float = Field(ge=0)
    value_umol_per_l: float = Field(ge=0)
    use_in_objective: bool
    exclusion_reason: str | None = None
    source_artifact_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1)


class RuntimeMetric(StrictBlueprintModel):
    metric_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    endpoint: Literal["auc_0_t", "auc_0_inf", "cmax", "tmax", "half_life"]
    observed_value: float = Field(ge=0)
    unit: str = Field(min_length=1)
    source_artifact_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1)


class RuntimeScenario(StrictBlueprintModel):
    name: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    split: Literal[
        "calibration",
        "internal_holdout",
        "external_validation",
        "external_validation_secondary",
    ]
    aggregation_weight: float = Field(gt=0, le=1)
    pkml: RuntimeInputArtifact
    observations: list[RuntimeObservation] = Field(default_factory=list)


class CalibrationThresholds(StrictBlueprintModel):
    minimum_converged_starts: Literal[3] = 3
    top_three_objective_relative_tolerance: Literal[0.0001] = 0.0001
    top_three_parameter_relative_tolerance: Literal[0.001] = 0.001
    boundary_margin_fraction: Literal[0.01] = 0.01
    hessian_condition_number_max: Literal[100000000.0] = 100000000.0
    repeatability_relative_tolerance: Literal[1e-10] = 1e-10
    gmfe_max: Literal[2.0] = 2.0
    geometric_bias_min: Literal[0.5] = 0.5
    geometric_bias_max: Literal[2.0] = 2.0
    minimum_two_fold_fraction: Literal[0.8] = 0.8
    zero_baseline_max_umol_per_l: Literal[1e-9] = 1e-9
    endpoint_fold_error_min: Literal[0.5] = 0.5
    endpoint_fold_error_max: Literal[2.0] = 2.0
    tmax_absolute_error_h_max: Literal[1.0] = 1.0
    external_coverage_min: Literal[0.8] = 0.8


class UncertaintyProtocol(StrictBlueprintModel):
    sample_count: Literal[512] = 512
    seed: Literal[20260731] = 20260731
    batch_size: Literal[32] = 32
    quantiles: list[float] = Field(default_factory=lambda: [0.05, 0.5, 0.95])
    pmc572_auc_end_h: Literal[24.0] = 24.0
    pmc572_terminal_times_h: list[float] = Field(
        default_factory=lambda: [8.0, 10.0, 12.0, 16.0, 24.0]
    )
    caffeine_molecular_weight_g_per_mol: Literal[194.19] = 194.19


class CaffeineCalibrationRunConfig(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    protocol_version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    generated_at: datetime
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameter_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameter_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_audit_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    ospsuite_r_version: Literal[EXPECTED_OSPSUITE_VERSION] = (
        EXPECTED_OSPSUITE_VERSION
    )
    target_output: Literal[TARGET_OUTPUT] = TARGET_OUTPUT
    runtime_root: str = Field(min_length=1)
    output_directory: str = Field(min_length=1)
    parameters: list[RuntimeParameter] = Field(min_length=8, max_length=8)
    scenarios: list[RuntimeScenario] = Field(min_length=7, max_length=7)
    secondary_metrics: list[RuntimeMetric] = Field(min_length=5, max_length=5)
    start_grid: list[list[float]] = Field(min_length=9, max_length=9)
    thresholds: CalibrationThresholds = Field(default_factory=CalibrationThresholds)
    uncertainty: UncertaintyProtocol = Field(default_factory=UncertaintyProtocol)
    checks: dict[str, bool]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


class CaffeineCalibrationAudit(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    auditor_version: Literal["caffeine-calibration-auditor-1.0.0"] = (
        "caffeine-calibration-auditor-1.0.0"
    )
    generated_at: datetime
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_checks: dict[str, bool]
    metric_checks: dict[str, bool]
    scientific_status: str = Field(min_length=1)
    registration_allowed: bool
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


def _load_jsonl(path: Path, model: type[Any]) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(model.model_validate_json(line))
            except Exception as exc:
                raise CaffeineCalibrationError(
                    f"Invalid JSONL row at {path}:{line_number}: {exc}"
                ) from exc
    return rows


def _manifest_checksum_valid(manifest: Any) -> bool:
    return canonical_sha256(
        manifest.model_dump(mode="json", exclude={"checksum"})
    ) == manifest.checksum


def _resolve_pkml(runtime_root: Path, scenario_name: str) -> Path:
    candidates = [
        runtime_root / "run_1" / scenario_name / f"{scenario_name}.pkml",
        runtime_root / scenario_name / f"{scenario_name}.pkml",
        runtime_root / f"{scenario_name}.pkml",
    ]
    matches = [path.resolve() for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise CaffeineCalibrationError(
            f"Expected exactly one PKML for {scenario_name}; found {matches}."
        )
    return matches[0]


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise CaffeineCalibrationError(
            f"Runtime artifact escapes runtime root: {path}"
        ) from exc


def _runtime_audit(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("accepted") is not True:
        raise CaffeineCalibrationError("PK-Sim runtime audit is not accepted.")
    checksum = payload.get("checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise CaffeineCalibrationError("Runtime audit has no valid checksum.")
    recomputed = canonical_sha256(
        {key: value for key, value in payload.items() if key != "checksum"}
    )
    if recomputed != checksum:
        raise CaffeineCalibrationError("Runtime audit checksum is invalid.")
    scenario_names = payload.get("scenario_names")
    if scenario_names != list(EXPECTED_SCENARIOS):
        raise CaffeineCalibrationError("Runtime audit scenario set or order drifted.")
    return payload, checksum


def _parameter_specs(
    manifest: CaffeineParameterManifest,
) -> list[RuntimeParameter]:
    output: list[RuntimeParameter] = []
    for item in manifest.parameters:
        try:
            runtime_path = RUNTIME_PARAMETER_PATHS[item.parameter_id]
        except KeyError as exc:
            raise CaffeineCalibrationError(
                f"No runtime path for {item.parameter_id}."
            ) from exc
        bootstrap_values = (
            [
                value.normalized_clearance_ml_per_min_per_kg
                for value in item.derivation_inputs
            ]
            if item.parameter_id == "caffeine_renal_plasma_clearance"
            else []
        )
        output.append(
            RuntimeParameter(
                parameter_id=item.parameter_id,
                runtime_path=runtime_path,
                origin=item.origin,
                value=item.value,
                unit=item.unit,
                uncertainty_type=item.uncertainty_type,
                uncertainty=item.uncertainty,
                reported_lower=item.reported_lower,
                reported_upper=item.reported_upper,
                lower_bound=item.lower_bound,
                upper_bound=item.upper_bound,
                calibrate=item.calibrate,
                bootstrap_values=bootstrap_values,
            )
        )
    if [item.parameter_id for item in output] != list(RUNTIME_PARAMETER_PATHS):
        raise CaffeineCalibrationError("Parameter order or identity drifted.")
    return output


def _observations(
    rows: list[ObservationPoint], dataset_id: str
) -> list[RuntimeObservation]:
    selected = [item for item in rows if item.dataset_id == dataset_id]
    selected.sort(key=lambda item: (item.time_min, item.observation_id))
    return [
        RuntimeObservation(
            observation_id=item.observation_id,
            dataset_id=item.dataset_id,
            split=item.split,
            time_min=item.time_min,
            value_umol_per_l=item.canonical_value_umol_per_l,
            use_in_objective=item.use_in_objective,
            exclusion_reason=item.exclusion_reason,
            source_artifact_id=item.provenance.artifact_id,
            source_location=item.provenance.source_location,
        )
        for item in selected
    ]


def prepare_caffeine_calibration(
    *,
    dataset_directory: Path,
    parameter_directory: Path,
    scenario_directory: Path,
    runtime_root: Path,
    runtime_audit_path: Path,
    output_directory: Path,
    config_path: Path,
) -> CaffeineCalibrationRunConfig:
    dataset_path = dataset_directory / "dataset_split_manifest.json"
    observations_path = dataset_directory / "observations.jsonl"
    metrics_path = dataset_directory / "pk_metrics.jsonl"
    parameter_path = parameter_directory / "caffeine_parameter_manifest.json"
    scenario_path = scenario_directory / "study_scenario_manifest.json"
    dataset = CaffeineDatasetManifest.model_validate_json(
        dataset_path.read_text(encoding="utf-8")
    )
    parameter = CaffeineParameterManifest.model_validate_json(
        parameter_path.read_text(encoding="utf-8")
    )
    scenario = CaffeineStudyScenarioManifest.model_validate_json(
        scenario_path.read_text(encoding="utf-8")
    )
    observation_rows = _load_jsonl(observations_path, ObservationPoint)
    metric_rows = _load_jsonl(metrics_path, PKMetric)
    runtime_audit, runtime_audit_checksum = _runtime_audit(runtime_audit_path)

    runtime_root = runtime_root.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise CaffeineCalibrationError(
            f"Calibration output directory is not empty: {output_directory}"
        )
    scenario_by_name = {item.name: item for item in scenario.simulations}
    runtime_scenarios: list[RuntimeScenario] = []
    for name in EXPECTED_SCENARIOS:
        item = scenario_by_name.get(name)
        if item is None:
            raise CaffeineCalibrationError(f"Scenario is missing: {name}")
        pkml = _resolve_pkml(runtime_root, name)
        runtime_scenarios.append(
            RuntimeScenario(
                name=name,
                dataset_id=item.dataset_id,
                split=item.split,
                aggregation_weight=item.aggregation_weight,
                pkml=RuntimeInputArtifact(
                    relative_path=_relative(pkml, runtime_root),
                    sha256=sha256_file(pkml),
                    byte_size=pkml.stat().st_size,
                ),
                observations=_observations(observation_rows, item.dataset_id),
            )
        )
    secondary_metrics = [
        RuntimeMetric(
            metric_id=item.metric_id,
            dataset_id=item.dataset_id,
            endpoint=item.endpoint,
            observed_value=item.value,
            unit=item.unit,
            source_artifact_id=item.provenance.artifact_id,
            source_location=item.provenance.source_location,
        )
        for item in sorted(metric_rows, key=lambda value: value.endpoint)
    ]
    parameter_specs = _parameter_specs(parameter)
    estimated = [item for item in parameter_specs if item.calibrate]
    if [item.parameter_id for item in estimated] != [
        "caffeine_cyp1a2_vmax_effective_n3",
        "caffeine_effective_lipophilicity",
    ]:
        raise CaffeineCalibrationError("Estimated parameter set drifted.")
    vmax = estimated[0]
    logp = estimated[1]
    assert vmax.lower_bound is not None and vmax.upper_bound is not None
    assert logp.lower_bound is not None and logp.upper_bound is not None
    vmax_grid = [vmax.lower_bound, (vmax.lower_bound + vmax.upper_bound) / 2, vmax.upper_bound]
    logp_grid = [logp.lower_bound, (logp.lower_bound + logp.upper_bound) / 2, logp.upper_bound]
    start_grid = [[left, right] for left in vmax_grid for right in logp_grid]
    checks = {
        "dataset_manifest_accepted": dataset.accepted and all(dataset.checks.values()),
        "dataset_manifest_checksum_reproducible": _manifest_checksum_valid(dataset),
        "parameter_manifest_accepted": parameter.accepted and all(parameter.checks.values()),
        "parameter_manifest_checksum_reproducible": _manifest_checksum_valid(parameter),
        "scenario_manifest_accepted": scenario.accepted and all(scenario.checks.values()),
        "scenario_manifest_checksum_reproducible": _manifest_checksum_valid(scenario),
        "runtime_audit_accepted": runtime_audit["accepted"] is True,
        "runtime_audit_root_matches": Path(runtime_audit["runtime_root"]).resolve()
        == runtime_root,
        "seven_scenarios_exact": tuple(item.name for item in runtime_scenarios)
        == EXPECTED_SCENARIOS,
        "fifty_observations_exact": sum(
            len(item.observations) for item in runtime_scenarios
            if item.name not in {"EXT2_PMC572_PO_MALE", "EXT2_PMC572_PO_FEMALE"}
        )
        == 50,
        "five_secondary_metrics_exact": len(secondary_metrics) == 5,
        "calibration_split_exact": {
            item.dataset_id
            for item in runtime_scenarios
            if item.split == "calibration"
        }
        == set(CALIBRATION_DATASETS),
        "nine_start_grid_exact": len(start_grid) == 9
        and len({tuple(item) for item in start_grid}) == 9,
        "all_pkml_nonempty": all(item.pkml.byte_size > 0 for item in runtime_scenarios),
    }
    payload = {
        "generated_at": datetime.now(UTC),
        "dataset_manifest_sha256": sha256_file(dataset_path),
        "dataset_manifest_checksum": dataset.checksum,
        "parameter_manifest_sha256": sha256_file(parameter_path),
        "parameter_manifest_checksum": parameter.checksum,
        "scenario_manifest_sha256": sha256_file(scenario_path),
        "scenario_manifest_checksum": scenario.checksum,
        "runtime_audit_sha256": sha256_file(runtime_audit_path),
        "runtime_audit_checksum": runtime_audit_checksum,
        "runtime_root": runtime_root.as_posix(),
        "output_directory": output_directory.as_posix(),
        "parameters": parameter_specs,
        "scenarios": runtime_scenarios,
        "secondary_metrics": secondary_metrics,
        "start_grid": start_grid,
        "checks": checks,
        "accepted": all(checks.values()),
    }
    draft = CaffeineCalibrationRunConfig(**payload, checksum="0" * 64)
    config = draft.model_copy(
        update={
            "checksum": canonical_sha256(
                draft.model_dump(mode="json", exclude={"checksum"})
            )
        }
    )
    if not config.accepted:
        failed = [name for name, value in checks.items() if not value]
        raise CaffeineCalibrationError(
            f"Calibration input checks failed: {', '.join(failed)}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(config_path, config, canonical=True)
    return config


def _float(value: str, *, field: str) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise CaffeineCalibrationError(f"Invalid {field}: {value!r}") from exc
    if not math.isfinite(output):
        raise CaffeineCalibrationError(f"Non-finite {field}: {value!r}")
    return output


def concentration_metrics(
    rows: list[dict[str, str]],
    *,
    zero_baseline_max: float = 1e-9,
) -> dict[str, float | bool | int]:
    positive_ratios: list[float] = []
    log_errors: list[float] = []
    zero_pass = True
    included = 0
    for row in rows:
        if row.get("included", "true").casefold() != "true":
            continue
        observed = _float(row["observed_umol_per_l"], field="observed")
        predicted = _float(row["predicted_umol_per_l"], field="predicted")
        if predicted < 0:
            raise CaffeineCalibrationError("Negative concentration prediction.")
        included += 1
        if observed == 0:
            zero_pass = zero_pass and predicted <= zero_baseline_max
            continue
        if predicted <= 0:
            raise CaffeineCalibrationError(
                "Non-positive prediction cannot enter log metrics."
            )
        ratio = predicted / observed
        positive_ratios.append(ratio)
        log_errors.append(math.log(ratio))
    if not positive_ratios:
        raise CaffeineCalibrationError("No positive observations for metrics.")
    gmfe = math.exp(sum(abs(value) for value in log_errors) / len(log_errors))
    bias = math.exp(sum(log_errors) / len(log_errors))
    log_rmse = math.sqrt(
        sum(value * value for value in log_errors) / len(log_errors)
    )
    within = sum(0.5 <= value <= 2.0 for value in positive_ratios)
    return {
        "included_count": included,
        "positive_count": len(positive_ratios),
        "gmfe": gmfe,
        "geometric_bias": bias,
        "log_rmse": log_rmse,
        "two_fold_fraction": within / len(positive_ratios),
        "zero_baseline_pass": zero_pass,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_caffeine_calibration(
    *,
    config_path: Path,
    run_manifest_path: Path,
    output_path: Path,
) -> CaffeineCalibrationAudit:
    config = CaffeineCalibrationRunConfig.model_validate_json(
        config_path.read_text(encoding="utf-8")
    )
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    output_directory = Path(config.output_directory).resolve()
    point_path = output_directory / "point_predictions.csv"
    metric_path = output_directory / "validation_metrics.csv"
    artifact_inventory = manifest.get("artifacts", [])
    artifacts: dict[str, dict[str, Any]] = {
        item["relative_path"]: item for item in artifact_inventory
    }
    artifact_checks = {
        "config_checksum_reproducible": canonical_sha256(
            config.model_dump(mode="json", exclude={"checksum"})
        )
        == config.checksum,
        "manifest_config_sha256_matches": manifest.get("config_sha256")
        == sha256_file(config_path),
        "run_completed": manifest.get("run_completed") is True,
        "point_predictions_present": point_path.is_file(),
        "validation_metrics_present": metric_path.is_file(),
        "nine_starts_executed": manifest.get("calibration", {}).get(
            "start_count"
        )
        == 9,
        "all_seven_scenarios_executed": manifest.get("scenario_count") == 7,
        "uncertainty_512_executed": manifest.get("uncertainty", {}).get(
            "successful_sample_count"
        )
        == 512,
        "artifact_inventory_nonempty": bool(artifacts),
    }
    inventory_valid = True
    for relative_path, expected in artifacts.items():
        path = (output_directory / relative_path).resolve()
        try:
            path.relative_to(output_directory)
        except ValueError:
            inventory_valid = False
            continue
        if (
            not path.is_file()
            or path.stat().st_size != expected.get("byte_size")
            or sha256_file(path) != expected.get("sha256")
        ):
            inventory_valid = False
    artifact_checks["artifact_inventory_verified"] = inventory_valid

    metric_checks: dict[str, bool] = {}
    if point_path.is_file() and metric_path.is_file():
        point_rows = _read_csv(point_path)
        reported = {
            row["dataset_id"]: row for row in _read_csv(metric_path)
        }
        for dataset_id in (
            "blanchard_1983_iv_5mgkg",
            "blanchard_1983_oral_5mgkg",
            "wahllander_1989_iv_3mgkg",
            "eudract_2011_002291_16_oral_50mg",
            "pmc5350996_synthetic_200mg",
        ):
            rows = [row for row in point_rows if row["dataset_id"] == dataset_id]
            recalculated = concentration_metrics(rows)
            row = reported.get(dataset_id)
            if row is None:
                metric_checks[f"{dataset_id}_reported"] = False
                continue
            for key in ("gmfe", "geometric_bias", "log_rmse", "two_fold_fraction"):
                actual = _float(row[key], field=f"{dataset_id}.{key}")
                expected = float(recalculated[key])
                metric_checks[f"{dataset_id}_{key}_reproduced"] = math.isclose(
                    actual,
                    expected,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            metric_checks[f"{dataset_id}_zero_baseline_reproduced"] = (
                row["zero_baseline_pass"].casefold()
                == str(recalculated["zero_baseline_pass"]).casefold()
            )
    scientific_status = str(manifest.get("scientific_status", "run_incomplete"))
    registration_allowed = (
        scientific_status == "external_study_validated"
        and manifest.get("registration_allowed") is True
    )
    accepted = all(artifact_checks.values()) and all(metric_checks.values())
    payload = {
        "generated_at": datetime.now(UTC),
        "config_sha256": sha256_file(config_path),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "artifact_checks": artifact_checks,
        "metric_checks": metric_checks,
        "scientific_status": scientific_status,
        "registration_allowed": registration_allowed,
        "accepted": accepted,
    }
    draft = CaffeineCalibrationAudit(**payload, checksum="0" * 64)
    audit = draft.model_copy(
        update={
            "checksum": canonical_sha256(
                draft.model_dump(mode="json", exclude={"checksum"})
            )
        }
    )
    write_json_atomic(output_path, audit, canonical=True)
    return audit


def diagnose_caffeine_calibration(
    *,
    config_path: Path,
    source_run_manifest_path: Path,
    output_directory: Path,
    rscript_path: Path,
    script_path: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    source_run_manifest_path = source_run_manifest_path.resolve()
    output_directory = output_directory.resolve()
    config = CaffeineCalibrationRunConfig.model_validate_json(
        config_path.read_text(encoding="utf-8")
    )
    if canonical_sha256(
        config.model_dump(mode="json", exclude={"checksum"})
    ) != config.checksum:
        raise CaffeineCalibrationError("Calibration config checksum is invalid.")
    expected_manifest = Path(config.output_directory).resolve() / "run_manifest.json"
    if source_run_manifest_path != expected_manifest:
        raise CaffeineCalibrationError(
            "Diagnostic source manifest is not the configured calibration run."
        )
    manifest = json.loads(source_run_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("config_sha256") != sha256_file(config_path)
        or manifest.get("scientific_status")
        not in {"calibration_failed", "parameter_unidentifiable"}
        or manifest.get("registration_allowed") is not False
        or manifest.get("validation") is not None
    ):
        raise CaffeineCalibrationError(
            "Diagnostic source is not an isolated calibration failure."
        )
    if output_directory.exists() and any(output_directory.iterdir()):
        raise CaffeineCalibrationError(
            f"Diagnostic output directory is not empty: {output_directory}"
        )
    subprocess.run(
        [
            str(rscript_path.resolve()),
            str(script_path.resolve()),
            str(config_path),
            "diagnostic",
            str(output_directory),
        ],
        check=True,
    )
    diagnostic_path = output_directory / "diagnostic_manifest.json"
    payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    checks = payload.get("checks", {})
    if not checks or not all(checks.values()):
        raise CaffeineCalibrationError("Calibration diagnostic checks failed.")
    if (
        payload.get("holdout_or_external_scenarios_loaded") is not False
        or payload.get("holdout_or_external_scenarios_run") is not False
    ):
        raise CaffeineCalibrationError(
            "Calibration diagnostic violated split isolation."
        )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, execute, diagnose, or audit the locked caffeine PBPK fit."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--dataset-directory", type=Path, required=True)
    prepare.add_argument("--parameter-directory", type=Path, required=True)
    prepare.add_argument("--scenario-directory", type=Path, required=True)
    prepare.add_argument("--runtime-root", type=Path, required=True)
    prepare.add_argument("--runtime-audit", type=Path, required=True)
    prepare.add_argument("--output-directory", type=Path, required=True)
    prepare.add_argument("--config", type=Path, required=True)
    execute = commands.add_parser("execute")
    execute.add_argument("--config", type=Path, required=True)
    execute.add_argument("--rscript", type=Path, required=True)
    execute.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).with_name("R") / "caffeine_fit_validate.R",
    )
    diagnose = commands.add_parser("diagnose")
    diagnose.add_argument("--config", type=Path, required=True)
    diagnose.add_argument("--source-run-manifest", type=Path, required=True)
    diagnose.add_argument("--output-directory", type=Path, required=True)
    diagnose.add_argument("--rscript", type=Path, required=True)
    diagnose.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).with_name("R") / "caffeine_fit_validate.R",
    )
    audit = commands.add_parser("audit")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--run-manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        config = prepare_caffeine_calibration(
            dataset_directory=args.dataset_directory,
            parameter_directory=args.parameter_directory,
            scenario_directory=args.scenario_directory,
            runtime_root=args.runtime_root,
            runtime_audit_path=args.runtime_audit,
            output_directory=args.output_directory,
            config_path=args.config,
        )
        print(json.dumps(config.model_dump(mode="json"), ensure_ascii=False))
        return
    if args.command == "execute":
        subprocess.run(
            [str(args.rscript), str(args.script), str(args.config.resolve())],
            check=True,
        )
        return
    if args.command == "diagnose":
        payload = diagnose_caffeine_calibration(
            config_path=args.config,
            source_run_manifest_path=args.source_run_manifest,
            output_directory=args.output_directory,
            rscript_path=args.rscript,
            script_path=args.script,
        )
        print(json.dumps(payload, ensure_ascii=False))
        return
    audit = audit_caffeine_calibration(
        config_path=args.config,
        run_manifest_path=args.run_manifest,
        output_path=args.output,
    )
    print(json.dumps(audit.model_dump(mode="json"), ensure_ascii=False))
    if not audit.accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
