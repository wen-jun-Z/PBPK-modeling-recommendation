from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from pbpk_kb.blueprints.caffeine_v2_parameters import (
    CaffeineV2ParameterManifest,
)
from pbpk_kb.blueprints.caffeine_v3_sources import CaffeineV3SourceManifest
from pbpk_kb.blueprints.schemas import StrictBlueprintModel
from pbpk_kb.model_assets.execution_plan import (
    canonical_sha256,
    write_json_atomic,
)
from pbpk_kb.model_assets.license_policy import sha256_file


PARAMETER_BUILDER_VERSION = "caffeine-v3-parameter-builder-1.0.0"
PARAMETER_MANIFEST_VERSION = "caffeine_parameter_manifest_v3"
MODEL_VERSION = "caffeine_parent_pathway_model_v3"
DATASET_VERSION = "caffeine_dataset_v3"
DISPOSITION_DATASET = "blanchard_1983_iv_5mgkg"
ABSORPTION_DATASET = "blanchard_1983_oral_5mgkg"
DISPOSITION_OBJECTIVE = (
    "mean((log(Cpred_blanchard_iv)-log(Cobs_blanchard_iv))^2)"
)
CLEARANCE_START_FRACTIONS = (
    0.01,
    0.05,
    0.10,
    0.20,
    0.40,
    0.60,
    0.80,
    0.90,
    0.99,
)
PERMEABILITY_LOG10_MIN = -9.0
PERMEABILITY_LOG10_MAX = -2.0
PERMEABILITY_LOG10_STEP = 0.1
PERMEABILITY_INITIAL_VALUE = 1e-5


class CaffeineV3ParameterError(RuntimeError):
    pass


class RuntimeMapping(StrictBlueprintModel):
    runtime_path: str = Field(min_length=1)
    multiplier: float = Field(gt=0)
    mapping_rule: str = Field(min_length=1)


class ParameterLineage(StrictBlueprintModel):
    source_artifact_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    source_value_text: str = Field(min_length=1)
    source_value_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_parameter_id: str | None = None


class CaffeineV3ParameterRecord(StrictBlueprintModel):
    parameter_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    role: Literal[
        "compound_input",
        "renal_clearance",
        "pathway_fraction",
        "total_metabolic_capacity",
        "intestinal_absorption",
    ]
    origin: Literal["observed", "derived", "estimated", "set_identified"]
    identification_mode: Literal["fixed", "point_estimated", "set_identified"]
    value: float
    unit: str = Field(min_length=1)
    lower_bound: float | None = None
    upper_bound: float | None = None
    calibrate: bool
    calibration_dataset_ids: list[str] = Field(default_factory=list)
    constraint_dataset_ids: list[str] = Field(default_factory=list)
    objective_function: str | None = None
    runtime_mappings: list[RuntimeMapping] = Field(default_factory=list)
    lineage: list[ParameterLineage] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identification_contract(self) -> "CaffeineV3ParameterRecord":
        if not math.isfinite(self.value):
            raise ValueError("Parameter value must be finite.")
        if self.identification_mode == "point_estimated":
            if self.origin != "estimated" or not self.calibrate:
                raise ValueError("Point-estimated parameter contract drifted.")
            if self.lower_bound is None or self.upper_bound is None:
                raise ValueError("Point-estimated parameters require bounds.")
            if not self.lower_bound <= self.value <= self.upper_bound:
                raise ValueError("Estimated initial value is outside bounds.")
            if self.calibration_dataset_ids != [DISPOSITION_DATASET]:
                raise ValueError("Clearance must use IV disposition data only.")
            if self.constraint_dataset_ids:
                raise ValueError("Point-estimated parameter has constraints.")
            if self.objective_function != DISPOSITION_OBJECTIVE:
                raise ValueError("Disposition objective drifted.")
        elif self.identification_mode == "set_identified":
            if self.origin != "set_identified" or self.calibrate:
                raise ValueError("Set-identified parameter contract drifted.")
            if self.lower_bound is None or self.upper_bound is None:
                raise ValueError("Set identification requires search guards.")
            if not self.lower_bound <= self.value <= self.upper_bound:
                raise ValueError("Set-search initial value is outside guards.")
            if self.calibration_dataset_ids:
                raise ValueError("Set-identified permeability is not point-fitted.")
            if self.constraint_dataset_ids != [
                DISPOSITION_DATASET,
                ABSORPTION_DATASET,
            ]:
                raise ValueError("Absorption constraint datasets drifted.")
            if self.objective_function is not None:
                raise ValueError("Set identification cannot define an objective.")
        else:
            if self.origin in {"estimated", "set_identified"}:
                raise ValueError("Fixed parameter origin drifted.")
            if self.calibrate or self.calibration_dataset_ids:
                raise ValueError("Fixed parameters cannot be calibrated.")
            if self.constraint_dataset_ids:
                raise ValueError("Fixed parameters cannot define constraints.")
            if self.lower_bound is not None or self.upper_bound is not None:
                raise ValueError("Fixed parameters cannot define search guards.")
            if self.objective_function is not None:
                raise ValueError("Fixed parameters cannot define an objective.")
        if not self.runtime_mappings and self.role != "pathway_fraction":
            raise ValueError("Runtime parameter requires at least one mapping.")
        if self.role == "pathway_fraction" and self.runtime_mappings:
            raise ValueError("Pathway fractions are applied through clearance.")
        return self


class ClearanceStart(StrictBlueprintModel):
    start_id: int = Field(ge=1, le=9)
    normalized_fraction: float = Field(gt=0, lt=1)
    total_cyp1a2_specific_clearance: float = Field(gt=0)


class PermeabilityCandidate(StrictBlueprintModel):
    candidate_id: int = Field(ge=1, le=71)
    log10_dm_per_min: float = Field(ge=-9.0, le=-2.0)
    value_dm_per_min: float = Field(gt=0)


class AbsorptionSetProtocol(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    identification_type: Literal["set_identified_nuisance_parameter"] = (
        "set_identified_nuisance_parameter"
    )
    candidate_count: Literal[71] = 71
    log10_lower_guard: Literal[-9.0] = -9.0
    log10_upper_guard: Literal[-2.0] = -2.0
    log10_step: Literal[0.1] = 0.1
    bioavailability_lower: Literal[0.99] = 0.99
    bioavailability_upper: Literal[1.01] = 1.01
    tmax_lower_min: Literal[11.5] = 11.5
    tmax_upper_min: Literal[48.1] = 48.1
    plateau_multiplier: Literal[10.0] = 10.0
    plateau_bioavailability_delta_max: Literal[0.005] = 0.005
    plateau_tmax_delta_h_max: Literal[0.05] = 0.05
    plateau_concentration_symmetric_relative_delta_max: Literal[0.02] = 0.02
    reference_selection: Literal["minimum_admissible_candidate"] = (
        "minimum_admissible_candidate"
    )
    upper_bound_interpretation: Literal["right_censored_numerical_guard"] = (
        "right_censored_numerical_guard"
    )


class CaffeineV3ParameterManifest(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    builder_version: Literal[PARAMETER_BUILDER_VERSION] = (
        PARAMETER_BUILDER_VERSION
    )
    manifest_version: Literal[PARAMETER_MANIFEST_VERSION] = (
        PARAMETER_MANIFEST_VERSION
    )
    model_version: Literal[MODEL_VERSION] = MODEL_VERSION
    generated_at: datetime
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_parameter_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_parameter_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_version: Literal[DATASET_VERSION] = DATASET_VERSION
    dataset_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameters: list[CaffeineV3ParameterRecord] = Field(
        min_length=11,
        max_length=11,
    )
    disposition_objective: Literal[DISPOSITION_OBJECTIVE] = DISPOSITION_OBJECTIVE
    clearance_starts: list[ClearanceStart] = Field(min_length=9, max_length=9)
    permeability_candidates: list[PermeabilityCandidate] = Field(
        min_length=71,
        max_length=71,
    )
    absorption_set_protocol: AbsorptionSetProtocol = Field(
        default_factory=AbsorptionSetProtocol
    )
    structural_scope: list[str] = Field(min_length=7)
    checks: dict[str, bool]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _verify_manifest(
    path: Path,
    *,
    version_key: str,
    version: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checksum = payload.get("checksum")
    if payload.get(version_key) != version or payload.get("accepted") is not True:
        raise CaffeineV3ParameterError(f"Unaccepted or drifted manifest: {path}")
    if canonical_sha256(
        {key: value for key, value in payload.items() if key != "checksum"}
    ) != checksum:
        raise CaffeineV3ParameterError(f"Manifest checksum is invalid: {path}")
    return payload


def _lineage_from_parent(record: Any) -> list[ParameterLineage]:
    source_text = json.dumps(
        record.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return [
        ParameterLineage(
            source_artifact_id="caffeine_parameter_manifest_v2",
            source_location=f"parameters[{record.parameter_id}]",
            source_value_text=source_text,
            source_value_text_sha256=_text_hash(source_text),
            parent_parameter_id=record.parameter_id,
        )
    ]


def _method_lineage() -> ParameterLineage:
    source_text = (
        "Caco-2 Papp cannot be mapped one-to-one to PK-Sim specific "
        "intestinal permeability; identify a model-specific admissible set."
    )
    return ParameterLineage(
        source_artifact_id="pksim_compound_workflow_html",
        source_location="Absorption/Specific Intestinal Permeability",
        source_value_text=source_text,
        source_value_text_sha256=_text_hash(source_text),
    )


def _runtime_mappings(record: Any) -> list[RuntimeMapping]:
    return [
        RuntimeMapping(
            runtime_path=item.runtime_path,
            multiplier=item.multiplier,
            mapping_rule=item.mapping_rule,
        )
        for item in record.runtime_mappings
    ]


def _write_parameter_csv(
    path: Path,
    parameters: list[CaffeineV3ParameterRecord],
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "parameter_id",
            "role",
            "origin",
            "identification_mode",
            "value",
            "unit",
            "lower_bound",
            "upper_bound",
            "calibrate",
            "calibration_dataset_ids",
            "constraint_dataset_ids",
            "runtime_mapping_count",
        ]
    )
    for item in parameters:
        writer.writerow(
            [
                item.parameter_id,
                item.role,
                item.origin,
                item.identification_mode,
                item.value,
                item.unit,
                item.lower_bound,
                item.upper_bound,
                item.calibrate,
                "|".join(item.calibration_dataset_ids),
                "|".join(item.constraint_dataset_ids),
                len(item.runtime_mappings),
            ]
        )
    path.write_text(buffer.getvalue(), encoding="utf-8")


def _permeability_grid() -> list[PermeabilityCandidate]:
    output: list[PermeabilityCandidate] = []
    for index in range(71):
        log_value = round(
            PERMEABILITY_LOG10_MIN + index * PERMEABILITY_LOG10_STEP,
            10,
        )
        output.append(
            PermeabilityCandidate(
                candidate_id=index + 1,
                log10_dm_per_min=log_value,
                value_dm_per_min=10.0**log_value,
            )
        )
    return output


def build_caffeine_v3_parameter_manifest(
    *,
    source_manifest_path: Path,
    dataset_manifest_path: Path,
    output_dir: Path,
) -> CaffeineV3ParameterManifest:
    if output_dir.exists():
        raise CaffeineV3ParameterError(f"Immutable output exists: {output_dir}")
    source_payload = _verify_manifest(
        source_manifest_path,
        version_key="package_version",
        version="caffeine_absorption_sources_v3",
    )
    source = CaffeineV3SourceManifest.model_validate(source_payload)
    parent_path = source_manifest_path.parent / "caffeine_parameter_manifest_v2.json"
    parent_payload = _verify_manifest(
        parent_path,
        version_key="manifest_version",
        version="caffeine_parameter_manifest_v2",
    )
    parent = CaffeineV2ParameterManifest.model_validate(parent_payload)
    if (
        source.parent_parameter_manifest_checksum != parent.checksum
        or source.parent_parameter_manifest_sha256 != sha256_file(parent_path)
    ):
        raise CaffeineV3ParameterError("v3 source-to-parent lineage drifted.")
    dataset = _verify_manifest(
        dataset_manifest_path,
        version_key="dataset_version",
        version=DATASET_VERSION,
    )
    dataset_splits = {
        item["dataset_id"]: item["split"] for item in dataset["datasets"]
    }
    if dataset_splits.get(DISPOSITION_DATASET) != "calibration" or (
        dataset_splits.get(ABSORPTION_DATASET) != "calibration"
    ):
        raise CaffeineV3ParameterError("Blanchard dataset split drifted.")

    parent_by_id = {item.parameter_id: item for item in parent.parameters}
    fixed_ids = (
        "caffeine_molecular_weight",
        "caffeine_pka_base",
        "caffeine_water_solubility_25c",
        "caffeine_fraction_unbound_plasma",
        "caffeine_renal_plasma_clearance",
        "caffeine_effective_lipophilicity",
        "caffeine_fraction_n3_paraxanthine",
        "caffeine_fraction_n1_theobromine",
        "caffeine_fraction_n7_theophylline",
    )
    parameters: list[CaffeineV3ParameterRecord] = []
    for parameter_id in fixed_ids:
        record = parent_by_id[parameter_id]
        parameters.append(
            CaffeineV3ParameterRecord(
                parameter_id=record.parameter_id,
                name=record.name,
                role=record.role,
                origin=record.origin,
                identification_mode="fixed",
                value=record.value,
                unit=record.unit,
                calibrate=False,
                runtime_mappings=_runtime_mappings(record),
                lineage=_lineage_from_parent(record),
            )
        )

    clearance_parent = parent_by_id["total_cyp1a2_specific_clearance"]
    if clearance_parent.lower_bound is None or clearance_parent.upper_bound is None:
        raise CaffeineV3ParameterError("Parent clearance search domain is missing.")
    parameters.append(
        CaffeineV3ParameterRecord(
            parameter_id=clearance_parent.parameter_id,
            name=clearance_parent.name,
            role=clearance_parent.role,
            origin="estimated",
            identification_mode="point_estimated",
            value=clearance_parent.upper_bound * 0.30,
            unit=clearance_parent.unit,
            lower_bound=clearance_parent.lower_bound,
            upper_bound=clearance_parent.upper_bound,
            calibrate=True,
            calibration_dataset_ids=[DISPOSITION_DATASET],
            objective_function=DISPOSITION_OBJECTIVE,
            runtime_mappings=_runtime_mappings(clearance_parent),
            lineage=_lineage_from_parent(clearance_parent),
        )
    )

    permeability_parent = parent_by_id[
        "specific_intestinal_permeability_transcellular"
    ]
    parameters.append(
        CaffeineV3ParameterRecord(
            parameter_id=permeability_parent.parameter_id,
            name=permeability_parent.name,
            role=permeability_parent.role,
            origin="set_identified",
            identification_mode="set_identified",
            value=PERMEABILITY_INITIAL_VALUE,
            unit="dm/min",
            lower_bound=10.0**PERMEABILITY_LOG10_MIN,
            upper_bound=10.0**PERMEABILITY_LOG10_MAX,
            calibrate=False,
            constraint_dataset_ids=[DISPOSITION_DATASET, ABSORPTION_DATASET],
            runtime_mappings=[
                RuntimeMapping(
                    runtime_path=(
                        "Caffeine|Specific intestinal permeability "
                        "(transcellular)"
                    ),
                    multiplier=1.0,
                    mapping_rule="set_search_candidate_not_point_estimate",
                )
            ],
            lineage=_lineage_from_parent(permeability_parent)
            + [_method_lineage()],
        )
    )

    starts = [
        ClearanceStart(
            start_id=index + 1,
            normalized_fraction=fraction,
            total_cyp1a2_specific_clearance=(
                clearance_parent.upper_bound * fraction
            ),
        )
        for index, fraction in enumerate(CLEARANCE_START_FRACTIONS)
    ]
    candidates = _permeability_grid()
    modes = {item.parameter_id: item.identification_mode for item in parameters}
    checks = {
        "source_manifest_checksum_valid": source.checksum
        == source_payload["checksum"],
        "parent_parameter_checksum_valid": parent.checksum
        == parent_payload["checksum"],
        "dataset_manifest_checksum_valid": canonical_sha256(
            {key: value for key, value in dataset.items() if key != "checksum"}
        )
        == dataset["checksum"],
        "parameter_ids_unique": len(parameters)
        == len({item.parameter_id for item in parameters}),
        "only_clearance_point_estimated": [
            item.parameter_id
            for item in parameters
            if item.identification_mode == "point_estimated"
        ]
        == ["total_cyp1a2_specific_clearance"],
        "only_permeability_set_identified": [
            item.parameter_id
            for item in parameters
            if item.identification_mode == "set_identified"
        ]
        == ["specific_intestinal_permeability_transcellular"],
        "nine_fixed_parameters": list(modes.values()).count("fixed") == 9,
        "nine_deterministic_clearance_starts": len(starts) == 9
        and [item.normalized_fraction for item in starts]
        == list(CLEARANCE_START_FRACTIONS),
        "seventy_one_log_candidates_exact": len(candidates) == 71
        and math.isclose(candidates[0].value_dm_per_min, 1e-9)
        and math.isclose(candidates[-1].value_dm_per_min, 1e-2)
        and all(
            math.isclose(
                right.log10_dm_per_min - left.log10_dm_per_min,
                0.1,
                abs_tol=1e-12,
            )
            for left, right in zip(candidates, candidates[1:])
        ),
        "validation_not_used_for_parameterization": all(
            split == "calibration"
            for dataset_id, split in dataset_splits.items()
            if dataset_id in {DISPOSITION_DATASET, ABSORPTION_DATASET}
        ),
        "direct_caco2_mapping_prohibited": any(
            "one-to-one" in item
            for item in source.prohibited_transforms
        ),
    }
    payload = {
        "schema_version": "1.0.0",
        "builder_version": PARAMETER_BUILDER_VERSION,
        "manifest_version": PARAMETER_MANIFEST_VERSION,
        "model_version": MODEL_VERSION,
        "generated_at": datetime.now(UTC),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_manifest_checksum": source.checksum,
        "parent_parameter_manifest_sha256": sha256_file(parent_path),
        "parent_parameter_manifest_checksum": parent.checksum,
        "dataset_version": DATASET_VERSION,
        "dataset_manifest_checksum": dataset["checksum"],
        "parameters": parameters,
        "disposition_objective": DISPOSITION_OBJECTIVE,
        "clearance_starts": starts,
        "permeability_candidates": candidates,
        "absorption_set_protocol": AbsorptionSetProtocol(),
        "structural_scope": [
            "Parent caffeine concentrations remain the modeled PK output.",
            "N3, N1, and N7 human pathway fractions remain fixed.",
            "Only total CYP1A2 specific clearance is point-estimated.",
            "Clearance estimation uses Blanchard IV concentrations only.",
            "Permeability is set-identified from F and Tmax constraints.",
            "Caco-2 and human permeability values support structure only.",
            "Validation scenarios remain checksum-locked until both gates pass.",
            "Metabolite PK and DDI qualification remain outside v3 scope.",
        ],
        "checks": checks,
        "accepted": all(checks.values()),
    }
    draft = CaffeineV3ParameterManifest(**payload, checksum="0" * 64)
    manifest = draft.model_copy(
        update={
            "checksum": canonical_sha256(
                draft.model_dump(mode="json", exclude={"checksum"})
            )
        }
    )
    if not manifest.accepted:
        failed = [name for name, passed in checks.items() if not passed]
        raise CaffeineV3ParameterError(
            "Caffeine v3 parameter acceptance failed: " + ", ".join(failed)
        )

    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        raise CaffeineV3ParameterError(f"Stale output exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        write_json_atomic(
            temporary / "caffeine_parameter_manifest.json",
            manifest,
            canonical=False,
        )
        _write_parameter_csv(temporary / "parameter_values.csv", parameters)
        write_json_atomic(
            temporary / "clearance_start_grid.json",
            [item.model_dump(mode="json") for item in starts],
            canonical=False,
        )
        write_json_atomic(
            temporary / "permeability_search_grid.json",
            [item.model_dump(mode="json") for item in candidates],
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the caffeine v3 point/set parameter contract."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = build_caffeine_v3_parameter_manifest(
        source_manifest_path=args.source_manifest,
        dataset_manifest_path=args.dataset_manifest,
        output_dir=args.output_dir,
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
