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

from pbpk_kb.blueprints.caffeine_v2_sources import (
    CaffeineV2SourceManifest,
)
from pbpk_kb.blueprints.schemas import StrictBlueprintModel
from pbpk_kb.model_assets.execution_plan import (
    canonical_sha256,
    write_json_atomic,
)
from pbpk_kb.model_assets.license_policy import sha256_file


PARAMETER_BUILDER_VERSION = "caffeine-v2-parameter-builder-1.0.0"
PARAMETER_MANIFEST_VERSION = "caffeine_parameter_manifest_v2"
MODEL_VERSION = "caffeine_parent_pathway_model_v2"
DATASET_VERSION = "caffeine_dataset_v3"
CALIBRATION_DATASETS = (
    "blanchard_1983_iv_5mgkg",
    "blanchard_1983_oral_5mgkg",
)
CALIBRATION_OBJECTIVE = (
    "0.5*mean((log(Cpred_iv)-log(Cobs_iv))^2)+"
    "0.5*mean((log(Cpred_oral)-log(Cobs_oral))^2)"
)
START_FRACTIONS = (0.01, 0.10, 0.30, 0.60, 0.90)
RENAL_PROCESS_DATA_SOURCE = "Birkett 1991 Table 1 rederived PBPK-KB-v1"


class CaffeineV2ParameterError(RuntimeError):
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


class CaffeineV2ParameterRecord(StrictBlueprintModel):
    parameter_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    role: Literal[
        "compound_input",
        "renal_clearance",
        "pathway_fraction",
        "total_metabolic_capacity",
        "intestinal_absorption",
    ]
    origin: Literal["observed", "derived", "estimated"]
    value: float
    unit: str = Field(min_length=1)
    lower_bound: float | None = None
    upper_bound: float | None = None
    calibrate: bool
    calibration_dataset_ids: list[str] = Field(default_factory=list)
    objective_function: str | None = None
    runtime_mappings: list[RuntimeMapping] = Field(default_factory=list)
    lineage: list[ParameterLineage] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_parameter(self) -> "CaffeineV2ParameterRecord":
        if not math.isfinite(self.value):
            raise ValueError("Parameter value must be finite.")
        if self.origin == "estimated":
            if not self.calibrate:
                raise ValueError("Estimated parameters must be calibrated.")
            if self.lower_bound is None or self.upper_bound is None:
                raise ValueError("Estimated parameters require bounds.")
            if not self.lower_bound <= self.value <= self.upper_bound:
                raise ValueError("Estimated initial value is outside bounds.")
            if tuple(self.calibration_dataset_ids) != CALIBRATION_DATASETS:
                raise ValueError("Estimated parameter datasets drifted.")
            if self.objective_function != CALIBRATION_OBJECTIVE:
                raise ValueError("Estimated parameter objective drifted.")
            if not self.runtime_mappings:
                raise ValueError("Estimated parameters require runtime mappings.")
        else:
            if self.calibrate or self.calibration_dataset_ids:
                raise ValueError("Fixed parameters cannot be calibrated.")
            if self.lower_bound is not None or self.upper_bound is not None:
                raise ValueError("Fixed parameters cannot define fit bounds.")
            if self.objective_function is not None:
                raise ValueError("Fixed parameters cannot define an objective.")
        if self.role == "pathway_fraction" and self.runtime_mappings:
            raise ValueError("Pathway fractions are applied through total clearance.")
        return self


class CalibrationStart(StrictBlueprintModel):
    start_id: int = Field(ge=1, le=25)
    total_cyp1a2_specific_clearance: float = Field(gt=0)
    specific_intestinal_permeability_transcellular: float = Field(gt=0)


class CaffeineV2ParameterManifest(StrictBlueprintModel):
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
    parent_parameter_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_version: Literal[DATASET_VERSION] = DATASET_VERSION
    dataset_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameters: list[CaffeineV2ParameterRecord] = Field(
        min_length=11,
        max_length=11,
    )
    calibration_objective: Literal[CALIBRATION_OBJECTIVE] = CALIBRATION_OBJECTIVE
    calibration_starts: list[CalibrationStart] = Field(
        min_length=25,
        max_length=25,
    )
    structural_scope: list[str] = Field(min_length=5)
    checks: dict[str, bool]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _verify_manifest(path: Path, *, version_key: str, version: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checksum = payload.get("checksum")
    if payload.get(version_key) != version:
        raise CaffeineV2ParameterError(
            f"Manifest version drifted at {path}: {payload.get(version_key)!r}"
        )
    if payload.get("accepted") is not True:
        raise CaffeineV2ParameterError(f"Manifest is not accepted: {path}")
    if canonical_sha256(
        {key: value for key, value in payload.items() if key != "checksum"}
    ) != checksum:
        raise CaffeineV2ParameterError(f"Manifest checksum is invalid: {path}")
    return payload


def _parent_lineage(record: dict[str, Any]) -> list[ParameterLineage]:
    source_text = json.dumps(
        {
            "parameter_id": record["parameter_id"],
            "origin": record["origin"],
            "value": record["value"],
            "unit": record["unit"],
            "provenance": record["provenance"],
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return [
        ParameterLineage(
            source_artifact_id="caffeine_parameter_manifest_v1",
            source_location=f"parameters[{record['parameter_id']}]",
            source_value_text=source_text,
            source_value_text_sha256=_text_hash(source_text),
            parent_parameter_id=record["parameter_id"],
        )
    ]


def _source_lineage(
    *, artifact_id: str, source_location: str, source_value_text: str
) -> list[ParameterLineage]:
    return [
        ParameterLineage(
            source_artifact_id=artifact_id,
            source_location=source_location,
            source_value_text=source_value_text,
            source_value_text_sha256=_text_hash(source_value_text),
        )
    ]


def _fixed_runtime_path(parameter_id: str) -> str:
    paths = {
        "caffeine_molecular_weight": "Caffeine|Molecular weight",
        "caffeine_pka_base": "Caffeine|pKa value 0",
        "caffeine_water_solubility_25c": "Caffeine|Solubility at reference pH",
        "caffeine_fraction_unbound_plasma": (
            "Caffeine|Fraction unbound (plasma, reference value)"
        ),
        "caffeine_renal_plasma_clearance": (
            "Neighborhoods|Kidney_pls_Kidney_ur|Caffeine|"
            f"Renal Clearances-{RENAL_PROCESS_DATA_SOURCE}-Caffeine|"
            "Plasma clearance"
        ),
        "caffeine_effective_lipophilicity": "Caffeine|Lipophilicity",
    }
    try:
        return paths[parameter_id]
    except KeyError as exc:
        raise CaffeineV2ParameterError(
            f"No fixed runtime path for {parameter_id}."
        ) from exc


def _write_parameter_csv(path: Path, parameters: list[CaffeineV2ParameterRecord]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "parameter_id",
            "role",
            "origin",
            "value",
            "unit",
            "lower_bound",
            "upper_bound",
            "calibrate",
            "runtime_mapping_count",
        ]
    )
    for item in parameters:
        writer.writerow(
            [
                item.parameter_id,
                item.role,
                item.origin,
                item.value,
                item.unit,
                item.lower_bound,
                item.upper_bound,
                item.calibrate,
                len(item.runtime_mappings),
            ]
        )
    path.write_text(buffer.getvalue(), encoding="utf-8")


def build_caffeine_v2_parameter_manifest(
    *,
    source_manifest_path: Path,
    dataset_manifest_path: Path,
    output_dir: Path,
) -> CaffeineV2ParameterManifest:
    if output_dir.exists():
        raise CaffeineV2ParameterError(f"Immutable output exists: {output_dir}")
    source_payload = _verify_manifest(
        source_manifest_path,
        version_key="package_version",
        version="caffeine_parameter_sources_v2",
    )
    source_manifest = CaffeineV2SourceManifest.model_validate(source_payload)
    dataset = _verify_manifest(
        dataset_manifest_path,
        version_key="dataset_version",
        version=DATASET_VERSION,
    )
    dataset_ids = {item["dataset_id"]: item["split"] for item in dataset["datasets"]}
    if any(dataset_ids.get(item) != "calibration" for item in CALIBRATION_DATASETS):
        raise CaffeineV2ParameterError("Calibration dataset split drifted.")

    source_root = source_manifest_path.parent
    parent_manifest_path = source_root / "caffeine_parameter_manifest_v1.json"
    parent = _verify_manifest(
        parent_manifest_path,
        version_key="manifest_version",
        version="caffeine_parameter_manifest_v1",
    )
    parent_records = {item["parameter_id"]: item for item in parent["parameters"]}
    fixed_parent_ids = (
        "caffeine_molecular_weight",
        "caffeine_pka_base",
        "caffeine_water_solubility_25c",
        "caffeine_fraction_unbound_plasma",
        "caffeine_renal_plasma_clearance",
        "caffeine_effective_lipophilicity",
    )
    if any(item not in parent_records for item in fixed_parent_ids):
        raise CaffeineV2ParameterError("Parent fixed parameter set is incomplete.")

    parameters: list[CaffeineV2ParameterRecord] = []
    for parameter_id in fixed_parent_ids:
        record = parent_records[parameter_id]
        role: Literal["compound_input", "renal_clearance"] = (
            "renal_clearance"
            if parameter_id == "caffeine_renal_plasma_clearance"
            else "compound_input"
        )
        parameters.append(
            CaffeineV2ParameterRecord(
                parameter_id=parameter_id,
                name=record["name"],
                role=role,
                origin=(
                    "observed"
                    if parameter_id == "caffeine_effective_lipophilicity"
                    else record["origin"]
                ),
                value=record["value"],
                unit=record["unit"],
                calibrate=False,
                runtime_mappings=[
                    RuntimeMapping(
                        runtime_path=_fixed_runtime_path(parameter_id),
                        multiplier=1.0,
                        mapping_rule="direct_fixed_value",
                    )
                ],
                lineage=_parent_lineage(record),
            )
        )

    fractions = {item.pathway_id: item for item in source_manifest.pathway_fractions}
    fraction_specs = (
        ("caffeine_fraction_n3_paraxanthine", "n3_paraxanthine", "N3 to paraxanthine"),
        ("caffeine_fraction_n1_theobromine", "n1_theobromine", "N1 to theobromine"),
        ("caffeine_fraction_n7_theophylline", "n7_theophylline", "N7 to theophylline"),
    )
    for parameter_id, pathway_id, name in fraction_specs:
        source = fractions[pathway_id]
        parameters.append(
            CaffeineV2ParameterRecord(
                parameter_id=parameter_id,
                name=name,
                role="pathway_fraction",
                origin="observed",
                value=source.fraction,
                unit="fraction",
                calibrate=False,
                lineage=_source_lineage(
                    artifact_id=source.source_artifact_id,
                    source_location=source.source_location,
                    source_value_text=source.source_text,
                ),
            )
        )

    domains = {item.parameter_id: item for item in source_manifest.search_domains}
    clearance_domain = domains["total_cyp1a2_specific_clearance"]
    permeability_domain = domains[
        "specific_intestinal_permeability_transcellular"
    ]
    estimated_common = {
        "origin": "estimated",
        "calibrate": True,
        "calibration_dataset_ids": list(CALIBRATION_DATASETS),
        "objective_function": CALIBRATION_OBJECTIVE,
    }
    parameters.append(
        CaffeineV2ParameterRecord(
            parameter_id="total_cyp1a2_specific_clearance",
            name="Total CYP1A2 specific clearance",
            role="total_metabolic_capacity",
            value=clearance_domain.upper_bound * 0.30,
            unit=clearance_domain.unit,
            lower_bound=clearance_domain.lower_bound,
            upper_bound=clearance_domain.upper_bound,
            runtime_mappings=[
                RuntimeMapping(
                    runtime_path=(
                        "Caffeine-CYP1A2-N3-Paraxanthine|Specific clearance"
                    ),
                    multiplier=fractions["n3_paraxanthine"].fraction,
                    mapping_rule="CL_N3=CL_total*0.839",
                ),
                RuntimeMapping(
                    runtime_path=(
                        "Caffeine-CYP1A2-N1-Theobromine|Specific clearance"
                    ),
                    multiplier=fractions["n1_theobromine"].fraction,
                    mapping_rule="CL_N1=CL_total*0.121",
                ),
                RuntimeMapping(
                    runtime_path=(
                        "Caffeine-CYP1A2-N7-Theophylline|Specific clearance"
                    ),
                    multiplier=fractions["n7_theophylline"].fraction,
                    mapping_rule="CL_N7=CL_total*0.040",
                ),
            ],
            lineage=_source_lineage(
                artifact_id=clearance_domain.source_artifact_id,
                source_location=clearance_domain.source_location,
                source_value_text=(
                    "search domain only: [0, "
                    f"{clearance_domain.upper_bound}] {clearance_domain.unit}"
                ),
            ),
            **estimated_common,
        )
    )
    parameters.append(
        CaffeineV2ParameterRecord(
            parameter_id="specific_intestinal_permeability_transcellular",
            name="Specific intestinal permeability (transcellular)",
            role="intestinal_absorption",
            value=permeability_domain.upper_bound * 0.30,
            unit=permeability_domain.unit,
            lower_bound=permeability_domain.lower_bound,
            upper_bound=permeability_domain.upper_bound,
            runtime_mappings=[
                RuntimeMapping(
                    runtime_path=(
                        "Caffeine|Specific intestinal permeability (transcellular)"
                    ),
                    multiplier=1.0,
                    mapping_rule="direct_estimated_value",
                )
            ],
            lineage=_source_lineage(
                artifact_id=permeability_domain.source_artifact_id,
                source_location=permeability_domain.source_location,
                source_value_text=(
                    "search domain only: [0, "
                    f"{permeability_domain.upper_bound}] {permeability_domain.unit}"
                ),
            ),
            **estimated_common,
        )
    )

    starts = [
        CalibrationStart(
            start_id=index + 1,
            total_cyp1a2_specific_clearance=(
                clearance_domain.upper_bound * clearance_fraction
            ),
            specific_intestinal_permeability_transcellular=(
                permeability_domain.upper_bound * permeability_fraction
            ),
        )
        for index, (clearance_fraction, permeability_fraction) in enumerate(
            (left, right)
            for left in START_FRACTIONS
            for right in START_FRACTIONS
        )
    ]
    ids = [item.parameter_id for item in parameters]
    estimated_ids = {item.parameter_id for item in parameters if item.calibrate}
    fixed_lipophilicity = next(
        item for item in parameters if item.parameter_id == "caffeine_effective_lipophilicity"
    )
    total_mapping = next(
        item
        for item in parameters
        if item.parameter_id == "total_cyp1a2_specific_clearance"
    )
    checks = {
        "source_manifest_checksum_valid": canonical_sha256(
            {
                key: value
                for key, value in source_payload.items()
                if key != "checksum"
            }
        )
        == source_payload["checksum"],
        "dataset_manifest_checksum_valid": canonical_sha256(
            {key: value for key, value in dataset.items() if key != "checksum"}
        )
        == dataset["checksum"],
        "parameter_ids_unique": len(ids) == len(set(ids)),
        "only_two_external_parameters_estimated": estimated_ids
        == {
            "total_cyp1a2_specific_clearance",
            "specific_intestinal_permeability_transcellular",
        },
        "lipophilicity_fixed_at_public_value": (
            not fixed_lipophilicity.calibrate
            and fixed_lipophilicity.origin == "observed"
            and math.isclose(fixed_lipophilicity.value, -0.1)
        ),
        "three_pathway_fractions_sum_one": math.isclose(
            sum(item.multiplier for item in total_mapping.runtime_mappings),
            1.0,
            rel_tol=0,
            abs_tol=1e-12,
        ),
        "twenty_five_deterministic_starts": len(starts) == 25
        and [item.start_id for item in starts] == list(range(1, 26)),
        "calibration_split_only": not (
            set(CALIBRATION_DATASETS)
            & {
                dataset_id
                for dataset_id, split in dataset_ids.items()
                if split != "calibration"
            }
        ),
        "fendt_values_not_imported": all(
            item.decision == "excluded_from_v2_model_input"
            for item in source_manifest.contamination_ledger
        ),
    }
    payload = {
        "schema_version": "1.0.0",
        "builder_version": PARAMETER_BUILDER_VERSION,
        "manifest_version": PARAMETER_MANIFEST_VERSION,
        "model_version": MODEL_VERSION,
        "generated_at": datetime.now(UTC),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_manifest_checksum": source_payload["checksum"],
        "parent_parameter_manifest_checksum": parent["checksum"],
        "dataset_version": DATASET_VERSION,
        "dataset_manifest_checksum": dataset["checksum"],
        "parameters": parameters,
        "calibration_objective": CALIBRATION_OBJECTIVE,
        "calibration_starts": starts,
        "structural_scope": [
            "Parent caffeine concentrations are the only calibrated outputs.",
            "N3, N1, and N7 demethylation fractions are fixed human observations.",
            "The three first-order pathways share one fitted total capacity.",
            "Transcellular intestinal permeability is fitted directly.",
            "Metabolite PK and caffeine DDI qualification are outside this model version.",
            "Holdout and external simulations remain locked until calibration passes.",
        ],
        "checks": checks,
        "accepted": all(checks.values()),
    }
    draft = CaffeineV2ParameterManifest(**payload, checksum="0" * 64)
    manifest = draft.model_copy(
        update={
            "checksum": canonical_sha256(
                draft.model_dump(mode="json", exclude={"checksum"})
            )
        }
    )
    if not manifest.accepted:
        failed = [name for name, passed in checks.items() if not passed]
        raise CaffeineV2ParameterError(
            "Caffeine v2 parameter acceptance failed: " + ", ".join(failed)
        )

    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        raise CaffeineV2ParameterError(f"Stale output exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        write_json_atomic(
            temporary / "caffeine_parameter_manifest.json",
            manifest,
            canonical=False,
        )
        _write_parameter_csv(temporary / "parameter_values.csv", parameters)
        write_json_atomic(
            temporary / "calibration_start_grid.json",
            [item.model_dump(mode="json") for item in starts],
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
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the caffeine v2 pathway/absorption parameter manifest."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = build_caffeine_v2_parameter_manifest(
        source_manifest_path=args.source_manifest,
        dataset_manifest_path=args.dataset_manifest,
        output_dir=args.output_dir,
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
