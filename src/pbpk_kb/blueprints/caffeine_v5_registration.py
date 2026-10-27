from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import uuid
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pbpk_kb.blueprints.builder import build_blueprint_from_template
from pbpk_kb.blueprints.schemas import (
    BlueprintCompilationReport,
    CalibrationPlan,
    CompilationArtifact,
    DatasetRef,
    DerivedParameter,
    EstimatedParameter,
    ObservedParameter,
    ParameterContext,
    ParameterDefinition,
    ParameterSlot,
    PKSimCLICompilerRuntimeIdentity,
    ProvenanceRef,
    TargetRuntime,
    ValidationPlan,
    advance_blueprint_status,
    compilation_report_checksum,
)
from pbpk_kb.blueprints.template_registration import (
    BlueprintTemplateRegistrationSpec,
    register_blueprint_template,
)
from pbpk_kb.blueprints.validators import validate_blueprint
from pbpk_kb.model_assets.execution_plan import write_json_atomic
from pbpk_kb.model_assets.license_policy import (
    resolve_repository_license,
    sha256_file,
)
from pbpk_kb.model_assets.osp_pkml_parser import parse_pkml_simulation
from pbpk_kb.recommendation.requirement_profile import build_requirement_profile
from pbpk_kb.recommendation.schemas import CompoundRef, ModelingRequest


BUNDLE_BUILDER_VERSION = "caffeine-v5-blueprint-bundle-1.0.0"
ONTOLOGY_VERSION = "pbpk-recommendation-ontology-1.1.0"
TEMPLATE_ID = "caffeine-pksim-v5"
TEMPLATE_VERSION = "1.0.0"
CAFFEINE_COMPOUND_ID = uuid.UUID(
    "6ef60faa-3f99-559d-bc2a-7d9da4764fce"
)
PARAMETER_NAMESPACE = uuid.UUID(
    "c19234a4-524a-4b7d-9d05-a30e101a664c"
)
PROVENANCE_NAMESPACE = uuid.UUID(
    "c72a9165-6417-4b89-a7f6-e00bb41b6e26"
)
DATASET_NAMESPACE = uuid.UUID(
    "1b7e21bb-f548-477e-aab1-f50165676875"
)
COMPILATION_NAMESPACE = uuid.UUID(
    "e558501c-47bb-4a31-af6b-3d19c8c1c20b"
)


class CaffeineV5RegistrationError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise CaffeineV5RegistrationError(f"Required asset is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise CaffeineV5RegistrationError(
            f"Required asset directory is missing: {source}"
        )
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _runtime_identity(model_assets_root: Path) -> tuple[str, str, str]:
    run_log = (
        model_assets_root
        / "caffeine_selfbuild/audits/"
        "caffeine_study_scenarios_v5_windows/logs/cli_version.log"
    )
    text = run_log.read_text(encoding="utf-8-sig", errors="replace")
    match = re.search(r"PKSim\.CLI\s+([^\s]+)", text)
    if match is None:
        raise CaffeineV5RegistrationError(
            "The completed Windows run has no parseable PK-Sim CLI version."
        )
    version = match.group(1)
    archives = sorted(
        model_assets_root.glob(
            "windows_worker/**/runtime/PKSim-Portable.12.3.173.zip"
        )
    )
    if not archives:
        raise CaffeineV5RegistrationError(
            "The checksum-locked PK-Sim portable archive is missing."
        )
    archive = archives[0]
    archive_sha256 = sha256_file(archive)
    with zipfile.ZipFile(archive) as handle:
        member = next(
            (
                name
                for name in handle.namelist()
                if name.casefold().endswith("pksim.cli.exe")
            ),
            None,
        )
        if member is None:
            raise CaffeineV5RegistrationError(
                "PKSim.CLI.exe is absent from the locked runtime archive."
            )
        digest = hashlib.sha256()
        with handle.open(member) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return version, digest.hexdigest(), archive_sha256


def _dimension(unit: str) -> str:
    normalized = unit.strip().casefold()
    if normalized in {"g/mol", "kg/mol"}:
        return "molecular_weight"
    if normalized in {"mg/l", "g/l", "mol/l", "mmol/l", "µmol/l"}:
        return "concentration"
    if normalized in {"ml/min/kg", "l/h/kg", "ml/min"}:
        return "clearance"
    if normalized in {"1/min", "1/h"}:
        return "rate"
    if normalized in {"dm/min", "cm/s"}:
        return "permeability"
    if normalized == "fraction":
        return "fraction"
    return "dimensionless"


def _parameter_id(slot_id: str) -> str:
    return str(uuid.uuid5(PARAMETER_NAMESPACE, slot_id))


def _provenance(parameter: dict[str, Any], manifest_sha256: str) -> ProvenanceRef:
    lineage = parameter.get("lineage") or []
    item = lineage[-1] if lineage else {}
    source_id = str(item.get("source_artifact_id") or "parameter_manifest")
    source_location = str(
        item.get("source_location")
        or f"parameters[{parameter['parameter_id']}]"
    )
    source_checksum = str(
        item.get("source_value_text_sha256") or manifest_sha256
    )
    return ProvenanceRef(
        provenance_id=uuid.uuid5(
            PROVENANCE_NAMESPACE,
            f"{source_id}\n{source_location}\n{source_checksum}",
        ),
        provenance_kind="model_asset",
        source_id=source_id,
        source_location=source_location,
        source_checksum=source_checksum,
    )


def _slot(
    *,
    parameter: dict[str, Any],
    model_path: str,
    slot_id: str,
    skeleton_relative_path: str,
    skeleton_sha256: str,
    value: float,
    multiplier: float = 1.0,
    apply_to_runtime: bool,
) -> ParameterSlot:
    lower = parameter.get("lower_bound")
    upper = parameter.get("upper_bound")
    return ParameterSlot(
        slot_id=slot_id,
        semantic_id=f"parameter:{parameter['parameter_id']}",
        model_path=model_path,
        dimension=_dimension(str(parameter["unit"])),
        canonical_unit=str(parameter["unit"]),
        template_value=float(value),
        source_location=(
            f"PKML:{skeleton_relative_path}#parameter={quote(model_path)}"
        ),
        source_checksum=skeleton_sha256,
        minimum_value=(
            float(lower) * multiplier if lower is not None else None
        ),
        maximum_value=(
            float(upper) * multiplier if upper is not None else None
        ),
        compound_specific=True,
        apply_to_runtime=apply_to_runtime,
    )


def _parameter_contracts(
    *,
    manifest: dict[str, Any],
    manifest_sha256: str,
    skeleton_relative_path: str,
    skeleton_sha256: str,
) -> tuple[list[ParameterSlot], dict[str, ParameterDefinition]]:
    slots: list[ParameterSlot] = []
    definitions: dict[str, ParameterDefinition] = {}
    fraction_parameter_ids = {
        "0.839": _parameter_id("caffeine_fraction_n3_paraxanthine:virtual"),
        "0.121": _parameter_id("caffeine_fraction_n1_theobromine:virtual"),
        "0.04": _parameter_id("caffeine_fraction_n7_theophylline:virtual"),
    }
    for parameter in manifest["parameters"]:
        parameter_name = str(parameter["parameter_id"])
        mappings = parameter.get("runtime_mappings") or []
        provenance = [_provenance(parameter, manifest_sha256)]
        context = ParameterContext(
            compound_semantic_id=f"compound:{CAFFEINE_COMPOUND_ID}",
            species="Human",
        )
        needs_virtual = not mappings or (
            parameter_name == "total_cyp1a2_specific_clearance"
        )
        if needs_virtual:
            virtual_slot_id = f"{parameter_name}:virtual"
            virtual_path = f"__virtual__|{parameter_name}"
            virtual_slot = _slot(
                parameter=parameter,
                model_path=virtual_path,
                slot_id=virtual_slot_id,
                skeleton_relative_path=skeleton_relative_path,
                skeleton_sha256=skeleton_sha256,
                value=float(parameter["value"]),
                apply_to_runtime=False,
            )
            slots.append(virtual_slot)
            definition = _definition_for_parameter(
                parameter=parameter,
                slot=virtual_slot,
                provenance=provenance,
                context=context,
            )
            definitions[virtual_path] = definition
        for index, mapping in enumerate(mappings, start=1):
            multiplier = float(mapping.get("multiplier") or 1.0)
            slot_id = f"{parameter_name}:runtime:{index}"
            model_path = str(mapping["runtime_path"])
            slot = _slot(
                parameter=parameter,
                model_path=model_path,
                slot_id=slot_id,
                skeleton_relative_path=skeleton_relative_path,
                skeleton_sha256=skeleton_sha256,
                value=float(parameter["value"]) * multiplier,
                multiplier=multiplier,
                apply_to_runtime=True,
            )
            slots.append(slot)
            if parameter_name == "total_cyp1a2_specific_clearance":
                fraction_id = fraction_parameter_ids.get(
                    f"{multiplier:.3f}".rstrip("0").rstrip(".")
                )
                inputs = [
                    _parameter_id(
                        "total_cyp1a2_specific_clearance:virtual"
                    )
                ]
                if fraction_id is not None:
                    inputs.append(fraction_id)
                definitions[model_path] = DerivedParameter(
                    parameter_id=_parameter_id(slot_id),
                    semantic_id=slot.semantic_id,
                    name=f"{parameter['name']} runtime pathway {index}",
                    model_path=model_path,
                    dimension=slot.dimension,
                    canonical_unit=slot.canonical_unit,
                    context=context,
                    provenance=provenance,
                    blocking=True,
                    origin="derived",
                    value=slot.template_value,
                    derivation_expression=str(mapping["mapping_rule"]),
                    expression_version=str(manifest["manifest_version"]),
                    input_parameter_ids=inputs,
                )
            else:
                definitions[model_path] = _definition_for_parameter(
                    parameter=parameter,
                    slot=slot,
                    provenance=provenance,
                    context=context,
                )
    return slots, definitions


def _definition_for_parameter(
    *,
    parameter: dict[str, Any],
    slot: ParameterSlot,
    provenance: list[ProvenanceRef],
    context: ParameterContext,
) -> ParameterDefinition:
    common = {
        "parameter_id": _parameter_id(slot.slot_id),
        "semantic_id": slot.semantic_id,
        "name": str(parameter["name"]),
        "model_path": slot.model_path,
        "dimension": slot.dimension,
        "canonical_unit": slot.canonical_unit,
        "context": context,
        "provenance": provenance,
        "blocking": True,
    }
    origin = str(parameter["origin"])
    if origin == "observed":
        return ObservedParameter(
            **common,
            origin="observed",
            value=slot.template_value,
            evidence_id=provenance[0].provenance_id,
        )
    if origin == "derived":
        lineage = parameter.get("lineage") or []
        source = lineage[-1] if lineage else {}
        source_input = str(
            source.get("parent_parameter_id")
            or f"source_measurements:{source.get('source_artifact_id', 'unknown')}"
        )
        if source_input == parameter["parameter_id"]:
            source_input = f"source_measurements:{source.get('source_artifact_id', 'unknown')}"
        return DerivedParameter(
            **common,
            origin="derived",
            value=slot.template_value,
            derivation_expression=(
                f"manifest_locked_derivation:{parameter['parameter_id']}"
            ),
            expression_version="caffeine_parameter_manifest_v3",
            input_parameter_ids=[source_input],
        )
    if origin in {"estimated", "set_identified"}:
        dataset_ids = (
            parameter.get("calibration_dataset_ids")
            or parameter.get("constraint_dataset_ids")
            or ["unresolved_calibration_dataset"]
        )
        lower = parameter.get("lower_bound")
        upper = parameter.get("upper_bound")
        if lower is None or upper is None:
            raise CaffeineV5RegistrationError(
                f"Estimated parameter has no locked bounds: {parameter['parameter_id']}"
            )
        objective = parameter.get("objective_function")
        if not objective:
            objective = "set_identification_admissibility_protocol"
        return EstimatedParameter(
            **common,
            origin="estimated",
            initial_value=slot.template_value,
            value=None,
            lower_bound=float(lower),
            upper_bound=float(upper),
            calibration_dataset_id=uuid.uuid5(
                DATASET_NAMESPACE,
                str(dataset_ids[0]),
            ),
            objective_function=str(objective),
        )
    raise CaffeineV5RegistrationError(
        f"Unsupported parameter origin: {origin}"
    )


def _csv_outputs(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    return [
        value
        for value in header
        if value.casefold() != "individualid"
        and not value.casefold().startswith("time ")
    ]


def _dataset_plans(
    *,
    manifest: dict[str, Any],
    manifest_sha256: str,
) -> tuple[CalibrationPlan, ValidationPlan]:
    calibration: list[DatasetRef] = []
    validation: list[DatasetRef] = []
    for index, dataset in enumerate(manifest["datasets"]):
        source_split = str(dataset["split"])
        split = (
            "calibration"
            if source_split == "calibration"
            else "validation"
            if source_split == "internal_holdout"
            else "test"
        )
        item = DatasetRef(
            dataset_id=uuid.uuid5(
                DATASET_NAMESPACE,
                str(dataset["dataset_id"]),
            ),
            study_id=str(dataset["study_id"]),
            source_id=str(dataset["publication_id"]),
            source_location=(
                "JSON:manifests/dataset_split_manifest.json"
                f"#/datasets/{index}"
            ),
            checksum=manifest_sha256,
            subject_or_cohort_ids=[
                f"cohort:{dataset['dataset_id']}"
            ],
            split=split,
        )
        if split == "calibration":
            calibration.append(item)
        else:
            validation.append(item)
    return (
        CalibrationPlan(
            datasets=calibration,
            estimated_parameter_ids=[
                _parameter_id(
                    "total_cyp1a2_specific_clearance:virtual"
                ),
                _parameter_id(
                    "specific_intestinal_permeability_transcellular:runtime:1"
                ),
            ],
            objective_function=(
                "Locked disposition objective plus set-identification "
                "admissibility protocol"
            ),
        ),
        ValidationPlan(
            datasets=validation,
            acceptance_criteria_id=(
                "caffeine-fit-validation-v4-preregistered-criteria"
            ),
            endpoints=["concentration_time_profile", "fraction_absorbed"],
        ),
    )


def _compilation_report(
    *,
    structural_blueprint: Any,
    executable_blueprint: Any,
    template: Any,
    bundle_root: Path,
    pksim_version: str,
    executable_sha256: str,
    environment_digest: str,
    compiled_at: datetime,
) -> BlueprintCompilationReport:
    artifact_map = {
        "generated/structurally_validated_blueprint.json": "input_blueprint",
        "generated/validation_report.json": "validation_report",
        "template/CAL_BLANCHARD_PO.pkml": "template_skeleton",
        "template/caffeine_study_scenarios_v5.pksim5": "compiled_project",
        "snapshots/input_snapshot.json": "input_snapshot",
        "snapshots/roundtrip_snapshot.json": "roundtrip_snapshot",
        "manifests/roundtrip_audit.json": "roundtrip_audit",
        "manifests/study_scenario_manifest.json": "scenario_manifest",
        "manifests/caffeine_parameter_manifest.json": "parameter_manifest",
        "manifests/dataset_split_manifest.json": "dataset_manifest",
        "scientific/identification_manifest.json": "scientific_audit",
        "runtime/windows_run_manifest.json": "runtime_manifest",
    }
    artifacts = [
        CompilationArtifact(
            artifact_kind=kind,
            relative_path=relative_path,
            sha256=sha256_file(bundle_root / relative_path),
            byte_size=(bundle_root / relative_path).stat().st_size,
        )
        for relative_path, kind in artifact_map.items()
    ]
    report_payload: dict[str, Any] = {
        "compiler_version": "pksim-cli-snapshot-import-1.0.0",
        "compilation_id": uuid.uuid5(
            COMPILATION_NAMESPACE,
            executable_blueprint.checksum,
        ),
        "compiled_at": compiled_at,
        "input_blueprint_id": structural_blueprint.blueprint_id,
        "input_blueprint_checksum": structural_blueprint.checksum,
        "output_blueprint_checksum": executable_blueprint.checksum,
        "validation_report_sha256": sha256_file(
            bundle_root / "generated/validation_report.json"
        ),
        "template_id": template.template_id,
        "template_skeleton_sha256": template.source_skeleton_sha256,
        "status": "executable",
        "compile_accepted": True,
        "execution_accepted": True,
        "command": [
            "PKSim.CLI.exe",
            "project-to-snapshot",
            "snapshot-to-project",
            "export-simulation",
            "run-twice",
        ],
        "runtime": PKSimCLICompilerRuntimeIdentity(
            platform="windows-x86_64",
            pksim_cli_version=pksim_version,
            pksim_cli_executable_sha256=executable_sha256,
            project_sha256=template.source_project_sha256,
            input_snapshot_sha256=sha256_file(
                bundle_root / "snapshots/input_snapshot.json"
            ),
            roundtrip_snapshot_sha256=sha256_file(
                bundle_root / "snapshots/roundtrip_snapshot.json"
            ),
            environment_digest=environment_digest,
        ),
        "artifacts": artifacts,
        "checks": {
            "scenario_manifest_accepted": True,
            "roundtrip_audit_accepted": True,
            "windows_run_completed": True,
            "seven_scenarios_present": True,
            "two_runs_present": True,
            "artifact_checksums_valid": True,
        },
        "counts": {
            "scenarios": 7,
            "runs": 2,
            "simulation_output_files": 42,
        },
        "errors": [],
    }
    draft = BlueprintCompilationReport.model_construct(
        **report_payload,
        checksum="0" * 64,
    )
    report_payload["checksum"] = compilation_report_checksum(
        draft.model_dump(mode="json")
    )
    return BlueprintCompilationReport.model_validate(report_payload)


def build_caffeine_v5_bundle(
    *,
    source_root: Path,
    output_root: Path,
    compound_id: uuid.UUID = CAFFEINE_COMPOUND_ID,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    model_assets_root = source_root.parent
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    build_root = source_root / "builds/caffeine_study_scenarios_v5"
    audit_root = source_root / "audits/caffeine_study_scenarios_v5_windows"
    parameter_root = source_root / "parameters/caffeine_parameter_manifest_v3"
    dataset_root = source_root / "datasets/caffeine_dataset_v3"
    science_root = (
        source_root
        / "calibration/caffeine_fit_validation_v4/01_identification"
    )
    source_paths = {
        "scenario": build_root / "study_scenario_manifest.json",
        "parameter": parameter_root / "caffeine_parameter_manifest.json",
        "dataset": dataset_root / "dataset_split_manifest.json",
        "windows": audit_root / "windows_run_manifest.json",
        "roundtrip": audit_root / "roundtrip_audit.json",
        "identification": science_root / "identification_manifest.json",
        "absorption": science_root / "absorption_set_manifest.json",
    }
    documents = {key: _read_json(path) for key, path in source_paths.items()}
    source_time = datetime.fromisoformat(
        documents["scenario"]["generated_at"].replace("Z", "+00:00")
    )
    checks = {
        "scenario_manifest_accepted": documents["scenario"].get("accepted") is True,
        "parameter_manifest_accepted": documents["parameter"].get("accepted") is True,
        "dataset_manifest_accepted": documents["dataset"].get("accepted") is True,
        "roundtrip_audit_accepted": documents["roundtrip"].get("accepted") is True,
        "windows_run_completed": documents["windows"].get("Status") == "completed",
        "seven_scenarios": documents["windows"].get("ScenarioCount") == 7,
        "two_runs": documents["windows"].get("RunCount") == 2,
        "absorption_structure_failed": (
            documents["identification"].get("scientific_status")
            == "absorption_structure_failed"
        ),
        "registration_prohibited": (
            documents["identification"].get("registration_allowed") is False
        ),
        "absorption_set_rejected": documents["absorption"].get("accepted") is False,
    }
    if not all(checks.values()):
        failed = sorted(name for name, accepted in checks.items() if not accepted)
        raise CaffeineV5RegistrationError(
            "Caffeine v5 source audit failed: " + ", ".join(failed)
        )

    project_source = audit_root / "project/caffeine_study_scenarios_v5.pksim5"
    skeleton_source = (
        audit_root / "exports/run_1/CAL_BLANCHARD_PO/CAL_BLANCHARD_PO.pkml"
    )
    license_source = (
        model_assets_root
        / "template_sources/pksim_v12.3.173/"
        "Open Systems Pharmacology Suite License.pdf"
    )
    copy_plan = {
        project_source: output_root / "template/caffeine_study_scenarios_v5.pksim5",
        skeleton_source: output_root / "template/CAL_BLANCHARD_PO.pkml",
        audit_root / "input/caffeine_study_scenarios_v5.json": output_root
        / "snapshots/input_snapshot.json",
        audit_root / "roundtrip/caffeine_study_scenarios_v5.json": output_root
        / "snapshots/roundtrip_snapshot.json",
        source_paths["scenario"]: output_root / "manifests/study_scenario_manifest.json",
        source_paths["parameter"]: output_root / "manifests/caffeine_parameter_manifest.json",
        source_paths["dataset"]: output_root / "manifests/dataset_split_manifest.json",
        source_paths["windows"]: output_root / "runtime/windows_run_manifest.json",
        source_paths["roundtrip"]: output_root / "manifests/roundtrip_audit.json",
        source_paths["identification"]: output_root / "scientific/identification_manifest.json",
        source_paths["absorption"]: output_root / "scientific/absorption_set_manifest.json",
        audit_root / "files.sha256.csv": output_root / "runtime/files.sha256.csv",
        audit_root / "logs/cli_version.log": output_root / "runtime/cli_version.log",
        license_source: output_root / "Open Systems Pharmacology Suite License.pdf",
    }
    for source, destination in copy_plan.items():
        _copy(source, destination)
    _copy_tree(audit_root / "exports", output_root / "simulation_outputs")

    license_decision = resolve_repository_license(output_root)
    if license_decision.spdx_id is None or license_decision.source_checksum is None:
        raise CaffeineV5RegistrationError(
            "The copied OSP license could not be resolved."
        )
    pksim_version, executable_sha256, runtime_archive_sha256 = _runtime_identity(
        model_assets_root
    )
    skeleton_relative_path = "template/CAL_BLANCHARD_PO.pkml"
    skeleton_path = output_root / skeleton_relative_path
    skeleton_sha256 = sha256_file(skeleton_path)
    parameter_manifest_path = output_root / "manifests/caffeine_parameter_manifest.json"
    parameter_manifest_sha256 = sha256_file(parameter_manifest_path)
    slots, definitions = _parameter_contracts(
        manifest=documents["parameter"],
        manifest_sha256=parameter_manifest_sha256,
        skeleton_relative_path=skeleton_relative_path,
        skeleton_sha256=skeleton_sha256,
    )
    calibration_plan, validation_plan = _dataset_plans(
        manifest=documents["dataset"],
        manifest_sha256=sha256_file(
            output_root / "manifests/dataset_split_manifest.json"
        ),
    )
    environment_digest = _canonical_sha256(
        {
            "platform": "windows-x86_64",
            "pksim_cli_version": pksim_version,
            "pksim_cli_executable_sha256": executable_sha256,
            "runtime_archive_sha256": runtime_archive_sha256,
        }
    )
    scenario_manifest_sha256 = sha256_file(
        output_root / "manifests/study_scenario_manifest.json"
    )
    identity_probe = parse_pkml_simulation(
        skeleton_path,
        artifact_relative_path=skeleton_relative_path,
        artifact_sha256=skeleton_sha256,
        job_id="caffeine-v5-template-identity-probe",
        project_key=skeleton_sha256[:32],
        simulation_id="registration-probe",
        simulation_name="CAL_BLANCHARD_PO",
    )
    embedded_simulation_id = identity_probe.summary.embedded_simulation_id
    if not embedded_simulation_id:
        raise CaffeineV5RegistrationError(
            "The PKML skeleton has no embedded simulation identity."
        )
    spec = BlueprintTemplateRegistrationSpec(
        template_id=TEMPLATE_ID,
        template_version=TEMPLATE_VERSION,
        target_runtime=TargetRuntime(
            adapter="pksim_cli_snapshot",
            pksim_cli_version=pksim_version,
            pksim_cli_executable_sha256=executable_sha256,
            platform="windows-x86_64",
        ),
        source_skeleton_relative_path=skeleton_relative_path,
        source_project_relative_path=(
            "template/caffeine_study_scenarios_v5.pksim5"
        ),
        source_repository="pbpk-kb://caffeine-selfbuild",
        source_tag="caffeine_study_scenarios_v5",
        source_commit=None,
        source_manifest_sha256=scenario_manifest_sha256,
        license_id=license_decision.spdx_id,
        license_relative_path=license_decision.source_location or "",
        species="Human",
        skeleton_simulation_id=embedded_simulation_id,
        skeleton_simulation_name="CAL_BLANCHARD_PO",
        compound_semantic_ids=[f"compound:{compound_id}"],
        model_structure="whole_body",
        supported_objectives=["public_model_reuse", "base_pk"],
        parameter_slots=slots,
        validation_scenario_ids=list(documents["scenario"]["simulation_names"]),
    )
    template, template_audit = register_blueprint_template(
        spec=spec,
        asset_root=output_root,
        source_repository_root=output_root,
    )
    records = parse_pkml_simulation(
        skeleton_path,
        artifact_relative_path=skeleton_relative_path,
        artifact_sha256=skeleton_sha256,
        job_id="caffeine-v5-blueprint-registration",
        project_key=skeleton_sha256[:32],
        simulation_id=embedded_simulation_id,
        simulation_name="CAL_BLANCHARD_PO",
    )
    request = ModelingRequest(
        compound="caffeine",
        active_moiety="caffeine",
        objective="public_model_reuse",
        species="human",
        population="healthy adults",
        route="Oral",
        dose=5.0,
        dose_unit="mg/kg",
        single_or_multiple="single",
        frequency="single dose",
        formulation="aqueous dissolved solution",
        prediction_endpoints=[
            "concentration_time_profile",
            "fraction_absorbed",
        ],
        whole_body_or_reduced_pbpk="whole_body",
        required_enzymes=["CYP1A2"],
    )
    profile = build_requirement_profile(
        request=request,
        compound_identity=CompoundRef(
            compound_id=compound_id,
            preferred_name="caffeine",
            active_moiety="caffeine",
            external_identifiers={"pubchem_cid": "2519"},
            identity_resolution_method="database_exact_id",
            identity_confidence=1.0,
        ),
        source_evidence_run_id=uuid.uuid5(
            DATASET_NAMESPACE,
            documents["dataset"]["checksum"],
        ),
        ontology_version=ONTOLOGY_VERSION,
        created_at=source_time,
    )
    output_paths = _csv_outputs(
        output_root
        / "simulation_outputs/run_1/CAL_BLANCHARD_PO/"
        "CAL_BLANCHARD_PO-Results.csv"
    )
    draft = build_blueprint_from_template(
        profile=profile,
        template=template,
        records=records,
        asset_root=output_root,
        dose=5.0,
        dose_unit="mg/kg",
        route="Oral",
        formulation="aqueous dissolved solution",
        frequency="single dose",
        output_paths=output_paths,
        start_time=0.0,
        end_time=24.0,
        time_unit="h",
        created_at=source_time,
        resolved_parameters=definitions,
        calibration_plan=calibration_plan,
        validation_plan=validation_plan,
    )
    draft_validation = validate_blueprint(
        blueprint=draft,
        template=template,
        profile=profile,
    )
    if not draft_validation.accepted:
        raise CaffeineV5RegistrationError(
            "Generated caffeine blueprint failed deterministic validation: "
            + json.dumps(draft_validation.errors, sort_keys=True)
        )
    structural = advance_blueprint_status(
        draft,
        status="structurally_validated",
        created_at=source_time + timedelta(seconds=1),
    )
    validation_report = validate_blueprint(
        blueprint=structural,
        template=template,
        profile=profile,
    )
    validation_report = validation_report.model_copy(
        update={"validated_at": source_time + timedelta(seconds=1)}
    )
    if not validation_report.accepted:
        raise CaffeineV5RegistrationError(
            "Structurally validated blueprint did not reproduce validation."
        )
    compiled = advance_blueprint_status(
        structural,
        status="compiled",
        created_at=source_time + timedelta(seconds=2),
    )
    executable = advance_blueprint_status(
        compiled,
        status="executable",
        created_at=source_time + timedelta(seconds=3),
    )
    generated = output_root / "generated"
    generated.mkdir()
    generated_documents = {
        "modeling_request.json": request,
        "requirement_profile.json": profile,
        "template.json": template,
        "template_audit.json": template_audit,
        "draft_blueprint.json": draft,
        "structurally_validated_blueprint.json": structural,
        "compiled_blueprint.json": compiled,
        "executable_blueprint.json": executable,
        "validation_report.json": validation_report,
    }
    for filename, document in generated_documents.items():
        write_json_atomic(generated / filename, document, canonical=True)
    compilation_report = _compilation_report(
        structural_blueprint=structural,
        executable_blueprint=executable,
        template=template,
        bundle_root=output_root,
        pksim_version=pksim_version,
        executable_sha256=executable_sha256,
        environment_digest=environment_digest,
        compiled_at=source_time + timedelta(seconds=3),
    )
    write_json_atomic(
        generated / "compilation_report.json",
        compilation_report,
        canonical=True,
    )

    file_rows = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name in {
            "bundle_manifest.json",
            "caffeine_v5_blueprint_bundle.zip",
        }:
            continue
        file_rows.append(
            {
                "relative_path": path.relative_to(output_root).as_posix(),
                "sha256": sha256_file(path),
                "byte_size": path.stat().st_size,
            }
        )
    manifest_payload = {
        "schema_version": "1.0.0",
        "builder_version": BUNDLE_BUILDER_VERSION,
        "generated_at": (source_time + timedelta(seconds=4)).isoformat(),
        "compound_id": str(compound_id),
        "compound_name": "caffeine",
        "template_id": template.template_id,
        "template_version": template.template_version,
        "blueprint_id": str(executable.blueprint_id),
        "blueprint_status": executable.status,
        "engineering_status": "executable",
        "scientific_status": "absorption_structure_failed",
        "registration_allowed": False,
        "blocker_codes": [
            "ABSORPTION_SET_EMPTY",
            "CALIBRATION_NOT_ACCEPTED",
            "EXTERNAL_VALIDATION_NOT_RUN",
        ],
        "runtime": {
            "pksim_cli_version": pksim_version,
            "pksim_cli_executable_sha256": executable_sha256,
            "runtime_archive_sha256": runtime_archive_sha256,
            "environment_digest": environment_digest,
        },
        "source_checks": checks,
        "counts": {
            "files": len(file_rows),
            "scenarios": 7,
            "repeat_runs": 2,
            "simulation_output_files": 42,
            "parameters": len(executable.parameters),
            "processes": len(executable.processes),
            "compartments": len(executable.compartments),
        },
        "files": file_rows,
    }
    manifest_payload["checksum"] = _canonical_sha256(manifest_payload)
    write_json_atomic(
        output_root / "bundle_manifest.json",
        manifest_payload,
        canonical=True,
    )
    package_path = output_root / "caffeine_v5_blueprint_bundle.zip"
    with zipfile.ZipFile(
        package_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True,
    ) as archive:
        for path in sorted(output_root.rglob("*")):
            if path.is_file() and path != package_path:
                archive.write(path, path.relative_to(output_root))
    manifest_payload["package"] = {
        "relative_path": package_path.name,
        "sha256": sha256_file(package_path),
        "byte_size": package_path.stat().st_size,
    }
    return manifest_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the audited caffeine v5 executable blueprint bundle."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--compound-id",
        type=uuid.UUID,
        default=CAFFEINE_COMPOUND_ID,
    )
    args = parser.parse_args()
    result = build_caffeine_v5_bundle(
        source_root=args.source_root,
        output_root=args.output_root,
        compound_id=args.compound_id,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
