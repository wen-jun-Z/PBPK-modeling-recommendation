from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from pbpk_kb.blueprints.builder import (
    derive_compartments,
    derive_equation_templates,
)
from pbpk_kb.blueprints.schemas import (
    BlueprintTemplate,
    ParameterSlot,
    StrictBlueprintModel,
    TargetRuntime,
)
from pbpk_kb.blueprints.template_catalog import (
    TemplateCatalog,
    TemplateSkeletonAudit,
    audit_template_skeleton,
    instantiated_processes,
    resolve_template_asset_path,
)
from pbpk_kb.model_assets.execution_plan import write_json_atomic
from pbpk_kb.model_assets.license_policy import (
    resolve_repository_license,
    sha256_file,
)
from pbpk_kb.model_assets.models import LicenseStatus
from pbpk_kb.model_assets.osp_pkml_parser import (
    parse_pkml_simulation,
)


class TemplateRegistrationError(RuntimeError):
    pass


class BlueprintTemplateRegistrationSpec(StrictBlueprintModel):
    template_id: str = Field(min_length=1)
    template_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    target_runtime: TargetRuntime
    source_skeleton_relative_path: str = Field(min_length=1)
    source_project_relative_path: str = Field(min_length=1)
    source_repository: str = Field(min_length=1)
    source_tag: str = Field(min_length=1)
    source_commit: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{40}$",
    )
    source_manifest_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    license_id: str = Field(min_length=1)
    license_relative_path: str = Field(min_length=1)
    species: str = Field(min_length=1)
    skeleton_simulation_id: str = Field(min_length=1)
    skeleton_simulation_name: str = Field(min_length=1)
    compound_semantic_ids: list[str] = Field(min_length=1)
    model_structure: str = Field(pattern=r"^(whole_body|reduced)$")
    supported_objectives: list[str] = Field(min_length=1)
    parameter_slots: list[ParameterSlot] = Field(min_length=1)
    allowed_topology_transformations: list[str] = Field(
        default_factory=list
    )
    validation_scenario_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def source_revision_is_locked(
        self,
    ) -> "BlueprintTemplateRegistrationSpec":
        if self.source_commit is None and self.source_manifest_sha256 is None:
            raise ValueError(
                "Template source requires a Git commit or manifest checksum."
            )
        return self


def register_blueprint_template(
    *,
    spec: BlueprintTemplateRegistrationSpec,
    asset_root: Path,
    source_repository_root: Path,
) -> tuple[BlueprintTemplate, TemplateSkeletonAudit]:
    skeleton_path = resolve_template_asset_path(
        asset_root,
        spec.source_skeleton_relative_path,
    )
    project_path = resolve_template_asset_path(
        asset_root,
        spec.source_project_relative_path,
    )
    license_path = resolve_template_asset_path(
        asset_root,
        spec.license_relative_path,
    )
    for kind, path in (
        ("skeleton", skeleton_path),
        ("project", project_path),
        ("license", license_path),
    ):
        if not path.is_file():
            raise TemplateRegistrationError(
                f"Template {kind} asset is missing: {path}"
            )
    license_decision = resolve_repository_license(
        source_repository_root.resolve()
    )
    if license_decision.status != LicenseStatus.ALLOWED:
        raise TemplateRegistrationError(
            "Template source license is not explicitly allowed: "
            f"{license_decision.reason_code}."
        )
    if license_decision.spdx_id != spec.license_id:
        raise TemplateRegistrationError(
            "Template license identifier does not match the parsed source "
            "license."
        )
    if license_decision.source_checksum != sha256_file(license_path):
        raise TemplateRegistrationError(
            "Template license copy does not match the source repository "
            "license."
        )
    skeleton_sha256 = sha256_file(skeleton_path)
    records = parse_pkml_simulation(
        skeleton_path,
        artifact_relative_path=spec.source_skeleton_relative_path,
        artifact_sha256=skeleton_sha256,
        job_id=f"template-registration:{spec.template_id}",
        project_key=skeleton_sha256[:32],
        simulation_id=spec.skeleton_simulation_id,
        simulation_name=spec.skeleton_simulation_name,
    )
    if records.summary.parse_status != "parsed":
        raise TemplateRegistrationError(
            f"Template PKML parse failed: {records.summary.error}"
        )
    processes = instantiated_processes(records)
    if not processes:
        raise TemplateRegistrationError(
            "Template has no instantiated OSP processes."
        )
    if any(process.process_type == "other" for process in processes):
        raise TemplateRegistrationError(
            "Template has unsupported instantiated process types."
        )
    process_counts: dict[str, int] = {}
    for process in processes:
        process_counts[process.process_type] = (
            process_counts.get(process.process_type, 0) + 1
        )
    species = {item.casefold() for item in records.summary.species}
    if spec.species.casefold() not in species:
        raise TemplateRegistrationError(
            "Registration species is absent from the PKML."
        )
    if not records.summary.routes:
        raise TemplateRegistrationError(
            "Template PKML has no administration route."
        )
    if not records.summary.compounds:
        raise TemplateRegistrationError(
            "Template PKML has no compound identity."
        )
    for slot in spec.parameter_slots:
        if slot.source_checksum != skeleton_sha256:
            raise TemplateRegistrationError(
                f"Parameter slot source checksum mismatch: {slot.slot_id}"
            )
        if not slot.source_location.startswith("PKML:"):
            raise TemplateRegistrationError(
                f"Parameter slot is not grounded in PKML: {slot.slot_id}"
            )
    template = BlueprintTemplate(
        template_id=spec.template_id,
        template_version=spec.template_version,
        target_runtime=spec.target_runtime,
        source_skeleton_relative_path=(
            spec.source_skeleton_relative_path
        ),
        source_skeleton_sha256=skeleton_sha256,
        source_project_relative_path=spec.source_project_relative_path,
        source_project_sha256=sha256_file(project_path),
        source_repository=spec.source_repository,
        source_tag=spec.source_tag,
        source_commit=spec.source_commit,
        source_manifest_sha256=spec.source_manifest_sha256,
        license_id=spec.license_id,
        license_relative_path=spec.license_relative_path,
        license_sha256=sha256_file(license_path),
        license_decision=license_decision,
        species=spec.species,
        skeleton_simulation_id=spec.skeleton_simulation_id,
        skeleton_simulation_name=spec.skeleton_simulation_name,
        skeleton_compound_names=records.summary.compounds,
        skeleton_instantiated_process_count=len(processes),
        skeleton_process_type_counts=dict(sorted(process_counts.items())),
        compound_semantic_ids=spec.compound_semantic_ids,
        model_structure=spec.model_structure,
        supported_objectives=spec.supported_objectives,
        supported_routes=records.summary.routes,
        supported_process_types=sorted(process_counts),
        required_compartment_ids=[
            item.compartment_id for item in derive_compartments(records)
        ],
        parameter_slots=spec.parameter_slots,
        equation_templates=derive_equation_templates(records),
        allowed_topology_transformations=(
            spec.allowed_topology_transformations
        ),
        validation_scenario_ids=spec.validation_scenario_ids,
        validation_status="structural_template_runtime_verified",
    )
    audit = audit_template_skeleton(
        template=template,
        asset_root=asset_root,
    )
    if not audit.accepted:
        raise TemplateRegistrationError(
            "Registered template failed skeleton audit: "
            + ",".join(audit.errors)
        )
    return template, audit


def write_template_catalog(
    *,
    path: Path,
    templates: list[BlueprintTemplate],
) -> None:
    template_ids = [item.template_id for item in templates]
    if len(template_ids) != len(set(template_ids)):
        raise TemplateRegistrationError("Template IDs must be unique.")
    catalog = TemplateCatalog(
        catalog_schema_version="1.0.0",
        templates=sorted(
            templates,
            key=lambda item: (
                item.template_id,
                item.template_version,
            ),
        ),
    )
    write_json_atomic(path, catalog, canonical=True)
