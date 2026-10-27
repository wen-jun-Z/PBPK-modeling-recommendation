from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath

import yaml
from pydantic import Field

from pbpk_kb.blueprints.schemas import (
    BlueprintTemplate,
    StrictBlueprintModel,
)
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.models import (
    LicenseStatus,
    OSPPKMLProcessRecord,
)
from pbpk_kb.model_assets.osp_pkml_parser import (
    OSPPKMLParsedRecords,
    parse_pkml_simulation,
)
from pbpk_kb.recommendation.schemas import RequirementProfile


class TemplateCatalogError(RuntimeError):
    pass


class TemplateCatalog(StrictBlueprintModel):
    catalog_schema_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    templates: list[BlueprintTemplate] = Field(min_length=1)


class TemplateSkeletonAudit(StrictBlueprintModel):
    template_id: str
    skeleton_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parse_accepted: bool
    simulation_identity_accepted: bool
    compound_identity_accepted: bool
    species_accepted: bool
    route_accepted: bool
    process_inventory_accepted: bool
    participant_resolution_accepted: bool
    instantiated_process_count: int = Field(ge=0)
    process_type_counts: dict[str, int]
    errors: list[str] = Field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return all(
            (
                self.parse_accepted,
                self.simulation_identity_accepted,
                self.compound_identity_accepted,
                self.species_accepted,
                self.route_accepted,
                self.process_inventory_accepted,
                self.participant_resolution_accepted,
            )
        )


def instantiated_processes(
    records: OSPPKMLParsedRecords,
) -> list[OSPPKMLProcessRecord]:
    return [
        process
        for process in records.processes
        if "/Model[" in process.source.source_location
        and process.element_type in {"Reaction", "Transport"}
    ]


def audit_template_skeleton(
    *,
    template: BlueprintTemplate,
    asset_root: Path,
) -> TemplateSkeletonAudit:
    skeleton_path = resolve_template_asset_path(
        asset_root,
        template.source_skeleton_relative_path,
    )
    actual_sha256 = sha256_file(skeleton_path)
    records = parse_pkml_simulation(
        skeleton_path,
        artifact_relative_path=template.source_skeleton_relative_path,
        artifact_sha256=actual_sha256,
        job_id=f"template-audit:{template.template_id}",
        project_key=actual_sha256[:32],
        simulation_id=template.skeleton_simulation_id,
        simulation_name=template.skeleton_simulation_name,
    )
    processes = instantiated_processes(records)
    process_counts = Counter(
        process.process_type for process in processes
    )
    errors: list[str] = []
    parse_accepted = records.summary.parse_status == "parsed"
    simulation_identity_accepted = (
        records.summary.embedded_simulation_id
        == template.skeleton_simulation_id
        and records.summary.simulation_name
        == template.skeleton_simulation_name
    )
    compound_identity_accepted = {
        item.casefold() for item in records.summary.compounds
    } == {
        item.casefold() for item in template.skeleton_compound_names
    }
    species_accepted = template.species.casefold() in {
        item.casefold() for item in records.summary.species
    }
    route_accepted = set(map(str.casefold, template.supported_routes)).issubset(
        set(map(str.casefold, records.summary.routes))
    )
    process_inventory_accepted = (
        len(processes) == template.skeleton_instantiated_process_count
        and dict(sorted(process_counts.items()))
        == dict(sorted(template.skeleton_process_type_counts.items()))
        and set(template.supported_process_types).issubset(process_counts)
    )
    participant_resolution_accepted = all(
        not process.unresolved_partner_identifiers for process in processes
    )
    checks = {
        "PKML_PARSE_FAILED": parse_accepted,
        "SIMULATION_IDENTITY_MISMATCH": simulation_identity_accepted,
        "COMPOUND_IDENTITY_MISMATCH": compound_identity_accepted,
        "SPECIES_MISMATCH": species_accepted,
        "ROUTE_MISMATCH": route_accepted,
        "PROCESS_INVENTORY_MISMATCH": process_inventory_accepted,
        "UNRESOLVED_PROCESS_PARTICIPANTS": participant_resolution_accepted,
    }
    errors.extend(reason for reason, accepted in checks.items() if not accepted)
    return TemplateSkeletonAudit(
        template_id=template.template_id,
        skeleton_sha256=actual_sha256,
        parse_accepted=parse_accepted,
        simulation_identity_accepted=simulation_identity_accepted,
        compound_identity_accepted=compound_identity_accepted,
        species_accepted=species_accepted,
        route_accepted=route_accepted,
        process_inventory_accepted=process_inventory_accepted,
        participant_resolution_accepted=participant_resolution_accepted,
        instantiated_process_count=len(processes),
        process_type_counts=dict(sorted(process_counts.items())),
        errors=errors,
    )


def resolve_template_asset_path(
    asset_root: Path,
    relative_path: str,
) -> Path:
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
    ):
        raise TemplateCatalogError(
            f"Unsafe template asset path: {relative_path!r}"
        )
    path = asset_root.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(asset_root.resolve())
    except ValueError as exc:
        raise TemplateCatalogError(
            f"Template asset escapes root: {relative_path!r}"
        ) from exc
    return path


def load_template_catalog(
    *,
    config_path: Path,
    asset_root: Path,
) -> TemplateCatalog:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    catalog = TemplateCatalog.model_validate(payload)
    template_ids = [item.template_id for item in catalog.templates]
    if len(template_ids) != len(set(template_ids)):
        raise TemplateCatalogError("Template IDs must be unique.")
    for template in catalog.templates:
        decision = template.license_decision
        if decision.status != LicenseStatus.ALLOWED:
            raise TemplateCatalogError(
                "Template source license is not explicitly allowed: "
                f"{template.template_id}"
            )
        if decision.spdx_id != template.license_id:
            raise TemplateCatalogError(
                "Template license identifier drift: "
                f"{template.template_id}"
            )
        if decision.source_checksum != template.license_sha256:
            raise TemplateCatalogError(
                "Template license decision checksum drift: "
                f"{template.template_id}"
            )
        expected_files = [
            (
                template.source_skeleton_relative_path,
                template.source_skeleton_sha256,
            ),
            (
                template.source_project_relative_path,
                template.source_project_sha256,
            ),
            (
                template.license_relative_path,
                template.license_sha256,
            ),
        ]
        for relative_path, expected_sha256 in expected_files:
            path = resolve_template_asset_path(asset_root, relative_path)
            if not path.is_file():
                raise TemplateCatalogError(
                    f"Template asset is missing: {relative_path}"
                )
            if sha256_file(path) != expected_sha256:
                raise TemplateCatalogError(
                    f"Template asset checksum drift: {relative_path}"
                )
        audit = audit_template_skeleton(
            template=template,
            asset_root=asset_root,
        )
        if not audit.accepted:
            raise TemplateCatalogError(
                "Template skeleton audit failed: "
                + ",".join(audit.errors)
            )
    return catalog


def required_process_type_options(
    profile: RequirementProfile,
) -> list[set[str]]:
    mapping = {
        "absorption": {"absorption"},
        "distribution": {"distribution"},
        "metabolism": {"metabolism"},
        "transport": {"transport"},
        "excretion": {"renal_excretion", "biliary_excretion"},
        "renal_excretion": {"renal_excretion"},
        "biliary_excretion": {"biliary_excretion"},
        "ddi": {"ddi"},
    }
    required: list[set[str]] = []
    for requirement in profile.requirements:
        if (
            requirement.necessity != "hard_required"
            or requirement.category != "adme_process"
        ):
            continue
        normalized = str(requirement.expected_value).strip().casefold()
        process_types = mapping.get(normalized)
        if process_types is not None and process_types not in required:
            required.append(process_types)
    return required


def required_process_types(profile: RequirementProfile) -> set[str]:
    return {
        next(iter(options))
        for options in required_process_type_options(profile)
        if len(options) == 1
    }


def select_template(
    *,
    catalog: TemplateCatalog,
    profile: RequirementProfile,
) -> BlueprintTemplate:
    if profile.ranking_eligibility != "ready":
        raise TemplateCatalogError(
            "Blocked profiles cannot select executable templates."
        )
    compound_semantic_id = (
        f"compound:{profile.compound_identity.compound_id}"
    )
    species = {
        str(item.expected_value).casefold()
        for item in profile.requirements
        if item.category == "biological_scope"
        and item.necessity == "hard_required"
    }
    objectives = {
        str(item.expected_value).casefold()
        for item in profile.requirements
        if item.category == "validation_context"
        and item.necessity == "hard_required"
    }
    needed_process_options = required_process_type_options(profile)
    candidates = [
        template
        for template in catalog.templates
        if compound_semantic_id in template.compound_semantic_ids
        and (
            not species
            or template.species.casefold() in species
        )
        and (
            not objectives
            or bool(
                objectives
                & set(map(str.casefold, template.supported_objectives))
            )
        )
        and all(
            options & set(template.supported_process_types)
            for options in needed_process_options
        )
    ]
    if not candidates:
        raise TemplateCatalogError("NO_EXECUTABLE_TEMPLATE")
    return sorted(
        candidates,
        key=lambda item: (
            item.template_id,
            item.template_version,
        ),
    )[0]
