from __future__ import annotations

import uuid

from pbpk_kb.blueprints.schemas import (
    BlueprintTemplate,
    DerivedParameter,
    MissingParameter,
    ParameterContext,
    ParameterDefinition,
    ProvenanceRef,
)


PARAMETER_NAMESPACE = uuid.UUID(
    "2717e118-2028-48d4-883b-56485bd19809"
)
PROVENANCE_NAMESPACE = uuid.UUID(
    "7fa29666-499c-4bc6-90fc-cee42b741c4c"
)


def resolve_template_parameters(
    *,
    template: BlueprintTemplate,
    compound_semantic_id: str,
    species: str,
    process_ids_by_semantic_id: dict[str, list[str]] | None = None,
    resolved_parameters: dict[str, ParameterDefinition] | None = None,
) -> list[ParameterDefinition]:
    process_ids_by_semantic_id = process_ids_by_semantic_id or {}
    resolved_parameters = resolved_parameters or {}
    parameters: list[ParameterDefinition] = []
    for slot in sorted(
        template.parameter_slots,
        key=lambda item: (item.semantic_id, item.model_path),
    ):
        parameter_id = str(
            uuid.uuid5(
                PARAMETER_NAMESPACE,
                f"{template.template_id}\n{slot.model_path}",
            )
        )
        provenance_id = uuid.uuid5(
            PROVENANCE_NAMESPACE,
            f"{slot.source_checksum}\n{slot.source_location}",
        )
        resolved = resolved_parameters.get(slot.model_path)
        if resolved is not None:
            if resolved.semantic_id != slot.semantic_id:
                raise ValueError(
                    f"Resolved parameter semantic ID drift: {slot.model_path}"
                )
            if resolved.model_path != slot.model_path:
                raise ValueError(
                    f"Resolved parameter path drift: {slot.model_path}"
                )
            if resolved.dimension != slot.dimension:
                raise ValueError(
                    f"Resolved parameter dimension drift: {slot.model_path}"
                )
            if resolved.canonical_unit != slot.canonical_unit:
                raise ValueError(
                    f"Resolved parameter unit drift: {slot.model_path}"
                )
            parameters.append(resolved)
            continue
        provenance = [
            ProvenanceRef(
                provenance_id=provenance_id,
                provenance_kind="model_asset",
                source_id="pksim_source_repository",
                source_location=slot.source_location,
                source_checksum=slot.source_checksum,
            )
        ]
        context = ParameterContext(
            compound_semantic_id=(
                compound_semantic_id if slot.compound_specific else None
            ),
            species=species,
        )
        if slot.compound_specific:
            parameters.append(
                MissingParameter(
                    parameter_id=parameter_id,
                    semantic_id=slot.semantic_id,
                    name=slot.slot_id,
                    model_path=slot.model_path,
                    dimension=slot.dimension,
                    canonical_unit=slot.canonical_unit,
                    context=context,
                    used_by_process_ids=process_ids_by_semantic_id.get(
                        slot.semantic_id,
                        [],
                    ),
                    provenance=provenance,
                    blocking=True,
                    origin="missing",
                    value=None,
                    missing_reason_code=(
                        "COMPOUND_SPECIFIC_PARAMETER_NOT_RESOLVED"
                    ),
                )
            )
            continue
        parameters.append(
            DerivedParameter(
                parameter_id=parameter_id,
                semantic_id=slot.semantic_id,
                name=slot.slot_id,
                model_path=slot.model_path,
                dimension=slot.dimension,
                canonical_unit=slot.canonical_unit,
                context=context,
                used_by_process_ids=process_ids_by_semantic_id.get(
                    slot.semantic_id,
                    [],
                ),
                provenance=provenance,
                blocking=True,
                origin="derived",
                value=slot.template_value,
                derivation_expression=(
                    f"inherit_template_value:{slot.model_path}"
                ),
                expression_version="template-inheritance-1.0.0",
                input_parameter_ids=[],
            )
        )
    return parameters
