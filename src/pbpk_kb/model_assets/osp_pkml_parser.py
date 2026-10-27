from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from defusedxml import ElementTree as SafeET

from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.models import (
    OSPArtifactSource,
    OSPPKMLAdministrationRecord,
    OSPPKMLComponentRecord,
    OSPPKMLOutputColumn,
    OSPPKMLParameterRecord,
    OSPPKMLProcessParticipant,
    OSPPKMLProcessRecord,
    OSPPKMLResultCSVSummary,
    OSPPKMLSimulationSummary,
)


PARSER_VERSION = "1.2.0"
PARAMETER_TAGS = {
    "Parameter",
    "IndividualParameter",
    "ExpressionParameter",
    "DistributedParameter",
}
PROCESS_TAGS = {"ReactionBuilder", "Reaction", "TransportBuilder", "Transport"}
COMPARTMENT_TYPES = {
    "Compartment",
    "Container",
    "Organ",
    "Physical",
    "Tissue",
}
UNIT_SUFFIX = re.compile(r"^(?P<name>.*?)\s*\[(?P<unit>[^\[\]]+)]\s*$")


class OSPPKMLParseError(RuntimeError):
    pass


@dataclass(slots=True)
class _Frame:
    tag: str
    attrs: dict[str, str]
    ordinal: int


@dataclass(slots=True)
class _PendingParameter:
    depth: int
    element_type: str
    attrs: dict[str, str]
    source: OSPArtifactSource
    scope_path: str
    value_origin_source: str | None = None
    value_origin_description: str | None = None


@dataclass(slots=True)
class _PendingProcess:
    depth: int
    element_type: str
    attrs: dict[str, str]
    source: OSPArtifactSource
    container_id: str | None = None
    educts: list[tuple[str, float | None]] = field(default_factory=list)
    products: list[tuple[str, float | None]] = field(default_factory=list)
    modifiers: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _CompletedProcess:
    element_type: str
    attrs: dict[str, str]
    source: OSPArtifactSource
    container_id: str | None
    educt_partners: list[tuple[str, float | None]]
    product_partners: list[tuple[str, float | None]]
    modifier_identifiers: list[str]


@dataclass(slots=True)
class OSPPKMLParsedRecords:
    summary: OSPPKMLSimulationSummary
    parameters: list[OSPPKMLParameterRecord]
    components: list[OSPPKMLComponentRecord]
    processes: list[OSPPKMLProcessRecord]
    administrations: list[OSPPKMLAdministrationRecord]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _coerce_value(value: str | None) -> float | str | None:
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    return number if math.isfinite(number) else value


def _bool_attribute(value: str | None) -> bool | None:
    if value == "1":
        return True
    if value == "0":
        return False
    return None


def _path_segment(frame: _Frame) -> str:
    identity = frame.attrs.get("id") or frame.attrs.get("name")
    if identity:
        escaped = identity.replace("\\", "\\\\").replace("]", "\\]")
        return f"{frame.tag}[{escaped}]"
    return f"{frame.tag}[{frame.ordinal}]"


def _element_path(frames: list[_Frame]) -> str:
    return "/" + "/".join(_path_segment(frame) for frame in frames)


def _source(
    *,
    artifact_relative_path: str,
    artifact_sha256: str,
    job_id: str,
    frames: list[_Frame],
) -> OSPArtifactSource:
    element_path = _element_path(frames)
    return OSPArtifactSource(
        relative_path=artifact_relative_path,
        sha256=artifact_sha256,
        source_location=(
            f"OSP_EXECUTION_RESULT:{job_id}/{artifact_relative_path}"
            f"#element_path={element_path}"
        ),
    )


def _component_type(attrs: dict[str, str], tag: str) -> str:
    if tag == "Individual":
        return "individual"
    if tag == "Solver":
        return "solver"
    if tag == "Observer":
        return "observer"
    if tag == "Container":
        return "compartment"

    quantity_type = attrs.get("quantityType", "").casefold()
    normalized_quantity_type = re.sub(r"[\s_-]+", "", quantity_type)
    icon = attrs.get("icon", "").casefold()
    if quantity_type == "drug" or icon == "drug":
        return "compound"
    if quantity_type == "metabolite":
        return "metabolite"
    if quantity_type == "enzyme" or icon == "enzyme":
        return "enzyme"
    if quantity_type == "transporter" or icon == "transporter":
        return "transporter"
    if normalized_quantity_type in {"otherprotein", "protein"} or icon == "protein":
        return "protein"
    if normalized_quantity_type == "complex":
        return "molecular_complex"
    if tag in {"MoleculeBuilder", "MoleculeAmount"}:
        return "molecular_entity"
    return "other"


def _process_type(attrs: dict[str, str], tag: str) -> str:
    text = " ".join(
        filter(
            None,
            (
                attrs.get("name"),
                attrs.get("description"),
                attrs.get("icon"),
            ),
        )
    ).casefold()
    if "glomerular" in text or "renal" in text or "urine" in text:
        return "renal_excretion"
    if "biliar" in text or "gallbladder" in text or "feces" in text:
        return "biliary_excretion"
    if "absorp" in text or "lumen drug transport" in text:
        return "absorption"
    if tag.startswith("Reaction") or "metabolism" in text:
        return "metabolism"
    if "passive diffusion" in text:
        return "distribution"
    if tag.startswith("Transport"):
        return "transport"
    return "other"


def _ancestor_formulation(frames: list[_Frame]) -> str | None:
    for frame in reversed(frames[:-1]):
        if (
            frame.tag == "EventGroupBuilder"
            and frame.attrs.get("containerType") == "Formulation"
        ):
            return frame.attrs.get("name")
    return None


def _partner_section(frames: list[_Frame]) -> str | None:
    for frame in reversed(frames):
        if frame.tag in {"Educts", "Products"}:
            return frame.tag
    return None


def _ancestor_container(
    frames: list[_Frame],
) -> tuple[str | None, str | None]:
    for frame in reversed(frames[:-1]):
        if frame.tag == "Container":
            return frame.attrs.get("id"), frame.attrs.get("name")
    return None, None


def _stoichiometric_coefficient(value: str | None) -> float:
    if value is None:
        return 1.0
    try:
        coefficient = float(value)
    except ValueError as exc:
        raise OSPPKMLParseError(
            f"Invalid stoichiometric coefficient: {value!r}."
        ) from exc
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise OSPPKMLParseError(
            f"Invalid stoichiometric coefficient: {value!r}."
        )
    return coefficient


def _resolve_process_participants(
    partners: list[tuple[str, float | None]],
    *,
    components_by_id: dict[str, tuple[str, str]],
    components_by_name: dict[str, set[tuple[str, str]]],
) -> tuple[list[str], list[OSPPKMLProcessParticipant], list[str]]:
    names: set[str] = set()
    participants: list[OSPPKMLProcessParticipant] = []
    unresolved: set[str] = set()
    coefficients: dict[str, float | None] = {}
    for identifier, coefficient in partners:
        previous = coefficients.get(identifier)
        if previous is not None and previous != coefficient:
            raise OSPPKMLParseError(
                "A process participant has conflicting stoichiometric "
                f"coefficients: {identifier!r}."
            )
        coefficients[identifier] = coefficient
    for identifier in sorted(coefficients, key=str.casefold):
        coefficient = coefficients[identifier]
        resolved = components_by_id.get(identifier)
        if resolved is not None:
            name, component_type = resolved
            names.add(name)
            participants.append(
                OSPPKMLProcessParticipant(
                    identifier=identifier,
                    name=name,
                    component_type=component_type,
                    resolution_status="resolved_by_component_id",
                    stoichiometric_coefficient=coefficient,
                )
            )
            continue

        name_matches = components_by_name.get(identifier.casefold(), set())
        specific_name_matches = {
            match
            for match in name_matches
            if match[1] != "molecular_entity"
        }
        if len(specific_name_matches) == 1:
            name_matches = specific_name_matches
        if len(name_matches) == 1:
            name, component_type = next(iter(name_matches))
            names.add(name)
            participants.append(
                OSPPKMLProcessParticipant(
                    identifier=identifier,
                    name=name,
                    component_type=component_type,
                    resolution_status="resolved_by_component_name",
                    stoichiometric_coefficient=coefficient,
                )
            )
            continue

        unresolved.add(identifier)
        participants.append(
            OSPPKMLProcessParticipant(
                identifier=identifier,
                resolution_status="unresolved",
                stoichiometric_coefficient=coefficient,
            )
        )
    return (
        sorted(names, key=str.casefold),
        participants,
        sorted(unresolved, key=str.casefold),
    )


def parse_result_csv(
    path: Path,
    *,
    artifact_relative_path: str,
    source_relative_path: str,
    expected_sha256: str | None = None,
) -> OSPPKMLResultCSVSummary:
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise OSPPKMLParseError(
            f"CSV checksum mismatch for {artifact_relative_path}."
        )
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            metadata: dict[str, str] = {}

            def data_lines() -> Any:
                for raw_line in handle:
                    if raw_line.startswith("#"):
                        key, separator, value = raw_line[1:].partition(":")
                        if separator:
                            metadata[key.strip()] = value.strip()
                        continue
                    if raw_line.strip():
                        yield raw_line

            reader = csv.reader(data_lines())
            headers = next(reader)
            if not headers:
                raise OSPPKMLParseError("CSV has no columns.")

            columns: list[OSPPKMLOutputColumn] = []
            for header in headers:
                match = UNIT_SUFFIX.match(header)
                columns.append(
                    OSPPKMLOutputColumn(
                        name=(match.group("name").strip() if match else header),
                        unit=(match.group("unit").strip() if match else None),
                    )
                )
            normalized_names = [column.name.casefold() for column in columns]
            source_name = source_relative_path.casefold()
            if source_name.endswith("-results.csv"):
                result_kind = "timeseries"
            elif source_name.endswith("-population.csv"):
                result_kind = "population"
            elif source_name.endswith("-pk-analyses.csv"):
                result_kind = "pk_analysis"
            elif "time" in normalized_names:
                result_kind = "timeseries"
            elif {"quantitypath", "parameter", "value"} <= set(normalized_names):
                result_kind = "pk_analysis"
            elif {"gender", "population"} <= set(normalized_names):
                result_kind = "population"
            else:
                result_kind = "other"

            individual_index = (
                normalized_names.index("individualid")
                if "individualid" in normalized_names
                else None
            )
            time_index = (
                normalized_names.index("time")
                if "time" in normalized_names
                else None
            )
            time_name = (
                columns[time_index].name if time_index is not None else None
            )
            time_unit = (
                columns[time_index].unit if time_index is not None else None
            )
            output_columns = [
                column
                for index, column in enumerate(columns)
                if result_kind == "timeseries"
                and index not in {individual_index, time_index}
            ]
            individuals: set[str] = set()
            row_count = 0
            time_min: float | None = None
            time_max: float | None = None
            numeric_values = 0
            text_values = 0
            blank_values = 0
            non_finite = 0
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(headers):
                    raise OSPPKMLParseError(
                        f"CSV row {row_number} has {len(row)} values; "
                        f"expected {len(headers)}."
                    )
                row_count += 1
                if individual_index is not None and row[individual_index]:
                    individuals.add(row[individual_index])
                for column_index, value in enumerate(row):
                    if not value:
                        blank_values += 1
                        continue
                    try:
                        number = float(value)
                    except ValueError:
                        text_values += 1
                        if (
                            result_kind == "timeseries"
                            and column_index not in {individual_index}
                        ):
                            raise OSPPKMLParseError(
                                f"CSV row {row_number} has a non-numeric "
                                f"timeseries value in {headers[column_index]!r}."
                            )
                        continue
                    numeric_values += 1
                    if not math.isfinite(number):
                        non_finite += 1
                    if column_index != time_index or not math.isfinite(number):
                        continue
                    time_min = (
                        number if time_min is None else min(time_min, number)
                    )
                    time_max = (
                        number if time_max is None else max(time_max, number)
                    )
            if result_kind == "timeseries" and time_index is None:
                raise OSPPKMLParseError(
                    "Timeseries CSV does not contain a Time column."
                )
            if result_kind == "timeseries" and row_count == 0:
                raise OSPPKMLParseError("Timeseries CSV has no data rows.")
            if result_kind == "timeseries":
                for index, column in enumerate(columns):
                    if index in {individual_index, time_index}:
                        continue
                    if not column.name:
                        raise OSPPKMLParseError(
                            "Timeseries CSV contains an unnamed output column."
                        )
        return OSPPKMLResultCSVSummary(
            relative_path=artifact_relative_path,
            source_relative_path=source_relative_path,
            sha256=actual_sha256,
            result_kind=result_kind,
            row_count=row_count,
            individual_ids=sorted(individuals, key=str.casefold),
            columns=columns,
            metadata=metadata,
            time_column=time_name,
            time_unit=time_unit,
            time_min=time_min,
            time_max=time_max,
            outputs=output_columns,
            numeric_value_count=numeric_values,
            text_value_count=text_values,
            blank_value_count=blank_values,
            non_finite_value_count=non_finite,
            parse_status="parsed",
        )
    except Exception as exc:
        if isinstance(exc, OSPPKMLParseError):
            error = str(exc)
        else:
            error = f"{type(exc).__name__}: {exc}"
        return OSPPKMLResultCSVSummary(
            relative_path=artifact_relative_path,
            source_relative_path=source_relative_path,
            sha256=actual_sha256,
            result_kind="other",
            row_count=0,
            individual_ids=[],
            columns=[],
            outputs=[],
            numeric_value_count=0,
            text_value_count=0,
            blank_value_count=0,
            non_finite_value_count=0,
            parse_status="failed",
            error=error,
        )


def parse_pkml_simulation(
    path: Path,
    *,
    artifact_relative_path: str,
    artifact_sha256: str,
    job_id: str,
    project_key: str,
    simulation_id: str,
    simulation_name: str,
    csv_results: list[OSPPKMLResultCSVSummary] | None = None,
) -> OSPPKMLParsedRecords:
    if sha256_file(path) != artifact_sha256:
        raise OSPPKMLParseError(
            f"PKML checksum mismatch for {artifact_relative_path}."
        )

    frames: list[_Frame] = []
    sibling_counts: list[dict[str, int]] = [{}]
    pending_parameters: list[_PendingParameter] = []
    pending_processes: list[_PendingProcess] = []
    parameters: list[OSPPKMLParameterRecord] = []
    components: list[OSPPKMLComponentRecord] = []
    completed_processes: list[_CompletedProcess] = []
    administrations: list[OSPPKMLAdministrationRecord] = []
    embedded_simulation_id: str | None = None
    embedded_simulation_name: str | None = None
    xml_version: str | None = None
    species: set[str] = set()
    populations: set[str] = set()
    compounds: set[str] = set()
    metabolites: set[str] = set()
    enzymes: set[str] = set()
    transporters: set[str] = set()
    routes: set[str] = set()
    formulations: set[str] = set()
    compartments: set[str] = set()
    observers: set[str] = set()
    solver: str | None = None
    warnings: list[str] = []
    participant_components_by_id: dict[str, tuple[str, str]] = {}
    participant_components_by_name: dict[
        str, set[tuple[str, str]]
    ] = {}
    component_container_by_id: dict[str, str] = {}

    try:
        iterator = SafeET.iterparse(path, events=("start", "end"))
        for event, element in iterator:
            tag = _local_name(element.tag)
            if event == "start":
                siblings = sibling_counts[-1]
                ordinal = siblings.get(tag, 0) + 1
                siblings[tag] = ordinal
                frame = _Frame(tag=tag, attrs=dict(element.attrib), ordinal=ordinal)
                frames.append(frame)
                sibling_counts.append({})
                attrs = frame.attrs
                source = _source(
                    artifact_relative_path=artifact_relative_path,
                    artifact_sha256=artifact_sha256,
                    job_id=job_id,
                    frames=frames,
                )
                if tag == "SimulationTransfer":
                    xml_version = attrs.get("version")
                elif tag == "Simulation":
                    embedded_simulation_id = attrs.get("id")
                    embedded_simulation_name = attrs.get("name")

                if tag in PARAMETER_TAGS and attrs.get("id") and attrs.get("name"):
                    pending_parameters.append(
                        _PendingParameter(
                            depth=len(frames),
                            element_type=tag,
                            attrs=attrs,
                            source=source,
                            scope_path=_element_path(frames[:-1]),
                        )
                    )
                elif tag == "ValueOrigin" and pending_parameters:
                    pending_parameters[-1].value_origin_source = attrs.get("source")
                    pending_parameters[-1].value_origin_description = attrs.get(
                        "description"
                    )

                if tag in PROCESS_TAGS and attrs.get("id") and attrs.get("name"):
                    pending_process = _PendingProcess(
                        depth=len(frames),
                        element_type=tag,
                        attrs=attrs,
                        source=source,
                        container_id=_ancestor_container(frames)[0],
                    )
                    source_partner = attrs.get(
                        "sourceAmount"
                    ) or attrs.get("sourceMoleculeName")
                    target_partner = attrs.get(
                        "targetAmount"
                    ) or attrs.get("targetMoleculeName")
                    if source_partner:
                        pending_process.educts.append((source_partner, 1.0))
                    if target_partner:
                        pending_process.products.append((target_partner, 1.0))
                    pending_processes.append(pending_process)
                elif pending_processes and tag in {
                    "ReactionPartnerBuilder",
                    "ReactionPartner",
                }:
                    partner = attrs.get("moleculeName") or attrs.get("partner")
                    coefficient = _stoichiometric_coefficient(
                        attrs.get("stoichiometricCoefficient")
                    )
                    section = _partner_section(frames)
                    if partner and section == "Educts":
                        pending_processes[-1].educts.append(
                            (partner, coefficient)
                        )
                    elif partner and section == "Products":
                        pending_processes[-1].products.append(
                            (partner, coefficient)
                        )

                if tag in {
                    "MoleculeBuilder",
                    "MoleculeAmount",
                    "Individual",
                    "Container",
                    "Observer",
                    "Solver",
                }:
                    component_id = attrs.get("id")
                    name = attrs.get("name")
                    component_type = _component_type(attrs, tag)
                    if component_id and name:
                        parent_container_id, parent_container_name = (
                            _ancestor_container(frames)
                        )
                        identity = (name, component_type)
                        if tag in {"MoleculeBuilder", "MoleculeAmount"}:
                            existing_identity = (
                                participant_components_by_id.get(
                                    component_id
                                )
                            )
                            if (
                                existing_identity is not None
                                and existing_identity != identity
                                and "DUPLICATE_COMPONENT_ID_CONFLICT"
                                not in warnings
                            ):
                                warnings.append(
                                    "DUPLICATE_COMPONENT_ID_CONFLICT"
                                )
                            else:
                                participant_components_by_id[
                                    component_id
                                ] = identity
                            participant_components_by_name.setdefault(
                                name.casefold(), set()
                            ).add(identity)
                            if parent_container_id is not None:
                                component_container_by_id[
                                    component_id
                                ] = parent_container_id
                        components.append(
                            OSPPKMLComponentRecord(
                                parser_version=PARSER_VERSION,
                                project_key=project_key,
                                simulation_id=simulation_id,
                                simulation_name=simulation_name,
                                component_id=component_id,
                                name=name,
                                component_type=component_type,
                                element_type=tag,
                                description=attrs.get("description"),
                                quantity_type=attrs.get("quantityType"),
                                icon=attrs.get("icon"),
                                dimension=attrs.get("dim")
                                or attrs.get("dimension"),
                                display_unit=attrs.get("displayUnit"),
                                parent_container_id=parent_container_id,
                                parent_container_name=parent_container_name,
                                source=source,
                            )
                        )
                    if component_type == "compound" and name:
                        compounds.add(name)
                    elif component_type == "metabolite" and name:
                        metabolites.add(name)
                    elif component_type == "enzyme" and name:
                        enzymes.add(name)
                    elif component_type == "transporter" and name:
                        transporters.add(name)
                    elif component_type == "observer" and name:
                        observers.add(name)
                    elif component_type == "individual" and name:
                        populations.add(name)
                        if attrs.get("icon"):
                            species.add(attrs["icon"])
                    elif component_type == "compartment" and name:
                        container_type = attrs.get("containerType")
                        if container_type in COMPARTMENT_TYPES or not container_type:
                            compartments.add(name)
                    elif component_type == "solver" and name:
                        solver = name

                if tag == "ApplicationBuilder":
                    administration_id = attrs.get("id")
                    name = attrs.get("name")
                    route = attrs.get("eventGroupType") or attrs.get("icon")
                    if administration_id and name and route:
                        formulation = _ancestor_formulation(frames)
                        administrations.append(
                            OSPPKMLAdministrationRecord(
                                parser_version=PARSER_VERSION,
                                project_key=project_key,
                                simulation_id=simulation_id,
                                simulation_name=simulation_name,
                                administration_id=administration_id,
                                name=name,
                                route=route,
                                molecule_name=attrs.get("moleculeName"),
                                formulation=formulation,
                                source=source,
                            )
                        )
                        routes.add(route)
                        if formulation:
                            formulations.add(formulation)
                continue

            if (
                tag == "ModifierNamesChild"
                and pending_processes
                and element.text
                and element.text.strip()
            ):
                pending_processes[-1].modifiers.append(element.text.strip())

            if (
                pending_parameters
                and pending_parameters[-1].depth == len(frames)
                and pending_parameters[-1].element_type == tag
            ):
                pending = pending_parameters.pop()
                attrs = pending.attrs
                parameters.append(
                    OSPPKMLParameterRecord(
                        parser_version=PARSER_VERSION,
                        project_key=project_key,
                        simulation_id=simulation_id,
                        simulation_name=simulation_name,
                        element_type=pending.element_type,
                        parameter_id=attrs["id"],
                        name=attrs["name"],
                        value=_coerce_value(attrs.get("value")),
                        formula_id=attrs.get("formula"),
                        dimension=attrs.get("dim") or attrs.get("dimension"),
                        display_unit=attrs.get("displayUnit"),
                        is_default=_bool_attribute(attrs.get("isDefault")),
                        quantity_type=attrs.get("quantityType"),
                        scope_path=pending.scope_path,
                        value_origin_source=pending.value_origin_source,
                        value_origin_description=pending.value_origin_description,
                        source=pending.source,
                    )
                )

            if (
                pending_processes
                and pending_processes[-1].depth == len(frames)
                and pending_processes[-1].element_type == tag
            ):
                pending_process = pending_processes.pop()
                attrs = pending_process.attrs
                completed_processes.append(
                    _CompletedProcess(
                        element_type=pending_process.element_type,
                        attrs=attrs,
                        source=pending_process.source,
                        container_id=pending_process.container_id,
                        educt_partners=pending_process.educts,
                        product_partners=pending_process.products,
                        modifier_identifiers=sorted(
                            set(pending_process.modifiers), key=str.casefold
                        ),
                    )
                )

            element.clear()
            frames.pop()
            sibling_counts.pop()
    except Exception as exc:
        if isinstance(exc, OSPPKMLParseError):
            raise
        raise OSPPKMLParseError(
            f"Unable to parse PKML {artifact_relative_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if embedded_simulation_name != simulation_name:
        raise OSPPKMLParseError(
            "PKML simulation name does not match the execution manifest: "
            f"{embedded_simulation_name!r} != {simulation_name!r}."
        )
    processes: list[OSPPKMLProcessRecord] = []
    unresolved_process_partner_count = 0
    for completed_process in completed_processes:
        educts, educt_participants, unresolved_educts = (
            _resolve_process_participants(
                completed_process.educt_partners,
                components_by_id=participant_components_by_id,
                components_by_name=participant_components_by_name,
            )
        )
        products, product_participants, unresolved_products = (
            _resolve_process_participants(
                completed_process.product_partners,
                components_by_id=participant_components_by_id,
                components_by_name=participant_components_by_name,
            )
        )
        modifiers, modifier_participants, unresolved_modifiers = (
            _resolve_process_participants(
                [
                    (identifier, None)
                    for identifier in completed_process.modifier_identifiers
                ],
                components_by_id=participant_components_by_id,
                components_by_name=participant_components_by_name,
            )
        )
        unresolved_identifiers = sorted(
            set(
                unresolved_educts
                + unresolved_products
                + unresolved_modifiers
            ),
            key=str.casefold,
        )
        unresolved_process_partner_count += len(unresolved_identifiers)
        attrs = completed_process.attrs
        educt_identifiers = [
            identifier
            for identifier, _ in completed_process.educt_partners
        ]
        product_identifiers = [
            identifier
            for identifier, _ in completed_process.product_partners
        ]
        input_compartment_id = completed_process.container_id
        output_compartment_id = completed_process.container_id
        if completed_process.element_type == "Transport":
            input_compartment_id = next(
                (
                    component_container_by_id[identifier]
                    for identifier in educt_identifiers
                    if identifier in component_container_by_id
                ),
                None,
            )
            output_compartment_id = next(
                (
                    component_container_by_id[identifier]
                    for identifier in product_identifiers
                    if identifier in component_container_by_id
                ),
                None,
            )
        processes.append(
            OSPPKMLProcessRecord(
                parser_version=PARSER_VERSION,
                project_key=project_key,
                simulation_id=simulation_id,
                simulation_name=simulation_name,
                process_id=attrs["id"],
                name=attrs["name"],
                process_type=_process_type(
                    attrs, completed_process.element_type
                ),
                element_type=completed_process.element_type,
                description=attrs.get("description"),
                dimension=attrs.get("dim") or attrs.get("dimension"),
                formula_id=attrs.get("formula"),
                educts=educts,
                products=products,
                modifiers=modifiers,
                educt_identifiers=sorted(
                    set(educt_identifiers), key=str.casefold
                ),
                product_identifiers=sorted(
                    set(product_identifiers), key=str.casefold
                ),
                modifier_identifiers=completed_process.modifier_identifiers,
                educt_participants=educt_participants,
                product_participants=product_participants,
                modifier_participants=modifier_participants,
                input_compartment_id=input_compartment_id,
                output_compartment_id=output_compartment_id,
                unresolved_partner_identifiers=unresolved_identifiers,
                source=completed_process.source,
            )
        )
    if unresolved_process_partner_count:
        warnings.append("UNRESOLVED_PROCESS_PARTICIPANTS")
    if not parameters:
        warnings.append("NO_PARAMETERS_PARSED")
    if not components:
        warnings.append("NO_COMPONENTS_PARSED")
    csv_results = csv_results or []
    if any(result.parse_status != "parsed" for result in csv_results):
        warnings.append("CSV_RESULT_PARSE_FAILED")
    if any(result.non_finite_value_count for result in csv_results):
        warnings.append("CSV_RESULT_NON_FINITE_VALUES")

    summary = OSPPKMLSimulationSummary(
        parser_version=PARSER_VERSION,
        project_key=project_key,
        simulation_id=simulation_id,
        simulation_name=simulation_name,
        embedded_simulation_id=embedded_simulation_id,
        pkml_source=OSPArtifactSource(
            relative_path=artifact_relative_path,
            sha256=artifact_sha256,
            source_location=(
                f"OSP_EXECUTION_RESULT:{job_id}/{artifact_relative_path}"
            ),
        ),
        xml_version=xml_version,
        species=sorted(species, key=str.casefold),
        populations=sorted(populations, key=str.casefold),
        compounds=sorted(compounds, key=str.casefold),
        metabolites=sorted(metabolites, key=str.casefold),
        enzymes=sorted(enzymes, key=str.casefold),
        transporters=sorted(transporters, key=str.casefold),
        routes=sorted(routes, key=str.casefold),
        formulations=sorted(formulations, key=str.casefold),
        compartments=sorted(compartments, key=str.casefold),
        observers=sorted(observers, key=str.casefold),
        solver=solver,
        parameter_count=len(parameters),
        component_count=len(components),
        process_count=len(processes),
        administration_count=len(administrations),
        csv_results=csv_results,
        parse_status="parsed",
        warnings=warnings,
    )
    return OSPPKMLParsedRecords(
        summary=summary,
        parameters=parameters,
        components=components,
        processes=processes,
        administrations=administrations,
    )
