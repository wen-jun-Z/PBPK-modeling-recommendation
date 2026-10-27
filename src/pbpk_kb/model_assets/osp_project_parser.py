from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from defusedxml import ElementTree as SafeET

from pbpk_kb.model_assets.compression import OSPContentError, decompress_osp_content
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.models import (
    ExecutionProbeStatus,
    OSPBuildingBlockSummary,
    OSPCompressedContentAudit,
    OSPCompoundSummary,
    OSPParameterSummary,
    OSPProjectParseResult,
    OSPProtocolSummary,
    OSPSimulationSummary,
    ParseStatus,
)


PARSER_VERSION = "1.2.0"


class OSPProjectError(RuntimeError):
    pass


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _read_only_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path.resolve()))}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _schema_fingerprint(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        """
        SELECT type, name, COALESCE(sql, '') AS sql
        FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()
    payload = json.dumps(
        [dict(row) for row in rows],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _table_row_counts(connection: sqlite3.Connection) -> dict[str, int]:
    names = [
        row["name"]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        if not row["name"].startswith("sqlite_")
    ]
    return {
        name: int(connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
        for name in names
    }


def _root_attributes(xml: bytes) -> tuple[str, dict[str, str]]:
    for _event, element in SafeET.iterparse(io.BytesIO(xml), events=("start",)):
        return _local_name(element.tag), dict(element.attrib)
    raise OSPProjectError("Compressed XML content has no root element.")


def _coerce_number(value: str | None) -> float | str | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return value


def _parameters(root: Any) -> list[OSPParameterSummary]:
    output: list[OSPParameterSummary] = []
    for element in root.iter():
        if _local_name(element.tag) != "Para":
            continue
        attrs = element.attrib
        parameter_id = attrs.get("id")
        name = attrs.get("name")
        if not parameter_id or not name:
            continue
        is_default: bool | None = None
        if attrs.get("isDefault") in {"0", "1"}:
            is_default = attrs["isDefault"] == "1"
        output.append(
            OSPParameterSummary(
                parameter_id=parameter_id,
                name=name,
                value=_coerce_number(attrs.get("value")),
                formula_id=attrs.get("formula"),
                dimension=attrs.get("dim"),
                display_unit=attrs.get("displayUnit"),
                is_default=is_default,
            )
        )
    return output


def _parse_compound(xml: bytes) -> OSPCompoundSummary:
    root = SafeET.fromstring(xml)
    processes: set[str] = set()
    for element in root.iter():
        tag = _local_name(element.tag)
        if tag.endswith("Process"):
            processes.add(element.attrib.get("name") or tag)
    return OSPCompoundSummary(
        compound_id=root.attrib["id"],
        name=root.attrib["name"],
        parameters=_parameters(root),
        process_names=sorted(processes, key=str.casefold),
    )


def _parse_protocol(xml: bytes) -> OSPProtocolSummary:
    root = SafeET.fromstring(xml)
    parameters = {
        parameter.name: parameter.value
        for parameter in _parameters(root)
        if parameter.value is not None
    }
    return OSPProtocolSummary(
        protocol_id=root.attrib["id"],
        name=root.attrib["name"],
        application_type=root.attrib.get("applicationType"),
        dosing_interval=root.attrib.get("dosingInterval"),
        parameters=parameters,
    )


def _summarize_simulation(
    xml: bytes,
    *,
    simulation_id: str,
    name: str,
    simulation_mode: str,
    content_id: int,
    has_stored_results: bool,
    used_building_block_count: int,
    observed_dataset_count: int,
) -> OSPSimulationSummary:
    organs: set[str] = set()
    molecules: set[str] = set()
    reactions: set[str] = set()
    transports: set[str] = set()
    observers: set[str] = set()
    parameter_count = 0
    equation_count = 0
    for event, element in SafeET.iterparse(io.BytesIO(xml), events=("start", "end")):
        if event == "end":
            element.clear()
            continue
        tag = _local_name(element.tag)
        attrs = element.attrib
        name_value = attrs.get("name")
        if tag == "Para":
            parameter_count += 1
        elif tag == "Formula":
            equation_count += 1
        elif tag == "Organ" and name_value:
            organs.add(name_value)
        elif tag == "Container" and attrs.get("containerType") == "Molecule" and name_value:
            molecules.add(name_value)
        elif tag == "Reaction" and name_value:
            reactions.add(name_value)
        elif tag == "Transport" and name_value:
            transports.add(name_value)
        elif tag == "Observer" and name_value:
            observers.add(name_value)
    sort = lambda values: sorted(values, key=str.casefold)
    return OSPSimulationSummary(
        simulation_id=simulation_id,
        name=name,
        simulation_mode=simulation_mode,
        content_id=content_id,
        has_stored_results=has_stored_results,
        used_building_block_count=used_building_block_count,
        observed_dataset_count=observed_dataset_count,
        organ_names=sort(organs),
        molecule_names=sort(molecules),
        reaction_names=sort(reactions),
        transport_names=sort(transports),
        observer_names=sort(observers),
        parameter_count=parameter_count,
        equation_count=equation_count,
    )


def _names_for_icon(rows: Iterable[sqlite3.Row], icon: str) -> list[str]:
    return sorted(
        {str(row["Name"]) for row in rows if str(row["Icon"]).casefold() == icon.casefold()},
        key=str.casefold,
    )


def parse_osp_project(
    path: Path,
    *,
    repository_commit: str,
    parser_version: str = PARSER_VERSION,
) -> OSPProjectParseResult:
    path = path.resolve()
    warnings: list[str] = []
    try:
        connection = _read_only_connection(path)
    except sqlite3.Error as exc:
        raise OSPProjectError(f"Unable to open OSP project {path}: {exc}") from exc
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise OSPProjectError(f"SQLite integrity check failed for {path}: {integrity}")
        projects = connection.execute("SELECT * FROM PROJECTS ORDER BY Id").fetchall()
        if len(projects) != 1:
            raise OSPProjectError(
                f"Expected exactly one PROJECTS row in {path}; found {len(projects)}."
            )
        project = projects[0]
        table_counts = _table_row_counts(connection)
        block_rows = connection.execute(
            """
            SELECT Id, Name, Icon, Version, StructureVersion, ContentId
            FROM BUILDING_BLOCKS
            ORDER BY Icon, Name, Id
            """
        ).fetchall()
        referenced_content_ids = {
            int(row["ContentId"]) for row in block_rows if row["ContentId"] is not None
        }
        available_content_ids = {
            int(row["Id"])
            for row in connection.execute(
                "SELECT Id FROM CONTENTS WHERE Data IS NOT NULL"
            ).fetchall()
        }
        missing_referenced_content = sorted(
            referenced_content_ids - available_content_ids
        )
        if missing_referenced_content:
            raise OSPProjectError(
                "Building blocks reference empty or absent content IDs: "
                + ", ".join(map(str, missing_referenced_content))
            )
        null_content_ids = [
            int(row["Id"])
            for row in connection.execute(
                "SELECT Id FROM CONTENTS WHERE Data IS NULL ORDER BY Id"
            ).fetchall()
        ]
        if null_content_ids:
            warnings.append(f"NULL_OPTIONAL_CONTENT_RECORDS:{len(null_content_ids)}")

        simulation_rows = connection.execute(
            """
            SELECT
                s.SimulationId,
                s.SimulationMode,
                s.SimulationResultsId,
                b.Name,
                b.ContentId,
                (SELECT COUNT(*) FROM USED_BUILDING_BLOCKS u
                 WHERE u.SimulationId = s.SimulationId) AS UsedBlockCount,
                (SELECT COUNT(*) FROM USED_OBSERVED_DATA o
                 WHERE o.SimulationId = s.SimulationId) AS ObservedCount
            FROM SIMULATIONS s
            JOIN BUILDING_BLOCKS b ON b.Id = s.SimulationId
            ORDER BY b.Name, s.SimulationId
            """
        ).fetchall()
        simulation_by_content_id = {
            int(row["ContentId"]): row for row in simulation_rows
        }
        block_by_content_id = {
            int(row["ContentId"]): row for row in block_rows
        }
        content_audits: list[OSPCompressedContentAudit] = []
        building_blocks: list[OSPBuildingBlockSummary] = []
        compound_details: list[OSPCompoundSummary] = []
        protocols: list[OSPProtocolSummary] = []
        simulations: list[OSPSimulationSummary] = []
        for content_row in connection.execute(
            "SELECT Id, Data FROM CONTENTS WHERE Data IS NOT NULL ORDER BY Id"
        ):
            content_id = int(content_row["Id"])
            blob = bytes(content_row["Data"])
            try:
                decoded = decompress_osp_content(blob)
                root_name, root_attrs = _root_attributes(decoded.data)
            except (OSPContentError, OSPProjectError, ValueError) as exc:
                raise OSPProjectError(
                    f"Failed to decode CONTENTS.Id={content_id} in {path}: {exc}"
                ) from exc
            if decoded.crc_valid is False:
                raise OSPProjectError(
                    f"CRC mismatch for CONTENTS.Id={content_id} in {path}."
                )
            content_audits.append(
                OSPCompressedContentAudit(
                    content_id=content_id,
                    filename=decoded.filename,
                    storage_format=decoded.storage_format,
                    compressed_size=decoded.compressed_size,
                    uncompressed_size=decoded.uncompressed_size,
                    crc32=decoded.crc32,
                    crc_valid=decoded.crc_valid,
                    xml_root=root_name,
                    xml_version=root_attrs.get("xmlVersion"),
                )
            )
            row = block_by_content_id.get(content_id)
            if row is None:
                continue
            building_blocks.append(
                OSPBuildingBlockSummary(
                    building_block_id=str(row["Id"]),
                    name=str(row["Name"]),
                    icon=str(row["Icon"]),
                    version=int(row["Version"]),
                    structure_version=int(row["StructureVersion"]),
                    content_id=content_id,
                    xml_root=root_name,
                    xml_version=root_attrs.get("xmlVersion"),
                )
            )
            icon = str(row["Icon"]).casefold()
            if icon == "compound":
                compound_details.append(_parse_compound(decoded.data))
            elif icon == "protocol":
                protocols.append(_parse_protocol(decoded.data))
            simulation_row = simulation_by_content_id.get(content_id)
            if simulation_row is not None:
                simulations.append(
                    _summarize_simulation(
                        decoded.data,
                        simulation_id=str(simulation_row["SimulationId"]),
                        name=str(simulation_row["Name"]),
                        simulation_mode=str(simulation_row["SimulationMode"]),
                        content_id=content_id,
                        has_stored_results=(
                            simulation_row["SimulationResultsId"] is not None
                        ),
                        used_building_block_count=int(
                            simulation_row["UsedBlockCount"]
                        ),
                        observed_dataset_count=int(simulation_row["ObservedCount"]),
                    )
                )
        if len(building_blocks) != len(block_rows):
            raise OSPProjectError(
                f"Parsed {len(building_blocks)} of {len(block_rows)} building blocks."
            )
        simulations.sort(key=lambda item: (item.name.casefold(), item.simulation_id))
        protocols.sort(key=lambda item: (item.name.casefold(), item.protocol_id))
        compound_details.sort(key=lambda item: (item.name.casefold(), item.compound_id))
        observed_datasets = [
            str(row["Name"])
            for row in connection.execute(
                "SELECT Name FROM DATA_REPOSITORIES ORDER BY Name, Id"
            ).fetchall()
        ]
        parameter_identifications = [
            str(row["Name"])
            for row in connection.execute(
                "SELECT Name FROM PARAMETER_IDENTIFICATIONS ORDER BY Name, Id"
            ).fetchall()
        ]
        icons = Counter(str(row["Icon"]) for row in block_rows)
        if not simulations:
            warnings.append("PROJECT_HAS_NO_SIMULATIONS")
        if not compound_details:
            warnings.append("PROJECT_HAS_NO_COMPOUND_BUILDING_BLOCK")
        if len(icons) == 1:
            warnings.append("PROJECT_HAS_SINGLE_BUILDING_BLOCK_TYPE")
        return OSPProjectParseResult(
            parser_version=parser_version,
            project_path=str(path),
            project_sha256=sha256_file(path),
            repository_commit=repository_commit,
            project_id=int(project["Id"]),
            project_name=str(project["Name"]),
            project_version=int(project["Version"]),
            sqlite_integrity="ok",
            schema_fingerprint=_schema_fingerprint(connection),
            parse_status=ParseStatus.REGISTERED_NOT_EXECUTABLE,
            execution_probe_status=ExecutionProbeStatus.BLOCKED_RUNTIME_UNAVAILABLE,
            compounds=[item.name for item in compound_details],
            compound_details=compound_details,
            expression_profiles=_names_for_icon(block_rows, "ENZYME")
            + _names_for_icon(block_rows, "TRANSPORTER")
            + _names_for_icon(block_rows, "PROTEIN"),
            formulations=_names_for_icon(block_rows, "Formulation"),
            individuals=_names_for_icon(block_rows, "Human")
            + _names_for_icon(block_rows, "Individual"),
            protocols=protocols,
            simulations=simulations,
            observed_datasets=observed_datasets,
            parameter_identifications=parameter_identifications,
            building_blocks=building_blocks,
            compressed_content_audit=content_audits,
            null_content_ids=null_content_ids,
            table_row_counts=table_counts,
            warnings=warnings,
        )
    except sqlite3.Error as exc:
        raise OSPProjectError(f"SQLite parse failed for {path}: {exc}") from exc
    finally:
        connection.close()
