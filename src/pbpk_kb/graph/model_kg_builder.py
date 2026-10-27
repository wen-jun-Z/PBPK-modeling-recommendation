from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from pbpk_kb.graph.model_kg_schema import (
    MODEL_KG_SCHEMA_VERSION,
    ModelKGEdge,
    ModelKGNode,
    ModelKGProvenance,
    canonical_json,
    schema_payload,
    stable_id,
    validate_relation,
)
from pbpk_kb.model_assets.models import (
    ModelValidationRunLabel,
    OSPExecutionPlan,
    OSPModelVersionBuildSummary,
    OSPModelVersionSnapshot,
)


BUILDER_VERSION = "pbpk-model-kg-builder-1.2.0"
_OPAQUE_PATH_ID = re.compile(r"\[[A-Za-z0-9_-]{8,}\]")


class ModelKGBuildError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ModelKGBuildError(f"Expected a JSON object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ModelKGBuildError(
                    f"Expected an object at {path}:{line_number}"
                )
            yield value


def _fact_path(directory: Path, stem: str) -> Path:
    candidates = (
        directory / f"{stem}.jsonl",
        directory / f"{stem}.jsonl.gz",
    )
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ModelKGBuildError(
            f"Expected exactly one {stem} JSONL input in {directory}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _write_json_atomic(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = (canonical_json(row) + "\n").encode("utf-8")
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return digest.hexdigest()


def _write_gzip_jsonl_atomic(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=6,
            mtime=0,
        ) as compressed:
            with io.TextIOWrapper(
                compressed,
                encoding="utf-8",
                newline="\n",
            ) as text:
                for row in rows:
                    text.write(canonical_json(row))
                    text.write("\n")
        raw.flush()
        os.fsync(raw.fileno())
    temporary.replace(path)
    return _sha256_file(path)


def _provenance(
    *,
    source_id: str,
    source_location: str,
    asset_sha256: str,
    extraction_method: str,
    evidence_id: str | None = None,
) -> ModelKGProvenance:
    return ModelKGProvenance(
        source_id=source_id,
        source_location=source_location,
        asset_sha256=asset_sha256,
        extraction_method=extraction_method,
        evidence_id=evidence_id,
    )


class _GraphStaging:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-131072")
        self.connection.executescript(
            """
            CREATE TABLE nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                name TEXT NOT NULL,
                properties_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                occurrence_count INTEGER NOT NULL DEFAULT 1
            ) WITHOUT ROWID;
            CREATE TABLE edges (
                edge_id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_id TEXT NOT NULL,
                object_type TEXT NOT NULL,
                properties_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                occurrence_count INTEGER NOT NULL DEFAULT 1
            ) WITHOUT ROWID;
            CREATE TABLE unresolved_process_partners (
                identifier TEXT PRIMARY KEY,
                normalized_identifier TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_location TEXT NOT NULL,
                asset_sha256 TEXT NOT NULL,
                extraction_method TEXT NOT NULL,
                occurrence_count INTEGER NOT NULL DEFAULT 1
            ) WITHOUT ROWID;
            """
        )
        self.node_buffer: dict[str, list[Any]] = {}
        self.edge_buffer: dict[str, list[Any]] = {}
        self.unresolved_partner_buffer: dict[str, list[Any]] = {}
        self.buffer_limit = 100_000

    def add_node(self, node: ModelKGNode) -> None:
        buffered = self.node_buffer.get(node.node_id)
        if buffered is not None:
            buffered[-1] += node.occurrence_count
        else:
            self.node_buffer[node.node_id] = [
                node.node_id,
                node.node_type,
                node.name,
                canonical_json(node.properties),
                canonical_json(node.provenance.model_dump(mode="json")),
                node.occurrence_count,
            ]
        self._flush_if_needed()

    def add_edge(self, edge: ModelKGEdge) -> None:
        relation_error = validate_relation(
            edge.predicate,
            edge.subject_type,
            edge.object_type,
        )
        if relation_error:
            raise ModelKGBuildError(relation_error)
        buffered = self.edge_buffer.get(edge.edge_id)
        if buffered is not None:
            buffered[-1] += edge.occurrence_count
        else:
            self.edge_buffer[edge.edge_id] = [
                edge.edge_id,
                edge.subject_id,
                edge.subject_type,
                edge.predicate,
                edge.object_id,
                edge.object_type,
                canonical_json(edge.properties),
                canonical_json(edge.provenance.model_dump(mode="json")),
                edge.occurrence_count,
            ]
        self._flush_if_needed()

    def add_unresolved_partner(
        self,
        identifier: str,
        provenance: ModelKGProvenance,
        occurrence_count: int = 1,
    ) -> None:
        if occurrence_count < 1:
            raise ValueError("occurrence_count must be positive.")
        buffered = self.unresolved_partner_buffer.get(identifier)
        if buffered is not None:
            buffered[-1] += occurrence_count
        else:
            self.unresolved_partner_buffer[identifier] = [
                identifier,
                identifier.casefold(),
                provenance.source_id,
                provenance.source_location,
                provenance.asset_sha256,
                provenance.extraction_method,
                occurrence_count,
            ]
        self._flush_if_needed()

    def _flush_if_needed(self) -> None:
        if (
            len(self.node_buffer)
            + len(self.edge_buffer)
            + len(self.unresolved_partner_buffer)
            >= self.buffer_limit
        ):
            self.flush()

    def flush(self) -> None:
        if self.node_buffer:
            self.connection.executemany(
                """
                INSERT INTO nodes (
                    node_id, node_type, name, properties_json,
                    provenance_json, occurrence_count
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    occurrence_count = occurrence_count
                        + excluded.occurrence_count
                """,
                self.node_buffer.values(),
            )
            self.node_buffer.clear()
        if self.edge_buffer:
            self.connection.executemany(
                """
                INSERT INTO edges (
                    edge_id, subject_id, subject_type, predicate,
                    object_id, object_type, properties_json,
                    provenance_json, occurrence_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(edge_id) DO UPDATE SET
                    occurrence_count = occurrence_count
                        + excluded.occurrence_count
                """,
                self.edge_buffer.values(),
            )
            self.edge_buffer.clear()
        if self.unresolved_partner_buffer:
            self.connection.executemany(
                """
                INSERT INTO unresolved_process_partners (
                    identifier, normalized_identifier, source_id,
                    source_location, asset_sha256, extraction_method,
                    occurrence_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(identifier) DO UPDATE SET
                    occurrence_count = occurrence_count
                        + excluded.occurrence_count
                """,
                self.unresolved_partner_buffer.values(),
            )
            self.unresolved_partner_buffer.clear()
        self.connection.commit()

    def finish(self) -> None:
        self.flush()
        self.connection.executescript(
            """
            CREATE INDEX edges_subject_idx
                ON edges(subject_id, predicate, object_id);
            CREATE INDEX edges_object_idx
                ON edges(object_id, predicate, subject_id);
            CREATE INDEX edges_predicate_idx
                ON edges(predicate, subject_type, object_type);
            CREATE INDEX nodes_type_idx ON nodes(node_type, name);
            CREATE INDEX unresolved_partner_normalized_idx
                ON unresolved_process_partners(normalized_identifier);
            """
        )
        self.connection.commit()
        self.connection.execute("PRAGMA optimize")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def _node(
    *,
    node_id: str,
    node_type: str,
    name: str,
    properties: dict[str, Any],
    provenance: ModelKGProvenance,
) -> ModelKGNode:
    return ModelKGNode(
        node_id=node_id,
        node_type=node_type,
        name=name,
        properties=properties,
        provenance=provenance,
    )


def _edge(
    *,
    subject_id: str,
    subject_type: str,
    predicate: str,
    object_id: str,
    object_type: str,
    provenance: ModelKGProvenance,
    properties: dict[str, Any] | None = None,
) -> ModelKGEdge:
    edge_properties = properties or {}
    return ModelKGEdge(
        edge_id=stable_id(
            "model-kg-edge",
            subject_id,
            predicate,
            object_id,
            edge_properties,
        ),
        subject_id=subject_id,
        subject_type=subject_type,
        predicate=predicate,
        object_id=object_id,
        object_type=object_type,
        properties=edge_properties,
        provenance=provenance,
    )


def _entity_id(node_type: str, name: str) -> str:
    return stable_id(
        f"model-kg-{node_type.casefold()}",
        name.strip().casefold(),
    )


def _entity(
    staging: _GraphStaging,
    *,
    node_type: str,
    name: str,
    provenance: ModelKGProvenance,
) -> str:
    node_id = _entity_id(node_type, name)
    staging.add_node(
        _node(
            node_id=node_id,
            node_type=node_type,
            name=name,
            properties={"normalized_name": name.strip().casefold()},
            provenance=provenance,
        )
    )
    return node_id


def _project_provenance(
    model: OSPModelVersionSnapshot,
) -> ModelKGProvenance:
    project_asset = next(
        (
            asset
            for asset in model.assets
            if asset.asset_kind == "pksim_project"
            and asset.relative_path == model.project_relative_path
        ),
        None,
    )
    if project_asset is None:
        raise ModelKGBuildError(
            f"ModelVersion has no project asset: {model.model_version_id}"
        )
    return _provenance(
        source_id=model.source_id,
        source_location=project_asset.source_location,
        asset_sha256=project_asset.sha256,
        extraction_method="osp_project_and_pkml_parser",
    )


def _add_model_version_core(
    staging: _GraphStaging,
    *,
    model: OSPModelVersionSnapshot,
    plan: OSPExecutionPlan,
) -> None:
    provenance = _project_provenance(model)
    validation_label = (
        model.validation_run_label.model_dump(mode="json")
        if model.validation_run_label is not None
        else None
    )
    staging.add_node(
        _node(
            node_id=model.model_id,
            node_type="PBPKModel",
            name=model.model_family,
            properties={
                "source_id": model.source_id,
                "repository_url": model.repository_url,
            },
            provenance=provenance,
        )
    )
    staging.add_node(
        _node(
            node_id=model.model_version_id,
            node_type="ModelVersion",
            name=f"{model.model_family} @ {model.repository_commit[:12]}",
            properties={
                "model_family": model.model_family,
                "project_key": model.project_key,
                "project_relative_path": model.project_relative_path,
                "project_sha256": model.project_sha256,
                "repository_commit": model.repository_commit,
                "software": model.software,
                "software_version": model.software_version,
                "license": model.license.model_dump(mode="json"),
                "static_parse_status": model.static_parse_status.value,
                "execution_probe_status": model.execution_probe_status.value,
                "pkml_parse_status": model.pkml_parse_status.value,
                "reusable_candidate": model.reusable_candidate,
                "blocking_reasons": model.blocking_reasons,
                "simulation_count": model.simulation_count,
                "parameter_record_count": model.parameter_record_count,
                "component_record_count": model.component_record_count,
                "process_record_count": model.process_record_count,
                "validation_run_id": (
                    model.validation_run_label.validation_run_id
                    if model.validation_run_label is not None
                    else None
                ),
                "validation_grade": (
                    model.validation_run_label.validation_grade
                    if model.validation_run_label is not None
                    else None
                ),
                "run_label_grade": (
                    model.validation_run_label.run_label_grade
                    if model.validation_run_label is not None
                    else 0
                ),
                "ranking_label": (
                    model.validation_run_label.ranking_label
                    if model.validation_run_label is not None
                    else None
                ),
                "validation_run_label": validation_label,
            },
            provenance=provenance,
        )
    )
    staging.add_edge(
        _edge(
            subject_id=model.model_id,
            subject_type="PBPKModel",
            predicate="HAS_VERSION",
            object_id=model.model_version_id,
            object_type="ModelVersion",
            provenance=provenance,
        )
    )

    runtime = plan.runtime.pksim_portable
    software_provenance = _provenance(
        source_id="osp_suite_release",
        source_location=runtime.url,
        asset_sha256=runtime.sha256,
        extraction_method="locked_runtime_manifest",
    )
    software_id = stable_id(
        "model-kg-software",
        model.software,
        model.software_version,
        runtime.sha256,
    )
    staging.add_node(
        _node(
            node_id=software_id,
            node_type="Software",
            name=f"{model.software} {model.software_version}",
            properties={
                "name": model.software,
                "version": model.software_version,
                "suite_version": plan.runtime.suite_version,
            },
            provenance=software_provenance,
        )
    )
    staging.add_edge(
        _edge(
            subject_id=model.model_version_id,
            subject_type="ModelVersion",
            predicate="IMPLEMENTED_IN",
            object_id=software_id,
            object_type="Software",
            provenance=software_provenance,
        )
    )

    structure_id = stable_id(
        "model-kg-structure",
        model.model_version_id,
        model.compartments,
        model.process_types,
    )
    staging.add_node(
        _node(
            node_id=structure_id,
            node_type="ModelStructure",
            name=f"{model.model_family} structure",
            properties={
                "compartment_count": len(model.compartments),
                "process_types": model.process_types,
            },
            provenance=provenance,
        )
    )
    staging.add_edge(
        _edge(
            subject_id=model.model_version_id,
            subject_type="ModelVersion",
            predicate="HAS_STRUCTURE",
            object_id=structure_id,
            object_type="ModelStructure",
            provenance=provenance,
        )
    )

    for compartment in model.compartments:
        entity_id = _entity(
            staging,
            node_type="Compartment",
            name=compartment,
            provenance=provenance,
        )
        staging.add_edge(
            _edge(
                subject_id=structure_id,
                subject_type="ModelStructure",
                predicate="HAS_COMPARTMENT",
                object_id=entity_id,
                object_type="Compartment",
                provenance=provenance,
            )
        )

    collections = (
        ("Compound", model.compounds, "MODELS_COMPOUND"),
        ("AdministrationRoute", model.routes, "SUPPORTS_ROUTE"),
        ("Formulation", model.formulations, "SUPPORTS_FORMULATION"),
        ("Population", model.populations, "APPLIES_TO"),
        ("Species", model.species, "APPLIES_TO"),
    )
    for node_type, names, predicate in collections:
        for name in names:
            entity_id = _entity(
                staging,
                node_type=node_type,
                name=name,
                provenance=provenance,
            )
            staging.add_edge(
                _edge(
                    subject_id=model.model_version_id,
                    subject_type="ModelVersion",
                    predicate=predicate,
                    object_id=entity_id,
                    object_type=node_type,
                    provenance=provenance,
                )
            )

    for asset in model.assets:
        asset_provenance = _provenance(
            source_id=model.source_id,
            source_location=asset.source_location,
            asset_sha256=asset.sha256,
            extraction_method="checksum_locked_asset_inventory",
        )
        asset_id = stable_id(
            "model-kg-asset",
            asset.source_location,
            asset.sha256,
        )
        staging.add_node(
            _node(
                node_id=asset_id,
                node_type="ModelAsset",
                name=Path(asset.relative_path).name,
                properties={
                    "asset_kind": asset.asset_kind,
                    "relative_path": asset.relative_path,
                    "sha256": asset.sha256,
                },
                provenance=asset_provenance,
            )
        )
        staging.add_edge(
            _edge(
                subject_id=model.model_version_id,
                subject_type="ModelVersion",
                predicate="HAS_ASSET",
                object_id=asset_id,
                object_type="ModelAsset",
                provenance=asset_provenance,
            )
        )


def _source_provenance(
    row: dict[str, Any],
    *,
    source_id: str,
    extraction_method: str,
) -> ModelKGProvenance:
    source = row["source"]
    return _provenance(
        source_id=source_id,
        source_location=str(source["source_location"]),
        asset_sha256=str(source["sha256"]),
        extraction_method=extraction_method,
    )


def _component_entity_type(component_type: str) -> str | None:
    return {
        "compound": "Compound",
        "metabolite": "Metabolite",
        "enzyme": "Enzyme",
        "transporter": "Transporter",
        "protein": "Protein",
        "molecular_complex": "MolecularComplex",
        "molecular_entity": "MolecularEntity",
        "compartment": "Compartment",
        "individual": "Population",
    }.get(component_type)


def _molecule_type(
    name: str,
    model: OSPModelVersionSnapshot,
    *,
    modifier: bool = False,
) -> str:
    folded = name.casefold()
    if folded in {item.casefold() for item in model.metabolites}:
        return "Metabolite"
    if folded in {item.casefold() for item in model.enzymes}:
        return "Enzyme"
    if folded in {item.casefold() for item in model.transporters}:
        return "Transporter"
    if modifier and (
        "cyp" in folded
        or "ugt" in folded
        or "enzyme" in folded
    ):
        return "Enzyme"
    return "Compound"


def _process_participant_semantics(
    process: dict[str, Any],
) -> tuple[list[dict[str, str]], list[str]]:
    semantics: list[dict[str, str]] = []
    unresolved: set[str] = set()
    for role in ("educt", "product", "modifier"):
        participant_field = f"{role}_participants"
        if participant_field not in process:
            raise ModelKGBuildError(
                f"Process record lacks {participant_field}; rebuild Gate 1 "
                "with osp_pkml_stream_parser >= 1.2.0."
            )
        for participant in process.get(participant_field) or []:
            identifier = str(participant["identifier"])
            if participant["resolution_status"] == "unresolved":
                unresolved.add(identifier)
                continue
            name = participant.get("name")
            component_type = participant.get("component_type")
            if not name or not component_type:
                raise ModelKGBuildError(
                    "Resolved process participant lacks name or type: "
                    f"{process.get('process_id')}:{identifier}"
                )
            entity_type = _component_entity_type(str(component_type))
            semantics.append(
                {
                    "role": role,
                    "name": str(name),
                    "component_type": str(component_type),
                    "entity_type": entity_type or "",
                }
            )
    projected_unresolved = {
        str(value)
        for value in process.get("unresolved_partner_identifiers") or []
    }
    if projected_unresolved != unresolved:
        raise ModelKGBuildError(
            "Process unresolved participant projection is inconsistent: "
            f"{process.get('process_id')}"
        )
    semantics.sort(
        key=lambda item: (
            item["role"],
            item["component_type"],
            item["name"].casefold(),
        )
    )
    return semantics, sorted(unresolved, key=str.casefold)


def _normalized_scope_path(value: str | None) -> str | None:
    if value is None:
        return None
    return _OPAQUE_PATH_ID.sub("[]", value)


def _add_reports(
    staging: _GraphStaging,
    *,
    reports_path: Path,
    versions_by_family: dict[str, list[OSPModelVersionSnapshot]],
) -> None:
    for report in _iter_jsonl(reports_path):
        relative_path = str(report["relative_path"])
        family = relative_path.split("/", 1)[0]
        versions = versions_by_family.get(family, [])
        if not versions:
            continue
        report_sha256 = str(report["asset_sha256"])
        commit = versions[0].repository_commit
        source_location = (
            f"GITHUB_BLOB:{commit}/{relative_path}"
        )
        provenance = _provenance(
            source_id=versions[0].source_id,
            source_location=source_location,
            asset_sha256=report_sha256,
            extraction_method=str(report["extraction_method"]),
        )
        document_id = stable_id(
            "model-kg-source-document",
            source_location,
            report_sha256,
        )
        staging.add_node(
            _node(
                node_id=document_id,
                node_type="SourceDocument",
                name=Path(relative_path).name,
                properties={
                    "relative_path": relative_path,
                    "format": report["format"],
                    "title": report.get("title"),
                    "section_count": len(report.get("sections") or []),
                    "table_count": len(report.get("tables") or []),
                },
                provenance=provenance,
            )
        )
        for model in versions:
            staging.add_edge(
                _edge(
                    subject_id=model.model_version_id,
                    subject_type="ModelVersion",
                    predicate="DESCRIBED_IN",
                    object_id=document_id,
                    object_type="SourceDocument",
                    provenance=provenance,
                )
            )

        for dataset in report.get("dataset_references") or []:
            role = str(dataset.get("role") or "unresolved")
            if role == "calibration":
                node_type = "CalibrationDataset"
                predicate = "CALIBRATED_WITH"
            elif role in {"qualification", "validation"}:
                node_type = "QualificationDataset"
                predicate = "VALIDATED_WITH"
            else:
                node_type = "ModelDataset"
                predicate = "USES_DATASET"
            evidence_id = stable_id(
                "model-kg-evidence",
                report_sha256,
                dataset["dataset_id"],
                dataset["source_location"],
            )
            evidence_provenance = _provenance(
                source_id=versions[0].source_id,
                source_location=str(dataset["source_location"]),
                asset_sha256=report_sha256,
                extraction_method=str(report["extraction_method"]),
                evidence_id=evidence_id,
            )
            dataset_id = stable_id(
                f"model-kg-{node_type.casefold()}",
                report_sha256,
                dataset["dataset_id"],
            )
            staging.add_node(
                _node(
                    node_id=dataset_id,
                    node_type=node_type,
                    name=str(dataset.get("name") or dataset["dataset_id"]),
                    properties={
                        "role": role,
                        "reference": dataset.get("reference"),
                        "description": dataset.get("description"),
                        "table_id": dataset.get("table_id"),
                        "row_number": dataset.get("row_number"),
                    },
                    provenance=evidence_provenance,
                )
            )
            staging.add_node(
                _node(
                    node_id=evidence_id,
                    node_type="Evidence",
                    name=f"{node_type} evidence",
                    properties={
                        "evidence_kind": "report_dataset_reference",
                        "supporting_text": dataset.get("description")
                        or dataset.get("reference")
                        or dataset.get("name"),
                        "role": role,
                    },
                    provenance=evidence_provenance,
                )
            )
            staging.add_edge(
                _edge(
                    subject_id=dataset_id,
                    subject_type=node_type,
                    predicate="SUPPORTED_BY",
                    object_id=evidence_id,
                    object_type="Evidence",
                    provenance=evidence_provenance,
                )
            )
            staging.add_edge(
                _edge(
                    subject_id=evidence_id,
                    subject_type="Evidence",
                    predicate="DERIVED_FROM",
                    object_id=document_id,
                    object_type="SourceDocument",
                    provenance=evidence_provenance,
                )
            )
            for model in versions:
                staging.add_edge(
                    _edge(
                        subject_id=model.model_version_id,
                        subject_type="ModelVersion",
                        predicate=predicate,
                        object_id=dataset_id,
                        object_type=node_type,
                        provenance=evidence_provenance,
                        properties={"dataset_role": role},
                    )
                )

        metrics = report.get("metrics") or []
        if not metrics:
            continue
        for model in versions:
            run_id = stable_id(
                "model-kg-validation-run",
                model.model_version_id,
                report_sha256,
            )
            staging.add_node(
                _node(
                    node_id=run_id,
                    node_type="ValidationRun",
                    name=f"{model.model_family} reported validation",
                    properties={
                        "status": "reported_not_reproduced",
                        "report_relative_path": relative_path,
                    },
                    provenance=provenance,
                )
            )
            staging.add_edge(
                _edge(
                    subject_id=model.model_version_id,
                    subject_type="ModelVersion",
                    predicate="HAS_VALIDATION_RUN",
                    object_id=run_id,
                    object_type="ValidationRun",
                    provenance=provenance,
                )
            )
            for metric in metrics:
                evidence_id = stable_id(
                    "model-kg-evidence",
                    report_sha256,
                    metric["metric_id"],
                    metric["source_location"],
                )
                metric_provenance = _provenance(
                    source_id=model.source_id,
                    source_location=str(metric["source_location"]),
                    asset_sha256=report_sha256,
                    extraction_method=str(report["extraction_method"]),
                    evidence_id=evidence_id,
                )
                metric_id = stable_id(
                    "model-kg-validation-metric",
                    model.model_version_id,
                    report_sha256,
                    metric["metric_id"],
                )
                staging.add_node(
                    _node(
                        node_id=metric_id,
                        node_type="ValidationMetric",
                        name=str(metric["metric_name"]),
                        properties={
                            "value": metric.get("value"),
                            "unit": metric.get("unit"),
                            "group": metric.get("group"),
                            "status": "reported_not_reproduced",
                        },
                        provenance=metric_provenance,
                    )
                )
                staging.add_node(
                    _node(
                        node_id=evidence_id,
                        node_type="Evidence",
                        name=f"{metric['metric_name']} evidence",
                        properties={
                            "evidence_kind": "reported_validation_metric",
                            "metric_name": metric["metric_name"],
                            "value": metric.get("value"),
                            "unit": metric.get("unit"),
                            "group": metric.get("group"),
                        },
                        provenance=metric_provenance,
                    )
                )
                staging.add_edge(
                    _edge(
                        subject_id=run_id,
                        subject_type="ValidationRun",
                        predicate="REPORTS",
                        object_id=metric_id,
                        object_type="ValidationMetric",
                        provenance=metric_provenance,
                    )
                )
                staging.add_edge(
                    _edge(
                        subject_id=metric_id,
                        subject_type="ValidationMetric",
                        predicate="SUPPORTED_BY",
                        object_id=evidence_id,
                        object_type="Evidence",
                        provenance=metric_provenance,
                    )
                )
                staging.add_edge(
                    _edge(
                        subject_id=evidence_id,
                        subject_type="Evidence",
                        predicate="DERIVED_FROM",
                        object_id=document_id,
                        object_type="SourceDocument",
                        provenance=metric_provenance,
                    )
                )


def _add_validation_runs(
    staging: _GraphStaging,
    *,
    validation_runs_path: Path,
    version_by_id: dict[str, OSPModelVersionSnapshot],
) -> int:
    run_count = 0
    for row in _iter_jsonl(validation_runs_path):
        model_version_id = str(row.get("model_version_id") or "")
        model = version_by_id.get(model_version_id)
        if model is None:
            raise ModelKGBuildError(
                "Validation run references an unknown ModelVersion: "
                f"{model_version_id}"
            )
        label = ModelValidationRunLabel.model_validate(row.get("label"))
        if model.validation_run_label is None or (
            model.validation_run_label.model_dump(mode="json")
            != label.model_dump(mode="json")
        ):
            raise ModelKGBuildError(
                "Validation run label does not match the ModelVersion "
                f"snapshot: {model_version_id}"
            )
        datasets = row.get("datasets")
        if not isinstance(datasets, list):
            raise ModelKGBuildError(
                f"Validation datasets must be a list: {label.validation_run_id}"
            )
        if len(datasets) != label.verification_dataset_count:
            raise ModelKGBuildError(
                "Validation dataset count does not match the grade label: "
                f"{label.validation_run_id}"
            )

        audit_provenance = _provenance(
            source_id=model.source_id,
            source_location=(
                f"VALIDATION_AUDIT:{label.validation_run_id}"
            ),
            asset_sha256=label.audit_report_sha256,
            extraction_method="licensed_snapshot_validation_auditor",
        )
        audit_document_id = stable_id(
            "model-kg-source-document",
            label.audit_report_sha256,
            label.validation_run_id,
        )
        staging.add_node(
            _node(
                node_id=audit_document_id,
                node_type="SourceDocument",
                name="snapshot_reuse_audit.json",
                properties={
                    "document_kind": "independent_validation_audit",
                    "sha256": label.audit_report_sha256,
                    "validation_run_id": label.validation_run_id,
                },
                provenance=audit_provenance,
            )
        )
        staging.add_edge(
            _edge(
                subject_id=model.model_version_id,
                subject_type="ModelVersion",
                predicate="DESCRIBED_IN",
                object_id=audit_document_id,
                object_type="SourceDocument",
                provenance=audit_provenance,
            )
        )

        snapshot_asset = next(
            (
                asset
                for asset in model.assets
                if asset.asset_kind == "repository_snapshot"
                and asset.sha256 == label.source_snapshot_sha256
            ),
            None,
        )
        if snapshot_asset is None:
            raise ModelKGBuildError(
                "Grade-3 validation label is not bound to a registered "
                f"repository snapshot: {model_version_id}"
            )
        snapshot_provenance = _provenance(
            source_id=model.source_id,
            source_location=snapshot_asset.source_location,
            asset_sha256=snapshot_asset.sha256,
            extraction_method="licensed_snapshot_protocol_parser",
        )
        snapshot_document_id = stable_id(
            "model-kg-source-document",
            snapshot_asset.source_location,
            snapshot_asset.sha256,
        )
        staging.add_node(
            _node(
                node_id=snapshot_document_id,
                node_type="SourceDocument",
                name=Path(snapshot_asset.relative_path).name,
                properties={
                    "document_kind": "locked_model_snapshot",
                    "relative_path": snapshot_asset.relative_path,
                    "sha256": snapshot_asset.sha256,
                },
                provenance=snapshot_provenance,
            )
        )
        staging.add_edge(
            _edge(
                subject_id=model.model_version_id,
                subject_type="ModelVersion",
                predicate="DESCRIBED_IN",
                object_id=snapshot_document_id,
                object_type="SourceDocument",
                provenance=snapshot_provenance,
            )
        )

        run_id = label.validation_run_id
        staging.add_node(
            _node(
                node_id=run_id,
                node_type="ValidationRun",
                name=f"{model.model_family} external validation",
                properties={
                    "status": label.validation_grade,
                    "run_label_grade": label.run_label_grade,
                    "ranking_label": label.ranking_label,
                    "engineering_accepted": label.engineering_accepted,
                    "scientific_accepted": label.scientific_accepted,
                    "reusable_candidate": label.reusable_candidate,
                    "protocol_id": label.protocol_id,
                    "verification_dataset_count": (
                        label.verification_dataset_count
                    ),
                    "pooled_observation_count": (
                        label.pooled_observation_count
                    ),
                    "validated_at": label.validated_at.isoformat(),
                    "protocol_sha256": label.protocol_sha256,
                    "windows_manifest_sha256": (
                        label.windows_manifest_sha256
                    ),
                    "audit_report_sha256": label.audit_report_sha256,
                    "source_snapshot_sha256": (
                        label.source_snapshot_sha256
                    ),
                },
                provenance=audit_provenance,
            )
        )
        staging.add_edge(
            _edge(
                subject_id=model.model_version_id,
                subject_type="ModelVersion",
                predicate="HAS_VALIDATION_RUN",
                object_id=run_id,
                object_type="ValidationRun",
                provenance=audit_provenance,
            )
        )
        run_evidence_id = stable_id(
            "model-kg-evidence",
            run_id,
            "grade-label",
            label.audit_report_sha256,
        )
        run_evidence_provenance = _provenance(
            source_id=model.source_id,
            source_location=(
                f"VALIDATION_AUDIT:{run_id}#/validation_grade"
            ),
            asset_sha256=label.audit_report_sha256,
            extraction_method="licensed_snapshot_validation_auditor",
            evidence_id=run_evidence_id,
        )
        staging.add_node(
            _node(
                node_id=run_evidence_id,
                node_type="Evidence",
                name="external validation acceptance evidence",
                properties={
                    "evidence_kind": "reproduced_validation_grade",
                    "supporting_text": (
                        f"{model.model_family} passed the locked grade "
                        f"{label.run_label_grade} validation protocol with "
                        f"{label.verification_dataset_count} external datasets."
                    ),
                    "validation_run_id": run_id,
                },
                provenance=run_evidence_provenance,
            )
        )
        staging.add_edge(
            _edge(
                subject_id=run_id,
                subject_type="ValidationRun",
                predicate="SUPPORTED_BY",
                object_id=run_evidence_id,
                object_type="Evidence",
                provenance=run_evidence_provenance,
            )
        )
        staging.add_edge(
            _edge(
                subject_id=run_evidence_id,
                subject_type="Evidence",
                predicate="DERIVED_FROM",
                object_id=audit_document_id,
                object_type="SourceDocument",
                provenance=run_evidence_provenance,
            )
        )

        pooled_metrics = (
            ("pooled_gmfe", label.pooled_gmfe),
            ("pooled_geometric_bias", label.pooled_geometric_bias),
            ("pooled_two_fold_fraction", label.pooled_two_fold_fraction),
            (
                "endpoint_scenario_pass_fraction",
                label.endpoint_scenario_pass_fraction,
            ),
        )
        for metric_name, value in pooled_metrics:
            if value is None:
                continue
            metric_id = stable_id(
                "model-kg-validation-metric",
                run_id,
                "pooled",
                metric_name,
            )
            metric_provenance = _provenance(
                source_id=model.source_id,
                source_location=(
                    f"VALIDATION_AUDIT:{run_id}#/pooled_metrics/"
                    f"{metric_name}"
                ),
                asset_sha256=label.audit_report_sha256,
                extraction_method="licensed_snapshot_validation_auditor",
                evidence_id=run_evidence_id,
            )
            staging.add_node(
                _node(
                    node_id=metric_id,
                    node_type="ValidationMetric",
                    name=metric_name,
                    properties={
                        "value": value,
                        "scope": "pooled_external_validation",
                        "status": "reproduced",
                    },
                    provenance=metric_provenance,
                )
            )
            staging.add_edge(
                _edge(
                    subject_id=run_id,
                    subject_type="ValidationRun",
                    predicate="REPORTS",
                    object_id=metric_id,
                    object_type="ValidationMetric",
                    provenance=metric_provenance,
                )
            )
            staging.add_edge(
                _edge(
                    subject_id=metric_id,
                    subject_type="ValidationMetric",
                    predicate="SUPPORTED_BY",
                    object_id=run_evidence_id,
                    object_type="Evidence",
                    provenance=metric_provenance,
                )
            )

        dataset_metric_names = (
            "gmfe",
            "geometric_bias",
            "two_fold_fraction",
            "log_rmse",
            "auc0_last_ratio",
            "cmax_ratio",
        )
        for dataset in datasets:
            if not isinstance(dataset, dict) or not dataset.get("dataset_name"):
                raise ModelKGBuildError(
                    f"Malformed validation dataset in {run_id}."
                )
            dataset_name = str(dataset["dataset_name"])
            dataset_id = stable_id(
                "model-kg-qualification-dataset",
                run_id,
                dataset_name,
                dataset.get("source_location"),
            )
            dataset_location = str(
                dataset.get("source_location")
                or snapshot_asset.source_location
            )
            dataset_evidence_id = stable_id(
                "model-kg-evidence",
                dataset_id,
                label.source_snapshot_sha256,
            )
            dataset_provenance = _provenance(
                source_id=model.source_id,
                source_location=dataset_location,
                asset_sha256=label.source_snapshot_sha256,
                extraction_method="licensed_snapshot_protocol_parser",
                evidence_id=dataset_evidence_id,
            )
            dataset_properties = {
                key: dataset.get(key)
                for key in (
                    "reference",
                    "simulation_name",
                    "simulation_sequence",
                    "observation_count",
                    "positive_pair_count",
                    "endpoint_evaluable",
                    "endpoint_passed",
                    "dataset_guard_passed",
                    *dataset_metric_names,
                )
            }
            dataset_properties.update(
                {
                    "role": "external_validation",
                    "validation_run_id": run_id,
                    "source_location": dataset_location,
                }
            )
            staging.add_node(
                _node(
                    node_id=dataset_id,
                    node_type="QualificationDataset",
                    name=dataset_name,
                    properties=dataset_properties,
                    provenance=dataset_provenance,
                )
            )
            staging.add_node(
                _node(
                    node_id=dataset_evidence_id,
                    node_type="Evidence",
                    name="external validation dataset evidence",
                    properties={
                        "evidence_kind": "frozen_external_dataset",
                        "supporting_text": " | ".join(
                            value
                            for value in (
                                dataset_name,
                                str(dataset.get("reference") or ""),
                            )
                            if value
                        ),
                        "validation_run_id": run_id,
                    },
                    provenance=dataset_provenance,
                )
            )
            for subject_id, subject_type in (
                (model.model_version_id, "ModelVersion"),
                (run_id, "ValidationRun"),
            ):
                staging.add_edge(
                    _edge(
                        subject_id=subject_id,
                        subject_type=subject_type,
                        predicate="VALIDATED_WITH",
                        object_id=dataset_id,
                        object_type="QualificationDataset",
                        provenance=dataset_provenance,
                    )
                )
            staging.add_edge(
                _edge(
                    subject_id=dataset_id,
                    subject_type="QualificationDataset",
                    predicate="SUPPORTED_BY",
                    object_id=dataset_evidence_id,
                    object_type="Evidence",
                    provenance=dataset_provenance,
                )
            )
            staging.add_edge(
                _edge(
                    subject_id=dataset_evidence_id,
                    subject_type="Evidence",
                    predicate="DERIVED_FROM",
                    object_id=snapshot_document_id,
                    object_type="SourceDocument",
                    provenance=dataset_provenance,
                )
            )

            for metric_name in dataset_metric_names:
                value = dataset.get(metric_name)
                if value is None:
                    continue
                metric_id = stable_id(
                    "model-kg-validation-metric",
                    run_id,
                    dataset_id,
                    metric_name,
                )
                metric_evidence_id = stable_id(
                    "model-kg-evidence",
                    metric_id,
                    label.audit_report_sha256,
                )
                metric_location = (
                    f"VALIDATION_AUDIT:{run_id}#/datasets/"
                    f"{dataset.get('simulation_sequence')}/{metric_name}"
                )
                metric_provenance = _provenance(
                    source_id=model.source_id,
                    source_location=metric_location,
                    asset_sha256=label.audit_report_sha256,
                    extraction_method="licensed_snapshot_validation_auditor",
                    evidence_id=metric_evidence_id,
                )
                staging.add_node(
                    _node(
                        node_id=metric_id,
                        node_type="ValidationMetric",
                        name=metric_name,
                        properties={
                            "value": value,
                            "scope": "external_validation_dataset",
                            "dataset_id": dataset_id,
                            "status": "reproduced",
                        },
                        provenance=metric_provenance,
                    )
                )
                staging.add_node(
                    _node(
                        node_id=metric_evidence_id,
                        node_type="Evidence",
                        name=f"{metric_name} reproduced metric evidence",
                        properties={
                            "evidence_kind": "reproduced_validation_metric",
                            "supporting_text": (
                                f"{dataset_name}: {metric_name}={value}"
                            ),
                            "validation_run_id": run_id,
                        },
                        provenance=metric_provenance,
                    )
                )
                for predicate, object_id, object_type in (
                    ("REPORTS", metric_id, "ValidationMetric"),
                ):
                    staging.add_edge(
                        _edge(
                            subject_id=run_id,
                            subject_type="ValidationRun",
                            predicate=predicate,
                            object_id=object_id,
                            object_type=object_type,
                            provenance=metric_provenance,
                        )
                    )
                staging.add_edge(
                    _edge(
                        subject_id=metric_id,
                        subject_type="ValidationMetric",
                        predicate="EVALUATES_DATASET",
                        object_id=dataset_id,
                        object_type="QualificationDataset",
                        provenance=metric_provenance,
                    )
                )
                staging.add_edge(
                    _edge(
                        subject_id=metric_id,
                        subject_type="ValidationMetric",
                        predicate="SUPPORTED_BY",
                        object_id=metric_evidence_id,
                        object_type="Evidence",
                        provenance=metric_provenance,
                    )
                )
                staging.add_edge(
                    _edge(
                        subject_id=metric_evidence_id,
                        subject_type="Evidence",
                        predicate="DERIVED_FROM",
                        object_id=audit_document_id,
                        object_type="SourceDocument",
                        provenance=metric_provenance,
                    )
                )
        run_count += 1
    return run_count


def _node_rows(connection: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    for row in connection.execute(
        """
        SELECT node_id, node_type, name, properties_json,
               provenance_json, occurrence_count
        FROM nodes ORDER BY node_id
        """
    ):
        yield {
            "node_id": row[0],
            "node_type": row[1],
            "name": row[2],
            "properties": json.loads(row[3]),
            "provenance": json.loads(row[4]),
            "occurrence_count": row[5],
        }


def _edge_rows(connection: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    for row in connection.execute(
        """
        SELECT edge_id, subject_id, subject_type, predicate,
               object_id, object_type, properties_json,
               provenance_json, occurrence_count
        FROM edges ORDER BY edge_id
        """
    ):
        yield {
            "edge_id": row[0],
            "subject_id": row[1],
            "subject_type": row[2],
            "predicate": row[3],
            "object_id": row[4],
            "object_type": row[5],
            "properties": json.loads(row[6]),
            "provenance": json.loads(row[7]),
            "occurrence_count": row[8],
        }


def _unresolved_partner_rows(
    connection: sqlite3.Connection,
) -> Iterator[dict[str, Any]]:
    for row in connection.execute(
        """
        SELECT identifier, normalized_identifier, source_id,
               source_location, asset_sha256, extraction_method,
               occurrence_count
        FROM unresolved_process_partners ORDER BY identifier
        """
    ):
        yield {
            "identifier": row[0],
            "normalized_identifier": row[1],
            "provenance": {
                "source_id": row[2],
                "source_location": row[3],
                "asset_sha256": row[4],
                "extraction_method": row[5],
            },
            "occurrence_count": row[6],
        }


def _counts(
    connection: sqlite3.Connection,
    table: str,
    column: str,
) -> dict[str, int]:
    if (table, column) not in {
        ("nodes", "node_type"),
        ("edges", "predicate"),
    }:
        raise ValueError("Unsupported count query.")
    return {
        str(key): int(value)
        for key, value in connection.execute(
            f"SELECT {column}, COUNT(*) FROM {table} GROUP BY {column} "
            f"ORDER BY {column}"
        )
    }


def build_model_kg(
    *,
    model_version_directory: Path,
    gate1_audit_path: Path,
    plan_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    model_version_directory = model_version_directory.resolve()
    output_directory = output_directory.resolve()
    working_directory = output_directory.with_name(
        output_directory.name + ".tmp"
    )
    if output_directory.exists() or working_directory.exists():
        raise ModelKGBuildError(
            f"Output or temporary directory already exists: {output_directory}"
        )
    working_directory.mkdir(parents=True)

    gate1_audit = _read_json(gate1_audit_path)
    if not gate1_audit.get("accepted"):
        raise ModelKGBuildError("Gate 1 audit is not accepted.")
    build_summary = OSPModelVersionBuildSummary.model_validate(
        _read_json(model_version_directory / "summary.json")
    )
    plan = OSPExecutionPlan.model_validate(_read_json(plan_path))
    if (
        gate1_audit.get("input_checksums", {}).get(
            "model_version_summary"
        )
        != _sha256_file(model_version_directory / "summary.json")
    ):
        raise ModelKGBuildError(
            "Gate 1 audit does not cover this ModelVersion package."
        )

    model_versions = [
        OSPModelVersionSnapshot.model_validate(row)
        for row in _iter_jsonl(
            _fact_path(model_version_directory, "model_versions")
        )
    ]
    if len(model_versions) != build_summary.model_version_count:
        raise ModelKGBuildError("ModelVersion count does not match summary.")
    version_by_project = {
        model.project_key: model for model in model_versions
    }
    version_by_id = {
        model.model_version_id: model for model in model_versions
    }
    versions_by_family: dict[
        str, list[OSPModelVersionSnapshot]
    ] = defaultdict(list)
    for model in model_versions:
        versions_by_family[model.model_family].append(model)

    staging = _GraphStaging(working_directory / "model_kg.sqlite")
    progress = {
        "components": 0,
        "processes": 0,
        "parameters": 0,
        "simulations": 0,
        "administrations": 0,
        "reports": 0,
        "validation_runs": 0,
        "unresolved_process_partner_occurrences": 0,
    }
    scenario_ids: dict[tuple[str, str], str] = {}
    try:
        for model in model_versions:
            _add_model_version_core(staging, model=model, plan=plan)

        for simulation in _iter_jsonl(
            _fact_path(model_version_directory, "simulations")
        ):
            model = version_by_project[str(simulation["project_key"])]
            provenance = _provenance(
                source_id=model.source_id,
                source_location=str(
                    simulation["pkml_source"]["source_location"]
                ),
                asset_sha256=str(simulation["pkml_source"]["sha256"]),
                extraction_method="osp_pkml_stream_parser",
            )
            scenario_id = stable_id(
                "model-kg-simulation",
                model.model_version_id,
                simulation["simulation_id"],
            )
            scenario_ids[
                (model.project_key, str(simulation["simulation_id"]))
            ] = scenario_id
            csv_results = simulation.get("csv_results") or []
            staging.add_node(
                _node(
                    node_id=scenario_id,
                    node_type="SimulationScenario",
                    name=str(simulation["simulation_name"]),
                    properties={
                        "simulation_id": simulation["simulation_id"],
                        "embedded_simulation_id": simulation.get(
                            "embedded_simulation_id"
                        ),
                        "solver": simulation.get("solver"),
                        "parse_status": simulation["parse_status"],
                        "result_kinds": sorted(
                            {
                                str(item["result_kind"])
                                for item in csv_results
                            }
                        ),
                        "result_file_count": len(csv_results),
                        "non_finite_value_count": sum(
                            int(item["non_finite_value_count"])
                            for item in csv_results
                        ),
                        "parameter_count": simulation["parameter_count"],
                        "component_count": simulation["component_count"],
                        "process_count": simulation["process_count"],
                        "administration_count": simulation[
                            "administration_count"
                        ],
                    },
                    provenance=provenance,
                )
            )
            staging.add_edge(
                _edge(
                    subject_id=model.model_version_id,
                    subject_type="ModelVersion",
                    predicate="HAS_SIMULATION",
                    object_id=scenario_id,
                    object_type="SimulationScenario",
                    provenance=provenance,
                )
            )
            for node_type, values, predicate in (
                (
                    "AdministrationRoute",
                    simulation.get("routes") or [],
                    "SUPPORTS_ROUTE",
                ),
                (
                    "Formulation",
                    simulation.get("formulations") or [],
                    "SUPPORTS_FORMULATION",
                ),
                (
                    "Population",
                    simulation.get("populations") or [],
                    "APPLIES_TO",
                ),
                (
                    "Species",
                    simulation.get("species") or [],
                    "APPLIES_TO",
                ),
            ):
                for value in values:
                    entity_id = _entity(
                        staging,
                        node_type=node_type,
                        name=str(value),
                        provenance=provenance,
                    )
                    staging.add_edge(
                        _edge(
                            subject_id=scenario_id,
                            subject_type="SimulationScenario",
                            predicate=predicate,
                            object_id=entity_id,
                            object_type=node_type,
                            provenance=provenance,
                        )
                    )
            progress["simulations"] += 1

        for component in _iter_jsonl(
            _fact_path(model_version_directory, "components")
        ):
            model = version_by_project[str(component["project_key"])]
            provenance = _source_provenance(
                component,
                source_id=model.source_id,
                extraction_method="osp_pkml_stream_parser",
            )
            component_id = stable_id(
                "model-kg-component",
                model.model_version_id,
                component["component_type"],
                component["name"],
                component["element_type"],
                component.get("dimension"),
                component.get("display_unit"),
            )
            staging.add_node(
                _node(
                    node_id=component_id,
                    node_type="ModelComponent",
                    name=str(component["name"]),
                    properties={
                        "component_kind": component["component_type"],
                        "element_type": component["element_type"],
                        "dimension": component.get("dimension"),
                        "display_unit": component.get("display_unit"),
                        "component_variants_in_fact_layer": True,
                        "fact_layer": "components.jsonl.gz",
                    },
                    provenance=provenance,
                )
            )
            staging.add_edge(
                _edge(
                    subject_id=model.model_version_id,
                    subject_type="ModelVersion",
                    predicate="HAS_COMPONENT",
                    object_id=component_id,
                    object_type="ModelComponent",
                    provenance=provenance,
                )
            )
            entity_type = _component_entity_type(
                str(component["component_type"])
            )
            if entity_type:
                entity_id = _entity(
                    staging,
                    node_type=entity_type,
                    name=str(component["name"]),
                    provenance=provenance,
                )
                staging.add_edge(
                    _edge(
                        subject_id=component_id,
                        subject_type="ModelComponent",
                        predicate="REPRESENTS",
                        object_id=entity_id,
                        object_type=entity_type,
                        provenance=provenance,
                    )
                )
            progress["components"] += 1
            if progress["components"] % 250_000 == 0:
                print(
                    canonical_json(
                        {
                            "phase": "components",
                            "records_read": progress["components"],
                        }
                    ),
                    flush=True,
                )

        for process in _iter_jsonl(
            _fact_path(model_version_directory, "processes")
        ):
            model = version_by_project[str(process["project_key"])]
            provenance = _source_provenance(
                process,
                source_id=model.source_id,
                extraction_method="osp_pkml_stream_parser",
            )
            participant_semantics, unresolved_partners = (
                _process_participant_semantics(process)
            )
            participant_key = [
                {
                    "role": participant["role"],
                    "name": participant["name"].strip().casefold(),
                    "component_type": participant["component_type"],
                }
                for participant in participant_semantics
            ]
            process_id = stable_id(
                "model-kg-component",
                model.model_version_id,
                "process",
                process["name"],
                process["process_type"],
                process["element_type"],
                participant_key,
            )
            staging.add_node(
                _node(
                    node_id=process_id,
                    node_type="ModelComponent",
                    name=str(process["name"]),
                    properties={
                        "component_kind": "process",
                        "process_type": process["process_type"],
                        "element_type": process["element_type"],
                        "participants": participant_semantics,
                        "process_variants_in_fact_layer": True,
                        "fact_layer": "processes.jsonl.gz",
                    },
                    provenance=provenance,
                )
            )
            staging.add_edge(
                _edge(
                    subject_id=model.model_version_id,
                    subject_type="ModelVersion",
                    predicate="HAS_COMPONENT",
                    object_id=process_id,
                    object_type="ModelComponent",
                    provenance=provenance,
                )
            )
            predicate_by_role = {
                "educt": "HAS_EDUCT",
                "product": "HAS_PRODUCT",
                "modifier": "MODIFIED_BY",
            }
            for participant in participant_semantics:
                entity_type = participant["entity_type"]
                if entity_type not in {
                    "Compound",
                    "Metabolite",
                    "Enzyme",
                    "Transporter",
                    "Protein",
                    "MolecularComplex",
                    "MolecularEntity",
                }:
                    continue
                value = participant["name"]
                predicate = predicate_by_role[participant["role"]]
                entity_id = _entity(
                    staging,
                    node_type=entity_type,
                    name=value,
                    provenance=provenance,
                )
                staging.add_edge(
                    _edge(
                        subject_id=process_id,
                        subject_type="ModelComponent",
                        predicate=predicate,
                        object_id=entity_id,
                        object_type=entity_type,
                        provenance=provenance,
                    )
                )
            for identifier in unresolved_partners:
                staging.add_unresolved_partner(identifier, provenance)
                progress[
                    "unresolved_process_partner_occurrences"
                ] += 1
            progress["processes"] += 1
            if progress["processes"] % 100_000 == 0:
                print(
                    canonical_json(
                        {
                            "phase": "processes",
                            "records_read": progress["processes"],
                        }
                    ),
                    flush=True,
                )

        for parameter in _iter_jsonl(
            _fact_path(model_version_directory, "parameters")
        ):
            model = version_by_project[str(parameter["project_key"])]
            provenance = _source_provenance(
                parameter,
                source_id=model.source_id,
                extraction_method="osp_pkml_stream_parser",
            )
            normalized_scope = _normalized_scope_path(
                parameter.get("scope_path")
            )
            parameter_id = stable_id(
                "model-kg-parameter",
                model.model_version_id,
                parameter["name"],
                normalized_scope,
                parameter["element_type"],
                parameter.get("dimension"),
                parameter.get("display_unit"),
                parameter.get("quantity_type"),
            )
            staging.add_node(
                _node(
                    node_id=parameter_id,
                    node_type="ModelParameter",
                    name=str(parameter["name"]),
                    properties={
                        "element_type": parameter["element_type"],
                        "dimension": parameter.get("dimension"),
                        "display_unit": parameter.get("display_unit"),
                        "quantity_type": parameter.get("quantity_type"),
                        "normalized_scope_path": normalized_scope,
                        "value_variants_in_fact_layer": True,
                        "fact_layer": "parameters.jsonl.gz",
                    },
                    provenance=provenance,
                )
            )
            staging.add_edge(
                _edge(
                    subject_id=model.model_version_id,
                    subject_type="ModelVersion",
                    predicate="USES_PARAMETER",
                    object_id=parameter_id,
                    object_type="ModelParameter",
                    provenance=provenance,
                )
            )
            progress["parameters"] += 1
            if progress["parameters"] % 500_000 == 0:
                print(
                    canonical_json(
                        {
                            "phase": "parameters",
                            "records_read": progress["parameters"],
                        }
                    ),
                    flush=True,
                )

        for administration in _iter_jsonl(
            _fact_path(model_version_directory, "administrations")
        ):
            model = version_by_project[str(administration["project_key"])]
            provenance = _source_provenance(
                administration,
                source_id=model.source_id,
                extraction_method="osp_pkml_stream_parser",
            )
            scenario_id = scenario_ids[
                (
                    model.project_key,
                    str(administration["simulation_id"]),
                )
            ]
            administration_id = stable_id(
                "model-kg-administration",
                model.model_version_id,
                administration["simulation_id"],
                administration["administration_id"],
                administration["name"],
            )
            staging.add_node(
                _node(
                    node_id=administration_id,
                    node_type="AdministrationEvent",
                    name=str(administration["name"]),
                    properties={
                        "administration_id": administration[
                            "administration_id"
                        ],
                        "molecule_name": administration.get(
                            "molecule_name"
                        ),
                        "route": administration.get("route"),
                        "formulation": administration.get("formulation"),
                    },
                    provenance=provenance,
                )
            )
            staging.add_edge(
                _edge(
                    subject_id=scenario_id,
                    subject_type="SimulationScenario",
                    predicate="HAS_ADMINISTRATION",
                    object_id=administration_id,
                    object_type="AdministrationEvent",
                    provenance=provenance,
                )
            )
            molecule_name = administration.get("molecule_name")
            if molecule_name:
                entity_type = _molecule_type(str(molecule_name), model)
                entity_id = _entity(
                    staging,
                    node_type=entity_type,
                    name=str(molecule_name),
                    provenance=provenance,
                )
                staging.add_edge(
                    _edge(
                        subject_id=administration_id,
                        subject_type="AdministrationEvent",
                        predicate="ADMINISTERS_COMPOUND",
                        object_id=entity_id,
                        object_type=entity_type,
                        provenance=provenance,
                    )
                )
            for field, node_type, predicate in (
                ("route", "AdministrationRoute", "USES_ROUTE"),
                ("formulation", "Formulation", "USES_FORMULATION"),
            ):
                value = administration.get(field)
                if not value:
                    continue
                entity_id = _entity(
                    staging,
                    node_type=node_type,
                    name=str(value),
                    provenance=provenance,
                )
                staging.add_edge(
                    _edge(
                        subject_id=administration_id,
                        subject_type="AdministrationEvent",
                        predicate=predicate,
                        object_id=entity_id,
                        object_type=node_type,
                        provenance=provenance,
                    )
                )
            progress["administrations"] += 1

        reports_path = _fact_path(model_version_directory, "reports")
        _add_reports(
            staging,
            reports_path=reports_path,
            versions_by_family=versions_by_family,
        )
        progress["reports"] = sum(
            1
            for _ in _iter_jsonl(reports_path)
        )
        validation_runs_candidates = (
            model_version_directory / "validation_runs.jsonl",
            model_version_directory / "validation_runs.jsonl.gz",
        )
        validation_runs_paths = [
            path for path in validation_runs_candidates if path.is_file()
        ]
        if len(validation_runs_paths) > 1:
            raise ModelKGBuildError(
                "Both compressed and uncompressed validation run inputs exist."
            )
        if validation_runs_paths:
            progress["validation_runs"] = _add_validation_runs(
                staging,
                validation_runs_path=validation_runs_paths[0],
                version_by_id=version_by_id,
            )
        staging.finish()

        node_type_counts = _counts(
            staging.connection,
            "nodes",
            "node_type",
        )
        predicate_counts = _counts(
            staging.connection,
            "edges",
            "predicate",
        )
        node_count = sum(node_type_counts.values())
        edge_count = sum(predicate_counts.values())
        unresolved_process_partner_unique_count = int(
            staging.connection.execute(
                "SELECT COUNT(*) FROM unresolved_process_partners"
            ).fetchone()[0]
        )
        output_checksums = {
            "model_kg_nodes.jsonl.gz": _write_gzip_jsonl_atomic(
                working_directory / "model_kg_nodes.jsonl.gz",
                _node_rows(staging.connection),
            ),
            "model_kg_edges.jsonl.gz": _write_gzip_jsonl_atomic(
                working_directory / "model_kg_edges.jsonl.gz",
                _edge_rows(staging.connection),
            ),
            "model_kg_schema.json": _write_json_atomic(
                working_directory / "model_kg_schema.json",
                schema_payload(),
            ),
            "unresolved_process_partners.jsonl.gz": (
                _write_gzip_jsonl_atomic(
                    working_directory
                    / "unresolved_process_partners.jsonl.gz",
                    _unresolved_partner_rows(staging.connection),
                )
            ),
            "failures.jsonl": _write_jsonl_atomic(
                working_directory / "failures.jsonl",
                [],
            ),
        }
        staging.close()
        output_checksums["model_kg.sqlite"] = _sha256_file(
            working_directory / "model_kg.sqlite"
        )
        summary = {
            "build_schema_version": "1.0.0",
            "builder_version": BUILDER_VERSION,
            "model_kg_schema_version": MODEL_KG_SCHEMA_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "accepted_input_gate1_audit": True,
            "source_model_version_directory": str(
                model_version_directory
            ),
            "source_model_version_summary_sha256": _sha256_file(
                model_version_directory / "summary.json"
            ),
            "gate1_audit_sha256": _sha256_file(gate1_audit_path),
            "execution_plan_sha256": _sha256_file(plan_path),
            "model_count": len({model.model_id for model in model_versions}),
            "model_version_count": len(model_versions),
            "node_count": node_count,
            "edge_count": edge_count,
            "node_type_counts": node_type_counts,
            "predicate_counts": predicate_counts,
            "input_record_counts": progress,
            "unresolved_process_partner_unique_count": (
                unresolved_process_partner_unique_count
            ),
            "output_checksums": output_checksums,
            "policy_blockers": build_summary.blocking_reasons,
        }
        _write_json_atomic(
            working_directory / "summary.json",
            summary,
        )
        working_directory.replace(output_directory)
        return summary
    except Exception as exc:
        try:
            staging.close()
        except Exception:
            pass
        _write_json_atomic(
            working_directory / "failure.json",
            {
                "failed_at": datetime.now(UTC).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
                "progress": progress,
            },
        )
        raise
