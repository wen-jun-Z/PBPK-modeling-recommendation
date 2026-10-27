from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from pbpk_kb.model_assets.models import OSPModelVersionSnapshot
from pbpk_kb.recommendation.schemas import (
    ModelingRequest,
    RankedModelCandidate,
    RequirementProfile,
)


MODEL_RELEVANCE_POLICY_VERSION = "pbpk-model-relevance-gate-1.1.0"
MODEL_KG_SQLITE_ENV = "PBPK_MODEL_KG_SQLITE"
MODEL_KG_AUDIT_ENV = "PBPK_MODEL_KG_AUDIT"

def normalize_biological_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


@dataclass(frozen=True)
class TargetMechanism:
    mechanism_type: str
    canonical_name: str
    aliases: tuple[str, ...]
    origins: tuple[str, ...]
    relationship_types: tuple[str, ...]
    relationship_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    source_locations: tuple[str, ...]

    @property
    def normalized_aliases(self) -> frozenset[str]:
        return frozenset(
            normalize_biological_name(value)
            for value in (self.canonical_name, *self.aliases)
            if value.strip()
        )


@dataclass(frozen=True)
class ModelMechanismPath:
    model_version_id: str
    project_relative_path: str
    process_node_id: str
    process_name: str
    process_type: str
    mechanism_type: str
    mechanism_name: str
    compound_educt_names: tuple[str, ...]
    ownership_edge_id: str
    mechanism_edge_id: str
    source_id: str
    source_location: str
    asset_sha256: str
    extraction_method: str


@dataclass(frozen=True)
class ModelMechanismIndex:
    status: str
    reason_code: str | None
    graph_summary_sha256: str | None
    graph_sqlite_sha256: str | None
    model_version_count: int
    paths: tuple[ModelMechanismPath, ...]

    def paths_for(
        self,
        model: OSPModelVersionSnapshot,
    ) -> tuple[ModelMechanismPath, ...]:
        return tuple(
            path
            for path in self.paths
            if path.model_version_id == model.model_version_id
            or path.project_relative_path == model.project_relative_path
        )


def unavailable_model_mechanism_index(
    reason_code: str,
) -> ModelMechanismIndex:
    return ModelMechanismIndex(
        status="unavailable_fail_closed",
        reason_code=reason_code,
        graph_summary_sha256=None,
        graph_sqlite_sha256=None,
        model_version_count=0,
        paths=(),
    )


def _merge_target_mechanisms(
    rows: list[dict[str, Any]],
) -> list[TargetMechanism]:
    grouped: dict[tuple[str, str], dict[str, set[str] | str]] = {}
    for row in rows:
        mechanism_type = str(row["mechanism_type"])
        canonical_name = str(row["canonical_name"])
        key = (
            mechanism_type,
            normalize_biological_name(canonical_name),
        )
        bucket = grouped.setdefault(
            key,
            {
                "mechanism_type": mechanism_type,
                "canonical_name": canonical_name,
                "aliases": set(),
                "origins": set(),
                "relationship_types": set(),
                "relationship_ids": set(),
                "evidence_ids": set(),
                "source_locations": set(),
            },
        )
        for field in (
            "aliases",
            "origins",
            "relationship_types",
            "relationship_ids",
            "evidence_ids",
            "source_locations",
        ):
            values = row.get(field) or []
            if isinstance(values, str):
                values = [values]
            bucket[field].update(str(value) for value in values if value)
    return [
        TargetMechanism(
            mechanism_type=str(bucket["mechanism_type"]),
            canonical_name=str(bucket["canonical_name"]),
            aliases=tuple(sorted(bucket["aliases"], key=str.casefold)),
            origins=tuple(sorted(bucket["origins"])),
            relationship_types=tuple(
                sorted(bucket["relationship_types"])
            ),
            relationship_ids=tuple(sorted(bucket["relationship_ids"])),
            evidence_ids=tuple(sorted(bucket["evidence_ids"])),
            source_locations=tuple(sorted(bucket["source_locations"])),
        )
        for _, bucket in sorted(grouped.items())
    ]


def target_mechanisms_for_request(
    *,
    session: Session,
    compound_id: str,
    request: ModelingRequest,
) -> list[TargetMechanism]:
    rows: list[dict[str, Any]] = []
    for mechanism_type, values in (
        (
            "enzyme",
            set(request.required_enzymes) | set(request.enzyme),
        ),
        (
            "transporter",
            set(request.required_transporters) | set(request.transporter),
        ),
    ):
        for value in sorted(values, key=str.casefold):
            rows.append(
                {
                    "mechanism_type": mechanism_type,
                    "canonical_name": value,
                    "aliases": [value],
                    "origins": ["user_request"],
                    "relationship_types": ["explicit_requirement"],
                }
            )

    relationships = session.execute(
        text(
            """
            SELECT
                rel.relationship_id::text,
                rel.relationship_type,
                rel.processed_evidence_id::text AS evidence_id,
                rel.source_location,
                target.preferred_name,
                target.gene_symbol,
                target.target_type,
                target.synonyms
            FROM enzyme_transporter_relationship rel
            JOIN protein_target_identity target
              ON target.protein_target_id = rel.protein_target_id
            JOIN auto_validated_evidence evidence
              ON evidence.evidence_id = rel.processed_evidence_id
            WHERE rel.compound_id = CAST(:compound_id AS uuid)
              AND rel.curation_status = 'auto_validated'
              AND target.curation_status = 'auto_validated'
              AND COALESCE(target.confidence_score, 0) >= 0.90
              AND evidence.curation_status = 'auto_validated'
              AND rel.relationship_type IN (
                    'substrate_of',
                    'transported_by',
                    'metabolized_by'
              )
              AND COALESCE(rel.confidence_score, 0) >= 0.90
            ORDER BY target.target_type, target.preferred_name,
                     rel.relationship_id
            """
        ),
        {"compound_id": compound_id},
    ).mappings()
    for relationship in relationships:
        aliases = list(relationship.get("synonyms") or [])
        if relationship.get("gene_symbol"):
            aliases.append(str(relationship["gene_symbol"]))
        rows.append(
            {
                "mechanism_type": relationship["target_type"],
                "canonical_name": relationship["preferred_name"],
                "aliases": aliases,
                "origins": ["auto_validated_evidence"],
                "relationship_types": [
                    relationship["relationship_type"]
                ],
                "relationship_ids": [
                    relationship["relationship_id"]
                ],
                "evidence_ids": [relationship["evidence_id"]],
                "source_locations": [relationship["source_location"]],
            }
        )
    return _merge_target_mechanisms(rows)


def _mechanism_index_cache_key(
    sqlite_path: Path,
    audit_path: Path,
) -> tuple[str, int, int, str, int, int]:
    sqlite_stat = sqlite_path.stat()
    audit_stat = audit_path.stat()
    return (
        str(sqlite_path),
        sqlite_stat.st_mtime_ns,
        sqlite_stat.st_size,
        str(audit_path),
        audit_stat.st_mtime_ns,
        audit_stat.st_size,
    )


def model_mechanism_index_from_environment(
    *,
    expected_model_version_count: int,
) -> ModelMechanismIndex:
    sqlite_value = os.getenv(MODEL_KG_SQLITE_ENV, "").strip()
    audit_value = os.getenv(MODEL_KG_AUDIT_ENV, "").strip()
    if not sqlite_value or not audit_value:
        return unavailable_model_mechanism_index(
            "MODEL_KG_PATH_NOT_CONFIGURED"
        )
    sqlite_path = Path(sqlite_value).expanduser().resolve()
    audit_path = Path(audit_value).expanduser().resolve()
    try:
        cache_key = _mechanism_index_cache_key(sqlite_path, audit_path)
        index = _load_model_mechanism_index(*cache_key)
    except (OSError, ValueError, json.JSONDecodeError, sqlite3.Error):
        return unavailable_model_mechanism_index(
            "MODEL_KG_INTEGRITY_OR_READ_FAILURE"
        )
    if index.model_version_count != expected_model_version_count:
        return unavailable_model_mechanism_index(
            "MODEL_KG_CATALOG_VERSION_COUNT_MISMATCH"
        )
    return index


@lru_cache(maxsize=4)
def _load_model_mechanism_index(
    sqlite_value: str,
    _sqlite_mtime_ns: int,
    _sqlite_size: int,
    audit_value: str,
    _audit_mtime_ns: int,
    _audit_size: int,
) -> ModelMechanismIndex:
    sqlite_path = Path(sqlite_value)
    audit_path = Path(audit_value)
    summary_path = sqlite_path.with_name("summary.json")
    summary = _read_json(summary_path)
    audit = _read_json(audit_path)
    summary_sha256 = _sha256_file(summary_path)
    expected_sqlite_sha256 = str(
        (summary.get("output_checksums") or {}).get("model_kg.sqlite")
        or ""
    )
    if (
        audit.get("accepted") is not True
        or audit.get("graph_summary_sha256") != summary_sha256
        or not expected_sqlite_sha256
    ):
        raise ValueError("Model KG audit binding is invalid.")
    actual_sqlite_sha256 = _sha256_file(sqlite_path)
    if actual_sqlite_sha256 != expected_sqlite_sha256:
        raise ValueError("Model KG SQLite checksum mismatch.")

    connection = sqlite3.connect(
        f"file:{sqlite_path}?mode=ro",
        uri=True,
    )
    try:
        connection.execute("PRAGMA query_only = ON")
        version_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM nodes WHERE node_type = 'ModelVersion'"
            ).fetchone()[0]
        )
        query = """
            SELECT
                version.node_id,
                version.properties_json,
                ownership.edge_id,
                process.node_id,
                process.name,
                process.properties_json,
                mechanism.node_type,
                mechanism.name,
                relation.edge_id,
                relation.provenance_json
            FROM edges ownership
            JOIN nodes version
              ON version.node_id = ownership.subject_id
             AND version.node_type = 'ModelVersion'
            JOIN nodes process
              ON process.node_id = ownership.object_id
             AND process.node_type = 'ModelComponent'
            JOIN edges relation
              ON relation.subject_id = process.node_id
             AND relation.predicate = 'MODIFIED_BY'
            JOIN nodes mechanism
              ON mechanism.node_id = relation.object_id
             AND mechanism.node_type IN ('Enzyme', 'Transporter')
            WHERE ownership.predicate = 'HAS_COMPONENT'
            ORDER BY version.node_id, process.node_id, mechanism.node_id,
                     relation.edge_id
        """
        paths: dict[tuple[str, ...], ModelMechanismPath] = {}
        for row in connection.execute(query):
            version_properties = json.loads(row[1])
            process_properties = json.loads(row[5])
            provenance = json.loads(row[9])
            if process_properties.get("component_kind") != "process":
                continue
            educt_names = sorted(
                {
                    str(participant.get("name"))
                    for participant in (
                        process_properties.get("participants") or []
                    )
                    if participant.get("role") == "educt"
                    and participant.get("entity_type")
                    in {"Compound", "Metabolite"}
                    and participant.get("name")
                },
                key=str.casefold,
            )
            if not educt_names:
                continue
            path = ModelMechanismPath(
                model_version_id=str(row[0]),
                project_relative_path=str(
                    version_properties["project_relative_path"]
                ),
                process_node_id=str(row[3]),
                process_name=str(row[4]),
                process_type=str(
                    process_properties.get("process_type") or ""
                ),
                mechanism_type=str(row[6]).casefold(),
                mechanism_name=str(row[7]),
                compound_educt_names=tuple(educt_names),
                ownership_edge_id=str(row[2]),
                mechanism_edge_id=str(row[8]),
                source_id=str(provenance.get("source_id") or ""),
                source_location=str(
                    provenance.get("source_location") or ""
                ),
                asset_sha256=str(provenance.get("asset_sha256") or ""),
                extraction_method=str(
                    provenance.get("extraction_method") or ""
                ),
            )
            key = (
                path.model_version_id,
                path.process_node_id,
                path.mechanism_type,
                normalize_biological_name(path.mechanism_name),
                path.mechanism_edge_id,
            )
            paths[key] = path
    finally:
        connection.close()
    return ModelMechanismIndex(
        status="accepted",
        reason_code=None,
        graph_summary_sha256=summary_sha256,
        graph_sqlite_sha256=actual_sqlite_sha256,
        model_version_count=version_count,
        paths=tuple(paths[key] for key in sorted(paths)),
    )


def candidate_relevance_decision(
    *,
    profile: RequirementProfile,
    model: OSPModelVersionSnapshot,
    candidate: RankedModelCandidate,
    target_mechanisms: list[TargetMechanism],
    mechanism_index: ModelMechanismIndex,
) -> dict[str, Any]:
    requested_names = {
        normalize_biological_name(profile.compound_identity.preferred_name),
        normalize_biological_name(profile.compound_identity.active_moiety),
    }
    matched_compounds = sorted(
        {
            value
            for value in model.compounds
            if normalize_biological_name(value) in requested_names
        },
        key=str.casefold,
    )
    if candidate.candidate_class != "mechanistic_template":
        return {
            "is_relevant": True,
            "relevance_class": "same_compound",
            "relevance_reason_codes": ["SAME_COMPOUND_MATCH"],
            "relevance_basis": [
                {
                    "basis_type": "same_compound",
                    "matched_compounds": matched_compounds,
                }
            ],
        }
    if not target_mechanisms:
        return {
            "is_relevant": False,
            "relevance_class": "unrelated",
            "relevance_reason_codes": [
                "NO_VERIFIED_TARGET_DISPOSITION_MECHANISM"
            ],
            "relevance_basis": [],
        }
    if mechanism_index.status != "accepted":
        return {
            "is_relevant": False,
            "relevance_class": "unrelated",
            "relevance_reason_codes": [
                mechanism_index.reason_code
                or "MODEL_KG_UNAVAILABLE_FAIL_CLOSED"
            ],
            "relevance_basis": [],
        }

    model_names = {
        normalize_biological_name(value)
        for value in (*model.compounds, *model.metabolites)
        if value.strip()
    }
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for mechanism in target_mechanisms:
        for path in mechanism_index.paths_for(model):
            if path.mechanism_type != mechanism.mechanism_type:
                continue
            if (
                normalize_biological_name(path.mechanism_name)
                not in mechanism.normalized_aliases
            ):
                continue
            if not model_names.intersection(
                normalize_biological_name(value)
                for value in path.compound_educt_names
            ):
                continue
            key = (
                mechanism.mechanism_type,
                normalize_biological_name(mechanism.canonical_name),
            )
            basis = grouped.setdefault(
                key,
                {
                    "basis_type": "shared_verified_disposition_process",
                    "mechanism_type": mechanism.mechanism_type,
                    "mechanism_name": mechanism.canonical_name,
                    "target_origins": list(mechanism.origins),
                    "target_relationship_types": list(
                        mechanism.relationship_types
                    ),
                    "target_relationship_ids": list(
                        mechanism.relationship_ids
                    ),
                    "target_evidence_ids": list(mechanism.evidence_ids),
                    "target_source_locations": list(
                        mechanism.source_locations
                    ),
                    "model_process_paths": [],
                },
            )
            process_record = {
                "process_node_id": path.process_node_id,
                "process_name": path.process_name,
                "process_type": path.process_type,
                "compound_educt_names": list(path.compound_educt_names),
                "ownership_edge_id": path.ownership_edge_id,
                "mechanism_edge_id": path.mechanism_edge_id,
                "source_id": path.source_id,
                "source_location": path.source_location,
                "asset_sha256": path.asset_sha256,
                "extraction_method": path.extraction_method,
            }
            if process_record not in basis["model_process_paths"]:
                basis["model_process_paths"].append(process_record)
    bases = [grouped[key] for key in sorted(grouped)]
    for basis in bases:
        basis["model_process_paths"].sort(
            key=lambda value: (
                value["process_name"].casefold(),
                value["mechanism_edge_id"],
            )
        )
    return {
        "is_relevant": bool(bases),
        "relevance_class": (
            "shared_verified_mechanism" if bases else "unrelated"
        ),
        "relevance_reason_codes": (
            ["SHARED_VERIFIED_DISPOSITION_PROCESS"]
            if bases
            else ["NO_SHARED_VERIFIED_MODEL_PROCESS_PATH"]
        ),
        "relevance_basis": bases,
    }
