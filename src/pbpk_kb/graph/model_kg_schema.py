from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


MODEL_KG_SCHEMA_VERSION = "pbpk-model-kg-1.1.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ModelKGProvenance(StrictModel):
    source_id: str
    source_location: str
    asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    extraction_method: str
    evidence_id: str | None = None


class ModelKGNode(StrictModel):
    node_id: str
    node_type: str
    name: str
    properties: dict[str, Any] = Field(default_factory=dict)
    provenance: ModelKGProvenance
    occurrence_count: int = Field(default=1, ge=1)


class ModelKGEdge(StrictModel):
    edge_id: str
    subject_id: str
    subject_type: str
    predicate: str
    object_id: str
    object_type: str
    properties: dict[str, Any] = Field(default_factory=dict)
    provenance: ModelKGProvenance
    occurrence_count: int = Field(default=1, ge=1)


@dataclass(frozen=True)
class RelationSpec:
    subject_types: frozenset[str]
    object_types: frozenset[str]


NODE_TYPES = frozenset(
    {
        "PBPKModel",
        "ModelVersion",
        "ModelAsset",
        "Software",
        "ModelStructure",
        "Compartment",
        "ModelComponent",
        "ModelParameter",
        "Compound",
        "Metabolite",
        "Enzyme",
        "Transporter",
        "Protein",
        "MolecularComplex",
        "MolecularEntity",
        "Species",
        "Population",
        "AdministrationRoute",
        "Formulation",
        "SimulationScenario",
        "AdministrationEvent",
        "CalibrationDataset",
        "QualificationDataset",
        "ModelDataset",
        "ValidationRun",
        "ValidationMetric",
        "Evidence",
        "SourceDocument",
    }
)

MODEL_ENTITIES = frozenset(
    {
        "Compound",
        "Metabolite",
        "Enzyme",
        "Transporter",
        "Protein",
        "MolecularComplex",
        "MolecularEntity",
        "Species",
        "Population",
        "AdministrationRoute",
        "Formulation",
        "Compartment",
    }
)
DATASET_TYPES = frozenset(
    {"CalibrationDataset", "QualificationDataset", "ModelDataset"}
)

RELATIONS: dict[str, RelationSpec] = {
    "HAS_VERSION": RelationSpec(
        frozenset({"PBPKModel"}),
        frozenset({"ModelVersion"}),
    ),
    "HAS_ASSET": RelationSpec(
        frozenset({"ModelVersion"}),
        frozenset({"ModelAsset"}),
    ),
    "IMPLEMENTED_IN": RelationSpec(
        frozenset({"ModelVersion"}),
        frozenset({"Software"}),
    ),
    "MODELS_COMPOUND": RelationSpec(
        frozenset({"ModelVersion"}),
        frozenset({"Compound"}),
    ),
    "HAS_STRUCTURE": RelationSpec(
        frozenset({"ModelVersion"}),
        frozenset({"ModelStructure"}),
    ),
    "HAS_COMPARTMENT": RelationSpec(
        frozenset({"ModelStructure"}),
        frozenset({"Compartment"}),
    ),
    "HAS_COMPONENT": RelationSpec(
        frozenset({"ModelVersion"}),
        frozenset({"ModelComponent"}),
    ),
    "USES_PARAMETER": RelationSpec(
        frozenset({"ModelVersion", "SimulationScenario"}),
        frozenset({"ModelParameter"}),
    ),
    "SUPPORTS_ROUTE": RelationSpec(
        frozenset({"ModelVersion", "SimulationScenario"}),
        frozenset({"AdministrationRoute"}),
    ),
    "SUPPORTS_FORMULATION": RelationSpec(
        frozenset({"ModelVersion", "SimulationScenario"}),
        frozenset({"Formulation"}),
    ),
    "APPLIES_TO": RelationSpec(
        frozenset({"ModelVersion", "SimulationScenario"}),
        frozenset({"Population", "Species"}),
    ),
    "CALIBRATED_WITH": RelationSpec(
        frozenset({"ModelVersion"}),
        frozenset({"CalibrationDataset"}),
    ),
    "VALIDATED_WITH": RelationSpec(
        frozenset({"ModelVersion", "ValidationRun"}),
        frozenset({"QualificationDataset"}),
    ),
    "USES_DATASET": RelationSpec(
        frozenset({"ModelVersion"}),
        DATASET_TYPES,
    ),
    "HAS_VALIDATION_RUN": RelationSpec(
        frozenset({"ModelVersion"}),
        frozenset({"ValidationRun"}),
    ),
    "REPORTS": RelationSpec(
        frozenset({"ValidationRun"}),
        frozenset({"ValidationMetric"}),
    ),
    "EVALUATES_DATASET": RelationSpec(
        frozenset({"ValidationMetric"}),
        frozenset({"QualificationDataset"}),
    ),
    "HAS_SIMULATION": RelationSpec(
        frozenset({"ModelVersion"}),
        frozenset({"SimulationScenario"}),
    ),
    "HAS_ADMINISTRATION": RelationSpec(
        frozenset({"SimulationScenario"}),
        frozenset({"AdministrationEvent"}),
    ),
    "ADMINISTERS_COMPOUND": RelationSpec(
        frozenset({"AdministrationEvent"}),
        frozenset({"Compound", "Metabolite"}),
    ),
    "USES_ROUTE": RelationSpec(
        frozenset({"AdministrationEvent"}),
        frozenset({"AdministrationRoute"}),
    ),
    "USES_FORMULATION": RelationSpec(
        frozenset({"AdministrationEvent"}),
        frozenset({"Formulation"}),
    ),
    "REPRESENTS": RelationSpec(
        frozenset({"ModelComponent"}),
        MODEL_ENTITIES,
    ),
    "HAS_EDUCT": RelationSpec(
        frozenset({"ModelComponent"}),
        frozenset(
            {
                "Compound",
                "Metabolite",
                "Enzyme",
                "Transporter",
                "Protein",
                "MolecularComplex",
                "MolecularEntity",
            }
        ),
    ),
    "HAS_PRODUCT": RelationSpec(
        frozenset({"ModelComponent"}),
        frozenset(
            {
                "Compound",
                "Metabolite",
                "Enzyme",
                "Transporter",
                "Protein",
                "MolecularComplex",
                "MolecularEntity",
            }
        ),
    ),
    "MODIFIED_BY": RelationSpec(
        frozenset({"ModelComponent"}),
        frozenset(
            {
                "Enzyme",
                "Transporter",
                "Protein",
                "Compound",
                "Metabolite",
                "MolecularComplex",
                "MolecularEntity",
            }
        ),
    ),
    "DESCRIBED_IN": RelationSpec(
        frozenset({"ModelVersion"}),
        frozenset({"SourceDocument"}),
    ),
    "SUPPORTED_BY": RelationSpec(
        NODE_TYPES - frozenset({"Evidence", "SourceDocument", "ModelAsset"}),
        frozenset({"Evidence"}),
    ),
    "DERIVED_FROM": RelationSpec(
        frozenset({"Evidence"}),
        frozenset({"SourceDocument", "ModelAsset"}),
    ),
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_id(namespace: str, *parts: Any) -> str:
    body = "\x1f".join(
        canonical_json(part)
        if isinstance(part, (dict, list, tuple))
        else str(part)
        for part in parts
    )
    return (
        f"{namespace}:"
        f"{hashlib.sha256(body.encode('utf-8')).hexdigest()}"
    )


def validate_relation(
    predicate: str,
    subject_type: str,
    object_type: str,
) -> str | None:
    spec = RELATIONS.get(predicate)
    if spec is None:
        return f"unknown_relation:{predicate}"
    if subject_type not in spec.subject_types:
        return (
            f"invalid_subject_type:{predicate}:{subject_type}:"
            f"{sorted(spec.subject_types)}"
        )
    if object_type not in spec.object_types:
        return (
            f"invalid_object_type:{predicate}:{object_type}:"
            f"{sorted(spec.object_types)}"
        )
    return None


def schema_payload() -> dict[str, Any]:
    return {
        "schema_version": MODEL_KG_SCHEMA_VERSION,
        "node_types": sorted(NODE_TYPES),
        "relations": {
            name: {
                "subject_types": sorted(spec.subject_types),
                "object_types": sorted(spec.object_types),
            }
            for name, spec in sorted(RELATIONS.items())
        },
    }
