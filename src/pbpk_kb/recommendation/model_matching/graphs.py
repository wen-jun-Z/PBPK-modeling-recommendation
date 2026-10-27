from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping


GRAPH_SCHEMA_VERSION = "pbpk-cross-graph-1.2.0"


def normalize_text(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def semantic_key(node_type: str, value: object) -> str:
    if node_type == "MolecularStructure":
        structure = str(value).strip()
        digest = hashlib.sha256(structure.encode("utf-8")).hexdigest()
        return f"{node_type}:sha256:{digest}"
    return f"{node_type}:{normalize_text(value)}"


@dataclass(frozen=True, slots=True)
class GraphNode:
    node_id: str
    node_type: str
    semantic_key: str
    label: str
    hierarchy: str
    required: bool = False
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source_id: str
    predicate: str
    target_id: str
    required: bool = False
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TypedGraph:
    graph_id: str
    graph_kind: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "graph_id": self.graph_id,
            "graph_kind": self.graph_kind,
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
            "metadata": dict(self.metadata),
        }

    @property
    def checksum(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def graph_from_dict(value: Mapping[str, Any]) -> TypedGraph:
    return TypedGraph(
        graph_id=str(value["graph_id"]),
        graph_kind=str(value["graph_kind"]),
        nodes=tuple(GraphNode(**node) for node in value.get("nodes") or []),
        edges=tuple(GraphEdge(**edge) for edge in value.get("edges") or []),
        metadata=dict(value.get("metadata") or {}),
    )


_TYPE_HIERARCHY = {
    "Compound": "identity",
    "Objective": "context",
    "Species": "context",
    "Population": "context",
    "Route": "context",
    "Formulation": "context",
    "Regimen": "context",
    "ADMEProcess": "mechanism",
    "Enzyme": "mechanism",
    "Transporter": "mechanism",
    "DDIRole": "mechanism",
    "Compartment": "structure",
    "Endpoint": "structure",
    "ModelStructure": "structure",
    "Dataset": "validation",
    "Software": "validation",
    "License": "validation",
    "ExecutionStatus": "validation",
    "ParseStatus": "validation",
    "PBPKModel": "identity",
    "ModelVersion": "root",
    "ModelAsset": "validation",
    "ModelComponent": "mechanism",
    "ModelParameter": "structure",
    "Metabolite": "mechanism",
    "Protein": "mechanism",
    "MolecularComplex": "mechanism",
    "MolecularEntity": "mechanism",
    "AdministrationRoute": "context",
    "SimulationScenario": "context",
    "AdministrationEvent": "context",
    "CalibrationDataset": "validation",
    "QualificationDataset": "validation",
    "ModelDataset": "validation",
    "ValidationRun": "validation",
    "ValidationMetric": "validation",
    "Evidence": "validation",
    "SourceDocument": "validation",
    "MolecularStructure": "identity",
    "MolecularDescriptor": "identity",
}


def node_hierarchy(node_type: str) -> str:
    return _TYPE_HIERARCHY.get(node_type, "other")


def _values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def population_categories(values: Iterable[str]) -> list[str]:
    text = " ".join(values).casefold()
    categories: list[str] = []
    markers = {
        "pediatric": ("pediatric", "paediatric", "child", "infant", "neonat"),
        "geriatric": ("geriatric", "elderly"),
        "pregnancy": ("pregnan", "lactat"),
        "renal_impairment": ("renal impairment", "kidney impairment"),
        "hepatic_impairment": ("hepatic impairment", "liver impairment"),
        "adult": ("adult", "standard european", "japanese", "healthy"),
    }
    for category, aliases in markers.items():
        if any(alias in text for alias in aliases):
            categories.append(category)
    return categories or sorted({normalize_text(value) for value in values if value})


def endpoint_categories(values: Iterable[str]) -> list[str]:
    categories: set[str] = set()
    for value in values:
        text = normalize_text(value)
        if "auc" in text:
            categories.add("auc")
        if "cmax" in text or "maximum concentration" in text:
            categories.add("cmax")
        if "concentration" in text or "plasma" in text or "blood" in text:
            categories.add("concentration_time")
        if "urine" in text or "renal" in text:
            categories.add("urinary_excretion")
        if "bile" in text or "feces" in text:
            categories.add("biliary_fecal_excretion")
        if "absorbed" in text or "dissolved" in text:
            categories.add("absorption_fraction")
        if "tissue" in text or "organ" in text:
            categories.add("tissue_concentration")
    if "concentration_time" in categories:
        categories.update({"auc", "cmax"})
    return sorted(categories)


class _GraphBuilder:
    def __init__(self, graph_id: str, graph_kind: str, root_type: str) -> None:
        self.graph_id = graph_id
        self.graph_kind = graph_kind
        self.root_id = f"{graph_id}:root"
        self.nodes: dict[str, GraphNode] = {
            self.root_id: GraphNode(
                node_id=self.root_id,
                node_type=root_type,
                semantic_key=semantic_key(root_type, graph_id),
                label=graph_id,
                hierarchy="root",
            )
        }
        self.edges: dict[tuple[str, str, str], GraphEdge] = {}

    def add(
        self,
        *,
        node_type: str,
        value: object,
        predicate: str,
        required: bool = False,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        label = str(value).strip()
        if not label:
            return
        key = semantic_key(node_type, label)
        node_id = f"{self.graph_id}:{key}"
        self.nodes.setdefault(
            node_id,
            GraphNode(
                node_id=node_id,
                node_type=node_type,
                semantic_key=key,
                label=label,
                hierarchy=_TYPE_HIERARCHY.get(node_type, "other"),
                required=required,
                attributes=dict(attributes or {}),
            ),
        )
        edge_key = (self.root_id, predicate, node_id)
        self.edges.setdefault(
            edge_key,
            GraphEdge(
                source_id=self.root_id,
                predicate=predicate,
                target_id=node_id,
                required=required,
            ),
        )

    def build(self, metadata: Mapping[str, Any]) -> TypedGraph:
        return TypedGraph(
            graph_id=self.graph_id,
            graph_kind=self.graph_kind,
            nodes=tuple(sorted(self.nodes.values(), key=lambda node: node.node_id)),
            edges=tuple(
                sorted(
                    self.edges.values(),
                    key=lambda edge: (edge.source_id, edge.predicate, edge.target_id),
                )
            ),
            metadata=dict(metadata),
        )


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Expected mapping or Pydantic model, received {type(value)!r}.")


def build_model_graph(snapshot: object) -> TypedGraph:
    data = _mapping(snapshot)
    model_id = str(data.get("model_version_id") or data.get("model_id") or "unknown")
    builder = _GraphBuilder(model_id, "executable_model", "ModelVersion")

    fields = (
        ("Compound", "MODELS_COMPOUND", _values(data.get("compounds"))),
        ("Species", "SUPPORTS_SPECIES", _values(data.get("species"))),
        (
            "Population",
            "SUPPORTS_POPULATION",
            population_categories(_values(data.get("populations"))),
        ),
        ("Route", "SUPPORTS_ROUTE", _values(data.get("routes"))),
        ("Formulation", "SUPPORTS_FORMULATION", _values(data.get("formulations"))),
        ("ADMEProcess", "IMPLEMENTS_PROCESS", _values(data.get("process_types"))),
        ("Enzyme", "IMPLEMENTS_ENZYME", _values(data.get("enzymes"))),
        ("Transporter", "IMPLEMENTS_TRANSPORTER", _values(data.get("transporters"))),
        ("Compartment", "HAS_COMPARTMENT", _values(data.get("compartments"))),
        (
            "Endpoint",
            "PRODUCES_ENDPOINT",
            endpoint_categories(_values(data.get("output_names"))),
        ),
        ("Software", "IMPLEMENTED_IN", _values(data.get("software"))),
        (
            "MolecularStructure",
            "HAS_MOLECULAR_STRUCTURE",
            _values(data.get("canonical_smiles")),
        ),
    )
    for node_type, predicate, values in fields:
        for value in sorted(set(values), key=str.casefold):
            builder.add(node_type=node_type, value=value, predicate=predicate)

    license_value = data.get("license") or data.get("license_status") or "unknown"
    if isinstance(license_value, Mapping):
        license_value = license_value.get("status") or "unknown"
    builder.add(node_type="License", value=license_value, predicate="HAS_LICENSE")
    builder.add(
        node_type="ExecutionStatus",
        value=data.get("execution_probe_status") or "unknown",
        predicate="HAS_EXECUTION_STATUS",
    )
    parse_status = (
        data.get("pkml_parse_status")
        or data.get("static_parse_status")
        or "unknown"
    )
    builder.add(
        node_type="ParseStatus",
        value=parse_status,
        predicate="HAS_PARSE_STATUS",
    )
    builder.add(
        node_type="ModelStructure",
        value="whole_body" if len(_values(data.get("compartments"))) >= 10 else "reduced_or_unknown",
        predicate="HAS_MODEL_STRUCTURE",
    )
    molecular_descriptors = data.get("molecular_descriptors") or {}
    if isinstance(molecular_descriptors, Mapping):
        for name, value in sorted(molecular_descriptors.items()):
            if value is None:
                continue
            builder.add(
                node_type="MolecularDescriptor",
                value=f"{name}={value}",
                predicate="HAS_MOLECULAR_DESCRIPTOR",
            )
    dataset_count = int(data.get("dataset_reference_count") or 0)
    validation_metric_count = int(data.get("validation_metric_count") or 0)
    builder.add(
        node_type="QualificationDataset",
        value=f"dataset_count={dataset_count}",
        predicate="HAS_QUALIFICATION_DATASET_COVERAGE",
        attributes={"count": dataset_count},
    )
    scientific_status = (
        "scientific_passed"
        if data.get("scientific_accepted")
        else "engineering_only"
        if data.get("engineering_accepted")
        else "execution_failed"
    )
    builder.add(
        node_type="ValidationRun",
        value=scientific_status,
        predicate="HAS_VALIDATION_RUN",
    )
    builder.add(
        node_type="ValidationMetric",
        value=(
            f"metric_count={validation_metric_count};"
            f"run_grade={int(data.get('run_label_grade') or 0)}"
        ),
        predicate="HAS_VALIDATION_METRICS",
        attributes={
            "metric_count": validation_metric_count,
            "run_label_grade": int(data.get("run_label_grade") or 0),
        },
    )
    for scenario in data.get("validated_scenarios") or []:
        if not isinstance(scenario, Mapping):
            continue
        dose_value = scenario.get("dose_value")
        dose_unit = scenario.get("dose_unit")
        frequency = scenario.get("frequency") or "unknown_frequency"
        route = scenario.get("route") or "unknown_route"
        if dose_value is None or not dose_unit:
            continue
        builder.add(
            node_type="Regimen",
            value=(
                f"{scenario.get('compound') or 'unspecified_compound'}|"
                f"{frequency}|{dose_value} {dose_unit}|{route}"
            ),
            predicate="HAS_VALIDATED_REGIMEN",
            required=False,
            attributes={
                "dose_value": dose_value,
                "dose_unit": dose_unit,
                "compound": scenario.get("compound"),
                "dose_basis": scenario.get("dose_basis"),
                "frequency": frequency,
                "route": route,
                "start_time_value": scenario.get("start_time_value"),
                "start_time_unit": scenario.get("start_time_unit"),
                "end_time_value": scenario.get("end_time_value"),
                "end_time_unit": scenario.get("end_time_unit"),
                "infusion_duration_value": scenario.get(
                    "infusion_duration_value"
                ),
                "infusion_duration_unit": scenario.get(
                    "infusion_duration_unit"
                ),
                "validation_status": "executed_verification",
            },
        )
    return builder.build(
        {
            "model_family": data.get("model_family"),
            "active_moieties": sorted(
                set(_values(data.get("compounds"))), key=str.casefold
            ),
            "candidate_split": data.get("candidate_split"),
            "project_relative_path": data.get("project_relative_path"),
            "repository_commit": data.get("repository_commit"),
            "simulation_count": int(data.get("simulation_count") or 0),
            "parameter_record_count": int(data.get("parameter_record_count") or 0),
            "dataset_reference_count": int(data.get("dataset_reference_count") or 0),
            "validation_metric_count": int(data.get("validation_metric_count") or 0),
            "license_status": str(license_value),
            "parse_status": str(parse_status),
            "execution_probe_status": str(
                data.get("execution_probe_status") or "unknown"
            ),
            "engineering_accepted": (
                bool(data.get("engineering_accepted"))
                if "engineering_accepted" in data
                else None
            ),
            "scientific_accepted": (
                bool(data.get("scientific_accepted"))
                if "scientific_accepted" in data
                else None
            ),
            "reusable_candidate": (
                bool(data.get("reusable_candidate"))
                if "reusable_candidate" in data
                else None
            ),
        }
    )


_REQUIREMENT_TYPE = {
    "compound_identity": "Compound",
    "biological_scope": "Species",
    "validation_context": "Objective",
    "administration_route": "Route",
    "formulation": "Formulation",
    "dose_regimen": "Regimen",
    "adme_process": "ADMEProcess",
    "enzyme": "Enzyme",
    "transporter": "Transporter",
    "ddi_role": "DDIRole",
    "compartment": "Compartment",
    "prediction_endpoint": "Endpoint",
    "population_feature": "Population",
    "model_structure": "ModelStructure",
    "request_constraint": "Dataset",
}


def build_requirement_graph(profile: object) -> TypedGraph:
    data = _mapping(profile)
    graph_id = str(data.get("profile_id") or data.get("request_id") or "requirement")
    builder = _GraphBuilder(graph_id, "requirement", "ModelingRequest")
    for requirement in data.get("requirements") or []:
        item = _mapping(requirement)
        category = str(item.get("category") or "")
        node_type = _REQUIREMENT_TYPE.get(category)
        if node_type is None:
            continue
        value = item.get("expected_value") or item.get("preferred_name")
        builder.add(
            node_type=node_type,
            value=value,
            predicate=f"REQUIRES_{node_type.upper()}",
            required=item.get("necessity") == "hard_required",
            attributes={
                "origin": item.get("origin"),
                "confidence": item.get("confidence"),
                "evidence_ids": item.get("evidence_ids") or [],
                "canonical_unit": item.get("canonical_unit"),
            },
        )
    identity = _mapping(data.get("compound_identity") or {})
    if identity.get("preferred_name"):
        builder.add(
            node_type="Compound",
            value=identity["preferred_name"],
            predicate="REQUIRES_COMPOUND",
            required=True,
        )
    return builder.build(
        {
            "critical_gap_count": len(data.get("critical_gaps") or []),
            "conflict_count": len(data.get("conflicts") or []),
            "profile_checksum": data.get("checksum"),
        }
    )


def build_execution_requirement_graph(task: Mapping[str, Any]) -> TypedGraph:
    """Build a request graph from a hidden-validation execution task."""

    graph_id = str(task["task_id"])
    builder = _GraphBuilder(graph_id, "requirement", "ModelingRequest")
    for compound in _values(task.get("active_moieties")):
        builder.add(
            node_type="Compound",
            value=compound,
            predicate="REQUIRES_COMPOUND",
            required=True,
        )
    for analyte in _values(task.get("observed_analytes")):
        builder.add(
            node_type="Compound",
            value=analyte,
            predicate="REQUIRES_OBSERVED_ANALYTE",
            required=True,
        )
    for structure in _values(task.get("canonical_smiles")):
        builder.add(
            node_type="MolecularStructure",
            value=structure,
            predicate="REQUIRES_MOLECULAR_STRUCTURE",
            required=True,
        )
    descriptors = task.get("molecular_descriptors") or {}
    if isinstance(descriptors, Mapping):
        for name, value in sorted(descriptors.items()):
            if value is None:
                continue
            builder.add(
                node_type="MolecularDescriptor",
                value=f"{name}={value}",
                predicate="REQUIRES_MOLECULAR_DESCRIPTOR",
                required=False,
            )
    builder.add(
        node_type="Species",
        value=task.get("species") or "human",
        predicate="REQUIRES_SPECIES",
        required=True,
    )
    for node_type, predicate, values in (
        (
            "ADMEProcess",
            "REQUIRES_ADMEPROCESS",
            _values(task.get("process_types")),
        ),
        ("Enzyme", "REQUIRES_ENZYME", _values(task.get("enzymes"))),
        (
            "Transporter",
            "REQUIRES_TRANSPORTER",
            _values(task.get("transporters")),
        ),
        ("DDIRole", "REQUIRES_DDIROLE", _values(task.get("ddi_roles"))),
    ):
        for value in sorted(set(values), key=str.casefold):
            builder.add(
                node_type=node_type,
                value=value,
                predicate=predicate,
                required=True,
            )
    scenario = _mapping(task.get("scenario") or {})
    raw_regimens = scenario.get("regimens") or [scenario]
    for raw_regimen in raw_regimens:
        regimen = _mapping(raw_regimen)
        builder.add(
            node_type="Route",
            value=regimen.get("route"),
            predicate="REQUIRES_ROUTE",
            required=True,
        )
        formulations = _values(regimen.get("formulations")) or _values(
            regimen.get("formulation")
        )
        for formulation in formulations:
            builder.add(
                node_type="Formulation",
                value=formulation,
                predicate="PREFERS_FORMULATION",
                required=False,
            )
        builder.add(
            node_type="Regimen",
            value=(
                f"{regimen.get('compound') or 'unspecified_compound'}|"
                f"{regimen.get('frequency') or 'single_dose'}"
            ),
            predicate="REQUIRES_REGIMEN",
            required=False,
            attributes={
                "compound": regimen.get("compound"),
                "dose_value": regimen.get("dose_value"),
                "dose_unit": regimen.get("dose_unit"),
                "dose_basis": regimen.get("dose_basis"),
                "start_time_value": regimen.get("start_time_value"),
                "start_time_unit": regimen.get("start_time_unit"),
                "end_time_value": regimen.get("end_time_value"),
                "end_time_unit": regimen.get("end_time_unit"),
                "infusion_duration_value": regimen.get(
                    "infusion_duration_value"
                ),
                "infusion_duration_unit": regimen.get(
                    "infusion_duration_unit"
                ),
                "continuous_value": True,
            },
        )
    for population in (
        scenario.get("population"),
        scenario.get("sex"),
    ):
        builder.add(
            node_type="Population",
            value=population,
            predicate="REQUIRES_POPULATION",
            required=False,
        )
    for endpoint in endpoint_categories(_values(task.get("required_outputs"))):
        builder.add(
            node_type="Endpoint",
            value=endpoint,
            predicate="REQUIRES_ENDPOINT",
            required=True,
        )
    builder.add(
        node_type="Software",
        value="PK-Sim",
        predicate="REQUIRES_SOFTWARE",
        required=True,
    )
    return builder.build(
        {
            "benchmark_level": "execution_derived",
            "validation_group_id": task.get("validation_group_id"),
            "split": task.get("split"),
        }
    )


def requirement_graph_from_model(
    snapshot: Mapping[str, Any],
    *,
    query_id: str,
    include_identity: bool = True,
) -> TypedGraph:
    """Build a candidate-retrieval sanity query without model/version IDs."""

    builder = _GraphBuilder(query_id, "requirement", "ModelingRequest")
    if include_identity:
        for value in _values(snapshot.get("model_family") or snapshot.get("compounds")):
            builder.add(
                node_type="Compound",
                value=value,
                predicate="REQUIRES_COMPOUND",
                required=True,
            )
    for node_type, predicate, values in (
        ("Species", "REQUIRES_SPECIES", _values(snapshot.get("species"))),
        (
            "Population",
            "REQUIRES_POPULATION",
            population_categories(_values(snapshot.get("populations"))),
        ),
        ("Route", "REQUIRES_ROUTE", _values(snapshot.get("routes"))),
        ("Formulation", "REQUIRES_FORMULATION", _values(snapshot.get("formulations"))),
        ("ADMEProcess", "REQUIRES_ADMEPROCESS", _values(snapshot.get("process_types"))),
        ("Enzyme", "REQUIRES_ENZYME", _values(snapshot.get("enzymes"))),
        ("Transporter", "REQUIRES_TRANSPORTER", _values(snapshot.get("transporters"))),
        (
            "Endpoint",
            "REQUIRES_ENDPOINT",
            endpoint_categories(_values(snapshot.get("output_names"))),
        ),
    ):
        for value in sorted(set(values), key=str.casefold):
            builder.add(
                node_type=node_type,
                value=value,
                predicate=predicate,
                required=True,
            )
    builder.add(
        node_type="Software",
        value=snapshot.get("software") or "PK-Sim",
        predicate="REQUIRES_SOFTWARE",
        required=True,
    )
    return builder.build(
        {
            "benchmark_level": "candidate_retrieval_sanity",
            "source_family": snapshot.get("model_family"),
            "identity_included": include_identity,
        }
    )
