from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.models import (
    ModelValidationRunLabel,
    OSPModelVersionSnapshot,
)
from pbpk_kb.recommendation.formal_graph import (
    FORMAL_GRAPH_VERSION,
    stable_id,
)
from pbpk_kb.recommendation.kg_models import (
    KGEdge,
    KGEvent,
    KGNode,
    Provenance,
)
from pbpk_kb.recommendation.ontology import ONTOLOGY_VERSION


PROJECTOR_VERSION = "licensed-model-recommendation-graph-projector-1.0.0"


class LicensedModelGraphError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise LicensedModelGraphError(f"Expected a JSON object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise LicensedModelGraphError(
                    f"Expected an object at {path}:{line_number}"
                )
            yield value


def _write_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = (_canonical_json(row) + "\n").encode("utf-8")
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return digest.hexdigest()


def _single_row(path: Path) -> dict[str, Any]:
    rows = list(_iter_jsonl(path))
    if len(rows) != 1:
        raise LicensedModelGraphError(
            f"Expected one row in {path}; found {len(rows)}."
        )
    return rows[0]


def _supporting_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _provenance(
    *,
    evidence_id: str,
    source_id: str,
    source_location: str,
    supporting_text: str,
    extraction_method: str,
    context_id: str | None = None,
) -> Provenance:
    return Provenance(
        evidence_id=evidence_id,
        source_id=source_id,
        source_location=source_location,
        supporting_text=supporting_text,
        source_text_hash=_supporting_text_hash(supporting_text),
        extraction_method=extraction_method,
        extractor_version=PROJECTOR_VERSION,
        confidence=1.0,
        quality_status="auto_validated",
        negation_status="affirmed",
        speculation_status="asserted",
        context_id=context_id,
    )


def project_licensed_model_into_recommendation_graph(
    *,
    base_graph_directory: Path,
    base_gate1_audit_path: Path,
    registration_directory: Path,
    validation_audit_path: Path,
    compound_node_id: str,
    output_directory: Path,
) -> dict[str, Any]:
    base_graph_directory = base_graph_directory.resolve()
    base_gate1_audit_path = base_gate1_audit_path.resolve()
    registration_directory = registration_directory.resolve()
    validation_audit_path = validation_audit_path.resolve()
    output_directory = output_directory.resolve()
    working = output_directory.with_name(output_directory.name + ".tmp")
    if output_directory.exists() or working.exists():
        raise LicensedModelGraphError(
            f"Output or temporary directory already exists: {output_directory}"
        )

    base_summary = _read_json(base_graph_directory / "kg_build_summary.json")
    base_manifest_path = base_graph_directory / "kg_extraction_manifest.json"
    base_manifest = _read_json(base_manifest_path)
    base_audit = _read_json(base_gate1_audit_path)
    expected_manifest_sha = (
        (base_audit.get("inputs") or {})
        .get("graph_manifest", {})
        .get("sha256")
    )
    base_checks = {
        "base_graph_accepted": base_summary.get("accepted") is True,
        "base_gate1_audit_accepted": base_audit.get("accepted") is True,
        "base_manifest_bound_to_audit": (
            expected_manifest_sha == sha256_file(base_manifest_path)
        ),
    }
    if not all(base_checks.values()):
        raise LicensedModelGraphError(
            "Base recommendation graph is not an accepted audited snapshot."
        )

    registration_summary_path = (
        registration_directory / "registration_summary.json"
    )
    registration_summary = _read_json(registration_summary_path)
    model_path = registration_directory / "model_versions.jsonl.gz"
    simulations_path = registration_directory / "simulations.jsonl.gz"
    label_path = registration_directory / "validation_run_labels.jsonl.gz"
    model = OSPModelVersionSnapshot.model_validate(_single_row(model_path))
    label = ModelValidationRunLabel.model_validate(_single_row(label_path))
    validation_audit = _read_json(validation_audit_path)
    metrics_path = Path(label.validation_dataset_metrics_path)
    datasets = list(_iter_jsonl(metrics_path))
    simulations = list(_iter_jsonl(simulations_path))
    registration_checks = {
        "registration_accepted": registration_summary.get("accepted") is True,
        "registration_model_matches": (
            registration_summary.get("model_id") == model.model_id
            and registration_summary.get("model_version_id")
            == model.model_version_id
        ),
        "model_label_matches": model.validation_run_label == label,
        "grade_3_label": (
            label.run_label_grade == 3
            and label.validation_grade == "external_study_validated"
            and label.ranking_label == "reuse_positive"
            and label.engineering_accepted
            and label.scientific_accepted
            and label.reusable_candidate
        ),
        "validation_audit_bound": (
            sha256_file(validation_audit_path)
            == label.audit_report_sha256
            == (registration_summary.get("input_checksums") or {}).get(
                "audit_report"
            )
        ),
        "validation_dataset_count_matches": (
            len(datasets) == label.verification_dataset_count
        ),
        "simulation_count_matches": len(simulations) == model.simulation_count,
    }
    for filename, expected in sorted(
        (registration_summary.get("output_checksums") or {}).items()
    ):
        path = registration_directory / filename
        registration_checks[f"checksum:{filename}"] = (
            path.is_file() and sha256_file(path) == expected
        )
    if not all(registration_checks.values()):
        failed = sorted(
            name for name, passed in registration_checks.items() if not passed
        )
        raise LicensedModelGraphError(
            "Licensed model registration integrity failed: "
            + ", ".join(failed)
        )

    nodes = list(_iter_jsonl(base_graph_directory / "kg_nodes.jsonl"))
    edges = list(_iter_jsonl(base_graph_directory / "kg_edges.jsonl"))
    events = list(_iter_jsonl(base_graph_directory / "kg_events.jsonl"))
    nodes_by_id = {str(row["node_id"]): row for row in nodes}
    edges_by_id = {str(row["edge_id"]): row for row in edges}
    events_by_id = {str(row["event_id"]): row for row in events}
    compound = nodes_by_id.get(compound_node_id)
    if compound is None or compound.get("node_type") != "Compound":
        raise LicensedModelGraphError(
            f"Locked compound node is absent: {compound_node_id}"
        )
    compound_names = {
        str(compound.get("preferred_name") or "").casefold(),
        *(
            str(value).casefold()
            for value in (compound.get("properties") or {}).get("aliases") or []
        ),
    }
    if not any(name.casefold() in compound_names for name in model.compounds):
        raise LicensedModelGraphError(
            "Locked compound node does not match the registered model compound."
        )

    def add_node(node: KGNode) -> None:
        row = node.model_dump(mode="json")
        existing = nodes_by_id.get(node.node_id)
        if existing is not None and _canonical_json(existing) != _canonical_json(row):
            raise LicensedModelGraphError(
                f"Node collision with different content: {node.node_id}"
            )
        if existing is None:
            nodes.append(row)
            nodes_by_id[node.node_id] = row

    def add_edge(edge: KGEdge) -> str:
        row = edge.model_dump(mode="json")
        existing = edges_by_id.get(edge.edge_id)
        if existing is not None and _canonical_json(existing) != _canonical_json(row):
            raise LicensedModelGraphError(
                f"Edge collision with different content: {edge.edge_id}"
            )
        if existing is None:
            edges.append(row)
            edges_by_id[edge.edge_id] = row
        return edge.edge_id

    def add_event(event: KGEvent) -> None:
        row = event.model_dump(mode="json")
        existing = events_by_id.get(event.event_id)
        if existing is not None and _canonical_json(existing) != _canonical_json(row):
            raise LicensedModelGraphError(
                f"Event collision with different content: {event.event_id}"
            )
        if existing is None:
            events.append(row)
            events_by_id[event.event_id] = row

    def make_edge(
        subject_id: str,
        subject_type: str,
        predicate: str,
        object_id: str,
        object_type: str,
        provenance: Provenance,
        properties: dict[str, Any] | None = None,
    ) -> str:
        edge_properties = properties or {}
        return add_edge(
            KGEdge(
                edge_id=stable_id(
                    "licensed-model-edge",
                    subject_id,
                    predicate,
                    object_id,
                    edge_properties,
                    provenance.evidence_id,
                ),
                subject_id=subject_id,
                subject_type=subject_type,
                predicate=predicate,
                object_id=object_id,
                object_type=object_type,
                provenance=provenance,
                properties=edge_properties,
            )
        )

    def resolve_or_add_concept(node_type: str, name: str) -> str:
        folded = " ".join(name.casefold().split())
        matches = []
        for row in nodes:
            if row.get("node_type") != node_type:
                continue
            names = {str(row.get("preferred_name") or "")}
            names.update(
                str(value)
                for value in (row.get("properties") or {}).get("aliases") or []
            )
            if folded in {" ".join(value.casefold().split()) for value in names}:
                matches.append(str(row["node_id"]))
        if len(matches) > 1:
            raise LicensedModelGraphError(
                f"Ambiguous {node_type} concept mapping: {name}"
            )
        if matches:
            return matches[0]
        node_id = stable_id("licensed-model-concept", node_type, folded)
        add_node(
            KGNode(
                node_id=node_id,
                node_type=node_type,
                preferred_name=name,
                standard_identifiers={"OSP_MODEL_NAME": name},
                properties={
                    "aliases": [name],
                    "mapping_status": "exact_registered_model_name",
                },
            )
        )
        return node_id

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
        raise LicensedModelGraphError(
            "Registered model lacks the validation-bound source snapshot."
        )
    snapshot_document_id = stable_id(
        "source-document",
        snapshot_asset.source_location,
        snapshot_asset.sha256,
    )
    add_node(
        KGNode(
            node_id=snapshot_document_id,
            node_type="SourceDocument",
            preferred_name=Path(snapshot_asset.relative_path).name,
            standard_identifiers={"SHA256": snapshot_asset.sha256},
            properties={
                "source_id": model.source_id,
                "source_location": snapshot_asset.source_location,
                "sha256": snapshot_asset.sha256,
                "document_kind": "locked_model_snapshot",
            },
        )
    )
    audit_document_id = stable_id(
        "source-document",
        label.validation_run_id,
        label.audit_report_sha256,
    )
    add_node(
        KGNode(
            node_id=audit_document_id,
            node_type="SourceDocument",
            preferred_name="snapshot_reuse_audit.json",
            standard_identifiers={"SHA256": label.audit_report_sha256},
            properties={
                "source_id": model.source_id,
                "source_location": (
                    f"VALIDATION_AUDIT:{label.validation_run_id}"
                ),
                "sha256": label.audit_report_sha256,
                "document_kind": "independent_validation_audit",
            },
        )
    )

    registration_text = (
        f"{model.model_family} public PBPK model release "
        f"{model.repository_commit} is licensed as {model.license.spdx_id}; "
        f"all {model.simulation_count} simulations were parsed and executed."
    )
    registration_evidence_id = stable_id(
        "evidence",
        model.model_version_id,
        snapshot_asset.sha256,
        "registered-model",
    )
    registration_provenance = _provenance(
        evidence_id=registration_evidence_id,
        source_id=model.source_id,
        source_location=snapshot_asset.source_location,
        supporting_text=registration_text,
        extraction_method="licensed_model_registration_projection",
    )
    add_node(
        KGNode(
            node_id=registration_evidence_id,
            node_type="Evidence",
            preferred_name="public_model:registered_executable_model",
            standard_identifiers={
                "MODEL_VERSION": model.model_version_id,
                "SOURCE_SHA256": snapshot_asset.sha256,
            },
            properties={
                "quality_status": "auto_validated",
                "evidence_kind": "registered_executable_model",
                "supporting_text": registration_text,
                "source_id": model.source_id,
                "source_location": snapshot_asset.source_location,
            },
        )
    )
    make_edge(
        registration_evidence_id,
        "Evidence",
        "DERIVED_FROM",
        snapshot_document_id,
        "SourceDocument",
        registration_provenance,
    )

    model_node_id = model.model_id
    add_node(
        KGNode(
            node_id=model_node_id,
            node_type="PBPKModel",
            preferred_name=f"{model.model_family} PBPK model",
            standard_identifiers={
                "MODEL_CHECKSUM": model.project_sha256,
                "OSP_REPOSITORY_COMMIT": model.repository_commit,
            },
            properties={
                "model_family": model.model_family,
                "software": model.software,
                "software_version": model.software_version,
                "repository_url": model.repository_url,
                "repository_commit": model.repository_commit,
                "license_spdx_id": model.license.spdx_id,
                "availability_status": "downloadable_executable_model",
                "model_checksum": model.project_sha256,
                "source_snapshot_checksum": label.source_snapshot_sha256,
                "reusable_candidate": True,
                "validation_grade": label.validation_grade,
                "run_label_grade": label.run_label_grade,
                "ranking_label": label.ranking_label,
                "validation_run_id": label.validation_run_id,
            },
        )
    )
    version_node_id = model.model_version_id
    add_node(
        KGNode(
            node_id=version_node_id,
            node_type="ModelVersion",
            preferred_name=(
                f"{model.model_family} v1.0 @ {model.repository_commit[:12]}"
            ),
            standard_identifiers={
                "MODEL_VERSION": model.model_version_id,
                "PROJECT_SHA256": model.project_sha256,
            },
            properties={
                "model_family": model.model_family,
                "repository_commit": model.repository_commit,
                "project_relative_path": model.project_relative_path,
                "project_sha256": model.project_sha256,
                "software": model.software,
                "software_version": model.software_version,
                "license": model.license.model_dump(mode="json"),
                "simulation_count": model.simulation_count,
                "parameter_record_count": model.parameter_record_count,
                "component_record_count": model.component_record_count,
                "process_record_count": model.process_record_count,
                "validation_run_id": label.validation_run_id,
                "validation_grade": label.validation_grade,
                "run_label_grade": label.run_label_grade,
                "ranking_label": label.ranking_label,
                "reusable_candidate": True,
                "routes": model.routes,
                "formulations": model.formulations,
                "species": model.species,
                "enzymes": model.enzymes,
                "transporters": model.transporters,
                "process_types": model.process_types,
            },
        )
    )
    core_edge_ids = [
        make_edge(
            compound_node_id,
            "Compound",
            "DESCRIBED_BY_MODEL",
            model_node_id,
            "PBPKModel",
            registration_provenance,
        ),
        make_edge(
            model_node_id,
            "PBPKModel",
            "HAS_MODEL_VERSION",
            version_node_id,
            "ModelVersion",
            registration_provenance,
        ),
        make_edge(
            model_node_id,
            "PBPKModel",
            "SUPPORTED_BY",
            registration_evidence_id,
            "Evidence",
            registration_provenance,
        ),
        make_edge(
            version_node_id,
            "ModelVersion",
            "SUPPORTED_BY",
            registration_evidence_id,
            "Evidence",
            registration_provenance,
        ),
    ]

    structure_id = stable_id(
        "model-structure",
        version_node_id,
        model.compartments,
        model.process_types,
    )
    add_node(
        KGNode(
            node_id=structure_id,
            node_type="ModelStructure",
            preferred_name=f"{model.model_family} whole-body structure",
            standard_identifiers={"MODEL_VERSION": version_node_id},
            properties={
                "structure_type": "whole_body_pbpk",
                "compartment_count": len(model.compartments),
                "process_types": model.process_types,
            },
        )
    )
    core_edge_ids.append(
        make_edge(
            version_node_id,
            "ModelVersion",
            "USES_STRUCTURE",
            structure_id,
            "ModelStructure",
            registration_provenance,
        )
    )
    for compartment in model.compartments:
        concept_id = resolve_or_add_concept("Compartment", compartment)
        core_edge_ids.append(
            make_edge(
                version_node_id,
                "ModelVersion",
                "HAS_COMPARTMENT",
                concept_id,
                "Compartment",
                registration_provenance,
            )
        )
    for node_type, values, predicate in (
        ("Enzyme", model.enzymes, "USES_ENZYME"),
        ("Transporter", model.transporters, "USES_TRANSPORTER"),
        ("AdministrationRoute", model.routes, "SUPPORTS_ROUTE"),
        ("Formulation", model.formulations, "SUPPORTS_FORMULATION"),
        ("Species", model.species, "APPLIES_TO"),
    ):
        for value in values:
            concept_id = resolve_or_add_concept(node_type, value)
            core_edge_ids.append(
                make_edge(
                    version_node_id,
                    "ModelVersion",
                    predicate,
                    concept_id,
                    node_type,
                    registration_provenance,
                )
            )

    for simulation in simulations:
        simulation_id = stable_id(
            "simulation-scenario",
            version_node_id,
            simulation["simulation_id"],
        )
        add_node(
            KGNode(
                node_id=simulation_id,
                node_type="SimulationScenario",
                preferred_name=str(simulation["simulation_name"]),
                standard_identifiers={
                    "MODEL_SIMULATION": str(simulation["simulation_id"]),
                    "PKML_SHA256": str(
                        simulation["pkml_source"]["sha256"]
                    ),
                },
                properties={
                    key: simulation.get(key)
                    for key in (
                        "simulation_id",
                        "simulation_name",
                        "protocol_sequence",
                        "protocol_role",
                        "validation_dataset_name",
                        "solver",
                        "routes",
                        "formulations",
                        "species",
                        "populations",
                        "parameter_count",
                        "component_count",
                        "process_count",
                        "administration_count",
                    )
                },
            )
        )
        core_edge_ids.append(
            make_edge(
                version_node_id,
                "ModelVersion",
                "HAS_SIMULATION",
                simulation_id,
                "SimulationScenario",
                registration_provenance,
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
            ("Species", simulation.get("species") or [], "APPLIES_TO"),
            (
                "Population",
                simulation.get("populations") or [],
                "APPLIES_TO",
            ),
        ):
            for value in values:
                concept_id = resolve_or_add_concept(node_type, str(value))
                core_edge_ids.append(
                    make_edge(
                        simulation_id,
                        "SimulationScenario",
                        predicate,
                        concept_id,
                        node_type,
                        registration_provenance,
                    )
                )

    audit_text = (
        f"Independent validation accepted {len(datasets)} external datasets "
        f"with pooled GMFE {label.pooled_gmfe:.6g}, geometric bias "
        f"{label.pooled_geometric_bias:.6g}, and two-fold fraction "
        f"{label.pooled_two_fold_fraction:.6g}."
    )
    audit_evidence_id = stable_id(
        "evidence",
        label.validation_run_id,
        label.audit_report_sha256,
        "validation-run",
    )
    audit_provenance = _provenance(
        evidence_id=audit_evidence_id,
        source_id=model.source_id,
        source_location=f"VALIDATION_AUDIT:{label.validation_run_id}",
        supporting_text=audit_text,
        extraction_method="licensed_snapshot_validation_projection",
        context_id=label.validation_run_id,
    )
    add_node(
        KGNode(
            node_id=audit_evidence_id,
            node_type="Evidence",
            preferred_name="public_model_validation:grade_3",
            standard_identifiers={
                "VALIDATION_RUN": label.validation_run_id,
                "AUDIT_SHA256": label.audit_report_sha256,
            },
            properties={
                "quality_status": "auto_validated",
                "evidence_kind": "reproduced_external_validation",
                "supporting_text": audit_text,
                "source_id": model.source_id,
                "source_location": (
                    f"VALIDATION_AUDIT:{label.validation_run_id}"
                ),
            },
        )
    )
    make_edge(
        audit_evidence_id,
        "Evidence",
        "DERIVED_FROM",
        audit_document_id,
        "SourceDocument",
        audit_provenance,
    )
    validation_event_id = label.validation_run_id
    add_node(
        KGNode(
            node_id=validation_event_id,
            node_type="ModelValidationEvent",
            preferred_name=f"{model.model_family} grade-3 validation",
            standard_identifiers={
                "VALIDATION_RUN": label.validation_run_id,
                "PROTOCOL": label.protocol_id,
            },
            properties={
                "validation_grade": label.validation_grade,
                "run_label_grade": label.run_label_grade,
                "ranking_label": label.ranking_label,
                "engineering_accepted": label.engineering_accepted,
                "scientific_accepted": label.scientific_accepted,
                "reusable_candidate": label.reusable_candidate,
                "verification_dataset_count": (
                    label.verification_dataset_count
                ),
                "pooled_observation_count": label.pooled_observation_count,
                "pooled_gmfe": label.pooled_gmfe,
                "pooled_geometric_bias": label.pooled_geometric_bias,
                "pooled_two_fold_fraction": (
                    label.pooled_two_fold_fraction
                ),
                "endpoint_scenario_pass_fraction": (
                    label.endpoint_scenario_pass_fraction
                ),
                "validated_at": label.validated_at.isoformat(),
                "audit_report_sha256": label.audit_report_sha256,
            },
        )
    )
    validation_edge_ids = [
        make_edge(
            version_node_id,
            "ModelVersion",
            "HAS_VALIDATION_RUN",
            validation_event_id,
            "ModelValidationEvent",
            audit_provenance,
        ),
        make_edge(
            validation_event_id,
            "ModelValidationEvent",
            "SUPPORTED_BY",
            audit_evidence_id,
            "Evidence",
            audit_provenance,
        ),
    ]

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
        metric_id = stable_id(
            "validation-metric",
            validation_event_id,
            "pooled",
            metric_name,
        )
        add_node(
            KGNode(
                node_id=metric_id,
                node_type="ValidationMetric",
                preferred_name=metric_name,
                standard_identifiers={
                    "VALIDATION_RUN": validation_event_id,
                },
                properties={
                    "value": value,
                    "scope": "pooled_external_validation",
                    "status": "reproduced",
                },
            )
        )
        validation_edge_ids.extend(
            [
                make_edge(
                    validation_event_id,
                    "ModelValidationEvent",
                    "HAS_VALIDATION_METRIC",
                    metric_id,
                    "ValidationMetric",
                    audit_provenance,
                ),
                make_edge(
                    metric_id,
                    "ValidationMetric",
                    "SUPPORTED_BY",
                    audit_evidence_id,
                    "Evidence",
                    audit_provenance,
                ),
            ]
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
        dataset_name = str(dataset["dataset_name"])
        dataset_id = stable_id(
            "qualification-dataset",
            validation_event_id,
            dataset_name,
            dataset.get("source_location"),
        )
        add_node(
            KGNode(
                node_id=dataset_id,
                node_type="QualificationDataset",
                preferred_name=dataset_name,
                standard_identifiers={
                    "VALIDATION_RUN": validation_event_id,
                    "DATASET_SEQUENCE": str(
                        dataset.get("simulation_sequence")
                    ),
                },
                properties={
                    key: dataset.get(key)
                    for key in (
                        "reference",
                        "source_location",
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
                | {
                    "role": "external_validation",
                    "validation_run_id": validation_event_id,
                },
            )
        )
        validation_edge_ids.extend(
            [
                make_edge(
                    version_node_id,
                    "ModelVersion",
                    "VALIDATED_WITH",
                    dataset_id,
                    "QualificationDataset",
                    audit_provenance,
                ),
                make_edge(
                    validation_event_id,
                    "ModelValidationEvent",
                    "VALIDATED_WITH",
                    dataset_id,
                    "QualificationDataset",
                    audit_provenance,
                ),
                make_edge(
                    dataset_id,
                    "QualificationDataset",
                    "SUPPORTED_BY",
                    audit_evidence_id,
                    "Evidence",
                    audit_provenance,
                ),
            ]
        )
        for metric_name in dataset_metric_names:
            value = dataset.get(metric_name)
            if value is None:
                continue
            metric_id = stable_id(
                "validation-metric",
                validation_event_id,
                dataset_id,
                metric_name,
            )
            add_node(
                KGNode(
                    node_id=metric_id,
                    node_type="ValidationMetric",
                    preferred_name=metric_name,
                    standard_identifiers={
                        "VALIDATION_RUN": validation_event_id,
                        "QUALIFICATION_DATASET": dataset_id,
                    },
                    properties={
                        "value": value,
                        "scope": "external_validation_dataset",
                        "dataset_id": dataset_id,
                        "status": "reproduced",
                    },
                )
            )
            validation_edge_ids.extend(
                [
                    make_edge(
                        validation_event_id,
                        "ModelValidationEvent",
                        "HAS_VALIDATION_METRIC",
                        metric_id,
                        "ValidationMetric",
                        audit_provenance,
                    ),
                    make_edge(
                        metric_id,
                        "ValidationMetric",
                        "EVALUATES_DATASET",
                        dataset_id,
                        "QualificationDataset",
                        audit_provenance,
                    ),
                    make_edge(
                        metric_id,
                        "ValidationMetric",
                        "SUPPORTED_BY",
                        audit_evidence_id,
                        "Evidence",
                        audit_provenance,
                    ),
                ]
            )

    add_event(
        KGEvent(
            event_id=validation_event_id,
            event_type="ModelValidationEvent",
            evidence_id=audit_evidence_id,
            trigger_text="external study validation accepted",
            argument_edge_ids=validation_edge_ids,
            context={
                "model_version_id": version_node_id,
                "validation_run_id": validation_event_id,
                "run_label_grade": 3,
                "dataset_count": len(datasets),
            },
            extraction_method="licensed_snapshot_validation_projection",
            extractor_version=PROJECTOR_VERSION,
            confidence=1.0,
            validation_status="validated",
            validation_reasons=[
                "engineering_checks_passed",
                "scientific_checks_passed",
                "external_dataset_coverage_complete",
            ],
        )
    )

    endpoint_errors = []
    for edge in edges:
        if str(edge["subject_id"]) not in nodes_by_id:
            endpoint_errors.append(f"missing_subject:{edge['edge_id']}")
        if str(edge["object_id"]) not in nodes_by_id:
            endpoint_errors.append(f"missing_object:{edge['edge_id']}")
    event_errors = []
    for event in events:
        missing = sorted(
            set(str(value) for value in event.get("argument_edge_ids") or [])
            - set(edges_by_id)
        )
        if missing:
            event_errors.append(
                f"event_missing_edges:{event['event_id']}:{','.join(missing[:5])}"
            )

    working.mkdir(parents=True)
    node_rows = sorted(nodes, key=lambda row: str(row["node_id"]))
    edge_rows = sorted(edges, key=lambda row: str(row["edge_id"]))
    event_rows = sorted(events, key=lambda row: str(row["event_id"]))
    output_checksums = {
        "kg_nodes.jsonl": _write_jsonl(
            working / "kg_nodes.jsonl",
            node_rows,
        ),
        "kg_edges.jsonl": _write_jsonl(
            working / "kg_edges.jsonl",
            edge_rows,
        ),
        "kg_events.jsonl": _write_jsonl(
            working / "kg_events.jsonl",
            event_rows,
        ),
    }
    node_type_counts = Counter(str(row["node_type"]) for row in node_rows)
    edge_type_counts = Counter(str(row["predicate"]) for row in edge_rows)
    event_type_counts = Counter(str(row["event_type"]) for row in event_rows)
    summary = {
        "accepted": not endpoint_errors and not event_errors,
        "run_id": f"{base_summary.get('run_id')}-licensed-model-projection",
        "ontology_version": ONTOLOGY_VERSION,
        "formal_graph_version": FORMAL_GRAPH_VERSION,
        "projector_version": PROJECTOR_VERSION,
        "base_graph_directory": str(base_graph_directory),
        "base_graph_manifest_sha256": sha256_file(base_manifest_path),
        "registration_summary_sha256": sha256_file(
            registration_summary_path
        ),
        "validation_audit_sha256": sha256_file(validation_audit_path),
        "node_count": len(node_rows),
        "edge_count": len(edge_rows),
        "event_count": len(event_rows),
        "node_type_counts": dict(sorted(node_type_counts.items())),
        "edge_type_counts": dict(sorted(edge_type_counts.items())),
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "projected_model_count": 1,
        "projected_model_version_count": 1,
        "projected_simulation_count": len(simulations),
        "projected_validation_dataset_count": len(datasets),
        "projected_validation_metric_count": (
            4
            + sum(
                dataset.get(name) is not None
                for dataset in datasets
                for name in dataset_metric_names
            )
        ),
        "output_checksums": output_checksums,
        "endpoint_errors": endpoint_errors,
        "event_errors": event_errors,
    }
    _write_json(working / "kg_build_summary.json", summary)
    manifest = {
        "manifest_schema_version": "1.0.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": summary["run_id"],
        "ontology_version": ONTOLOGY_VERSION,
        "formal_graph_version": FORMAL_GRAPH_VERSION,
        "projector_version": PROJECTOR_VERSION,
        "base_graph_manifest": {
            "path": str(base_manifest_path),
            "sha256": sha256_file(base_manifest_path),
        },
        "base_gate1_audit": {
            "path": str(base_gate1_audit_path),
            "sha256": sha256_file(base_gate1_audit_path),
        },
        "licensed_model_registration": {
            "path": str(registration_summary_path),
            "sha256": sha256_file(registration_summary_path),
        },
        "validation_audit": {
            "path": str(validation_audit_path),
            "sha256": sha256_file(validation_audit_path),
        },
        "compound_node_id": compound_node_id,
        "model_id": model.model_id,
        "model_version_id": model.model_version_id,
        "validation_run_id": label.validation_run_id,
        "output_checksums": output_checksums,
    }
    manifest_path = working / "kg_extraction_manifest.json"
    _write_json(manifest_path, manifest)
    checks = {
        **base_checks,
        **registration_checks,
        "compound_identity_exact": True,
        "all_edge_endpoints_exist": not endpoint_errors,
        "all_event_arguments_exist": not event_errors,
        "grade_3_model_projected": (
            nodes_by_id[model_node_id]["properties"]["run_label_grade"] == 3
            and nodes_by_id[model_node_id]["properties"][
                "reusable_candidate"
            ]
            is True
        ),
        "validation_dataset_coverage_complete": len(datasets)
        == label.verification_dataset_count,
        "simulation_coverage_complete": len(simulations)
        == model.simulation_count,
    }
    audit = {
        "audit_schema_version": "1.0.0",
        "auditor_version": PROJECTOR_VERSION,
        "audited_at": datetime.now(UTC).isoformat(),
        "accepted": all(checks.values()),
        "checks": checks,
        "counts": {
            "nodes": len(node_rows),
            "edges": len(edge_rows),
            "events": len(event_rows),
            "projected_simulations": len(simulations),
            "projected_validation_datasets": len(datasets),
            "projected_validation_metrics": summary[
                "projected_validation_metric_count"
            ],
        },
        "inputs": {
            "graph_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "base_graph_manifest": {
                "path": str(base_manifest_path),
                "sha256": sha256_file(base_manifest_path),
            },
            "base_gate1_audit": {
                "path": str(base_gate1_audit_path),
                "sha256": sha256_file(base_gate1_audit_path),
            },
            "registration_summary": {
                "path": str(registration_summary_path),
                "sha256": sha256_file(registration_summary_path),
            },
            "validation_audit": {
                "path": str(validation_audit_path),
                "sha256": sha256_file(validation_audit_path),
            },
        },
        "errors": {
            "endpoint_errors": endpoint_errors,
            "event_errors": event_errors,
        },
    }
    _write_json(working / "projection_audit.json", audit)
    if not summary["accepted"] or not audit["accepted"]:
        raise LicensedModelGraphError(
            "Licensed model recommendation graph projection failed audit."
        )
    working.replace(output_directory)
    return {
        **summary,
        "projection_audit_sha256": sha256_file(
            output_directory / "projection_audit.json"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Project a grade-3 licensed PBPK model into an audited "
            "recommendation graph snapshot."
        )
    )
    parser.add_argument("--base-graph", type=Path, required=True)
    parser.add_argument("--base-audit", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--validation-audit", type=Path, required=True)
    parser.add_argument("--compound-node-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = project_licensed_model_into_recommendation_graph(
        base_graph_directory=args.base_graph,
        base_gate1_audit_path=args.base_audit,
        registration_directory=args.registration,
        validation_audit_path=args.validation_audit,
        compound_node_id=args.compound_node_id,
        output_directory=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
