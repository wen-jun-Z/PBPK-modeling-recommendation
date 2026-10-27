from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.recommendation.ontology import validate_relation
from pbpk_kb.recommendation.schemas import RecommendationRequest


AUDITOR_VERSION = "pbpk-recommendation-gate1-auditor-1.0.0"
EVENT_REQUIRED_PREDICATES = {
    "DDIEvent": {
        "REPORTS",
        "PERPETRATOR_IN",
        "VICTIM_IN",
        "SUPPORTED_BY",
    },
    "PKMeasurement": {
        "REPORTS",
        "HAS_MEASUREMENT",
        "OF_COMPOUND",
        "MEASURES",
        "SUPPORTED_BY",
    },
    "AbsorptionEvent": {
        "REPORTS",
        "ABSORBED_VIA",
        "OF_COMPOUND",
        "MEASURES",
        "SUPPORTED_BY",
    },
    "ExcretionEvent": {
        "REPORTS",
        "EXCRETED_VIA",
        "OF_COMPOUND",
        "MEASURES",
        "SUPPORTED_BY",
    },
    "MetabolismEvent": {
        "REPORTS",
        "OF_COMPOUND",
        "MEASURES",
        "SUPPORTED_BY",
    },
}
STABLE_COMPOUND_NAMESPACES = {
    "PUBCHEM.COMPOUND",
    "INCHIKEY",
    "CHEMBL",
    "DRUGCENTRAL",
    "RXNORM",
    "UNII",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent PBPK recommendation Gate 1 audit.")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--ner-report", type=Path, required=True)
    parser.add_argument("--entity-link-report", type=Path, required=True)
    parser.add_argument("--supervised-re-report", type=Path, required=True)
    parser.add_argument("--ddi-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_gate1(
        evidence_path=args.evidence,
        graph_dir=args.graph_dir,
        ner_report_path=args.ner_report,
        entity_link_report_path=args.entity_link_report,
        supervised_re_report_path=args.supervised_re_report,
        ddi_report_path=args.ddi_report,
    )
    _atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise RuntimeError("Gate 1 audit failed; inspect the generated report.")


def audit_gate1(
    *,
    evidence_path: Path,
    graph_dir: Path,
    ner_report_path: Path,
    entity_link_report_path: Path,
    supervised_re_report_path: Path,
    ddi_report_path: Path,
) -> dict[str, Any]:
    nodes = list(_iter_jsonl(graph_dir / "kg_nodes.jsonl"))
    edges = list(_iter_jsonl(graph_dir / "kg_edges.jsonl"))
    events = list(_iter_jsonl(graph_dir / "kg_events.jsonl"))
    mentions = list(_iter_jsonl(graph_dir / "kg_entity_mentions.jsonl"))
    unresolved = list(_iter_jsonl(graph_dir / "kg_unresolved_entities.jsonl"))
    evidence = list(_iter_jsonl(evidence_path))
    graph_summary = _read_json(graph_dir / "kg_build_summary.json")
    graph_manifest = _read_json(graph_dir / "kg_extraction_manifest.json")
    ner_report = _read_json(ner_report_path)
    entity_link_report = _read_json(entity_link_report_path)
    supervised_re_report = _read_json(supervised_re_report_path)
    ddi_report = _read_json(ddi_report_path)

    node_ids = [str(row["node_id"]) for row in nodes]
    edge_ids = [str(row["edge_id"]) for row in edges]
    event_ids = [str(row["event_id"]) for row in events]
    node_by_id = {str(row["node_id"]): row for row in nodes}
    edge_by_id = {str(row["edge_id"]): row for row in edges}
    event_predicates: dict[str, set[str]] = defaultdict(set)
    ontology_errors = []
    provenance_errors = []
    endpoint_errors = []
    for edge in edges:
        edge_id = str(edge["edge_id"])
        subject = node_by_id.get(str(edge["subject_id"]))
        object_ = node_by_id.get(str(edge["object_id"]))
        if subject is None or object_ is None:
            endpoint_errors.append(edge_id)
            continue
        error = validate_relation(
            str(edge["predicate"]),
            str(edge["subject_type"]),
            str(edge["object_type"]),
        )
        if error:
            ontology_errors.append(f"{edge_id}:{error}")
        event_predicates[str(edge["subject_id"])].add(str(edge["predicate"]))
        event_predicates[str(edge["object_id"])].add(str(edge["predicate"]))
        provenance = edge.get("provenance") or {}
        if not all(
            provenance.get(key)
            for key in (
                "evidence_id",
                "source_id",
                "source_location",
                "supporting_text",
                "source_text_hash",
                "extraction_method",
                "extractor_version",
            )
        ):
            provenance_errors.append(edge_id)
        elif provenance.get("quality_status") != "auto_validated":
            provenance_errors.append(edge_id)
        elif len(str(provenance.get("source_text_hash"))) != 64:
            provenance_errors.append(edge_id)

    event_errors = []
    for event in events:
        event_id = str(event["event_id"])
        required = EVENT_REQUIRED_PREDICATES.get(str(event["event_type"]), set())
        missing_predicates = sorted(required - event_predicates.get(event_id, set()))
        missing_arguments = sorted(
            set(str(value) for value in event.get("argument_edge_ids") or []) - set(edge_by_id)
        )
        if missing_predicates or missing_arguments:
            event_errors.append(
                {
                    "event_id": event_id,
                    "event_type": event["event_type"],
                    "missing_predicates": missing_predicates,
                    "missing_argument_edges": missing_arguments,
                }
            )

    regimen_errors = []
    regimen_ids = {
        str(row["node_id"])
        for row in nodes
        if row.get("node_type") == "DoseRegimen"
    }
    incoming_under_regimen = {
        str(edge["object_id"])
        for edge in edges
        if edge.get("predicate") == "UNDER_REGIMEN"
    }
    route_edges = {
        str(edge["subject_id"])
        for edge in edges
        if edge.get("predicate") == "HAS_ROUTE"
    }
    formulation_edges = {
        str(edge["subject_id"])
        for edge in edges
        if edge.get("predicate") == "HAS_FORMULATION"
    }
    for regimen_id in sorted(regimen_ids):
        node = node_by_id[regimen_id]
        properties = node.get("properties") or {}
        if not any(properties.get(key) not in (None, "") for key in ("dose", "route", "formulation", "frequency")):
            regimen_errors.append(f"{regimen_id}:empty_regimen")
        if regimen_id not in incoming_under_regimen:
            regimen_errors.append(f"{regimen_id}:not_used_by_event")
        if properties.get("route") not in (None, "") and regimen_id not in route_edges:
            regimen_errors.append(f"{regimen_id}:route_property_without_edge")
        if properties.get("formulation") not in (None, "") and regimen_id not in formulation_edges:
            regimen_errors.append(f"{regimen_id}:formulation_property_without_edge")

    name_only_compounds = []
    for node in nodes:
        if node.get("node_type") != "Compound":
            continue
        namespaces = set((node.get("standard_identifiers") or {}).keys())
        if not namespaces & STABLE_COMPOUND_NAMESPACES:
            name_only_compounds.append(str(node["node_id"]))

    evidence_sha = _sha256(evidence_path)
    report_paths = {
        "evidence": evidence_path,
        "graph_summary": graph_dir / "kg_build_summary.json",
        "graph_manifest": graph_dir / "kg_extraction_manifest.json",
        "ner_report": ner_report_path,
        "entity_link_report": entity_link_report_path,
        "supervised_re_report": supervised_re_report_path,
        "ddi_report": ddi_report_path,
    }
    request_schema_checks = _request_schema_checks()
    checks = {
        "graph_builder_accepted": bool(graph_summary.get("accepted")),
        "ner_gate_accepted": bool(ner_report.get("accepted")),
        "entity_link_gate_accepted": bool(entity_link_report.get("accepted")),
        "supervised_re_gate_accepted": bool(supervised_re_report.get("accepted")),
        "ddi_revalidation_accepted": bool(ddi_report.get("accepted")),
        "evidence_input_hash_consistent": (
            graph_summary.get("input_sha256") == evidence_sha
            and graph_manifest.get("input_sha256") == evidence_sha
            and ner_report.get("input_sha256") == evidence_sha
        ),
        "auto_validated_partition_only": bool(evidence)
        and all(row.get("quality_status") == "auto_validated" for row in evidence),
        "node_ids_unique": len(node_ids) == len(set(node_ids)),
        "edge_ids_unique": len(edge_ids) == len(set(edge_ids)),
        "event_ids_unique": len(event_ids) == len(set(event_ids)),
        "all_edge_endpoints_exist": not endpoint_errors,
        "ontology_domain_range_valid": not ontology_errors,
        "all_scientific_edges_have_provenance": not provenance_errors,
        "required_event_roles_complete": not event_errors,
        "dose_regimen_consistent": not regimen_errors,
        "name_only_compounds_excluded": not name_only_compounds,
        "unresolved_mentions_not_edge_endpoints": all(
            str(row["mention_id"]) not in {
                str(edge["subject_id"]) for edge in edges
            }
            | {str(edge["object_id"]) for edge in edges}
            for row in unresolved
        ),
        "modeling_request_schema_validated": all(request_schema_checks.values()),
    }
    return {
        "accepted": all(checks.values()),
        "auditor_version": AUDITOR_VERSION,
        "checks": checks,
        "counts": {
            "evidence": len(evidence),
            "nodes": len(nodes),
            "edges": len(edges),
            "events": len(events),
            "mentions": len(mentions),
            "unresolved_mentions": len(unresolved),
            "dose_regimens": len(regimen_ids),
            "name_only_compounds": len(name_only_compounds),
        },
        "event_type_counts": dict(
            sorted(Counter(str(row["event_type"]) for row in events).items())
        ),
        "errors": {
            "endpoint_errors": endpoint_errors[:100],
            "ontology_errors": ontology_errors[:100],
            "provenance_errors": provenance_errors[:100],
            "event_errors": event_errors[:100],
            "regimen_errors": regimen_errors[:100],
            "name_only_compounds": name_only_compounds[:100],
        },
        "request_schema_checks": request_schema_checks,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in report_paths.items()
        },
        "limitations": [
            "NER and entity-link metrics are locked silver benchmarks, not expert gold accuracy.",
            "Unresolved entities and relations remain in audit outputs and are excluded from accepted facts.",
            "Gate 1 validates extraction and ontology integrity; it does not validate GNN recommendation quality.",
        ],
    }


def _request_schema_checks() -> dict[str, bool]:
    base = RecommendationRequest(
        compound="midazolam",
        objective="base_pk",
        route="oral",
        dose=5,
        dose_unit="mg",
        frequency="single dose",
        prediction_endpoints=["AUC", "Cmax"],
    )
    ddi = RecommendationRequest(
        compound="midazolam",
        objective="ddi_victim",
        perpetrator_drug="ketoconazole",
        victim_regimen={"route": "oral", "dose": 5, "dose_unit": "mg"},
        prediction_endpoints=["DDI_AUC_ratio"],
    )
    population = RecommendationRequest(
        compound="midazolam",
        objective="hepatic_impairment",
        population="adults with hepatic impairment",
        hepatic_function="Child-Pugh A-C",
    )
    return {
        "base_request_has_no_gap": not base.request_gaps(),
        "ddi_request_has_no_role_gap": not ddi.request_gaps(),
        "population_request_has_no_population_gap": not population.request_gaps(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object.")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            yield value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
