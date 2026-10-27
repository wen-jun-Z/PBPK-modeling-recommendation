from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.recommendation.entity_extraction import (
    EntityCatalog,
    GlinerNERBackend,
    NullNERBackend,
    PBPKEntityExtractor,
)
from pbpk_kb.recommendation.kg_models import ExtractionManifest
from pbpk_kb.recommendation.ontology import ONTOLOGY_VERSION
from pbpk_kb.recommendation.relation_extraction import PBPKRelationExtractor


OUTPUT_DIR_NAME = "08_recommendation_kg"
STATE_FILE_NAME = "row_extraction_state.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ontology-constrained PBPK recommendation graph.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--ner-backend", choices=("gliner", "none"), default="gliner")
    parser.add_argument("--ner-model", default="Ihor/gliner-biomed-large-v1.0")
    parser.add_argument("--ner-threshold", type=float, default=0.45)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--external-catalog", type=Path)
    parser.add_argument("--ddi-decisions", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or args.input.parent.parent / OUTPUT_DIR_NAME
    summary = run_pipeline(
        input_path=args.input,
        output_dir=output_dir,
        run_id=args.run_id or args.input.parent.parent.name,
        ner_backend=args.ner_backend,
        ner_model=args.ner_model,
        ner_threshold=args.ner_threshold,
        device=args.device,
        limit=args.limit,
        external_catalog_path=args.external_catalog,
        ddi_decisions_path=args.ddi_decisions,
        resume=args.resume,
        reset=args.reset,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def run_pipeline(
    *,
    input_path: Path,
    output_dir: Path,
    run_id: str,
    ner_backend: str,
    ner_model: str,
    ner_threshold: float,
    device: str,
    limit: int | None = None,
    external_catalog_path: Path | None = None,
    ddi_decisions_path: Path | None = None,
    resume: bool = False,
    reset: bool = False,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if reset and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / STATE_FILE_NAME
    if state_path.exists() and not resume and not reset:
        raise RuntimeError(f"State exists at {state_path}; use --resume or --reset.")

    rows = list(_iter_jsonl(input_path))
    if limit is not None:
        rows = rows[:limit]
    _validate_input_partition(rows)
    external_entries = list(_iter_jsonl(external_catalog_path)) if external_catalog_path else []
    ddi_decision_rows = list(_iter_jsonl(ddi_decisions_path)) if ddi_decisions_path else []
    ddi_decisions = {str(row["evidence_id"]): row for row in ddi_decision_rows}
    if len(ddi_decisions) != len(ddi_decision_rows):
        raise ValueError("Duplicate evidence_id values in DDI relation decisions.")
    unknown_decisions = sorted(set(ddi_decisions) - {str(row["evidence_id"]) for row in rows})
    if unknown_decisions:
        raise ValueError(f"DDI decisions reference unknown evidence IDs: {unknown_decisions[:5]}")
    catalog = EntityCatalog(rows, external_entries)
    ner = (
        GlinerNERBackend(ner_model, threshold=ner_threshold, device=device)
        if ner_backend == "gliner"
        else NullNERBackend()
    )
    entity_extractor = PBPKEntityExtractor(catalog, ner)
    relation_extractor = PBPKRelationExtractor(catalog, ddi_decisions)
    completed = _load_state(state_path) if resume else {}

    for index, row in enumerate(rows, start=1):
        evidence_id = str(row["evidence_id"])
        if evidence_id in completed:
            continue
        mentions, unresolved = entity_extractor.extract(row)
        relations = relation_extractor.extract(row, mentions)
        bundle = {
            "evidence_id": evidence_id,
            "mentions": [item.model_dump(mode="json") for item in mentions],
            "unresolved": [item.model_dump(mode="json") for item in unresolved],
            "nodes": [item.model_dump(mode="json") for item in relations.nodes],
            "edges": [item.model_dump(mode="json") for item in relations.edges],
            "events": [item.model_dump(mode="json") for item in relations.events],
            "rejected": [item.model_dump(mode="json") for item in relations.rejected],
        }
        _append_jsonl(state_path, bundle)
        completed[evidence_id] = bundle
        if index % 25 == 0 or index == len(rows):
            print(json.dumps({"processed": index, "total": len(rows)}, sort_keys=True), flush=True)

    if len(completed) != len(rows):
        raise RuntimeError(f"Extraction state is incomplete: {len(completed)} of {len(rows)}.")
    ordered = [completed[str(row["evidence_id"])] for row in rows]
    return _finalize(
        input_path=input_path,
        external_catalog_path=external_catalog_path,
        external_catalog_count=len(external_entries),
        ddi_decisions_path=ddi_decisions_path,
        ddi_relation_decision_count=len(ddi_decision_rows),
        output_dir=output_dir,
        run_id=run_id,
        rows=rows,
        bundles=ordered,
        catalog=catalog,
        ner_backend=ner_backend,
        ner_model=ner.model_name,
        ner_threshold=ner_threshold,
    )


def _finalize(
    *,
    input_path: Path,
    external_catalog_path: Path | None,
    external_catalog_count: int,
    ddi_decisions_path: Path | None,
    ddi_relation_decision_count: int,
    output_dir: Path,
    run_id: str,
    rows: list[dict[str, Any]],
    bundles: list[dict[str, Any]],
    catalog: EntityCatalog,
    ner_backend: str,
    ner_model: str,
    ner_threshold: float,
) -> dict[str, Any]:
    mentions = _deduplicate((item for bundle in bundles for item in bundle["mentions"]), "mention_id")
    unresolved = _deduplicate((item for bundle in bundles for item in bundle["unresolved"]), "mention_id")
    nodes = {
        item.node_id: item.model_dump(mode="json")
        for item in catalog.nodes()
    }
    for item in (item for bundle in bundles for item in bundle["nodes"]):
        current = nodes.get(item["node_id"])
        if current is not None and current["node_type"] != item["node_type"]:
            raise RuntimeError(f"Node type collision for {item['node_id']}.")
        nodes[item["node_id"]] = item
    edges = _deduplicate((item for bundle in bundles for item in bundle["edges"]), "edge_id")
    events = _deduplicate((item for bundle in bundles for item in bundle["events"]), "event_id")
    rejected = _deduplicate((item for bundle in bundles for item in bundle["rejected"]), "rejection_id")

    endpoint_errors = []
    for edge in edges:
        subject = nodes.get(edge["subject_id"])
        object_ = nodes.get(edge["object_id"])
        if subject is None:
            endpoint_errors.append(f"missing_subject:{edge['edge_id']}:{edge['subject_id']}")
        elif subject["node_type"] != edge["subject_type"]:
            endpoint_errors.append(f"subject_type_mismatch:{edge['edge_id']}")
        if object_ is None:
            endpoint_errors.append(f"missing_object:{edge['edge_id']}:{edge['object_id']}")
        elif object_["node_type"] != edge["object_type"]:
            endpoint_errors.append(f"object_type_mismatch:{edge['edge_id']}")
    event_errors = []
    edge_ids = {edge["edge_id"] for edge in edges}
    for event in events:
        missing = sorted(set(event["argument_edge_ids"]) - edge_ids)
        if missing:
            event_errors.append(f"event_missing_edges:{event['event_id']}:{','.join(missing)}")

    quality_counts = Counter(str(row.get("quality_status")) for row in rows)
    relation_method_counts = Counter(
        edge["provenance"]["extraction_method"]
        for edge in edges
    )
    node_type_counts = Counter(node["node_type"] for node in nodes.values())
    edge_type_counts = Counter(edge["predicate"] for edge in edges)
    event_type_counts = Counter(event["event_type"] for event in events)
    acceptance_checks = {
        "input_auto_validated_only": quality_counts == {"auto_validated": len(rows)},
        "all_rows_processed_once": len(bundles) == len(rows),
        "all_edge_endpoints_exist": not endpoint_errors,
        "all_event_arguments_exist": not event_errors,
        "all_scientific_edges_have_provenance": all(
            bool(edge.get("provenance", {}).get("evidence_id"))
            and bool(edge.get("provenance", {}).get("supporting_text"))
            for edge in edges
        ),
        "all_source_quotes_hash_bound": all(
            len(edge.get("provenance", {}).get("source_text_hash", "")) == 64
            for edge in edges
        ),
        "unlinked_mentions_excluded_from_edges": all(
            mention["linked_node_id"] is not None
            or all(
                edge["subject_id"] != mention["mention_id"] and edge["object_id"] != mention["mention_id"]
                for edge in edges
            )
            for mention in mentions
        ),
    }
    accepted = all(acceptance_checks.values())

    _atomic_jsonl(output_dir / "kg_entity_mentions.jsonl", mentions)
    _atomic_jsonl(output_dir / "kg_unresolved_entities.jsonl", unresolved)
    _atomic_jsonl(output_dir / "kg_nodes.jsonl", sorted(nodes.values(), key=lambda item: item["node_id"]))
    _atomic_jsonl(output_dir / "kg_edges.jsonl", sorted(edges, key=lambda item: item["edge_id"]))
    _atomic_jsonl(output_dir / "kg_events.jsonl", sorted(events, key=lambda item: item["event_id"]))
    _atomic_jsonl(output_dir / "kg_rejected_relations.jsonl", rejected)

    manifest = ExtractionManifest(
        run_id=run_id,
        input_path=str(input_path),
        input_sha256=_sha256(input_path),
        external_catalog_path=str(external_catalog_path) if external_catalog_path else None,
        external_catalog_sha256=_sha256(external_catalog_path) if external_catalog_path else None,
        external_catalog_count=external_catalog_count,
        ddi_relation_decisions_path=str(ddi_decisions_path) if ddi_decisions_path else None,
        ddi_relation_decisions_sha256=_sha256(ddi_decisions_path) if ddi_decisions_path else None,
        ddi_relation_decision_count=ddi_relation_decision_count,
        ontology_version=ONTOLOGY_VERSION,
        ner_backend=ner_backend,
        ner_model=ner_model,
        ner_threshold=ner_threshold,
        relation_models=["structured_event_extractor", "dependency_rule_relation_extractor"],
        evidence_count=len(rows),
        mention_count=len(mentions),
        linked_mention_count=sum(item["linked_node_id"] is not None for item in mentions),
        unresolved_mention_count=len(unresolved),
        node_count=len(nodes),
        edge_count=len(edges),
        event_count=len(events),
        rejected_relation_count=len(rejected),
        quality_status_counts=dict(sorted(quality_counts.items())),
    )
    _atomic_json(output_dir / "kg_extraction_manifest.json", manifest.model_dump(mode="json"))
    summary = {
        "accepted": accepted,
        "run_id": run_id,
        "ontology_version": ONTOLOGY_VERSION,
        "ner_backend": manifest.ner_backend,
        "ner_model": manifest.ner_model,
        "ner_threshold": manifest.ner_threshold,
        "input_path": str(input_path),
        "input_sha256": manifest.input_sha256,
        "external_catalog_path": manifest.external_catalog_path,
        "external_catalog_sha256": manifest.external_catalog_sha256,
        "ddi_relation_decisions_path": manifest.ddi_relation_decisions_path,
        "ddi_relation_decisions_sha256": manifest.ddi_relation_decisions_sha256,
        "ddi_relation_decision_count": manifest.ddi_relation_decision_count,
        "external_catalog_count": manifest.external_catalog_count,
        "evidence_count": len(rows),
        "mention_count": len(mentions),
        "linked_mention_count": manifest.linked_mention_count,
        "unresolved_mention_count": len(unresolved),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "event_count": len(events),
        "rejected_relation_count": len(rejected),
        "node_type_counts": dict(sorted(node_type_counts.items())),
        "edge_type_counts": dict(sorted(edge_type_counts.items())),
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "relation_method_counts": dict(sorted(relation_method_counts.items())),
        "acceptance_checks": acceptance_checks,
        "endpoint_errors": endpoint_errors,
        "event_errors": event_errors,
    }
    _atomic_json(output_dir / "kg_build_summary.json", summary)
    if not accepted:
        raise RuntimeError("Final KG acceptance checks failed; inspect kg_build_summary.json.")
    return summary


def _validate_input_partition(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("The final auto-validated evidence input is empty.")
    evidence_ids = [str(row.get("evidence_id") or "") for row in rows]
    if any(not value for value in evidence_ids):
        raise ValueError("Every evidence row requires evidence_id.")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("Duplicate evidence_id values in final input.")
    invalid = [value for value, row in zip(evidence_ids, rows) if row.get("quality_status") != "auto_validated"]
    if invalid:
        raise ValueError(f"Non-auto-validated evidence leaked into KG input: {invalid[:5]}")


def _load_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    output = {}
    for row in _iter_jsonl(path):
        output[str(row["evidence_id"])] = row
    return output


def _deduplicate(rows: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_key = str(row[key])
        current = output.get(row_key)
        if current is not None and current != row:
            raise RuntimeError(f"Conflicting duplicate {key}: {row_key}")
        output[row_key] = row
    return list(output.values())


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            yield value


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


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
