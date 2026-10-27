from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.recommendation.formal_graph import FormalGraphStore, load_json, load_jsonl, sha256_file


AUDITOR_VERSION = "pbpk-recommendation-gate2-auditor-1.1.0"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _content_sha(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {key: value for key, value in row.items() if key != "content_sha256"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _unique(rows: Iterable[dict[str, Any]], key: str, errors: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            errors.append(f"missing_{key}")
        elif value in result:
            errors.append(f"duplicate_{key}:{value}")
        else:
            result[value] = row
    return result


def audit_gate2(
    *,
    graph_dir: Path,
    gate1_audit_path: Path,
    subgraph_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    store = FormalGraphStore(graph_dir, gate1_audit_path)
    manifest = load_json(subgraph_dir / "gate2_manifest.json")
    summary = load_json(subgraph_dir / "gate2_build_summary.json")
    subgraphs = {
        hop: load_jsonl(subgraph_dir / f"subgraphs_{hop}hop.jsonl")
        for hop in (0, 1, 2)
    }
    paths = load_jsonl(subgraph_dir / "traversal_paths.jsonl")

    errors: dict[str, list[str]] = {
        "binding": [],
        "subgraph": [],
        "path": [],
        "citation": [],
        "eligibility": [],
        "serialization": [],
    }
    expected_graph_hashes = store.graph_hashes()
    for name, observed in expected_graph_hashes.items():
        expected = (manifest.get("graph_hashes") or {}).get(name)
        if expected != observed:
            errors["binding"].append(f"graph_hash_mismatch:{name}")

    output_paths = {
        f"subgraphs_{hop}hop": subgraph_dir / f"subgraphs_{hop}hop.jsonl"
        for hop in (0, 1, 2)
    }
    output_paths["traversal_paths"] = subgraph_dir / "traversal_paths.jsonl"
    for name, path in output_paths.items():
        expected = (manifest.get("output_hashes") or {}).get(name)
        if expected != sha256_file(path):
            errors["binding"].append(f"output_hash_mismatch:{name}")

    path_by_id = _unique(paths, "path_id", errors["path"])
    eligible_centers = {
        str(node["node_id"])
        for node in store.eligible_centers(
            frozenset(str(value) for value in manifest.get("center_types") or [])
        )
    }
    subgraph_by_hop_center: dict[int, dict[str, dict[str, Any]]] = {}

    for hop, rows in subgraphs.items():
        by_center: dict[str, dict[str, Any]] = {}
        seen_subgraph_ids: set[str] = set()
        for row in rows:
            subgraph_id = str(row.get("subgraph_id") or "")
            center_id = str(row.get("center_id") or "")
            if not subgraph_id or subgraph_id in seen_subgraph_ids:
                errors["subgraph"].append(f"invalid_subgraph_id:{hop}:{subgraph_id}")
            seen_subgraph_ids.add(subgraph_id)
            if center_id in by_center:
                errors["subgraph"].append(f"duplicate_center:{hop}:{center_id}")
            by_center[center_id] = row
            if row.get("content_sha256") != _content_sha(row):
                errors["binding"].append(f"content_hash_mismatch:{subgraph_id}")
            if int(row.get("hop_limit", -1)) != hop:
                errors["subgraph"].append(f"hop_label_mismatch:{subgraph_id}")
            if center_id not in eligible_centers:
                errors["eligibility"].append(f"ineligible_center:{center_id}")
            center = store.node_by_id.get(center_id)
            if not center or not store.eligible_center(center):
                errors["eligibility"].append(f"name_only_or_unknown_center:{center_id}")

            node_ids = {str(node["node_id"]) for node in row.get("nodes") or []}
            edge_ids = {str(edge["edge_id"]) for edge in row.get("edges") or []}
            if center_id not in node_ids:
                errors["subgraph"].append(f"center_absent:{subgraph_id}")
            if int(row.get("node_count", -1)) != len(node_ids):
                errors["subgraph"].append(f"node_count_mismatch:{subgraph_id}")
            if int(row.get("edge_count", -1)) != len(edge_ids):
                errors["subgraph"].append(f"edge_count_mismatch:{subgraph_id}")
            if hop == 0 and (node_ids != {center_id} or edge_ids):
                errors["subgraph"].append(f"invalid_zero_hop:{subgraph_id}")

            for node in row.get("nodes") or []:
                node_id = str(node["node_id"])
                if store.node_by_id.get(node_id) != node:
                    errors["binding"].append(f"node_not_from_formal_graph:{subgraph_id}:{node_id}")
            for edge in row.get("edges") or []:
                edge_id = str(edge["edge_id"])
                source_edge = store.edge_by_id.get(edge_id)
                observed_edge = {key: value for key, value in edge.items() if key != "traversal_depth"}
                if source_edge != observed_edge:
                    errors["binding"].append(f"edge_not_from_formal_graph:{subgraph_id}:{edge_id}")
                if edge["subject_id"] not in node_ids or edge["object_id"] not in node_ids:
                    errors["subgraph"].append(f"edge_endpoint_absent:{subgraph_id}:{edge_id}")
                if int(edge.get("traversal_depth", 99)) > hop:
                    errors["subgraph"].append(f"edge_exceeds_hop:{subgraph_id}:{edge_id}")

            cited_evidence: set[tuple[str, str]] = set()
            for edge in row.get("edges") or []:
                provenance = edge.get("provenance") or {}
                if provenance.get("evidence_id") and provenance.get("source_text_hash"):
                    cited_evidence.add(
                        (str(provenance["evidence_id"]), str(provenance["source_text_hash"]))
                    )
            row_citations = {
                (str(item.get("evidence_id") or ""), str(item.get("source_text_hash") or ""))
                for item in row.get("citations") or []
            }
            if row_citations != cited_evidence:
                errors["citation"].append(f"citation_edge_mismatch:{subgraph_id}")
            for citation in row.get("citations") or []:
                evidence_id = str(citation.get("evidence_id") or "")
                evidence_node = store.node_by_id.get(evidence_id)
                if not evidence_node or evidence_node.get("node_type") != "Evidence":
                    errors["citation"].append(f"citation_evidence_absent:{subgraph_id}:{evidence_id}")
                elif (
                    (evidence_node.get("properties") or {}).get("source_text_hash")
                    != citation.get("source_text_hash")
                ):
                    errors["citation"].append(f"citation_hash_mismatch:{subgraph_id}:{evidence_id}")

            embedding_text = str(row.get("embedding_text") or "")
            if not embedding_text.startswith("CENTER ["):
                errors["serialization"].append(f"center_missing:{subgraph_id}")
            if "ONTOLOGY VERSION " not in embedding_text:
                errors["serialization"].append(f"ontology_version_missing:{subgraph_id}")
            if "SOURCE FAMILIES " not in embedding_text:
                errors["serialization"].append(f"source_summary_missing:{subgraph_id}")
            if row.get("edges") and "RELATION TYPES " not in embedding_text:
                errors["serialization"].append(f"relation_summary_missing:{subgraph_id}")
            if row.get("edges") and "\nEDGE " not in embedding_text:
                errors["serialization"].append(f"edge_line_missing:{subgraph_id}")
            if row.get("events") and "\nEVENT " not in embedding_text:
                errors["serialization"].append(f"event_line_missing:{subgraph_id}")
            if row.get("citations") and "\nCITATION " not in embedding_text:
                errors["serialization"].append(f"citation_line_missing:{subgraph_id}")
            relation_definitions = (
                (row.get("ontology_semantics") or {}).get("relation_definitions") or {}
            )
            for predicate in sorted(
                {str(edge["predicate"]) for edge in row.get("edges") or []}
            ):
                if not relation_definitions.get(predicate):
                    errors["serialization"].append(
                        f"structured_relation_gloss_missing:{subgraph_id}:{predicate}"
                    )
                if f"RELATION GLOSS {predicate}=" not in embedding_text:
                    errors["serialization"].append(
                        f"text_relation_gloss_missing:{subgraph_id}:{predicate}"
                    )

            for path_id in row.get("path_ids") or []:
                path = path_by_id.get(str(path_id))
                if not path:
                    errors["path"].append(f"path_absent:{subgraph_id}:{path_id}")
                elif int(path.get("hop", 99)) > hop:
                    errors["path"].append(f"path_exceeds_hop:{subgraph_id}:{path_id}")
        subgraph_by_hop_center[hop] = by_center

    for center_id in sorted(eligible_centers):
        for hop in (0, 1, 2):
            if center_id not in subgraph_by_hop_center.get(hop, {}):
                errors["subgraph"].append(f"missing_center_subgraph:{hop}:{center_id}")
        if not all(center_id in subgraph_by_hop_center.get(hop, {}) for hop in (0, 1, 2)):
            continue
        zero = subgraph_by_hop_center[0][center_id]
        one = subgraph_by_hop_center[1][center_id]
        two = subgraph_by_hop_center[2][center_id]
        node_sets = [
            {str(node["node_id"]) for node in value["nodes"]}
            for value in (zero, one, two)
        ]
        edge_sets = [
            {str(edge["edge_id"]) for edge in value["edges"]}
            for value in (zero, one, two)
        ]
        if not node_sets[0] <= node_sets[1] <= node_sets[2]:
            errors["subgraph"].append(f"node_hop_not_monotonic:{center_id}")
        if not edge_sets[0] <= edge_sets[1] <= edge_sets[2]:
            errors["subgraph"].append(f"edge_hop_not_monotonic:{center_id}")

    for path in paths:
        path_id = str(path.get("path_id") or "")
        center_id = str(path.get("center_id") or "")
        target_id = str(path.get("target_id") or "")
        node_ids = [str(value) for value in path.get("node_ids") or []]
        edge_ids = [str(value) for value in path.get("edge_ids") or []]
        directions = [str(value) for value in path.get("directions") or []]
        if (
            not node_ids
            or node_ids[0] != center_id
            or node_ids[-1] != target_id
            or len(edge_ids) != len(node_ids) - 1
            or len(directions) != len(edge_ids)
            or int(path.get("hop", -1)) != len(edge_ids)
            or len(edge_ids) > 2
        ):
            errors["path"].append(f"invalid_path_shape:{path_id}")
            continue
        for index, (edge_id, direction) in enumerate(zip(edge_ids, directions, strict=True)):
            edge = store.edge_by_id.get(edge_id)
            if not edge:
                errors["path"].append(f"path_edge_absent:{path_id}:{edge_id}")
                continue
            expected = (
                (edge["subject_id"], edge["object_id"])
                if direction == "out"
                else (edge["object_id"], edge["subject_id"])
            )
            if expected != (node_ids[index], node_ids[index + 1]):
                errors["path"].append(f"path_edge_direction_invalid:{path_id}:{edge_id}")

    checks = {
        "gate1_binding_valid": not errors["binding"],
        "all_eligible_centers_indexed": not any(
            value.startswith("missing_center_subgraph") for value in errors["subgraph"]
        ),
        "subgraph_content_matches_formal_graph": not any(
            "not_from_formal_graph" in value for value in errors["binding"]
        ),
        "zero_one_two_hop_are_true_and_monotonic": not any(
            value.startswith(("invalid_zero_hop", "edge_exceeds_hop", "node_hop_not_monotonic", "edge_hop_not_monotonic"))
            for value in errors["subgraph"]
        ),
        "all_paths_reconstruct_from_adjacency": not errors["path"],
        "all_citations_trace_to_evidence": not errors["citation"],
        "name_only_and_nonreusable_models_excluded": not errors["eligibility"],
        "output_hashes_valid": not any("hash_mismatch" in value for value in errors["binding"]),
        "canonical_serialization_complete": not errors["serialization"],
    }
    accepted = all(checks.values()) and not any(errors.values())
    report = {
        "accepted": accepted,
        "auditor_version": AUDITOR_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "counts": {
            "eligible_centers": len(eligible_centers),
            "subgraphs_0hop": len(subgraphs[0]),
            "subgraphs_1hop": len(subgraphs[1]),
            "subgraphs_2hop": len(subgraphs[2]),
            "paths": len(paths),
            "center_type_counts": dict(
                sorted(
                    Counter(
                        str(store.node_by_id[center_id]["node_type"])
                        for center_id in eligible_centers
                    ).items()
                )
            ),
        },
        "errors": errors,
        "inputs": {
            "graph_dir": str(graph_dir),
            "gate1_audit_path": str(gate1_audit_path),
            "subgraph_dir": str(subgraph_dir),
            "manifest_sha256": sha256_file(subgraph_dir / "gate2_manifest.json"),
            "summary_sha256": sha256_file(subgraph_dir / "gate2_build_summary.json"),
        },
        "limitations": [
            "Gate 2 validates graph traversal integrity and provenance, not semantic retrieval quality.",
            "Expansion caps are deterministic and audited; truncated neighborhoods remain discoverable through the manifest counts.",
            "Only reusable repository PBPK models are eligible model centers; literature mentions remain evidence nodes in traversed subgraphs.",
        ],
        "build_summary": summary,
    }
    _atomic_json(output_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit formal PBPK KG subgraphs and traversal paths.")
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--gate1-audit", type=Path, required=True)
    parser.add_argument("--subgraph-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_gate2(
        graph_dir=args.graph_dir,
        gate1_audit_path=args.gate1_audit,
        subgraph_dir=args.subgraph_dir,
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
