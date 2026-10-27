from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from pbpk_kb.recommendation.model_matching.training import _load_benchmark, _read_jsonl


GOLD_SCHEMA_VERSION = "pbpk-expert-gold-annotation-1.0.0"
ALLOWED_DECISIONS = {"reuse", "adapt", "reference", "reject"}


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = (json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n").encode("utf-8")
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def build_blinded_packets(
    *,
    benchmark_directory: Path,
    output_path: Path,
    candidates_per_query: int = 10,
) -> dict[str, Any]:
    queries, query_graphs, model_graphs, model_gate_graphs, labels = _load_benchmark(
        benchmark_directory
    )
    uniform_rows = _read_jsonl(
        benchmark_directory / "predictions_uniform_rule.jsonl"
    )
    score_by_pair = {
        (str(row["query_id"]), str(row["candidate_id"])): float(row["score"])
        for row in uniform_rows
    }
    packets = []
    for query in queries:
        query_id = str(query["query_id"])
        ordered = sorted(
            model_graphs,
            key=lambda model_id: (
                -score_by_pair[(query_id, model_id)],
                hashlib.sha256(f"{query_id}:{model_id}".encode()).hexdigest(),
            ),
        )
        positives = [model_id for model_id in ordered if labels[(query_id, model_id)] >= 2]
        weak = [model_id for model_id in ordered if labels[(query_id, model_id)] == 1]
        negatives = [model_id for model_id in ordered if labels[(query_id, model_id)] == 0]
        candidates = positives[: max(1, candidates_per_query - 2)]
        if weak:
            candidates.append(weak[0])
        if negatives:
            candidates.append(negatives[0])
        candidates.extend(
            model_id
            for model_id in ordered
            if model_id not in candidates
        )
        candidates = candidates[:candidates_per_query]
        for candidate_id in candidates:
            pair_id = hashlib.sha256(
                f"{GOLD_SCHEMA_VERSION}:{query_id}:{candidate_id}".encode()
            ).hexdigest()[:24]
            packets.append(
                {
                    "schema_version": GOLD_SCHEMA_VERSION,
                    "pair_id": pair_id,
                    "query_id": query_id,
                    "candidate_id": candidate_id,
                    "query_graph": query_graphs[query_id].to_dict(),
                    "candidate_graph": model_graphs[candidate_id].to_dict(),
                    "candidate_gate_graph": model_gate_graphs[candidate_id].to_dict(),
                    "annotation": {
                        "assessor_id": None,
                        "relevance_grade": None,
                        "decision": None,
                        "hard_constraint_failures": [],
                        "adaptation_operations": [],
                        "reason_codes": [],
                        "evidence_locations": [],
                        "confidence": None,
                        "comment": None,
                    },
                }
            )
    checksum = _atomic_jsonl(output_path, packets)
    return {
        "schema_version": GOLD_SCHEMA_VERSION,
        "packet_count": len(packets),
        "query_count": len(queries),
        "candidates_per_query": candidates_per_query,
        "sha256": checksum,
        "scores_and_silver_labels_blinded": True,
    }


def validate_and_adjudicate(
    *,
    annotation_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in annotation_paths:
        for row in _read_jsonl(path):
            annotation = row.get("annotation") or {}
            if annotation.get("relevance_grade") not in {0, 1, 2, 3}:
                raise ValueError(f"Invalid relevance grade in {path}.")
            if annotation.get("decision") not in ALLOWED_DECISIONS:
                raise ValueError(f"Invalid decision in {path}.")
            if not annotation.get("assessor_id"):
                raise ValueError(f"Missing assessor ID in {path}.")
            if not annotation.get("evidence_locations"):
                raise ValueError(f"Missing evidence locations in {path}.")
            grouped.setdefault(str(row["pair_id"]), []).append(row)
    adjudicated = []
    unresolved = []
    for pair_id, rows in sorted(grouped.items()):
        assessors = {row["annotation"]["assessor_id"] for row in rows}
        if len(assessors) < 2:
            unresolved.append({"pair_id": pair_id, "reason": "FEWER_THAN_TWO_ASSESSORS"})
            continue
        votes = {
            (row["annotation"]["relevance_grade"], row["annotation"]["decision"])
            for row in rows
        }
        if len(votes) != 1:
            unresolved.append({"pair_id": pair_id, "reason": "REQUIRES_THIRD_ASSESSOR_ADJUDICATION"})
            continue
        grade, decision = votes.pop()
        adjudicated.append(
            {
                "schema_version": GOLD_SCHEMA_VERSION,
                "pair_id": pair_id,
                "query_id": rows[0]["query_id"],
                "candidate_id": rows[0]["candidate_id"],
                "relevance_grade": grade,
                "decision": decision,
                "assessor_ids": sorted(assessors),
                "evidence_locations": sorted(
                    {
                        location
                        for row in rows
                        for location in row["annotation"]["evidence_locations"]
                    }
                ),
            }
        )
    checksum = _atomic_jsonl(output_path, adjudicated)
    unresolved_path = output_path.with_name("unresolved_adjudication.jsonl")
    _atomic_jsonl(unresolved_path, unresolved)
    return {
        "adjudicated_pair_count": len(adjudicated),
        "unresolved_pair_count": len(unresolved),
        "expert_gold_eligible": not unresolved and bool(adjudicated),
        "sha256": checksum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or adjudicate PBPK expert-gold packets.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--benchmark", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--candidates-per-query", type=int, default=10)
    adjudicate = subparsers.add_parser("adjudicate")
    adjudicate.add_argument("--annotations", action="append", type=Path, required=True)
    adjudicate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_blinded_packets(
            benchmark_directory=args.benchmark,
            output_path=args.output,
            candidates_per_query=args.candidates_per_query,
        )
    else:
        result = validate_and_adjudicate(
            annotation_paths=args.annotations,
            output_path=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
