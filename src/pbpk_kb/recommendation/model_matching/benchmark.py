from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from pbpk_kb.recommendation.model_matching.graphs import (
    TypedGraph,
    build_model_graph,
    requirement_graph_from_model,
)
from pbpk_kb.recommendation.model_matching.metrics import evaluate_rankings
from pbpk_kb.recommendation.model_matching.model_kg_graphs import (
    condense_model_graph,
    load_model_graphs,
)
from pbpk_kb.recommendation.model_matching.scoring import (
    assess_candidate_use,
    bm25_scores,
    cross_graph_attention_score,
    domain_ged_score,
    hierarchical_match_score,
    score_candidate,
)


BENCHMARK_VERSION = "pbpk-candidate-retrieval-sanity-benchmark-2.0.0"
EXPERIMENT_VERSION = "pbpk-model-matching-experiment-1.2.0"

IMPLEMENTED_METHODS = (
    "hard_gate_only",
    "uniform_rule",
    "bm25",
    "domain_ged",
    "typed_cross_attention",
    "hierarchical_matching",
    "cg_hhgm_deterministic_fallback",
)

GPU_METHODS = (
    "morgan_tanimoto",
    "molclr",
    "molecular_graph_transformer",
    "equivariant_3d",
    "greed",
    "gedgnn",
    "mata_star",
    "gmn",
    "cgsim",
    "cosimgnn",
    "h2mn",
    "segmn",
    "hgt_dual_encoder",
    "lambdamart",
    "cg_hhgm",
)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
        default=str,
    ).encode("utf-8") + b"\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = _canonical(row)
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}.")
            yield value


def load_model_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _iter_jsonl(path):
            model_version_id = str(row.get("model_version_id") or "")
            if not model_version_id:
                raise ValueError(f"Missing model_version_id in {path}.")
            records[model_version_id] = row
    return sorted(
        records.values(),
        key=lambda row: (
            str(row.get("model_family") or "").casefold(),
            str(row["model_version_id"]),
        ),
    )


def _normalized(values: object) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    return {" ".join(str(value).casefold().split()) for value in values if str(value).strip()}


def relevance_grade(query: Mapping[str, Any], candidate: Mapping[str, Any]) -> int:
    if query["model_version_id"] == candidate["model_version_id"]:
        return 3
    same_family = (
        str(query.get("model_family") or "").casefold()
        == str(candidate.get("model_family") or "").casefold()
    )
    species_overlap = bool(_normalized(query.get("species")) & _normalized(candidate.get("species")))
    route_overlap = bool(_normalized(query.get("routes")) & _normalized(candidate.get("routes")))
    if same_family:
        return 2 if species_overlap and route_overlap else 1
    query_mechanisms = set().union(
        _normalized(query.get("enzymes")),
        _normalized(query.get("transporters")),
        _normalized(query.get("process_types")),
    )
    candidate_mechanisms = set().union(
        _normalized(candidate.get("enzymes")),
        _normalized(candidate.get("transporters")),
        _normalized(candidate.get("process_types")),
    )
    shared = query_mechanisms & candidate_mechanisms
    return int(bool(shared) and species_overlap and route_overlap)


def _split(model_family: str) -> str:
    bucket = int(hashlib.sha256(model_family.casefold().encode("utf-8")).hexdigest()[:8], 16) % 10
    return "train" if bucket < 6 else "validation" if bucket < 8 else "test"


def build_benchmark(
    model_records: list[dict[str, Any]],
    *,
    full_model_kg: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, TypedGraph], dict[str, TypedGraph]]:
    if full_model_kg is not None:
        full_graphs = load_model_graphs(
            sqlite_path=full_model_kg,
            model_version_ids=[str(row["model_version_id"]) for row in model_records],
        )
        model_graphs = {
            model_id: condense_model_graph(graph)
            for model_id, graph in full_graphs.items()
        }
    else:
        model_graphs = {
            str(row["model_version_id"]): build_model_graph(row)
            for row in model_records
        }
    query_graphs: dict[str, TypedGraph] = {}
    queries: list[dict[str, Any]] = []
    for row in model_records:
        model_id = str(row["model_version_id"])
        for task, include_identity in (
            ("exact_model_recovery", True),
            ("mechanistic_template_recovery", False),
        ):
            query_id = "query:" + hashlib.sha256(
                f"{BENCHMARK_VERSION}:{task}:{model_id}".encode("utf-8")
            ).hexdigest()[:24]
            graph = requirement_graph_from_model(
                row,
                query_id=query_id,
                include_identity=include_identity,
            )
            query_graphs[query_id] = graph
            queries.append(
                {
                    "query_id": query_id,
                    "task": task,
                    "benchmark_level": "candidate_retrieval_sanity",
                    "source_model_version_id": model_id,
                    "source_model_family": row.get("model_family"),
                    "split": _split(str(row.get("model_family") or "")),
                    "identity_included": include_identity,
                    "query_graph_checksum": graph.checksum,
                    "label_policy": (
                        "source-version=3;same-family-compatible=2;"
                        "shared-mechanism-context=1;otherwise=0"
                    ),
                }
            )
    return queries, query_graphs, model_graphs


def _method_rows(
    *,
    method: str,
    queries: list[dict[str, Any]],
    query_graphs: Mapping[str, TypedGraph],
    model_records: list[dict[str, Any]],
    model_graphs: Mapping[str, TypedGraph],
    model_gate_graphs: Mapping[str, TypedGraph],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered_model_graphs = [model_graphs[str(row["model_version_id"])] for row in model_records]
    for query in queries:
        query_graph = query_graphs[query["query_id"]]
        bm25 = bm25_scores(query_graph, ordered_model_graphs) if method == "bm25" else None
        source = next(
            row for row in model_records
            if row["model_version_id"] == query["source_model_version_id"]
        )
        for index, candidate in enumerate(model_records):
            candidate_id = str(candidate["model_version_id"])
            graph = model_graphs[candidate_id]
            match = score_candidate(query_graph, graph)
            # Full Model KG graphs contain provenance, simulation, and validation
            # nodes that are useful for ranking but are not request constraints.
            # Evaluate biological hard gates against the canonical model card.
            gate_match = score_candidate(query_graph, model_gate_graphs[candidate_id])
            assessment = assess_candidate_use(
                query_graph,
                model_gate_graphs[candidate_id],
                match=gate_match,
            )
            if method == "hard_gate_only":
                score = float(assessment.direct_reuse_gate_passed)
            elif method == "uniform_rule":
                score = sum(match.hierarchy_scores.values()) / len(match.hierarchy_scores)
            elif method == "bm25":
                score = float(bm25[index])
            elif method == "domain_ged":
                score = domain_ged_score(query_graph, graph)[0]
            elif method == "typed_cross_attention":
                score = cross_graph_attention_score(query_graph, graph)
            elif method == "hierarchical_matching":
                score = hierarchical_match_score(query_graph, graph)[0]
            elif method == "cg_hhgm_deterministic_fallback":
                score = match.score
            else:
                raise ValueError(f"Unsupported implemented method: {method}")
            reuse_decision = bool(
                query["identity_included"]
                and assessment.direct_reuse_gate_passed
                and score >= 0.90
            )
            rows.append(
                {
                    "query_id": query["query_id"],
                    "query_task": query["task"],
                    "query_split": query["split"],
                    "candidate_id": candidate_id,
                    "candidate_family": candidate.get("model_family"),
                    "method": method,
                    "score": score,
                    "relevance_grade": relevance_grade(source, candidate),
                    "raw_score": score,
                    "hard_gate_passed": (
                        assessment.direct_reuse_gate_passed
                    ),
                    "structural_gate_passed": (
                        assessment.structural_gate_passed
                    ),
                    "maximum_use_tier": assessment.maximum_use_tier,
                    "predicted_use_tier": (
                        "direct_reuse"
                        if reuse_decision
                        else "reject"
                        if assessment.maximum_use_tier == "reject"
                        else "mechanistic_reference"
                    ),
                    "reuse_decision": reuse_decision,
                    "adaptation_decision": False,
                    "adaptation_cost": match.adaptation_cost,
                    "uncertainty": match.uncertainty,
                }
            )
    return rows


def run_experiment(
    *,
    model_paths: list[Path],
    output_directory: Path,
    full_model_kg: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_directory.mkdir(parents=True, exist_ok=False)
    records = load_model_records(model_paths)
    model_gate_graphs = {
        str(row["model_version_id"]): build_model_graph(row)
        for row in records
    }
    queries, query_graphs, model_graphs = build_benchmark(
        records,
        full_model_kg=full_model_kg,
    )
    checksums: dict[str, str] = {}
    checksums["queries.jsonl"] = _atomic_jsonl(output_directory / "queries.jsonl", queries)
    checksums["query_graphs.jsonl"] = _atomic_jsonl(
        output_directory / "query_graphs.jsonl",
        (query_graphs[query["query_id"]].to_dict() for query in queries),
    )
    checksums["model_graphs.jsonl"] = _atomic_jsonl(
        output_directory / "model_graphs.jsonl",
        (model_graphs[str(row["model_version_id"])].to_dict() for row in records),
    )
    checksums["model_gate_graphs.jsonl"] = _atomic_jsonl(
        output_directory / "model_gate_graphs.jsonl",
        (
            model_gate_graphs[str(row["model_version_id"])].to_dict()
            for row in records
        ),
    )
    label_rows = [
        {
            "query_id": query["query_id"],
            "query_task": query["task"],
            "query_split": query["split"],
            "candidate_id": str(candidate["model_version_id"]),
            "candidate_family": candidate.get("model_family"),
            "relevance_grade": relevance_grade(
                next(
                    record
                    for record in records
                    if record["model_version_id"]
                    == query["source_model_version_id"]
                ),
                candidate,
            ),
        }
        for query in queries
        for candidate in records
    ]
    checksums["labels.jsonl"] = _atomic_jsonl(
        output_directory / "labels.jsonl",
        label_rows,
    )
    comparison: dict[str, Any] = {}
    test_comparison: dict[str, Any] = {}
    for method in IMPLEMENTED_METHODS:
        method_rows = _method_rows(
            method=method,
            queries=queries,
            query_graphs=query_graphs,
            model_records=records,
            model_graphs=model_graphs,
            model_gate_graphs=model_gate_graphs,
        )
        result_name = f"predictions_{method}.jsonl"
        metric_name = f"metrics_{method}.json"
        checksums[result_name] = _atomic_jsonl(output_directory / result_name, method_rows)
        metrics = evaluate_rankings(method_rows)
        checksums[metric_name] = _atomic_json(output_directory / metric_name, metrics)
        comparison[method] = metrics["aggregate"]
        test_metrics = evaluate_rankings(
            row for row in method_rows if row["query_split"] == "test"
        )
        test_metric_name = f"metrics_{method}_test.json"
        checksums[test_metric_name] = _atomic_json(
            output_directory / test_metric_name,
            test_metrics,
        )
        test_comparison[method] = test_metrics["aggregate"]

    split_counts = Counter(query["split"] for query in queries)
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_level": "candidate_retrieval_sanity",
        "evaluation_scope": "candidate_retrieval_and_representation_sanity_only",
        "scientific_claim_eligible": False,
        "scientific_claim_blockers": [
            "NOT_EXECUTION_DERIVED",
            "LABELS_DERIVED_FROM_CANDIDATE_METADATA",
            "ONLY_ONE_GRADE3_REUSE_POSITIVE_IN_CATALOG",
            "NO_INDEPENDENT_ADAPTATION_OUTCOMES",
        ],
        "split_policy": (
            "SHA256 model-family grouped 60/20/20 split; a model family is "
            "assigned to exactly one query split. Neural training must also "
            "exclude validation/test candidate families from training pairs."
        ),
        "model_graph_source": (
            "full_model_kg_sqlite_semantic_condensation"
            if full_model_kg is not None
            else "compact_model_card_graph"
        ),
        "hard_gate_graph_source": "canonical_model_card_graph",
        "full_model_kg_path": str(full_model_kg) if full_model_kg else None,
        "model_count": len(records),
        "model_family_count": len({str(row.get("model_family")) for row in records}),
        "query_count": len(queries),
        "pair_count_per_method": len(queries) * len(records),
        "split_query_counts": dict(sorted(split_counts.items())),
        "implemented_methods": list(IMPLEMENTED_METHODS),
        "gpu_or_external_methods": {
            method: {
                "status": "not_executed",
                "reason": (
                    "Requires the locked GPU environment and method-specific "
                    "features/checkpoints; no local surrogate is reported under "
                    "the paper method name."
                ),
            }
            for method in GPU_METHODS
        },
        "comparison": comparison,
        "test_comparison": test_comparison,
        "input_files": [str(path) for path in model_paths],
        "output_checksums": checksums,
        "elapsed_seconds": time.perf_counter() - started,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(output_directory / "experiment_manifest.json", manifest)
    return manifest


def _default_output() -> Path:
    stamp = datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    return Path("data_process/model_matching") / stamp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the PBPK candidate-retrieval sanity benchmark."
    )
    parser.add_argument("--model-versions", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--full-model-kg", type=Path)
    args = parser.parse_args()
    manifest = run_experiment(
        model_paths=args.model_versions,
        output_directory=args.output or _default_output(),
        full_model_kg=args.full_model_kg,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
