from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from pbpk_kb.recommendation.model_matching.benchmark import _atomic_json, _atomic_jsonl
from pbpk_kb.recommendation.model_matching.metrics import evaluate_rankings
from pbpk_kb.recommendation.model_matching.scoring import (
    assess_candidate_use,
    score_candidate,
)
from pbpk_kb.recommendation.model_matching.training import (
    _load_benchmark,
    load_benchmark_metadata,
)


MOLECULAR_BASELINE_VERSION = "pbpk-molecular-domain-baseline-1.2.0"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize(value: object) -> str:
    return " ".join(str(value).casefold().split())


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _compound_set_similarity(
    *,
    query_compounds: list[str],
    candidate_compounds: list[str],
    representations: Mapping[str, Any],
    similarity: Any,
) -> tuple[float, bool]:
    if (
        not query_compounds
        or not candidate_compounds
        or any(value not in representations for value in query_compounds)
        or any(value not in representations for value in candidate_compounds)
    ):
        return 0.0, False
    score = sum(
        max(
            float(
                similarity(
                    representations[query_compound],
                    representations[candidate_compound],
                )
            )
            for candidate_compound in candidate_compounds
        )
        for query_compound in query_compounds
    ) / len(query_compounds)
    return score, True


def run_molecular_baseline(
    *,
    benchmark_directory: Path,
    structures_path: Path,
    output_directory: Path,
    pretrained_embeddings_path: Path | None = None,
) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.DataStructs import TanimotoSimilarity

    queries, query_graphs, model_graphs, model_gate_graphs, labels = _load_benchmark(
        benchmark_directory
    )
    benchmark_metadata = load_benchmark_metadata(benchmark_directory)
    structures = {
        _normalize(row["compound"]): str(row["canonical_smiles"])
        for row in _read_jsonl(structures_path)
        if row.get("compound") and row.get("canonical_smiles")
    }
    embedding_rows = _read_jsonl(pretrained_embeddings_path) if pretrained_embeddings_path else []
    embedding_methods = {
        str(row.get("embedding_method"))
        for row in embedding_rows
        if row.get("embedding_method")
    }
    if len(embedding_methods) > 1:
        raise ValueError("Pretrained embedding file mixes multiple embedding methods")
    embedding_method = next(iter(embedding_methods), "pretrained_molecular_embedding")
    embeddings = {
        _normalize(row["compound"]): [float(value) for value in row["embedding"]]
        for row in embedding_rows
    }
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = {}
    for compound, smiles in structures.items():
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            fingerprints[compound] = generator.GetFingerprint(molecule)

    methods = ["morgan_tanimoto"] + ([embedding_method] if embeddings else [])
    test_query_ids = {
        str(query["query_id"])
        for query in queries
        if query["split"] == "test"
    }
    output_directory.mkdir(parents=True, exist_ok=False)
    results = {}
    for method in methods:
        rows = []
        covered = 0
        test_covered = 0
        test_pair_count = 0
        for query in queries:
            query_id = str(query["query_id"])
            query_compounds = sorted(
                {
                    _normalize(value)
                    for value in query.get("required_model_compounds")
                    or query.get("active_moieties")
                    or []
                }
            )
            for candidate_id, graph in sorted(model_graphs.items()):
                deterministic = score_candidate(
                    query_graphs[query_id], model_gate_graphs[candidate_id]
                )
                assessment = assess_candidate_use(
                    query_graphs[query_id],
                    model_gate_graphs[candidate_id],
                    match=deterministic,
                )
                candidate_compounds = sorted(
                    {
                        _normalize(node.label)
                        for node in graph.nodes
                        if node.node_type == "Compound"
                    }
                )
                if method == "morgan_tanimoto":
                    score, available = _compound_set_similarity(
                        query_compounds=query_compounds,
                        candidate_compounds=candidate_compounds,
                        representations=fingerprints,
                        similarity=TanimotoSimilarity,
                    )
                else:
                    score, available = _compound_set_similarity(
                        query_compounds=query_compounds,
                        candidate_compounds=candidate_compounds,
                        representations=embeddings,
                        similarity=_cosine,
                    )
                covered += int(available)
                if query["split"] == "test":
                    test_covered += int(available)
                    test_pair_count += 1
                rows.append(
                    {
                        "query_id": query_id,
                        "candidate_id": candidate_id,
                        "score": score,
                        "relevance_grade": labels[(query_id, candidate_id)],
                        "hard_gate_passed": (
                            assessment.direct_reuse_gate_passed
                        ),
                        "structural_gate_passed": (
                            assessment.structural_gate_passed
                        ),
                        "maximum_use_tier": assessment.maximum_use_tier,
                        "predicted_use_tier": (
                            "reject"
                            if assessment.maximum_use_tier == "reject"
                            else "mechanistic_reference"
                        ),
                        "reuse_decision": False,
                        "adaptation_decision": False,
                        "feature_available": available,
                    }
                )
        _atomic_jsonl(output_directory / f"predictions_{method}.jsonl", rows)
        metrics = evaluate_rankings(rows)
        _atomic_json(output_directory / f"metrics_{method}.json", metrics)
        test_metrics = evaluate_rankings(
            row
            for row in rows
            if str(row["query_id"]) in test_query_ids
        )
        _atomic_json(
            output_directory / f"metrics_{method}_test.json",
            test_metrics,
        )
        results[method] = {
            "coverage": covered / len(rows) if rows else 0.0,
            "metrics": metrics["aggregate"],
            "test_coverage": (
                test_covered / test_pair_count if test_pair_count else 0.0
            ),
            "test_metrics": test_metrics["aggregate"],
        }
    manifest = {
        "version": MOLECULAR_BASELINE_VERSION,
        **benchmark_metadata,
        "role": "compound_domain_auxiliary_feature_only",
        "structures_path": str(structures_path),
        "pretrained_embeddings_path": str(pretrained_embeddings_path) if pretrained_embeddings_path else None,
        "pretrained_embedding_method": embedding_method if embeddings else None,
        "results": results,
    }
    _atomic_json(output_directory / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run molecular-domain PBPK baselines.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--pretrained-embeddings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_molecular_baseline(
        benchmark_directory=args.benchmark,
        structures_path=args.structures,
        pretrained_embeddings_path=args.pretrained_embeddings,
        output_directory=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
