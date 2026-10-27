from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from pbpk_kb.recommendation.model_matching.benchmark import (
    IMPLEMENTED_METHODS,
    _atomic_json,
    _atomic_jsonl,
)
from pbpk_kb.recommendation.model_matching.execution_benchmark import (
    BENCHMARK_VERSION,
    _iter_jsonl,
)
from pbpk_kb.recommendation.model_matching.graphs import (
    GRAPH_SCHEMA_VERSION,
    TypedGraph,
    build_execution_requirement_graph,
    build_model_graph,
)
from pbpk_kb.recommendation.model_matching.metrics import evaluate_rankings
from pbpk_kb.recommendation.model_matching.perturbation_evaluation import (
    evaluate_ranking_continuity,
)
from pbpk_kb.recommendation.model_matching.scoring import (
    assess_candidate_use,
    bm25_scores,
    cross_graph_attention_score,
    domain_ged_score,
    hierarchical_match_score,
    score_candidate,
)


EXPERIMENT_VERSION = "pbpk-execution-derived-ranking-experiment-1.3.0"
DETERMINISTIC_ADAPTATION_COST_SCALE = 8.0


def _normalized(values: object) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    return {
        " ".join(str(value).casefold().split())
        for value in values
        if str(value).strip()
    }


def _structure_index(
    structures_path: Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if structures_path is None:
        return {}, {
            "status": "not_supplied",
            "source_path": None,
            "source_sha256": None,
        }
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    rows = list(_iter_jsonl(structures_path))
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        compound = " ".join(str(row.get("compound") or "").casefold().split())
        isomeric_smiles = str(row.get("isomeric_smiles") or "").strip()
        connectivity_smiles = str(row.get("canonical_smiles") or "").strip()
        smiles = isomeric_smiles or connectivity_smiles
        if not compound or not smiles:
            continue
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"RDKit could not parse canonical SMILES for {compound}.")
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        value = {
            "canonical_smiles": canonical,
            "structure_representation": (
                "canonicalized_isomeric_smiles"
                if isomeric_smiles
                else "canonicalized_connectivity_smiles_fallback"
            ),
            "source_isomeric_smiles": isomeric_smiles or None,
            "source_connectivity_smiles": connectivity_smiles or None,
            "inchi_key": row.get("inchi_key"),
            "molecular_descriptors": {
                "molecular_weight": round(float(Descriptors.MolWt(molecule)), 6),
                "tpsa": round(float(rdMolDescriptors.CalcTPSA(molecule)), 6),
                "xlogp": round(float(Crippen.MolLogP(molecule)), 6),
                "hbd": int(Lipinski.NumHDonors(molecule)),
                "hba": int(Lipinski.NumHAcceptors(molecule)),
                "rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule)),
            },
            "structure_source_id": row.get("source_id"),
            "structure_source_url": row.get("source_url"),
            "structure_response_sha256": row.get("response_sha256"),
        }
        existing = index.get(compound)
        if existing is not None and existing["canonical_smiles"] != canonical:
            raise ValueError(f"Conflicting structures for {compound}.")
        index[compound] = value
    digest = hashlib.sha256(structures_path.read_bytes()).hexdigest()
    return index, {
        "status": "loaded",
        "source_path": str(structures_path.resolve()),
        "source_sha256": digest,
        "resolved_compound_count": len(index),
    }


def _structure_payload(
    *,
    compounds: object,
    structures: Mapping[str, dict[str, Any]],
    missing: set[str],
) -> dict[str, Any]:
    normalized = sorted(_normalized(compounds))
    rows = []
    for compound in normalized:
        structure = structures.get(compound)
        if structure is None:
            missing.add(compound)
            continue
        rows.append({"compound": compound, **structure})
    payload: dict[str, Any] = {
        "compound_structures": rows,
        "canonical_smiles": [row["canonical_smiles"] for row in rows],
    }
    if len(rows) == 1:
        payload["molecular_descriptors"] = rows[0]["molecular_descriptors"]
    return payload


def _score_rows(
    *,
    method: str,
    queries: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    query_graphs: Mapping[str, TypedGraph],
    model_graphs: Mapping[str, TypedGraph],
    labels: Mapping[tuple[str, str], dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    ordered_graphs = [model_graphs[str(row["candidate_id"])] for row in candidates]
    rows = []
    for query in queries:
        query_id = str(query["query_id"])
        query_graph = query_graphs[query_id]
        bm25 = bm25_scores(query_graph, ordered_graphs) if method == "bm25" else None
        query_compounds = _normalized(
            query.get("required_model_compounds")
            or query["active_moieties"]
        )
        for index, candidate in enumerate(candidates):
            candidate_id = str(candidate["candidate_id"])
            graph = model_graphs[candidate_id]
            match = score_candidate(query_graph, graph)
            assessment = assess_candidate_use(
                query_graph,
                graph,
                match=match,
            )
            if method == "hard_gate_only":
                score = float(assessment.direct_reuse_gate_passed)
            elif method == "uniform_rule":
                score = sum(match.hierarchy_scores.values()) / len(match.hierarchy_scores)
            elif method == "bm25":
                assert bm25 is not None
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
                raise ValueError(f"Unsupported method: {method}")
            label = labels.get((query_id, candidate_id)) if labels is not None else None
            same_active_moiety = query_compounds.issubset(
                _normalized(candidate.get("active_moieties"))
            )
            reuse_decision = bool(
                same_active_moiety
                and assessment.direct_reuse_gate_passed
                and score >= 0.90
            )
            predicted_use_tier = (
                "direct_reuse"
                if reuse_decision
                else "reject"
                if assessment.maximum_use_tier == "reject"
                else "mechanistic_reference"
            )
            rows.append(
                {
                    "query_id": query_id,
                    "query_task": query["task"],
                    "query_split": query["split"],
                    "candidate_id": candidate_id,
                    "candidate_family": candidate.get("model_family"),
                    "method": method,
                    "score": score,
                    "relevance_grade": (
                        int(label["relevance_grade"]) if label is not None else None
                    ),
                    "raw_score": score,
                    "hard_gate_passed": (
                        assessment.direct_reuse_gate_passed
                    ),
                    "structural_gate_passed": (
                        assessment.structural_gate_passed
                    ),
                    "direct_reuse_gate_failures": list(
                        assessment.direct_reuse_gate_failures
                    ),
                    "maximum_use_tier": assessment.maximum_use_tier,
                    "predicted_use_tier": predicted_use_tier,
                    "reuse_decision": reuse_decision,
                    "adaptation_decision": False,
                    "adaptation_cost": match.adaptation_cost,
                    "predicted_adaptation_cost": (
                        match.adaptation_cost
                        * DETERMINISTIC_ADAPTATION_COST_SCALE
                    ),
                    "adaptation_cost_label": (
                        label.get("adaptation_cost_label") if label is not None else None
                    ),
                    "adaptation_cost_source": (
                        "deterministic_pbpk_graph_edit_cost_scaled"
                    ),
                    "adaptation_operation_labels": (
                        label.get("adaptation_operation_multilabel")
                        if label is not None and label.get("adaptation_executed")
                        else None
                    ),
                    "predicted_adaptation_operations": (
                        []
                        if label is not None and label.get("adaptation_executed")
                        else None
                    ),
                    "uncertainty": match.uncertainty,
                    "scenario_scientific_passed": (
                        label.get("scenario_scientific_passed")
                        if label is not None
                        else None
                    ),
                }
            )
    return rows


def run_execution_experiment(
    *,
    benchmark_directory: Path,
    labels_directory: Path,
    output_directory: Path,
    structures_path: Path | None = None,
    perturbed_tasks_path: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    tasks = list(_iter_jsonl(benchmark_directory / "benchmark_tasks.jsonl"))
    candidates = list(_iter_jsonl(benchmark_directory / "candidates.jsonl"))
    label_rows = list(_iter_jsonl(labels_directory / "labels.jsonl"))
    benchmark_manifest = json.loads(
        (benchmark_directory / "manifest.json").read_text(encoding="utf-8")
    )
    leakage_path = benchmark_directory / "leakage_audit.json"
    expected_leakage_sha256 = str(
        (benchmark_manifest.get("output_checksums") or {}).get(
            "leakage_audit.json"
        )
        or ""
    )
    actual_leakage_sha256 = (
        hashlib.sha256(leakage_path.read_bytes()).hexdigest()
        if leakage_path.is_file()
        else ""
    )
    if not expected_leakage_sha256 or actual_leakage_sha256 != expected_leakage_sha256:
        raise ValueError("Benchmark leakage audit is missing or its checksum drifted.")
    leakage_audit = json.loads(leakage_path.read_text(encoding="utf-8"))
    if not leakage_audit.get("passed"):
        raise ValueError("Benchmark leakage audit did not pass.")
    candidate_split_rows = list(leakage_audit.get("candidate_assignments") or [])
    candidate_split_by_id = {
        str(row["candidate_id"]): str(row["split"])
        for row in candidate_split_rows
    }
    expected_candidate_ids = {str(row["candidate_id"]) for row in candidates}
    if set(candidate_split_by_id) != expected_candidate_ids:
        raise ValueError("Leakage audit does not cover the frozen candidate inventory.")
    task_split_by_id = {
        str(row["task_id"]): str(row["split"])
        for row in leakage_audit.get("task_assignments") or []
    }
    if task_split_by_id != {
        str(row["task_id"]): str(row["split"]) for row in tasks
    }:
        raise ValueError("Leakage-audit task splits differ from benchmark tasks.")
    structures, structure_manifest = _structure_index(structures_path)
    group_labels = {
        (str(row["validation_group_id"]), str(row["candidate_id"])): row
        for row in _iter_jsonl(labels_directory / "candidate_group_labels.jsonl")
    }
    labels: dict[tuple[str, str], dict[str, Any]] = {}
    task_by_id = {str(row["task_id"]): row for row in tasks}
    for row in label_rows:
        task = task_by_id[str(row["task_id"])]
        group = group_labels[(str(task["validation_group_id"]), str(row["candidate_id"]))]
        labels[(str(row["task_id"]), str(row["candidate_id"]))] = {
            **row,
            "adaptation_cost_label": (
                float(row["adaptation_weighted_cost"])
                if row.get("adaptation_executed")
                and row.get("adaptation_weighted_cost") is not None
                else None
            ),
            "adaptation_group_weighted_cost": float(
                group.get("adaptation_weighted_cost") or 0.0
            ),
        }
    expected_pairs = {
        (str(task["task_id"]), str(candidate["candidate_id"]))
        for task in tasks
        for candidate in candidates
    }
    missing = expected_pairs - set(labels)
    extra = set(labels) - expected_pairs
    if missing or extra:
        raise ValueError(
            f"Execution-derived labels are incomplete: missing={len(missing)}, extra={len(extra)}"
        )
    queries = []
    enriched_tasks = []
    missing_query_structures: set[str] = set()
    for task in tasks:
        required_compounds = task.get("required_model_compounds") or task[
            "active_moieties"
        ]
        structure = _structure_payload(
            compounds=required_compounds,
            structures=structures,
            missing=missing_query_structures,
        )
        enriched_tasks.append({**task, **structure})
        queries.append(
            {
            "query_id": task["task_id"],
            "task": task["task_type"],
            "benchmark_level": "execution_derived",
            "split": task["split"],
            "identity_included": True,
            "source_model_family": "+".join(required_compounds),
            "active_moieties": task["active_moieties"],
            "observed_analytes": task.get("observed_analytes") or [],
            "required_model_compounds": task.get("required_model_compounds")
            or task["active_moieties"],
            "validation_group_id": task["validation_group_id"],
            "scenario": task["scenario"],
            **structure,
            }
        )
    model_records = []
    missing_candidate_structures: set[str] = set()
    for candidate in candidates:
        structure = _structure_payload(
            compounds=candidate.get("active_moieties") or [],
            structures=structures,
            missing=missing_candidate_structures,
        )
        model_records.append(
            {
            **candidate,
            "model_version_id": candidate["candidate_id"],
            "candidate_split": candidate_split_by_id[str(candidate["candidate_id"])],
            "compounds": candidate.get("active_moieties") or [],
            **structure,
            }
        )
    query_graphs = {
        str(task["task_id"]): build_execution_requirement_graph(task)
        for task in enriched_tasks
    }
    model_graphs = {
        str(record["model_version_id"]): build_model_graph(record)
        for record in model_records
    }
    perturbed_queries: list[dict[str, Any]] = []
    perturbed_query_graphs: dict[str, TypedGraph] = {}
    perturbation_links: dict[str, str] = {}
    perturbed_split_by_id: dict[str, str] = {}
    if perturbed_tasks_path is not None:
        perturbed_tasks = list(_iter_jsonl(perturbed_tasks_path))
        seen_perturbed_ids: set[str] = set()
        for task in perturbed_tasks:
            query_id = str(task["task_id"])
            base_task_id = str(task["base_task_id"])
            if query_id in seen_perturbed_ids:
                raise ValueError(f"Duplicate perturbed task ID: {query_id}")
            if base_task_id not in task_by_id:
                raise ValueError(
                    f"Perturbed task {query_id} references unknown base task {base_task_id}."
                )
            base_split = str(task_by_id[base_task_id]["split"])
            if str(task.get("split") or "") != base_split:
                raise ValueError(
                    f"Perturbed task {query_id} has split={task.get('split')!r}; "
                    f"base task {base_task_id} has split={base_split!r}."
                )
            seen_perturbed_ids.add(query_id)
            structure = _structure_payload(
                compounds=task.get("required_model_compounds")
                or task["active_moieties"],
                structures=structures,
                missing=missing_query_structures,
            )
            enriched = {**task, **structure}
            perturbed_queries.append(
                {
                    "query_id": query_id,
                    "base_query_id": base_task_id,
                    "task": task["task_type"],
                    "benchmark_level": "execution_derived_perturbation",
                    "split": task["split"],
                    "identity_included": True,
                    "source_model_family": "+".join(
                        task.get("required_model_compounds")
                        or task["active_moieties"]
                    ),
                    "active_moieties": task["active_moieties"],
                    "observed_analytes": task.get("observed_analytes") or [],
                    "required_model_compounds": task.get(
                        "required_model_compounds"
                    )
                    or task["active_moieties"],
                    "validation_group_id": task["validation_group_id"],
                    "scenario": task["scenario"],
                    "perturbation_group_id": task["perturbation_group_id"],
                    "perturbation_type": task["perturbation_type"],
                    "domain_status": task["domain_status"],
                    "scientific_grade_generated": False,
                    **structure,
                }
            )
            perturbed_query_graphs[query_id] = build_execution_requirement_graph(
                enriched
            )
            perturbation_links[query_id] = base_task_id
            perturbed_split_by_id[query_id] = str(task["split"])
    checksums = {
        "queries.jsonl": _atomic_jsonl(output_directory / "queries.jsonl", queries),
        "query_graphs.jsonl": _atomic_jsonl(
            output_directory / "query_graphs.jsonl",
            (query_graphs[str(query["query_id"])].to_dict() for query in queries),
        ),
        "model_graphs.jsonl": _atomic_jsonl(
            output_directory / "model_graphs.jsonl",
            (model_graphs[str(record["model_version_id"])].to_dict() for record in model_records),
        ),
        "model_gate_graphs.jsonl": _atomic_jsonl(
            output_directory / "model_gate_graphs.jsonl",
            (model_graphs[str(record["model_version_id"])].to_dict() for record in model_records),
        ),
        "labels.jsonl": _atomic_jsonl(
            output_directory / "labels.jsonl",
            (
                {
                    "query_id": task_id,
                    "query_task": task_by_id[task_id]["task_type"],
                    "query_split": task_by_id[task_id]["split"],
                    "candidate_id": candidate_id,
                    **label,
                }
                for (task_id, candidate_id), label in sorted(labels.items())
            ),
        ),
    }
    shutil.copy2(leakage_path, output_directory / "leakage_audit.json")
    checksums["leakage_audit.json"] = actual_leakage_sha256
    if perturbed_queries:
        checksums["perturbed_queries.jsonl"] = _atomic_jsonl(
            output_directory / "perturbed_queries.jsonl", perturbed_queries
        )
        checksums["perturbed_query_graphs.jsonl"] = _atomic_jsonl(
            output_directory / "perturbed_query_graphs.jsonl",
            (
                perturbed_query_graphs[str(query["query_id"])].to_dict()
                for query in perturbed_queries
            ),
        )
    comparison: dict[str, Any] = {}
    test_comparison: dict[str, Any] = {}
    perturbation_comparison: dict[str, Any] = {}
    test_perturbation_comparison: dict[str, Any] = {}
    for method in IMPLEMENTED_METHODS:
        rows = _score_rows(
            method=method,
            queries=queries,
            candidates=candidates,
            query_graphs=query_graphs,
            model_graphs=model_graphs,
            labels=labels,
        )
        checksums[f"predictions_{method}.jsonl"] = _atomic_jsonl(
            output_directory / f"predictions_{method}.jsonl", rows
        )
        metrics = evaluate_rankings(rows)
        checksums[f"metrics_{method}.json"] = _atomic_json(
            output_directory / f"metrics_{method}.json", metrics
        )
        comparison[method] = metrics["aggregate"]
        test_metrics = evaluate_rankings(
            row for row in rows if row["query_split"] == "test"
        )
        checksums[f"metrics_{method}_test.json"] = _atomic_json(
            output_directory / f"metrics_{method}_test.json", test_metrics
        )
        test_comparison[method] = test_metrics["aggregate"]
        if perturbed_queries:
            perturbed_rows = _score_rows(
                method=method,
                queries=perturbed_queries,
                candidates=candidates,
                query_graphs=perturbed_query_graphs,
                model_graphs=model_graphs,
                labels=None,
            )
            checksums[f"perturbation_predictions_{method}.jsonl"] = _atomic_jsonl(
                output_directory / f"perturbation_predictions_{method}.jsonl",
                perturbed_rows,
            )
            continuity = evaluate_ranking_continuity(
                base_rows=rows,
                perturbed_rows=perturbed_rows,
                task_links=perturbation_links,
            )
            checksums[f"perturbation_metrics_{method}.json"] = _atomic_json(
                output_directory / f"perturbation_metrics_{method}.json",
                continuity,
            )
            perturbation_comparison[method] = {
                key: value for key, value in continuity.items() if key != "details"
            }
            test_links = {
                perturbed_id: base_id
                for perturbed_id, base_id in perturbation_links.items()
                if perturbed_split_by_id[perturbed_id] == "test"
            }
            test_continuity = evaluate_ranking_continuity(
                base_rows=(row for row in rows if row["query_split"] == "test"),
                perturbed_rows=(
                    row
                    for row in perturbed_rows
                    if row["query_split"] == "test"
                ),
                task_links=test_links,
            )
            checksums[f"perturbation_metrics_{method}_test.json"] = _atomic_json(
                output_directory / f"perturbation_metrics_{method}_test.json",
                test_continuity,
            )
            test_perturbation_comparison[method] = {
                key: value
                for key, value in test_continuity.items()
                if key != "details"
            }
    split_counts = Counter(str(query["split"]) for query in queries)
    source_label_manifest = json.loads(
        (labels_directory / "manifest.json").read_text(encoding="utf-8")
    )
    same_compound_candidate_counts = {}
    for task in tasks:
        active = {
            str(value).casefold()
            for value in task.get("required_model_compounds")
            or task.get("active_moieties")
            or []
        }
        same_compound_candidate_counts[str(task["task_id"])] = sum(
            1
            for candidate in candidates
            if active
            and active.issubset(
                {
                    str(value).casefold()
                    for value in candidate.get("active_moieties") or []
                }
            )
        )
    non_degenerate_query_count = sum(
        count >= 2 for count in same_compound_candidate_counts.values()
    )
    has_complete_splits = all(split_counts.get(split, 0) > 0 for split in ("train", "validation", "test"))
    blockers = list(source_label_manifest.get("scientific_claim_blockers") or [])
    if non_degenerate_query_count == 0:
        blockers.append("NON_DEGENERATE_RANKING_QUERY_COUNT_ZERO")
    if not has_complete_splits:
        blockers.append("TRAIN_VALIDATION_TEST_SPLITS_NOT_ALL_PRESENT")
    if missing_query_structures or missing_candidate_structures:
        blockers.append("MOLECULAR_REPRESENTATION_COVERAGE_INCOMPLETE")
    blockers = sorted(set(blockers))
    scientific_eligible = bool(
        source_label_manifest.get("scientific_claim_eligible")
        and non_degenerate_query_count > 0
        and has_complete_splits
        and not blockers
    )
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_level": "execution_derived",
        "scientific_claim_eligible": scientific_eligible,
        "scientific_claim_blockers": [] if scientific_eligible else blockers,
        "query_count": len(queries),
        "candidate_count": len(candidates),
        "non_degenerate_ranking_query_count": non_degenerate_query_count,
        "same_compound_candidate_counts": same_compound_candidate_counts,
        "pair_count_per_method": len(expected_pairs),
        "perturbed_query_count": len(perturbed_queries),
        "perturbed_pair_count_per_method": len(perturbed_queries) * len(candidates),
        "split_query_counts": dict(sorted(split_counts.items())),
        "label_coverage": len(labels) / len(expected_pairs) if expected_pairs else 0.0,
        "adaptation_supervision_available": bool(
            source_label_manifest.get("adaptation_supervision_available")
        ),
        "adaptation_executed_count": int(
            source_label_manifest.get("adaptation_executed_count") or 0
        ),
        "adaptation_scientific_passed_count": int(
            source_label_manifest.get(
                "adaptation_scientific_passed_count"
            )
            or 0
        ),
        "molecular_representation": {
            **structure_manifest,
            "query_coverage": (
                1.0
                - len(missing_query_structures)
                / len(
                    {
                        " ".join(str(value).casefold().split())
                        for task in tasks
                        for value in task.get("required_model_compounds")
                        or task["active_moieties"]
                    }
                )
                if tasks
                else 0.0
            ),
            "candidate_coverage": (
                1.0
                - len(missing_candidate_structures)
                / len(
                    {
                        " ".join(str(value).casefold().split())
                        for candidate in candidates
                        for value in candidate["active_moieties"]
                    }
                )
                if candidates
                else 0.0
            ),
            "missing_query_compounds": sorted(missing_query_structures),
            "missing_candidate_compounds": sorted(
                missing_candidate_structures
            ),
        },
        "implemented_methods": list(IMPLEMENTED_METHODS),
        "comparison": comparison,
        "test_comparison": test_comparison,
        "perturbation_comparison": perturbation_comparison,
        "test_perturbation_comparison": test_perturbation_comparison,
        "perturbed_tasks_path": (
            str(perturbed_tasks_path.resolve()) if perturbed_tasks_path else None
        ),
        "source_label_manifest": str((labels_directory / "manifest.json").resolve()),
        "source_leakage_audit": str(leakage_path.resolve()),
        "source_leakage_audit_sha256": actual_leakage_sha256,
        "candidate_split_counts": dict(
            sorted(Counter(candidate_split_by_id.values()).items())
        ),
        "output_checksums": checksums,
        "elapsed_seconds": time.perf_counter() - started,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(output_directory / "experiment_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic baselines on execution-derived PBPK labels."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--structures", type=Path)
    parser.add_argument("--perturbed-tasks", type=Path)
    args = parser.parse_args()
    result = run_execution_experiment(
        benchmark_directory=args.benchmark,
        labels_directory=args.labels,
        output_directory=args.output,
        structures_path=args.structures,
        perturbed_tasks_path=args.perturbed_tasks,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
