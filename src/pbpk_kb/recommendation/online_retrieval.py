from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pbpk_kb.recommendation.embedding_benchmark import _model_factory, load_jsonl
from pbpk_kb.recommendation.embedding_index import LOCKED_DENSE_MODELS
from pbpk_kb.recommendation.gnn_benchmark import (
    CATEGORY_ORDER,
    QUERY_MODEL,
    _blend_category_scores,
    _pair_forward,
    build_graph_tensors,
    build_pair_tensors,
    build_ranker,
)
from pbpk_kb.recommendation.graph_prompt import (
    GraphPromptCorpus,
    RetrievedSubgraph,
    _objective_evidence_ids,
    _quality_ok,
    _subgraph_has_compound_path,
    audit_silver_examples,
    build_silver_examples,
    canonical_json,
    canonical_objective,
    sha256_file,
)
from pbpk_kb.recommendation.reranker_benchmark import (
    CANDIDATE_POOL_SIZE,
    CANDIDATE_TOP_K_PER_RETRIEVER,
    RERANKER_MAX_LENGTH,
    RRF_K,
    Qwen3Reranker,
    build_candidate_pool,
)
from pbpk_kb.recommendation.schemas import ModelingRequest


ONLINE_RETRIEVAL_VERSION = "pbpk-gate5-online-retrieval-1.2.0"
MAX_SELECTED_SUBGRAPHS = 8
QUERY_CONDITIONED_DIMENSION = 256


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _release_accelerator_objects(*objects: Any) -> None:
    for value in objects:
        if value is None:
            continue
        release = getattr(value, "release", None)
        if callable(release):
            release()
            continue
        move = getattr(value, "to", None)
        if callable(move):
            try:
                move("cpu")
            except (RuntimeError, TypeError):
                pass
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def request_query_text(request: ModelingRequest) -> str:
    objective = canonical_objective(request.objective)
    fields = [
        f"PBPK modeling evidence for compound {request.compound}.",
        f"Objective: {objective}.",
        f"Species: {request.species}.",
    ]
    if objective == "ddi_victim":
        fields.append(
            f"{request.compound} is the victim drug and "
            f"{request.perpetrator_drug} is the perpetrator drug."
        )
        fields.append(
            "Retrieve the directed DDI event, affected PK parameter, mechanism, "
            "dose context, numerical value, unit, and provenance."
        )
    elif objective == "ddi_perpetrator":
        fields.append(
            f"{request.compound} is the perpetrator drug and "
            f"{request.victim_drug} is the victim drug."
        )
        fields.append(
            "Retrieve the directed DDI event, affected PK parameter, mechanism, "
            "dose context, numerical value, unit, and provenance."
        )
    elif objective == "public_model_reuse":
        fields.append(
            "Retrieve a directly linked reusable public PBPK model, model version, "
            "software, structure, validation information, and provenance."
        )
    elif objective in {
        "renal_impairment",
        "hepatic_impairment",
        "pediatric",
        "pregnancy",
        "special_population",
    }:
        fields.append(
            f"Population: {request.population or objective}. Retrieve population-specific "
            "PK measurements, regimen, values, units, and provenance."
        )
    else:
        fields.append(
            "Retrieve physicochemical, absorption, distribution, metabolism, "
            "transport, excretion, clinical PK parameter, value, unit, regimen, "
            "population, and provenance evidence."
        )
    for label, value in (
        ("Route", request.route),
        ("Population", request.population),
        ("Formulation", request.formulation),
        ("Dose unit", request.dose_unit),
        ("Constraints", request.constraints),
    ):
        if value:
            fields.append(f"{label}: {value}.")
    if request.dose is not None:
        fields.append(f"Dose: {request.dose} {request.dose_unit or ''}.".strip())
    return " ".join(fields)


def request_category(request: ModelingRequest) -> str:
    objective = canonical_objective(request.objective)
    if objective in {"ddi_victim", "ddi_perpetrator"}:
        return "ddi_role"
    if objective in {
        "renal_impairment",
        "hepatic_impairment",
        "pediatric",
        "pregnancy",
        "special_population",
    }:
        return "special_population"
    if objective == "public_model_reuse":
        return "multi_hop_path"
    return "scenario_query"


def request_path_relations(request: ModelingRequest) -> list[str]:
    objective = canonical_objective(request.objective)
    if objective in {"ddi_victim", "ddi_perpetrator"}:
        return [
            "AFFECTS_PARAMETER",
            "MEDIATED_BY",
            "PERPETRATOR_IN",
            "VICTIM_IN",
        ]
    if objective == "public_model_reuse":
        return ["DESCRIBED_BY_MODEL"]
    if objective in {
        "renal_impairment",
        "hepatic_impairment",
        "pediatric",
        "pregnancy",
        "special_population",
    }:
        return ["OF_COMPOUND", "STUDIED_IN"]
    return ["HAS_UNIT", "HAS_VALUE", "MEASURES", "OF_COMPOUND"]


def _hard_graph_compatibility(
    corpus: GraphPromptCorpus,
    request: ModelingRequest,
    subgraph: dict[str, Any],
) -> tuple[bool, list[str]]:
    compound_id = str(corpus.resolve_compound(request.compound)["node_id"])
    errors = []
    if not (
        str(subgraph.get("center_id") or "") == compound_id
        or _subgraph_has_compound_path(subgraph, compound_id)
    ):
        errors.append("target_compound_path_absent")
    item = RetrievedSubgraph(
        subgraph_id=str(subgraph["subgraph_id"]),
        hop=int(subgraph["hop_limit"]),
        score=0.0,
        score_components={},
        row=subgraph,
        vector=None,
    )
    objective_evidence = _objective_evidence_ids(
        corpus=corpus,
        request=request,
        subgraphs=[item],
        compound_id=compound_id,
    )
    if not objective_evidence:
        errors.append("objective_evidence_absent")
    text = str(subgraph.get("embedding_text") or "").lower()
    for label, value in (
        ("route", request.route),
        ("population", request.population),
        ("formulation", request.formulation),
    ):
        if value and str(value).lower() not in text:
            errors.append(f"{label}_context_absent")
    return not errors, errors


def _lexical_rankings(
    records_by_hop: dict[int, list[dict[str, Any]]],
    query_rows: list[dict[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    rankings: dict[str, dict[str, list[str]]] = {}
    query_texts = [str(row["query_text"]) for row in query_rows]
    for hop, records in records_by_hop.items():
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        documents = [str(row["embedding_text"]) for row in records]
        document_matrix = vectorizer.fit_transform(documents)
        query_matrix = vectorizer.transform(query_texts)
        values: dict[str, list[str]] = {}
        for row_index, query in enumerate(query_rows):
            scores = (document_matrix @ query_matrix[row_index].T).toarray()[:, 0]
            order = np.argsort(-scores, kind="stable")[
                :CANDIDATE_TOP_K_PER_RETRIEVER
            ]
            values[str(query["query_id"])] = [
                str(records[int(index)]["object_id"]) for index in order
            ]
        rankings[f"lexical_word_bigram::{hop}hop"] = values
    return rankings


def _dense_model_query_rankings(
    *,
    model_name: str,
    index_dir: Path,
    index_manifest: dict[str, Any],
    records_by_hop: dict[int, list[dict[str, Any]]],
    query_rows: list[dict[str, Any]],
    query_texts: list[str],
    cache_dir: Path,
    device: str,
    batch_size: int,
    blas_threads: int,
) -> tuple[dict[str, dict[str, list[str]]], np.ndarray, str]:
    from threadpoolctl import threadpool_limits

    encoder = None
    try:
        encoder = _model_factory(
            model_name,
            cache_dir=cache_dir,
            device=device,
            batch_size=batch_size,
            query_only=True,
        )
        expected = str(index_manifest["models"][model_name]["revision"])
        if str(encoder.revision) != expected:
            raise RuntimeError(
                f"Embedding revision mismatch for {model_name}: "
                f"{encoder.revision} != {expected}"
            )
        query_vectors = np.asarray(
            encoder.encode_queries(query_texts),
            dtype=np.float32,
        )
        if query_vectors.shape != (len(query_rows), 768):
            raise RuntimeError(
                f"Unexpected query vectors for {model_name}: {query_vectors.shape}"
            )
        rankings: dict[str, dict[str, list[str]]] = {}
        for hop, records in records_by_hop.items():
            artifact = index_manifest["models"][model_name]["artifacts"][
                f"subgraph_{hop}hop"
            ]
            vector_path = index_dir / str(artifact["vectors_path"])
            if sha256_file(vector_path) != str(artifact["vectors_sha256"]):
                raise RuntimeError(f"Dense index checksum mismatch: {vector_path}")
            document_vectors = np.load(vector_path, mmap_mode="r")
            with threadpool_limits(limits=blas_threads, user_api="blas"):
                scores = np.asarray(document_vectors @ query_vectors.T)
            values: dict[str, list[str]] = {}
            for row_index, query in enumerate(query_rows):
                order = np.argsort(-scores[:, row_index], kind="stable")[
                    :CANDIDATE_TOP_K_PER_RETRIEVER
                ]
                values[str(query["query_id"])] = [
                    str(records[int(index)]["object_id"]) for index in order
                ]
            rankings[f"{model_name}::{hop}hop"] = values
        return rankings, query_vectors, str(encoder.revision)
    finally:
        _release_accelerator_objects(encoder)


def _dense_rankings(
    *,
    index_dir: Path,
    index_manifest: dict[str, Any],
    records_by_hop: dict[int, list[dict[str, Any]]],
    query_rows: list[dict[str, Any]],
    cache_dir: Path,
    device: str,
    batch_size: int,
    blas_threads: int = 1,
) -> tuple[dict[str, dict[str, list[str]]], np.ndarray, dict[str, str]]:
    rankings: dict[str, dict[str, list[str]]] = {}
    qwen_queries: np.ndarray | None = None
    revisions: dict[str, str] = {}
    query_texts = [str(row["query_text"]) for row in query_rows]
    for model_name in LOCKED_DENSE_MODELS:
        model_rankings, query_vectors, revision = _dense_model_query_rankings(
            model_name=model_name,
            index_dir=index_dir,
            index_manifest=index_manifest,
            records_by_hop=records_by_hop,
            query_rows=query_rows,
            query_texts=query_texts,
            cache_dir=cache_dir,
            device=device,
            batch_size=batch_size,
            blas_threads=blas_threads,
        )
        rankings.update(model_rankings)
        revisions[model_name] = revision
        if model_name == QUERY_MODEL:
            qwen_queries = query_vectors.copy()
        del query_vectors
    if qwen_queries is None:
        raise RuntimeError("Qwen3 query embeddings were not produced.")
    return rankings, qwen_queries, revisions


def _run_cross_encoder(
    *,
    candidates_by_query: dict[str, list[str]],
    query_rows: list[dict[str, Any]],
    record_by_id: dict[str, dict[str, Any]],
    cache_dir: Path,
    device: str,
    batch_size: int,
) -> tuple[dict[tuple[str, str], float], list[dict[str, Any]], str]:
    reranker = Qwen3Reranker(
        cache_dir=cache_dir,
        device=device,
        batch_size=batch_size,
        max_length=RERANKER_MAX_LENGTH,
    )
    pairs = []
    keys = []
    for query in query_rows:
        query_id = str(query["query_id"])
        for subgraph_id in candidates_by_query[query_id]:
            keys.append((query_id, subgraph_id))
            pairs.append(
                (
                    str(query["query_text"]),
                    str(record_by_id[subgraph_id]["embedding_text"]),
                )
            )
    try:
        scores, token_audit = reranker.score(pairs)
        if any(bool(row["truncated"]) for row in token_audit):
            raise RuntimeError("Gate 5 cross-encoder input was truncated.")
        revision = str(reranker.revision)
    finally:
        _release_accelerator_objects(reranker)
        del reranker
    return dict(zip(keys, scores, strict=True)), token_audit, revision


def _load_gnn(
    *,
    graph_dir: Path,
    embedding_index_dir: Path,
    gnn_training_dir: Path,
    gnn_index_dir: Path,
    device: str,
) -> tuple[Any, Any, Any, Any, dict[str, Any], dict[str, Any]]:
    import torch

    manifest = json.loads(
        (gnn_index_dir / "gnn_index_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("accepted") is not True:
        raise RuntimeError("Gate 4 HGT index is not accepted.")
    seed = int(manifest["selected_seed"])
    hgt_path = gnn_training_dir / "hgt" / f"seed_{seed}.pt"
    teacher_path = gnn_training_dir / "no_gnn" / f"seed_{seed}.pt"
    if sha256_file(hgt_path) != str(manifest["checkpoint_sha256"]):
        raise RuntimeError("Gate 4 HGT checkpoint checksum mismatch.")
    if sha256_file(teacher_path) != str(manifest["teacher_checkpoint_sha256"]):
        raise RuntimeError("Gate 4 NoGNN checkpoint checksum mismatch.")
    hgt_checkpoint = torch.load(hgt_path, map_location="cpu", weights_only=False)
    teacher_checkpoint = torch.load(
        teacher_path,
        map_location="cpu",
        weights_only=False,
    )
    graph = build_graph_tensors(
        graph_dir=graph_dir,
        embedding_index_dir=embedding_index_dir,
    )
    target = torch.device(device)
    hgt = build_ranker(
        architecture="hgt",
        graph=graph,
        category_count=len(CATEGORY_ORDER),
    )
    teacher = build_ranker(
        architecture="no_gnn",
        graph=graph,
        category_count=len(CATEGORY_ORDER),
    )
    hgt.load_state_dict(hgt_checkpoint["state_dict"])
    teacher.load_state_dict(teacher_checkpoint["state_dict"])
    hgt.to(target).eval()
    teacher.to(target).eval()
    graph = graph.to(target)
    with torch.inference_mode():
        hgt_nodes = hgt.encode_nodes(graph)
        teacher_nodes = teacher.encode_nodes(graph)
    return (
        hgt,
        teacher,
        hgt_nodes,
        teacher_nodes,
        graph,
        hgt_checkpoint,
    )


def _gnn_rerank(
    *,
    query_rows: list[dict[str, Any]],
    candidates_by_query: dict[str, list[str]],
    candidate_audit: list[dict[str, Any]],
    cross_scores: dict[tuple[str, str], float],
    corpus: GraphPromptCorpus,
    query_embeddings: np.ndarray,
    graph_dir: Path,
    subgraph_dir: Path,
    embedding_index_dir: Path,
    gnn_benchmark_dir: Path,
    gnn_training_dir: Path,
    gnn_index_dir: Path,
    device: str,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    import torch

    (
        hgt,
        teacher,
        hgt_nodes,
        teacher_nodes,
        graph,
        hgt_checkpoint,
    ) = _load_gnn(
        graph_dir=graph_dir,
        embedding_index_dir=embedding_index_dir,
        gnn_training_dir=gnn_training_dir,
        gnn_index_dir=gnn_index_dir,
        device=device,
    )
    subgraph_index = load_jsonl(gnn_benchmark_dir / "subgraph_index.jsonl")
    subgraph_by_id = {
        str(row["subgraph_id"]): row for row in subgraph_index
    }
    node_index = {
        node_id: index for index, node_id in enumerate(graph.node_ids)
    }
    query_by_id = {str(row["query_id"]): row for row in query_rows}
    audit_by_key = {
        (str(row["query_id"]), str(row["candidate_subgraph_id"])): row
        for row in candidate_audit
    }
    query_tensor = torch.from_numpy(query_embeddings).to(device)
    result_rows = []
    selected_vectors = np.zeros(
        (len(query_rows), MAX_SELECTED_SUBGRAPHS, QUERY_CONDITIONED_DIMENSION),
        dtype=np.float32,
    )
    selected_masks = np.zeros(
        (len(query_rows), MAX_SELECTED_SUBGRAPHS),
        dtype=np.bool_,
    )
    for query_index, query in enumerate(query_rows):
        query_id = str(query["query_id"])
        request = ModelingRequest.model_validate(query["modeling_request"])
        category = str(query["category"])
        pair_rows = []
        graph_checks = {}
        for rank, subgraph_id in enumerate(candidates_by_query[query_id], 1):
            subgraph = corpus.subgraphs[subgraph_id]
            graph_pass, graph_errors = _hard_graph_compatibility(
                corpus,
                request,
                subgraph,
            )
            graph_checks[subgraph_id] = (graph_pass, graph_errors)
            audit = audit_by_key[(query_id, subgraph_id)]
            pair_rows.append(
                {
                    "query_id": query_id,
                    "subgraph_id": subgraph_id,
                    "candidate_rank": rank,
                    "rrf_score": float(audit["rrf_score"]),
                    "retriever_ranks": dict(audit["retriever_ranks"]),
                    "reranker_score": float(cross_scores[(query_id, subgraph_id)]),
                    "final_ranking_score": float(
                        cross_scores[(query_id, subgraph_id)]
                    ),
                    "graph_constraint_pass": graph_pass,
                    "category": category,
                    "path_relation_targets": request_path_relations(request),
                    "relevance_label": 0,
                    "hard_negative": False,
                    "parameter_ranking_eligible": False,
                    "citation_selection_label": 0.0,
                }
            )
        pairs = build_pair_tensors(
            rows=pair_rows,
            query_row_by_id={query_id: query_index},
            subgraph_by_id=subgraph_by_id,
            node_index=node_index,
            relation_vocab=graph.relation_vocab,
        ).to(device)
        with torch.inference_mode():
            hgt_output = _pair_forward(
                hgt,
                hgt_nodes,
                query_tensor,
                pairs,
                return_attention=True,
            )
            teacher_output = _pair_forward(
                teacher,
                teacher_nodes,
                query_tensor,
                pairs,
            )
        hgt_scores = hgt_output["scores"].detach().cpu().tolist()
        teacher_scores = teacher_output["scores"].detach().cpu().tolist()
        alpha = float(
            hgt_checkpoint["validation_alphas"].get(category, 0.0)
        )
        final_scores = _blend_category_scores(
            rows=pair_rows,
            teacher_scores=teacher_scores,
            graph_scores=hgt_scores,
            alphas={category: alpha},
        )
        representations = (
            hgt_output["pair_representation"].detach().cpu().numpy().astype(np.float32)
        )
        compound_id = str(corpus.resolve_compound(request.compound)["node_id"])
        order = sorted(
            range(len(pair_rows)),
            key=lambda index: (
                not bool(pair_rows[index]["graph_constraint_pass"]),
                str(
                    corpus.subgraphs[
                        str(pair_rows[index]["subgraph_id"])
                    ].get("center_id")
                    or ""
                )
                != compound_id,
                -int(
                    corpus.subgraphs[
                        str(pair_rows[index]["subgraph_id"])
                    ].get("hop_limit")
                    or 0
                )
                if str(
                    corpus.subgraphs[
                        str(pair_rows[index]["subgraph_id"])
                    ].get("center_id")
                    or ""
                )
                == compound_id
                else 0,
                -float(final_scores[index]),
                str(pair_rows[index]["subgraph_id"]),
            ),
        )
        compatible = [
            index
            for index in order
            if bool(pair_rows[index]["graph_constraint_pass"])
        ]
        anchors = [
            index
            for index in compatible
            if str(
                corpus.subgraphs[
                    str(pair_rows[index]["subgraph_id"])
                ].get("center_id")
                or ""
            )
            == compound_id
        ]
        anchors.sort(
            key=lambda index: (
                -int(
                    corpus.subgraphs[
                        str(pair_rows[index]["subgraph_id"])
                    ].get("hop_limit")
                    or 0
                ),
                -float(final_scores[index]),
                str(pair_rows[index]["subgraph_id"]),
            )
        )
        objective_evidence_by_index: dict[int, set[str]] = {}
        for pair_index in compatible:
            subgraph_id = str(pair_rows[pair_index]["subgraph_id"])
            item = RetrievedSubgraph(
                subgraph_id=subgraph_id,
                hop=int(corpus.subgraphs[subgraph_id]["hop_limit"]),
                score=float(final_scores[pair_index]),
                score_components={},
                row=corpus.subgraphs[subgraph_id],
                vector=None,
            )
            objective_evidence_by_index[pair_index] = _objective_evidence_ids(
                corpus=corpus,
                request=request,
                subgraphs=[item],
                compound_id=compound_id,
            )
        selected_indices = anchors[:MAX_SELECTED_SUBGRAPHS]
        covered_evidence = set().union(
            *(
                objective_evidence_by_index[index]
                for index in selected_indices
            ),
            set(),
        )
        while (
            len(selected_indices) < MAX_SELECTED_SUBGRAPHS
            and len(selected_indices) < len(compatible)
        ):
            pair_index = min(
                (
                    index
                    for index in compatible
                    if index not in selected_indices
                ),
                key=lambda index: (
                    -len(
                        objective_evidence_by_index[index]
                        - covered_evidence
                    ),
                    -float(final_scores[index]),
                    str(pair_rows[index]["subgraph_id"]),
                ),
            )
            selected_indices.append(pair_index)
            covered_evidence.update(
                objective_evidence_by_index[pair_index]
            )
        ranked = []
        for final_rank, pair_index in enumerate(order, 1):
            row = pair_rows[pair_index]
            subgraph_id = str(row["subgraph_id"])
            graph_pass, graph_errors = graph_checks[subgraph_id]
            ranked.append(
                {
                    **row,
                    "final_rank": final_rank,
                    "graph_constraint_errors": graph_errors,
                    "teacher_score": float(teacher_scores[pair_index]),
                    "raw_hgt_score": float(hgt_scores[pair_index]),
                    "residual_alpha": alpha,
                    "final_score": float(final_scores[pair_index]),
                }
            )
        for token_index, pair_index in enumerate(selected_indices):
            selected_vectors[query_index, token_index] = representations[pair_index]
            selected_masks[query_index, token_index] = True
        selected_ids = [
            str(pair_rows[index]["subgraph_id"]) for index in selected_indices
        ]
        if not selected_ids:
            raise RuntimeError(f"No graph-compatible result for {query_id}.")
        positive_ids = set(map(str, query.get("positive_subgraph_ids") or ()))
        target = query["target"]
        selected_evidence = {
            str(citation.get("evidence_id") or "")
            for subgraph_id in selected_ids
            for citation in corpus.subgraphs[subgraph_id].get("citations") or ()
            if citation.get("quality_status") == "auto_validated"
        }
        selected_models = {
            str(node.get("node_id") or "")
            for subgraph_id in selected_ids
            for node in corpus.subgraphs[subgraph_id].get("nodes") or ()
            if node.get("node_type") == "PBPKModel"
        }
        target_evidence_recall = len(
            selected_evidence & set(map(str, target["evidence_ids"]))
        ) / max(1, len(target["evidence_ids"]))
        target_model_recall = (
            len(
                selected_models
                & set(map(str, target["public_model_ids"]))
            )
            / len(target["public_model_ids"])
            if target["public_model_ids"]
            else 1.0
        )
        result_rows.append(
            {
                "query_row": query_index,
                "example_id": query["example_id"],
                "query_id": query_id,
                "query_text": query["query_text"],
                "category": category,
                "modeling_request": query["modeling_request"],
                "selected_subgraph_ids": selected_ids,
                "ranked_candidates": ranked,
                "candidate_hit_at_100": bool(
                    set(candidates_by_query[query_id]) & positive_ids
                ),
                "candidate_recall_at_100": len(
                    set(candidates_by_query[query_id]) & positive_ids
                )
                / max(1, len(positive_ids)),
                "selected_hit_at_8": bool(set(selected_ids) & positive_ids),
                "selected_recall_at_8": len(set(selected_ids) & positive_ids)
                / max(1, len(positive_ids)),
                "target_evidence_recall_at_8": target_evidence_recall,
                "target_public_model_recall_at_8": target_model_recall,
                "retrieval_status": (
                    "retrieval_complete"
                    if target_evidence_recall == 1.0
                    and target_model_recall == 1.0
                    else "retrieval_failure"
                ),
            }
        )
    _release_accelerator_objects(hgt, teacher, graph)
    del (
        hgt,
        teacher,
        hgt_nodes,
        teacher_nodes,
        graph,
        query_tensor,
        pairs,
        hgt_output,
        teacher_output,
    )
    _release_accelerator_objects()
    return result_rows, selected_vectors, selected_masks


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = (
        "candidate_hit_at_100",
        "candidate_recall_at_100",
        "selected_hit_at_8",
        "selected_recall_at_8",
        "target_evidence_recall_at_8",
        "target_public_model_recall_at_8",
    )

    def means(values: list[dict[str, Any]]) -> dict[str, float]:
        return {
            name: sum(float(row[name]) for row in values) / max(1, len(values))
            for name in names
        }

    categories = sorted({str(row["category"]) for row in rows})
    return {
        "overall": {
            **means(rows),
            "complete_target_coverage_rate": sum(
                row["retrieval_status"] == "retrieval_complete"
                for row in rows
            )
            / max(1, len(rows)),
        },
        "by_category": {
            category: {
                **means(
                    selected := [
                        row for row in rows if row["category"] == category
                    ]
                ),
                "complete_target_coverage_rate": sum(
                    row["retrieval_status"] == "retrieval_complete"
                    for row in selected
                )
                / max(1, len(selected)),
            }
            for category in categories
        },
    }


def run_online_retrieval(
    *,
    graph_dir: Path,
    subgraph_dir: Path,
    embedding_index_dir: Path,
    gnn_benchmark_dir: Path,
    gnn_training_dir: Path,
    gnn_index_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    device: str,
    dense_batch_size: int,
    reranker_batch_size: int,
) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = GraphPromptCorpus(
        graph_dir=graph_dir,
        subgraph_dir=subgraph_dir,
        gnn_index_dir=gnn_index_dir,
    )
    examples = build_silver_examples(corpus)
    silver_audit = audit_silver_examples(corpus, examples)
    if not silver_audit["accepted"]:
        _atomic_json(output_dir / "silver_ownership_audit.json", silver_audit)
        raise RuntimeError("Independent Gate 5 silver audit failed.")
    query_rows = [
        {
            **row,
            "query_id": "gate5-query:"
            + hashlib.sha256(
                canonical_json(row["modeling_request"]).encode("utf-8")
            ).hexdigest()[:24],
            "query_text": request_query_text(
                ModelingRequest.model_validate(row["modeling_request"])
            ),
            "category": request_category(
                ModelingRequest.model_validate(row["modeling_request"])
            ),
            "target_hop": 2,
        }
        for row in examples
    ]
    index_manifest_path = embedding_index_dir / "embedding_index_manifest.json"
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    records_by_hop = {
        hop: load_jsonl(
            embedding_index_dir
            / str(
                index_manifest["source_records"][
                    f"subgraph_{hop}hop"
                ]["records_path"]
            )
        )
        for hop in (1, 2)
    }
    record_by_id = {
        str(row["object_id"]): row
        for records in records_by_hop.values()
        for row in records
    }
    lexical = _lexical_rankings(records_by_hop, query_rows)
    dense, qwen_queries, dense_revisions = _dense_rankings(
        index_dir=embedding_index_dir,
        index_manifest=index_manifest,
        records_by_hop=records_by_hop,
        query_rows=query_rows,
        cache_dir=cache_dir,
        device=device,
        batch_size=dense_batch_size,
    )
    candidates_by_query, candidate_audit = build_candidate_pool(
        retriever_rankings={**lexical, **dense},
        queries=query_rows,
        top_k_per_retriever=CANDIDATE_TOP_K_PER_RETRIEVER,
        pool_size=CANDIDATE_POOL_SIZE,
        rrf_k=RRF_K,
    )
    cross_scores, cross_token_audit, reranker_revision = _run_cross_encoder(
        candidates_by_query=candidates_by_query,
        query_rows=query_rows,
        record_by_id=record_by_id,
        cache_dir=cache_dir,
        device=device,
        batch_size=reranker_batch_size,
    )
    result_rows, selected_vectors, selected_masks = _gnn_rerank(
        query_rows=query_rows,
        candidates_by_query=candidates_by_query,
        candidate_audit=candidate_audit,
        cross_scores=cross_scores,
        corpus=corpus,
        query_embeddings=qwen_queries,
        graph_dir=graph_dir,
        subgraph_dir=subgraph_dir,
        embedding_index_dir=embedding_index_dir,
        gnn_benchmark_dir=gnn_benchmark_dir,
        gnn_training_dir=gnn_training_dir,
        gnn_index_dir=gnn_index_dir,
        device=device,
    )
    metrics = _aggregate_metrics(result_rows)
    category_hit_ok = all(
        values["candidate_hit_at_100"] >= 0.90
        for values in metrics["by_category"].values()
    )
    acceptance = {
        "candidate_hit_at_100": (
            metrics["overall"]["candidate_hit_at_100"] >= 0.95
        ),
        "category_candidate_hit_at_100": category_hit_ok,
        "complete_target_coverage_rate": (
            sum(
                row["retrieval_status"] == "retrieval_complete"
                for row in result_rows
            )
            / max(1, len(result_rows))
            >= 0.95
        ),
        "target_public_model_recall_at_8": (
            all(
                float(row["target_public_model_recall_at_8"]) == 1.0
                for row in result_rows
            )
        ),
        "cross_encoder_no_truncation": not any(
            bool(row["truncated"]) for row in cross_token_audit
        ),
        "silver_ownership_audit": bool(silver_audit["accepted"]),
    }
    paths = {
        "silver": output_dir / "recommendation_silver.jsonl",
        "queries": output_dir / "online_queries.jsonl",
        "candidate_audit": output_dir / "rrf_candidate_audit.jsonl",
        "cross_token_audit": output_dir / "cross_encoder_token_audit.jsonl",
        "results": output_dir / "online_retrieval_results.jsonl",
        "failures": output_dir / "retrieval_failures.jsonl",
        "vectors": output_dir / "query_conditioned_subgraph_vectors.npy",
        "masks": output_dir / "query_conditioned_subgraph_masks.npy",
        "silver_audit": output_dir / "silver_ownership_audit.json",
    }
    _atomic_jsonl(paths["silver"], examples)
    _atomic_jsonl(
        paths["queries"],
        [
            {
                key: row[key]
                for key in (
                    "example_id",
                    "query_id",
                    "query_text",
                    "category",
                    "modeling_request",
                )
            }
            for row in query_rows
        ],
    )
    _atomic_jsonl(paths["candidate_audit"], candidate_audit)
    _atomic_jsonl(paths["cross_token_audit"], cross_token_audit)
    _atomic_jsonl(paths["results"], result_rows)
    _atomic_jsonl(
        paths["failures"],
        [
            row
            for row in result_rows
            if row["retrieval_status"] == "retrieval_failure"
        ],
    )
    _atomic_npy(paths["vectors"], selected_vectors)
    _atomic_npy(paths["masks"], selected_masks)
    _atomic_json(paths["silver_audit"], silver_audit)
    manifest = {
        "accepted": all(acceptance.values()),
        "online_retrieval_version": ONLINE_RETRIEVAL_VERSION,
        "generated_at": utc_now(),
        "query_count": len(query_rows),
        "query_category_counts": dict(
            sorted(Counter(str(row["category"]) for row in query_rows).items())
        ),
        "dense_models": list(LOCKED_DENSE_MODELS),
        "dense_revisions": dense_revisions,
        "lexical_revision": "sklearn-word-bigram-v1",
        "rrf": {
            "k": RRF_K,
            "top_k_per_retriever_per_hop": CANDIDATE_TOP_K_PER_RETRIEVER,
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
        },
        "reranker": {
            "model": "Qwen/Qwen3-Reranker-0.6B",
            "revision": reranker_revision,
            "max_length": RERANKER_MAX_LENGTH,
        },
        "gnn": {
            "selected_architecture": "hgt",
            "query_conditioned_dimension": QUERY_CONDITIONED_DIMENSION,
            "checkpoint_sha256": json.loads(
                (gnn_index_dir / "gnn_index_manifest.json").read_text(
                    encoding="utf-8"
                )
            )["checkpoint_sha256"],
        },
        "metrics": metrics,
        "acceptance_checks": acceptance,
        "silver_audit": silver_audit,
        "vector_shape": list(selected_vectors.shape),
        "mask_shape": list(selected_masks.shape),
        "artifacts": {
            key: {
                "path": path.name,
                "sha256": sha256_file(path),
            }
            for key, path in paths.items()
        },
        "input_hashes": {
            "graph_manifest": sha256_file(
                graph_dir / "kg_extraction_manifest.json"
            ),
            "subgraph_manifest": sha256_file(
                subgraph_dir / "gate2_manifest.json"
            ),
            "embedding_index_manifest": sha256_file(index_manifest_path),
            "gnn_index_manifest": sha256_file(
                gnn_index_dir / "gnn_index_manifest.json"
            ),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    _atomic_json(output_dir / "online_retrieval_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked Gate 3 + Gate 4 online retrieval chain."
    )
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--subgraph-dir", type=Path, required=True)
    parser.add_argument("--embedding-index-dir", type=Path, required=True)
    parser.add_argument("--gnn-benchmark-dir", type=Path, required=True)
    parser.add_argument("--gnn-training-dir", type=Path, required=True)
    parser.add_argument("--gnn-index-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dense-batch-size", type=int, default=32)
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_online_retrieval(
        graph_dir=args.graph_dir,
        subgraph_dir=args.subgraph_dir,
        embedding_index_dir=args.embedding_index_dir,
        gnn_benchmark_dir=args.gnn_benchmark_dir,
        gnn_training_dir=args.gnn_training_dir,
        gnn_index_dir=args.gnn_index_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        device=args.device,
        dense_batch_size=args.dense_batch_size,
        reranker_batch_size=args.reranker_batch_size,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
