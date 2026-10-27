from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pbpk_kb.paths import recommendation_model_cache_dir
from pbpk_kb.recommendation.embedding_benchmark import (
    _aggregate,
    _atomic_json,
    _atomic_jsonl,
    _evaluate_dense,
    _evaluate_hashing,
    _evaluate_tfidf,
    _model_factory,
    _rank_metrics,
    MODEL_INPUT_SERIALIZATION_VERSION,
    load_jsonl,
    sha256_file,
)


BENCHMARK_VERSION = "pbpk-retrieval-silver-1.9.0"
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_MAX_LENGTH = 2048
RERANKER_INSTRUCTION = (
    "Given a PBPK modeling or pharmacokinetics query, judge whether the "
    "provenance-backed knowledge-graph subgraph is directly relevant and "
    "contains the entities, relations, study context, and evidence needed "
    "to answer the query."
)
CANDIDATE_TOP_K_PER_RETRIEVER = 50
CANDIDATE_POOL_SIZE = 100
RRF_K = 60
DEFAULT_DENSE_MODELS = (
    "ncbi/MedCPT",
    "NeuML/pubmedbert-base-embeddings",
    "Alibaba-NLP/gte-modernbert-base",
    "BAAI/bge-base-en-v1.5",
    "Qwen/Qwen3-Embedding-0.6B",
)
ACCEPTANCE_THRESHOLDS = {
    "minimum_query_count": 120,
    "minimum_candidate_hit_rate_at_100": 0.95,
    "minimum_category_candidate_hit_rate_at_100": 0.90,
    "minimum_reranked_hit_rate_at_10": 0.80,
    "minimum_reranked_mrr": 0.45,
    "minimum_reranked_ndcg_at_10": 0.50,
    "maximum_reranker_truncated_pair_count": 0,
    "minimum_composite_improvement_over_best_baseline": 0.005,
    "maximum_category_hit_rate_regression_vs_best_baseline": 0.10,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_query_references(
    queries: list[dict[str, Any]],
    subgraph_dir: Path,
) -> dict[str, int]:
    corpus_ids = {
        hop: {
            str(row["subgraph_id"])
            for row in load_jsonl(subgraph_dir / f"subgraphs_{hop}hop.jsonl")
        }
        for hop in (0, 1, 2)
    }
    positive_count = 0
    hard_negative_count = 0
    for query in queries:
        hop = int(query["target_hop"])
        if hop not in corpus_ids:
            raise RuntimeError(f"Query uses unsupported hop: {hop}")
        positives = {str(value) for value in query.get("positive_subgraph_ids") or []}
        negatives = {str(value) for value in query.get("hard_negative_subgraph_ids") or []}
        if not positives:
            raise RuntimeError(f"Query has no positive targets: {query.get('query_id')}")
        missing_positives = positives - corpus_ids[hop]
        missing_negatives = negatives - corpus_ids[hop]
        if missing_positives or missing_negatives:
            raise RuntimeError(
                f"Query target reference mismatch for {query.get('query_id')}: "
                f"missing_positives={len(missing_positives)}, "
                f"missing_hard_negatives={len(missing_negatives)}"
            )
        positive_count += len(positives)
        hard_negative_count += len(negatives)
    return {
        "query_count": len(queries),
        "positive_reference_count": positive_count,
        "hard_negative_reference_count": hard_negative_count,
    }


def prepare_locked_benchmark(
    source_dir: Path,
    subgraph_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_query = source_dir / "locked_retrieval_silver.jsonl"
    source_manifest = json.loads(
        (source_dir / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("query_sha256") != sha256_file(source_query):
        raise RuntimeError("Source retrieval benchmark checksum mismatch.")

    queries = load_jsonl(source_query)
    reference_audit = validate_query_references(queries, subgraph_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    query_path = output_dir / "locked_retrieval_silver.jsonl"
    shutil.copy2(source_query, query_path)
    acceptance_path = output_dir / "preregistered_acceptance.json"
    _atomic_json(
        acceptance_path,
        {
            "benchmark_version": BENCHMARK_VERSION,
            "locked_before_reranker_evaluation": True,
            "candidate_generation": {
                "dense_models": list(DEFAULT_DENSE_MODELS),
                "lexical_model": "sklearn-word-bigram-v1",
                "model_input_serialization_version": MODEL_INPUT_SERIALIZATION_VERSION,
                "top_k_per_retriever": CANDIDATE_TOP_K_PER_RETRIEVER,
                "pool_size": CANDIDATE_POOL_SIZE,
                "fusion": f"RRF(k={RRF_K})",
            },
            "reranker": {
                "model": RERANKER_MODEL,
                "instruction": RERANKER_INSTRUCTION,
                "max_length": RERANKER_MAX_LENGTH,
                "score": "softmax_probability_of_yes_over_yes_no_logits",
            },
            "thresholds": ACCEPTANCE_THRESHOLDS,
            "created_at": utc_now(),
        },
    )
    manifest = {
        "accepted": True,
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_level": "silver",
        "query_count": len(queries),
        "query_sha256": sha256_file(query_path),
        "acceptance_sha256": sha256_file(acceptance_path),
        "source_benchmark_version": source_manifest.get("benchmark_version"),
        "source_query_sha256": source_manifest.get("query_sha256"),
        "subgraph_hashes": {
            f"{hop}hop": sha256_file(subgraph_dir / f"subgraphs_{hop}hop.jsonl")
            for hop in (0, 1, 2)
        },
        "query_reference_audit": reference_audit,
        "limitations": [
            "This development benchmark reuses locked KG-derived silver queries.",
            "No positive IDs, label_basis fields, or hard-negative labels may enter retrieval inputs.",
            "Independent expert gold evaluation remains mandatory at Gate 6.",
        ],
        "generated_at": utc_now(),
    }
    _atomic_json(output_dir / "benchmark_manifest.json", manifest)
    return manifest


def _candidate_metrics(
    candidates_by_query: dict[str, list[str]],
    queries: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        query_id = str(query["query_id"])
        candidates = candidates_by_query[query_id]
        positives = {str(value) for value in query["positive_subgraph_ids"]}
        row = {
            "query_id": query_id,
            "category": str(query["category"]),
            "target_hop": int(query["target_hop"]),
            "candidate_count": len(candidates),
            "hit_rate_at_50": float(bool(set(candidates[:50]) & positives)),
            "hit_rate_at_100": float(bool(set(candidates[:100]) & positives)),
            "recall_at_50": len(set(candidates[:50]) & positives) / len(positives),
            "recall_at_100": len(set(candidates[:100]) & positives) / len(positives),
        }
        rows.append(row)

    def averages(values: list[dict[str, Any]]) -> dict[str, float]:
        names = ("hit_rate_at_50", "hit_rate_at_100", "recall_at_50", "recall_at_100")
        return {
            name: sum(float(row[name]) for row in values) / len(values)
            for name in names
        }

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
    return {
        "overall": averages(rows),
        "by_category": {
            category: averages(category_rows)
            for category, category_rows in sorted(by_category.items())
        },
        "per_query": rows,
    }


def build_candidate_pool(
    *,
    retriever_rankings: dict[str, dict[str, list[str]]],
    queries: list[dict[str, Any]],
    top_k_per_retriever: int = CANDIDATE_TOP_K_PER_RETRIEVER,
    pool_size: int = CANDIDATE_POOL_SIZE,
    rrf_k: int = RRF_K,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    candidates_by_query: dict[str, list[str]] = {}
    audit_rows: list[dict[str, Any]] = []
    retriever_names = sorted(retriever_rankings)
    for query in queries:
        query_id = str(query["query_id"])
        scores: dict[str, float] = defaultdict(float)
        source_ranks: dict[str, dict[str, int]] = defaultdict(dict)
        for retriever_name in retriever_names:
            ranking = retriever_rankings[retriever_name][query_id][:top_k_per_retriever]
            for rank, object_id in enumerate(ranking, 1):
                scores[object_id] += 1.0 / (rrf_k + rank)
                source_ranks[object_id][retriever_name] = rank
        ordered = sorted(scores, key=lambda value: (-scores[value], value))[:pool_size]
        candidates_by_query[query_id] = ordered
        for rank, object_id in enumerate(ordered, 1):
            audit_rows.append(
                {
                    "query_id": query_id,
                    "target_hop": int(query["target_hop"]),
                    "candidate_subgraph_id": object_id,
                    "candidate_rank": rank,
                    "rrf_score": scores[object_id],
                    "retriever_ranks": source_ranks[object_id],
                }
            )
    return candidates_by_query, audit_rows


def infer_query_graph_constraints(query_text: str) -> dict[str, Any]:
    text = " ".join(query_text.lower().replace("-", " ").split())
    if "affected compound" in text and any(
        value in text for value in ("endpoint", "exposure", "two edge")
    ):
        return {"required_path": ["VICTIM_IN", "AFFECTS_PARAMETER"]}
    if "causal interacting compound" in text and any(
        value in text for value in ("metabolic machinery", "enzyme", "transporter", "two edge")
    ):
        return {"required_path": ["PERPETRATOR_IN", "MEDIATED_BY"]}
    if (
        ("exposure is altered" in text and "perpetrator" in text)
        or ("auc or cmax changed by another drug" in text)
        or ("receiving the pharmacokinetic interaction effect" in text)
    ):
        return {"required_center_predicate": "VICTIM_IN"}
    if "victim" in text and any(
        value in text for value in ("parameter", "endpoint", "outcome", "exposure", "two edge")
    ):
        return {"required_path": ["VICTIM_IN", "AFFECTS_PARAMETER"]}
    if "perpetrator" in text and any(
        value in text for value in ("enzyme", "transporter", "mediator", "machinery", "two edge")
    ):
        return {"required_path": ["PERPETRATOR_IN", "MEDIATED_BY"]}
    if "compound" in text and any(
        value in text for value in ("quantitative value", "numerical", "numeric observation")
    ):
        return {"required_path": ["OF_COMPOUND", "HAS_VALUE"]}
    if "victim" in text:
        return {"required_center_predicate": "VICTIM_IN"}
    if "perpetrator" in text:
        return {"required_center_predicate": "PERPETRATOR_IN"}

    relation_terms = (
        ("DESCRIBED_BY_MODEL", ("pbpk model", "physiologically based pharmacokinetic model")),
        ("STUDIED_IN", ("population specific", "population context", "demographic", "disease context")),
        ("INHIBITS", ("inhibitor", "inhibition", "inhibitory", "suppresses")),
        ("SUBSTRATE_OF", ("substrate of", "processed by an enzyme")),
        ("TRANSPORTED_BY", ("transporter dependent", "membrane transporter", "carried by")),
        ("ABSORBED_VIA", ("absorption", "absorbed", "enters systemic circulation")),
        ("EXCRETED_VIA", ("excretion", "excreted", "renal biliary", "urinary removal")),
        ("UNDER_REGIMEN", ("dose route and schedule", "administration regimen", "dosing regimen")),
        ("HAS_VALUE", ("numerical value", "numeric observation", "quantitative measurement")),
        ("HAS_UNIT", ("measurement unit", "dimensional unit", "unit annotated")),
    )
    for predicate, terms in relation_terms:
        if any(term in text for term in terms):
            return {"required_predicate": predicate}
    return {}


def graph_constraint_passes(
    query_text: str,
    subgraph: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    constraint = infer_query_graph_constraints(query_text)
    if not constraint:
        return True, {}
    edges = subgraph.get("edges") or []
    if "required_center_predicate" in constraint:
        predicate = str(constraint["required_center_predicate"])
        center_id = str(subgraph["center_id"])
        passes = any(
            edge.get("predicate") == predicate
            and center_id in {str(edge.get("subject_id")), str(edge.get("object_id"))}
            for edge in edges
        )
        return passes, constraint
    if "required_predicate" in constraint:
        predicate = str(constraint["required_predicate"])
        return any(edge.get("predicate") == predicate for edge in edges), constraint

    first_predicate, second_predicate = constraint["required_path"]
    center_id = str(subgraph["center_id"])
    middle_nodes: set[str] = set()
    for edge in edges:
        if edge.get("predicate") != first_predicate:
            continue
        subject_id = str(edge.get("subject_id"))
        object_id = str(edge.get("object_id"))
        if subject_id == center_id:
            middle_nodes.add(object_id)
        elif object_id == center_id:
            middle_nodes.add(subject_id)
    passes = any(
        edge.get("predicate") == second_predicate
        and (
            str(edge.get("subject_id")) in middle_nodes
            or str(edge.get("object_id")) in middle_nodes
        )
        for edge in edges
    )
    return passes, constraint


class Qwen3Reranker:
    def __init__(
        self,
        *,
        cache_dir: Path,
        device: str,
        batch_size: int,
        max_length: int,
    ) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        snapshot = Path(
            snapshot_download(
                RERANKER_MODEL,
                cache_dir=str(cache_dir),
                ignore_patterns=["*.bin", "*.onnx", "openvino/*"],
            )
        )
        self.name = RERANKER_MODEL
        self.revision = snapshot.name
        self.device = torch.device(device)
        self.torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot), padding_side="left")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            torch_dtype=dtype,
            attn_implementation="sdpa",
        ).to(self.device).eval()
        self.false_token_id = self.tokenizer.convert_tokens_to_ids("no")
        self.true_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements "
            'based on the Query and the Instruct provided. Note that the answer can only be "yes" '
            'or "no".<|im_end|>\n<|im_start|>user\n'
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def _format(self, query: str, document: str) -> str:
        return (
            f"<Instruct>: {RERANKER_INSTRUCTION}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    def score(
        self,
        pairs: list[tuple[str, str]],
    ) -> tuple[list[float], list[dict[str, Any]]]:
        scores: list[float] = []
        token_audit: list[dict[str, Any]] = []
        body_limit = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        for start in range(0, len(pairs), self.batch_size):
            formatted = [
                self._format(query, document)
                for query, document in pairs[start : start + self.batch_size]
            ]
            untruncated = self.tokenizer(
                formatted,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]
            encoded = self.tokenizer(
                formatted,
                add_special_tokens=False,
                padding=False,
                truncation="longest_first",
                max_length=body_limit,
            )
            input_ids = [
                self.prefix_tokens + token_ids + self.suffix_tokens
                for token_ids in encoded["input_ids"]
            ]
            model_inputs = self.tokenizer.pad(
                {"input_ids": input_ids},
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            model_inputs = {key: value.to(self.device) for key, value in model_inputs.items()}
            with self.torch.inference_mode():
                logits = self.model(**model_inputs).logits[:, -1, :]
                yes_logits = logits[:, self.true_token_id]
                no_logits = logits[:, self.false_token_id]
                probabilities = self.torch.softmax(
                    self.torch.stack((no_logits, yes_logits), dim=1).float(),
                    dim=1,
                )[:, 1]
            scores.extend(float(value) for value in probabilities.detach().cpu())
            for original, actual in zip(untruncated, input_ids):
                original_count = len(self.prefix_tokens) + len(original) + len(self.suffix_tokens)
                token_audit.append(
                    {
                        "original_token_count": original_count,
                        "model_input_token_count": len(actual),
                        "truncated": original_count > self.max_length,
                        "model_max_length": self.max_length,
                        "tokenizer_class": type(self.tokenizer).__name__,
                        "tokenizer_revision": self.revision,
                    }
                )
        return scores, token_audit

    def release(self) -> None:
        model = getattr(self, "model", None)
        if model is not None:
            model.to("cpu")
            self.model = None
        self.tokenizer = None


def evaluate_reranker(
    *,
    subgraph_dir: Path,
    benchmark_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    device: str,
    dense_batch_size: int,
    reranker_batch_size: int,
    reranker_max_length: int,
) -> dict[str, Any]:
    import torch

    manifest = json.loads((benchmark_dir / "benchmark_manifest.json").read_text(encoding="utf-8"))
    query_path = benchmark_dir / "locked_retrieval_silver.jsonl"
    acceptance_path = benchmark_dir / "preregistered_acceptance.json"
    if manifest.get("benchmark_version") != BENCHMARK_VERSION:
        raise RuntimeError("Unexpected reranker benchmark version.")
    if manifest.get("query_sha256") != sha256_file(query_path):
        raise RuntimeError("Locked reranker query checksum mismatch.")
    if manifest.get("acceptance_sha256") != sha256_file(acceptance_path):
        raise RuntimeError("Preregistered reranker acceptance checksum mismatch.")
    for hop in (0, 1, 2):
        expected = (manifest.get("subgraph_hashes") or {}).get(f"{hop}hop")
        observed = sha256_file(subgraph_dir / f"subgraphs_{hop}hop.jsonl")
        if expected != observed:
            raise RuntimeError(f"Locked {hop}-hop subgraph checksum mismatch.")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance["candidate_generation"]["dense_models"] != list(DEFAULT_DENSE_MODELS):
        raise RuntimeError("Candidate generator differs from preregistered model set.")
    if (
        acceptance["candidate_generation"]["model_input_serialization_version"]
        != MODEL_INPUT_SERIALIZATION_VERSION
    ):
        raise RuntimeError("Model input serialization differs from preregistered version.")
    if int(acceptance["reranker"]["max_length"]) != int(reranker_max_length):
        raise RuntimeError("Reranker max length differs from preregistered value.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA reranker run requested but no CUDA device is available.")

    corpora = {
        hop: load_jsonl(subgraph_dir / f"subgraphs_{hop}hop.jsonl")
        for hop in (0, 1, 2)
    }
    corpus_by_id = {
        str(row["subgraph_id"]): row
        for rows in corpora.values()
        for row in rows
    }
    queries = load_jsonl(query_path)
    validate_query_references(queries, subgraph_dir)
    lexical_result, lexical_rankings = _evaluate_tfidf(corpora, queries)
    hash_result = _evaluate_hashing(corpora, queries)
    retriever_rankings: dict[str, dict[str, list[str]]] = {
        "lexical_tfidf": lexical_rankings
    }
    dense_revisions: dict[str, str] = {}
    for model_name in DEFAULT_DENSE_MODELS:
        print(json.dumps({"candidate_encoder": model_name}), flush=True)
        encoder = _model_factory(
            model_name,
            cache_dir=cache_dir,
            device=device,
            batch_size=dense_batch_size,
        )
        result, _, rankings = _evaluate_dense(
            encoder=encoder,
            corpora=corpora,
            queries=queries,
        )
        retriever_rankings[model_name] = rankings
        dense_revisions[model_name] = str(result["revision"])
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    candidates_by_query, candidate_rows = build_candidate_pool(
        retriever_rankings=retriever_rankings,
        queries=queries,
    )
    candidate_metrics = _candidate_metrics(candidates_by_query, queries)
    thresholds = acceptance["thresholds"]
    candidate_checks = {
        "overall_candidate_hit_rate_at_100": (
            candidate_metrics["overall"]["hit_rate_at_100"]
            >= float(thresholds["minimum_candidate_hit_rate_at_100"])
        ),
        "category_candidate_hit_rate_at_100": (
            min(
                (
                    values["hit_rate_at_100"]
                    for values in candidate_metrics["by_category"].values()
                ),
                default=0.0,
            )
            >= float(thresholds["minimum_category_candidate_hit_rate_at_100"])
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(output_dir / "candidate_pool.jsonl", candidate_rows)
    _atomic_json(output_dir / "candidate_retrieval_metrics.json", candidate_metrics)
    if not all(candidate_checks.values()):
        failure = {
            "accepted": False,
            "stage": "candidate_generation",
            "checks": candidate_checks,
            "thresholds": thresholds,
            "candidate_metrics": candidate_metrics,
            "generated_at": utc_now(),
        }
        _atomic_json(output_dir / "reranker_comparison.json", failure)
        return failure

    flat_pairs: list[tuple[str, str]] = []
    flat_metadata: list[tuple[str, str, int]] = []
    for query in queries:
        query_id = str(query["query_id"])
        for candidate_rank, object_id in enumerate(candidates_by_query[query_id], 1):
            flat_pairs.append(
                (str(query["query_text"]), str(corpus_by_id[object_id]["embedding_text"]))
            )
            flat_metadata.append((query_id, object_id, candidate_rank))

    print(json.dumps({"loading_reranker": RERANKER_MODEL, "pair_count": len(flat_pairs)}), flush=True)
    started = time.time()
    reranker = Qwen3Reranker(
        cache_dir=cache_dir,
        device=device,
        batch_size=reranker_batch_size,
        max_length=reranker_max_length,
    )
    scores, token_audit = reranker.score(flat_pairs)
    rows_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_score_rows: list[dict[str, Any]] = []
    for metadata, score, token_stats in zip(flat_metadata, scores, token_audit):
        query_id, object_id, candidate_rank = metadata
        row = {
            "query_id": query_id,
            "candidate_subgraph_id": object_id,
            "candidate_rank": candidate_rank,
            "reranker_score": score,
            **token_stats,
        }
        rows_by_query[query_id].append(row)
        all_score_rows.append(row)

    per_query: list[dict[str, Any]] = []
    for query in queries:
        query_id = str(query["query_id"])
        parsed_constraint = infer_query_graph_constraints(str(query["query_text"]))
        for row in rows_by_query[query_id]:
            passes, observed_constraint = graph_constraint_passes(
                str(query["query_text"]),
                corpus_by_id[str(row["candidate_subgraph_id"])],
            )
            if observed_constraint != parsed_constraint:
                raise RuntimeError("Graph constraint parser produced inconsistent results.")
            row["graph_constraint"] = parsed_constraint
            row["graph_constraint_pass"] = passes
            row["final_ranking_score"] = (
                float(row["reranker_score"])
                + (2.0 if parsed_constraint and passes else 0.0)
            )
        reranked = sorted(
            rows_by_query[query_id],
            key=lambda row: (
                -float(row["final_ranking_score"]),
                int(row["candidate_rank"]),
            ),
        )
        ranking = [str(row["candidate_subgraph_id"]) for row in reranked]
        metrics = _rank_metrics(
            [float(row["final_ranking_score"]) for row in reranked],
            ranking,
            {str(value) for value in query["positive_subgraph_ids"]},
        )
        per_query.append(
            {
                "query_id": query_id,
                "category": str(query["category"]),
                "target_hop": int(query["target_hop"]),
                "graph_constraint": parsed_constraint,
                "metrics": metrics,
            }
        )
    reranked_result = _aggregate(per_query)
    reranked_result.update(
        {
            "model": RERANKER_MODEL,
            "revision": reranker.revision,
            "retrieval_method": "multi_dense_lexical_rrf100_qwen3_cross_encoder",
            "instruction": RERANKER_INSTRUCTION,
            "query_count": len(queries),
            "pair_count": len(flat_pairs),
            "elapsed_seconds": round(time.time() - started, 3),
            "per_query": sorted(per_query, key=lambda row: str(row["query_id"])),
        }
    )
    _atomic_jsonl(output_dir / "reranker_scores.jsonl", all_score_rows)
    _atomic_json(output_dir / "qwen3_reranker_metrics.json", reranked_result)

    baselines = [lexical_result, hash_result]
    best_baseline = max(baselines, key=lambda row: float(row["composite_score"]))
    category_regressions = {
        category: max(
            float(row["by_category"].get(category, {}).get("hit_rate_at_10", 0.0))
            for row in baselines
        )
        - float(reranked_result["by_category"][category]["hit_rate_at_10"])
        for category in sorted(reranked_result["by_category"])
    }
    truncated_pair_count = sum(
        1 for row in token_audit if bool(row["truncated"])
    )
    checks = {
        **candidate_checks,
        "query_count_meets_minimum": (
            len(queries) >= int(thresholds["minimum_query_count"])
        ),
        "reranked_hit_rate_at_10_meets_threshold": (
            float(reranked_result["overall"]["hit_rate_at_10"])
            >= float(thresholds["minimum_reranked_hit_rate_at_10"])
        ),
        "reranked_mrr_meets_threshold": (
            float(reranked_result["overall"]["reciprocal_rank"])
            >= float(thresholds["minimum_reranked_mrr"])
        ),
        "reranked_ndcg_at_10_meets_threshold": (
            float(reranked_result["overall"]["ndcg_at_10"])
            >= float(thresholds["minimum_reranked_ndcg_at_10"])
        ),
        "reranker_truncation_within_limit": (
            truncated_pair_count
            <= int(thresholds["maximum_reranker_truncated_pair_count"])
        ),
        "composite_beats_best_baseline": (
            float(reranked_result["composite_score"])
            - float(best_baseline["composite_score"])
            >= float(thresholds["minimum_composite_improvement_over_best_baseline"])
        ),
        "no_category_regresses_beyond_limit": (
            max(category_regressions.values(), default=0.0)
            <= float(thresholds["maximum_category_hit_rate_regression_vs_best_baseline"])
        ),
    }
    comparison = {
        "accepted": all(checks.values()),
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_level": "silver",
        "benchmark_query_sha256": sha256_file(query_path),
        "acceptance_sha256": sha256_file(acceptance_path),
        "checks": checks,
        "thresholds": thresholds,
        "candidate_generation": {
            "dense_models": list(DEFAULT_DENSE_MODELS),
            "dense_revisions": dense_revisions,
            "lexical_revision": "sklearn-word-bigram-v1",
            "model_input_serialization_version": MODEL_INPUT_SERIALIZATION_VERSION,
            "top_k_per_retriever": CANDIDATE_TOP_K_PER_RETRIEVER,
            "pool_size": CANDIDATE_POOL_SIZE,
            "rrf_k": RRF_K,
            "metrics": candidate_metrics,
        },
        "reranker": {
            "model": reranker.name,
            "revision": reranker.revision,
            "instruction": RERANKER_INSTRUCTION,
            "max_length": reranker.max_length,
            "truncated_pair_count": truncated_pair_count,
            "pair_count": len(token_audit),
        },
        "reranked_metrics": {
            "composite_score": reranked_result["composite_score"],
            "hit_rate_at_10": reranked_result["overall"]["hit_rate_at_10"],
            "mrr": reranked_result["overall"]["reciprocal_rank"],
            "ndcg_at_10": reranked_result["overall"]["ndcg_at_10"],
            "by_category": reranked_result["by_category"],
        },
        "best_baseline": {
            "model": best_baseline["model"],
            "composite_score": best_baseline["composite_score"],
        },
        "composite_improvement": (
            float(reranked_result["composite_score"])
            - float(best_baseline["composite_score"])
        ),
        "category_hit_rate_regressions": category_regressions,
        "generated_at": utc_now(),
        "limitations": [
            "This is a KG-derived development silver benchmark, not expert clinical gold validation.",
            "The reranker sees only query text and candidate subgraph serialization.",
            "Architecture changes after observing this result require a new locked benchmark revision.",
        ],
    }
    _atomic_json(output_dir / "reranker_comparison.json", comparison)
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run locked PBPK multi-retriever reranking.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-benchmark-dir", type=Path, required=True)
    prepare.add_argument("--subgraph-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--subgraph-dir", type=Path, required=True)
    run.add_argument("--benchmark-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--cache-dir", type=Path, default=recommendation_model_cache_dir())
    run.add_argument("--device", default="cuda")
    run.add_argument("--dense-batch-size", type=int, default=32)
    run.add_argument("--reranker-batch-size", type=int, default=16)
    run.add_argument("--reranker-max-length", type=int, default=RERANKER_MAX_LENGTH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        result = prepare_locked_benchmark(
            args.source_benchmark_dir,
            args.subgraph_dir,
            args.output_dir,
        )
    else:
        result = evaluate_reranker(
            subgraph_dir=args.subgraph_dir,
            benchmark_dir=args.benchmark_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            device=args.device,
            dense_batch_size=args.dense_batch_size,
            reranker_batch_size=args.reranker_batch_size,
            reranker_max_length=args.reranker_max_length,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result.get("accepted") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
