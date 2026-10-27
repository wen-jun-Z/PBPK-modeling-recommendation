#!/usr/bin/env python3
"""Batch frozen hybrid retrieval, then local Qwen text-graph component selection.

No training, no recommendation-quality relabelling, and no remote services.
The old learned graph projector is deliberately not used with a different LLM.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

from prepare_component_model_integration import (ROOT, OUT, GRAPH, INDEX, GNN,
    GNN_INDEX, WEIGHTS, VERSION, artifact, digest, read, rows, scoped, sha, write, write_rows)
sys.path.insert(0, str(ROOT / "src"))
from pbpk_kb.recommendation.graph_prompt import (GraphPromptCorpus, RetrievedSubgraph,
    assemble_recommendation, serialize_text_graph)
from pbpk_kb.recommendation.component_model_bridge import validate_component_output
from pbpk_kb.recommendation.schemas import ModelingRequest

RERANKER_REPAIR_OLD_PRODUCER = "023164f73d870c608072808ee58f0c13e1cbf87531436550944f8d2055429e0b"
CONTEXT_RESUME_OLD_PRODUCER = "5b21381a2853c5546168cee01490f4ba414209a57bc2016dd74b18e308de25bb"
RERANKER_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
DENSE_PREPROCESSING = "complete-token-chunks-token-weighted-l2-v1"
DENSE_PREFLIGHT = ROOT / "runtime/cache/component_integration_dense_token_preflight.json"


def corpus_for(output):
    return GraphPromptCorpus(graph_dir=output / "inputs/graph", subgraph_dir=output / "inputs/subgraphs")


def assert_inputs(output):
    protocol = read(output / "protocol.json")
    if protocol["version"] != VERSION or not isinstance(protocol.get("ranking_and_evaluation"), dict):
        raise ValueError("Complete generation and evaluation protocol must be frozen first")
    for ref in [*protocol["source_artifacts"], *protocol["prepared_input_artifacts"]]:
        if sha(ref["path"]) != ref["sha256"]:
            raise ValueError("Registered input changed: " + ref["path"])
    binding = {"protocol_sha256": sha(output / "protocol.json"),
               "prepare_script": artifact(ROOT / "scripts/prepare_component_model_integration.py"),
               "generate_script": artifact(Path(__file__))}
    path = output / "run_binding.json"
    if path.exists() and read(path) != binding:
        raise ValueError("Protocol/producer changed after this run began")
    write(path, binding)
    return protocol


def masked_graph_bundle(output, *, graph_dir, embedding_index_dir):
    """Preserve checkpoint vocabularies while removing prohibited message paths."""
    import torch
    from pbpk_kb.recommendation.gnn_benchmark import build_graph_tensors
    graph = build_graph_tensors(graph_dir=graph_dir, embedding_index_dir=embedding_index_dir)
    mask = read(output / "inputs/graph_mask.json")
    node_allow, edge_allow = set(mask["allowed_node_ids"]), set(mask["allowed_edge_ids"])
    off = torch.tensor([nid not in node_allow for nid in graph.node_ids], dtype=torch.bool)
    for key in ("text_features", "numeric_features", "source_ids", "unit_ids", "context_ids"):
        getattr(graph, key)[off] = 0
    keep = torch.tensor([eid in edge_allow for eid in graph.edge_ids], dtype=torch.bool)
    full_keep = torch.cat([keep, keep])
    graph.forward_edge_index = graph.forward_edge_index[:, keep]
    graph.full_edge_index = graph.full_edge_index[:, full_keep]
    for key in ("full_edge_type", "full_edge_numeric", "full_edge_source_ids", "full_edge_version_ids"):
        setattr(graph, key, getattr(graph, key)[full_keep])
    graph.edge_ids = [eid for eid, flag in zip(graph.edge_ids, keep.tolist()) if flag]
    graph.relation_ids = [rid for rid, flag in zip(graph.relation_ids, keep.tolist()) if flag]
    # HGT uses type-local edge arrays, not the full_edge_index above.
    lookup = {typ: {graph.node_ids[int(g)]: i for i, g in enumerate(indices)}
              for typ, indices in graph.hetero_node_indices.items()}
    buffers = {key: [[], []] for key in graph.hetero_edge_indices}
    node_type = dict(zip(graph.node_ids, graph.node_types))
    for edge in rows(output / "inputs/graph/kg_edges.jsonl"):
        a, b = edge["subject_id"], edge["object_id"]
        at, bt, rel = node_type[a], node_type[b], edge["predicate"]
        for key, left, right in [((at, "FWD__" + rel, bt), lookup[at][a], lookup[bt][b]),
                                 ((bt, "REV__" + rel, at), lookup[bt][b], lookup[at][a])]:
            buffers[key][0].append(left)
            buffers[key][1].append(right)
    graph.hetero_edge_indices = {key: torch.tensor(value, dtype=torch.long) for key, value in buffers.items()}
    assert not any(graph.node_ids[int(i)] not in node_allow for i in graph.full_edge_index.flatten())
    return graph


def progress(output, phase, **values):
    value = {"phase": phase, "updated_at_unix": time.time(), **values}
    write(output / "progress.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def component_context_audit(corpus, request, subgraph):
    """Admit supported mechanisms without claiming unreported clinical fit."""
    from pbpk_kb.recommendation.graph_prompt import _objective_evidence_ids, _subgraph_has_compound_path
    from pbpk_kb.recommendation.component_model_bridge import _contexts, _context_audit
    compound_id = str(corpus.resolve_compound(request.compound)["node_id"])
    errors = []
    if not (str(subgraph.get("center_id") or "") == compound_id
            or _subgraph_has_compound_path(subgraph, compound_id)):
        errors.append("target_compound_path_absent")
    item = RetrievedSubgraph(subgraph_id=str(subgraph["subgraph_id"]), hop=int(subgraph["hop_limit"]),
        score=0., score_components={}, row=subgraph, vector=None)
    evidence = sorted(_objective_evidence_ids(corpus=corpus, request=request,
        subgraphs=[item], compound_id=compound_id))
    audits = {eid: _context_audit(request, _contexts(corpus, eid, compound_id)) for eid in evidence}
    eligible = [eid for eid in evidence if audits[eid]["status"] != "conflicting"]
    if not evidence:
        errors.append("objective_evidence_absent")
    elif not eligible:
        errors.append("all_objective_evidence_explicitly_conflicting")
    return {"pass": not errors, "errors": errors, "evidence_context": audits,
        "eligible_evidence_ids": eligible,
        "interpretation": "Unknown conditions permit soft mechanism preferences only; no assertion of clinical-condition fit."}


def soft_graph_compatibility(corpus, request, subgraph):
    audit = component_context_audit(corpus, request, subgraph)
    return audit["pass"], audit["errors"]


def token_count(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


def complete_text_chunks(tokenizer, text, *, prefix="", limit):
    """Split at tokenizer offsets, preserving every source character in order."""
    if token_count(tokenizer, prefix + text) <= limit:
        return [(text, max(1, len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])))]
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Long-input splitting requires exact token-to-character offsets")
    encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    budget = limit - token_count(tokenizer, prefix) - 4
    if budget < 1:
        raise ValueError("Retrieval instruction alone exceeds the tokenizer limit")
    result, token_start, char_start = [], 0, 0
    while token_start < len(offsets):
        stop = min(token_start + budget, len(offsets))
        while True:
            char_end = len(text) if stop == len(offsets) else int(offsets[stop][0])
            piece = text[char_start:char_end]
            if piece and token_count(tokenizer, prefix + piece) <= limit:
                break
            stop -= 1
            if stop <= token_start:
                raise ValueError("A complete source character cannot fit the token budget")
        weight = len(tokenizer(piece, add_special_tokens=False, truncation=False)["input_ids"])
        result.append((piece, max(1, weight)))
        token_start, char_start = stop, char_end
    if "".join(piece for piece, _ in result) != text:
        raise ValueError("Token chunking omitted or reordered source characters")
    return result


def complete_dense_encode(encoder, texts, *, role):
    """Keep native long context; bounded encoders use all token chunks."""
    import numpy as np
    is_medcpt = hasattr(encoder, "query_tokenizer")
    if is_medcpt:
        tokenizer = encoder.query_tokenizer if role == "query" else encoder.article_tokenizer
        model = encoder.query_model if role == "query" else encoder.article_model
        original_limit = limit = 64 if role == "query" else 512
        prefix = ""
    else:
        model = encoder.model
        tokenizer = model.tokenizer
        original_limit = int(model.max_seq_length)
        prompt_name = encoder.query_prompt_name if role == "query" else None
        prompt_name = prompt_name or getattr(model, "default_prompt_name", None)
        prompt = (getattr(model, "prompts", {}) or {}).get(prompt_name, "")
        prefix = prompt + (encoder.query_prefix if role == "query" else "")
        native = int(getattr(model[0].auto_model.config, "max_position_embeddings", original_limit))
        if encoder.name in {"Qwen/Qwen3-Embedding-0.6B", "Alibaba-NLP/gte-modernbert-base"}:
            needed = max((token_count(tokenizer, prefix + t) for t in texts), default=original_limit)
            limit = min(native, max(original_limit, min(32768, needed)))
            model.max_seq_length = limit
        else:
            limit = min(original_limit, native)
    lengths = [token_count(tokenizer, prefix + text) for text in texts]
    chunks, parents, weights, counts = [], [], [], []
    for i, text in enumerate(texts):
        pieces = complete_text_chunks(tokenizer, text, prefix=prefix, limit=limit)
        counts.append(len(pieces))
        for piece, weight in pieces:
            chunks.append(piece)
            parents.append(i)
            weights.append(weight)
    # Verify the actual wrapper input for every chunk, including its instruction.
    actual_counts = [token_count(tokenizer, prefix + chunk) for chunk in chunks]
    if any(count > limit for count in actual_counts):
        raise ValueError("Complete dense chunk would still be truncated")
    # All five wrappers already normalize chunk embeddings. Preserve that score
    # convention, then take a token-weighted mean and L2 normalize each full text.
    function = encoder.encode_queries if role == "query" else encoder.encode_documents
    vectors = np.asarray(function(chunks), dtype=np.float32)
    if vectors.shape[0] != len(chunks):
        raise ValueError("Missing dense chunk embeddings")
    result = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
    for i, vector, weight in zip(parents, vectors, weights, strict=True):
        result[i] += vector * weight
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if not np.isfinite(result).all() or np.any(norms <= 0):
        raise ValueError("Invalid token-aggregated dense embedding")
    result /= norms
    audit = {"policy": DENSE_PREPROCESSING, "role": role, "texts": len(texts),
        "original_max_seq_length": original_limit, "effective_max_seq_length": limit,
        "original_token_counts": lengths, "original_excess_count": sum(n > original_limit for n in lengths),
        "chunk_counts": counts, "maximum_actual_chunk_tokens": max(actual_counts, default=0),
        "all_source_characters_preserved": True, "actual_chunk_truncation_count": 0,
        "aggregation": "effective content token weighted mean of normalized chunk vectors, then L2; instructions repeated but excluded from weights"}
    return result, audit


def complete_pair_lengths(reranker, pairs):
    """Measure the exact unchanged yes/no prompt, without any truncation."""
    counts = []
    extra = len(reranker.prefix_tokens) + len(reranker.suffix_tokens)
    for start in range(0, len(pairs), 64):
        texts = [reranker._format(q, d) for q, d in pairs[start:start + 64]]
        tokens = reranker.tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        counts.extend(len(ids) + extra for ids in tokens)
    return counts


def select_context_limit(lengths, model_limit, ceiling=32768):
    """Raise before scoring if a complete pair cannot fit the registered model."""
    maximum = max(lengths, default=0)
    supported = min(int(model_limit), int(ceiling))
    if maximum > supported:
        raise ValueError(f"Complete reranker input needs {maximum} tokens; supported ceiling is {supported}. No truncation allowed.")
    limit = 4096
    while limit < maximum:
        limit *= 2
    return min(limit, supported)


def score_complete_pairs(reranker, pairs, lengths, *, maximum_batch=8, token_budget=16384,
                         initial_records=None, checkpoint=None, on_progress=None):
    """Use the frozen scorer with larger context; verify every actual token count.

    Results are restored to original pair order. Length bucketing affects padding
    only. CUDA OOM halves the batch at most three times for an eight-row batch;
    a single-row failure stops, preserving any completed checkpoint.
    """
    done = dict(initial_records or {})
    if len(lengths) != len(pairs) or any(i < 0 or i >= len(pairs) for i in done):
        raise ValueError("Pair/length/checkpoint index mismatch")
    pending = sorted((i for i in range(len(pairs)) if i not in done), key=lambda i: (lengths[i], i))
    maximum_batch = min(8, max(1, int(maximum_batch)))
    since_checkpoint = 0
    while pending:
        size = min(maximum_batch, len(pending))
        while size > 1 and max(lengths[i] for i in pending[:size]) * size > token_budget:
            size //= 2
        selected = pending[:size]
        reranker.batch_size = size
        try:
            scores, audits = reranker.score([pairs[i] for i in selected])
        except reranker.torch.cuda.OutOfMemoryError:
            gc.collect()
            reranker.torch.cuda.empty_cache()
            if size == 1:
                if checkpoint:
                    checkpoint(done)
                raise
            maximum_batch = max(1, size // 2)
            if on_progress:
                on_progress(completed=len(done), total=len(pairs), batch_size=maximum_batch, event="oom_halve")
            continue
        if len(scores) != size or len(audits) != size:
            raise ValueError("Reranker returned an incomplete batch")
        for i, score, audit in zip(selected, scores, audits, strict=True):
            if (audit.get("truncated") or audit.get("original_token_count") != lengths[i]
                    or audit.get("model_input_token_count") != lengths[i]):
                raise ValueError("Reranker input was truncated or changed despite complete-input preflight")
            if not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
                raise ValueError("Nonfinite or out-of-range reranker probability")
            done[i] = {"score": float(score), "audit": audit}
        del pending[:size]
        since_checkpoint += size
        if since_checkpoint >= 256 or not pending:
            if checkpoint:
                checkpoint(done)
            if on_progress:
                on_progress(completed=len(done), total=len(pairs), batch_size=maximum_batch, event="scored_complete_inputs")
            since_checkpoint = 0
    return [done[i]["score"] for i in range(len(pairs))], [done[i]["audit"] for i in range(len(pairs))]


def run_complete_cross_encoder(*, output, candidates_by_query, query_rows, record_by_id, cache_dir, device, batch_size):
    from pbpk_kb.recommendation.reranker_benchmark import Qwen3Reranker, RERANKER_INSTRUCTION
    from pbpk_kb.recommendation.online_retrieval import _release_accelerator_objects
    keys, pairs = [], []
    for query in query_rows:
        for sid in candidates_by_query[query["query_id"]]:
            keys.append((query["query_id"], sid))
            pairs.append((query["query_text"], record_by_id[sid]["embedding_text"]))
    reranker = Qwen3Reranker(cache_dir=cache_dir, device=device, batch_size=min(8, batch_size), max_length=32768)
    try:
        if reranker.revision != RERANKER_REVISION:
            raise ValueError("The existing frozen Qwen3 reranker revision changed")
        lengths = complete_pair_lengths(reranker, pairs)
        maximum = getattr(reranker.model.config, "max_position_embeddings", 0)
        reranker.max_length = select_context_limit(lengths, maximum)
        binding = digest({"keys": keys, "pairs": pairs, "revision": reranker.revision,
                          "instruction": RERANKER_INSTRUCTION, "context_limit": reranker.max_length})
        preflight = {"identity": binding, "pair_count": len(pairs), "minimum_tokens": min(lengths, default=0),
            "maximum_tokens": max(lengths, default=0), "over_historical_2048": sum(n > 2048 for n in lengths),
            "context_limit": reranker.max_length, "model_max_position_embeddings": maximum,
            "token_counts": lengths, "truncation": False, "scoring": "Unchanged Qwen3Reranker.score yes/no softmax",
            "revision": reranker.revision, "maximum_batch_size": min(8, batch_size), "padded_token_budget": 16384}
        write(output / "retrieval/cross_encoder_preflight.json", preflight)
        progress(output, "component_cross_encoder_complete_preflight", **{k: preflight[k] for k in
                 ("pair_count", "maximum_tokens", "over_historical_2048", "context_limit")})
        partial = output / "retrieval/cross_encoder_partial.json"
        restored = {}
        if partial.exists():
            value = read(partial)
            if value["identity"] != binding:
                raise ValueError("Partial complete-input reranker checkpoint changed")
            restored = {int(i): r for i, r in value["records"].items()}
            for i, record in restored.items():
                audit = record["audit"]
                if not 0 <= i < len(pairs) or audit.get("truncated") or audit.get("model_input_token_count") != lengths[i]:
                    raise ValueError("Invalid partial reranker token audit")
        scores, audit = score_complete_pairs(reranker, pairs, lengths, maximum_batch=min(8, batch_size),
            initial_records=restored,
            checkpoint=lambda done: write(partial, {"identity": binding, "records": done}),
            on_progress=lambda **v: progress(output, "component_cross_encoder", **v))
        return dict(zip(keys, scores, strict=True)), audit, reranker.revision
    finally:
        _release_accelerator_objects(reranker)


def record_reranker_repair(output):
    """Explicit, narrow migration after a failed, unpublished reranker stage."""
    output = scoped(output)
    revision_dir = output / "revisions/001_complete_retrieval_inputs"
    if (revision_dir / "repair.json").exists():
        record = read(revision_dir / "repair.json")
        if read(output / "run_binding.json") != record["new_binding"]:
            raise ValueError("Previously recorded repair does not match current binding")
        return record
    old = read(output / "run_binding.json")
    if old["generate_script"]["sha256"] != RERANKER_REPAIR_OLD_PRODUCER:
        raise ValueError("Repair applies only to the documented first failed producer")
    if (old["protocol_sha256"] != sha(output / "protocol.json")
            or old["prepare_script"] != artifact(ROOT / "scripts/prepare_component_model_integration.py")):
        raise ValueError("Unrelated inputs/protocol changed before reranker repair")
    if any((output / path).exists() for path in ("retrieval/cross_encoder.json", "retrieval/manifest.json", "components/manifest.json")):
        raise ValueError("Cannot migrate a completed or published reranker stage")
    if any((output / "components").glob("*.json")):
        raise ValueError("Cannot migrate after component generation began")
    cached = []
    for path in sorted((output / "retrieval").glob("*.npz")):
        registry = path.with_suffix(".json")
        if read(registry)["sha256"] != sha(path):
            raise ValueError("Earlier dense cache is not intact")
        cached.extend((artifact(path), artifact(registry)))
    revision_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("run_binding.json", "protocol.json"):
        (revision_dir / ("before_" + filename)).write_bytes((output / filename).read_bytes())
    protocol = read(output / "protocol.json")
    protocol["retrieval"]["cross_encoder_complete_input"] = {
        "revision": RERANKER_REVISION, "input_truncation": False,
        "context_limit_selection": "Measure all original prompt tokens first; smallest power of two >= max(4096, observed maximum), capped by min(model context,32768).",
        "maximum_batch_size": 8, "padded_token_budget": 16384, "oom": "halve; single-row OOM stops with checkpoint",
        "weights_instruction_yes_no_scoring_unchanged": True, "previous_truncated_scores_reused": False}
    if not DENSE_PREFLIGHT.exists():
        raise ValueError("Independent CPU dense-token audit is required before cache migration")
    dense_preflight = read(DENSE_PREFLIGHT)
    affected, unaffected = [], []
    for model_name, audit in dense_preflight["models"].items():
        (affected if audit["query"]["over_limit"] or audit["document"]["over_limit"] else unaffected).append(model_name)
    protocol["retrieval"]["dense_complete_input"] = {
        "policy": DENSE_PREPROCESSING, "query_prefix_and_prompt_included_in_preflight": True,
        "long_context_models": "Qwen/GTE use declared native context, at most32768; longer inputs are chunked",
        "bounded_context_models": "Keep original wrapper budget (MedCPT query64/document512; BERT/BGE512); exact-character token-offset chunks",
        "aggregation": "Content-token-weighted normalized chunk embeddings, followed by L2; repeat query instruction on every chunk",
        "old_cache_invalidation": affected, "unchanged_complete_cache_allowed": unaffected,
        "audit": artifact(DENSE_PREFLIGHT), "no_new_training_or_outcome_access": True}
    protocol["retrieval"]["fixed_subgraph_retrieval_summary"] = {
        "node_entries": 12, "citation_entries": 4, "characters_per_citation": 600,
        "selection_order": "Frozen source subgraph order; restriction existed before first run, documented here.",
        "full_corpus_retained": True, "generator_prompt": "Existing compact serializer: <=8 evidence entries, first600 characters; full sources preserved for validator/audit."}
    write(output / "protocol.json", protocol)
    new = {"protocol_sha256": sha(output / "protocol.json"), "prepare_script": old["prepare_script"],
           "generate_script": artifact(Path(__file__))}
    repair = {"revision": 1, "failure_type": "RuntimeError: Gate 5 cross-encoder input was truncated.",
        "failed_stage": "cross_encoder", "discarded_unpublished_pairs": 10100,
        "repair": "Complete-input reranker preflight and larger context; dense tokenizer audit and complete text chunk aggregation. Frozen model weights/instructions/score formula unchanged; truncated dense caches invalidated, candidate pools recomputed.",
        "old_binding": old, "new_binding": new, "reused_dense_cache_artifacts": cached,
        "dense_models_recomputed": affected, "dense_models_complete_before_repair": unaffected,
        "frozen_old_library_modified": False, "evaluation_labels_or_results_accessed": False}
    for cached_ref in cached:
        source = scoped(cached_ref["path"])
        target = revision_dir / "before_dense_cache" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        # Preserve the actual prior cache (under 8 MB total), not just a hash.
        target.write_bytes(source.read_bytes())
    # Record the transition before installing its producer binding.
    write(revision_dir / "repair.json", repair)
    write(output / "run_binding.json", new)
    return repair


def record_context_resume_repair(output):
    """Archive the completed old gate result; reuse only its unaffected scores."""
    output = scoped(output)
    revision_dir = output / "revisions/002_context_and_attempt_resume"
    record_path = revision_dir / "repair.json"
    if record_path.exists():
        record = read(record_path)
        if read(output / "run_binding.json") != record["new_binding"]:
            raise ValueError("Recorded revision002 does not match current producer binding")
        return record
    old = read(output / "run_binding.json")
    if old["generate_script"]["sha256"] != CONTEXT_RESUME_OLD_PRODUCER:
        raise ValueError("Revision002 applies only to the completed audited retrieval producer")
    if old["protocol_sha256"] != sha(output / "protocol.json") or old["prepare_script"] != artifact(
            ROOT / "scripts/prepare_component_model_integration.py"):
        raise ValueError("Unrelated preparation/protocol changed before revision002")
    before_protocol = read(output / "protocol.json")
    for ref in [*before_protocol.get("source_artifacts", []), *before_protocol.get("prepared_input_artifacts", [])]:
        if sha(ref["path"]) != ref["sha256"]:
            raise ValueError("Registered input changed before revision002: " + ref["path"])
    producer_backup = revision_dir / "before_generate_component_model_integration.py"
    if not producer_backup.exists() or sha(producer_backup) != old["generate_script"]["sha256"]:
        raise ValueError("Exact old producer must be retained before revision002")
    if list((output / "components").rglob("*.json")):
        raise ValueError("Context migration is forbidden after component generation began")
    old_manifest = read(output / "retrieval/manifest.json")
    if old_manifest.get("status") != "completed":
        raise ValueError("Previous retrieval must have a completed manifest")
    for ref in old_manifest["artifacts"]:
        if sha(ref["path"]) != ref["sha256"]:
            raise ValueError("Previous retrieval artifact changed before archive")
    preserved = []
    for name in ("run_binding.json", "protocol.json", "retrieval/manifest.json", "retrieval/results.jsonl"):
        source = output / name
        target = revision_dir / ("before_" + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        preserved.append({"original": artifact(source), "archive": artifact(target)})
    for source in sorted((output / "inputs/prompts").glob("*.json")):
        target = revision_dir / "before_prompts" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        preserved.append({"original": artifact(source), "archive": artifact(target)})
    caches = []
    for path in sorted((output / "retrieval").iterdir()):
        if path.name in {"manifest.json", "results.jsonl"} or path.suffix not in {".json", ".npz"}:
            continue
        caches.append(artifact(path))
    protocol = read(output / "protocol.json")
    protocol["retrieval"]["component_condition_compatibility"] = {
        "scope": "Soft mechanism preference bridge; not proof of complete clinical-condition match",
        "unchanged_hard_gates": ["target_compound_path", "objective_evidence"],
        "context_audit": "Typed bridge context: explicit conflicts are excluded; missing or incompletely parsed conditions remain unknown.",
        "aggregation": "A subgraph needs at least one nonconflicting objective evidence; each selected evidence is audited again by the bridge.",
        "original_request_unchanged": True, "old_literal_population_formulation_contains_gate_removed": True,
        "score_cache_reuse": "All complete-input lexical/dense/cross scores unchanged; rebuild masked HGT compatibility and prompt selection."}
    protocol["generation_attempt_checkpoint"] = {
        "maximum_attempts": 2, "maximum_repairs": 1,
        "storage": "Atomic fsynced per-request reservations and full-batch raw-output receipts before validation",
        "resume": "Validate saved raw text without another model call; reconstruct first repair exactly; never reset attempt count",
        "interrupted_unrecorded_call": "Count consumed call and fail explicitly without replay",
        "binding": ["query_id", "global_index", "request_sha256", "raw_query_sha256", "prompt_graph", "protocol_sha256", "producer_sha256"]}
    bridge = artifact(ROOT / "src/pbpk_kb/recommendation/component_model_bridge.py")
    existing = [ref for ref in protocol["source_artifacts"] if ref["path"] != bridge["path"]]
    protocol["source_artifacts"] = [*existing, bridge]
    # The migration is registered before invalidating only the affected stage.
    write(output / "protocol.json", protocol)
    new = {"protocol_sha256": sha(output / "protocol.json"), "prepare_script": old["prepare_script"],
           "generate_script": artifact(Path(__file__))}
    repair = {"revision": 2, "defects": ["Legacy literal population/formulation substring gate falsely rejects complete clinical requests",
        "Unpersisted first generation attempt can reset the two-call budget after interruption"],
        "old_binding": old, "new_binding": new, "original_producer_archive": artifact(producer_backup),
        "preserved_previous_stage": preserved, "reused_complete_input_scores": caches, "context_bridge": bridge,
        "previous_retrieval_statuses": old_manifest["statuses"],
        "rerun_only": ["masked_hgt_compatibility", "selected_subgraphs", "text_graph_prompts"],
        "model_weights_prompts_decoding_and_attempt_budget_unchanged": True,
        "original_requests_and_filtered_corpus_unchanged": True,
        "evaluation_labels_or_enhanced_results_accessed": False, "frozen_old_library_modified": False}
    write(record_path, repair)
    for saved in preserved:
        original = scoped(saved["original"]["path"])
        if original.parent == output / "inputs/prompts" or original in {
                output / "retrieval/manifest.json", output / "retrieval/results.jsonl"}:
            original.unlink()
    write(output / "run_binding.json", new)
    return repair


def retrieve_all(output, *, device="cuda:0", batch_size=16):
    import numpy as np
    from pbpk_kb.recommendation import online_retrieval as online
    from pbpk_kb.recommendation.embedding_benchmark import _model_factory
    from pbpk_kb.recommendation.embedding_index import LOCKED_DENSE_MODELS
    from pbpk_kb.recommendation.gnn_benchmark import QUERY_MODEL
    from pbpk_kb.recommendation.online_agent import _prompt_row
    corpus = corpus_for(output)
    entries = rows(output / "inputs/requests.jsonl")
    pending = [e for e in entries if e["status"] == "pending_retrieval"]
    if (output / "retrieval/manifest.json").exists():
        manifest = read(output / "retrieval/manifest.json")
        for ref in manifest["artifacts"]:
            if sha(ref["path"]) != ref["sha256"]:
                raise ValueError("Stored retrieval artifact changed")
        return manifest
    queries = [{"query_id": e["query_id"], "example_id": e["query_id"],
        "modeling_request": e["modeling_request"], "query_text": online.request_query_text(ModelingRequest.model_validate(e["modeling_request"])),
        "category": online.request_category(ModelingRequest.model_validate(e["modeling_request"])),
        "target_hop": 2, "positive_subgraph_ids": [], "target": {"evidence_ids": [], "public_model_ids": []}}
        for e in pending]
    benchmark = {r["subgraph_id"]: r for r in rows(output / "inputs/gnn_benchmark/subgraph_index.jsonl")}
    records = {hop: [{"object_id": sid, "embedding_text": corpus.subgraphs[sid]["embedding_text"]}
                     for sid in sorted(benchmark) if benchmark[sid]["hop"] == hop] for hop in (1, 2)}
    by_id = {r["object_id"]: r for rs in records.values() for r in rs}
    lexical = online._lexical_rankings(records, queries)
    dense, revisions, query_vectors = {}, {}, None
    old_index = read(INDEX / "embedding_index_manifest.json")
    cache = ROOT / "runtime/hf_cache/embedding"
    for model_name in LOCKED_DENSE_MODELS:
        progress(output, "component_dense_retrieval", model=model_name, requests=len(queries))
        saved = output / "retrieval" / (model_name.replace("/", "_") + ".npz")
        registry = saved.with_suffix(".json")
        old_identity = digest({"queries": queries, "records": records, "revision": old_index["models"][model_name]["revision"]})
        identity = digest({"legacy_input_identity": old_identity, "preprocessing": DENSE_PREPROCESSING})
        if registry.exists() and saved.exists() and read(registry).get("identity") == old_identity:
            preflight = read(DENSE_PREFLIGHT)["models"][model_name]
            if (not preflight["query"]["over_limit"] and not preflight["document"]["over_limit"]
                    and read(registry)["sha256"] == sha(saved)):
                write(registry, {**read(registry), "identity": identity,
                    "encoding_audit": "Independent full-token audit proves previous cache had zero excess inputs; values reused unchanged.",
                    "token_preflight": artifact(DENSE_PREFLIGHT)})
        if registry.exists() and saved.exists() and read(registry).get("identity") == identity and read(registry)["sha256"] == sha(saved):
            with np.load(saved, allow_pickle=False) as arrays:
                qv, dvs = arrays["queries"], {hop: arrays[f"hop{hop}"] for hop in (1, 2)}
            revision = read(registry)["revision"]
        else:
            encoder = _model_factory(model_name, cache_dir=cache, device=device, batch_size=batch_size, query_only=False)
            try:
                revision = str(encoder.revision)
                if revision != old_index["models"][model_name]["revision"]:
                    raise ValueError("Frozen retriever revision mismatch")
                qv, query_audit = complete_dense_encode(encoder, [q["query_text"] for q in queries], role="query")
                dvs, document_audit = {}, {}
                for hop, rs in records.items():
                    dvs[hop], document_audit[hop] = complete_dense_encode(encoder, [r["embedding_text"] for r in rs], role="document")
            finally:
                online._release_accelerator_objects(encoder)
                del encoder
            saved.parent.mkdir(parents=True, exist_ok=True)
            temp = saved.with_suffix(".tmp")
            with temp.open("wb") as stream:
                np.savez_compressed(stream, queries=qv, **{f"hop{h}": v for h, v in dvs.items()})
            temp.replace(saved)
            write(registry, {"identity": identity, "sha256": sha(saved), "revision": revision,
                "encoding_audit": {"query": query_audit, "document_by_hop": document_audit}})
        revisions[model_name] = revision
        if model_name == QUERY_MODEL:
            query_vectors = qv.copy()
        for hop, rs in records.items():
            scores = dvs[hop] @ qv.T
            dense[f"{model_name}::{hop}hop"] = {q["query_id"]: [rs[int(i)]["object_id"] for i in
                np.argsort(-scores[:, j], kind="stable")[:online.CANDIDATE_TOP_K_PER_RETRIEVER]] for j, q in enumerate(queries)}
    candidates, audit = online.build_candidate_pool(retriever_rankings={**lexical, **dense}, queries=queries,
        top_k_per_retriever=online.CANDIDATE_TOP_K_PER_RETRIEVER, pool_size=online.CANDIDATE_POOL_SIZE, rrf_k=online.RRF_K)
    score_path = output / "retrieval/cross_encoder.json"
    if score_path.exists():
        saved_scores = read(score_path)
        if saved_scores["identity"] != digest({"queries": queries, "candidates": candidates}):
            raise ValueError("Cross-encoder resume binding changed")
        cross = {(q, s): float(v) for q, s, v in saved_scores["scores"]}
        token_audit, cross_revision = saved_scores["token_audit"], saved_scores["revision"]
    else:
        progress(output, "component_cross_encoder", pairs=sum(map(len, candidates.values())))
        cross, token_audit, cross_revision = run_complete_cross_encoder(output=output,
            candidates_by_query=candidates, query_rows=queries,
            record_by_id=by_id, cache_dir=cache, device=device, batch_size=batch_size)
        write(score_path, {"identity": digest({"queries": queries, "candidates": candidates}),
            "scores": [[q, s, v] for (q, s), v in sorted(cross.items())], "token_audit": token_audit, "revision": cross_revision})
    if query_vectors is None:
        raise ValueError("Missing fixed Qwen query embeddings")
    graph_eligible = [i for i, q in enumerate(queries) if any(soft_graph_compatibility(
        corpus, ModelingRequest.model_validate(q["modeling_request"]), corpus.subgraphs[s])[0]
        for s in candidates[q["query_id"]])]
    eligible_queries = [queries[i] for i in graph_eligible]
    progress(output, "component_masked_hgt", requests=len(eligible_queries))
    with patch.object(online, "build_graph_tensors", lambda **kw: masked_graph_bundle(output, **kw)), \
            patch.object(online, "_hard_graph_compatibility", soft_graph_compatibility):
        result, _, _ = online._gnn_rerank(query_rows=eligible_queries, candidates_by_query=candidates, candidate_audit=audit,
            cross_scores=cross, corpus=corpus, query_embeddings=query_vectors[graph_eligible], graph_dir=GRAPH,
            subgraph_dir=output / "inputs/subgraphs", embedding_index_dir=INDEX,
            gnn_benchmark_dir=output / "inputs/gnn_benchmark", gnn_training_dir=GNN, gnn_index_dir=GNN_INDEX, device=device)
    for result_row in result:
        request = ModelingRequest.model_validate(result_row["modeling_request"])
        result_row["component_context_audits"] = {sid: component_context_audit(corpus, request, corpus.subgraphs[sid])
            for sid in result_row["selected_subgraph_ids"]}
    online._release_accelerator_objects()
    result_by_id = {r["query_id"]: r for r in result}
    artifacts, statuses = [], Counter()
    for entry in pending:
        request = ModelingRequest.model_validate(entry["modeling_request"])
        retrieval = result_by_id.get(entry["query_id"], {"selected_subgraph_ids": [], "ranked_candidates": [],
            "retrieval_status": "no_graph_compatible_candidate_in_hybrid_pool"})
        row, selected = online_prompt(corpus, request, retrieval, entry, _prompt_row)
        path = output / "inputs/prompts" / (entry["key"] + ".json")
        write(path, row)
        artifacts.append(artifact(path))
        statuses[row["status"]] += 1
    write_rows(output / "retrieval/results.jsonl", result)
    artifacts.append(artifact(output / "retrieval/results.jsonl"))
    manifest = {"version": VERSION, "status": "completed", "requests": len(pending), "statuses": dict(statuses),
        "dense_revisions": revisions, "cross_encoder_revision": cross_revision, "hgt_weights": artifact(GNN / "hgt/seed_17.pt"),
        "source_mask": artifact(output / "inputs/graph_mask.json"), "artifacts": artifacts,
        "producer_binding": artifact(output / "run_binding.json"),
        "condition_audit_implementation": artifact(ROOT / "src/pbpk_kb/recommendation/component_model_bridge.py")}
    write(output / "retrieval/manifest.json", manifest)
    return manifest


def online_prompt(corpus, request, retrieval, entry, prompt_builder):
    row, selected = prompt_builder(corpus=corpus, request=request, retrieval=retrieval)
    # The compact graph is already evidence-bound. Remove the soft-token catalog
    # because there are no soft tokens in the new generator.
    text = row["prompts"]["hybrid"].split("\nSOFT_GRAPH_TOKEN_CATALOG ", 1)[0]
    row["prompts"] = {"text_graph": text}
    row.pop("soft_graph_token_catalog", None)
    row.update(query_id=entry["query_id"], global_index=entry["global_index"], request_sha256=entry["request_sha256"],
        raw_query_sha256=entry["raw_query_sha256"], allowed_subgraph_ids=row["retrieved_subgraph_ids"],
        status="ready" if row["allowed_output_ids"]["evidence_ids"] else "unavailable",
        method="filtered_hybrid_retrieval_frozen_hgt_qwen36_text_graph",
        full_text_graph_audit=serialize_text_graph(corpus, request, selected))
    row["component_context_audits"] = retrieval.get("component_context_audits", {})
    assert not row["allowed_output_ids"]["public_model_ids"]
    return row, selected


def selected_subgraphs(corpus, row):
    traces = {r["subgraph_id"]: r for r in row["retrieval_trace"]}
    return [RetrievedSubgraph(subgraph_id=s, hop=int(corpus.subgraphs[s]["hop_limit"]),
        score=traces[s]["final_score"], score_components=traces[s]["score_components"],
        row=corpus.subgraphs[s], vector=None) for s in row["retrieved_subgraph_ids"]]


def validate_output(corpus, entry, row, raw):
    from pbpk_kb.recommendation.soft_graph_prompt import _decode_and_validate_handles, _translate_handles
    selection, errors, normalizations = _decode_and_validate_handles(raw, row)
    if selection is None or errors:
        return None, {"status": "invalid_selection", "errors": errors}, normalizations
    translated = _translate_handles(selection, row)
    request = ModelingRequest.model_validate(entry["modeling_request"])
    recommendation = assemble_recommendation(corpus=corpus, request=request, subgraphs=selected_subgraphs(corpus, row), selection=translated)
    validation = validate_component_output(corpus=corpus, request=request, recommendation=recommendation,
        allowed_subgraph_ids=set(row["retrieved_subgraph_ids"]))
    return recommendation, validation, normalizations


def selection_schema(row):
    properties = {}
    for key in ("component_ids", "evidence_ids", "public_model_ids", "gap_codes"):
        ids = sorted(row["output_handle_map"][key])
        properties[key] = {"type": "array", "items": {"type": "string", "enum": ids} if ids else {"type": "string"},
                           "maxItems": min(len(ids), 8 if key == "evidence_ids" else 16), "uniqueItems": True}
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


class ComponentGenerator:
    def __init__(self, memory_gib=24):
        import torch
        from transformers import AutoTokenizer, BitsAndBytesConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        from pbpk_kb.blueprints.qwen_proposals import verify_weights
        from frontier_reasoned_fast_judge import verify_fast_environment
        from frontier_modern_judge import prepare_json_constraint
        self.fast_environment = verify_fast_environment()
        weights, _ = verify_weights(WEIGHTS, project_root=ROOT)
        free, total = torch.cuda.mem_get_info(0)
        if free < (memory_gib + 2) * 1024**3:
            raise MemoryError("Need component cap plus 2 GiB free headroom; do not inspect or stop other jobs")
        torch.cuda.set_per_process_memory_fraction(memory_gib * 1024**3 / total, 0)
        self.torch, self.memory_gib = torch, memory_gib
        path = scoped(weights["path"])
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model, info = Qwen3_5ForConditionalGeneration.from_pretrained(path, local_files_only=True,
            trust_remote_code=False, device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("Generator weight loading mismatch")
        self.model.eval()
        self.constraint = prepare_json_constraint(self.tokenizer, self.model.generation_config.eos_token_id)

    def messages(self, row, repair=None):
        messages = [{"role": "system", "content": row["system_instruction"]},
                    {"role": "user", "content": row["prompts"]["text_graph"]}]
        if repair:
            messages += [{"role": "assistant", "content": repair["raw"]}, {"role": "user", "content":
                "One bounded correction: the previous selection failed evidence validation: " +
                json.dumps(repair["errors"]) + ". Return only legal handle IDs, with evidence supporting each selected component. "
                "Do not invent support; use the declared missing-component gap handles where appropriate."}]
        return messages

    def generate(self, tasks):
        from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        callbacks = [build_transformers_prefix_allowed_tokens_fn(self.constraint.tokenizer_data,
            JsonSchemaParser(selection_schema(t["prompt"]), config=CharacterLevelParserConfig(max_consecutive_whitespaces=2))) for t in tasks]
        texts = [self.tokenizer.apply_chat_template(self.messages(t["prompt"], t.get("repair")),
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for t in tasks]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
        width = int(inputs["input_ids"].shape[1])
        if width > 12000 or (width + 256) * len(tasks) > 24000:
            raise MemoryError("Complete component prompt batch exceeds registered token budget")
        lengths = inputs["attention_mask"].sum(1).tolist()
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        torch = self.torch
        torch.cuda.reset_peak_memory_stats(0)
        start = time.monotonic()
        with torch.inference_mode():
            tokens = self.model.generate(**inputs, max_new_tokens=256, do_sample=False, use_cache=True,
                max_time=300, pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.model.generation_config.eos_token_id,
                prefix_allowed_tokens_fn=lambda i, values: callbacks[i](i, values))
        torch.cuda.synchronize()
        outputs = self.tokenizer.batch_decode(tokens[:, width:], skip_special_tokens=True)
        metadata = {"wall_seconds": time.monotonic() - start, "batch_size": len(tasks), "input_tokens": lengths,
            "output_token_width": int(tokens.shape[1] - width), "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "enable_thinking": False, "memory_cap_gib": self.memory_gib}
        del inputs, tokens
        return outputs, metadata


def durable_write(path, value):
    """Atomic, flushed checkpoints survive SSH loss and avoid partial JSON."""
    path = scoped(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def attempt_binding(output, task):
    entry = task["entry"]
    return {"query_id": entry["query_id"], "global_index": entry["global_index"],
        "request_sha256": entry["request_sha256"], "raw_query_sha256": entry["raw_query_sha256"],
        "prompt_graph": task["prompt_ref"], "protocol_sha256": sha(output / "protocol.json"),
        "producer_sha256": sha(Path(__file__)), "maximum_attempts": 2}


def save_attempts(output, task):
    if len(task["attempts"]) > 2:
        raise ValueError("Registered component generation attempt limit exceeded")
    durable_write(output / "components/attempts" / (task["entry"]["key"] + ".json"),
        {"binding": attempt_binding(output, task), "attempts": task["attempts"]})


def recover_attempts(output, task):
    path = output / "components/attempts" / (task["entry"]["key"] + ".json")
    if not path.exists():
        return
    saved = read(path)
    if saved["binding"] != attempt_binding(output, task):
        raise ValueError("Component attempt request/prompt/protocol/producer binding changed")
    attempts = saved["attempts"]
    if not 1 <= len(attempts) <= 2 or any(a["state"] != "evaluated" for a in attempts[:-1]):
        raise ValueError("Malformed or over-budget component attempt checkpoint")
    task["attempts"] = attempts
    attempt = attempts[-1]
    if attempt["state"] == "started":
        receipt_path = scoped(attempt["receipt_path"])
        expected_parent = scoped(output / "components/generation_batches")
        if receipt_path.parent != expected_parent:
            raise ValueError("Component generation receipt leaves its run directory")
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt["batch_id"] != attempt["batch_id"] or digest(receipt["bindings"]) != attempt["batch_id"]:
                raise ValueError("Component batch receipt identity changed")
            match = [i for i, b in enumerate(receipt["bindings"]) if b == {
                **attempt_binding(output, task), "attempt_number": len(attempts)}]
            if len(match) != 1 or len(receipt["outputs"]) != len(receipt["bindings"]):
                raise ValueError("Component batch receipt request binding changed")
            attempt.update(state="generated", raw=receipt["outputs"][match[0]], timing=receipt["timing"],
                           receipt=artifact(receipt_path))
        else:
            # It is impossible to prove whether an interrupted call consumed
            # tokens. Count it and fail explicitly; never silently replay it.
            attempt.update(state="generated", raw="", timing={"failure": "InterruptedUnrecordedGeneration",
                "explanation": "Reserved call has no complete batch receipt; no automatic replay."})
        save_attempts(output, task)
    elif attempt["state"] not in {"generated", "evaluated"}:
        raise ValueError("Unknown component attempt state")


def settle_attempt(output, corpus, task):
    """Validate recorded raw text or finalize without consuming a new call."""
    if not task["attempts"]:
        return None
    attempt = task["attempts"][-1]
    if attempt["state"] == "generated":
        try:
            recommendation, validation, normalizations = validate_output(corpus, task["entry"], task["prompt"], attempt["raw"])
        except (ValueError, KeyError, TypeError) as exc:
            recommendation, validation, normalizations = None, {"status": "assembly_failed",
                "errors": [type(exc).__name__ + ": " + str(exc)[:400]]}, []
        attempt.update(state="evaluated", validation=validation, normalizations=normalizations,
                       recommendation=recommendation)
        save_attempts(output, task)
    if attempt["state"] != "evaluated":
        raise ValueError("Component attempt must be recorded before validation")
    valid = attempt["validation"].get("status") == "recommendation_validated"
    if not valid and len(task["attempts"]) < 2 and "failure" not in attempt["timing"]:
        task["repair"] = {"raw": attempt["raw"], "errors": attempt["validation"].get("errors", [])}
        return None
    entry = task["entry"]
    result = {"query_id": entry["query_id"], "global_index": entry["global_index"],
        "request_sha256": entry["request_sha256"], "status": "validated" if valid else "failed",
        "prompt_graph": task["prompt_ref"], "validation": attempt["validation"],
        "attempts": [{k: v for k, v in a.items() if k != "recommendation"} for a in task["attempts"]],
        "recommendation": deepcopy(attempt["recommendation"]) if valid else None}
    if valid:
        result["recommendation"]["status"] = "recommendation_validated"
    durable_write(output / "components" / (entry["key"] + ".json"), result)
    return result


def record_batch(output, batch, values, timing):
    bindings = [{**attempt_binding(output, task), "attempt_number": len(task["attempts"])} for task in batch]
    batch_id = digest(bindings)
    receipt_path = output / "components/generation_batches" / (batch_id + ".json")
    if len(values) != len(batch):
        raise ValueError("Generation returned a different number of rows")
    durable_write(receipt_path, {"batch_id": batch_id, "bindings": bindings, "outputs": values, "timing": timing})
    for task, raw in zip(batch, values, strict=True):
        attempt = task["attempts"][-1]
        if attempt["state"] != "started" or attempt["batch_id"] != batch_id:
            raise ValueError("Generation reservation changed")
        attempt.update(state="generated", raw=raw, timing=timing, receipt=artifact(receipt_path))
        save_attempts(output, task)


def reserve_batch(output, batch):
    bindings = [{**attempt_binding(output, task), "attempt_number": len(task["attempts"]) + 1} for task in batch]
    batch_id = digest(bindings)
    receipt_path = output / "components/generation_batches" / (batch_id + ".json")
    if receipt_path.exists():
        raise ValueError("Refusing to repeat a recorded generation batch")
    for task in batch:
        if len(task["attempts"]) >= 2:
            raise ValueError("Registered component generation attempt limit exceeded")
        task["attempts"].append({"state": "started", "batch_id": batch_id,
            "receipt_path": str(receipt_path.relative_to(ROOT))})
        save_attempts(output, task)


def cancel_oom_reservations(output, batch):
    """No output was produced; bounded batch reduction is not a repair call."""
    for task in batch:
        if task["attempts"][-1]["state"] != "started":
            raise ValueError("Cannot cancel a completed generation")
        task["attempts"].pop()
        if task["attempts"]:
            save_attempts(output, task)
        else:
            path = output / "components/attempts" / (task["entry"]["key"] + ".json")
            path.unlink()


def generate_all(output, *, batch_size=4, memory_gib=24):
    corpus, entries = corpus_for(output), rows(output / "inputs/requests.jsonl")
    pending, completed = [], []
    for entry in entries:
        path = output / "components" / (entry["key"] + ".json")
        if path.exists():
            saved = read(path)
            if saved["request_sha256"] != entry["request_sha256"] or saved["query_id"] != entry["query_id"]:
                raise ValueError("Component request binding changed")
            completed.append(saved)
            continue
        prompt_path = output / "inputs/prompts" / (entry["key"] + ".json")
        prompt = read(prompt_path) if prompt_path.exists() else None
        if entry["status"] == "unavailable" or (prompt and prompt["status"] != "ready"):
            result = {"query_id": entry["query_id"], "global_index": entry["global_index"], "status": "unavailable",
                "request_sha256": entry["request_sha256"], "reason": entry.get("reason") or "retrieval_no_supported_component_evidence"}
            write(path, result)
            completed.append(result)
        else:
            if prompt is None:
                raise ValueError("Hybrid retrieval must finish before generation")
            task = {"entry": entry, "prompt": prompt, "prompt_ref": artifact(prompt_path), "attempts": []}
            recover_attempts(output, task)
            result = settle_attempt(output, corpus, task)
            if result is None:
                pending.append(task)
            else:
                completed.append(result)
    generator = ComponentGenerator(memory_gib) if pending else None
    size = min(max(1, batch_size), 8)
    while pending:
        batch = pending[:size]
        reserve_batch(output, batch)
        try:
            values, timing = generator.generate(batch)
        except (MemoryError, generator.torch.cuda.OutOfMemoryError) as exc:
            gc.collect()
            generator.torch.cuda.empty_cache()
            if len(batch) > 1:
                cancel_oom_reservations(output, batch)
                size = max(1, len(batch) // 2)
                progress(output, "component_generation_batch_reduction", batch_size=size, completed=len(completed), total=186)
                continue
            values, timing = [""], {"failure": type(exc).__name__, "batch_size": 1}
        # Save the entire returned batch before validating any individual row.
        record_batch(output, batch, values, timing)
        pending = pending[len(batch):]
        for task in batch:
            result = settle_attempt(output, corpus, task)
            if result is None:
                pending.append(task)
                continue
            completed.append(result)
        progress(output, "component_generation", completed=len(completed), pending=len(pending), total=186,
                 statuses=dict(Counter(r["status"] for r in completed)), batch_size=size)
    manifest = {"version": VERSION, "status": "completed", "total": 186,
        "statuses": dict(Counter(r["status"] for r in completed)), "protocol": artifact(output / "protocol.json"),
        "graph_dir": str((output / "inputs/graph").relative_to(ROOT)), "subgraph_dir": str((output / "inputs/subgraphs").relative_to(ROOT)),
        "records": [{"query_id": e["query_id"], **artifact(output / "components" / (e["key"] + ".json"))} for e in entries]}
    write(output / "components/manifest.json", manifest)
    progress(output, "components_complete", completed=186, total=186, statuses=manifest["statuses"])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--stage", choices=("retrieve", "generate", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--retrieval-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--memory-gib", type=int, default=24)
    parser.add_argument("--record-reranker-repair", action="store_true",
                        help="Record the failed original stage and narrowly migrate producer binding; no GPU work")
    parser.add_argument("--record-context-resume-repair", action="store_true",
                        help="Archive the literal-context retrieval result and register revision002; no GPU work")
    args = parser.parse_args()
    if args.device != "cuda:0" or args.memory_gib != 24:
        raise ValueError("This registered generation uses cuda:0 with the measured 24 GiB cap")
    for name, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "USE_HUB_KERNELS": "0", "TOKENIZERS_PARALLELISM": "false"}.items():
        os.environ[name] = value
    output = scoped(args.output)
    if args.record_context_resume_repair:
        record = record_context_resume_repair(output)
        print(json.dumps({"recorded_revision": record["revision"],
            "reused_complete_score_files": len(record["reused_complete_input_scores"])}))
        return
    if args.record_reranker_repair:
        record = record_reranker_repair(output)
        print(json.dumps({"recorded_revision": record["revision"], "retained_dense_files": len(record["reused_dense_cache_artifacts"])}, ensure_ascii=False))
        return
    assert_inputs(output)
    if args.stage in {"retrieve", "all"}:
        retrieve_all(output, device=args.device, batch_size=args.retrieval_batch_size)
    if args.stage in {"generate", "all"}:
        generate_all(output, batch_size=args.batch_size, memory_gib=args.memory_gib)


if __name__ == "__main__":
    main()
