from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from functools import wraps
from concurrent.futures import Future
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from pbpk_kb.paths import resolve_recorded_path
from pbpk_kb.recommendation.embedding_benchmark import load_jsonl
from pbpk_kb.recommendation.graph_prompt import (
    CITATION_CONTEXT_VERSION,
    GRAPH_PROMPT_VERSION,
    RECOMMENDATION_SCHEMA_VERSION,
    VALIDATOR_VERSION,
    GraphPromptCorpus,
    RetrievedSubgraph,
    assemble_recommendation,
    canonical_json,
    sha256_file,
    validate_recommendation,
)
from pbpk_kb.recommendation.graph_prompt_dataset import (
    SYSTEM_INSTRUCTION,
    _allowed_ids,
    _compact_online_text_graph,
    _handle_map,
    _user_prompt,
)
from pbpk_kb.recommendation.online_retrieval import (
    _dense_rankings,
    _gnn_rerank,
    _lexical_rankings,
    _release_accelerator_objects,
    _run_cross_encoder,
    request_category,
    request_query_text,
)
from pbpk_kb.recommendation.reranker_benchmark import (
    CANDIDATE_POOL_SIZE,
    CANDIDATE_TOP_K_PER_RETRIEVER,
    RRF_K,
    build_candidate_pool,
)
from pbpk_kb.recommendation.schemas import ModelingRequest
from pbpk_kb.recommendation.soft_graph_prompt import (
    _decode_and_validate_handles,
    _generate,
    _translate_handles,
    build_projector,
)


ONLINE_AGENT_VERSION = "pbpk-online-recommendation-agent-1.2.0"
_ONLINE_GENERATION_LOCK = Lock()
_ONLINE_STATUS_LOCK = Lock()
_ONLINE_INFLIGHT = {}
_ONLINE_RECENT = {}


def online_generation_status():
    with _ONLINE_STATUS_LOCK:
        entries = list(_ONLINE_INFLIGHT.values())
        return {
            "running": sum(x["state"] == "running" for x in entries),
            "queued": sum(x["state"] == "queued" for x in entries),
            "shared_waiters": sum(x["followers"] for x in entries),
        }


def _single_online_generation(function):
    """Serialize GPU work; identical calls share one result, including DB import latency."""
    @wraps(function)
    def guarded(*args, **kwargs):
        encoded = json.dumps([args, kwargs], sort_keys=True, default=lambda v: v.model_dump(mode="json") if hasattr(v, "model_dump") else str(v))
        key = (function, hashlib.sha256(encoded.encode()).hexdigest())
        now = time.monotonic()
        with _ONLINE_STATUS_LOCK:
            for old in list(_ONLINE_RECENT):
                if now - _ONLINE_RECENT[old][0] > 60:
                    del _ONLINE_RECENT[old]
            if key in _ONLINE_RECENT:
                return deepcopy(_ONLINE_RECENT[key][1])
            entry = _ONLINE_INFLIGHT.get(key)
            owner = entry is None
            if owner:
                entry = {"future": Future(), "state": "queued", "followers": 0}
                _ONLINE_INFLIGHT[key] = entry
            else:
                entry["followers"] += 1
        if not owner:
            try:
                return deepcopy(entry["future"].result())
            finally:
                with _ONLINE_STATUS_LOCK:
                    entry["followers"] -= 1
        try:
            with _ONLINE_GENERATION_LOCK:
                with _ONLINE_STATUS_LOCK:
                    entry["state"] = "running"
                result = function(*args, **kwargs)
                with _ONLINE_STATUS_LOCK:
                    entry["state"] = "completed"
            with _ONLINE_STATUS_LOCK:
                _ONLINE_RECENT[key] = (time.monotonic(), deepcopy(result))
                while len(_ONLINE_RECENT) > 16:
                    del _ONLINE_RECENT[next(iter(_ONLINE_RECENT))]
            entry["future"].set_result(result)
            return deepcopy(result)
        except BaseException as exc:
            entry["future"].set_exception(exc)
            raise
        finally:
            with _ONLINE_STATUS_LOCK:
                _ONLINE_INFLIGHT.pop(key, None)
    return guarded


ONLINE_PATH_ENV = {
    "graph_dir": "PBPK_RECOMMENDATION_GRAPH_DIR",
    "subgraph_dir": "PBPK_RECOMMENDATION_SUBGRAPH_DIR",
    "embedding_index_dir": "PBPK_RECOMMENDATION_EMBEDDING_INDEX_DIR",
    "gnn_benchmark_dir": "PBPK_RECOMMENDATION_GNN_BENCHMARK_DIR",
    "gnn_training_dir": "PBPK_RECOMMENDATION_GNN_TRAINING_DIR",
    "gnn_index_dir": "PBPK_RECOMMENDATION_GNN_INDEX_DIR",
    "cache_dir": "PBPK_RECOMMENDATION_MODEL_CACHE_DIR",
    "gate5_dir": "PBPK_RECOMMENDATION_GATE5_DIR",
    "gate6_dir": "PBPK_RECOMMENDATION_GATE6_DIR",
    "model_path": "PBPK_RECOMMENDATION_LLM_MODEL_PATH",
    "output_root": "PBPK_RECOMMENDATION_ONLINE_OUTPUT_DIR",
}


def _positive_thread_count(environment: str, default: int) -> int:
    value = int(os.getenv(environment, str(default)))
    if value < 1:
        raise RuntimeError(f"{environment} must be a positive integer.")
    return value


def online_thread_settings() -> dict[str, int]:
    return {
        "blas_threads": _positive_thread_count(
            "PBPK_RECOMMENDATION_BLAS_THREADS", 1
        ),
        "torch_threads": _positive_thread_count(
            "PBPK_RECOMMENDATION_TORCH_THREADS", 4
        ),
        "torch_interop_threads": _positive_thread_count(
            "PBPK_RECOMMENDATION_TORCH_INTEROP_THREADS", 1
        ),
    }


def configured_online_agent_kwargs() -> dict[str, Any]:
    missing = [
        environment
        for environment in ONLINE_PATH_ENV.values()
        if not os.getenv(environment)
    ]
    if missing:
        raise RuntimeError(
            "Online recommendation paths are not configured: "
            + ", ".join(missing)
        )
    values: dict[str, Any] = {
        key: Path(os.environ[environment])
        for key, environment in ONLINE_PATH_ENV.items()
    }
    nonexistent = [
        f"{key}={path}"
        for key, path in values.items()
        if key != "output_root" and not path.exists()
    ]
    if nonexistent:
        raise RuntimeError(
            "Online recommendation artifacts are missing: "
            + ", ".join(nonexistent)
        )
    values["device"] = os.getenv(
        "PBPK_RECOMMENDATION_DEVICE",
        "cuda:0",
    )
    values["dense_batch_size"] = int(
        os.getenv("PBPK_RECOMMENDATION_DENSE_BATCH_SIZE", "32")
    )
    values["reranker_batch_size"] = int(
        os.getenv("PBPK_RECOMMENDATION_RERANKER_BATCH_SIZE", "16")
    )
    values.update(online_thread_settings())
    return values


def _configure_torch_runtime(
    torch: Any,
    *,
    torch_threads: int,
    torch_interop_threads: int,
) -> None:
    if int(torch.get_num_threads()) != torch_threads:
        torch.set_num_threads(torch_threads)
    if int(torch.get_num_interop_threads()) != torch_interop_threads:
        try:
            torch.set_num_interop_threads(torch_interop_threads)
        except RuntimeError as exc:
            if int(torch.get_num_interop_threads()) != torch_interop_threads:
                raise RuntimeError(
                    "PyTorch interop threads were initialized before the "
                    "online recommendation runtime could be bounded."
                ) from exc


def _resolve_projector_checkpoint(
    gate5_dir: Path,
    gate5_manifest: dict[str, Any],
) -> Path:
    projector = gate5_manifest.get("projector") or {}
    recorded_value = str(projector.get("checkpoint_path") or "").strip()
    if not recorded_value:
        raise FileNotFoundError(
            "Accepted Gate 5 does not record a projector checkpoint path."
        )
    recorded = Path(recorded_value)
    selected_seed = int(projector["selected_seed"])
    candidates = [resolve_recorded_path(recorded)] if recorded.is_absolute() else []
    if not recorded.is_absolute():
        candidates.extend(
            parent / recorded
            for parent in (Path.cwd(), gate5_dir, *gate5_dir.parents)
        )
    candidates.append(
        gate5_dir / "projectors" / f"seed_{selected_seed}.pt"
    )
    resolved = next(
        (
            candidate
            for candidate in dict.fromkeys(candidates)
            if candidate.is_file()
        ),
        None,
    )
    if resolved is None:
        raise FileNotFoundError(
            "Gate 5 projector checkpoint recorded by the accepted manifest "
            f"is unavailable: {recorded}"
        )
    expected = str(projector["checkpoint_sha256"])
    if sha256_file(resolved) != expected:
        raise RuntimeError("Online projector checksum mismatch.")
    return resolved


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def refresh_reassembled_run_manifest(
    *,
    run_dir: Path,
    runtime_artifact_hashes: dict[str, str],
    reason: str = (
        "Target-compound grounding, deterministic evidence-gap resolution, "
        "and cross-compound leakage hardening."
    ),
) -> dict[str, Any]:
    manifest_path = run_dir / "recommendation_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(
        (run_dir / "recommendation_validation.json").read_text(
            encoding="utf-8"
        )
    )
    recommendation = json.loads(
        (run_dir / "recommendation_final.json").read_text(encoding="utf-8")
    )
    if (
        validation.get("status") != "recommendation_validated"
        or recommendation.get("status") != "recommendation_validated"
    ):
        raise RuntimeError(
            "Only a validated deterministic reassembly can be finalized."
        )
    required_hashes = {
        "graph_manifest",
        "subgraph_manifest",
        "embedding_index_manifest",
        "benchmark_manifest",
        "gnn_comparison",
        "gnn_index_manifest",
    }
    if set(runtime_artifact_hashes) != required_hashes:
        raise RuntimeError(
            "Reassembled run is missing runtime artifact hashes."
        )
    preserved_llm_hash = sha256_file(run_dir / "llm_selection.json")
    preserved_retrieval_hash = sha256_file(run_dir / "online_retrieval.json")
    artifact_paths = sorted(
        path
        for path in run_dir.iterdir()
        if path.is_file() and path.name != manifest_path.name
    )
    manifest.update(
        {
            "online_agent_version": ONLINE_AGENT_VERSION,
            "status": "recommendation_validated",
            "assembly_version": GRAPH_PROMPT_VERSION,
            "recommendation_schema_version": RECOMMENDATION_SCHEMA_VERSION,
            "validator_version": VALIDATOR_VERSION,
            "citation_context_version": CITATION_CONTEXT_VERSION,
            "runtime_artifact_hashes": dict(
                sorted(runtime_artifact_hashes.items())
            ),
            "deterministic_reassembly": {
                "performed": True,
                "preserved_llm_selection_sha256": preserved_llm_hash,
                "preserved_online_retrieval_sha256": preserved_retrieval_hash,
                "reason": reason,
            },
            "artifacts": {
                path.name: sha256_file(path) for path in artifact_paths
            },
        }
    )
    _atomic_json(manifest_path, manifest)
    return manifest


def _validate_runtime_artifact_chain(
    *,
    gate5: dict[str, Any],
    graph_dir: Path,
    subgraph_dir: Path,
    embedding_index_dir: Path,
    gnn_benchmark_dir: Path,
    gnn_training_dir: Path,
    gnn_index_dir: Path,
) -> dict[str, str]:
    paths = {
        "graph_manifest": graph_dir / "kg_extraction_manifest.json",
        "subgraph_manifest": subgraph_dir / "gate2_manifest.json",
        "embedding_index_manifest": (
            embedding_index_dir / "embedding_index_manifest.json"
        ),
        "benchmark_manifest": (
            gnn_benchmark_dir / "gnn_benchmark_manifest.json"
        ),
        "gnn_comparison": gnn_training_dir / "gnn_comparison.json",
        "gnn_index_manifest": gnn_index_dir / "gnn_index_manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Online recommendation artifact chain is incomplete: "
            + ", ".join(missing)
        )
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    gate5_inputs = gate5.get("input_hashes") or {}
    for key in ("graph_manifest", "subgraph_manifest", "gnn_index_manifest"):
        if hashes[key] != str(gate5_inputs.get(key) or ""):
            raise RuntimeError(
                f"Runtime {key} does not match accepted Gate 5."
            )
    gnn_index = json.loads(
        paths["gnn_index_manifest"].read_text(encoding="utf-8")
    )
    if gnn_index.get("accepted") is not True:
        raise RuntimeError("Runtime GNN index is not accepted.")
    gnn_inputs = gnn_index.get("input_hashes") or {}
    linked = {
        "graph_manifest": "graph_manifest",
        "subgraph_manifest": "subgraph_manifest",
        "embedding_index_manifest": "embedding_index_manifest",
        "benchmark_manifest": "benchmark_manifest",
        "gnn_comparison": "comparison",
    }
    for actual_key, recorded_key in linked.items():
        if hashes[actual_key] != str(gnn_inputs.get(recorded_key) or ""):
            raise RuntimeError(
                f"Runtime {actual_key} does not match accepted GNN index."
            )
    embedding_index = json.loads(
        paths["embedding_index_manifest"].read_text(encoding="utf-8")
    )
    if embedding_index.get("accepted") is not True:
        raise RuntimeError("Runtime embedding index is not accepted.")
    embedding_inputs = embedding_index.get("input_hashes") or {}
    embedding_links = {
        "graph_manifest": "graph_manifest",
        "subgraph_manifest": "gate2_manifest",
    }
    for actual_key, recorded_key in embedding_links.items():
        if hashes[actual_key] != str(
            embedding_inputs.get(recorded_key) or ""
        ):
            raise RuntimeError(
                f"Runtime {actual_key} does not match accepted embedding index."
            )
    return hashes


def _runtime_id(request: ModelingRequest) -> str:
    digest = hashlib.sha256(
        canonical_json(request.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()
    return f"online-recommendation:{digest[:24]}"


def _retrieval_row(
    *,
    request: ModelingRequest,
    graph_dir: Path,
    subgraph_dir: Path,
    embedding_index_dir: Path,
    gnn_benchmark_dir: Path,
    gnn_training_dir: Path,
    gnn_index_dir: Path,
    cache_dir: Path,
    device: str,
    dense_batch_size: int,
    reranker_batch_size: int,
    blas_threads: int,
) -> tuple[GraphPromptCorpus, dict[str, Any], np.ndarray, np.ndarray]:
    corpus = GraphPromptCorpus(
        graph_dir=graph_dir,
        subgraph_dir=subgraph_dir,
        gnn_index_dir=gnn_index_dir,
    )
    runtime_id = _runtime_id(request)
    query = {
        "example_id": runtime_id,
        "query_id": runtime_id.replace("online-recommendation", "online-query"),
        "query_text": request_query_text(request),
        "category": request_category(request),
        "target_hop": 2,
        "modeling_request": request.model_dump(mode="json"),
        # Required only by the offline metric tail of _gnn_rerank. These empty
        # values are never used for candidate generation, compatibility, or
        # ranking.
        "positive_subgraph_ids": [],
        "target": {"evidence_ids": [], "public_model_ids": []},
    }
    index_manifest_path = embedding_index_dir / "embedding_index_manifest.json"
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    if index_manifest.get("accepted") is not True:
        raise RuntimeError("Online Agent requires an accepted embedding index.")
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
        for rows in records_by_hop.values()
        for row in rows
    }
    lexical = _lexical_rankings(records_by_hop, [query])
    dense, query_embeddings, dense_revisions = _dense_rankings(
        index_dir=embedding_index_dir,
        index_manifest=index_manifest,
        records_by_hop=records_by_hop,
        query_rows=[query],
        cache_dir=cache_dir,
        device=device,
        batch_size=dense_batch_size,
        blas_threads=blas_threads,
    )
    candidates, candidate_audit = build_candidate_pool(
        retriever_rankings={**lexical, **dense},
        queries=[query],
        top_k_per_retriever=CANDIDATE_TOP_K_PER_RETRIEVER,
        pool_size=CANDIDATE_POOL_SIZE,
        rrf_k=RRF_K,
    )
    cross_scores, token_audit, reranker_revision = _run_cross_encoder(
        candidates_by_query=candidates,
        query_rows=[query],
        record_by_id=record_by_id,
        cache_dir=cache_dir,
        device=device,
        batch_size=reranker_batch_size,
    )
    results, vectors, masks = _gnn_rerank(
        query_rows=[query],
        candidates_by_query=candidates,
        candidate_audit=candidate_audit,
        cross_scores=cross_scores,
        corpus=corpus,
        query_embeddings=query_embeddings,
        graph_dir=graph_dir,
        subgraph_dir=subgraph_dir,
        embedding_index_dir=embedding_index_dir,
        gnn_benchmark_dir=gnn_benchmark_dir,
        gnn_training_dir=gnn_training_dir,
        gnn_index_dir=gnn_index_dir,
        device=device,
    )
    result = results[0]
    result["dense_revisions"] = dense_revisions
    result["reranker_revision"] = reranker_revision
    result["cross_encoder_token_audit"] = token_audit
    return corpus, result, vectors[0], masks[0]


def _prompt_row(
    *,
    corpus: GraphPromptCorpus,
    request: ModelingRequest,
    retrieval: dict[str, Any],
) -> tuple[dict[str, Any], list[RetrievedSubgraph]]:
    ranked_by_id = {
        str(row["subgraph_id"]): row
        for row in retrieval["ranked_candidates"]
    }
    retrieved = []
    for subgraph_id in retrieval["selected_subgraph_ids"]:
        ranked = ranked_by_id[str(subgraph_id)]
        retrieved.append(
            RetrievedSubgraph(
                subgraph_id=str(subgraph_id),
                hop=int(corpus.subgraphs[str(subgraph_id)]["hop_limit"]),
                score=float(ranked["final_score"]),
                score_components={
                    "rrf": float(ranked["rrf_score"]),
                    "cross_encoder": float(ranked["reranker_score"]),
                    "no_gnn_teacher": float(ranked["teacher_score"]),
                    "raw_hgt": float(ranked["raw_hgt_score"]),
                    "residual_alpha": float(ranked["residual_alpha"]),
                    "graph_constraint": float(
                        bool(ranked["graph_constraint_pass"])
                    ),
                },
                row=corpus.subgraphs[str(subgraph_id)],
                vector=None,
            )
        )
    allowed = _allowed_ids(corpus, request, retrieved)
    handles, handle_map = _handle_map(allowed)
    allowed_handles = {
        key: sorted(handle_map[key])
        for key in handle_map
    }
    text_graph, online_context = _compact_online_text_graph(
        corpus=corpus,
        request=request,
        retrieved=retrieved,
        allowed=allowed,
        handles=handles,
    )
    component_gap_map = canonical_json(
        {
            "component_ids": handle_map["component_ids"],
            "gap_codes": handle_map["gap_codes"],
        }
    )
    text_graph += "\nOUTPUT_HANDLE_MAP " + component_gap_map
    token_catalog: dict[str, dict[str, list[str]]] = {}
    for token_index, item in enumerate(retrieved, 1):
        token_catalog[f"S{token_index:02d}"] = {
            "evidence_ids": sorted(
                {
                    handles["evidence_ids"][str(citation["evidence_id"])]
                    for citation in item.row.get("citations") or ()
                    if str(citation.get("evidence_id") or "")
                    in handles["evidence_ids"]
                }
            ),
            "public_model_ids": sorted(
                {
                    handles["public_model_ids"][str(node["node_id"])]
                    for node in item.row.get("nodes") or ()
                    if str(node.get("node_id") or "")
                    in handles["public_model_ids"]
                }
            ),
        }
    prompt = _user_prompt(
        request=request,
        allowed=allowed_handles,
        text_graph=text_graph,
    )
    prompt += "\nSOFT_GRAPH_TOKEN_CATALOG " + canonical_json(token_catalog)
    return (
        {
            "row": 0,
            "example_id": _runtime_id(request),
            "compound_id": str(corpus.resolve_compound(request.compound)["node_id"]),
            "modeling_request": request.model_dump(mode="json"),
            "retrieved_subgraph_ids": [
                item.subgraph_id for item in retrieved
            ],
            "retrieval_trace": [
                {
                    "subgraph_id": item.subgraph_id,
                    "hop": item.hop,
                    "final_score": item.score,
                    "score_components": item.score_components,
                }
                for item in retrieved
            ],
            "allowed_output_ids": allowed,
            "output_handle_map": handle_map,
            "soft_graph_token_catalog": token_catalog,
            "online_selection_context": online_context,
            "prompts": {"hybrid": prompt},
            "system_instruction": SYSTEM_INSTRUCTION,
        },
        retrieved,
    )


@_single_online_generation
def run_online_agent(
    *,
    request: ModelingRequest,
    graph_dir: Path,
    subgraph_dir: Path,
    embedding_index_dir: Path,
    gnn_benchmark_dir: Path,
    gnn_training_dir: Path,
    gnn_index_dir: Path,
    cache_dir: Path,
    gate5_dir: Path,
    gate6_dir: Path,
    model_path: Path,
    output_root: Path,
    device: str = "cuda:0",
    dense_batch_size: int = 32,
    reranker_batch_size: int = 16,
    blas_threads: int = 1,
    torch_threads: int = 4,
    torch_interop_threads: int = 1,
    maximum_prompt_tokens: int = 1408,
    maximum_new_tokens: int = 256,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _configure_torch_runtime(
        torch,
        torch_threads=torch_threads,
        torch_interop_threads=torch_interop_threads,
    )
    started = time.time()
    target_device = torch.device(device)
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but is unavailable; check the project GPU environment.")
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(target_device)
    runtime = {
        "requested_device": str(target_device),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    }
    gate5_path = gate5_dir / "gate5_manifest.json"
    gate6_path = gate6_dir / "gate6_manifest.json"
    gate5 = json.loads(gate5_path.read_text(encoding="utf-8"))
    gate6 = json.loads(gate6_path.read_text(encoding="utf-8"))
    if gate5.get("accepted") is not True:
        raise RuntimeError("Online Agent requires accepted Gate 5.")
    if gate6.get("engineering_acceptance") is not True:
        raise RuntimeError("Online Agent requires Gate 6 engineering acceptance.")
    if (gate6.get("input_hashes") or {}).get("gate5_manifest.json") != sha256_file(
        gate5_path
    ):
        raise RuntimeError("Gate 5 and Gate 6 manifests are not linked.")
    runtime_artifact_hashes = _validate_runtime_artifact_chain(
        gate5=gate5,
        graph_dir=graph_dir,
        subgraph_dir=subgraph_dir,
        embedding_index_dir=embedding_index_dir,
        gnn_benchmark_dir=gnn_benchmark_dir,
        gnn_training_dir=gnn_training_dir,
        gnn_index_dir=gnn_index_dir,
    )

    corpus, retrieval, vectors_numpy, mask_numpy = _retrieval_row(
        request=request,
        graph_dir=graph_dir,
        subgraph_dir=subgraph_dir,
        embedding_index_dir=embedding_index_dir,
        gnn_benchmark_dir=gnn_benchmark_dir,
        gnn_training_dir=gnn_training_dir,
        gnn_index_dir=gnn_index_dir,
        cache_dir=cache_dir,
        device=device,
        dense_batch_size=dense_batch_size,
        reranker_batch_size=reranker_batch_size,
        blas_threads=blas_threads,
    )
    row, retrieved = _prompt_row(
        corpus=corpus,
        request=request,
        retrieval=retrieval,
    )
    _release_accelerator_objects()

    tokenizer = None
    model = None
    projector = None
    vectors = None
    mask = None
    checkpoint = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map={"": device},
            dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        model.requires_grad_(False)
        model.eval()
        model.config.use_cache = True
        runtime["model_device"] = str(next(model.parameters()).device)
        runtime["model_dtype"] = str(next(model.parameters()).dtype)
        if target_device.type == "cuda":
            runtime["gpu_name"] = torch.cuda.get_device_name(target_device)
        projector = build_projector(
            input_dimension=256,
            hidden_dimension=2048,
            output_dimension=int(model.config.hidden_size),
        ).to(device)
        checkpoint_path = _resolve_projector_checkpoint(gate5_dir, gate5)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        projector.load_state_dict(checkpoint["state_dict"])
        projector.eval()
        vectors = torch.from_numpy(
            np.asarray(vectors_numpy, dtype=np.float32)
        ).to(device)
        mask = torch.from_numpy(
            np.asarray(mask_numpy, dtype=np.bool_)
        ).to(device)
        raw, generation = _generate(
            model=model,
            tokenizer=tokenizer,
            projector=projector,
            row=row,
            vectors=vectors,
            mask=mask,
            variant="hybrid",
            device=torch.device(device),
            maximum_prompt_tokens=maximum_prompt_tokens,
            maximum_new_tokens=maximum_new_tokens,
        )
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError(
            "Qwen3-8B BF16 online generation exhausted GPU memory after "
            "retrieval cleanup. The service will not silently substitute an "
            "unvalidated smaller model."
        ) from None
    finally:
        _release_accelerator_objects(model, projector)
        tokenizer = None
        model = None
        projector = None
        vectors = None
        mask = None
        checkpoint = None
        _release_accelerator_objects()
    selection, selection_errors, handle_normalizations = (
        _decode_and_validate_handles(raw, row)
    )
    if selection is None or selection_errors:
        raise RuntimeError(
            f"Online LLM selection failed validation: {selection_errors}"
        )
    translated = _translate_handles(selection, row)
    recommendation = assemble_recommendation(
        corpus=corpus,
        request=request,
        subgraphs=retrieved,
        selection=translated,
    )
    validation = validate_recommendation(
        corpus=corpus,
        request=request,
        recommendation=recommendation,
        allowed_subgraph_ids={
            item.subgraph_id for item in retrieved
        },
    )
    run_dir = output_root / _runtime_id(request)
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        run_dir / "recommendation_request.json",
        request.model_dump(mode="json"),
    )
    _atomic_json(run_dir / "online_retrieval.json", retrieval)
    _atomic_json(run_dir / "graph_prompt_input.json", row)
    _atomic_json(
        run_dir / "llm_selection.json",
        {
            "raw_output": raw,
            "selection_handles": selection,
            "selection_ids": translated,
            "handle_normalizations": handle_normalizations,
            "generation": generation,
        },
    )
    _atomic_jsonl(
        run_dir / "reranked_subgraphs.jsonl",
        [
            {
                "subgraph_id": item.subgraph_id,
                "hop": item.hop,
                "score": item.score,
                "score_components": item.score_components,
            }
            for item in retrieved
        ],
    )
    _atomic_json(run_dir / "recommendation_draft.json", recommendation)
    _atomic_json(run_dir / "recommendation_validation.json", validation)
    if validation["status"] == "recommendation_validated":
        recommendation["status"] = "recommendation_validated"
        _atomic_json(run_dir / "recommendation_final.json", recommendation)
    else:
        _atomic_json(
            run_dir / "recommendation_final.json",
            {
                "status": "not_emitted",
                "validation_status": validation["status"],
            },
        )
    artifact_paths = sorted(
        path for path in run_dir.iterdir() if path.is_file()
    )
    manifest = {
        "online_agent_version": ONLINE_AGENT_VERSION,
        "assembly_version": GRAPH_PROMPT_VERSION,
        "recommendation_schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "citation_context_version": CITATION_CONTEXT_VERSION,
        "status": validation["status"],
        "request_id": _runtime_id(request),
        "gate5_manifest_sha256": sha256_file(gate5_path),
        "gate6_manifest_sha256": sha256_file(gate6_path),
        "model": (gate5.get("llm") or {}).get("model"),
        "model_revision": (gate5.get("llm") or {}).get("revision"),
        "projector_checkpoint_sha256": (
            gate5.get("projector") or {}
        ).get("checkpoint_sha256"),
        "runtime_artifact_hashes": runtime_artifact_hashes,
        "deterministic_reassembly": {"performed": False},
        "runtime": {
            **runtime,
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(target_device)
                if target_device.type == "cuda" else None
            ),
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "artifacts": {
            path.name: sha256_file(path) for path in artifact_paths
        },
    }
    _atomic_json(run_dir / "recommendation_run_manifest.json", manifest)
    return {
        "run_dir": str(run_dir),
        "manifest": manifest,
        "recommendation": (
            recommendation
            if validation["status"] == "recommendation_validated"
            else None
        ),
        "validation": validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one server-side PBPK KG recommendation request."
    )
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--subgraph-dir", type=Path, required=True)
    parser.add_argument("--embedding-index-dir", type=Path, required=True)
    parser.add_argument("--gnn-benchmark-dir", type=Path, required=True)
    parser.add_argument("--gnn-training-dir", type=Path, required=True)
    parser.add_argument("--gnn-index-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--gate5-dir", type=Path, required=True)
    parser.add_argument("--gate6-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dense-batch-size", type=int, default=32)
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = ModelingRequest.model_validate_json(
        args.request_json.read_text(encoding="utf-8")
    )
    result = run_online_agent(
        request=request,
        graph_dir=args.graph_dir,
        subgraph_dir=args.subgraph_dir,
        embedding_index_dir=args.embedding_index_dir,
        gnn_benchmark_dir=args.gnn_benchmark_dir,
        gnn_training_dir=args.gnn_training_dir,
        gnn_index_dir=args.gnn_index_dir,
        cache_dir=args.cache_dir,
        gate5_dir=args.gate5_dir,
        gate6_dir=args.gate6_dir,
        model_path=args.model_path,
        output_root=args.output_root,
        device=args.device,
        dense_batch_size=args.dense_batch_size,
        reranker_batch_size=args.reranker_batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
