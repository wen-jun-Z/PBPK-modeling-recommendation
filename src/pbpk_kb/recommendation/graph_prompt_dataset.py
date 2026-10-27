from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pbpk_kb.recommendation.graph_prompt import (
    GRAPH_PROMPT_VERSION,
    REQUIRED_COMPONENTS,
    GraphPromptCorpus,
    RetrievedSubgraph,
    _components_for_evidence,
    canonical_objective,
    canonical_json,
    dataset_summary,
    derive_independent_silver_target,
    sha256_file,
)
from pbpk_kb.recommendation.embedding_benchmark import load_jsonl
from pbpk_kb.recommendation.schemas import ModelingRequest


DATASET_VERSION = "pbpk-recommendation-silver-2.6.0"
MAX_SOFT_SUBGRAPHS = 8
SOFT_INPUT_DIMENSION = 256
SOFT_TOKEN_COUNT = MAX_SOFT_SUBGRAPHS + 1
DEFAULT_GATE5_LLM_MODEL = "Qwen/Qwen3-8B"
DEFAULT_GATE5_LLM_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DEFAULT_GATE5_LLM_HIDDEN_SIZE = 4096
DEFAULT_GATE5_LLM_DTYPE = "bfloat16"
GAP_TOKEN_LOSS_WEIGHT = 4.0
GATE5_ACCEPTANCE = {
    "minimum_json_schema_valid_rate": 0.95,
    "minimum_hybrid_component_f1_delta_vs_no_kg": 0.05,
    "minimum_hybrid_citation_recall": 0.90,
    "minimum_hybrid_evidence_gap_f1": 0.80,
    "required_citation_provenance_completeness": 1.0,
    "maximum_unsupported_numeric_parameter_rate": 0.0,
    "maximum_quarantined_evidence_leakage": 0,
    "maximum_ontology_invalid_recommendation": 0,
}

SYSTEM_INSTRUCTION = """You are a PBPK knowledge-graph selection model.
Return one JSON object and no prose. Use only IDs present in ALLOWED_OUTPUT_IDS.
Required keys: component_ids, evidence_ids, public_model_ids, gap_codes.
Each value must be a JSON list of strings. Do not invent facts, values, units, IDs,
models, citations, or missing-data conclusions. Select missing-component gap IDs
by applying GAP_SELECTION_RULE to REQUIRED_COMPONENTS and SUPPORTED_COMPONENTS."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _allowed_ids(
    corpus: GraphPromptCorpus,
    request: ModelingRequest,
    subgraphs: list[Any],
) -> dict[str, list[str]]:
    values = derive_independent_silver_target(
        corpus,
        request,
        subgraphs,
        max_evidence=8,
    )
    objective = canonical_objective(request.objective)
    required_components = REQUIRED_COMPONENTS.get(
        objective,
        REQUIRED_COMPONENTS["base_pk"],
    )
    eligible_gap_codes = {
        f"missing_component:{component}"
        for component in required_components
    }
    if objective == "public_model_reuse":
        eligible_gap_codes.add("no_reusable_public_model")
    return {
        "component_ids": sorted(
            {
                "physicochemical",
                "absorption",
                "distribution",
                "metabolism",
                "transport",
                "excretion",
                "ddi",
                "special_population",
                "public_model_template",
            }
        ),
        "evidence_ids": list(values["evidence_ids"]),
        "public_model_ids": list(values["public_model_ids"]),
        "gap_codes": sorted(eligible_gap_codes),
    }


def _user_prompt(
    *,
    request: ModelingRequest,
    allowed: dict[str, list[str]],
    text_graph: str | None,
) -> str:
    sections = [
        "MODELING_REQUEST " + canonical_json(request.model_dump(mode="json")),
    ]
    if text_graph:
        sections.append(text_graph)
    sections.append("ALLOWED_OUTPUT_IDS " + canonical_json(allowed))
    sections.append("Return the required JSON object.")
    return "\n".join(sections)


def _handle_map(
    allowed: dict[str, list[str]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    forward: dict[str, dict[str, str]] = {}
    reverse: dict[str, dict[str, str]] = {}
    prefixes = {
        "component_ids": "C",
        "evidence_ids": "E",
        "public_model_ids": "M",
        "gap_codes": "G",
    }
    for key, values in allowed.items():
        forward[key] = {
            value: f"{prefixes[key]}{index:02d}"
            for index, value in enumerate(values, 1)
        }
        reverse[key] = {
            handle: value for value, handle in forward[key].items()
        }
    return forward, reverse


def _compact_online_text_graph(
    *,
    corpus: GraphPromptCorpus,
    request: ModelingRequest,
    retrieved: list[RetrievedSubgraph],
    allowed: dict[str, list[str]],
    handles: dict[str, dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    evidence_ids = set(map(str, allowed["evidence_ids"]))
    public_model_ids = set(map(str, allowed["public_model_ids"]))
    objective = canonical_objective(request.objective)
    required_components = sorted(
        REQUIRED_COMPONENTS.get(
            objective,
            REQUIRED_COMPONENTS["base_pk"],
        )
    )
    supported_components: set[str] = set()
    for evidence_id in evidence_ids:
        supported_components.update(
            _components_for_evidence(corpus, evidence_id)
        )
    if public_model_ids:
        supported_components.add("public_model_template")

    selected_edges: dict[str, dict[str, Any]] = {}
    subgraph_rows = []
    for rank, item in enumerate(retrieved, 1):
        subgraph_rows.append(
            {
                "handle": f"S{rank:02d}",
                "rank": rank,
                "hop": item.hop,
                "score": round(item.score, 6),
            }
        )
        for edge in item.row.get("edges") or ():
            provenance = edge.get("provenance") or {}
            if str(provenance.get("evidence_id") or "") not in evidence_ids:
                continue
            edge_id = str(edge["edge_id"])
            selected_edges[edge_id] = {
                "subject_id": str(edge["subject_id"]),
                "predicate": str(edge["predicate"]),
                "object_id": str(edge["object_id"]),
                "evidence_id": str(provenance["evidence_id"]),
            }

    lines = [
        "GRAPH_PROMPT_VERSION " + GRAPH_PROMPT_VERSION,
        "REQUIRED_COMPONENTS " + canonical_json(required_components),
        "SUPPORTED_COMPONENTS "
        + canonical_json(sorted(supported_components)),
        (
            "GAP_SELECTION_RULE select missing_component:<component> for each "
            "required component absent from supported; for public_model_reuse "
            "also select no_reusable_public_model when no reusable model exists"
        ),
    ]
    lines.extend(
        "SUBGRAPH " + canonical_json(row)
        for row in subgraph_rows
    )
    for evidence_id in sorted(evidence_ids):
        record = corpus.evidence_record(evidence_id)
        lines.append(
            "EVIDENCE "
            + canonical_json(
                {
                    "handle": handles["evidence_ids"][evidence_id],
                    "components": sorted(
                        _components_for_evidence(corpus, evidence_id)
                    ),
                    "source_id": record.get("source_id"),
                    "supporting_text": str(
                        record.get("supporting_text") or ""
                    )[:600],
                }
            )
        )
    for model_id in sorted(public_model_ids):
        model = corpus.node_by_id.get(model_id) or {}
        lines.append(
            "PUBLIC_MODEL "
            + canonical_json(
                {
                    "handle": handles["public_model_ids"][model_id],
                    "preferred_name": model.get("preferred_name"),
                    "software": (model.get("properties") or {}).get(
                        "software"
                    ),
                }
            )
        )
    edges_by_evidence: dict[str, list[dict[str, Any]]] = {
        evidence_id: [] for evidence_id in evidence_ids
    }
    for edge in selected_edges.values():
        edges_by_evidence[edge["evidence_id"]].append(edge)
    for evidence_id in sorted(edges_by_evidence):
        for edge in sorted(
            edges_by_evidence[evidence_id],
            key=lambda row: (
                row["predicate"],
                row["subject_id"],
                row["object_id"],
            ),
        )[:2]:
            subject = corpus.node_by_id.get(edge["subject_id"]) or {}
            object_ = corpus.node_by_id.get(edge["object_id"]) or {}
            lines.append(
                "EDGE "
                + canonical_json(
                    {
                        "evidence_handle": handles["evidence_ids"][
                            evidence_id
                        ],
                        "subject_type": subject.get("node_type"),
                        "subject": subject.get("preferred_name"),
                        "predicate": edge["predicate"],
                        "object_type": object_.get("node_type"),
                        "object": object_.get("preferred_name"),
                    }
                )
            )
    for rank, item in enumerate(retrieved, 1):
        lines.append(
            "PATH_SUMMARY "
            + canonical_json(
                {
                    "subgraph_handle": f"S{rank:02d}",
                    "path_count": len(item.row.get("path_ids") or ()),
                }
            )
        )
    context = {
        "basis": "modeling_request_plus_online_allowed_evidence",
        "objective": objective,
        "required_components": required_components,
        "supported_components": sorted(supported_components),
        "input_evidence_ids": sorted(evidence_ids),
        "input_public_model_ids": sorted(public_model_ids),
        "selected_subgraph_ids": [
            item.subgraph_id for item in retrieved
        ],
    }
    return "\n".join(lines), context


def build_graph_prompt_dataset(
    *,
    graph_dir: Path,
    subgraph_dir: Path,
    gnn_index_dir: Path,
    online_retrieval_dir: Path,
    output_dir: Path,
    llm_model: str = DEFAULT_GATE5_LLM_MODEL,
    llm_revision: str = DEFAULT_GATE5_LLM_REVISION,
    llm_hidden_size: int = DEFAULT_GATE5_LLM_HIDDEN_SIZE,
    llm_dtype: str = DEFAULT_GATE5_LLM_DTYPE,
) -> dict[str, Any]:
    corpus = GraphPromptCorpus(
        graph_dir=graph_dir,
        subgraph_dir=subgraph_dir,
        gnn_index_dir=gnn_index_dir,
    )
    retrieval_manifest_path = (
        online_retrieval_dir / "online_retrieval_manifest.json"
    )
    retrieval_manifest = json.loads(
        retrieval_manifest_path.read_text(encoding="utf-8")
    )
    if retrieval_manifest.get("accepted") is not True:
        raise RuntimeError("Gate 5 online retrieval is not accepted.")
    all_examples = load_jsonl(
        online_retrieval_dir / "recommendation_silver.jsonl"
    )
    retrieval_rows = load_jsonl(
        online_retrieval_dir / "online_retrieval_results.jsonl"
    )
    retrieval_by_example = {
        str(row["example_id"]): row for row in retrieval_rows
    }
    complete_example_ids = {
        str(row["example_id"])
        for row in retrieval_rows
        if row.get("retrieval_status") == "retrieval_complete"
    }
    examples = [
        row
        for row in all_examples
        if str(row["example_id"]) in complete_example_ids
    ]
    online_vectors = np.load(
        online_retrieval_dir / "query_conditioned_subgraph_vectors.npy"
    )
    online_masks = np.load(
        online_retrieval_dir / "query_conditioned_subgraph_masks.npy"
    )
    if online_vectors.shape != (
        len(retrieval_rows),
        MAX_SOFT_SUBGRAPHS,
        SOFT_INPUT_DIMENSION,
    ):
        raise RuntimeError(
            f"Unexpected query-conditioned vector shape: {online_vectors.shape}"
        )
    if online_masks.shape != (
        len(retrieval_rows),
        MAX_SOFT_SUBGRAPHS,
    ):
        raise RuntimeError(
            f"Unexpected query-conditioned mask shape: {online_masks.shape}"
        )
    summary = dataset_summary(examples)
    if any(
        summary["compound_overlap_counts"].values()
    ):
        raise RuntimeError("Compound-disjoint split contains leakage.")
    if min(summary["compound_counts"].get(split, 0) for split in ("validation", "test")) < 2:
        raise RuntimeError("Validation and test splits each require at least two compounds.")

    prompt_rows = []
    vector_rows: list[np.ndarray] = []
    vector_masks: list[np.ndarray] = []
    for row_index, example in enumerate(examples):
        request = ModelingRequest.model_validate(example["modeling_request"])
        retrieval = retrieval_by_example.get(str(example["example_id"]))
        if retrieval is None:
            raise RuntimeError(
                f"Missing online retrieval row for {example['example_id']}."
            )
        ranked_by_id = {
            str(row["subgraph_id"]): row
            for row in retrieval["ranked_candidates"]
        }
        retrieved = []
        for subgraph_id in retrieval["selected_subgraph_ids"]:
            ranked = ranked_by_id[str(subgraph_id)]
            score_components = {
                "rrf": float(ranked["rrf_score"]),
                "cross_encoder": float(ranked["reranker_score"]),
                "no_gnn_teacher": float(ranked["teacher_score"]),
                "raw_hgt": float(ranked["raw_hgt_score"]),
                "residual_alpha": float(ranked["residual_alpha"]),
                "graph_constraint": float(
                    bool(ranked["graph_constraint_pass"])
                ),
            }
            retrieved.append(
                RetrievedSubgraph(
                    subgraph_id=str(subgraph_id),
                    hop=int(corpus.subgraphs[str(subgraph_id)]["hop_limit"]),
                    score=float(ranked["final_score"]),
                    score_components=score_components,
                    row=corpus.subgraphs[str(subgraph_id)],
                    vector=None,
                )
            )
        retrieval_row = int(retrieval["query_row"])
        mask = np.asarray(online_masks[retrieval_row], dtype=np.bool_)
        if int(mask.sum()) != len(retrieved):
            raise RuntimeError(
                f"Online vector mask differs from retrieval count for "
                f"{example['example_id']}."
            )
        vector_rows.append(
            np.asarray(online_vectors[retrieval_row], dtype=np.float32)
        )
        vector_masks.append(mask)
        allowed = _allowed_ids(corpus, request, retrieved)
        missing_target = {
            key: sorted(
                set(map(str, example["target"].get(key) or ()))
                - set(map(str, allowed[key]))
            )
            for key in (
                "component_ids",
                "evidence_ids",
                "public_model_ids",
                "gap_codes",
            )
        }
        if any(missing_target.values()):
            raise RuntimeError(
                f"Online prompt does not cover locked target for "
                f"{example['example_id']}: {missing_target}"
            )
        target_ids = {
            key: sorted(
                set(example["target"].get(key) or ()) & set(allowed[key])
            )
            for key in (
                "component_ids",
                "evidence_ids",
                "public_model_ids",
                "gap_codes",
            )
        }
        handles, handle_map = _handle_map(allowed)
        target = {
            key: [handles[key][value] for value in target_ids[key]]
            for key in target_ids
        }
        allowed_handles = {
            key: sorted(handle_map[key])
            for key in handle_map
        }
        text_graph, online_selection_context = _compact_online_text_graph(
            corpus=corpus,
            request=request,
            retrieved=retrieved,
            allowed=allowed,
            handles=handles,
        )
        if online_selection_context["input_evidence_ids"] != sorted(
            map(str, allowed["evidence_ids"])
        ):
            raise RuntimeError(
                "Compact graph evidence basis differs from allowed evidence."
            )
        handle_catalog = {
            key: handle_map[key]
            for key in handle_map
        }
        component_gap_map = canonical_json(
            {
                "component_ids": handle_catalog["component_ids"],
                "gap_codes": handle_catalog["gap_codes"],
            }
        )
        text_graph = (
            text_graph
            + "\nOUTPUT_HANDLE_MAP "
            + component_gap_map
        )
        token_catalog: dict[str, dict[str, list[str]]] = {}
        for token_index, item in enumerate(retrieved, 1):
            evidence_handles = sorted(
                {
                    handles["evidence_ids"][str(citation["evidence_id"])]
                    for citation in item.row.get("citations") or ()
                    if str(citation.get("evidence_id") or "")
                    in handles["evidence_ids"]
                }
            )
            model_handles = sorted(
                {
                    handles["public_model_ids"][str(node["node_id"])]
                    for node in item.row.get("nodes") or ()
                    if str(node.get("node_id") or "")
                    in handles["public_model_ids"]
                }
            )
            token_catalog[f"S{token_index:02d}"] = {
                "evidence_ids": evidence_handles,
                "public_model_ids": model_handles,
            }
        no_kg_prompt = _user_prompt(
            request=request,
            allowed={
                **allowed_handles,
                "evidence_ids": [],
                "public_model_ids": [],
            },
            text_graph=None,
        ) + "\nOUTPUT_HANDLE_MAP " + component_gap_map
        soft_prompt = _user_prompt(
            request=request,
            allowed=allowed_handles,
            text_graph=None,
        ) + "\nOUTPUT_HANDLE_MAP " + component_gap_map
        soft_prompt += "\nSOFT_GRAPH_TOKEN_CATALOG " + canonical_json(token_catalog)
        prompt_rows.append(
            {
                "row": row_index,
                "example_id": example["example_id"],
                "split": example["split"],
                "compound_id": example["compound_id"],
                "modeling_request": example["modeling_request"],
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
                "positive_subgraph_ids": example["positive_subgraph_ids"],
                "allowed_output_ids": allowed,
                "output_handle_map": handle_catalog,
                "soft_graph_token_catalog": token_catalog,
                "online_selection_context": online_selection_context,
                "target": target,
                "target_ids": target_ids,
                "target_json": canonical_json(target),
                "soft_token_target_texts": [
                    str(item.row.get("embedding_text") or "")[:4000]
                    for item in retrieved
                ],
                "prompts": {
                    "no_kg": no_kg_prompt,
                    "text_graph": _user_prompt(
                        request=request,
                        allowed=allowed_handles,
                        text_graph=text_graph,
                    ),
                    "soft_graph": soft_prompt,
                    "hybrid": _user_prompt(
                        request=request,
                        allowed=allowed_handles,
                        text_graph=text_graph,
                    )
                    + "\nSOFT_GRAPH_TOKEN_CATALOG "
                    + canonical_json(token_catalog),
                },
                "system_instruction": SYSTEM_INSTRUCTION,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    silver_path = output_dir / "recommendation_silver.jsonl"
    prompts_path = output_dir / "prompt_examples.jsonl"
    vectors_path = output_dir / "soft_graph_inputs.npy"
    masks_path = output_dir / "soft_graph_masks.npy"
    acceptance_path = output_dir / "preregistered_acceptance.json"
    _atomic_jsonl(silver_path, examples)
    _atomic_jsonl(prompts_path, prompt_rows)
    _atomic_npy(vectors_path, np.stack(vector_rows))
    _atomic_npy(masks_path, np.stack(vector_masks))
    _atomic_json(
        acceptance_path,
        {
            "dataset_version": DATASET_VERSION,
            "locked_before_training": True,
            "prompt_variants": [
                "no_kg",
                "text_graph",
                "soft_graph",
                "hybrid",
            ],
            "soft_graph": {
                "input_dimension": SOFT_INPUT_DIMENSION,
                "maximum_subgraph_tokens": MAX_SOFT_SUBGRAPHS,
                "global_pool_token_count": 1,
                "total_soft_token_count": SOFT_TOKEN_COUNT,
                "projector": [
                    "LayerNorm(256)",
                    "Linear(256,2048)",
                    "GELU",
                    "Dropout(0.10)",
                    f"Linear(2048,{llm_hidden_size})",
                    f"LayerNorm({llm_hidden_size})",
                ],
            },
            "llm": {
                "model": llm_model,
                "revision": llm_revision,
                "hidden_size": llm_hidden_size,
                "dtype": llm_dtype,
            },
            "training": {
                "frozen_llm": True,
                "frozen_hgt": True,
                "prompt_variants": ["soft_graph", "hybrid"],
                "validation_loss": "mean(soft_graph,hybrid)",
                "maximum_prompt_tokens": 1408,
                "optimizer": "AdamW",
                "learning_rate": 1e-4,
                "weight_decay": 1e-4,
                "gradient_clip": 1.0,
                "maximum_epochs": 40,
                "patience": 6,
                "gap_token_loss_weight": GAP_TOKEN_LOSS_WEIGHT,
                "seeds": [17, 29, 43],
                "selection_split": "validation",
            },
            "thresholds": GATE5_ACCEPTANCE,
            "created_at": utc_now(),
        },
    )
    artifacts = {}
    for key, path in (
        ("silver", silver_path),
        ("prompts", prompts_path),
        ("soft_graph_inputs", vectors_path),
        ("soft_graph_masks", masks_path),
        ("acceptance", acceptance_path),
    ):
        artifacts[key] = {
            "path": path.name,
            "sha256": sha256_file(path),
        }
    manifest = {
        "accepted": True,
        "dataset_version": DATASET_VERSION,
        "graph_prompt_version": GRAPH_PROMPT_VERSION,
        "benchmark_level": "silver",
        "generated_at": utc_now(),
        "summary": summary,
        "retrieval_failure_count": len(all_examples) - len(examples),
        "vector_shape": [len(prompt_rows), MAX_SOFT_SUBGRAPHS, SOFT_INPUT_DIMENSION],
        "mask_shape": [len(prompt_rows), MAX_SOFT_SUBGRAPHS],
        "text_graph_policy": {
            "version": "online-scenario-compatible-handle-graph-1.2.0",
            "evidence_basis": "online_allowed_auto_validated_evidence",
            "maximum_prompt_tokens": 1408,
            "target_fields_read": [],
        },
        "llm": {
            "model": llm_model,
            "revision": llm_revision,
            "hidden_size": llm_hidden_size,
            "dtype": llm_dtype,
        },
        "artifacts": artifacts,
        "input_hashes": {
            "graph_manifest": sha256_file(
                graph_dir / "kg_extraction_manifest.json"
            ),
            "subgraph_manifest": sha256_file(
                subgraph_dir / "gate2_manifest.json"
            ),
            "gnn_index_manifest": sha256_file(
                gnn_index_dir / "gnn_index_manifest.json"
            ),
            "online_retrieval_manifest": sha256_file(
                retrieval_manifest_path
            ),
        },
        "content_sha256": hashlib.sha256(
            canonical_json(
                {
                    "summary": summary,
                    "artifacts": artifacts,
                }
            ).encode("utf-8")
        ).hexdigest(),
    }
    _atomic_json(output_dir / "dataset_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the locked Gate 5 recommendation silver dataset."
    )
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--subgraph-dir", type=Path, required=True)
    parser.add_argument("--gnn-index-dir", type=Path, required=True)
    parser.add_argument("--online-retrieval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm-model", default=DEFAULT_GATE5_LLM_MODEL)
    parser.add_argument("--llm-revision", default=DEFAULT_GATE5_LLM_REVISION)
    parser.add_argument(
        "--llm-hidden-size",
        type=int,
        default=DEFAULT_GATE5_LLM_HIDDEN_SIZE,
    )
    parser.add_argument("--llm-dtype", default=DEFAULT_GATE5_LLM_DTYPE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_graph_prompt_dataset(
        graph_dir=args.graph_dir,
        subgraph_dir=args.subgraph_dir,
        gnn_index_dir=args.gnn_index_dir,
        online_retrieval_dir=args.online_retrieval_dir,
        output_dir=args.output_dir,
        llm_model=args.llm_model,
        llm_revision=args.llm_revision,
        llm_hidden_size=args.llm_hidden_size,
        llm_dtype=args.llm_dtype,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
