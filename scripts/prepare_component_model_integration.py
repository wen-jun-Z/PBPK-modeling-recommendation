#!/usr/bin/env python3
"""Freeze outcome-blind component inputs; this command never loads a GPU model."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pbpk_kb.recommendation.graph_prompt import GraphPromptCorpus, canonical_json
from pbpk_kb.recommendation.schemas import ModelingRequest

OUT = ROOT / "experiments/model_recommendation/component_integration_20261007"
FROZEN = ROOT / "experiments/model_recommendation/pbpk_match_v2/dataset.json"
BASE = ROOT / "experiments/component_recommendation"
GRAPH = BASE / "107_runtime_graph_gate5_locked"
SUBGRAPHS = BASE / "108_runtime_subgraphs_gate5_locked"
INDEX = BASE / "73_gate3_multiretriever_production_index"
GNN = BASE / "80_gate4_v3_gnn_training"
GNN_INDEX = BASE / "81_gate4_hgt_production_index"
WEIGHTS = ROOT / "experiments/model_recommendation/frontier_comparison_20260914/assessment_qwen36/model_weights/qwen.json"
VERSION = "pbpk-component-model-integration-20261007-v1"


def scoped(path):
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Artifact leaves project boundary")
    return path


def sha(path):
    with scoped(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def read(path):
    return json.loads(scoped(path).read_text())


def rows(path):
    return [json.loads(s) for s in scoped(path).read_text().splitlines() if s.strip()]


def write(path, value):
    path = scoped(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def write_rows(path, values):
    path = scoped(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        for value in values:
            stream.write(canonical_json(value) + "\n")
    tmp.replace(path)


def artifact(path):
    path = scoped(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": path.stat().st_size}


def request_key(query_id):
    return hashlib.sha256(query_id.encode()).hexdigest()[:24]


def design_only(scenario):
    """Whitelist clinical design, not evaluation_target, error or observations."""
    design = (scenario or {}).get("normalized_design", {})
    return {key: deepcopy(design[key]) for key in
            ("species", "route", "dose", "regimen", "food", "formulation", "population", "sampling", "matrix", "analyte", "endpoint_contracts")
            if key in design}


def make_request(query, scenario=None):
    features, design = query.get("features", {}), design_only(scenario)
    compounds = query.get("compounds", [])
    if len(compounds) != 1:
        raise ValueError("A unique target compound is required; no drug-name joining")
    task = query.get("task_type", "")
    objective = "ddi" if task == "ddi_exposure_prediction" else (
        "special_population" if task == "special_population_prediction" else "base_pk")
    # The full request is retained below: template requests are not converted to
    # public-model selection, because this upstream experiment excludes model IDs.
    first = lambda key: ", ".join(features.get(key, [])) or None
    endpoint_names = {"auc": "AUC", "cmax": "Cmax", "tmax": "Tmax", "half_life": "half_life",
                      "clearance": "clearance", "volume_of_distribution": "volume_of_distribution",
                      "auc_ratio": "DDI_AUC_ratio", "cmax_ratio": "DDI_Cmax_ratio"}
    dose, regimen = design.get("dose") or {}, design.get("regimen") or {}
    route = design.get("route") or dose.get("route") or first("routes")
    food = str(design.get("food") or "").lower()
    food_category = ("fasted" if "fast" in food else "fed" if re.search(r"\bfed\b|^fed_", food) else None)
    times = regimen.get("dose_times_h")
    values = dict(compound=compounds[0], objective=objective,
        species=design.get("species") or first("species") or "unspecified", route=route,
        population=canonical_json(design["population"]) if design.get("population") else first("populations"),
        enzyme=list(features.get("enzymes", [])), transporter=list(features.get("transporters", [])),
        prediction_endpoints=sorted({endpoint_names[e] for e in features.get("endpoints", []) if e in endpoint_names}),
        dose=dose.get("value"), dose_unit=dose.get("unit"), formulation=design.get("formulation") or dose.get("formulation"),
        fed_or_fasted=food_category, single_or_multiple=("single" if len(times) == 1 else "multiple") if times else None,
        frequency=canonical_json({"dose_times_h": times}) if times else None,
        infusion_duration=str(regimen["infusion_duration_h"]) + " h" if regimen.get("infusion_duration_h") is not None else None,
        constraints="ORIGINAL_MODEL_RECOMMENDATION_REQUEST\n" + query["text"] +
                    "\nEXPLICIT_DESIGN_ONLY " + canonical_json(design) +
                    "\nSelect evidence-backed mechanisms only; unspecified fields remain unspecified. "
                    "Do not recommend named PBPK candidates or infer target study outcomes.", top_k_subgraphs=8)
    return ModelingRequest.model_validate(values).model_dump(mode="json")


def source_blocklist(scenarios):
    """Only registered source identifiers/paths; never open sealed observations."""
    tokens, sources = set(), []
    for scenario in scenarios:
        identifiers = scenario.get("publication_ids") or {}
        if not any(identifiers.values()):
            raise ValueError("Execution scenario lacks publication identity: " + scenario["scenario_id"])
        local = {str(v).lower() for v in identifiers.values() if v}
        local.update(str(scenario.get(k) or "").lower() for k in ("study_id", "publication_cluster_id", "cohort_id"))
        source = scenario.get("source_protocol") or {}
        if source.get("path"):
            path = scoped(source["path"])
            if source.get("sha256") and sha(path) != source["sha256"]:
                raise ValueError("Clinical metadata checksum mismatch")
            protocol = read(path)
            exclusion = protocol.get("evidence_exclusion") or {}
            local.update(str(v).lower() for v in (exclusion.get("identifiers") or {}).values() if v)
            local.update(str(v).lower() for v in exclusion.get("trial_registry_ids", []))
            for name in (*exclusion.get("raw_source_files", []), *exclusion.get("curator_only_source_paths", [])):
                local.add(Path(name).name.lower())
            sources.append(artifact(path))
        tokens.update(x for x in local if x)
    return {"tokens": sorted(tokens), "metadata_sources": sources,
            "policy": "All 41 target publications/trials and registered raw batches excluded globally; metadata-only reads."}


def filter_graph(output, blocked):
    nodes, edges, events = rows(GRAPH / "kg_nodes.jsonl"), rows(GRAPH / "kg_edges.jsonl"), rows(GRAPH / "kg_events.jsonl")
    by_id = {n["node_id"]: n for n in nodes}
    by_evidence = {}
    for edge in edges:
        evidence_id = (edge.get("provenance") or {}).get("evidence_id")
        if evidence_id:
            by_evidence.setdefault(evidence_id, []).append(edge)
    rejected, allowed = {}, set()
    for eid, related in by_evidence.items():
        node = by_id.get(eid, {})
        kind = str(node.get("preferred_name") or "").split(":", 1)[0]
        provenance = related[0].get("provenance") or {}
        text = canonical_json([e.get("provenance") for e in related]).lower()
        if any(token in text for token in blocked["tokens"]):
            rejected[eid] = "target_publication_trial_or_raw_batch"
        elif kind not in {"adme", "ddi", "physicochemical"}:
            rejected[eid] = "clinical_population_or_model_record_excluded"
        elif re.search(r"\bgmfe\b|\brmse\b|\brmsle\b|fold.error|predicted.to.observed|observed.to.predicted|model validation|model performance", text):
            rejected[eid] = "model_performance_text_excluded"
        elif provenance.get("quality_status") != "auto_validated":
            rejected[eid] = "not_auto_validated"
        else:
            allowed.add(eid)
    kept_edges = [e for e in edges if (e.get("provenance") or {}).get("evidence_id") in allowed
                  and e.get("predicate") != "DESCRIBED_BY_MODEL"
                  and by_id[e["subject_id"]]["node_type"] != "PBPKModel"
                  and by_id[e["object_id"]]["node_type"] != "PBPKModel"]
    node_ids = {v for e in kept_edges for v in (e["subject_id"], e["object_id"])} | allowed
    node_ids.update(n["node_id"] for n in nodes if n["node_type"] == "Compound")
    edge_ids = {e["edge_id"] for e in kept_edges}
    kept_events = [e for e in events if e.get("evidence_id") in allowed]
    for name, values in [("kg_nodes.jsonl", [n for n in nodes if n["node_id"] in node_ids]),
                         ("kg_edges.jsonl", kept_edges), ("kg_events.jsonl", kept_events)]:
        write_rows(output / "inputs/graph" / name, values)
    index_rows = []
    for hop in range(3):
        filtered = []
        for original in rows(SUBGRAPHS / f"subgraphs_{hop}hop.jsonl"):
            row = deepcopy(original)
            row["edges"] = [e for e in row.get("edges", []) if e["edge_id"] in edge_ids]
            row["citations"] = [e for e in row.get("citations", []) if e.get("evidence_id") in allowed]
            members = {row.get("center_id")} | {v for e in row["edges"] for v in (e["subject_id"], e["object_id"])}
            row["nodes"] = [n for n in row["nodes"] if n["node_id"] in node_ids and n["node_id"] in members]
            row["events"] = [e for e in row.get("events", []) if e.get("evidence_id") in allowed]
            row["path_ids"] = []  # Old paths may cross excluded edges; retained edge lists are authoritative.
            row.update(node_count=len(row["nodes"]), edge_count=len(row["edges"]),
                       citation_count=len(row["citations"]), event_count=len(row["events"]))
            row["embedding_text"] = canonical_json({"center": row.get("center_name"),
                "nodes": [{"type": n["node_type"], "name": n.get("preferred_name")} for n in row["nodes"][:12]],
                "relations": sorted({e["predicate"] for e in row["edges"]}),
                "evidence": [str(c.get("supporting_text") or "")[:600] for c in row["citations"][:4]]})
            row["source_content_sha256"] = original.get("content_sha256")
            row["content_sha256"] = digest({k: v for k, v in row.items() if k != "content_sha256"})
            if row["nodes"]:
                filtered.append(row)
                if hop and row["citations"] and row["edges"]:
                    index_rows.append({"subgraph_id": row["subgraph_id"], "node_ids": [n["node_id"] for n in row["nodes"]],
                        "node_count": row["node_count"], "edge_count": row["edge_count"], "event_count": row["event_count"],
                        "citation_count": row["citation_count"], "hop": hop})
        write_rows(output / f"inputs/subgraphs/subgraphs_{hop}hop.jsonl", filtered)
    write_rows(output / "inputs/gnn_benchmark/subgraph_index.jsonl", index_rows)
    mask = {"allowed_node_ids": sorted(node_ids), "allowed_edge_ids": sorted(edge_ids),
            "allowed_evidence_ids": sorted(allowed), "rejected_evidence": rejected,
            "rejection_counts": dict(Counter(rejected.values())),
            "retained_subgraph_count": len(index_rows),
            "gnn_policy": "Original vocabularies and tensor indices; excluded nodes zeroed and excluded message edges removed BEFORE graph encoding."}
    write(output / "inputs/graph_mask.json", mask)
    return mask


def prepare(output=OUT):
    output = scoped(output)
    if (output / "components/manifest.json").exists():
        raise ValueError("Generation already exists; do not rewrite registered inputs")
    data = read(FROZEN)
    scenarios = {s["scenario_id"]: s for s in data["execution_scenarios"]}
    blocked = source_blocklist(list(scenarios.values()))
    write(output / "inputs/source_blocklist.json", blocked)
    mask = filter_graph(output, blocked)
    corpus = GraphPromptCorpus(graph_dir=output / "inputs/graph", subgraph_dir=output / "inputs/subgraphs")
    entries = []
    for cohort, indices in [("core", data["core_indices"]), ("execution", data["execution_indices"])]:
        for i in indices:
            query = deepcopy(data["query_records"][i])
            request = make_request(query, scenarios.get(query["query_id"]))
            item = {"query_id": query["query_id"], "global_index": int(i), "cohort": cohort,
                "query_record": query, "raw_query_sha256": digest(query), "modeling_request": request,
                "request_sha256": digest(request), "key": request_key(query["query_id"])}
            try:
                node = corpus.resolve_compound(request["compound"])
                available = corpus.retrieve(ModelingRequest.model_validate(request), max_subgraphs=30)
                item.update(status="pending_retrieval" if available else "unavailable",
                    reason=None if available else "no_eligible_source_disjoint_component_subgraph", compound_id=node["node_id"],
                    eligible_subgraph_count=len(available))
            except LookupError:
                item.update(status="unavailable", reason="compound_absent_from_locked_component_graph", eligible_subgraph_count=0)
            entries.append(item)
    if len(entries) != 186 or len({e["query_id"] for e in entries}) != 186:
        raise ValueError("Expected exact 145+41 unique original queries")
    write_rows(output / "inputs/requests.jsonl", entries)
    sources = [FROZEN, GRAPH / "kg_nodes.jsonl", GRAPH / "kg_edges.jsonl", GRAPH / "kg_events.jsonl",
        *(SUBGRAPHS / f"subgraphs_{hop}hop.jsonl" for hop in range(3)),
        INDEX / "embedding_index_manifest.json", GNN_INDEX / "gnn_index_manifest.json", WEIGHTS]
    protocol = {"version": VERSION, "purpose": "Frozen PBPK-Match head, with versus without generated component evidence; no new quality labels.",
        "cohorts": {"core": 145, "execution": 41}, "source_artifacts": [artifact(p) for p in sources],
        "retrieval": {"pipeline": "locked lexical+dense RRF; Qwen3-Reranker; frozen HGT and no-GNN teacher blend",
            "filtered_subgraph_text_reencoded": True, "hgt_input_mask_before_message_passing": True,
            "new_training": False, "fallback": "Unavailable/failed component request uses unchanged original PBPK-Match scores."},
        "generation": {"model": "Qwen/Qwen3.6-27B", "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
            "method": "text_graph_selection", "is_historical_soft_graph_hybrid": False,
            "projector_used": False, "weights_manifest": artifact(WEIGHTS), "quantization": "NF4 double quant bf16",
            "enable_thinking": False, "do_sample": False, "max_new_tokens": 256, "max_input_tokens": 12000,
            "initial_batch_size": 4, "maximum_batch_size": 8, "oom_policy": "halve and retry; single OOM fails explicitly",
            "repair_attempts": 1, "max_batch_seconds": 300, "memory_cap_gib": 24, "minimum_headroom_gib": 2},
        "evidence_policy": blocked["policy"], "excluded_information": ["target study outcomes", "PBPK candidate IDs", "model validation scores", "recommendation quality labels"],
        "reuse_old_components": False, "component_training_data_shift": "Frozen early graph; no claim of expanded knowledge-base coverage.",
        "ranking_and_evaluation": {"frozen_outer_heads": 5, "folds_alpha_temperature_unchanged": True,
            "main_comparison": "pbpk_match_light before/after components", "diagnostic_comparison": "pure document before/after components",
            "training": False, "candidates_per_query": 280, "unavailable_or_invalid": "Original input and scores; retain all 186 denominators",
            "primary_tests": ["NDCG@10", "Top1 ordinary Cmax absolute log error on common scorable scenarios"],
            "multiplicity": "Holm across two primary tests", "bootstrap": {"draws": 2000, "unit": "original grouped split units"},
            "other_endpoints": "Top1-3 endpoints, curves, omissions, coverage descriptive; one-publication curves no significance",
            "limitations": ["new component information differs from original baseline inputs", "27B generator shares model family with automatic quality judge",
                "not the historical 8B soft-graph Hybrid", "not independent new-drug validation"]},
        "prepared_input_artifacts": [artifact(p) for p in sorted((output / "inputs").rglob("*")) if p.is_file()]}
    if (output / "protocol.json").exists():
        old = read(output / "protocol.json")
        for key in ("ranking_and_evaluation", "ranking", "evaluation"):
            if key in old:
                protocol[key] = old[key]
    write(output / "protocol.json", protocol)
    summary = {"version": VERSION, "total": len(entries), "by_cohort": {}, "graph_mask": {k: mask[k] for k in ("rejection_counts", "retained_subgraph_count")}}
    for cohort in ("core", "execution"):
        subset = [e for e in entries if e["cohort"] == cohort]
        summary["by_cohort"][cohort] = {"total": len(subset), "statuses": dict(Counter(e["status"] for e in subset)),
                                      "reasons": dict(Counter(e["reason"] for e in subset if e["reason"]))}
    write(output / "preparation.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    prepare(args.output)
