"""Freeze the complete registered catalog and evidence-supported ranking benchmark.

The labels describe documented compatibility, never validated scientific use.
Requests are generated from the evidence side, independently of model cards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.recommendation.model_matching.benchmark import _atomic_json, _atomic_jsonl
from pbpk_kb.recommendation.model_matching.graphs import (
    GraphNode, TypedGraph, build_model_graph, build_requirement_graph,
    endpoint_categories, normalize_text, population_categories, semantic_key,
)


ROOT = Path(__file__).resolve().parents[4]
VERSION = "expanded-evidence-catalog-ranking-1.0.0"
LABEL_SEMANTICS = "metadata_compatibility_not_validated_use"
GRADE_DEFINITIONS = {
    "0": "no_documented_relevance; not a biological negative or scientific rejection",
    "1": "different_compound_with_documented_mechanism_overlap",
    "2": "same_compound_with_partial_or_unspecified_requirement_coverage",
    "3": "same_compound_with_documented_requirement_coverage",
}
OBJECTIVES = {
    "clinical_pk": "plasma_concentration_time_prediction",
    "adme": "mechanistic_template_recommendation",
    "ddi": "ddi_exposure_prediction",
    "special_population": "special_population_prediction",
    "physicochemical": "compound_parameterization_template",
}
ENZYME = re.compile(r"\b(?:CYP\s*[1234][A-Z]\s*\d{1,2}|UGT\s*[12][A-Z]\s*\d{1,2}|CES\s*[12]|NAT\s*[12])\b", re.I)
TRANSPORTER = re.compile(r"\b(?:OATP\s*[12][AB]\s*\d|OCT\s*[123]|OAT\s*[134]|MATE\s*[12](?:-K)?|BCRP|ABCB1|ABCG2|P[- ]?GP|PEPT\s*[12])\b", re.I)
PATTERNS = {
    "routes": {
        "oral": r"\b(?:oral(?:ly)?|per os)\b",
        "intravenous": r"\b(?:intravenous(?:ly)?|i\.v\.)\b",
        "subcutaneous": r"\bsubcutaneous(?:ly)?\b",
        "inhalation": r"\b(?:inhaled|inhalation)\b",
        "intramuscular": r"\bintramuscular(?:ly)?\b",
    },
    "species": {
        "human": r"\b(?:human|patients?|volunteers?|subjects?)\b",
        "rat": r"\brats?\b", "mouse": r"\b(?:mice|mouse)\b",
        "dog": r"\bdogs?\b", "monkey": r"\bmonkeys?\b",
    },
    "populations": {
        "pediatric": r"\b(?:pediatric|paediatric|children|infants?|neonat\w*)\b",
        "geriatric": r"\b(?:geriatric|elderly)\b",
        "pregnancy": r"\b(?:pregnan\w*|lactat\w*)\b",
        "renal_impairment": r"\b(?:renal|kidney) (?:impairment|failure|disease)\b",
        "hepatic_impairment": r"\b(?:hepatic|liver) (?:impairment|failure|disease)\b",
        "adult": r"\b(?:adults?|healthy volunteers?)\b",
    },
}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _project_path(path: Path) -> Path:
    path = path.resolve()
    path.relative_to(ROOT)
    return path


def read_registered_snapshot() -> tuple[list[dict], list[dict], list[dict]]:
    """Read only the explicitly registered project database, without exporting secrets."""
    import psycopg
    from sqlalchemy.engine import make_url
    from pbpk_kb.config.settings import get_settings
    from pbpk_kb.storage.final_evidence_query import serving_evidence_sql

    ports = json.loads((ROOT / "runtime/deployment/ports.json").read_text())
    url = make_url(get_settings().database_url)
    if url.host != "127.0.0.1" or url.port != int(ports["postgres"]) or url.database != "pbpk_kb":
        raise ValueError("Only the registered project database is allowed.")
    with psycopg.connect(host=url.host, port=url.port, dbname=url.database,
                         user=url.username, password=url.password, connect_timeout=10,
                         row_factory=psycopg.rows.dict_row) as connection:
        connection.read_only = True
        connection.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        with connection.cursor() as cursor:
            cursor.execute("""SELECT v.*, r.model_name, r.metadata_json AS registry_metadata
                              FROM public_pbpk_model_version v
                              JOIN public_pbpk_model_registry r USING (model_id)
                              ORDER BY v.model_version_id""")
            models = cursor.fetchall()
            cursor.execute("SELECT * FROM (" + serving_evidence_sql() + ") e ORDER BY compound_id,evidence_id")
            evidence = cursor.fetchall()
            cursor.execute("""SELECT compound_id::text, preferred_name::text, synonyms
                              FROM compound_identity ORDER BY compound_id""")
            identities = cursor.fetchall()
    return models, evidence, identities


def _canonicalizer(identities: list[dict]):
    aliases: dict[str, set[str]] = defaultdict(set)
    for identity in identities:
        name = str(identity["preferred_name"]).casefold()
        for alias in [name, *(identity.get("synonyms") or [])]:
            aliases[_key(alias)].add(name)

    def canonical(value: Any) -> str:
        matches = aliases.get(_key(value), set())
        return next(iter(matches)) if len(matches) == 1 else normalize_text(value)

    return canonical


def _model_record(row: dict, canonical) -> tuple[dict, dict]:
    metadata = row.get("metadata_json") or {}
    compounds = {canonical(value) for value in row.get("compounds") or []}
    coverage_bindings = metadata.get("identity_bindings") or []
    compounds.update(canonical(value["canonical_name"]) for value in coverage_bindings if value.get("canonical_name"))
    record = {key: row.get(key) for key in (
        "project_relative_path", "repository_commit", "software", "species", "routes",
        "formulations", "enzymes", "transporters", "compartments", "process_types",
        "output_names", "simulation_count", "parameter_record_count", "dataset_reference_count",
        "static_parse_status", "pkml_parse_status", "execution_probe_status", "reusable_candidate",
    )}
    record.update({
        "model_version_id": str(row["model_version_id"]),
        "external_model_version_id": row["external_model_version_id"],
        "model_family": str(row["model_id"]),
        "model_family_name": metadata.get("model_family") or row["model_name"],
        "compounds": sorted(compounds),
        "populations": population_categories(row.get("populations") or []),
        "license_status": row["license_status"],
        "scientific_accepted": bool(metadata.get("scientific_execution_verified")),
        "engineering_accepted": row["execution_probe_status"] == "passed",
    })
    for dimension in ("species", "routes", "enzymes", "transporters", "process_types"):
        record[dimension] = sorted({normalize_text(v) for v in record.get(dimension) or []})
    source = Path(str(row["project_relative_path"]))
    possible = [ROOT / source, ROOT / "artifacts/pbpk-model-assets/OSP-PBPK-Model-Library" / source]
    source_path = next((_project_path(path) for path in possible if _project_path(path).is_file()), None)
    source_checksum = _hash(source_path) if source_path else None
    source_status = "verified_source" if source_checksum == row["project_sha256"] else "source_hash_mismatch" if source_path else "source_not_available_at_registered_project_paths"
    inventory = {
        "candidate_id": record["model_version_id"], "external_model_version_id": record["external_model_version_id"],
        "model_family_id": record["model_family"], "model_family": record["model_family_name"],
        "compounds": sorted(compounds), "identity_coverage_bindings": coverage_bindings,
        "registered_source_path": row["project_relative_path"],
        "resolved_source_path": str(source_path.relative_to(ROOT)) if source_path else None,
        "registered_sha256": row["project_sha256"], "observed_sha256": source_checksum,
        "source_status": source_status, "source_format": source.suffix,
        "graph_materialization": "registered_parsed_features" if record["compounds"] and any(record.get(f) for f in ("species", "routes", "enzymes", "compartments")) else "registered_metadata_only",
        "static_parse_status": row["static_parse_status"], "pkml_parse_status": row["pkml_parse_status"],
        "execution_probe_status": row["execution_probe_status"], "license_status": row["license_status"],
        "scientific_use_label": None,
        "included_in_catalog": True, "excluded_from_catalog_reason": None,
    }
    return record, inventory


def _evidence_features(row: dict) -> dict[str, list[str]]:
    text = " ".join(str(row.get(key) or "") for key in (
        "parameter_name", "value_text", "context", "evidence_text", "population", "route", "species",
    ))
    features = {dimension: [value for value, pattern in mapping.items() if re.search(pattern, text, re.I)]
                for dimension, mapping in PATTERNS.items()}
    # Multiple routes/species mentioned by a source do not define one unambiguous request.
    features["ambiguous_routes"] = features["routes"] if len(features["routes"]) > 1 else []
    features["ambiguous_species"] = features["species"] if len(features["species"]) > 1 else []
    if features["ambiguous_routes"]:
        features["routes"] = []
    if features["ambiguous_species"]:
        features["species"] = []
    features["enzymes"] = sorted({normalize_text(re.sub(r"\s+", "", match)) for match in ENZYME.findall(text)})
    features["transporters"] = sorted({normalize_text(re.sub(r"\s+", "", match)) for match in TRANSPORTER.findall(text)})
    parameter = normalize_text(row.get("parameter_name") or "")
    features["endpoints"] = endpoint_categories([parameter])
    features["process_types"] = []
    for token, value in (("absorp", "absorption"), ("bioavailability", "absorption"),
                         ("metabol", "metabolism"), ("excret", "excretion"),
                         ("renal clearance", "excretion"), ("distribut", "distribution"),
                         ("protein binding", "distribution"), ("transport", "transport")):
        if token in parameter:
            features["process_types"].append(value)
    return {key: sorted(set(values)) for key, values in features.items()}


def _request_rows(evidence: list[dict], canonical) -> tuple[list[dict], dict[str, TypedGraph], list[dict]]:
    groups: dict[tuple, list[tuple[dict, dict]]] = defaultdict(list)
    evidence_audit = []
    for row in evidence:
        kind = str(row["evidence_type"])
        if kind not in OBJECTIVES:
            evidence_audit.append({"evidence_id": row["evidence_id"], "compound": canonical(row["compound_name"]),
                                   "evidence_type": kind, "query_ids": [],
                                   "role": "excluded_from_request_generation",
                                   "reason": "model_descriptive_evidence_reserved_for_target_side_not_used_to_generate_requests"})
            continue
        features = _evidence_features(row)
        # One request per drug and actual evidence-supported objective/context, not per article or paraphrase.
        signature = (canonical(row["compound_name"]), OBJECTIVES[kind],
                     tuple(features["routes"]), tuple(features["species"]),
                     tuple(features["populations"]), tuple(features["endpoints"]))
        groups[signature].append((row, features))
    queries, graphs = [], {}
    for signature, rows in sorted(groups.items()):
        compound, objective, routes, species, populations, endpoints = signature
        query_id = "evidence-request:" + hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:32]
        merged = {field: sorted({value for _, features in rows for value in features[field]})
                  for field in ("enzymes", "transporters", "process_types")}
        feature_values = {**merged, "routes": list(routes), "species": list(species),
                          "populations": list(populations), "endpoints": list(endpoints)}
        requirements = [{"category": "validation_context", "expected_value": objective,
                         "necessity": "soft_required", "origin": "evidence_supported_task_template"}]
        categories = {"routes": "administration_route", "species": "biological_scope",
                      "populations": "population_feature", "endpoints": "prediction_endpoint",
                      "enzymes": "enzyme", "transporters": "transporter", "process_types": "adme_process"}
        for dimension, category in categories.items():
            for value in feature_values[dimension]:
                supporting = [str(row["evidence_id"]) for row, features in rows if value in features[dimension]]
                requirements.append({"category": category, "expected_value": value,
                                     "necessity": "soft_required", "origin": "automatic_evidence_mention_extraction",
                                     "evidence_ids": sorted(set(supporting)), "confidence": None})
        evidence_ids = sorted({str(row["evidence_id"]) for row, _ in rows})
        graph = build_requirement_graph({"profile_id": query_id, "compound_identity": {"preferred_name": compound}, "requirements": requirements})
        graph = _generic_root(graph, "PBPK modeling request")
        graphs[query_id] = replace(graph, metadata={**graph.metadata, "evidence_ids": evidence_ids,
            "request_origin": "independent_evidence_side", "weak_evidence_extraction": True})
        queries.append({"query_id": query_id, "task": objective, "benchmark_level": "expanded_metadata_ranking",
                        "active_moieties": [compound], "required_model_compounds": [compound],
                        "compound_id": str(rows[0][0]["compound_id"]), "features": feature_values,
                        "evidence_ids": evidence_ids, "evidence_count": len(evidence_ids),
                        "missing_context_dimensions": [dimension for dimension in ("species", "routes", "populations") if not feature_values[dimension]],
                        "request_origin": "independent_evidence_side", "identity_included": True,
                        "supervision_mode": "metadata_ranking", "scientific_use_label": None,
                        "query_graph_checksum": graphs[query_id].checksum})
        for row, features in rows:
            evidence_audit.append({"evidence_id": str(row["evidence_id"]), "compound": compound,
                                   "evidence_type": row["evidence_type"], "query_ids": [query_id],
                                   "role": "request_objective_and_documented_component_context",
                                   "source_id": row.get("source_id"), "source_location": row.get("source_location"),
                                   "evidence_text_sha256": hashlib.sha256(str(row.get("evidence_text") or "").encode()).hexdigest(),
                                   "extracted_features": features,
                                   "numeric_values_role": "not_used_as_model_execution_labels"})
    return queries, graphs, sorted(evidence_audit, key=lambda row: row["evidence_id"])


def _generic_root(graph: TypedGraph, label: str) -> TypedGraph:
    # Do not embed random candidate/query identifiers as biological information.
    return replace(graph, nodes=tuple(replace(node, label=label, semantic_key=semantic_key(node.node_type, label))
                                     if node.hierarchy == "root" else node for node in graph.nodes))


def assign_disjoint_groups(queries: list[dict], models: list[dict]) -> dict:
    parents: dict[str, str] = {}

    def find(value):
        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = find(parents[value])
        return parents[value]

    def union(left, right):
        parents[find(left)] = find(right)

    for model in models:
        family = "family:" + model["model_family"]
        find(family)
        for compound in model["compounds"]:
            union(family, "compound:" + compound)
    for query in queries:
        find("compound:" + query["active_moieties"][0])
    components: dict[str, list[str]] = defaultdict(list)
    for node in sorted(parents):
        components[find(node)].append(node)
    covered = {query["active_moieties"][0] for query in queries}
    compound_count = {root: len({node[9:] for node in nodes if node.startswith("compound:")} & covered)
                      for root, nodes in components.items()}
    evidence_roots = sorted((root for root in components if compound_count[root]),
                            key=lambda root: (-compound_count[root], hashlib.sha256("|".join(components[root]).encode()).hexdigest()))
    if len(evidence_roots) < 3:
        raise ValueError("Fewer than three independent evidence-connected components; cannot create disjoint splits.")
    assignments = {evidence_roots[0]: "train"}
    split_compounds = Counter({"train": compound_count[evidence_roots[0]]})
    for root in evidence_roots[1:]:
        # Fixed 25/3/2 consequence of the present 25-drug DDI connected component.
        target = min(("validation", "test"), key=lambda split: (split_compounds[split] / (3 if split == "validation" else 2), split != "validation"))
        assignments[root] = target
        split_compounds[target] += compound_count[root]
    for root, nodes in components.items():
        if root not in assignments:
            bucket = int(hashlib.sha256("|".join(nodes).encode()).hexdigest()[:8], 16) % 10
            assignments[root] = "train" if bucket < 6 else "validation" if bucket < 8 else "test"
    for query in queries:
        root = find("compound:" + query["active_moieties"][0])
        query["split"] = assignments[root]
        query["split_group_id"] = hashlib.sha256("|".join(components[root]).encode()).hexdigest()[:24]
    for model in models:
        root = find("family:" + model["model_family"])
        model["candidate_split"] = assignments[root]
        model["split_group_id"] = hashlib.sha256("|".join(components[root]).encode()).hexdigest()[:24]
    compound_splits: dict[str, set[str]] = defaultdict(set)
    family_splits: dict[str, set[str]] = defaultdict(set)
    for model in models:
        family_splits[model["model_family"]].add(model["candidate_split"])
        for compound in model["compounds"]:
            compound_splits[compound].add(model["candidate_split"])
    for query in queries:
        compound_splits[query["active_moieties"][0]].add(query["split"])
    passed = all(len(values) == 1 for values in [*compound_splits.values(), *family_splits.values()])
    if not passed:
        raise ValueError("Compound or model-family leakage was detected.")
    return {"passed": passed, "policy": "entire_compound_model_family_connected_components_largest_to_train_remaining_3_to_2",
            "component_count": len(components), "evidence_component_count": len(evidence_roots),
            "compound_split_counts": dict(split_compounds),
            "query_split_counts": dict(Counter(query["split"] for query in queries)),
            "candidate_split_counts": dict(Counter(model["candidate_split"] for model in models)),
            "components": [{"group_id": hashlib.sha256("|".join(nodes).encode()).hexdigest()[:24],
                            "split": assignments[root], "members": nodes,
                            "evidence_covered_compounds": sorted({node[9:] for node in nodes if node.startswith("compound:")} & covered)}
                           for root, nodes in sorted(components.items())],
            "limitation": "DDI models connect most covered drugs; validation/test have few independent compound groups. No connected group was split."}


def compatibility_label(query: dict, model: dict) -> dict:
    required = query["features"]
    available = {**model, "endpoints": endpoint_categories(model.get("output_names") or [])}
    shared, missing, unmatched = {}, [], {}
    requested_count = matched_count = 0
    for dimension in ("species", "routes", "populations", "endpoints", "enzymes", "transporters", "process_types"):
        wanted = {normalize_text(value) for value in required[dimension]}
        offered = {normalize_text(value) for value in available.get(dimension) or []}
        shared[dimension] = sorted(wanted & offered)
        if wanted:
            requested_count += len(wanted)
            matched_count += len(wanted & offered)
            if not offered:
                missing.append(dimension)
            elif wanted - offered:
                unmatched[dimension] = sorted(wanted - offered)
    identity = bool(set(query["active_moieties"]) & set(model["compounds"]))
    coverage = matched_count / requested_count if requested_count else None
    has_context_mismatch = any(dimension in unmatched for dimension in ("species", "routes"))
    if identity:
        grade = 3 if requested_count and matched_count == requested_count and not missing else 2
    else:
        grade = int(any(shared[dimension] for dimension in ("enzymes", "transporters", "process_types")) and not has_context_mismatch)
    return {"query_id": query["query_id"], "candidate_id": model["model_version_id"],
            "relevance_grade": grade, "metadata_relevance_grade": grade,
            "label_origin": "frozen_deterministic_documented_compatibility_weak_supervision",
            "relevance_label_semantics": LABEL_SEMANTICS, "label_known": True,
            "query_split": query["split"], "candidate_split": model["candidate_split"],
            "same_compound": identity, "documented_requirement_coverage": coverage,
            "documented_matches": shared, "missing_candidate_dimensions": missing,
            "unmatched_documented_requirements": unmatched,
            "scientific_use_label": None, "scientific_use_label_known": False,
            "adaptation_executed": False, "adaptation_cost_label": None}


def build_expanded_benchmark(output: Path) -> dict:
    output = _project_path(output)
    if output.exists():
        raise FileExistsError("A frozen expanded benchmark must not be overwritten.")
    db_models, evidence, identities = read_registered_snapshot()
    if len(db_models) != 280 or len({row["compound_id"] for row in evidence}) != 30:
        raise ValueError("This frozen experiment expects the reviewed 280-version, 30-compound catalog.")
    canonical = _canonicalizer(identities)
    models, inventory = [], []
    for row in db_models:
        model, audit = _model_record(row, canonical)
        models.append(model)
        inventory.append(audit)
    queries, query_graphs, evidence_audit = _request_rows(evidence, canonical)
    split_audit = assign_disjoint_groups(queries, models)
    model_graphs = []
    for model in models:
        graph = _generic_root(build_model_graph(model), "Registered PBPK model")
        # Execution grades are neither an input shortcut nor a target of this compatibility experiment.
        excluded = {node.node_id for node in graph.nodes if node.node_type in ("ValidationMetric", "ValidationRun")}
        graph = replace(graph, nodes=tuple(node for node in graph.nodes if node.node_id not in excluded),
                        edges=tuple(edge for edge in graph.edges if edge.source_id not in excluded and edge.target_id not in excluded),
                        metadata={**graph.metadata, "candidate_split": model["candidate_split"],
                                  "split_group_id": model["split_group_id"], "model_family_name": model["model_family_name"],
                                  "supervision_mode": "metadata_ranking", "scientific_use_label": None})
        model_graphs.append(graph)
    for query in queries:
        graph = query_graphs[query["query_id"]]
        query_graphs[query["query_id"]] = replace(graph, metadata={**graph.metadata, "split": query["split"], "split_group_id": query["split_group_id"]})
        query["query_graph_checksum"] = query_graphs[query["query_id"]].checksum
    if len(evidence_audit) != len(evidence) or len({row["evidence_id"] for row in evidence_audit}) != len(evidence):
        raise ValueError("Every evidence record must appear exactly once in the utilization audit.")
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, rows in (
        ("queries.jsonl", queries), ("query_graphs.jsonl", (graph.to_dict() for graph in query_graphs.values())),
        ("model_graphs.jsonl", (graph.to_dict() for graph in model_graphs)),
        ("model_gate_graphs.jsonl", (graph.to_dict() for graph in model_graphs)),
        ("model_inventory.jsonl", inventory), ("evidence_utilization.jsonl", evidence_audit),
        ("model_records.jsonl", models),
    ):
        hashes[name] = _atomic_jsonl(output / name, rows)
    grade_counts, split_grade_counts = Counter(), defaultdict(Counter)

    def labels() -> Iterable[dict]:
        for query in queries:
            for model in models:
                label = compatibility_label(query, model)
                grade_counts[label["relevance_grade"]] += 1
                split_grade_counts[query["split"]][label["relevance_grade"]] += 1
                yield label

    hashes["labels.jsonl"] = _atomic_jsonl(output / "labels.jsonl", labels())
    hashes["split_audit.json"] = _atomic_json(output / "split_audit.json", split_audit)
    manifest = {
        "version": VERSION, "created_at": datetime.now(UTC).isoformat(), "accepted": True,
        "benchmark_level": "expanded_metadata_ranking", "supervision_mode": "metadata_ranking",
        "relevance_label_semantics": LABEL_SEMANTICS, "relevance_grade_definitions": GRADE_DEFINITIONS,
        "scientific_claim_eligible": False,
        "scientific_claim_blockers": ["NEW_CATALOG_LABELS_ARE_DOCUMENTED_COMPATIBILITY_WEAK_SUPERVISION",
                                      "NO_NEW_INDEPENDENT_EXECUTION_LABELS", "FEW_INDEPENDENT_HELD_OUT_COMPOUND_GROUPS"],
        "adaptation_supervision_available": False, "use_decision_supervision_available": False,
        "registered_candidate_count": len(models), "candidate_count": len(model_graphs),
        "excluded_candidate_count": 0, "model_family_count": len({model["model_family"] for model in models}),
        "query_count": len(queries), "label_count": len(queries) * len(models),
        "evidence_compound_count": len({query["compound_id"] for query in queries}),
        "usable_evidence_count": len(evidence), "evidence_utilization_counts": dict(Counter(row["role"] for row in evidence_audit)),
        "source_status_counts": dict(Counter(row["source_status"] for row in inventory)),
        "graph_materialization_counts": dict(Counter(row["graph_materialization"] for row in inventory)),
        "grade_counts": dict(grade_counts), "query_split_grade_counts": {split: dict(counts) for split, counts in split_grade_counts.items()},
        "query_split_counts": split_audit["query_split_counts"], "candidate_split_counts": split_audit["candidate_split_counts"],
        "compound_split_counts": split_audit["compound_split_counts"], "split_audit_passed": split_audit["passed"],
        "label_policy": "Frozen evidence-side requirements compared with registered source-parsed candidate features. No learned model generated these labels.",
        "query_policy": "All non-model-descriptive usable evidence aggregated by compound/objective/unambiguous route/species/population/endpoints; no model-card-derived recovery requests.",
        "evidence_extraction_limit": "Regex mentions are weak evidence of documented context/components, not adjudicated causal roles. Ambiguous species/routes remain unspecified.",
        "evaluation_limit": "Measures agreement with documented compatibility rules, not independent scientific model applicability or improved simulation accuracy.",
        "artifacts": hashes,
    }
    _atomic_json(output / "experiment_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_expanded_benchmark(args.output)
    print(json.dumps({key: manifest[key] for key in (
        "candidate_count", "query_count", "label_count", "evidence_compound_count", "usable_evidence_count",
        "compound_split_counts", "query_split_counts", "candidate_split_counts", "grade_counts",
    )}, indent=2))


if __name__ == "__main__":
    main()
