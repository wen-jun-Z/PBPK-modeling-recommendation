"""Freeze full-catalog, outcome-masked inputs for external recommendation methods.

This builds inference inputs, never scientific labels. Only the explicit ``text``
field is passed to text models; provenance and legacy split metadata are audit
fields. Existing evidence-derived requests are retained without turning their
weak compatibility labels into evaluation answers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "experiments/model_recommendation/runs/expanded_catalog_20260913/benchmark"
DEFAULT_OUTPUT = ROOT / "experiments/model_recommendation/frontier_comparison_20260914/inputs"
SCHEMA = "pbpk-frontier-inference-inputs-1.0"
TASKS = {
    "compound_parameterization_template": "find a compound parameterization template",
    "ddi_exposure_prediction": "predict exposure changes in drug-drug interactions",
    "mechanistic_template_recommendation": "find a mechanistic PBPK template",
    "plasma_concentration_time_prediction": "predict the plasma concentration-time profile",
    "special_population_prediction": "predict pharmacokinetics in a special population",
}
MECHANISMS = ("enzymes", "transporters", "process_types")
POPULATIONS = {
    "adult", "geriatric", "pediatric", "pregnancy", "renal_impairment",
    "hepatic_impairment", "healthy", "healthy_adult", "obese", "obesity",
    "neonate", "infant", "child", "adolescent", "elderly", "male", "female",
}
GRAPH_FIELDS = {
    "Compound": "compounds", "Species": "species", "Route": "routes",
    "Population": "populations", "Enzyme": "enzymes", "Transporter": "transporters",
    "ADMEProcess": "process_types", "Endpoint": "endpoints",
    "Compartment": "compartments", "ModelStructure": "model_structure",
    "Software": "software",
}


def scoped(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"Path is outside project: {path}")
    return resolved


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with scoped(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in scoped(path).read_text().splitlines() if line.strip()]


def unique(values) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()}, key=str.casefold)


def display(values) -> str:
    return ", ".join(value.replace("_", " ") for value in unique(values)) or "unspecified"


def safe_graph(graph: dict, allowed: set[str], *, candidate: bool = False) -> dict:
    """Keep typed structure but strip metadata, attributes and unsafe node types."""
    nodes = []
    for node in graph["nodes"]:
        kind, label = node["node_type"], node["label"]
        if kind not in allowed:
            continue
        # Study-bearing population names are not safe stand-ins for phenotype.
        if candidate and kind == "Population" and label.casefold().replace(" ", "_") not in POPULATIONS:
            continue
        nodes.append({key: node[key] for key in ("node_id", "node_type", "label", "required")})
    identifiers = {node["node_id"] for node in nodes}
    edges = [
        {key: edge[key] for key in ("source_id", "target_id", "predicate", "required")}
        for edge in graph["edges"]
        if edge["source_id"] in identifiers and edge["target_id"] in identifiers
    ]
    return {"graph_id": graph["graph_id"], "nodes": nodes, "edges": edges}


def query_lines(query: dict, features: dict) -> list[str]:
    compounds = query.get("required_model_compounds") or query["active_moieties"]
    lines = [
        f"Find PBPK model candidates for {display(compounds)}.",
        f"Intended task: {TASKS[query['task']]}.",
        f"Species context: {display(features['species'])}.",
        f"Administration route context: {display(features['routes'])}.",
        f"Population context: {display(features['populations'])}.",
        f"Target endpoints: {display(features['endpoints'])}.",
        f"Enzyme mentions in supplied evidence: {display(features['enzymes'])}.",
        f"Transporter mentions in supplied evidence: {display(features['transporters'])}.",
        f"ADME process mentions in supplied evidence: {display(features['process_types'])}.",
        "Evidence mentions are documented context, with causal roles not independently adjudicated.",
    ]
    return lines


def build_query(query: dict, graph: dict, evidence: dict) -> dict:
    evidence_ids = sorted(query["evidence_ids"])
    if any(eid not in evidence for eid in evidence_ids):
        raise ValueError(f"Missing evidence provenance for {query['query_id']}")
    features = {key: unique(values) for key, values in query["features"].items()}
    lines = query_lines(query, features)
    reordered = [lines[index] for index in (1, 0, 5, 4, 3, 2, 8, 7, 6, 9)]
    assert Counter(lines) == Counter(reordered)
    # This is a support ablation, not a changed-dose or changed-population task.
    ordered = sorted(evidence_ids, key=lambda eid: hashlib.sha256((query["query_id"] + eid).encode()).hexdigest())
    removed = sorted(ordered[: (len(ordered) + 1) // 2])
    retained = sorted(set(evidence_ids) - set(removed))
    ablated_features = dict(features)
    for key in MECHANISMS:
        available = {value for eid in retained for value in evidence[eid]["extracted_features"].get(key, [])}
        ablated_features[key] = sorted(set(features[key]) & available)
    ablated_text = "\n".join(query_lines(query, ablated_features))
    context_fields = ("species", "routes", "populations", "endpoints")
    assert all(ablated_features[key] == features[key] for key in context_fields)
    return {
        "query_id": query["query_id"],
        "text": "\n".join(lines),
        "compound_id": query["compound_id"],
        "compounds": unique(query.get("required_model_compounds") or query["active_moieties"]),
        "task_type": query["task"],
        "features": features,
        "input_evidence_ids": evidence_ids,
        "legacy_split": query["split"],
        "legacy_split_group_id": query["split_group_id"],
        "request_origin": "existing_evidence_derived_template",
        "graph": safe_graph(graph, set(GRAPH_FIELDS) | {"Objective", "ModelingRequest"}),
        "variants": {
            "reordered": {
                "text": "\n".join(reordered),
                "transformation": "fixed_permutation_of_identical_complete_sentences",
                "meaning_preserved_by_construction": True,
            },
            "missing_evidence": {
                "text": ablated_text,
                "features": ablated_features,
                "retained_evidence_ids": retained,
                "removed_evidence_ids": removed,
                "text_changed": ablated_text != "\n".join(lines),
                "transformation": "drop_hash_selected_half_of_supporting_evidence_recompute_mechanism_mentions_only",
                "explicit_request_context_preserved": True,
                "interpretation": "sensitivity_to_missing_mechanism_support_not_accuracy_or_required_invariance",
            },
        },
        "provenance": {
            "query_graph_checksum": query["query_graph_checksum"],
            "evidence": [{key: evidence[eid].get(key) for key in (
                "evidence_id", "source_id", "source_location", "evidence_text_sha256",
            )} for eid in evidence_ids],
            "independent_study_identity_verified": False,
            "source_content_independence_verified": False,
        },
    }


def formulation_categories(values: list[str]) -> list[str]:
    """Only explicit formulation words, never author/year/dose labels."""
    categories = set()
    patterns = {
        "tablet": r"\btablet", "capsule": r"\bcapsule", "solution": r"\bsolution\b",
        "suspension": r"\bsuspension\b", "intravenous infusion": r"\binfusion\b",
        "immediate release": r"\bimmediate.release\b|(?:\b|_)IR(?:\b|_)",
        "extended release": r"\bextended.release\b|(?:\b|_)XR(?:\b|_)",
        "sustained release": r"\bsustained.release\b|(?:\b|_)SR(?:\b|_)",
        "controlled release": r"\bcontrolled.release\b|(?:\b|_)CR(?:\b|_)",
    }
    for value in values:
        for name, pattern in patterns.items():
            if re.search(pattern, value, re.IGNORECASE):
                categories.add(name)
    return sorted(categories)


def build_candidate(record: dict, graph: dict, source: dict) -> dict:
    clean_graph = safe_graph(graph, set(GRAPH_FIELDS) | {"ModelVersion"}, candidate=True)
    features = {value: [] for value in GRAPH_FIELDS.values()}
    for node in clean_graph["nodes"]:
        if node["node_type"] in GRAPH_FIELDS:
            features[GRAPH_FIELDS[node["node_type"]]].append(node["label"])
    features = {key: unique(values) for key, values in features.items()}
    features["formulations"] = formulation_categories(record.get("formulations", []))
    fields = [
        ("Compounds represented", "compounds"), ("Software", "software"),
        ("Model structure", "model_structure"), ("Species", "species"),
        ("Administration routes", "routes"), ("Population categories", "populations"),
        ("Formulation categories explicitly named", "formulations"),
        ("Enzyme components", "enzymes"), ("Transporter components", "transporters"),
        ("ADME processes", "process_types"), ("Output endpoints", "endpoints"),
        ("Anatomical compartments", "compartments"),
    ]
    text = "\n".join([
        f"Registered PBPK model family: {record['model_family_name']}.",
        *[f"{name}: {display(features[key])}." for name, key in fields],
        "Descriptions report parsed model contents; applicability to a new study is not established by these contents alone.",
    ])
    return {
        "candidate_id": record["model_version_id"], "text": text,
        "family_id": record["model_family"], "family_name": record["model_family_name"],
        "compounds": features["compounds"], "features": features, "graph": clean_graph,
        "provenance": {
            "source_path": source["resolved_source_path"], "source_sha256": source["expected_sha256"],
            "repository_commit": record.get("repository_commit"),
            "license_status": record.get("license_status"),
            "source_content_independence_verified": False,
        },
    }


def write_frozen(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"Refusing to overwrite different frozen input: {path}")
        return
    path.write_text(content)


def jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source, output = scoped(args.source), scoped(args.output)
    names = ("queries.jsonl", "query_graphs.jsonl", "model_records.jsonl", "model_graphs.jsonl", "evidence_utilization.jsonl")
    frozen_manifest = json.loads((source / "experiment_manifest.json").read_text())
    hashes = {name: digest(source / name) for name in names}
    for name, sha in hashes.items():
        if frozen_manifest["artifacts"][name] != sha:
            raise ValueError(f"Frozen source changed: {name}")
    verification_path = source.parent / "benchmark_verification.json"
    verified = json.loads(verification_path.read_text())["catalog"]["source_checks"]
    sources = {row["candidate_id"]: row for row in verified}
    # Recheck all registered model assets, including the recorded frozen-path fallback.
    for row in verified:
        if digest(ROOT / row["resolved_source_path"]) != row["expected_sha256"]:
            raise ValueError(f"Model source changed: {row['candidate_id']}")
    query_graphs = {row["graph_id"]: row for row in read_rows(source / "query_graphs.jsonl")}
    model_graphs = {row["graph_id"]: row for row in read_rows(source / "model_graphs.jsonl")}
    evidence = {row["evidence_id"]: row for row in read_rows(source / "evidence_utilization.jsonl")}
    queries = [build_query(row, query_graphs[row["query_id"]], evidence) for row in read_rows(source / "queries.jsonl")]
    candidates = [build_candidate(row, model_graphs[row["model_version_id"]], sources[row["model_version_id"]]) for row in read_rows(source / "model_records.jsonl")]
    queries.sort(key=lambda row: row["query_id"])
    candidates.sort(key=lambda row: row["candidate_id"])
    if len(queries) != 772 or len(candidates) != 280:
        raise ValueError("This frozen full-data experiment requires exactly 772 requests and 280 candidates")
    if len({row["query_id"] for row in queries}) != len(queries) or len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise ValueError("Duplicate input identifier")
    output.mkdir(parents=True, exist_ok=True)
    write_frozen(output / "queries.jsonl", jsonl(queries))
    write_frozen(output / "candidates.jsonl", jsonl(candidates))
    manifest_path = output / "manifest.json"
    created_at = (json.loads(manifest_path.read_text())["created_at"] if manifest_path.exists()
                  else datetime.now(timezone.utc).isoformat())
    manifest = {
        "schema_version": SCHEMA, "created_at": created_at,
        "role": "full_catalog_zero_shot_inference_inputs_without_judgments",
        "query_count": len(queries), "candidate_count": len(candidates),
        "query_candidate_pairs_per_method": len(queries) * len(candidates),
        "compound_count": len({row["compound_id"] for row in queries}),
        "model_family_count": len({row["family_id"] for row in candidates}),
        "task_counts": dict(sorted(Counter(row["task_type"] for row in queries).items())),
        "legacy_split_counts": dict(sorted(Counter(row["legacy_split"] for row in queries).items())),
        "candidate_pool_policy": "every request scores all candidates sorted by candidate_id; no weak-label-based selection",
        "text_model_input_fields": ["text"],
        "graph_model_input_fields": ["graph"],
        "variant_counts": {
            "reordered": len(queries), "missing_evidence": len(queries),
            "missing_evidence_text_changed": sum(row["variants"]["missing_evidence"]["text_changed"] for row in queries),
        },
        "excluded_from_model_inputs": [
            "all weak relevance labels and label-derived features", "scientific or engineering acceptance",
            "execution and parse statuses", "qualification dataset counts", "validation metrics and observed curves",
            "raw study-bearing population and formulation names", "source paths and IDs", "legacy splits",
        ],
        "limitations": [
            "Requests are existing evidence-derived task templates, not independently collected prospective user requests.",
            "Regex-derived evidence mentions do not establish causal roles or scientific applicability.",
            "Candidate documents use a whitelisted static catalog projection, not full source reports; candidate versions can have identical text.",
            "Removing outcome fields does not verify content-level independence of studies or pretraining contamination.",
            "No independent judgments are generated; ranking accuracy, applicability, calibration and PK selection regret need separate justified targets.",
            "Reordering checks order invariance only, not natural-language paraphrase robustness.",
            "Missing-evidence changes measure sensitivity; stable rankings do not by themselves establish accuracy.",
            "No dose or population-change gold is invented for constraint robustness.",
        ],
        "independent_labels_created": False, "scientific_claim_eligible": False,
        "source_artifacts": [{"path": str((source / name).relative_to(ROOT)), "sha256": sha} for name, sha in hashes.items()] + [
            {"path": str(verification_path.relative_to(ROOT)), "sha256": digest(verification_path)},
        ],
        "model_sources_reverified": len(verified),
        "builder": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": digest(Path(__file__))},
        "outputs": {name: {"sha256": digest(output / name), "bytes": (output / name).stat().st_size} for name in ("queries.jsonl", "candidates.jsonl")},
    }
    write_frozen(manifest_path, json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("query_count", "candidate_count", "query_candidate_pairs_per_method", "compound_count", "model_family_count", "variant_counts", "model_sources_reverified")}, indent=2))


if __name__ == "__main__":
    main()
